"""
PLOFA — CONTINUOUS POSSESSION PHYSICS
=====================================
possession_physics.py

Philosophy:
    Football is a race. The match AI keeps deciding WHO does WHAT — which
    receiver to hit, which lane to carry, whether to shoot — but the OUTCOME
    is no longer a weighted roll. Every possession action is resolved on a
    shared 0.1-second clock by integrating player motion and ball travel, and
    contacts are awarded purely by geometry:

        * Ground pass / through ball  -> race between the travelling ball,
          the intended receiver and every defender (race-to-ball).
        * Dribble / take-on           -> carrier movement vs defender pursuit,
          settled by tackle radii and a sustained contact window.
        * Shot vs goalkeeper          -> a 3D shot-target trajectory vs the
          keeper's reaction, vertical reach and dive timing.

    Long balls, crosses, aerial duels and rebounds run on the ballistic
    delivery model (geometry_engine.resolve_aerial_delivery) — they are the
    natural 3D follow-up stage. Lofted / long PASSES now resolve through
    resolve_aerial_pass: a 3D arc (height over time) with the receiver's
    jump/reach contested by a defender's jump timing + vertical reach at the
    landing point, gated by a skill differential (Checkpoint 28). This module
    makes the ground game a continuous-motion simulation instead of a sequence
    of independent rolls.

    Checkpoint 6 — long balls go TRUE ballistic: PossessionEpisode.
    resolve_long_pass is the long-pass outcome authority. It builds the
    flight with geometry_engine.make_ballistic_flight — launch angle solved
    against range/height under gravity, so flight time and apex are DERIVED
    from the launch velocity vector, not tuned as free numbers — deflects the
    landing point with WeatherPhysics.pass_lateral_deflection (is_airborne
    = True → k_drag 0.09), and settles the descent as an aerial duel
    (acceleration-curve arrival + jump timing + standing/jump reach from
    physical.jumping). The pure geometric verdict is returned; the Checkpoint
    28 attribute gate is applied in the chain (PossessionChain.
    _aerial_pass_gate), matching where the ground-pass interception gate sits.

Calibration basis (real football ranges, 2026):
    | Quantity                          | Unit      | Real range           | Mapping used here        |
    |-----------------------------------|-----------|----------------------|--------------------------|
    | Sprint speed                      | m/s       | 36-38 km/h elite top | 5.0 + pace*0.042 (<=9.2) |
    | Sprint speed                      | km/h      | 18-31 in-match       | 5.0..9.2 m/s             |
    | Acceleration                      | m/s^2     | 3.1-5.1 (0-5m start) | 3.0 + accel*0.030        |
    | Reaction time                     | s         | 0.35 solo, 0.15 elite| 0.42 - ante*0.002       |
    | Control radius                    | m         | 0.7-1.6 first touch  | 0.7 + ball_ctrl*0.009    |
    | Tackle radius                     | m         | 0.9-1.9 leg reach    | 0.9 + tackling*0.009     |
    | Jump (outfield)                   | m         | 0.35-0.80            | 0.35 + jumping*0.0045    |
    | Standing reach (outfield)         | m         | 1.6-1.9              | 1.55 + jumping*0.0025    |
    | GK standing reach                 | m         | 2.05-2.35 (+ arms)   | 2.05 + jumping*0.003     |
    | GK vertical reach on dive         | m         | 2.4-3.0              | capped 2.9               |
    | GK lateral dive extension         | m         | 1.0-2.2              | 1.0 + diving*0.012       |
    | GK lateral dive speed             | m/s       | 2.2-3.6              | 2.2 + reflexes*0.012     |
    | Ground pass speed (short->driven) | m/s       | 10-23                | 10 + short_passing*0.13  |
    | Shot speed                        | m/s       | 20-34 (avg ~25-30)   | 20 + shot_power*0.14     |
    | Aerial delivery                   | m/s       | 12-24                | 12 + long_passing*0.12   |

Checkpoint 28 — Full Possession Physics Redesign:
    * Continuous player motion during ALL actions (passes, carries, dribbles, shots)
    * Ball-in-flight tracking with per-tick position updates
    * Pressure-driven speed and accuracy modifiers
    * Comprehensive physics trace for analytics consumers
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Dict, Any

from geometry_engine import (
    MovingPlayer, Vec2, Vec3, BallFlight, BallSpin, PassResolution,
    DribbleResolution, ShotResolution, AerialResolution, AerialPassResolution,
    make_flight, make_ballistic_flight, resolve_aerial_delivery,
    resolve_aerial_pass, resolve_ground_pass, resolve_dribble, resolve_shot,
    GoalkeeperState, BALL_ROLLING_DECEL, rolling_position, rolling_speed,
    rolling_arrival_time, resolve_body_duel, _body_contest_win_probability,
    _approach_speed_at, _race_motion, PlayerRaceContext, BodyDuel,
    MOVEMENT_DEAD_BAND_MPS,
)

# ─────────────────────────────────────────────
# SHARED CLOCK
# ─────────────────────────────────────────────
TICK_S = 0.1  # possession episodes step at 10 Hz

# ─────────────────────────────────────────────
# CALIBRATION — real-football mappings
# (documented in the module docstring)
# ─────────────────────────────────────────────

def sprint_speed(pace: float) -> float:
    """Pace 0..100 -> 5.0..9.2 m/s (18..33 km/h)."""
    return 5.0 + max(0.0, min(100.0, pace)) * 0.042


def acceleration(accel: float) -> float:
    """Acceleration rating 0..100 -> 3.0..6.0 m/s^2."""
    return 3.0 + max(0.0, min(100.0, accel)) * 0.030


def reaction_time(anticipation: float, reflexes: float = 60.0) -> float:
    """0.42s (slow reader) .. 0.15s (elite reflexes)."""
    base = 0.42 - max(0.0, min(100.0, anticipation)) * 0.002
    return base * (1.0 - max(0.0, min(100.0, reflexes)) * 0.002)


def control_radius(ball_control: float) -> float:
    """First-touch / dribble control footprint 0.7..1.6 m."""
    return 0.7 + max(0.0, min(100.0, ball_control)) * 0.009


def tackle_radius(tackling: float) -> float:
    """Leg/reach tackle envelope 0.9..1.9 m."""
    return 0.9 + max(0.0, min(100.0, tackling)) * 0.009


def jump_height(jumping: float) -> float:
    """Vertical leap 0.35..0.80 m."""
    return 0.35 + max(0.0, min(100.0, jumping)) * 0.0045


def standing_reach(jumping: float, is_gk: bool) -> float:
    if is_gk:
        return min(2.35, 2.05 + max(0.0, min(100.0, jumping)) * 0.003)
    return 1.55 + max(0.0, min(100.0, jumping)) * 0.0025


def vertical_reach_total(jumping: float, is_gk: bool) -> float:
    reach = standing_reach(jumping, is_gk) + jump_height(jumping)
    return min(2.9, reach) if is_gk else min(2.65, reach)


def dive_extension(diving: float) -> float:
    """Lateral distance a keeper covers beyond reach on a dive: 1.0..2.2 m."""
    return 1.0 + max(0.0, min(100.0, diving)) * 0.012


def dive_speed(reflexes: float) -> float:
    """Keeper horizontal dive speed 3.4..5.6 m/s (explosive full-stretch)."""
    return 3.4 + max(0.0, min(100.0, reflexes)) * 0.022


def ground_pass_speed(short_passing: float, driven: float = 0.0) -> float:
    """10 (loose) .. 23 (driven) m/s; pressing adds pace."""
    return max(10.0, 10.0 + max(0.0, min(100.0, short_passing)) * 0.13 + driven)


def shot_speed(shot_power: float) -> float:
    """20..34 m/s; average shots land ~25-30 m/s."""
    return max(20.0, 20.0 + max(0.0, min(100.0, shot_power)) * 0.14)


def header_shot_speed(power: float) -> float:
    """5..12 m/s launch velocity for headed efforts (firm headers 7-10 m/s).

    Headers carry far less velocity than struck shots — the ball is redirected
    off the noggin, not driven by a leg swing. Used for the Opta-style shot
    speed stamp on corner/header goals and efforts.
    """
    return max(5.0, 5.0 + max(0.0, min(100.0, power)) * 0.07)


def aerial_delivery_speed(long_passing: float) -> float:
    """12 (lofted) .. 24 (rapid cross) m/s."""
    return max(12.0, 12.0 + max(0.0, min(100.0, long_passing)) * 0.12)


def _clamped(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rolling_decel_for(weather=None) -> float:
    """Weather-scaled ground-pass rolling resistance (m/s²).

    Wet/waterlogged turf grips the ball harder, so passes bite and die
    earlier. Neutral when WeatherPhysics is disabled (returns the dry-turf
    baseline ``BALL_ROLLING_DECEL``).
    """
    from weather_physics import WeatherPhysics
    return BALL_ROLLING_DECEL * WeatherPhysics.rolling_decel_mult(weather)


def delivery_spin(
    skill: float,
    rng: Optional[random.Random] = None,
    kind: str = "side",
) -> BallSpin:
    """Skill-scaled spin for a delivery (curler / whipped cross / floated chip).

    ``skill`` is 0..100 (finishing / crossing / long-passing / free_kick).
    Poor technicians hit dead-straight unspun balls at the top end of the
    boot; elite one-timers and set-piece specialists put serious rotation on
    the ball — 15..70 rad/s (≈150..670 rpm), which is the real Magnus range.
    Returns a neutral dead ball for ``skill <= 0``.
    """
    _rng = rng or random
    skill = max(0.0, min(100.0, skill))
    if skill <= 0.0:
        return BallSpin()
    rate = 12.0 + skill * 0.58 * _rng.uniform(0.7, 1.35)
    if kind == "top":
        return BallSpin.topspin(rate)
    if kind == "back":
        return BallSpin.backspin(rate)
    sign = -1.0 if _rng.random() < 0.5 else 1.0
    return BallSpin.side(rate, sign)


def calculate_pressure_level(
    ball_x: float,
    ball_y: float,
    defenders: Iterable[MovingPlayer],
    max_pressure_dist: float = 5.0,
) -> float:
    """Calculate pressure level from 0.0 (no pressure) to 1.0 (intense pressure).
    
    Based on the distance and number of nearby defenders.
    Real football: pressure is highest when multiple defenders are within 2-3m.
    """
    pressure = 0.0
    nearby_count = 0
    
    for defender in defenders:
        if defender.is_goalkeeper:
            continue
        dist = math.hypot(
            defender.position.x - ball_x,
            defender.position.y - ball_y,
        )
        if dist < max_pressure_dist:
            # Contribution inversely proportional to distance
            contribution = 1.0 - (dist / max_pressure_dist)
            pressure += contribution * 0.5  # Each defender adds pressure
            nearby_count += 1
    
    # Multiple defenders compound pressure
    if nearby_count >= 2:
        pressure *= 1.3
    if nearby_count >= 3:
        pressure *= 1.2
    
    return min(1.0, pressure)


# ─────────────────────────────────────────────
# PER-TICK PLAYER STATE
# ─────────────────────────────────────────────

@dataclass
class PhysPlayer:
    """Mutable per-tick state wrapping the immutable MovingPlayer identity."""

    ref: object
    name: str
    motion: MovingPlayer
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    is_gk: bool = False


@dataclass
class MotionSnapshot:
    """One 0.1 s snapshot of the possession episode's motion."""

    tick: float
    player: str
    x: float
    y: float
    speed_mps: float
    ball_x: float
    ball_y: float
    ball_z: float = 0.0
    note: str = ""


@dataclass
class PressureSnapshot:
    """Snapshot of pressure state at a moment in the episode."""
    
    tick: float
    pressure_level: float
    nearest_defender_dist: float
    defenders_within_5m: int


# ─────────────────────────────────────────────
# AIM & TRAJECTORY HELPERS
# ─────────────────────────────────────────────

GOAL_LEFT = 30.34
GOAL_RIGHT = 37.66
GOAL_CENTER = 34.0
GOAL_HEIGHT = 2.44


def aim_shot_flight(
    shooter_dna,
    x: float,
    y: float,
    body_part: str,
    attacks_right: bool,
    rng: Optional[random.Random] = None,
    under_pressure: bool = False,
    pressure_level: float = 0.0,
) -> BallFlight:
    """Choose a goal-plane target and return a ballistic flight for it.

    The shooter aims with a skill-driven placement bias plus a genuine
    placement-error distribution (sigma grows with distance, shrinks with
    finishing/composure and worsens under pressure). The target geometry —
    NOT a probability roll — then decides whether the crossing is in frame,
    on the woodwork, or off target (wide / over the bar).
    
    Enhanced with pressure-driven accuracy degradation.
    """
    _rng = rng or random
    goal_x_plane = 105.0 if attacks_right else 0.0
    dist = math.hypot(goal_x_plane - x, GOAL_CENTER - y)

    finishing = float(getattr(getattr(shooter_dna, "technical", None), "finishing", 50.0))
    long_shots = float(getattr(getattr(shooter_dna, "technical", None), "long_shots", 45.0))
    composure = float(getattr(getattr(shooter_dna, "mental", None), "composure", 60.0))
    power = float(getattr(getattr(shooter_dna, "physical", None), "strength", 60.0))

    placement_skill = (finishing * 0.55 + composure * 0.25 + long_shots * 0.20) / 100.0

    # Closeness = rushed: from 6 yards there is no time to pick a corner, so
    # shots stay centralised but sloppier; from distance the shooter has time
    # to choose and spots the near/far post.
    leisure = min(1.0, dist / 16.0)

    # Placement draw: skilled finishers favour the corners; the natural side
    # of the pitch biases the chosen post slightly. Only available when the
    # shooter isn't crowded.
    natural_side = -1.0 if _rng.random() < 0.5 else 1.0
    corner_draw = natural_side * (0.2 + placement_skill * 1.5) * leisure * _rng.random()

    # Aim error: grows with distance, shrinks with skill; pressure hurts.
    pressure_w = 1.35 if under_pressure else 1.0
    # Additional pressure degradation based on pressure_level
    pressure_accuracy_mult = 1.0 + pressure_level * 0.5
    pressure_w *= pressure_accuracy_mult
    # Wide angles: the visible goal is a narrow aperture, so absolute-Y
    # accuracy collapses — crosses-body shots fly well wide of the frame.
    width_w = 1.0 + max(0.0, (abs(y - GOAL_CENTER) - 10.0) / 14.0)
    sigma_y = (2.4 + dist / 8.0) * (1.6 - placement_skill * 0.35) * pressure_w * width_w
    target_y = GOAL_CENTER + corner_draw + _rng.gauss(0.0, sigma_y)

    if body_part == "head":
        target_z = 0.9 + placement_skill * 0.7 + _rng.gauss(0.0, 0.35)
        apex_z = max(2.2, target_z * 1.6 + 0.5)
    else:
        low_share = 0.55 + placement_skill * 0.30
        low = _rng.random() < low_share
        base_z = _rng.uniform(0.08, 0.45) if low else _rng.uniform(0.7, 1.5)
        # Rushed shots balloon — vertical error grows sharply up close.
        sigma_z = 0.45 + (0.15 if low else 0.65) + (1.4 - placement_skill) * 0.6
        sigma_z += 1.1 * (1.0 - leisure)
        # Pressure adds vertical error
        sigma_z *= pressure_accuracy_mult
        target_z = base_z + _rng.gauss(0.0, sigma_z)
        apex_z = max(1.4, target_z * 1.9 + dist * 0.035)

    speed = shot_speed(power * 0.4 + finishing * 0.6) * (0.62 + 0.38 * leisure)
    
    # Pressure can cause rushed/hit shots (slightly different speed)
    if pressure_level > 0.5:
        speed *= (1.0 + _rng.uniform(-0.05, 0.08))  # Less control over power

    # Magnus: quality strikers curl the ball — an outside-of-the-boot curler
    # and a dead-straight strike at the same speed/angle now fly DIFFERENT
    # trajectories (the banana bend is real geometry, not a probability
    # boost). Skill drives spin rate; scuffed/laboured efforts stay straighter.
    spin = delivery_spin(
        finishing * 0.6 + composure * 0.4,
        _rng,
        kind="top" if (_rng.random() < 0.08 and dist > 18.0) else "side",
    )

    flight = make_flight(
        Vec3(x, y, 0.05),
        Vec3(goal_x_plane, target_y, max(0.0, target_z)),
        speed,
        apex_z=apex_z,
        spin=spin,
    )
    return flight


def build_ground_flight(
    start: Vec2, target: Vec2, speed: float, apex_ratio: float = 0.0
) -> BallFlight:
    """A mostly-flat pass flight; apex_ratio lifts the middle arc."""
    distance = start.distance_to(target)
    apex = 0.0
    if apex_ratio > 0.0:
        apex = distance * apex_ratio
    flight = make_flight(Vec3(start.x, start.y, 0.0), Vec3(target.x, target.y, 0.0), speed, apex_z=apex)
    flight.duration = distance / max(6.0, speed)
    return flight


# ─────────────────────────────────────────────
# POSSESSION EPISODE — the shared 0.1s clock
# ─────────────────────────────────────────────

class PossessionEpisode:
    """Resolve a full possession episode on a shared 10 Hz clock.

    The chain makes the tactical decisions (who, where, which action); this
    engine is the outcome authority. Every action advances the same
    ``elapsed`` clock in 0.1 s ticks, moves every tracked player continuously,
    and records a motion trace for analytics/exporters to consume.
    
    Enhanced with:
        * Per-tick ball position tracking during all actions
        * Pressure calculation and application
        * Comprehensive physics trace for analytics
    """

    dt: float = TICK_S

    # Fraction of top speed used to LABEL sustained race/pursuit effort in
    # the trace. A player accelerating to meet a pass, track a runner, or
    # press a carrier sustains roughly this share of his absolute top speed
    # over the race; only the quickest, most decisive duels go higher. The
    # label drives sprint/HSR band classification (see calculate_distance_stats)
    # so that races by top-end athletes land in the sprint band while most
    # work rates land in the high-speed-running band. Positions and race
    # timings are untouched.
    RACE_EFFORT = 0.85

    def __init__(self, dt: float = TICK_S):
        self.dt = float(dt)
        self.elapsed = 0.0
        self.players: dict[str, PhysPlayer] = {}
        self.trace: List[MotionSnapshot] = []
        # Persistent whole-possession ball path. Unlike _ball_trajectory
        # (which resets every single action), this accumulates EVERY tick of
        # ball motion for the entire episode — so a possession reads
        # carry -> pass -> carry as one continuous, time-ordered trace.
        self.ball_path: List[Dict[str, Any]] = []
        self.ball_x = 52.5
        self.ball_y = 34.0
        self.ball_z = 0.0
        self.notes: List[tuple[float, str, str]] = []
        self.pressure_history: List[PressureSnapshot] = []
        self._rng = random.Random(random.random())
        self._ball_trajectory: List[Tuple[float, Vec3]] = []
        # Per-player current race speed (m/s) during pass/dribble races, so a
        # player ramps to top speed (like a real sprint) instead of being
        # teleported at full speed by a lerp. Resets each episode.
        self._race_speed: dict[str, float] = {}
        # #3 Live per-player velocity memory: last measured (vx, vy) and the
        # (x, y, ts) they were measured at. The NEXT action that races these
        # players reads their real heading + entry speed here, so the geometry
        # engine applies turning cost and momentum carry (a player who is
        # already running toward the ball arrives earlier than one who must
        # pivot from a standstill). Curated per-tick deltas, so teleports and
        # stale skips don't poison it.
        self._vel: dict[str, Tuple[float, float]] = {}
        self._last_seen: dict[str, Tuple[float, float, float]] = {}

    # ── lifecycle ─────────────────────────────────────────────────
    def register(self, moving_players: Iterable[MovingPlayer]) -> None:
        """Add moving players to the episode's continuous state."""
        for mp in moving_players:
            if mp is None:
                continue
            name = getattr(mp.player, "name", str(mp.player))
            if name in self.players:
                continue
            self.players[name] = PhysPlayer(
                ref=mp.player,
                name=name,
                motion=mp,
                x=mp.position.x,
                y=mp.position.y,
                is_gk=mp.is_goalkeeper,
            )

    # ── #3 velocity memory ─────────────────────────────────────────────
    def _track_motion(self, name: str, x: float, y: float, ts: float) -> None:
        """Record a live per-player footstep, deriving real heading + speed.

        Only 0.03..0.5 s real-time steps feed the velocity (per-tick motion
        within an action); anything bigger is a hop/teleport and must not
        fabricate a sprint. A zero-motion sample clears the memory (the player
        stopped, so their next race starts from rest).
        """
        prev = self._last_seen.get(name)
        if prev is not None:
            dt = ts - prev[2]
            if 0.03 <= dt <= 0.5:
                dx = x - prev[0]
                dy = y - prev[1]
                step = math.hypot(dx, dy)
                if step > 1e-6:
                    self._vel[name] = (dx / dt, dy / dt)
                else:
                    self._vel[name] = (0.0, 0.0)
        self._last_seen[name] = (x, y, ts)

    def _velocity_of(self, name: str) -> Tuple[Optional[Vec2], Optional[float]]:
        """(heading unit Vec2, speed m/s) for the player's last known motion.

        Dead-band shuffles (< MOVEMENT_DEAD_BAND_MPS) and "never moved" are
        ``(None, None)`` -> the geometry engine treats the race as from rest.
        """
        v = self._vel.get(name)
        if v is None:
            return (None, None)
        vx, vy = v
        speed = math.hypot(vx, vy)
        if speed < MOVEMENT_DEAD_BAND_MPS:
            return (None, None)
        return (Vec2(vx / speed, vy / speed), speed)

    def _build_race_context(
        self, moving_players: Iterable[Optional[MovingPlayer]],
    ) -> Optional[PlayerRaceContext]:
        """Live heading/entry-speed context for a set of runners, keyed by id.

        Returns ``None`` when no one has motion history (from-rest baseline —
        the geometry resolvers then behave exactly as before, so standalone
        unit tests that never simulate footsteps are unchanged).
        """
        ctx: PlayerRaceContext = {}
        for mp in moving_players:
            if mp is None:
                continue
            name = getattr(mp.player, "name", str(mp.player))
            heading, speed = self._velocity_of(name)
            if heading is not None or speed is not None:
                ctx[id(mp)] = (heading, speed)
        return ctx or None

    def _approach_for(
        self, player: MovingPlayer, elapsed: float,
    ) -> float:
        """Sprint speed the player can actually produce at ``elapsed`` sec into
        a race: accel-ramp from rest, OR the live entry speed he carried into
        the race from the velocity memory (#3), whichever is higher (capped at
        top speed). The body-tie impulses use this, so a player who was already
        sprinting arrives with more momentum than an identical one from rest.
        """
        name = getattr(player.player, "name", str(player.player))
        _heading, spd = self._velocity_of(name)
        entry = 0.0 if spd is None else spd
        return min(player.top_speed, max(entry, _approach_speed_at(player, max(0.0, elapsed))))

    def set_ball(self, x: float, y: float, z: float = 0.0) -> None:
        self.ball_x = float(x)
        self.ball_y = float(y)
        self.ball_z = float(z)

    def record_ball_move(
        self,
        from_x: float, from_y: float,
        to_x: float, to_y: float,
        speed_mps: float,
        label: str = "move",
        pressure_level: float = 0.0,
        rolling_decel: Optional[float] = None,
    ) -> None:
        """Record a timed, per-tick ball flight into the PERSISTENT whole-
        possession path (unlike _ball_trajectory, which resets per action).

        Advances the ball in 0.1 s steps from (from_x, from_y) to
        (to_x, to_y) at ``speed_mps``, appending one point per tick. Time is
        monotonic across the entire episode, so a possession reads
        carry -> pass -> carry as one continuous trace. Used for action types
        that resolve without the geometry engine (e.g. the roll-based pass
        fallback when no position engine is wired in).

        Ground friction: like the geometry race, the ball rolls with constant
        deceleration (``rolling_decel``, default dry-turf baseline), so it
        arrives slower than it left and an under-struck ball dies short of
        ``to_x``/``to_y`` (the trace simply stops at its rest point).
        """
        distance = math.hypot(to_x - from_x, to_y - from_y)
        speed = max(1.0, float(speed_mps))
        k = BALL_ROLLING_DECEL if rolling_decel is None else max(0.0, rolling_decel)
        arrival = rolling_arrival_time(distance, speed, k)
        stopped_short = arrival is None
        if stopped_short:
            rest_dist = speed * speed / (2.0 * k) if k > 0 else 0.0
            travel = speed / k if k > 0 else 0.0
            end_x = from_x + (to_x - from_x) * min(1.0, rest_dist / distance) if distance else to_x
            end_y = from_y + (to_y - from_y) * min(1.0, rest_dist / distance) if distance else to_y
        else:
            travel = arrival
            end_x, end_y = to_x, to_y
        if travel <= 0.0:
            self.ball_path.append({
                "t": round(self.elapsed, 3), "x": round(end_x, 2), "y": round(end_y, 2),
                "kind": label, "p": round(pressure_level, 3),
            })
            self.set_ball(end_x, end_y)
            return
        steps = max(1, int(math.ceil(travel / self.dt)))
        for i in range(1, steps + 1):
            self.elapsed += self.dt
            t_now = travel * i / steps
            covered = rolling_position(speed, k, t_now)
            fx = min(1.0, covered / distance) if distance else 1.0
            bx = from_x + (to_x - from_x) * fx
            by = from_y + (to_y - from_y) * fx
            self.ball_path.append({
                "t": round(self.elapsed, 3), "x": round(bx, 2), "y": round(by, 2),
                "kind": label, "p": round(pressure_level, 3),
            })
        self.set_ball(end_x, end_y)

    def step_off_ball(
        self, ball_x: float, ball_y: float, dt: Optional[float] = None,
        speed: float = 2.0, exclude: Optional[set] = None,
    ) -> None:
        """Continuous off-ball movement (#2). Every registered player who is
        NOT in the current on-ball duel compacts toward the ball, so the whole
        team shape tracks play tick-by-tick instead of freezing between
        actions. Purely positional — it does not affect on-ball outcomes (those
        are decided by the position engine + race-to-ball). It enriches the
        motion trace and distance-covered stats.
        """
        dt = self.dt if dt is None else dt
        ex = exclude or set()
        for name, p in self.players.items():
            if name in ex:
                continue
            dx = (ball_x - p.x) * 0.08
            dy = (ball_y - p.y) * 0.04
            step_len = math.hypot(dx, dy)
            max_step = speed * dt
            if step_len > max_step and step_len > 0:
                dx *= max_step / step_len
                dy *= max_step / step_len
            p.x = _clamped(p.x + dx, 0.0, 105.0)
            p.y = _clamped(p.y + dy, 0.0, 68.0)

    def resolve_loose_ball(
        self,
        x: float, y: float,
        attackers: Iterable[MovingPlayer],
        defenders: Iterable[MovingPlayer],
        gravity: float = 0.0,
    ) -> str:
        """Continuous loose-ball / second-ball resolution (#4). A ball that is
        released (saved, blocked, miscontrolled) becomes a live entity that
        nearby players race toward on the shared 0.1 s clock; whoever reaches
        it first wins possession. Returns the winning team label
        (``"attack"`` / ``"defence"``) or ``"none"``.

        Body physics (#4): when the best attacker and best defender arrive at
        the same ball at the same instant (a 50/50 tie window), the shoulder
        decides — mass x momentum, not a coin flip. The heavier, faster, better
        balanced arrival shoves their way in; comparable impulses and physique
        leave the genuine gray zone to the impulse-weighted roll.
        """
        self.set_ball(x, y)
        att = [a for a in attackers if a is not None]
        defe = [d for d in defenders if d is not None]
        race_ctx = self._build_race_context(att + defe)
        best_team = "none"
        best_time = float("inf")
        for plist, team in ((att, "attack"), (defe, "defence")):
            for p in plist:
                tt = _race_motion(p, Vec2(x, y), p.control_radius, race_ctx)
                if tt < best_time:
                    best_time = tt
                    best_team = team
        # 50/50 tie window: if the best attacker and best defender reach the
        # same ball at the same instant, the shoulder decides. Body mass x
        # momentum (both approach speeds are accel-ramp/entry-aware) replaces
        # the coin flip that a rounding hair used to settle.
        if best_team != "none" and att and defe:
            best_att = min(
                _race_motion(p, Vec2(x, y), p.control_radius, race_ctx) for p in att
            )
            best_def = min(
                _race_motion(p, Vec2(x, y), p.control_radius, race_ctx) for p in defe
            )
            if abs(best_att - best_def) < 0.18:
                a = min(att, key=lambda p: _race_motion(p, Vec2(x, y), p.control_radius, race_ctx))
                d = min(defe, key=lambda p: _race_motion(p, Vec2(x, y), p.control_radius, race_ctx))
                duel = resolve_body_duel(d, a,
                                         self._approach_for(d, best_time),
                                         self._approach_for(a, best_time))
                if duel.winner == "defender":
                    best_team = "defence"
                elif duel.winner == "carrier":
                    best_team = "attack"
                else:
                    # Impulse-weighted gray zone: higher defender impulse tilts
                    # the roll toward the defending side.
                    if self._rng.random() >= _body_contest_win_probability(duel.impulse_ratio):
                        best_team = "attack"
                    else:
                        best_team = "defence"
        # Animate the scramble so the trace shows the loose ball being chased.
        if best_time < float("inf"):
            steps = max(1, int(math.ceil(best_time / self.dt)))
            for i in range(1, steps + 1):
                self.elapsed += self.dt
                self.ball_path.append({
                    "t": round(self.elapsed, 3), "x": round(x, 2), "y": round(y, 2),
                    "kind": "loose", "p": 0.0,
                })
        return best_team

    def note(self, label: str, detail: str = "") -> None:
        self.notes.append((self.elapsed, label, detail))

    def _snap(self, name: str, x: float, y: float, speed: float, note: str = "", z: float = 0.0) -> None:
        self.trace.append(MotionSnapshot(
            tick=round(self.elapsed, 3), player=name, x=round(x, 2),
            y=round(y, 2), speed_mps=round(speed, 2),
            ball_x=round(self.ball_x, 2), ball_y=round(self.ball_y, 2),
            ball_z=round(self.ball_z if z == 0.0 else z, 2), note=note,
        ))

    def _record_pressure(self, pressure: float, defenders: Iterable[MovingPlayer]) -> None:
        """Record pressure state for analytics."""
        nearest = float('inf')
        count = 0
        for d in defenders:
            if d.is_goalkeeper:
                continue
            dist = math.hypot(d.position.x - self.ball_x, d.position.y - self.ball_y)
            if dist < nearest:
                nearest = dist
            if dist < 5.0:
                count += 1
        self.pressure_history.append(PressureSnapshot(
            tick=self.elapsed,
            pressure_level=pressure,
            nearest_defender_dist=nearest if nearest < float('inf') else 99.0,
            defenders_within_5m=count,
        ))

    def advance(self, seconds: float) -> None:
        """Motion-inert tick law — everyone drifts toward a home anchor at
        accel-limited speed. Used to keep off-ball players alive
        between resolved moments (continuity, not teleportation)."""
        ticks = max(1, int(round(seconds / self.dt)))
        for _ in range(ticks):
            self.elapsed += self.dt
            for p in self.players.values():
                home_x = getattr(getattr(p.ref, "home_x", None), "x", None)
                home_y = getattr(getattr(p.ref, "home_y", None), "y", None)
                # Fall back to the MovingPlayer's static position.
                if home_x is None:
                    tx, ty = p.motion.position.x, p.motion.position.y
                else:
                    tx, ty = home_x, home_y
                dx, dy = tx - p.x, ty - p.y
                dist = math.hypot(dx, dy)
                heading = math.atan2(dy, dx) if dist > 1e-6 else 0.0
                target_speed = min(p.motion.top_speed * 0.75, max(0.0, dist - 0.5))
                new_speed = min(
                    target_speed,
                    math.hypot(p.vx, p.vy) + p.motion.acceleration * self.dt,
                )
                p.vx = math.cos(heading) * new_speed
                p.vy = math.sin(heading) * new_speed
                p.x = _clamped(p.x + p.vx * self.dt, 0.0, 105.0)
                p.y = _clamped(p.y + p.vy * self.dt, 0.0, 68.0)
            self._snap("ball", self.ball_x, self.ball_y, 0.0, note="drift")

    # ── race-to-ball (passes / through balls) ──────────────────────
    def resolve_ground_pass(
        self,
        start: Vec2,
        target: Vec2,
        receiver: MovingPlayer,
        defenders: Iterable[MovingPlayer],
        ball_speed: float,
        ball_owner_team: str = "",
        pressure_level: float = 0.0,
        rolling_decel: Optional[float] = None,
    ) -> PassResolution:
        """Alias for resolve_pass — event_chain calls this name."""
        return self.resolve_pass(start, target, receiver, defenders, ball_speed,
                                 ball_owner_team=ball_owner_team,
                                 pressure_level=pressure_level,
                                 rolling_decel=rolling_decel)

    def resolve_pass(
        self,
        start: Vec2,
        target: Vec2,
        receiver: MovingPlayer,
        defenders: Iterable[MovingPlayer],
        ball_speed: float,
        ball_owner_team: str = "",
        pressure_level: float = 0.0,
        rolling_decel: Optional[float] = None,
    ) -> PassResolution:
        """Continuous 0.1 s race-to-ball pass resolution (see module doc).

        Ground friction: the ball rolls with constant deceleration
        (``rolling_decel``), so long passes arrive slower than they left the
        boot and under-struck balls genuinely die short of the target. The
        per-tick trace is driven by the geometry resolution's friction-
        corrected trajectory so the whole-possession clock matches the race's
        physics, not a flat-distance lerp.
        
        Enhanced with:
        * Per-tick ball trajectory tracking
        * Pressure-driven speed and accuracy modifiers
        * Continuous defender motion updates
        """
        self.set_ball(start.x, start.y)
        self.note("pass", f"{getattr(receiver.player,'name','receiver')} {round(ball_speed,1)}m/s")
        
        # Record initial pressure
        defenders_list = list(defenders)
        self._record_pressure(pressure_level, defenders_list)
        recv_name = getattr(receiver.player, "name", "receiver")
        # Live heading/entry-speed context for this race from the velocity
        # memory (#3): a defender already tracking the ball runs it in faster
        # than one who must pivot from a standstill.
        race_ctx = self._build_race_context([receiver] + defenders_list)
        receiver_time = _race_motion(receiver, target, None, race_ctx)

        resolution = resolve_ground_pass(
            start, target, receiver, defenders_list, ball_speed,
            sample_step=self.dt, pressure_level=pressure_level,
            rolling_decel=rolling_decel, player_context=race_ctx,
        )
        
        # Trace the whole flight on the shared clock with per-tick ball
        # tracking, driven by the geometry's friction-corrected trajectory.
        speed = max(6.0, ball_speed)
        
        # Apply pressure-based speed degradation
        if pressure_level > 0.5:
            speed *= (1.0 - 0.1 * (pressure_level - 0.5))
        
        base_elapsed = self.elapsed
        
        # Track ball trajectory
        self._ball_trajectory = []
        traj = resolution.ball_trajectory or []
        last_ball = resolution.contact_point if traj else target
        
        for idx, (ts, ball) in enumerate(traj):
            self.elapsed = base_elapsed + max(0.0, ts)
            self.ball_x, self.ball_y = ball.x, ball.y
            self._ball_trajectory.append((self.elapsed, Vec3(ball.x, ball.y, 0.0)))
            last_ball = ball
            
            # Update receiver position
            rx = receiver.position.x
            ry = receiver.position.y
            if receiver_time > 0:
                rr = min(1.0, ts / max(0.05, receiver_time))
                rx = receiver.position.x + (target.x - receiver.position.x) * rr
                ry = receiver.position.y + (target.y - receiver.position.y) * rr
            self._track_motion(recv_name, rx, ry, self.elapsed)
            # Label the race as an acceleration/deceleration ARCH (peak at
            # ~55% of the race, trailing off into the control touch) instead
            # of a sustained plateau at RACE_EFFORT*top. A real receiver
            # decelerates to meet the ball; sustaining top effort across the
            # whole race classified far too much ground as HSR/sprint.
            _frac = min(1.0, rr if receiver_time > 0 else 1)
            _arch = min(_frac / 0.55, (1.0 - _frac) / 0.45) if _frac > 0 else 0.0
            self._snap(recv_name, rx, ry,
                       receiver.top_speed * self.RACE_EFFORT * max(0.0, _arch),
                       note="pass_race")
            
            # Update defender positions
            for d in defenders_list:
                if d is None:
                    continue
                def_name = getattr(d.player, "name", "defender")
                day = _race_motion(d, ball, d.control_radius, race_ctx)
                dr = min(1.0, ts / max(0.05, day)) if day > 0 else 1.0
                dxp = d.position.x + (ball.x - d.position.x) * dr
                dyp = d.position.y + (ball.y - d.position.y) * dr
                self._track_motion(def_name, dxp, dyp, self.elapsed)
                self._snap(def_name, dxp, dyp,
                           d.top_speed * self.RACE_EFFORT
                           * max(0.0, min(dr / 0.55, (1.0 - dr) / 0.45)),
                           note="intercept_race")

            # Off-ball players compact toward the ball each tick (#2)
            ex = {getattr(receiver.player, "name", "receiver")}
            ex.update(getattr(d.player, "name", "") for d in defenders_list)
            self.step_off_ball(ball.x, ball.y, exclude=ex)

        # Fold the whole flight into the persistent whole-possession path.
        for (t, v3) in self._ball_trajectory:
            self.ball_path.append({
                "t": round(t, 3), "x": round(v3.x, 2), "y": round(v3.y, 2),
                "kind": "pass", "p": round(pressure_level, 3),
            })
        self._snap("ball", last_ball.x, last_ball.y, speed, note="pass_end")
        return resolution

    # ── dribble / take-on (tackle radius + contact window) ─────────
    def resolve_dribble(
        self,
        start: Vec2,
        target: Vec2,
        attacker: MovingPlayer,
        defenders: Iterable[MovingPlayer],
        contact_window: float = 0.14,
        pressure_level: float = 0.0,
    ) -> DribbleResolution:
        """Continuous dribble: carrier motion vs defender pursuit, settled by
        tackle radius and a sustained contact window rather than a roll.
        Simultaneous contact is decided by body mass x momentum (#4).
        
        Enhanced with:
        * Continuous defender motion tracking
        * Pressure-driven tackle window calculation
        * Movement-cost-aware pursuit (turn costs via live heading context)
        * Body-collision / momentum duels on the 50/50 tie
        """
        self.set_ball(start.x, start.y)
        self.note("dribble", "")
        
        defenders_list = list(defenders)
        self._record_pressure(pressure_level, defenders_list)
        race_ctx = self._build_race_context([attacker] + defenders_list)

        resolution = resolve_dribble(
            start, target, attacker, defenders_list, sample_step=self.dt,
            pressure_level=pressure_level,
            player_context=race_ctx, rng=self._rng,
        )
        
        # Trace the carry on the shared clock with defender tracking
        distance = start.distance_to(target)
        dribble_speed = attacker.top_speed * 0.68
        duration = distance / max(2.5, dribble_speed)
        steps = max(1, int(math.ceil(duration / self.dt)))
        att_name = getattr(attacker.player, "name", "attacker")
        
        for i in range(1, steps + 1):
            self.elapsed += self.dt
            t = min(1.0, i / steps)
            ball = start.lerp(target, t)
            self.ball_x, self.ball_y = ball.x, ball.y
            self._track_motion(att_name, ball.x, ball.y, self.elapsed)
            
            self._snap(att_name,
                       ball.x, ball.y, dribble_speed, note="carry")

            # Fold the carry into the persistent whole-possession path.
            self.ball_path.append({
                "t": round(self.elapsed, 3), "x": round(ball.x, 2), "y": round(ball.y, 2),
                "kind": "carry", "p": round(pressure_level, 3),
            })

            # Track all defenders during dribble
            for d in defenders_list:
                if d is None:
                    continue
                def_name = getattr(d.player, "name", "defender")
                tackle_range = attacker.control_radius + d.tackle_radius
                arrival = _race_motion(d, ball, tackle_range, race_ctx)
                if arrival <= self.elapsed:
                    self._snap(def_name,
                               ball.x, ball.y,
                               d.top_speed * self.RACE_EFFORT, note="contact")
                else:
                    # Still moving toward contact point
                    dr = min(1.0, self.elapsed / max(0.05, arrival)) if arrival > 0 else 1.0
                    dxp = d.position.x + (ball.x - d.position.x) * dr
                    dyp = d.position.y + (ball.y - d.position.y) * dr
                    self._track_motion(def_name, dxp, dyp, self.elapsed)
                    self._snap(def_name, dxp, dyp,
                               d.top_speed * self.RACE_EFFORT * dr, note="pursuit")

            # Off-ball players compact toward the ball each tick (#2)
            ex = {getattr(attacker.player, "name", "attacker")}
            ex.update(getattr(d.player, "name", "") for d in defenders_list)
            self.step_off_ball(ball.x, ball.y, exclude=ex)

        return resolution

    # ── standing duel / press contest (contact window) ─────────────
    def resolve_duel_contest(
        self,
        attacker: MovingPlayer,
        challenger: MovingPlayer,
        ball_x: float,
        ball_y: float,
        rng: Optional[random.Random] = None,
        pressure_level: float = 0.0,
    ) -> bool:
        """True if the attacker holds the ball against a pressing challenger.

        Geometry: the challenger must arrive inside tackle range faster than
        the carrier can push the ball beyond that range — and keep the
        contact window. No completion roll; skill enters only through radii,
        reaction and speed.
        
        Enhanced with pressure-aware contact window calculation.
        """
        _rng = rng or self._rng
        race_ctx = self._build_race_context([attacker, challenger])
        tackle_range = attacker.control_radius + challenger.tackle_radius
        challenger_arrival = _race_motion(challenger, Vec2(ball_x, ball_y), tackle_range, race_ctx)
        
        # How long does the carrier need to escape the tackle envelope?
        escape_wheel = max(1.5, attacker.top_speed * 0.55)
        escape_time = (tackle_range + attacker.reaction_time) / escape_wheel
        
        # Contact window is tighter under pressure
        window = 0.10 + attacker.reaction_time
        if pressure_level > 0.3:
            window *= (1.0 - 0.2 * pressure_level)
        
        if challenger_arrival <= escape_time + window:
            # Body edge (#4): if the presser's shoulder would bowl the carrier
            # clean off the ball (mass x momentum at the instant of contact),
            # skill checks are over — the challenge is won physically.
            duel = resolve_body_duel(
                challenger, attacker,
                challenger.top_speed * 0.85, attacker.top_speed * 0.42,
            )
            if duel.winner == "defender":
                return False
            if duel.winner == "carrier":
                return True
            # Weighted gray zone: a big, fast presser tilts the physical 50/50.
            if _rng.random() < _body_contest_win_probability(duel.impulse_ratio):
                return False
            # Defensive skill can close the final gap; offensive agility opens
            # it — still geometric (radii/reaction) rather than a flat roll.
            skill_balance = (challenger.tackle_radius - attacker.control_radius)
            margin = (escape_time + window) - challenger_arrival
            if margin < skill_balance * 0.15:
                return _rng.random() < 0.35
            return True
        return True

    # ── shot vs goalkeeper ─────────────────────────────────────────
    def resolve_shot(
        self,
        flight: BallFlight,
        goalkeeper: Optional[MovingPlayer],
        blockers: Iterable[MovingPlayer] = (),
        attacks_right: bool = True,
        pressure_level: float = 0.0,
    ) -> ShotResolution:
        """Resolve a shot by trajectory vs keeper reach/dive timing.

        ``build_shot_flight`` already aimed the target; the keeper's vertical
        reach and lateral dive envelope are checked against the flight in
        real time (geometry_engine.resolve_shot at 0.1 s steps).
        
        Enhanced with:
        * Continuous ball tracking in 3D
        * Goalkeeper positioning updates
        * Pressure effects on shot speed
        """
        self.set_ball(flight.start.x, flight.start.y, flight.start.z)
        self.note("shot", f"{round(flight.duration,2)}s flight")
        
        # Record pressure at shot moment
        if goalkeeper is not None:
            self._record_pressure(pressure_level, [goalkeeper])
        
        resolution = resolve_shot(
            flight, goalkeeper, blockers, attacks_right=attacks_right,
            sample_step=self.dt,
        )
        
        # Trace the full 3D flight on the shared clock
        steps = max(1, int(math.ceil(flight.duration / self.dt)))
        for i in range(1, steps + 1):
            self.elapsed += self.dt
            t = min(1.0, i / steps)
            p = flight.position_at(t)
            self.ball_x, self.ball_y, self.ball_z = p.x, p.y, p.z
            
            ball_speed = flight.speed_at(t * flight.duration)
            self._snap("ball", p.x, p.y, ball_speed, note="shot", z=p.z)

            # Fold the shot into the persistent whole-possession path.
            self.ball_path.append({
                "t": round(self.elapsed, 3), "x": round(p.x, 2), "y": round(p.y, 2),
                "kind": "shot", "p": round(pressure_level, 3),
            })

            # Track goalkeeper movement during shot
            if goalkeeper is not None:
                name = getattr(goalkeeper.player, "name", "GK")
                # GK tracks ball position (dive is a short ~60-70% effort burst
                # off his line, not a top-speed sprint).
                gk_x = goalkeeper.position.x
                gk_y = goalkeeper.position.y
                self._snap(name, gk_x, gk_y,
                           goalkeeper.top_speed * self.RACE_EFFORT,
                           note="gk_track")
        
        return resolution

    # ── aerial delivery (cross / long ball) ────────────────────────
    def resolve_aerial(
        self,
        flight: BallFlight,
        attackers: Iterable[MovingPlayer],
        defenders: Iterable[MovingPlayer],
    ) -> AerialResolution:
        """Resolve aerial delivery with 3D tracking and movement-cost-aware
        arrival (turn/heading context from the velocity memory #3)."""
        self.set_ball(flight.start.x, flight.start.y, flight.start.z)
        self.note("aerial", "")
        
        att = [a for a in attackers if a is not None]
        defe = [d for d in defenders if d is not None]
        race_ctx = self._build_race_context(att + defe)
        resolution = resolve_aerial_delivery(
            flight, att, defe, sample_step=self.dt, player_context=race_ctx,
        )
        
        # Trace 3D trajectory
        steps = max(1, int(math.ceil(flight.duration / self.dt)))
        for i in range(1, steps + 1):
            self.elapsed += self.dt
            t = min(1.0, i / steps)
            p = flight.position_at(t)
            self.ball_x, self.ball_y, self.ball_z = p.x, p.y, p.z
            self._snap("ball", p.x, p.y, 0.0, note="aerial_flight", z=p.z)
            self.ball_path.append({
                "t": round(self.elapsed, 3), "x": round(p.x, 2), "y": round(p.y, 2),
                "kind": "pass", "p": 0.0,
            })

        return resolution

    def resolve_aerial_pass(
        self,
        flight: BallFlight,
        receiver: MovingPlayer,
        defenders: Iterable[MovingPlayer],
        pressure_level: float = 0.0,
    ) -> AerialPassResolution:
        """Resolve a lofted / long pass as a 3D aerial race on the shared clock.

        The long-ball counterpart of ``resolve_pass``: wraps
        ``geometry_engine.resolve_aerial_pass`` (ballistic arc + jump timing +
        vertical reach at the landing-point duel) and traces the 3D flight into
        the persistent whole-possession ball path.
        """
        self.set_ball(flight.start.x, flight.start.y, flight.start.z)
        self.note("aerial_pass", f"{round(flight.duration,2)}s flight")

        defenders_list = [d for d in defenders if d is not None]
        self._record_pressure(pressure_level, defenders_list)
        race_ctx = self._build_race_context([receiver] + defenders_list)

        resolution = resolve_aerial_pass(
            flight, receiver, defenders_list, sample_step=self.dt,
            player_context=race_ctx,
        )

        steps = max(1, int(math.ceil(flight.duration / self.dt)))
        recv_name = getattr(receiver.player, "name", "receiver")
        for i in range(1, steps + 1):
            self.elapsed += self.dt
            t = min(1.0, i / steps)
            p = flight.position_at(t)
            self.ball_x, self.ball_y, self.ball_z = p.x, p.y, p.z
            self._snap("ball", p.x, p.y, flight.speed_at(t * flight.duration),
                       note="aerial_pass_flight", z=p.z)
            self.ball_path.append({
                "t": round(self.elapsed, 3), "x": round(p.x, 2), "y": round(p.y, 2),
                "kind": "pass", "p": round(pressure_level, 3),
            })

            # Trace receiver chase to the landing point.
            rr = min(1.0, self.elapsed / max(0.05, resolution.receiver_arrival_time)) \
                if resolution.receiver_arrival_time > 0 else 1.0
            rx = receiver.position.x + (flight.target.x - receiver.position.x) * rr
            ry = receiver.position.y + (flight.target.y - receiver.position.y) * rr
            self._track_motion(recv_name, rx, ry, self.elapsed)
            self._snap(recv_name, rx, ry,
                       receiver.top_speed * self.RACE_EFFORT, note="aerial_race")

            ex = {recv_name}
            ex.update(getattr(d.player, "name", "") for d in defenders_list)
            self.step_off_ball(p.x, p.y, exclude=ex)

        return resolution

    def resolve_long_pass(
        self,
        start: Vec2,
        target: Vec2,
        ball_speed: float,
        receiver: MovingPlayer,
        defenders: Iterable[MovingPlayer],
        landing_z: float = 1.2,
        loft: float = 0.0,
        gravity: float = 9.81,
        weather=None,
        pressure_level: float = 0.0,
        spin: Optional[BallSpin] = None,
    ) -> AerialPassResolution:
        """Long-ball counterpart of ``resolve_pass`` — a REAL ballistic race.

        This is the resolution long passes/switches route through in the
        chain. Two things make it genuinely different from a ground pass:

        * BALLISTIC FLIGHT — the ball leaves the boot at ``ball_speed`` m/s
          at a launch angle solved (via ``make_ballistic_flight``) against the
          range/landing-height geometry under gravity (9.81). Flight time and
          peak height are DERIVED from that launch velocity vector, not tuned.
          A harder-driven diagonal and a soft lofted switch come out with
          naturally different, physically consistent arcs; a kick too weak for
          the distance genuinely falls short.

        * AERIAL LANDING DUEL — the contest is who can physically put their
          body under the ball at the descent point (delegated to
          ``geometry_engine.resolve_aerial_pass``): each candidate's arrival
          time uses the acceleration curve + reaction (``time_to_reach``) and
          must beat the ball's remaining flight, and vertical reach
          (standing reach + jump height, both derived from the DNA
          ``physical.jumping`` attribute) must cover the ball height at that
          instant — the jump is timed (t = sqrt(2h/g)) to meet the ball.

        Weather: the landing target is deflected by
        ``WeatherPhysics.pass_lateral_deflection`` with ``is_airborne=True`` —
        the k_drag = 0.09 airborne drag regime long balls were built for, so a
        crosswind bends the landing point BEFORE the ballistic flight is
        built and the duel is contested at the BENT target, not the aim point.

        Gate policy — geometry-pure here, by design. Like
        ``resolve_ground_pass``/``resolve_dribble``, this returns the pure
        geometric verdict; the Checkpoint-28 attribute gate long balls DO get
        (decision documented at ``PossessionChain._aerial_pass_gate``) is
        layered on in the chain, exactly where the ground-pass interception
        gate lives. One voice (geometry) is never the whole story for a
        contested action.
        """
        from weather_physics import WeatherPhysics

        to_x, to_y = target.x, target.y
        drift_m = 0.0
        if WeatherPhysics.is_active(weather):
            to_x, to_y, drift_m = WeatherPhysics.pass_lateral_deflection(
                target.x, target.y, start.x, start.y,
                weather=weather,
                is_airborne=True,  # long balls: k_drag = 0.09 regime
            )
        bent_target = Vec2(to_x, to_y)

        self.set_ball(start.x, start.y)
        self.note("long_pass", f"{round(ball_speed,1)}m/s ballistic")

        defenders_list = [d for d in defenders if d is not None]
        self._record_pressure(pressure_level, defenders_list)
        race_ctx = self._build_race_context([receiver] + defenders_list)

        flight = make_ballistic_flight(
            Vec3(start.x, start.y, 0.05),
            Vec3(bent_target.x, bent_target.y, landing_z),
            ball_speed,
            gravity=gravity,
            loft=loft,
            spin=spin,
        )
        resolution = resolve_aerial_pass(
            flight, receiver, defenders_list, sample_step=self.dt,
            player_context=race_ctx,
        )

        # Fold ballistic analytics + weather drift into the resolution so the
        # chain / exporters can stamp the launch without re-deriving it.
        resolution = AerialPassResolution(
            outcome=resolution.outcome,
            ball_travel_time=resolution.ball_travel_time,
            contact_point=resolution.contact_point,
            receiver_arrival_time=resolution.receiver_arrival_time,
            winner=resolution.winner,
            winner_is_receiver=resolution.winner_is_receiver,
            challenger=resolution.challenger,
            winner_jump_time=resolution.winner_jump_time,
            winner_reach_height=resolution.winner_reach_height,
            challenger_reach_height=resolution.challenger_reach_height,
            ball_trajectory=resolution.ball_trajectory,
            launch_speed=flight.launch_speed,
            launch_angle_deg=math.degrees(flight.launch_angle_rad),
            apex_height=flight.apex_z,
            wind_drift_m=drift_m,
            spin_rate=flight.spin.rate if flight.spin is not None else 0.0,
            spin_kind=flight.spin.kind if flight.spin is not None else "",
        )

        # Trace the ballistic 3D flight on the shared 0.1 s clock.
        steps = max(1, int(math.ceil(flight.duration / self.dt)))
        for i in range(1, steps + 1):
            self.elapsed += self.dt
            t = min(1.0, i / steps)
            p = flight.position_at(t)
            self.ball_x, self.ball_y, self.ball_z = p.x, p.y, p.z
            self._snap("ball", p.x, p.y, flight.speed_at(t * flight.duration),
                       note="long_pass_flight", z=p.z)

            rr = min(1.0, self.elapsed / max(0.05, resolution.receiver_arrival_time)) \
                if resolution.receiver_arrival_time > 0 else 1.0
            rx = receiver.position.x + (bent_target.x - receiver.position.x) * rr
            ry = receiver.position.y + (bent_target.y - receiver.position.y) * rr
            recv_name = getattr(receiver.player, "name", "receiver")
            self._track_motion(recv_name, rx, ry, self.elapsed)
            self._snap(recv_name, rx, ry,
                       receiver.top_speed * self.RACE_EFFORT, note="aerial_race")

            ex = {getattr(receiver.player, "name", "receiver")}
            ex.update(getattr(d.player, "name", "") for d in defenders_list)
            self.step_off_ball(p.x, p.y, exclude=ex)
            self.ball_path.append({
                "t": round(self.elapsed, 3), "x": round(p.x, 2), "y": round(p.y, 2),
                "kind": "pass", "p": round(pressure_level, 3),
            })

        return resolution

    def update_player_position(self, player_name: str, x: float, y: float) -> None:
        """Sync the episode's tracked PhysPlayer to the post-contact position."""
        pp = self.players.get(player_name)
        if pp is not None:
            pp.x = float(x)
            pp.y = float(y)

    def calculate_distance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate true per-player distance, sprints, and top speed from the trace.

        Returns:
            Dict mapping player name -> {
                "distance_m": total meters covered,
                "sprint_distance_m": meters covered while speed > sprint_threshold,
                "high_speed_sprint_distance_m": meters while speed > high_threshold,
                "sprint_count": number of sprint segments,
                "high_speed_sprint_count": number of high-speed sprint segments,
                "top_speed_mps": max observed speed,
            }
        """
        SPRINT_THRESHOLD = 7.0      # m/s
        HIGH_SPEED_THRESHOLD = 8.5  # m/s
        HSR_THRESHOLD = 5.5         # m/s (19.8 km/h, standard GPS band)
        MIN_SEGMENT_DIST = 3.5      # min sustained distance to register as a
                            # sprint/HSR segment (real GPS requires a
                            # genuinely sustained burst, not every dip)

        # Group trace rows by player
        player_traces: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for row in self.trace:
            if row.player in ("ball", "gk_track", "drift"):
                continue
            player_traces.setdefault(row.player, []).append(
                (row.x, row.y, row.speed_mps, row.tick)
            )

        stats: Dict[str, Dict[str, float]] = {}
        for player, trace in player_traces.items():
            if len(trace) < 2:
                stats[player] = {
                    "distance_m": 0.0,
                    "sprint_distance_m": 0.0,
                    "high_speed_sprint_distance_m": 0.0,
                    "sprint_count": 0.0,
                    "high_speed_sprint_count": 0.0,
                    "top_speed_mps": 0.0,
                }
                continue

            total_dist = 0.0
            sprint_dist = 0.0
            high_sprint_dist = 0.0
            hsr_dist = 0.0
            sprint_count = 0
            high_sprint_count = 0
            top_speed = 0.0
            in_sprint = False
            in_high_sprint = False
            sprint_seg_dist = 0.0
            high_sprint_seg_dist = 0.0

            for i in range(1, len(trace)):
                x0, y0, s0, t0 = trace[i - 1]
                x1, y1, s1, t1 = trace[i]
                dx = x1 - x0
                dy = y1 - y0
                seg_dist = math.hypot(dx, dy)
                total_dist += seg_dist
                # Speed-band classification uses the trace's speed label
                # (the intended race/carry effort of the action) rather than
                # geometry velocity: the coarse per-tick interpolation makes
                # receivers/defenders "jump" 1-2+ m per 0.1 s tick (an
                # implied 10-20 m/s), which a real GPS would never see.
                # The label already caps race effort at RACE_EFFORT * top_speed,
                # so sprint/HSR banding reflects the simulated effort.
                # Distance accumulated remains geometry-real (seg_dist).
                dt = t1 - t0
                # Only per-tick consecutive rows (≈ one 10 Hz step) carry a
                # measurable effort. Anything else is a position gap (sparse
                # trace rows, cross-action jumps) -> contributes to total
                # distance but not to sprint/HSR classification (GPS losing
                # sync).
                if 0.05 <= dt <= 0.15:
                    speed = s1
                else:
                    speed = 0.0
                top_speed = max(top_speed, speed)
                if speed >= HSR_THRESHOLD:
                    hsr_dist += seg_dist

                if speed >= SPRINT_THRESHOLD:
                    sprint_dist += seg_dist
                    sprint_seg_dist += seg_dist
                    in_sprint = True
                    if speed >= HIGH_SPEED_THRESHOLD:
                        high_sprint_dist += seg_dist
                        high_sprint_seg_dist += seg_dist
                        in_high_sprint = True
                    else:
                        in_high_sprint = False
                else:
                    if in_sprint and sprint_seg_dist >= MIN_SEGMENT_DIST:
                        sprint_count += 1
                    if in_high_sprint and high_sprint_seg_dist >= MIN_SEGMENT_DIST:
                        high_sprint_count += 1
                    in_sprint = False
                    in_high_sprint = False
                    sprint_seg_dist = 0.0
                    high_sprint_seg_dist = 0.0

            # Close any open segment at end of trace
            if in_sprint and sprint_seg_dist >= MIN_SEGMENT_DIST:
                sprint_count += 1
            if in_high_sprint and high_sprint_seg_dist >= MIN_SEGMENT_DIST:
                high_sprint_count += 1

            stats[player] = {
                "distance_m": round(total_dist, 2),
                "sprint_distance_m": round(sprint_dist, 2),
                "high_speed_sprint_distance_m": round(high_sprint_dist, 2),
                "hsr_distance_m": round(hsr_dist, 2),
                "sprint_count": float(sprint_count),
                "high_speed_sprint_count": float(high_sprint_count),
                "top_speed_mps": round(top_speed, 2),
            }

        return stats

    # ── trace export ───────────────────────────────────────────────
    def condensed_trace(self, limit: int = 60) -> List[dict]:
        """Compact per-tick trace for event metadata / analytics consumers."""
        rows = self.trace[-limit:] if self.dt >= 0 else self.trace
        return [
            {
                "t": r.tick, "p": r.player, "x": r.x, "y": r.y,
                "v": r.speed_mps, "bx": r.ball_x, "by": r.ball_y,
                "bz": r.ball_z, "n": r.note,
            }
            for r in rows
        ]

    def physics_meta(self, action: str = "ground") -> dict:
        return {
            "engine": "continuous_ticks",
            "tick_s": self.dt,
            "elapsed_s": round(self.elapsed, 2),
            "action": action,
            "ticks": len(self.trace),
            "pressure_samples": len(self.pressure_history),
        }
    
    def full_physics_report(self) -> Dict[str, Any]:
        """Comprehensive physics report for analytics consumers."""
        avg_pressure = 0.0
        if self.pressure_history:
            avg_pressure = sum(p.pressure_level for p in self.pressure_history) / len(self.pressure_history)
        
        return {
            "engine": "continuous_ticks_v2",
            "tick_s": self.dt,
            "elapsed_s": round(self.elapsed, 2),
            "ticks": len(self.trace),
            "ball_trajectory_points": len(self._ball_trajectory),
            "pressure": {
                "avg_level": round(avg_pressure, 3),
                "samples": len(self.pressure_history),
                "peak": round(max((p.pressure_level for p in self.pressure_history), default=0.0), 3),
            },
            "notes": [(round(t, 2), label, detail) for t, label, detail in self.notes],
        }
