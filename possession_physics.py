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

    Long balls, crosses, aerial duels and rebounds still run on the existing
    ballistic delivery model (geometry_engine.resolve_aerial_delivery) — they
    are the natural 3D follow-up stage. This module makes the ground game a
    continuous-motion simulation instead of a sequence of independent rolls.

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
    MovingPlayer, Vec2, Vec3, BallFlight, PassResolution, DribbleResolution,
    ShotResolution, AerialResolution, make_flight, resolve_aerial_delivery,
    resolve_ground_pass, resolve_dribble, resolve_shot, GoalkeeperState,
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


def aerial_delivery_speed(long_passing: float) -> float:
    """12 (lofted) .. 24 (rapid cross) m/s."""
    return max(12.0, 12.0 + max(0.0, min(100.0, long_passing)) * 0.12)


def _clamped(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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
    
    flight = make_flight(
        Vec3(x, y, 0.05),
        Vec3(goal_x_plane, target_y, max(0.0, target_z)),
        speed,
        apex_z=apex_z,
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

    def __init__(self, dt: float = TICK_S):
        self.dt = float(dt)
        self.elapsed = 0.0
        self.players: dict[str, PhysPlayer] = {}
        self.trace: List[MotionSnapshot] = []
        self.ball_x = 52.5
        self.ball_y = 34.0
        self.ball_z = 0.0
        self.notes: List[tuple[float, str, str]] = []
        self.pressure_history: List[PressureSnapshot] = []
        self._rng = random.Random()
        self._ball_trajectory: List[Tuple[float, Vec3]] = []

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

    def set_ball(self, x: float, y: float, z: float = 0.0) -> None:
        self.ball_x = float(x)
        self.ball_y = float(y)
        self.ball_z = float(z)

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
    ) -> PassResolution:
        """Alias for resolve_pass — event_chain calls this name."""
        return self.resolve_pass(start, target, receiver, defenders, ball_speed,
                                 ball_owner_team=ball_owner_team,
                                 pressure_level=pressure_level)

    def resolve_pass(
        self,
        start: Vec2,
        target: Vec2,
        receiver: MovingPlayer,
        defenders: Iterable[MovingPlayer],
        ball_speed: float,
        ball_owner_team: str = "",
        pressure_level: float = 0.0,
    ) -> PassResolution:
        """Continuous 0.1 s race-to-ball pass resolution (see module doc).
        
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

        resolution = resolve_ground_pass(
            start, target, receiver, defenders_list, ball_speed,
            sample_step=self.dt, pressure_level=pressure_level,
        )
        
        # Trace the whole flight on the shared clock with per-tick ball tracking
        distance = start.distance_to(target)
        speed = max(6.0, ball_speed)
        
        # Apply pressure-based speed degradation
        if pressure_level > 0.5:
            speed *= (1.0 - 0.1 * (pressure_level - 0.5))
        
        travel = distance / speed
        steps = max(1, int(math.ceil(travel / self.dt)))
        receiver_time = receiver.time_to_reach(target)
        
        # Track ball trajectory
        self._ball_trajectory = []
        
        for i in range(1, steps + 1):
            self.elapsed += self.dt
            t = min(1.0, i / steps)
            ball = start.lerp(target, t)
            self.ball_x, self.ball_y = ball.x, ball.y
            self._ball_trajectory.append((self.elapsed, Vec3(ball.x, ball.y, 0.0)))
            
            # Update receiver position
            rx = receiver.position.x
            ry = receiver.position.y
            if receiver_time > 0:
                rr = min(1.0, self.elapsed / max(0.05, receiver_time))
                rx = receiver.position.x + (target.x - receiver.position.x) * rr
                ry = receiver.position.y + (target.y - receiver.position.y) * rr
            self._snap(getattr(receiver.player, "name", "receiver"), rx, ry,
                       receiver.top_speed * min(1.0, rr if receiver_time > 0 else 1),
                       note="pass_race")
            
            # Update defender positions
            for d in defenders_list:
                if d is None:
                    continue
                day = d.time_to_reach(ball, d.control_radius)
                dr = min(1.0, self.elapsed / max(0.05, day)) if day > 0 else 1.0
                dxp = d.position.x + (ball.x - d.position.x) * dr
                dyp = d.position.y + (ball.y - d.position.y) * dr
                self._snap(getattr(d.player, "name", "defender"), dxp, dyp,
                           d.top_speed * min(1.0, dr), note="intercept_race")
        
        self._snap("ball", target.x, target.y, speed, note="pass_end")
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
        
        Enhanced with:
        * Continuous defender motion tracking
        * Pressure-driven tackle window calculation
        """
        self.set_ball(start.x, start.y)
        self.note("dribble", "")
        
        defenders_list = list(defenders)
        self._record_pressure(pressure_level, defenders_list)

        resolution = resolve_dribble(
            start, target, attacker, defenders_list, sample_step=self.dt,
            pressure_level=pressure_level,
        )
        
        # Trace the carry on the shared clock with defender tracking
        distance = start.distance_to(target)
        dribble_speed = attacker.top_speed * 0.68
        duration = distance / max(2.5, dribble_speed)
        steps = max(1, int(math.ceil(duration / self.dt)))
        
        for i in range(1, steps + 1):
            self.elapsed += self.dt
            t = min(1.0, i / steps)
            ball = start.lerp(target, t)
            self.ball_x, self.ball_y = ball.x, ball.y
            
            self._snap(getattr(attacker.player, "name", "attacker"),
                       ball.x, ball.y, dribble_speed, note="carry")
            
            # Track all defenders during dribble
            for d in defenders_list:
                if d is None:
                    continue
                tackle_range = attacker.control_radius + d.tackle_radius
                arrival = d.time_to_reach(ball, tackle_range)
                if arrival <= self.elapsed:
                    self._snap(getattr(d.player, "name", "defender"),
                               ball.x, ball.y, d.top_speed, note="contact")
                else:
                    # Still moving toward contact point
                    dr = min(1.0, self.elapsed / max(0.05, arrival)) if arrival > 0 else 1.0
                    dxp = d.position.x + (ball.x - d.position.x) * dr
                    dyp = d.position.y + (ball.y - d.position.y) * dr
                    self._snap(getattr(d.player, "name", "defender"),
                               dxp, dyp, d.top_speed * dr, note="pursuit")
        
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
        tackle_range = attacker.control_radius + challenger.tackle_radius
        challenger_arrival = challenger.time_to_reach(Vec2(ball_x, ball_y), tackle_range)
        
        # How long does the carrier need to escape the tackle envelope?
        escape_wheel = max(1.5, attacker.top_speed * 0.55)
        escape_time = (tackle_range + attacker.reaction_time) / escape_wheel
        
        # Contact window is tighter under pressure
        window = 0.10 + attacker.reaction_time
        if pressure_level > 0.3:
            window *= (1.0 - 0.2 * pressure_level)
        
        if challenger_arrival <= escape_time + window:
            # Defensive skill can close the final gap; offensive agility opens
            # it — still geometric (radii/reaction) rather than a flat roll.
            skill_balance = (challenger.tackle_radius - attacker.control_radius)
            margin = (escape_time + window) - challenger_arrival
            if margin < skill_balance * 0.15:
                return _rng.random() < 0.35
            return False
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
            
            # Track goalkeeper movement during shot
            if goalkeeper is not None:
                name = getattr(goalkeeper.player, "name", "GK")
                # GK tracks ball position
                gk_x = goalkeeper.position.x
                gk_y = goalkeeper.position.y
                self._snap(name, gk_x, gk_y, goalkeeper.top_speed, note="gk_track")
        
        return resolution

    # ── aerial delivery (cross / long ball) ────────────────────────
    def resolve_aerial(
        self,
        flight: BallFlight,
        attackers: Iterable[MovingPlayer],
        defenders: Iterable[MovingPlayer],
    ) -> AerialResolution:
        """Resolve aerial delivery with 3D tracking."""
        self.set_ball(flight.start.x, flight.start.y, flight.start.z)
        self.note("aerial", "")
        
        resolution = resolve_aerial_delivery(flight, attackers, defenders, sample_step=self.dt)
        
        # Trace 3D trajectory
        steps = max(1, int(math.ceil(flight.duration / self.dt)))
        for i in range(1, steps + 1):
            self.elapsed += self.dt
            t = min(1.0, i / steps)
            p = flight.position_at(t)
            self.ball_x, self.ball_y, self.ball_z = p.x, p.y, p.z
            self._snap("ball", p.x, p.y, 0.0, note="aerial_flight", z=p.z)
        
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
        MIN_SEGMENT_DIST = 2.0      # minimum distance to count as a segment

        # Group trace rows by player
        player_traces: Dict[str, List[Tuple[float, float, float]]] = {}
        for row in self.trace:
            if row.player in ("ball", "gk_track", "drift"):
                continue
            player_traces.setdefault(row.player, []).append(
                (row.x, row.y, row.speed_mps)
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
            sprint_count = 0
            high_sprint_count = 0
            top_speed = 0.0
            in_sprint = False
            in_high_sprint = False
            sprint_seg_dist = 0.0
            high_sprint_seg_dist = 0.0

            for i in range(1, len(trace)):
                x0, y0, s0 = trace[i - 1]
                x1, y1, s1 = trace[i]
                dx = x1 - x0
                dy = y1 - y0
                seg_dist = math.hypot(dx, dy)
                total_dist += seg_dist
                speed = s1
                top_speed = max(top_speed, speed)

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
