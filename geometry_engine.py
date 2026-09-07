"""Deterministic, two-dimensional action resolution for PLOFA.

This module deliberately separates *action selection* from *action outcome*.
The match AI can decide to pass or dribble, but the result is determined by
ball travel and player reach rather than a completion-probability roll.

Calibration (real football ranges):
    * Top speed       pace 0..100  -> 5.0..9.2 m/s (18-33 km/h)
    * Acceleration    3.0..6.0 m/s^2 (0-5m start ~1.3-1.7s)
    * Reaction        0.42s (slow) .. 0.15s (elite reflexes)
    * Control radius  0.7..1.6 m
    * Tackle radius   0.9..1.9 m
    * Jump height     0.35..0.80 m; total vertical reach capped 2.65 (OF) / 2.9 (GK)
    * GK dive         lateral extension 1.0..2.2 m beyond reach at 2.2..3.6 m/s
    * Ground pass     10..23 m/s; shots 20..34 m/s; aerial delivery 12..24 m/s

All contact sampling steps on a 0.1 s tick by default (10 Hz possession clock).

Checkpoint 28 — Full Possession Physics Redesign:
    * Continuous goalkeeper positioning during shot flights
    * 3D trajectory resolution with jump timing and vertical reach
    * Velocity-preserving rebounds after saves, woodwork, and blocks
    * Per-tick ball position tracking during all passes
    * Pressure-driven accuracy modifiers based on defender proximity

Long-ball registry (checkpoint 6):
    * ``make_ballistic_flight`` launches long passes as TRUE projectiles: the
      launch angle solves the range/height geometry for the passer's leg
      speed, and flight time + apex height are DERIVED from that launch
      velocity vector and gravity (9.81) — not picked as independent numbers.
    * ``resolve_aerial_pass`` remains the landing-point duel: whichever player
      physically reaches the descent point within their reach (standing +
      jump from physical.jumping) and arrival time (acceleration curve +
      reaction) wins. Ballistics plus the duel = no flat accuracy roll.

Ball spin / Magnus effect (checkpoint 7):
    * ``BallSpin`` decorates a flight with rotation. Spin bends the trajectory
      while keeping the target fixed (the striker's aim already accounts for
      the curve): a ``side`` spinner bulges away from the straight chord at
      mid-flight and lands on target — the mechanical signature that makes a
      Rashford curler structurally distinct from a straight strike at the same
      speed and angle; ``back`` spin floats the arc (chips land short, carry),
      ``top`` spin drives it down (dips, arrives faster/flatter).
    * Deviation scales with spin rate x ball speed (Magnus force ~ ω·v) and
      with the SQUARE of flight time, so the same curl bends far more over a
      long delivery than over a short one — 0 at both endpoints, because the
      aim already compensates, and the mid-flight banana is the distinguishable
      geometry (a swept blocker envelope sees it arc past the wall).

Ball ground friction / rolling deceleration (checkpoint 7):
    * ``resolve_ground_pass`` no longer rolls a flat ``distance / speed``.
      The ball loses speed to turf: rolling deceleration ``BALL_ROLLING_DECEL``
      (m/s^2, weather-scaled), so a 40 m driven pass arrives meaningfully
      slower than it left the boot and an under-struck pass can genuinely die
      before the target (``underhit``). Arrival time solves the constant-
      deceleration quadratic — ``rolling_arrival_time`` / ``rolling_position``.

Player movement costs — acceleration / braking / turning (checkpoint 8):
    * ``MovingPlayer.time_to_reach`` accepts the player's existing ``heading``
      and ``entry_speed``. Physics: a player ALREADY running the right way
      covers ground faster than from a standstill (momentum carries), but a
      sharp change of direction bleeds top speed (``turn_speed_factor``: ~70%
      through a 90° cut, ~50% for anything near a full reverse) and surplus
      entry speed must be braked off before the re-acceleration ramp. Every
      race recomputed-from-rest with a free instant pivot is gone: the chain
      feeds live velocity memory so a defender who has to turn to intercept
      genuinely loses time.
    * Back-compatible: callers that pass no heading/entry speed get the exact
      old accel-from-rest trapezoid, so every existing race is unchanged.

Body-collision / momentum duels (checkpoint 8):
    * ``resolve_body_duel`` settles who wins the shoulder-to-shoulder: impulse
      = effective mass (real body mass boosted by balance/leverage) x approach
      speed, compared across the two bodies. A 95 kg defender arriving at
      sprint displaces a featherweight carrier jogging with the ball; a strong,
      agile carrier rides the challenge and keeps possession. ``resolve_dribble``
      and ``resolve_loose_ball`` use it so tackles and 50/50s are won by mass x
      momentum tradeoffs instead of being decided only by reach radii.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def distance_to(self, other: "Vec2") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def lerp(self, other: "Vec2", t: float) -> "Vec2":
        return Vec2(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)
    
    def normalized(self) -> "Vec2":
        length = math.hypot(self.x, self.y)
        if length < 1e-6:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / length, self.y / length)
    
    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y
    
    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def horizontal(self) -> Vec2:
        return Vec2(self.x, self.y)
    
    def distance_to(self, other: "Vec3") -> float:
        return math.hypot(other.x - self.x, other.y - self.y, other.z - self.z)
    
    def lerp(self, other: "Vec3", t: float) -> "Vec3":
        return Vec3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )


TICK_S = 0.1  # default possession-clock step for contact sampling

# ─────────────────────────────────────────────────────────────────────────────
# BALL SPIN / MAGNUS EFFECT
# ─────────────────────────────────────────────────────────────────────────────

# Empirical Magnus coupling: lateral/vertical acceleration m/s² = K · ω · v.
# Calibrated so a strong free-kick curler (ω ≈ 60-80 rad/s, v ≈ 25 m/s,
# a ≈ 4-6 m/s²) bulges ~1.0-1.5 m off the straight chord at mid-flight, and a
# whipped cross (ω ≈ 55 rad/s, v ≈ 18 m/s, a ≈ 2.8 m/s²) bends ~0.8 m.
MAGNUS_K = 0.0028

# Maximum vertical Magnus acceleration allowed when adjusting ballistic apex.
# Keeps the effective gravity positive even for a violently-hooped delivery.
MAGNUS_VERTICAL_SPIN_CAP = 0.72  # fraction of 9.81

# Curl handedness for a Kodak moment vs the keeper — arbitrary per-aider,
# only the shape matters physically; the sign makes one side of the pitch
# bend one way and mirrors to the other.
SIDE_SPIN = "side"
TOP_SPIN = "top"
BACK_SPIN = "back"


@dataclass(frozen=True)
class BallSpin:
    """Rotation of the ball, which bends its flight via the Magnus effect.

    ``kind`` selects the plane of the Magnus force:
      * ``side``  — axis vertical; the ball curls perpendicular to travel
                    (the banana free-kick, the whipped banana cross).
      * ``top``   — forward rotation; Magnus drives the ball DOWN (dips,
                    flattens, arrives quicker).
      * ``back``  — backward rotation; Magnus holds the ball UP (floats,
                    carries longer, chips drop softer).

    ``rate`` is the angular speed in rad/s (a dead ball is 0 — and at high
    speed that is exactly the knuckleball regime where the engine stays
    honest: no spin, no bend). ``sign`` chooses curl handedness for ``side``.
    """

    rate: float = 0.0
    kind: str = SIDE_SPIN
    sign: float = 1.0

    @classmethod
    def side(cls, rate: float, sign: float = 1.0) -> "BallSpin":
        return cls(max(0.0, rate), SIDE_SPIN, 1.0 if sign >= 0 else -1.0)

    @classmethod
    def topspin(cls, rate: float) -> "BallSpin":
        return cls(max(0.0, rate), TOP_SPIN, 0.0)

    @classmethod
    def backspin(cls, rate: float) -> "BallSpin":
        return cls(max(0.0, rate), BACK_SPIN, 0.0)

    @property
    def active(self) -> bool:
        return self.rate > 0.0


def magnus_accel(spin: Optional[BallSpin], speed: float) -> float:
    """Magnus lateral/vertical acceleration (m/s²) for a spun flight."""
    if spin is None or not spin.active:
        return 0.0
    return MAGNUS_K * spin.rate * max(6.0, speed)


# ─────────────────────────────────────────────────────────────────────────────
# BALL GROUND FRICTION / ROLLING DECELERATION
# ─────────────────────────────────────────────────────────────────────────────

# Baseline rolling deceleration of a ball on dry turf (m/s²). A driven 22 m/s
# ball loses ~15-20% of its pace over 30 m and an under-struck 10 m/s loft
# crawls to a crawl and dies to short-of-target. Weather (pitch wetness) scales
# this up — see WeatherPhysics.rolling_decel_mult.
BALL_ROLLING_DECEL = 1.2


# ─────────────────────────────────────────────
# PLAYER MOVEMENT COSTS (#3)
# ─────────────────────────────────────────────
# A heading change shallower than this (≈20°) is "running the same line" —
# no turning penalty.
TURN_ANGLE_THRESHOLD_RAD = 0.35
# Braking is stronger than accelerating; surplus entry speed sheds at ~1.8x
# the player's own acceleration (capped at a realistic ±6 m/s^2).
DECEL_FACTOR = 1.8
# Any velocity slower than this is a shuffle, not a sprint into the race.
MOVEMENT_DEAD_BAND_MPS = 0.35


def turn_speed_factor(theta: float) -> float:
    """Share of top speed a player can sustain while turning through ``theta``.

    Aligned (θ≈0) → 1.0 (free); a 90° cut sustains ~0.70 of top speed; a full
    reverse forces the player to brake and repivot (≈0.40 — effective top of
    the re-acceleration ramp). Smooth monotonic curve, no cliffs.
    """
    theta = max(0.0, min(math.pi, theta))
    if theta <= TURN_ANGLE_THRESHOLD_RAD:
        return 1.0
    return 0.40 + 0.60 * math.cos(theta * 0.5) ** 2


# ─────────────────────────────────────────────
# BODY / MOMENTUM DUELS (#4)
# ─────────────────────────────────────────────
# Strength 0..100 -> body mass 55..95 kg (featherweight vs watertight tank).
BODY_MASS_MIN_KG = 55.0
BODY_MASS_MAX_KG = 95.0
# Impulse ratio ≥ this -> decisive shoulder win (bowl-over / stood up).
BODY_DUEL_DECISIVE_EDGE = 1.30
# Impulse ratio ≤ this -> carrier rides the challenge and keeps the ball.
BODY_DUEL_RODE_EDGE = 0.78
# Arrival within this many seconds of the carrier's escape moment is a genuine
# shoulder-to-shoulder (the defender hasn't cleanly beaten the ball; both
# bodies are at it at the same instant). Wider margins are clean early beats.
BODY_DUEL_TIE_WINDOW = 0.18
# Challenge momentum model: a defender who GETS to the ball to contest brings
# his committed challenge sprint (62% of top — a lunge-into-the-shoulder, not
# a stroll), while the carrier shields at his carry speed (68% of top). Equal
# body + the same approach fractions leave a true 50/50-adjacent contest;
# physique asymmetry decides the shoulder.
DEFENDER_CHALLENGE_FACTOR = 0.62
CARRIER_SHIELD_FACTOR = 0.68


def body_mass(strength: float) -> float:
    """Strength 0..100 -> body mass 55..95 kg."""
    return BODY_MASS_MIN_KG + max(0.0, min(100.0, strength)) * (
        (BODY_MASS_MAX_KG - BODY_MASS_MIN_KG) / 100.0
    )


def body_balance(strength: float, agility: float, ball_control: float) -> float:
    """0.3 (leggy, lead-footed) .. 0.95 (strong + agile + tidy on the ball).

    Balance is the leverage with which a player converts his mass at the point
    of contact — the difference between loafing through a shoulder and winning
    it with a planted frame.
    """
    raw = strength * 0.5 + agility * 0.3 + ball_control * 0.2
    return 0.30 + 0.65 * max(0.0, min(100.0, raw)) / 100.0


# Race context: per-player (heading unit Vec2, entry speed m/s) for movement-
# cost-aware races. Keyed by ``id(player)`` so resolvers need no naming.
PlayerRaceContext = Dict[int, Tuple[Optional[Vec2], Optional[float]]]


def _race_motion(
    player: "MovingPlayer",
    target: Vec2,
    radius: Optional[float] = None,
    player_context: Optional[PlayerRaceContext] = None,
) -> float:
    """Arrival time with (optional) movement context: heading + entry speed.

    Without context this is exactly the old accel-from-rest trapezoid; with
    context the player's existing heading/velocity feed the turn cost and
    momentum-carry model (see ``MovingPlayer.time_to_reach``).
    """
    if player_context is not None:
        entry = player_context.get(id(player))
        if entry is not None:
            heading, entry_speed = entry
            if heading is not None or entry_speed is not None:
                return player.time_to_reach(
                    target, radius, heading=heading, entry_speed=entry_speed,
                )
    return player.time_to_reach(target, radius)


def rolling_position(v0: float, decel: float, t: float) -> float:
    """Distance covered after ``t`` seconds against constant rolling friction.

    ``v0`` in m/s, ``decel`` in m/s². The ball stops (never reverses) once its
    speed reaches zero: s(t) = v0·t − ½·k·t² up to the stop instant, then sits
    at the rest distance (a trace sampling past the stop reports the rest
    point, not a backward slide).
    """
    v0 = max(0.0, v0)
    k = max(0.0, decel)
    if k <= 0.0:
        return v0 * max(0.0, t)
    t = min(max(0.0, t), v0 / k)
    return v0 * t - 0.5 * k * t * t


def rolling_speed(v0: float, decel: float, t: float) -> float:
    """Ball speed after ``t`` seconds of rolling, floor of 0 (rest)."""
    return max(0.0, v0 - max(0.0, decel) * max(0.0, t))


def rolling_arrival_time(distance: float, v0: float, decel: float) -> Optional[float]:
    """Time the ball first reaches ``distance`` under rolling friction.

    Returns None when the ball stops short of ``distance`` (an under-struck
    pass that genuinely dies on the pitch). When ``decel`` ≤ 0 the ball rolls
    at constant speed (distance / v0, the old flat model).
    """
    if distance <= 0.0:
        return 0.0
    v0 = max(0.0, v0)
    k = max(0.0, decel)
    if v0 <= 0.0:
        return None
    if k <= 0.0:
        return distance / v0
    rest_dist = v0 * v0 / (2.0 * k)
    if rest_dist < distance:
        return None
    # Solve distance = v0·t − ½·k·t² (the smaller root; the larger one has the
    # ball turning around — physically impossible for rolling friction).
    disc = v0 * v0 - 2.0 * k * distance
    if disc < 0.0:
        return None
    return (v0 - math.sqrt(disc)) / k


@dataclass(frozen=True)
class MovingPlayer:
    """A player's current physical state for one short action window."""

    player: object
    position: Vec2
    pace: float
    acceleration: float
    reaction_time: float
    control_radius: float = 1.05
    tackle_radius: float = 1.25
    is_goalkeeper: bool = False
    jump_height: float = 0.55
    standing_reach: float = 1.75
    # Goalkeeper dive envelope (calibrated): lateral extension beyond reach
    # and the horizontal dive speed the keeper can generate.
    dive_reach: float = 1.4
    dive_speed: float = 2.8
    # Vertical reach when diving (GK can reach higher when airborne)
    dive_vertical_reach: float = 2.9
    # Body identity (#4). mass in kg (55..95 via body_mass(strength));
    # balance 0..1 is the leverage converting that mass at contact.
    body_mass_kg: float = 76.0
    balance: float = 0.5

    @property
    def top_speed(self) -> float:
        """Pace 0..100 mapped to a real 5.0..9.2 m/s sprint range."""
        return 5.0 + max(0.0, min(100.0, self.pace)) * 0.042

    def time_to_reach(
        self,
        target: Vec2,
        radius: Optional[float] = None,
        heading: Optional[Vec2] = None,
        entry_speed: Optional[float] = None,
    ) -> float:
        """Earliest time at which the player can contact ``target`` in seconds.

        Movement costs (#3): without ``heading`` / ``entry_speed`` this is the
        classic accel-from-rest trapezoid (reaction + acceleration ramp + top-
        speed cruise), so every legacy call is unchanged. With a ``heading``
        and ``entry_speed`` (the player's live velocity going INTO this race):

        * momentum carry — an already-moving player starts the ramp at
          ``entry_speed`` instead of 0, so a defender tracking the ball the
          right way arrives earlier than from a standstill;
        * turning cost — a sharp change of direction bleeds the top speed the
          chase can sustain (``turn_speed_factor``); the wrong-way defender
          who must brake, pivot and re-accelerate pays extra time;
        * braking — entry speed above the (turn-limited) sustainable top must
          be shed at ~1.8x acceleration before the new sprint can build.
        """
        remaining = max(0.0, self.position.distance_to(target) - (radius or self.control_radius))
        if remaining == 0.0:
            return 0.0
        acceleration = max(1.5, self.acceleration)
        top_speed = self.top_speed

        # Turn cost: the player was moving somewhere; heavy heading changes
        # bleed the top speed this chase can carry into the new direction.
        if heading is not None:
            dx = target.x - self.position.x
            dy = target.y - self.position.y
            n = math.hypot(dx, dy)
            if n > 1e-9:
                cos_th = max(-1.0, min(1.0, (dx * heading.x + dy * heading.y) / n))
                top_speed *= turn_speed_factor(math.acos(cos_th))

        v_cur = 0.0
        if entry_speed is not None:
            v_cur = max(0.0, min(float(entry_speed), top_speed))

        # Braking: entering the race FASTER than the turning speed sustains
        # (e.g. full-sprint defender cutting 90°) means first shedding that
        # surplus, then cruising the new line at top_speed.
        if entry_speed is not None and entry_speed > top_speed:
            decel = min(6.0, acceleration * DECEL_FACTOR)
            brake_time = (float(entry_speed) - top_speed) / decel
            return self.reaction_time + brake_time + remaining / max(1.0, top_speed)

        if v_cur >= top_speed:
            return self.reaction_time + remaining / max(1.0, top_speed)

        time_to_top = (top_speed - v_cur) / acceleration
        distance_to_top = v_cur * time_to_top + 0.5 * acceleration * time_to_top * time_to_top
        if remaining <= distance_to_top:
            t_ramp = (
                math.sqrt(v_cur * v_cur + 2.0 * acceleration * remaining) - v_cur
            ) / acceleration
            return self.reaction_time + t_ramp
        return self.reaction_time + time_to_top + (remaining - distance_to_top) / top_speed

    def vertical_reach(self, airborne: bool) -> float:
        """Maximum height a player can reach when grounded or jumping."""
        if airborne:
            # Goalkeepers have a higher max vertical reach when diving
            if self.is_goalkeeper:
                return min(self.dive_vertical_reach, self.standing_reach + self.jump_height)
            return min(2.65, self.standing_reach + self.jump_height)
        return self.standing_reach

    def keeper_dive_reach(self, direction: Vec2) -> float:
        """Effective reach radius when diving toward a point in the goal mouth.

        The keeper extends ``dive_reach`` beyond the control radius in the
        direction of the shot; the lateral reach shrinks slightly when the
        ball is overhead (full extension overhead is harder than low).
        """
        return self.control_radius + self.dive_reach


@dataclass(frozen=True)
class GoalkeeperState:
    """Continuous goalkeeper positioning state during a shot sequence."""
    
    player: MovingPlayer
    # Current position (updated per tick)
    current_x: float
    current_y: float
    # Velocity (for momentum during dives)
    vx: float = 0.0
    vy: float = 0.0
    # Dive state
    is_diving: bool = False
    dive_direction: Optional[Vec2] = None
    dive_start_time: float = 0.0
    # Position on the goal line at start of shot sequence
    line_y: float = 34.0
    depth_x: float = 104.0
    
    @property
    def position(self) -> Vec2:
        return Vec2(self.current_x, self.current_y)
    
    def time_to_cover_lateral(self, target_y: float, time_available: float) -> bool:
        """Can the keeper reach a lateral position in the given time?"""
        lateral_dist = abs(target_y - self.current_y)
        if lateral_dist <= self.player.control_radius:
            return True  # Already in body coverage
        
        # Dive reach + body position
        effective_reach = self.player.control_radius + self.player.dive_reach
        
        # Time to react and dive
        reaction = self.player.reaction_time
        dive_time = (lateral_dist - self.player.control_radius) / max(1.0, self.player.dive_speed)
        
        return (reaction + dive_time) <= time_available and lateral_dist <= effective_reach


@dataclass(frozen=True)
class PassResolution:
    outcome: str  # received | intercepted | underhit
    ball_travel_time: float
    contact_point: Vec2
    receiver_arrival_time: float
    interceptor: Optional[MovingPlayer] = None
    interceptor_arrival_time: Optional[float] = None
    # Per-tick ball positions for analytics
    ball_trajectory: Optional[List[Tuple[float, Vec2]]] = None
    # Ball speed the moment it passes the contact point (rolling deceleration
    # means a 40 m ground pass arrives meaningfully slower than it left).
    ball_speed_at_arrival: float = 0.0
    # True when the ball stopped rolling before it physically could reach the
    # target (an under-struck pass that dies on the pitch).
    ball_stopped_short: bool = False
    # Magnus deco carried for analytics/exporters.
    spin_rate: float = 0.0
    spin_kind: str = ""


@dataclass(frozen=True)
class DribbleResolution:
    outcome: str  # retained | tackled
    contact_point: Vec2
    duration: float
    tackler: Optional[MovingPlayer] = None
    # Defender positions at each tick during the dribble
    defender_positions: Optional[List[Tuple[float, Vec2]]] = None
    # Body-collision deco (#4): how the take-on was actually decided.
    body_duel: Optional["BodyDuel"] = None
    resolution_note: str = ""  # bowl_over | shrugged_off | 50_50 | beat_them_geometrically


@dataclass(frozen=True)
class BodyDuel:
    """Outcome of a shoulder-to-shoulder contest, decided by mass x momentum."""

    winner: str  # "defender" | "carrier" | "contest"
    impulse_ratio: float  # defender impulse / carrier impulse
    defender_momentum: float  # kg·m/s
    carrier_momentum: float  # kg·m/s

    @property
    def decisive(self) -> bool:
        return self.winner != "contest"


def resolve_body_duel(
    defender: MovingPlayer,
    carrier: MovingPlayer,
    defender_approach_mps: float,
    carrier_approach_mps: float,
) -> BodyDuel:
    """Settle who wins the shoulder at ball contact — pure mass x momentum.

    Effective impulse = (real body mass boosted by balance/leverage) x approach
    speed. A defender arriving at sprint with 95 kg shoves a 60 kg carrier
    jogging with the ball clean off it (``defender``); a strong, planted carrier
    with momentum of his own rides the challenge (``carrier``); comparable
    impulses leave a genuine 50/50 (``contest``, caller resolves in a weighted
    gray zone). Real football: momentum trades, not a blind roll.
    """
    # Contact windows are short; the approach speed must still be physically
    # reachable from the players' acceleration curves. This keeps the challenge
    # model honest when a defender's "raw" approach is higher than a realistic
    # short-burst sprint allows.
    challenge_window_s = 0.5
    defender_approach_mps = min(defender_approach_mps, _approach_speed_at(defender, challenge_window_s))
    carrier_approach_mps = min(carrier_approach_mps, _approach_speed_at(carrier, challenge_window_s))

    m_d = defender.body_mass_kg * (1.0 + defender.balance)
    m_c = carrier.body_mass_kg * (1.0 + carrier.balance)
    mom_d = m_d * max(1.0, defender_approach_mps)
    mom_c = m_c * max(1.0, carrier_approach_mps)
    ratio = (mom_d / mom_c) if mom_c > 1e-9 else float("inf")
    if ratio >= BODY_DUEL_DECISIVE_EDGE:
        winner = "defender"
    elif ratio <= BODY_DUEL_RODE_EDGE:
        winner = "carrier"
    else:
        winner = "contest"
    return BodyDuel(winner, ratio, mom_d, mom_c)


def _body_contest_win_probability(impulse_ratio: float) -> float:
    """Gray-zone (contest) roll: p(defender wins the 50/50), weighted by impulse.

    Equal impulse -> a true 50/50; each step of momentum edge tilts the ball.
    Bounded so the contest zone never becomes a cliff.
    """
    return max(0.15, min(0.80, 0.5 + (impulse_ratio - 1.0) * 1.5))


def _approach_speed_at(player: MovingPlayer, elapsed: float) -> float:
    """Sprint speed physically reachable by ``player`` after ``elapsed`` s (accel-limited)."""
    if elapsed <= 0.0:
        return 0.0
    a = max(1.5, player.acceleration)
    return min(player.top_speed, a * elapsed)


@dataclass(frozen=True)
class BallFlight:
    """A ballistic 3D ball flight between two pitch coordinates.

    Two flight regimes coexist here:

    * PARAMETRIC (default) — the quadratic Bezier arch ``z_arc``. Apex and
      duration are chosen directly by the caller (shots, crosses, corners,
      ground passes). Controllable and deterministic, but the shape is tuned,
      not derived from launch physics.

* BALLISTIC (``ballistic=True``, built by ``make_ballistic_flight``) —
      a genuine projectile under gravity. Flight time, peak height and the
      whole arc fall out of an actual launch velocity vector (horizontal +
      vertical components at ``launch_angle_rad``) and ``gravity`` ≈ 9.81.
      This is the long-ball regime: a harder-struck ball and a soft lofted
      one produce naturally different, physically consistent trajectories
      instead of independently-tuned numbers.

    Spin (``spin``): a ``BallSpin`` bends the trajectory via Magnus. The
    deviation grows with spin rate x ball speed x flight-time² and is zero at
    both endpoints (the launcher aims for the curled target), so the ball
    still lands where it was aimed while its mid-flight path bulges — a side
    spinner's banana arc, a backspin chip's raised float, a topspin drive's
    sink. No spin → the old dead, perfectly-chordal flight.
    """

    start: Vec3
    target: Vec3
    duration: float
    apex_z: float
    # Ball velocity components (derived from trajectory)
    initial_speed: float = 20.0
    # True-projectile mode (launch velocity + gravity derived trajectory).
    ballistic: bool = False
    launch_speed: float = 0.0
    launch_angle_rad: float = 0.0
    gravity: float = 9.81
    # Magnus deco: how the ball is spinning through the air.
    spin: Optional[BallSpin] = None
    magnus_k: float = MAGNUS_K

    def _magnus_vertical_accel(self) -> float:
        """Signed Magnus acceleration in the vertical plane (m/s²).

        +upward for back spin (holds the ball up), −downward for top spin
        (drives it down), 0 for side spin / no spin.
        """
        a = magnus_accel(self.spin, self.initial_speed)
        if self.spin is None or not self.spin.active:
            return 0.0
        if self.spin.kind == TOP_SPIN:
            return -a
        if self.spin.kind == BACK_SPIN:
            return +a
        return 0.0

    def _magnus_side_accel(self) -> float:
        """Signed Magnus acceleration in the horizontal plane (m/s²)."""
        a = magnus_accel(self.spin, self.initial_speed)
        if self.spin is not None and self.spin.active and self.spin.kind == SIDE_SPIN:
            return a * self.spin.sign
        return 0.0

    def _magnus_deviation(self, time_s: float, duration: float) -> Vec3:
        """Chord-relative bend (m) from Magnus at ``time_s`` in a ``duration`` flight.

        Lateral: a constant side acceleration the launcher compensates for
        gives d(t) = ½·a·t·(t−T) — zero at both endpoints, peaking aT²/8 at
        mid-flight (the banana shape). Vertical: back spin floats the arc
        (apex lifted, same landing height), top spin sinks it — signed so the
        reported bend is zero at launch and target and the flown trajectory
        stays exactly aim-respecting.
        """
        if self.spin is None or not self.spin.active or duration <= 0.0:
            return Vec3(0.0, 0.0, 0.0)
        t = max(0.0, min(duration, time_s))
        T = duration
        a_lat = self._magnus_side_accel()
        a_vert = self._magnus_vertical_accel()
        if a_lat == 0.0 and a_vert == 0.0:
            return Vec3(0.0, 0.0, 0.0)
        # Perpendicular unit vector (in the x-y plane) of the travel chord.
        dx = self.target.x - self.start.x
        dy = self.target.y - self.start.y
        n_len = math.hypot(dx, dy)
        perp_x = 0.0
        perp_y = 1.0
        if n_len > 1e-9:
            perp_x = -dy / n_len
            perp_y = dx / n_len
        lat = 0.5 * a_lat * t * (t - T)
        vert = -0.5 * a_vert * t * (t - T)
        return Vec3(perp_x * lat, perp_y * lat, vert)

    def position_at(self, time_s: float) -> Vec3:
        if self.ballistic and self.launch_speed > 0:
            t = max(0.0, time_s)
            if self.duration > 0 and t > self.duration:
                t = self.duration
            tx = 0.0 if self.duration <= 0 else min(1.0, t / self.duration)
            # Horizontal motion is constant speed along the launch bearing.
            x = self.start.x + (self.target.x - self.start.x) * tx
            y = self.start.y + (self.target.y - self.start.y) * tx
            # Vertical motion is a true projectile: z(t) = z0 + v0z*t - ½ g t².
            # The Magnus deviation (below) adds the top/back-spin float or
            # sink on top of it; the effective-gravity shift keeps the landing
            # point exactly on target (deviation is 0 at both endpoints).
            v0z = self.launch_speed * math.sin(self.launch_angle_rad)
            z = self.start.z + v0z * t - 0.5 * self.gravity * t * t

            # Spin bend: side curl deviates horizontally; top/back spin alters
            # the arc (both are chord-relative, so landing stays on target).
            d = self._magnus_deviation(t, self.duration)
            return Vec3(x + d.x, y + d.y, max(0.0, z + d.z))
        t = max(0.0, min(1.0, time_s / max(0.001, self.duration)))
        x = self.start.x + (self.target.x - self.start.x) * t
        y = self.start.y + (self.target.y - self.start.y) * t
        # A quadratic Bezier arch gives a controllable, deterministic flight.
        z_linear = self.start.z + (self.target.z - self.start.z) * t
        z_arc = 4.0 * (self.apex_z - (self.start.z + self.target.z) / 2.0) * t * (1.0 - t)
        # Magnus bend applied at the absolute time coordinate for consistency
        # with the side-spin banana.
        d = self._magnus_deviation(time_s, self.duration)
        return Vec3(x + d.x, y + d.y, max(0.0, z_linear + z_arc + d.z))
    
    def velocity_at(self, time_s: float) -> Vec3:
        """Approximate instantaneous velocity at a given time."""
        dt = 0.01
        p1 = self.position_at(max(0.0, time_s - dt))
        p2 = self.position_at(min(self.duration, time_s + dt))
        return Vec3(
            (p2.x - p1.x) / (2 * dt),
            (p2.y - p1.y) / (2 * dt),
            (p2.z - p1.z) / (2 * dt),
        )
    
    def speed_at(self, time_s: float) -> float:
        """Ball speed at a given time."""
        v = self.velocity_at(time_s)
        return math.hypot(v.x, v.y, v.z)


@dataclass(frozen=True)
class AerialResolution:
    outcome: str  # controlled | contested | drops
    contact_point: Vec3
    contact_time: float
    winner: Optional[MovingPlayer] = None
    challenger: Optional[MovingPlayer] = None
    # Jump timing and vertical reach at contact
    winner_jump_time: float = 0.0
    winner_reach_height: float = 0.0
    challenger_reach_height: float = 0.0


@dataclass(frozen=True)
class AerialPassResolution:
    """Outcome of a lofted / long pass resolved as a 3D aerial duel.

    Mirrors ``PassResolution``'s outcome vocabulary (received | intercepted |
    underhit) but settles contact through ball height/arc over time, a
    receiver's jump/reach at the landing point, and a contesting defender's
    aerial-duel geometry (jump timing + vertical reach) — not a flat roll.
    """

    outcome: str  # received | intercepted | underhit
    ball_travel_time: float
    contact_point: Vec3
    receiver_arrival_time: float
    winner: Optional[MovingPlayer] = None
    winner_is_receiver: bool = False
    challenger: Optional[MovingPlayer] = None
    # Jump timing and vertical reach at the aerial contact
    winner_jump_time: float = 0.0
    winner_reach_height: float = 0.0
    challenger_reach_height: float = 0.0
    # Per-tick 3D ball positions for analytics
    ball_trajectory: Optional[List[Tuple[float, Vec3]]] = None
    # Ballistic launch analytics (populated by the long-ball resolver)
    launch_speed: float = 0.0
    launch_angle_deg: float = 0.0
    apex_height: float = 0.0
    wind_drift_m: float = 0.0
    # Magnus deco carried for analytics/exporters.
    spin_rate: float = 0.0
    spin_kind: str = ""


@dataclass(frozen=True)
class ShotResolution:
    outcome: str  # goal | saved | woodwork | wide | blocked
    goal_point: Vec3
    flight_time: float
    goalkeeper: Optional[MovingPlayer] = None
    blocker: Optional[MovingPlayer] = None
    rebound: Optional[Vec2] = None
    # Detailed physics for analytics
    gk_position_at_save: Optional[Vec2] = None
    gk_dive_time: float = 0.0
    ball_speed_at_contact: float = 20.0
    rebound_velocity: Optional[Vec2] = None


@dataclass(frozen=True)
class ReboundTrajectory:
    """Velocity-preserving rebound after a save, woodwork hit, or block."""
    
    origin: Vec3
    direction: Vec2
    speed: float
    decay: float = 0.85  # Energy loss on contact
    
    def position_at(self, time_s: float) -> Vec3:
        """Ground projection of the rebound (z=0 for simplicity)."""
        distance = self.speed * time_s * self.decay
        return Vec3(
            self.origin.x + self.direction.x * distance,
            self.origin.y + self.direction.y * distance,
            0.0,
        )


def make_flight(
    start: Vec3,
    target: Vec3,
    speed: float,
    apex_z: Optional[float] = None,
    spin: Optional[BallSpin] = None,
) -> BallFlight:
    distance = start.horizontal().distance_to(target.horizontal())
    duration = distance / max(6.0, speed)
    apex = apex_z if apex_z is not None else max(start.z, target.z) + min(8.0, distance * 0.08)
    return BallFlight(start, target, duration, apex, initial_speed=speed, spin=spin)


def make_ballistic_flight(
    start: Vec3,
    target: Vec3,
    speed: float,
    gravity: float = 9.81,
    loft: float = 0.0,
    spin: Optional[BallSpin] = None,
) -> BallFlight:
    """Launch a long ball as a true projectile under gravity.

    Opposite of ``make_flight``: the flight parameters are NOT picked by the
    caller. The ball is struck at ``speed`` m/s at a launch angle solved from
    the actual geometry (range and target height), and flight time plus peak
    height then follow from the launch velocity vector:

        u = tan(θ) solves  Δz = R·u − (g·R²/(2·v0²))·(1 + u²)   for  θ

    * ``loft`` blends between the low (driven, most efficient) launch angle
      (0.0) and the high (lofted, ballooned) solution (1.0) when the target
      is reachable at that leg speed.
    * If the target is OUT of physical range at ``speed`` (the quadratic has
      no real root), the ball is launched at 45° — the maximum-range angle —
      and the returned flight ends at its TRUE touchdown point short of the
      target, so the resolution genuinely reports an underhit drop instead of
      teleporting a low-speed hoof across the pitch.
    * ``spin`` decorates the flight with Magnus: the analytic launch-angle
      solve stays exact (the bend is zero at both endpoints) but the arc
      floats/sinks under top/back spin and the mid-flight banana of a side-spin
      curler becomes part of the resolved geometry.

    Gravity defaults to 9.81 m/s² (ballistic is never 'tuned' by an apex
    parameter; the arc is whatever physics makes of the launch vector).
    """

    dx = target.x - start.x
    dy = target.y - start.y
    range_m = math.hypot(dx, dy)
    bearing = Vec2(dx, dy).normalized() if range_m > 1e-9 else Vec2(0.0, 0.0)
    v0 = max(6.0, speed)
    dz = target.z - start.z
    g = gravity

    def _apex_with_vertical_spin(base_v0z: float, flight_time: float) -> float:
        """Peak of z(t) = z0 + v0z·t − ½·g·t² − ½·a_vert·t·(t−T).

        ``a_vert`` (>0 for back spin floats the arc up, <0 for top spin sinks
        it) shifts the effective gravity, so the reported apex follows the true
        bent arc instead of the raw launch vector. The landing point is
        untouched (deviation is 0 at both endpoints). Side spin is horizontal
        and leaves the vertical arc alone.
        """
        a_v = 0.0
        if spin is not None and spin.active and spin.kind in (TOP_SPIN, BACK_SPIN):
            a_v = magnus_accel(spin, v0)
            if spin.kind == TOP_SPIN:
                a_v = -a_v
            cap = g * MAGNUS_VERTICAL_SPIN_CAP
            a_v = max(-cap, min(cap, a_v))
        if abs(a_v) < 1e-6:
            return start.z + base_v0z * base_v0z / (2.0 * g)
        denom = g + a_v
        if denom <= 0.0:
            return start.z + base_v0z * base_v0z / (2.0 * g)
        t_apex = (base_v0z + 0.5 * a_v * flight_time) / denom
        z_apex = (
            start.z + base_v0z * t_apex - 0.5 * g * t_apex * t_apex
            - 0.5 * a_v * t_apex * (t_apex - flight_time)
        )
        return z_apex

    if range_m > 1e-9:
        # u = tan(θ). The target is reachable iff the quadratic has a real root.
        A = g * range_m * range_m / (2.0 * v0 * v0)
        disc = range_m * range_m - 4.0 * A * (dz + A)
        if disc > 1e-9:
            root = math.sqrt(disc)
            low_u = (range_m - root) / (2.0 * A)
            high_u = (range_m + root) / (2.0 * A)
            u = low_u * (1.0 - loft) + high_u * loft
            theta = math.atan2(u, 1.0)
            v0z = v0 * math.sin(theta)
            v0xy = v0 * math.cos(theta)
            t_arrive = range_m / v0xy
            apex = _apex_with_vertical_spin(v0z, t_arrive)
            landing = Vec3(target.x, target.y, max(0.0, target.z))
            return BallFlight(
                start, landing, t_arrive, max(apex, start.z + 0.2),
                initial_speed=v0, ballistic=True,
                launch_speed=v0, launch_angle_rad=theta, gravity=g,
                spin=spin,
            )

    # Unreachable at this leg speed: strike at max range angle and let the
    # ball fall where physics puts it — short of the intended target.
    theta = math.radians(45.0)
    v0z = v0 * math.sin(theta)
    v0xy = v0 * math.cos(theta)
    t_land = (v0z + math.sqrt(v0z * v0z + 2.0 * g * max(0.0, start.z))) / g
    short_range = v0xy * t_land
    landing = Vec3(
        start.x + bearing.x * short_range,
        start.y + bearing.y * short_range,
        0.0,
    )
    apex = _apex_with_vertical_spin(v0z, t_land)
    return BallFlight(
        start, landing, t_land, max(apex, start.z + 0.2),
        initial_speed=v0, ballistic=True,
        launch_speed=v0, launch_angle_rad=theta, gravity=g,
        spin=spin,
    )


def resolve_aerial_delivery(
    flight: BallFlight,
    attackers: Iterable[MovingPlayer],
    defenders: Iterable[MovingPlayer],
    sample_step: float = TICK_S,
    player_context: Optional[PlayerRaceContext] = None,
) -> AerialResolution:
    """Resolve a cross, clearance, or lofted pass by first physical contact.
    
    Enhanced with jump timing and vertical reach model for aerial duels, and
    movement-cost-aware arrival (heading/entry context) for each contestant.
    """
    attacking_players = tuple(attackers)
    defending_players = tuple(defenders)
    candidates: list[tuple[MovingPlayer, bool, float, Vec3, float, float]] = []
    
    steps = max(1, int(math.ceil(flight.duration / sample_step)))
    for index in range(1, steps + 1):
        time_s = flight.duration * index / steps
        point = flight.position_at(time_s)
        airborne = point.z > 1.15
        
        for player in attacking_players + defending_players:
            # Check vertical reach first (player must be able to reach this height)
            max_reach = player.vertical_reach(airborne)
            if point.z > max_reach:
                continue
            
            # Check horizontal arrival time (turn/entry-aware with context)
            arrival = _race_motion(player, point.horizontal(), player.control_radius, player_context)
            if arrival <= time_s:
                # Calculate jump timing: when must player leave ground to reach point.z?
                # Jump time is the time before contact when player initiates jump
                # Higher jumps require earlier takeoff
                jump_height_needed = max(0.0, point.z - player.standing_reach)
                if jump_height_needed > 0:
                    # Time to reach peak of jump (parabolic: t = sqrt(2h/g), g≈9.8)
                    jump_time_to_peak = math.sqrt(2.0 * jump_height_needed / 9.8)
                    # Player must leave ground before this time
                    jump_start_time = time_s - jump_time_to_peak
                else:
                    jump_start_time = time_s
                
                is_attacker = player in attacking_players
                candidates.append((
                    player, is_attacker, time_s, point,
                    jump_start_time, max_reach
                ))
        
        if candidates:
            break
    
    if not candidates:
        return AerialResolution("drops", flight.position_at(flight.duration), flight.duration)
    
    # Sort by: earliest contact time, then highest reach (taller/jumping players win)
    candidates.sort(key=lambda item: (item[2], -item[5]))
    
    winner, winner_is_attacker, time_s, point, jump_time, reach_height = candidates[0]
    
    # Find challenger (first player from opposite side who also could reach)
    challenger = None
    challenger_reach = 0.0
    for p, is_att, _, _, _, reach in candidates[1:]:
        if is_att != winner_is_attacker:
            challenger = p
            challenger_reach = reach
            break
    
    outcome = "contested" if challenger is not None else "controlled"
    
    return AerialResolution(
        outcome, point, time_s, winner, challenger,
        winner_jump_time=jump_time,
        winner_reach_height=reach_height,
        challenger_reach_height=challenger_reach,
    )


def resolve_aerial_pass(
    flight: BallFlight,
    receiver: MovingPlayer,
    defenders: Iterable[MovingPlayer],
    sample_step: float = TICK_S,
    player_context: Optional[PlayerRaceContext] = None,
) -> AerialPassResolution:
    """Resolve a lofted / long pass as a 3D aerial race (pass shape in 3D).

    The aerial equivalent of ``resolve_ground_pass`` / ``resolve_dribble``:
    the ball travels a ballistic arc (height over time, not just 2D x,y),
    and the outcome is decided by who can physically win the landing-point
    duel — a receiver's jump/reach to control it versus a contesting
    defender's aerial-duel geometry (jump timing + vertical reach).

    Outcomes:
        * ``received``     — the intended receiver wins the aerial duel and
                             can control the ball at the landing point.
        * ``intercepted``  — a defender wins the aerial duel first.
        * ``underhit``     — nobody reaches the landing point in time
                             (ball drops loose) or the receiver cannot arrive.
    """
    defenders_list = list(defenders)

    # Tick the 3D flight; at each sample gather every player who can reach
    # that point horizontally AND vertically (standing or airborne).
    candidates: list[tuple[MovingPlayer, bool, float, Vec3, float, float]] = []
    per_tick: List[Tuple[float, Vec3]] = []

    steps = max(1, int(math.ceil(flight.duration / sample_step)))
    for index in range(1, steps + 1):
        time_s = flight.duration * index / steps
        point = flight.position_at(time_s)
        per_tick.append((time_s, point))
        airborne = point.z > 1.15

        for player, is_receiver in (
            [(receiver, True)] + [(d, False) for d in defenders_list]
        ):
            max_reach = player.vertical_reach(airborne)
            if point.z > max_reach:
                continue
            if _race_motion(player, point.horizontal(), player.control_radius, player_context) > time_s:
                continue
            # Vertical height needed above standing reach drives the jump-time
            # geometry: the higher the point, the earlier the jump must start.
            jump_height_needed = max(0.0, point.z - player.standing_reach)
            if jump_height_needed > 0:
                jump_time_to_peak = math.sqrt(2.0 * jump_height_needed / 9.8)
                jump_start_time = time_s - jump_time_to_peak
            else:
                jump_start_time = time_s
            candidates.append((
                player, is_receiver, time_s, point, jump_start_time, max_reach,
            ))

        if candidates:
            break

    receiver_arrival = _race_motion(receiver, flight.target.horizontal(), None, player_context)

    if not candidates:
        # No one can get to the landing point in either plane — the ball drops.
        return AerialPassResolution(
            "underhit", flight.duration, flight.position_at(flight.duration),
            receiver_arrival,
            ball_trajectory=per_tick,
        )

    # Earliest reachable contact wins; ties broken by highest vertical reach
    # (the taller / better-jumping aerial player beats into the point).
    candidates.sort(key=lambda item: (item[2], -item[5]))
    winner, winner_is_receiver, time_s, point, jump_time, reach_height = candidates[0]

    if not winner_is_receiver:
        # A defender won the landing-point duel → intercepted.
        challenger_reach = 0.0
        for p, is_att, _, _, _, reach in candidates[1:]:
            if is_att:
                challenger_reach = reach
                break
        return AerialPassResolution(
            "intercepted", time_s, point, receiver_arrival, winner,
            winner_is_receiver=False, challenger=winner,
            winner_jump_time=jump_time, winner_reach_height=reach_height,
            challenger_reach_height=challenger_reach,
            ball_trajectory=per_tick,
        )

    # Receiver won the duel. First touch window analogous to a ground pass:
    # if the receiver can't arrive (or a defender pressed close enough to make
    # it a genuine contest), still decide by geometry.
    first_touch_window = 0.35 + receiver.control_radius / max(6.0, receiver.top_speed)
    challenger = None
    challenger_reach = 0.0
    for p, is_att, _, _, _, reach in candidates[1:]:
        if not is_att:
            challenger = p
            challenger_reach = reach
            break

    if receiver_arrival > time_s + first_touch_window:
        return AerialPassResolution(
            "underhit", time_s, point, receiver_arrival, winner,
            winner_is_receiver=True, challenger=challenger,
            winner_jump_time=jump_time, winner_reach_height=reach_height,
            challenger_reach_height=challenger_reach,
            ball_trajectory=per_tick,
        )

    return AerialPassResolution(
        "received", time_s, point, receiver_arrival, winner,
        winner_is_receiver=True, challenger=challenger,
        winner_jump_time=jump_time, winner_reach_height=reach_height,
        challenger_reach_height=challenger_reach,
        ball_trajectory=per_tick,
    )


def _calculate_rebound_velocity(
    flight: BallFlight,
    contact_point: Vec3,
    normal: Vec2,
    decay: float = 0.85,
) -> Tuple[Vec2, float]:
    """Calculate velocity-preserving rebound direction and speed.
    
    Uses reflection across the surface normal with energy decay.
    """
    incoming_vel = flight.velocity_at(flight.duration)
    incoming_2d = Vec2(incoming_vel.x, incoming_vel.y)
    
    # Reflect: v' = v - 2(v·n)n
    dot = incoming_2d.dot(normal)
    reflected = Vec2(
        incoming_2d.x - 2.0 * dot * normal.x,
        incoming_2d.y - 2.0 * dot * normal.y,
    )
    
    # Apply energy decay
    speed = math.hypot(incoming_vel.x, incoming_vel.y) * decay
    
    return reflected.normalized(), speed


def resolve_shot(
    flight: BallFlight,
    goalkeeper: Optional[MovingPlayer],
    blockers: Iterable[MovingPlayer] = (),
    attacks_right: bool = True,
    goal_left: float = 30.34,
    goal_right: float = 37.66,
    goal_height: float = 2.44,
    sample_step: float = TICK_S,
) -> ShotResolution:
    """Resolve a shot at the goal plane using flight time and keeper reach.
    
    Enhanced with:
    - Continuous goalkeeper positioning during flight
    - Velocity-preserving rebounds
    - Per-tick tracking of all participants
    
    The keeper must reach the goal-plane contact point: they react, then dive
    at ``dive_speed`` with ``dive_reach`` lateral extension beyond the control
    radius, and only a ball below their dive vertical reach can be saved.
    """
    # Track blockers throughout flight
    blocker_positions: List[Tuple[float, MovingPlayer, Vec2]] = []

    for blocker in blockers:
        steps = max(1, int(math.ceil(flight.duration / sample_step)))
        for index in range(1, steps):
            time_s = flight.duration * index / steps
            point = flight.position_at(time_s)
            blocker_positions.append((time_s, blocker, point))
            if point.z > blocker.vertical_reach(point.z > 1.15):
                continue
            if blocker.time_to_reach(point.horizontal(), blocker.tackle_radius) <= time_s:
                # Calculate rebound from blocker
                rebound_dir, rebound_speed = _calculate_rebound_velocity(
                    flight, point,
                    # Normal pointing away from goal
                    Vec2(-1.0 if attacks_right else 1.0, 0.0),
                    decay=0.75,
                )
                rebound = Vec2(
                    point.x + rebound_dir.x * 5.0,
                    point.y + rebound_dir.y * 5.0,
                )
                return ShotResolution(
                    "blocked", point, time_s, blocker=blocker,
                    rebound=rebound,
                    ball_speed_at_contact=flight.speed_at(time_s),
                    rebound_velocity=rebound_dir * rebound_speed,
                )
    
    point = flight.position_at(flight.duration)
    in_frame = goal_left <= point.y <= goal_right and 0.0 <= point.z <= goal_height
    near_post = min(abs(point.y - goal_left), abs(point.y - goal_right)) <= 0.12
    near_bar = abs(point.z - goal_height) <= 0.10
    
    if near_post or near_bar:
        # Woodwork rebound with velocity preservation
        normal = Vec2(0.0, 1.0 if abs(point.y - goal_left) < abs(point.y - goal_right) else -1.0)
        if near_bar:
            normal = Vec2(-1.0 if attacks_right else 1.0, 0.0)
        rebound_dir, rebound_speed = _calculate_rebound_velocity(flight, point, normal, decay=0.70)
        rebound = Vec2(
            point.x + rebound_dir.x * 6.0,
            point.y + rebound_dir.y * 6.0,
        )
        return ShotResolution(
            "woodwork", point, flight.duration,
            rebound=rebound,
            ball_speed_at_contact=flight.speed_at(flight.duration),
            rebound_velocity=rebound_dir * rebound_speed,
        )
    
    if not in_frame:
        return ShotResolution(
            "wide", point, flight.duration,
            ball_speed_at_contact=flight.speed_at(flight.duration),
        )
    
    if goalkeeper is None:
        return ShotResolution(
            "goal", point, flight.duration,
            ball_speed_at_contact=flight.speed_at(flight.duration),
        )

    # ── CONTINUOUS GOALKEEPER POSITIONING ─────────────────────────────
    # Keeper tracks the ball during flight, adjusting position along goal line
    # This is what real keepers do: they don't stand frozen, they anticipate
    
    # Initial keeper position (on the line)
    gk_line_x = 104.5 if attacks_right else 0.5
    gk_start_y = goalkeeper.position.y
    
    # During flight, keeper tracks ball trajectory
    # For central shots, keeper stays central; for wide shots, keeper shades toward near post
    ball_start_y = flight.start.y
    ball_target_y = point.y
    
    # Keeper anticipation: reads trajectory and pre-moves
    # Lateral movement starts BEFORE the ball reaches the goal plane
    anticipation_factor = min(1.0, flight.duration / 0.5)  # More time = more anticipation
    target_y = ball_start_y + (ball_target_y - ball_start_y) * anticipation_factor
    
    # Keeper can only move so fast along the line
    max_lateral_speed = goalkeeper.top_speed * 0.6  # Shuffling, not sprinting
    max_drift = max_lateral_speed * flight.duration
    drift = max(-max_drift, min(max_drift, target_y - gk_start_y))
    
    gk_final_y = max(goal_left + 0.5, min(goal_right - 0.5, gk_start_y + drift))
    gk_final_x = gk_line_x
    
    # Keeper must reach the goal-plane contact point after reacting. Reach
    # is governed by a genuine dive: the keeper reacts, then accelerates
    # laterally at ``dive_speed`` and can only cover ``dive_reach`` spread at
    # the goal-line plane before the ball crosses.
    horizontal_target = Vec2(point.x, point.y)
    lateral = abs(point.y - gk_final_y)
    longitudinal = abs(point.x - gk_final_x)
    # Coming forward from a few metres off the line costs a little extra.
    lateral += max(0.0, longitudinal - goalkeeper.dive_reach * 0.4) * 0.18
    overhead = point.z > 1.9
    vertical_ok = point.z <= (goalkeeper.vertical_reach(True) - (0.25 if overhead else 0.0))
    
    if not vertical_ok:
        return ShotResolution(
            "goal", point, flight.duration,
            gk_position_at_save=Vec2(gk_final_x, gk_final_y),
            ball_speed_at_contact=flight.speed_at(flight.duration),
        )

    # Body save: a ball within the keeper's immediate frame is smothered the
    # instant it is struck — no dive reaction time is needed because the
    # keeper's body is already in the goal mouth (this is what stops central
    # point-blank efforts from being automatic goals).
    body_cover = goalkeeper.control_radius + goalkeeper.dive_reach * 0.18
    if lateral <= body_cover:
        rebound_dir, rebound_speed = _calculate_rebound_velocity(
            flight, point,
            Vec2(-1.0 if attacks_right else 1.0, 0.0),
            decay=0.65,
        )
        rebound = Vec2(
            point.x + rebound_dir.x * 5.0,
            point.y + rebound_dir.y * 5.0,
        )
        return ShotResolution(
            "saved", point, flight.duration, goalkeeper,
            rebound=rebound,
            gk_position_at_save=Vec2(gk_final_x, gk_final_y),
            gk_dive_time=0.0,  # Body save, no dive needed
            ball_speed_at_contact=flight.speed_at(flight.duration),
            rebound_velocity=rebound_dir * rebound_speed,
        )

    # True dive: the keeper reacts, then covers ``dive_reach`` lateral spread
    # at ``dive_speed`` before the ball crosses.
    dive_reach = goalkeeper.dive_reach + 2.0
    # Anticipation: from distance the keeper reads the shot off the foot and
    # moves before impact — effective reaction shrinks as the shot travels.
    shot_dist = math.hypot(flight.target.x - flight.start.x, flight.target.y - flight.start.y)
    anticipation = 0.25 + 0.75 * min(1.0, 10.0 / max(5.0, shot_dist))
    dive_time = goalkeeper.reaction_time * anticipation + lateral / max(1.0, goalkeeper.dive_speed)
    
    if lateral <= dive_reach and dive_time <= flight.duration:
        rebound_dir, rebound_speed = _calculate_rebound_velocity(
            flight, point,
            Vec2(-1.0 if attacks_right else 1.0, 0.0),
            decay=0.70,
        )
        rebound = Vec2(
            point.x + rebound_dir.x * 6.0,
            point.y + rebound_dir.y * 6.0,
        )
        return ShotResolution(
            "saved", point, flight.duration, goalkeeper,
            rebound=rebound,
            gk_position_at_save=Vec2(gk_final_x, gk_final_y),
            gk_dive_time=dive_time,
            ball_speed_at_contact=flight.speed_at(flight.duration),
            rebound_velocity=rebound_dir * rebound_speed,
        )
    
    return ShotResolution(
        "goal", point, flight.duration,
        gk_position_at_save=Vec2(gk_final_x, gk_final_y),
        ball_speed_at_contact=flight.speed_at(flight.duration),
    )


def resolve_ground_pass(
    start: Vec2,
    target: Vec2,
    receiver: MovingPlayer,
    defenders: Iterable[MovingPlayer],
    ball_speed: float,
    sample_step: float = TICK_S,
    pressure_level: float = 0.0,
    rolling_decel: Optional[float] = None,
    player_context: Optional[PlayerRaceContext] = None,
) -> PassResolution:
    """Resolve a ground pass as a race between ball flight and player reach.

    A defender intercepts only when they can physically get within their
    control radius of the moving ball before it arrives. The receiver must
    reach the intended target no later than a short first-touch window after
    ball arrival. There is intentionally no random completion roll.

    Rolling friction: the ball decelerates against the turf at
    ``rolling_decel`` m/s² (default ``BALL_ROLLING_DECEL``, weather-scaled by
    the caller). Arrival time solves the constant-deceleration quadratic, so a
    ball that is under-struck for the distance genuinely dies short of the
    target (``underhit``, ``ball_stopped_short=True``) and every long pass
    arrives slower than it left the boot (``ball_speed_at_arrival``).
    
    Enhanced with:
    - Per-tick ball position tracking (friction-corrected)
    - Pressure-driven accuracy modifiers
    - Continuous defender motion tracking
    """
    distance = start.distance_to(target)
    speed = max(6.0, ball_speed)
    
    # Pressure affects ball speed (rushed passes are slower or less accurate)
    if pressure_level > 0.5:
        speed *= (1.0 - 0.1 * (pressure_level - 0.5))  # Up to 5% speed reduction
    
    k = BALL_ROLLING_DECEL if rolling_decel is None else max(0.0, rolling_decel)

    arrival_time = rolling_arrival_time(distance, speed, k)
    stopped_short = arrival_time is None
    if stopped_short:
        # The ball dies before covering the target distance.
        rest_time = speed / k if k > 0 else 0.0
        travel_time = rest_time
        rest_dist = speed * speed / (2.0 * k) if k > 0 else 0.0
        rest_point = start.lerp(target, min(1.0, rest_dist / distance)) if distance else target
    else:
        travel_time = arrival_time
    
    receiver_time = _race_motion(receiver, target, None, player_context)
    
    # Track ball trajectory per tick (friction-corrected positions)
    ball_trajectory: List[Tuple[float, Vec2]] = []
    
    earliest: Optional[tuple[MovingPlayer, float, Vec2]] = None
    defender_positions: List[Tuple[float, Vec2]] = []
    
    steps = max(1, int(math.ceil(max(0.001, travel_time) / sample_step)))
    last_point = target if not stopped_short else rest_point
    for index in range(1, steps + 1):
        time_s = travel_time * index / steps
        if stopped_short and time_s >= rest_time:
            point = rest_point
        else:
            covered = rolling_position(speed, k, time_s)
            point = start.lerp(target, min(1.0, covered / distance)) if distance else target
        
        # Record ball position for trajectory
        ball_trajectory.append((time_s, point))
        last_point = point
        
        for defender in defenders:
            if defender is None:
                continue
            intercept_radius = 1.0
            arrival = _race_motion(defender, point, intercept_radius, player_context)
            if arrival <= time_s:
                if earliest is None or time_s < earliest[1]:
                    earliest = (defender, time_s, point)
    
    # Arrival speed at contact (0 while still rolling if it died short).
    ball_speed_at_arrival = rolling_speed(speed, k, travel_time)
    
    # A receiver cannot control a ball they cannot arrive to within this
    # physically meaningful first-touch window, even if no defender wins it.
    first_touch_window = 0.35 + receiver.control_radius / max(6.0, receiver.top_speed)
    
    # Pressure affects first-touch window (harder to control under pressure)
    if pressure_level > 0.3:
        first_touch_window *= (1.0 + 0.2 * pressure_level)
    
    if earliest is not None and earliest[1] <= travel_time:
        defender, intercept_time, point = earliest
        return PassResolution(
            outcome="intercepted",
            ball_travel_time=travel_time,
            contact_point=point,
            receiver_arrival_time=receiver_time,
            interceptor=defender,
            interceptor_arrival_time=intercept_time,
            ball_trajectory=ball_trajectory,
            ball_speed_at_arrival=rolling_speed(speed, k, intercept_time),
            ball_stopped_short=False,
        )
    
    if stopped_short or receiver_time > travel_time + first_touch_window:
        return PassResolution(
            outcome="underhit",
            ball_travel_time=travel_time,
            contact_point=last_point,
            receiver_arrival_time=receiver_time,
            ball_trajectory=ball_trajectory,
            ball_speed_at_arrival=ball_speed_at_arrival,
            ball_stopped_short=stopped_short,
        )
    
    return PassResolution(
        outcome="received",
        ball_travel_time=travel_time,
        contact_point=target,
        receiver_arrival_time=receiver_time,
        ball_trajectory=ball_trajectory,
        ball_speed_at_arrival=ball_speed_at_arrival,
        ball_stopped_short=False,
    )


def resolve_dribble(
    start: Vec2,
    target: Vec2,
    attacker: MovingPlayer,
    defenders: Iterable[MovingPlayer],
    sample_step: float = TICK_S,
    pressure_level: float = 0.0,
    player_context: Optional[PlayerRaceContext] = None,
    rng: Optional[random.Random] = None,
) -> DribbleResolution:
    """Resolve a dribble by tracing the ball carrier and defenders in time.

    Take-ons are settled by reach geometry FIRST — a defender who cleanly
    beats the ball to the tackle envelope wins it outright (``reached_first``).
    When defender and carrier arrive at the same ball at the same instant (the
    tie window), the shoulder decides: body mass x momentum
    (``resolve_body_duel``). A decisive impulse bowls the carrier over
    (``bowled_over``); a decisive carrier edge shrugs the defender off
    (``shrugged_off``) and carries on; comparable momentum leaves a 50/50
    weighted by the impulse ratio.
    
    Enhanced with:
    - Continuous defender motion tracking
    - Pressure-driven contact window calculation
    - Movement-cost-aware pursuit (turn costs when a defender must change line)
    - Body-collision / momentum duels on simultaneous contact (#4)
    - Defender positions recorded at each tick
    """
    distance = start.distance_to(target)
    # Carrying the ball costs pace; good control improves the retained speed.
    dribble_speed = attacker.top_speed * 0.68
    duration = distance / max(2.5, dribble_speed)
    steps = max(1, int(math.ceil(duration / sample_step)))
    
    defender_positions: List[Tuple[float, Vec2]] = []
    defenders_list = list(defenders)
    shrugged_off: set[int] = set()
    _rng = rng or random

    for index in range(1, steps + 1):
        time_s = duration * index / steps
        ball = start.lerp(target, time_s / duration) if duration else target
        
        for defender in defenders_list:
            if id(defender) in shrugged_off:
                continue
            tackle_range = attacker.control_radius + defender.tackle_radius
            
            # Pressure affects tackle range (pressed defenders are more aggressive)
            effective_tackle_range = tackle_range
            if pressure_level > 0.3:
                effective_tackle_range *= (1.0 + 0.15 * pressure_level)
            
            arrival = _race_motion(defender, ball, effective_tackle_range, player_context)
            if arrival > time_s:
                continue
            
            margin = time_s - arrival
            if margin > BODY_DUEL_TIE_WINDOW:
                # The defender cleanly beat the ball on the ground first — the
                # carrier has no time to shield it. A pure reach win.
                return DribbleResolution(
                    "tackled", ball, time_s, defender,
                    defender_positions=defender_positions,
                    resolution_note="reached_first",
                )
            
            # Simultaneous contact: the shoulder-to-shoulder decides the
            # 50/50 via body mass x momentum (#4). The defender commits a
            # challenge sprint; the carrier shields at carry speed.
            defender_approach = defender.top_speed * DEFENDER_CHALLENGE_FACTOR
            carrier_approach = attacker.top_speed * CARRIER_SHIELD_FACTOR
            duel = resolve_body_duel(defender, attacker, defender_approach, carrier_approach)
            if duel.winner == "defender":
                return DribbleResolution(
                    "tackled", ball, time_s, defender,
                    defender_positions=defender_positions,
                    body_duel=duel, resolution_note="bowled_over",
                )
            if duel.winner == "carrier":
                # Strong, well-balanced carrier rides the shoulder and keeps
                # the ball — the dedicated defender is left behind for good.
                shrugged_off.add(id(defender))
                continue
            # Genuine 50/50 gray zone — weighted by the impulse ratio.
            if _rng.random() < _body_contest_win_probability(duel.impulse_ratio):
                return DribbleResolution(
                    "tackled", ball, time_s, defender,
                    defender_positions=defender_positions,
                    body_duel=duel, resolution_note="50_50",
                )
            shrugged_off.add(id(defender))
    
    return DribbleResolution(
        "retained", target, duration,
        defender_positions=defender_positions,
    )
