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
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Optional, Tuple


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

    @property
    def top_speed(self) -> float:
        """Pace 0..100 mapped to a real 5.0..9.2 m/s sprint range."""
        return 5.0 + max(0.0, min(100.0, self.pace)) * 0.042

    def time_to_reach(self, target: Vec2, radius: Optional[float] = None) -> float:
        """Earliest time at which the player can contact ``target`` in seconds."""
        remaining = max(0.0, self.position.distance_to(target) - (radius or self.control_radius))
        if remaining == 0.0:
            return 0.0
        acceleration = max(1.5, self.acceleration)
        top_speed = self.top_speed
        time_to_top = top_speed / acceleration
        distance_to_top = 0.5 * acceleration * time_to_top * time_to_top
        if remaining <= distance_to_top:
            return self.reaction_time + math.sqrt(2.0 * remaining / acceleration)
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


@dataclass(frozen=True)
class DribbleResolution:
    outcome: str  # retained | tackled
    contact_point: Vec2
    duration: float
    tackler: Optional[MovingPlayer] = None
    # Defender positions at each tick during the dribble
    defender_positions: Optional[List[Tuple[float, Vec2]]] = None


@dataclass(frozen=True)
class BallFlight:
    """A ballistic 3D ball flight between two pitch coordinates."""

    start: Vec3
    target: Vec3
    duration: float
    apex_z: float
    # Ball velocity components (derived from trajectory)
    initial_speed: float = 20.0

    def position_at(self, time_s: float) -> Vec3:
        t = max(0.0, min(1.0, time_s / max(0.001, self.duration)))
        x = self.start.x + (self.target.x - self.start.x) * t
        y = self.start.y + (self.target.y - self.start.y) * t
        # A quadratic Bezier arch gives a controllable, deterministic flight.
        z_linear = self.start.z + (self.target.z - self.start.z) * t
        z_arc = 4.0 * (self.apex_z - (self.start.z + self.target.z) / 2.0) * t * (1.0 - t)
        return Vec3(x, y, max(0.0, z_linear + z_arc))
    
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
) -> BallFlight:
    distance = start.horizontal().distance_to(target.horizontal())
    duration = distance / max(6.0, speed)
    apex = apex_z if apex_z is not None else max(start.z, target.z) + min(8.0, distance * 0.08)
    return BallFlight(start, target, duration, apex, initial_speed=speed)


def resolve_aerial_delivery(
    flight: BallFlight,
    attackers: Iterable[MovingPlayer],
    defenders: Iterable[MovingPlayer],
    sample_step: float = TICK_S,
) -> AerialResolution:
    """Resolve a cross, clearance, or lofted pass by first physical contact.
    
    Enhanced with jump timing and vertical reach model for aerial duels.
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
            
            # Check horizontal arrival time
            arrival = player.time_to_reach(point.horizontal(), player.control_radius)
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
) -> PassResolution:
    """Resolve a ground pass as a race between ball flight and player reach.

    A defender intercepts only when they can physically get within their
    control radius of the moving ball before it arrives. The receiver must
    reach the intended target no later than a short first-touch window after
    ball arrival. There is intentionally no random completion roll.
    
    Enhanced with:
    - Per-tick ball position tracking
    - Pressure-driven accuracy modifiers
    - Continuous defender motion tracking
    """
    distance = start.distance_to(target)
    speed = max(6.0, ball_speed)
    
    # Pressure affects ball speed (rushed passes are slower or less accurate)
    if pressure_level > 0.5:
        speed *= (1.0 - 0.1 * (pressure_level - 0.5))  # Up to 5% speed reduction
    
    travel_time = distance / speed
    receiver_time = receiver.time_to_reach(target)
    
    # Track ball trajectory per tick
    ball_trajectory: List[Tuple[float, Vec2]] = []
    
    earliest: Optional[tuple[MovingPlayer, float, Vec2]] = None
    defender_positions: List[Tuple[float, Vec2]] = []
    
    steps = max(1, int(math.ceil(travel_time / sample_step)))
    for index in range(1, steps + 1):
        time_s = travel_time * index / steps
        point = start.lerp(target, time_s / travel_time) if travel_time else target
        
        # Record ball position for trajectory
        ball_trajectory.append((time_s, point))
        
        for defender in defenders:
            intercept_radius = 1.0
            arrival = defender.time_to_reach(point, intercept_radius)
            if arrival <= time_s:
                if earliest is None or time_s < earliest[1]:
                    earliest = (defender, time_s, point)
    
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
        )
    
    if receiver_time > travel_time + first_touch_window:
        return PassResolution(
            outcome="underhit",
            ball_travel_time=travel_time,
            contact_point=target,
            receiver_arrival_time=receiver_time,
            ball_trajectory=ball_trajectory,
        )
    
    return PassResolution(
        outcome="received",
        ball_travel_time=travel_time,
        contact_point=target,
        receiver_arrival_time=receiver_time,
        ball_trajectory=ball_trajectory,
    )


def resolve_dribble(
    start: Vec2,
    target: Vec2,
    attacker: MovingPlayer,
    defenders: Iterable[MovingPlayer],
    sample_step: float = TICK_S,
    pressure_level: float = 0.0,
) -> DribbleResolution:
    """Resolve a dribble by tracing the ball carrier and defenders in time.
    
    Enhanced with:
    - Continuous defender motion tracking
    - Pressure-driven contact window calculation
    - Defender positions recorded at each tick
    """
    distance = start.distance_to(target)
    # Carrying the ball costs pace; good control improves the retained speed.
    dribble_speed = attacker.top_speed * 0.68
    duration = distance / max(2.5, dribble_speed)
    steps = max(1, int(math.ceil(duration / sample_step)))
    
    defender_positions: List[Tuple[float, Vec2]] = []
    defenders_list = list(defenders)

    for index in range(1, steps + 1):
        time_s = duration * index / steps
        ball = start.lerp(target, time_s / duration) if duration else target
        
        for defender in defenders_list:
            tackle_range = attacker.control_radius + defender.tackle_radius
            
            # Pressure affects tackle range (pressed defenders are more aggressive)
            effective_tackle_range = tackle_range
            if pressure_level > 0.3:
                effective_tackle_range *= (1.0 + 0.15 * pressure_level)
            
            if defender.time_to_reach(ball, effective_tackle_range) <= time_s:
                return DribbleResolution(
                    "tackled", ball, time_s, defender,
                    defender_positions=defender_positions,
                )
    
    return DribbleResolution(
        "retained", target, duration,
        defender_positions=defender_positions,
    )
