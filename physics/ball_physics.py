"""
physics/ball_physics.py — how long the ball takes to get there.
=============================================================

A pass in the current engine carries a start point and an end point and
nothing else::

    start_x, start_y, end_x, end_y

That is enough to draw a pass map and not enough to play against one. The
brief's §9 asks for more, and the reason is not realism for its own sake — it
is that a through ball is a *race*, and a race needs two finish times::

    Ball reaches location at:  t = 2.80 s
    Attacker reaches it at:    t = 2.45 s   -> attacker wins
    Defender reaches it at:    t = 2.71 s

Without travel time there is nothing to compare, so the outcome falls back to
``random.random() < success_probability`` — which §13 explicitly names as the
thing to replace.

Three models, in increasing order of sophistication, all of them closed-form
so the clock can jump straight to the event (§2)
------------------------------------------------
**Ground ball with drag.** Air resistance decelerates a rolling ball
exponentially, so ``v(t) = v0 * e^(-k t)`` and the distance covered is
``s(t) = (v0 / k) * (1 - e^(-k t))``. That inverts cleanly to
``t = -ln(1 - k*d/v0) / k``, which is what :func:`ground_travel_time` returns.
A closed form matters here: it means no stepping, so a match costs the same
whether it computes one pass or ten thousand.

**Ball deceleration is not optional.** §10's two requirements —
"LONGER DISTANCE + SAME BALL SPEED = LONGER TRAVEL TIME" and "SAME DISTANCE +
FASTER BALL = SHORGER TRAVEL TIME" — both fall out of the model, and a
hard-coded ``distance / speed`` would not honour the first.

**Vertical motion (§11).** A lightweight independent ``z`` with gravity::

    z(t) = z0 + vz * t - 0.5 * g * t^2

No spin, no Magnus, no turbulence — §11 says not to, and §24 says calibration
before complexity. A lofted ball launched at ``vz`` returns to its launch
height after ``2 * vz / g``, and its apex is ``vz^2 / (2g)``. The pitch stays
strictly x/y; ``z`` is one extra number on the ball, not a 3D engine.

Everything is deterministic. There is no randomness in this module, and
specifically no "ball travel jitter" — §9 forbids random travel times
unrelated to distance, and a randomised travel time would quietly undo the
whole point of the layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

#: Standard gravity, m/s^2.
GRAVITY = 9.80665

#: Exponential drag coefficient for a football rolling on grass, per second.
#: Calibrated so a 20 m/s ground pass loses roughly 15-20% of its speed over 30
#: m — a real ball, not a puck. PLOFA's own ``weather_physics`` exposes
#: ``rolling_decel_mult`` to scale this for conditions; that multiplier is
#: applied by the caller through :class:`BallPhysics`, keeping this module free
#: of weather imports.
BALL_DRAG_COEFFICIENT = 0.12

#: A ball cannot usefully exceed this (m/s). Beyond it the model is nonsense and
#: the value is a bug upstream, so it is clamped and detectable.
MAX_BALL_SPEED = 40.0

#: Net height for a goal, used as the reference landing height. A cross that
#: never comes down is not a cross.
GOAL_HEIGHT_M = 2.44


@dataclass(frozen=True)
class PassProfile:
    """A calibrated speed range for one kind of ball delivery.

    The ranges are the realistic ones for a top-flight player, in m/s. Midpoint
    is what :meth:`BallPhysics.nominal` uses when no explicit speed is given, so
    a pass's *character* comes from its type rather than from a random draw.
    """

    name: str
    min_speed: float
    max_speed: float
    #: 0.0 rolls along the ground, 1.0 is a full aerial delivery
    loft: float = 0.0
    #: initial vertical velocity for a lofted delivery, m/s
    vertical: float = 0.0

    @property
    def nominal(self) -> float:
        return (self.min_speed + self.max_speed) / 2.0

    def __str__(self) -> str:
        return (f"{self.name}: {self.min_speed:.0f}-{self.max_speed:.0f} m/s"
                + (f", vz {self.vertical:.1f} m/s" if self.vertical else ""))


#: §9's list, calibrated. These are the ranges that make a switch feel like a
#: switch and a short pass feel like a short pass.
PASS_PROFILES = {
    "short":       PassProfile("short pass", 8.0, 14.0),
    "progressive": PassProfile("progressive pass", 12.0, 18.0),
    "through":     PassProfile("through ball", 18.0, 24.0),
    "long":        PassProfile("long pass", 16.0, 22.0),
    "switch":      PassProfile("switch of play", 20.0, 26.0),
    "cross":       PassProfile("cross", 14.0, 20.0, loft=1.0, vertical=5.5),
    "chip":        PassProfile("chip", 6.0, 10.0, loft=1.0, vertical=7.5),
    "clearance":   PassProfile("clearance", 18.0, 25.0, loft=0.35, vertical=3.0),
    "shot":        PassProfile("shot", 20.0, 32.0),
    "throw":       PassProfile("throw-in", 6.0, 9.0, loft=0.4, vertical=2.4),
}
DEFAULT_PASS = "short"


def profile_for(kind: str) -> PassProfile:
    return PASS_PROFILES.get(kind, PASS_PROFILES[DEFAULT_PASS])


# ─────────────────────────────────────────────
# GROUND BALL
# ─────────────────────────────────────────────

def ground_travel_time(distance_m: float, speed_mps: float,
                       drag: float = BALL_DRAG_COEFFICIENT) -> float:
    """Seconds for a rolling ball to cover ``distance_m`` at ``speed_mps``.

    Solves ``s(t) = (v0/k)(1 - e^(-k t))`` for ``t``. A zero-drag ball is the
    ``k -> 0`` limit, i.e. plain ``distance / speed``, which is what the brief
    writes as the simple form; keeping drag in the default means the simple
    form is the approximation, not the truth.
    """
    if distance_m < 0:
        raise ValueError(f"distance must be non-negative, got {distance_m}")
    if distance_m == 0.0:
        return 0.0
    if speed_mps <= 0.0:
        raise ValueError(f"speed must be positive, got {speed_mps}")
    if drag <= 1e-9:
        return distance_m / speed_mps

    v0 = min(speed_mps, MAX_BALL_SPEED)
    # The ball must still be able to move: if drag would arrest it within the
    # distance, the pass never arrives.
    if drag * distance_m >= v0:
        return math.inf
    return -math.log(1.0 - drag * distance_m / v0) / drag


def ground_speed_at(t: float, speed_mps: float,
                    drag: float = BALL_DRAG_COEFFICIENT) -> float:
    """Speed of a ground ball ``t`` seconds after launch."""
    return min(speed_mps, MAX_BALL_SPEED) * math.exp(-drag * t)


# ─────────────────────────────────────────────
# VERTICAL MOTION (§11)
# ─────────────────────────────────────────────

def z_at(t: float, z0: float = 0.0, vz: float = 0.0,
         gravity: float = GRAVITY) -> float:
    """Height of a ball ``t`` seconds after launch: ``z0 + vz t - 0.5 g t^2``."""
    return z0 + vz * t - 0.5 * gravity * t * t


def loft_time(vz: float, gravity: float = GRAVITY) -> float:
    """Seconds a ball launched upward at ``vz`` stays airborne.

    Solves ``z(t) = 0`` for the positive root, giving ``2 vz / g``.
    """
    if vz <= 0.0:
        return 0.0
    return 2.0 * vz / gravity


def apex_height(vz: float, gravity: float = GRAVITY) -> float:
    """Peak height of a lofted ball: ``vz^2 / 2g``."""
    if vz <= 0.0:
        return 0.0
    return (vz * vz) / (2.0 * gravity)


def height_at(t: float, launch_height: float, apex: float,
              flight_time: float) -> float:
    """Height profile of a ball that leaves at ``launch_height``, peaks at
    ``apex`` and lands at ``launch_height`` again after ``flight_time``.

    A parabola through three known points. Using the symmetric ballistic shape
    means a cross has a believable arc without solving for a launch angle,
    which keeps the model explainable (§24).
    """
    if flight_time <= 0.0:
        return launch_height
    frac = t / flight_time
    return launch_height + 4.0 * (apex - launch_height) * frac * (1.0 - frac)


# ─────────────────────────────────────────────
# TRAJECTORY
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Trajectory:
    """A planned ball flight: where it goes, how fast, and when it lands."""

    kind: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    launch_speed: float
    launch_time: float
    travel_time: float
    #: peak height in metres; 0.0 for a ground ball
    apex: float = 0.0
    drag: float = BALL_DRAG_COEFFICIENT
    launch_height: float = 0.0

    @property
    def distance(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def arrival_time(self) -> float:
        return self.launch_time + self.travel_time

    @property
    def is_aerial(self) -> bool:
        return self.apex > 1e-9

    @property
    def flight_time(self) -> float:
        """Total time aloft; equals travel_time for a ground ball."""
        return self.travel_time

    @property
    def average_speed(self) -> float:
        if self.travel_time <= 0.0:
            return 0.0
        return self.distance / self.travel_time

    def height_at(self, t: float) -> float:
        """Ball height ``t`` seconds after launch."""
        if not self.is_aerial:
            return 0.0
        return height_at(t, self.launch_height, self.apex, self.travel_time)

    def position_at(self, t: float) -> Tuple[float, float, float]:
        """``(x, y, z)`` ``t`` seconds after launch.

        Horizontal motion is a straight line from start to end — a real pass is
        very nearly straight — while ``z`` follows the arc. That is the
        lightweight 3D §11 asks for and no more.
        """
        if self.travel_time <= 0.0:
            return (self.start[0], self.start[1], self.launch_height)
        frac = max(0.0, min(1.0, t / self.travel_time))
        x = self.start[0] + (self.end[0] - self.start[0]) * frac
        y = self.start[1] + (self.end[1] - self.start[1]) * frac
        return (x, y, self.height_at(t))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "start": [round(v, 3) for v in self.start],
            "end": [round(v, 3) for v in self.end],
            "distance": round(self.distance, 3),
            "launch_speed": round(self.launch_speed, 3),
            "launch_time": round(self.launch_time, 4),
            "travel_time": round(self.travel_time, 4),
            "arrival_time": round(self.arrival_time, 4),
            "apex": round(self.apex, 3),
            "aerial": self.is_aerial,
        }


@dataclass
class BallState:
    """A ball in flight. Carries the one extra number the brief asks for."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    @property
    def speed_horizontal(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def speed_3d(self) -> float:
        return math.hypot(self.vx, self.vy, self.vz)

    def to_dict(self) -> dict:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3),
                "vx": round(self.vx, 3), "vy": round(self.vy, 3),
                "vz": round(self.vz, 3)}


class BallPhysics:
    """Plans ball flights. One instance per match; stateless beyond config."""

    def __init__(self, drag: float = BALL_DRAG_COEFFICIENT,
                 gravity: float = GRAVITY) -> None:
        self.drag = drag
        self.gravity = gravity

    # -- planning --------------------------------------------------------
    def plan(self, start: Tuple[float, float], end: Tuple[float, float],
             kind: str = DEFAULT_PASS, launch_time: float = 0.0,
             speed: Optional[float] = None,
             launch_height: float = 0.0) -> Trajectory:
        """Plan a delivery and return its :class:`Trajectory`.

        ``speed`` defaults to the midpoint of the pass type's calibrated range,
        so a through ball is genuinely quicker than a short pass without
        anyone having to supply a number. Supply one to model a particular
        player striking it well or poorly.
        """
        prof = profile_for(kind)
        v0 = float(prof.nominal if speed is None else speed)
        v0 = max(0.5, min(v0, MAX_BALL_SPEED))

        horizontal = math.hypot(end[0] - start[0], end[1] - start[1])

        if prof.vertical > 0.0:
            # Aerial: flight time is set by the launch angle, and the horizontal
            # speed has to cover the distance within it.
            flight = loft_time(prof.vertical, self.gravity)
            if flight <= 0.0 or horizontal <= 0.0:
                travel = ground_travel_time(horizontal, v0, self.drag)
                apex = 0.0
            else:
                travel = flight
                apex = apex_height(prof.vertical, self.gravity)
        else:
            travel = ground_travel_time(horizontal, v0, self.drag)
            apex = 0.0

        return Trajectory(
            kind=kind, start=tuple(start), end=tuple(end),  # type: ignore[arg-type]
            launch_speed=v0, launch_time=float(launch_time),
            travel_time=travel, apex=apex, drag=self.drag,
            launch_height=launch_height,
        )

    def plan_ground(self, start, end, launch_time: float = 0.0,
                    speed: Optional[float] = None) -> Trajectory:
        return self.plan(start, end, "short", launch_time, speed)

    def plan_through_ball(self, start, end, launch_time: float = 0.0,
                          speed: Optional[float] = None) -> Trajectory:
        return self.plan(start, end, "through", launch_time, speed)

    def plan_cross(self, start, end, launch_time: float = 0.0,
                   speed: Optional[float] = None) -> Trajectory:
        return self.plan(start, end, "cross", launch_time, speed)

    def plan_shot(self, start, end, launch_time: float = 0.0,
                  speed: Optional[float] = None) -> Trajectory:
        return self.plan(start, end, "shot", launch_time, speed)

    # -- flight ---------------------------------------------------------
    def launch_state(self, traj: Trajectory) -> BallState:
        """The :class:`BallState` at the instant a trajectory is struck."""
        horizontal = traj.distance
        if traj.travel_time <= 0.0:
            return BallState(traj.start[0], traj.start[1], traj.launch_height)
        vh = horizontal / traj.travel_time
        if horizontal <= 1e-9:
            return BallState(traj.start[0], traj.start[1], traj.launch_height,
                             0.0, 0.0, 0.0)
        ux = (traj.end[0] - traj.start[0]) / horizontal
        uy = (traj.end[1] - traj.start[1]) / horizontal
        # Vertical launch velocity implied by the planned arc: for a parabola
        # peaking at `apex` over `travel_time`, vz = 4 * (apex - z0) / t.
        vz = 0.0
        if traj.apex > 0.0:
            vz = 4.0 * (traj.apex - traj.launch_height) / traj.travel_time
        return BallState(traj.start[0], traj.start[1], traj.launch_height,
                         vh * ux, vh * uy, vz)

    def state_at(self, traj: Trajectory, t: float) -> BallState:
        """Ball state ``t`` seconds after launch."""
        x, y, z = traj.position_at(t)
        launch = self.launch_state(traj)
        elapsed = max(0.0, min(traj.travel_time, t))
        # Horizontal speed bleeds off with the same drag model the flight time
        # was solved with, so speed and timing stay consistent.
        vh = traj.launch_speed * math.exp(-traj.drag * elapsed)
        if traj.distance <= 1e-9:
            return BallState(x, y, z, 0.0, 0.0, launch.vz)
        ux = (traj.end[0] - traj.start[0]) / traj.distance
        uy = (traj.end[1] - traj.start[1]) / traj.distance
        vz = launch.vz - self.gravity * elapsed
        return BallState(x, y, z, vh * ux, vh * uy, vz)

    def describe(self) -> str:
        return (f"BallPhysics(drag={self.drag:.4f}/s, g={self.gravity:.3f} m/s^2)")
