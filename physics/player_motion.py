"""
physics/player_motion.py — kinematics, and the arrival time that falls out.
=========================================================================

The engine currently moves players by *proportional steering* toward a tactical
target::

    tx += (trix - tx) * p_tri        # match_engine.py

That is a reasonable way to keep a team in shape, and this module does not
replace it. But it has no notion of velocity, acceleration or duration, so it
cannot answer "how long until he gets there?" — which is the only question a
through ball actually turns on. Line 2715 of the engine says as much in a
comment: *"position jumps we must not re-derive as distance/sprints from the
position gap."*

The model
---------
A player has position and velocity. Moving to a new target happens in three
phases, and each one costs time:

1. **REACTION.** The player does not respond instantly (§8). Velocity is not
   redirected; instead the component of velocity perpendicular to the new
   target has to be killed, which costs time proportional to how wrong the
   current heading is. A player already running at the ball reacts cheaply; a
   player running the wrong way pays for the turn.

2. **ACCELERATION.** Velocity builds toward ``max_speed`` at ``acceleration``
   (§7). Speed is capped — a fast player does not reach top speed
   instantaneously, which is exactly what makes an underestimating defender
   beat a faster one who is badly positioned.

3. **CRUISE.** Any remaining distance is covered at ``max_speed``.

Arrival time is then the closed-form sum, so the simulation advances the clock
directly to the event instead of stepping toward it. That is what makes this
package event-driven rather than a millisecond loop (§2).

Every quantity here is deterministic. There is no randomness in this module at
all, by design: a physics result must be a function of state, and execution
probability belongs to the brain, not the body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

#: A player "arrives" when within this many metres of the target. A footballer's
#: reach is not zero: stride length at speed is a metre or more.
REACH_M = 0.8

#: Below this speed a player is treated as stationary. Keeps the turning and
#: acceleration algebra well-conditioned at rest.
REST_EPSILON = 1e-6

#: Hard ceiling on any speed the model will produce (m/s). §15: impossible
#: movement must be impossible, not merely unlikely. 12 m/s is faster than any
#: human can run; reaching it is a bug, and this clamp turns that bug into a
#: detectable one.
ABSOLUTE_SPEED_CEILING = 12.0


@dataclass(frozen=True)
class KinematicState:
    """Where a player is and which way they are already moving."""

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def heading(self) -> float:
        """Direction of travel in radians, or 0.0 when stationary."""
        if self.speed < REST_EPSILON:
            return 0.0
        return math.atan2(self.vy, self.vx)

    def advanced(self, seconds: float) -> "KinematicState":
        """Constant-velocity extrapolation. Used for prediction only."""
        return KinematicState(
            self.x + self.vx * seconds,
            self.y + self.vy * seconds,
            self.vx, self.vy,
        )

    def to_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4),
                "vx": round(self.vx, 4), "vy": round(self.vy, 4),
                "speed": round(self.speed, 4)}


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def unit_towards(frm: Tuple[float, float],
                 to: Tuple[float, float]) -> Tuple[float, float]:
    """Unit vector from ``frm`` to ``to``; (1, 0) if they coincide."""
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    d = math.hypot(dx, dy)
    if d < REST_EPSILON:
        return (1.0, 0.0)
    return (dx / d, dy / d)


def _turn_time(speed: float, profile, desired: Tuple[float, float],
               state: KinematicState) -> Tuple[float, float, float]:
    """Time to kill the velocity component perpendicular to ``desired``.

    Returns ``(seconds, along_speed, remaining_perpendicular)``. Splitting the
    velocity this way is the standard way to model a course correction: the
    component already pointing the right way is kept, the wrong one is braked
    away at the deceleration limit.

    ``agility`` scales how quickly a player can change direction, so a nimble
    player pays less of this cost than a stiff one turning the same amount.
    """
    v_along = state.vx * desired[0] + state.vy * desired[1]
    perp_x = state.vx - v_along * desired[0]
    perp_y = state.vy - v_along * desired[1]
    v_perp = math.hypot(perp_x, perp_y)

    if v_perp < REST_EPSILON:
        return 0.0, v_along, 0.0

    # Agility 0..1 -> deceleration available for turning is between 1x and 2x
    # the straight-line figure. A player who cannot change direction is slower
    # in every direction, which is the honest reading of the attribute.
    turn_decel = profile.deceleration * (1.0 + profile.agility)
    t = v_perp / turn_decel
    return t, v_along, v_perp


def arrival_time(
    state: KinematicState,
    target: Tuple[float, float],
    profile,
    *,
    reaction: Optional[float] = None,
    reach: float = REACH_M,
) -> float:
    """Seconds until this player can physically be at ``target``.

    Closed form, so the caller can advance the clock straight to the event
    rather than simulating toward it. Returns 0.0 if already within ``reach``.

    The three phases, and why each is there:

    * **reaction** (§8) — the brief's distinction between what the brain wants
      and what the body can do. Defaults to the profile's calibrated value.
    * **turn** — killing wrong-way velocity, scaled by agility.
    * **accelerate / cruise** — the standard ``v0 + at`` capped at
      ``max_speed`` (§5, §7).
    """
    if profile.max_speed <= 0.0:
        raise ValueError("profile must have a positive max_speed")

    react = profile.reaction_time if reaction is None else float(reaction)
    if react < 0:
        raise ValueError(f"reaction must be non-negative, got {react}")

    d = distance((state.x, state.y), target)
    if d <= reach:
        return 0.0

    desired = unit_towards((state.x, state.y), target)
    t_turn, v_along, _ = _turn_time(state.speed, profile, desired, state)

    # During the turn the player keeps the along-target component but travels
    # slightly off-line. Treating the turn as purely lateral is the standard
    # approximation and keeps the answer honest to within a few centimetres.
    straight = max(0.0, d - reach - v_along * t_turn)

    a = profile.acceleration
    vmax = min(profile.max_speed, ABSOLUTE_SPEED_CEILING)
    if a <= REST_EPSILON:
        # No acceleration available: can only coast. Guarded so this returns a
        # finite number instead of dividing by zero.
        t_accel = 0.0
        t_cruise = straight / vmax if vmax > REST_EPSILON else 0.0
    else:
        v0 = max(v_along, 0.0)
        # Distance needed to reach vmax from v0 under constant acceleration.
        d_to_vmax = (vmax * vmax - v0 * v0) / (2.0 * a)
        if straight <= d_to_vmax:
            # Never reaches vmax before arriving: solve s = v0 t + 0.5 a t^2
            disc = v0 * v0 + 2.0 * a * straight
            t_accel = (math.sqrt(max(0.0, disc)) - v0) / a
            t_cruise = 0.0
        else:
            t_accel = (vmax - v0) / a
            t_cruise = (straight - d_to_vmax) / vmax

    return react + t_turn + t_accel + t_cruise


def integrate(
    state: KinematicState,
    target: Optional[Tuple[float, float]],
    dt: float,
    profile,
    *,
    reach: float = REACH_M,
) -> KinematicState:
    """Advance a player's kinematic state by ``dt`` seconds toward ``target``.

    Used when something needs a sampled path rather than a single arrival time
    — a run-tracker sample, or drawing where a player actually was. When
    ``target`` is ``None`` the player coasts and bleeds off speed through
    ``deceleration``, because a decelerating body is the default and an
    instantaneous stop is a teleport.

    Speed is clamped to :data:`ABSOLUTE_SPEED_CEILING` on the way out, so a
    miscalibrated profile degrades into "suspiciously fast" rather than into an
    impossible position jump that a later consumer would have to detect.
    """
    if dt < 0:
        raise ValueError(f"dt must be non-negative, got {dt}")
    if dt == 0.0:
        return state

    speed = state.speed

    if target is None or distance((state.x, state.y), target) <= reach:
        # Coasting to a stop.
        new_speed = max(0.0, speed - profile.deceleration * dt)
        if speed < REST_EPSILON:
            return KinematicState(state.x, state.y, 0.0, 0.0)
        scale = new_speed / speed
        return KinematicState(state.x + state.vx * dt,
                              state.y + state.vy * dt,
                              state.vx * scale, state.vy * scale)

    desired = unit_towards((state.x, state.y), target)
    v_along = state.vx * desired[0] + state.vy * desired[1]

    # Accelerate the along-target component, brake the perpendicular one.
    #
    # ``v_along`` is SIGNED. A player running *away* from his target has a
    # negative one, and the naive ``v_along + acceleration * dt`` then makes it
    # more negative — after multiplying by ``desired``, that accelerates him
    # directly away from where he is trying to get to. A 180-degree turn at
    # full pace came out as "keep sprinting the old way".
    #
    # Every arrival-time query this module was originally written for starts
    # from rest facing the target, so ``v_along >= 0`` and the sign never came
    # up. It appears the moment the stepper is used for continuous movement,
    # where a player who overshoots his shape socket, or whose target swings
    # behind him, is ordinary rather than exceptional.
    #
    # So: moving away, he BRAKES at the deceleration limit and is never allowed
    # to grow the backwards component. He coasts to a stop, then drives the
    # other way. Which is what a body does.
    if v_along >= 0.0:
        new_along = min(profile.max_speed, v_along + profile.acceleration * dt)
    else:
        new_along = min(0.0, v_along + profile.deceleration * dt)

    perp_x = state.vx - v_along * desired[0]
    perp_y = state.vy - v_along * desired[1]
    perp_speed = math.hypot(perp_x, perp_y)
    turn_decel = profile.deceleration * (1.0 + profile.agility)
    new_perp = max(0.0, perp_speed - turn_decel * dt)

    if perp_speed < REST_EPSILON:
        nvx, nvy = new_along * desired[0], new_along * desired[1]
    else:
        scale = new_perp / perp_speed
        nvx = new_along * desired[0] + perp_x * scale
        nvy = new_along * desired[1] + perp_y * scale

    final_speed = math.hypot(nvx, nvy)
    if final_speed > ABSOLUTE_SPEED_CEILING:
        scale = ABSOLUTE_SPEED_CEILING / final_speed
        nvx, nvy = nvx * scale, nvy * scale

    return KinematicState(state.x + nvx * dt, state.y + nvy * dt, nvx, nvy)


def run_to_arrival(
    state: KinematicState,
    target: Tuple[float, float],
    profile,
    *,
    reaction: Optional[float] = None,
    reach: float = REACH_M,
    max_dt: float = 0.05,
) -> Tuple[KinematicState, float]:
    """Simulate all the way to arrival. Returns ``(final_state, seconds)``.

    Provided for callers that need the path, not just the answer. The engine's
    own loop uses :func:`arrival_time` and jumps the clock instead, because
    stepping at 20 ms for a 90-minute match is exactly the millisecond loop
    §2 warns against.

    The returned elapsed time is reconciled against :func:`arrival_time` so a
    caller can see how much the sampled path differs from the closed form.
    """
    t = arrival_time(state, target, profile, reaction=reaction, reach=reach)
    if t <= 0.0:
        return state, 0.0

    elapsed = 0.0
    cur = state
    guard = 0
    # A generous cap: nobody needs more than 30 simulated seconds to cross a
    # pitch, and the guard turns a non-converging integrator into an error
    # rather than a hang.
    while elapsed < t + max_dt and guard < 20000:
        step = min(max_dt, t - elapsed)
        if step <= 0:
            break
        cur = integrate(cur, target, step, profile, reach=reach)
        elapsed += step
        guard += 1
        if distance((cur.x, cur.y), target) <= reach:
            break
    return cur, elapsed


@dataclass(frozen=True)
class PlayerMotion:
    """A player plus their kinematic state: what the collision code needs."""

    profile: object
    state: KinematicState

    @property
    def name(self) -> str:
        return getattr(self.profile, "name", "")

    def arrival(self, target: Tuple[float, float], **kw) -> float:
        return arrival_time(self.state, target, self.profile, **kw)

    def step(self, dt: float,
             target: Optional[Tuple[float, float]] = None) -> "PlayerMotion":
        return replace(self, state=integrate(
            self.state, target, dt, self.profile))
