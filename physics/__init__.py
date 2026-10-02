"""
PLOFA PHYSICS — a continuous physical-time layer.
==================================================

``physics/`` answers one question, and the question is narrow on purpose:

    "Given the current physical state of the football world, how long does this
     action actually take?"

What this package is NOT
------------------------
* It is **not** a rendering engine. There is no graphics, no animation, no
  frame loop, and nothing here ever sleeps. Ninety minutes of football is
  calculated, not waited out.
* It is **not** a second source of truth for position. ``PositionEngine``
  remains authoritative for where players are and where tactics want them;
  this layer calculates how long the body takes to get from here to there.
* It is **not** a replacement for the player brain. The brain answers "what
  should I do?"; this layer answers "how does that physically happen?". A
  target striker and a target defender are the same *race* — only the physics
  differ.

Design rules that the whole package obeys
------------------------------------------
1. **Event-driven, not millisecond-looped.** An action is planned, its next
   interesting time is computed, the clock is advanced to exactly that instant
   and the interaction is resolved. Fractional seconds throughout.
2. **The clock is simulated, never measured.** Nothing in this package reads
   ``time.time()``, ``datetime.now()`` or any CPU wall clock. CPU execution
   speed has zero influence on football time.
3. **No teleportation.** Every movement has non-zero physical duration, and
   :class:`~physics.violations.PhysicsViolation` exists so an impossible one
   is reported rather than silently tolerated.
4. **Deterministic.** Same inputs, same timeline. No unseeded randomness, and
   the one place probability is allowed (imperfect execution) takes an explicit
   seeded draw.
5. **Pure.** Nothing in ``physics/`` imports the live PLOFA pipeline. That is
   enforced by test, and it is what lets the physics be calibrated in
   isolation from a season that is currently running.

Integration is additive and opt-in. With the physics world disabled — the
default — the engine behaves exactly as it did, so the live 26/27 season is
unaffected.
"""
from __future__ import annotations

from physics.ball_physics import (
    BALL_DRAG_COEFFICIENT,
    GRAVITY,
    BallPhysics,
    BallState,
    PassProfile,
    Trajectory,
    loft_time,
)
from physics.calibration import (
    Calibration,
    PhysicalProfile,
    acceleration_for,
    calibrate,
    calibrate_from_values,
    fatigue_acceleration_multiplier,
    fatigue_speed_multiplier,
    reaction_for,
    speed_for_pace,
)
from physics.collision import (
    Candidate,
    ContestResult,
    arrival_contest,
    margin_table,
    race_preview,
    resolve_receiver,
)
from physics.continuous_time import (
    ContinuousClock,
    MatchTime,
    match_time_to_seconds,
    seconds_to_match_time,
    stoppage_for,
)
from physics.player_motion import (
    KinematicState,
    PlayerMotion,
    arrival_time,
    integrate,
    run_to_arrival,
)
from physics.violations import (
    PhysicsViolation,
    PhysicsViolationError,
    ViolationLog,
    check_move,
)
from physics.world import (
    PassResolution,
    PhysicsWorld,
    TimelineStep,
)

__all__ = [
    "BALL_DRAG_COEFFICIENT",
    "GRAVITY",
    "BallPhysics",
    "BallState",
    "Calibration",
    "Candidate",
    "ContestResult",
    "ContinuousClock",
    "KinematicState",
    "MatchTime",
    "PassProfile",
    "PassResolution",
    "PhysicalProfile",
    "PhysicsViolation",
    "PhysicsViolationError",
    "PhysicsWorld",
    "PlayerMotion",
    "TimelineStep",
    "Trajectory",
    "ViolationLog",
    "acceleration_for",
    "arrival_contest",
    "arrival_time",
    "calibrate",
    "calibrate_from_values",
    "check_move",
    "fatigue_acceleration_multiplier",
    "fatigue_speed_multiplier",
    "integrate",
    "loft_time",
    "margin_table",
    "match_time_to_seconds",
    "race_preview",
    "reaction_for",
    "resolve_receiver",
    "run_to_arrival",
    "seconds_to_match_time",
    "speed_for_pace",
    "stoppage_for",
]
