"""
Physical player movement: momentum, fatigue in the legs, and reaction delay.
===========================================================================

What the engine already had
---------------------------
The off-ball step is **not** naive proportional steering, and it is worth being
precise about that, because an earlier version of this work described it as
``tx += (trix - tx) * p_tri`` and implied the movement was unphysical. That
formula only steers the *target*. The step that follows is already respectable::

    speed = jog + (top_speed * CHASE_EFFORT - jog) * chase_ramp
    step  = min(speed * DT, dist_to_target)
    nx    = cx + (dx / dist) * step

It is time-stepped, pace-capped by position, ramps when a chase engages, and
records a real speed. What it does **not** have:

  1. **Momentum.** Direction is recomputed as "toward target" every tick, so a
     player at full pace can reverse instantaneously. Real players carry
     momentum through a turn and cannot.
  2. **An acceleration limit on direction change.** The chase ramp limits how
     fast *speed* grows, but nothing limits how fast the *heading* swings.
  3. **Stamina in the legs.** ``_top_speed_cache[p] = 5.0 + pace * 0.042`` is
     pure DNA. A player on 20% stamina runs at the same top speed as one on
     100% — the number is identical, so the legs never feel the match.
  4. **Reaction delay.** The target changes, and the player is already moving
     toward the new one on the same tick. Real perception costs ~0.2 s, and it
     is most of why pressing arrangements get broken.

(1), (2) and (3) are one thing: *step the body kinematically instead of
teleporting a scalar speed along a straight line.* :func:`physics.player_motion.integrate`
already does that — it carries velocity, accelerates the along-target
component, brakes the perpendicular one, and coasts to a stop rather than
teleporting. This module is the thin layer that decides *when* to step and
*what to step toward*.

(4) is the reaction gate below, and it is the one genuinely new piece.

Why the reaction gate holds the OLD target
------------------------------------------
A reaction delay modelled as "stand still for 0.2 s" is wrong twice over: it
wastes the player's legs and it looks like a statue. What actually happens is
that the player continues toward where he *thought* the stimulus was, and only
then corrects. So the gate holds the previously-held target until the reaction
timer expires, and the body keeps moving under momentum the whole time. The
error is in the player's information, not in his legs.

The gate only arms on a *material* target change. Shape targets drift by
centimetres every tick, and arming on those would freeze the whole team in
perpetual reaction, which is both wrong and slower than the steering it
replaces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from physics.player_motion import KinematicState, integrate

#: How far a shape target must move before the player is treated as having
#: noticed. Shape targets drift centimetres per tick; arming on that would
#: mean every outfield player spent the match in reaction.
REACTION_TRIGGER_M = 2.0

#: Outfield reaction time, seconds. Deliberately not a per-player constant at
#: this layer: PLOFA's DNA has no reaction trait, and inventing one would be a
#: new calibrated parameter pretending to be a measurement. A flat 0.20 s is
#: the accepted human figure and is honest about being a constant.
OUTFIELD_REACTION_S = 0.20


@dataclass
class MovementMemory:
    """What the body remembers between ticks: momentum, and what it last saw.

    One of these per player, owned by the adapter and dropped on substitution
    for the same reason a cached velocity is: across a substitution the finite
    difference is a teleport, and a teleport that persists is a player who
    appears to sprint 100 m between frames.
    """

    vx: float = 0.0
    vy: float = 0.0
    #: The target the player is currently *responding to*. Held steady while a
    #: reaction is pending, which is what makes the delay legible.
    target: Optional[Tuple[float, float]] = None
    #: Seconds of reaction still owed on the current target.
    reaction_left: float = 0.0
    #: Ticks spent in the current state — used only for reporting.
    ticks: int = 0

    def reset(self) -> None:
        self.vx = self.vy = 0.0
        self.target = None
        self.reaction_left = 0.0
        self.ticks = 0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def reacting(self) -> bool:
        return self.reaction_left > 0.0


def perceive(
    memory: MovementMemory,
    target: Optional[Tuple[float, float]],
    dt: float,
    *,
    reaction_s: float = OUTFIELD_REACTION_S,
    trigger_m: float = REACTION_TRIGGER_M,
) -> Optional[Tuple[float, float]]:
    """Advance perception one tick. Returns the target the body should move to.

    Perception and locomotion are on different clocks — a player notices
    something once, then runs for several ticks on that belief — so the
    countdown lives here rather than in :func:`step`.

    **The countdown has to live in exactly one place.** An earlier version put
    it in ``step`` and had ``step`` adopt the target on expiry. But ``step`` is
    handed the *already-stale* believed target, so on expiry it re-stored the
    old belief, and the next ``perceive`` saw a large difference and re-armed
    the reaction — forever. The player would have kept running to the first
    position he ever saw. Countdown and adoption happen together here, so the
    state cannot desynchronise.
    """
    if target is None:
        return None

    if memory.target is None:
        # First tick, or a fresh player: no previous belief, so there is
        # nothing to react *from*. He moves immediately — a reaction delay on
        # a player's very first action would be an artefact of the bookkeeping.
        memory.target = target
        return target

    if memory.reaction_left > 0.0:
        # Still digesting. Run on toward where he already thought the stimulus
        # was. The error is in his information, not in his legs.
        memory.reaction_left = max(0.0, memory.reaction_left - dt)
        if memory.reaction_left == 0.0:
            memory.target = target          # adopt, in the same tick
        return memory.target

    moved = math.hypot(target[0] - memory.target[0],
                       target[1] - memory.target[1])
    if moved >= trigger_m:
        memory.reaction_left = reaction_s
        return memory.target                # noticed, but has not corrected yet
    memory.target = target
    return target


def step(
    memory: MovementMemory,
    position: Tuple[float, float],
    target: Optional[Tuple[float, float]],
    profile,
    dt: float,
    *,
    reach: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Advance the body one tick. Returns ``(x, y, distance_moved)``.

    Purely kinematic: momentum is carried across calls through ``memory``, so
    this is a true integration rather than a per-tick speed assignment. All
    decision-making about *which* target is in :func:`perceive`. ``dt`` must be
    the real simulated step; the caller owns the clock.
    """
    if dt <= 0.0:
        return position[0], position[1], 0.0

    state = KinematicState(position[0], position[1], memory.vx, memory.vy)
    kwargs = {"reach": reach} if reach is not None else {}
    new = integrate(state, target, dt, profile, **kwargs)

    memory.vx, memory.vy = new.vx, new.vy
    memory.ticks += 1
    moved = math.hypot(new.x - position[0], new.y - position[1])
    return new.x, new.y, moved
