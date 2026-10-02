"""
physics/calibration.py — DNA in, physical units out.
=====================================================

This is the layer the brief is most insistent on, and the one most likely to
be fudged:

    "Do NOT simply make ``pace = movement speed`` unless the existing DNA
     explicitly defines it that way. Create a clear conversion/calibration
     layer."

PLOFA's ``PhysicalAttributes`` are 0-100 opinion scores::

    pace:         float = 60.0   # Raw sprint speed
    acceleration: float = 60.0   # 0 -> top speed quickness
    agility:      float = 60.0   # Change of direction
    stamina:      float = 65.0   # Endurance across 90 min

A score of 60 is not 60 m/s, and treating it as a speed would make every
player superhuman. Worse, a *linear* pass-through would make pace the only
thing that matters — the brief's §7 warning that "a slower player with better
positioning should sometimes reach the ball first" needs pace to be one input
among several.

So: a score becomes a physical quantity through a documented, bounded curve,
and the curves are anchored to real football numbers.

The anchors (§6's reference ranges, treated as calibration targets)
--------------------------------------------------------------------
======================  =========
walking                 1 - 2 m/s
jogging                 2 - 4 m/s
running                 4 - 6 m/s
high-speed running      5.5 - 7 m/s
maximum sprint          7 - 10 m/s
======================  =========

The mapping chosen here puts the DNA *default* (``pace = 60``) at **7.5 m/s**,
which is a realistic top-flight sprint, and spans 4.5 - 9.5 m/s across the full
0-100 score range. Those endpoints are deliberate: a 0-pace player is slow
rather than immobile, and a 100-pace player is an elite sprinter rather than a
cheetah.

Acceleration is anchored to the standard sprint splits rather than guessed. A
top-flight player covers roughly 5 m in 1.4 s, 10 m in 2.0 s, 20 m in 3.4 s and
reaches top speed around 4-5 s. That implies an average acceleration near
1.9 m/s^2 for a ``pace = 60`` player, which is what the curve produces.

Fatigue is deliberately NOT a flat speed tax (§14 forbids the "100 stamina =
fast, 50 stamina = slow" step). A tired player keeps most of his top speed but
loses acceleration, which is what actually happens: you can still sprint at the
end of a match, you just cannot get there as quickly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─────────────────────────────────────────────
# ANCHORS
# ─────────────────────────────────────────────

#: Score 0 -> this top speed (m/s). A poor sprinter, still a footballer.
MIN_SPRINT_MPS = 4.5
#: Score 100 -> this top speed (m/s). Elite, not superhuman.
MAX_SPRINT_MPS = 9.5

#: Score 0 -> this acceleration (m/s^2). Sluggish off the mark.
MIN_ACCEL_MPS2 = 0.6
#: Score 100 -> this acceleration (m/s^2). Explosive.
MAX_ACCEL_MPS2 = 2.8

#: Deceleration is a multiple of acceleration. Higher, because braking is
#: easier than accelerating in every sport that has been measured.
DECEL_FACTOR = 1.4

#: Reaction time at anticipation score 0 and 100 (seconds). Elite anticipation
#: is around 0.2 s; a slow reader is nearer 0.4 s.
SLOW_REACTION_S = 0.42
FAST_REACTION_S = 0.20

#: PITCH dimensions, for sanity checks in tests.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 100.0:
        return 100.0
    return value


def speed_for_pace(pace: float) -> float:
    """PLOFA pace score (0-100) -> top speed in m/s.

    Linear between two anchored endpoints, so the relationship stays
    explainable: a one-point pace gain is worth a fixed amount of speed, and
    there is no hidden curve to reverse-engineer from a fixture list.
    """
    return MIN_SPRINT_MPS + (_clamp01(pace) / 100.0) * (
        MAX_SPRINT_MPS - MIN_SPRINT_MPS)


def acceleration_for(acceleration: float) -> float:
    """PLOFA acceleration score (0-100) -> m/s^2."""
    return MIN_ACCEL_MPS2 + (_clamp01(acceleration) / 100.0) * (
        MAX_ACCEL_MPS2 - MIN_ACCEL_MPS2)


def reaction_for(anticipation: float) -> float:
    """PLOFA anticipation score (0-100) -> reaction delay in seconds."""
    return SLOW_REACTION_S + (_clamp01(anticipation) / 100.0) * (
        FAST_REACTION_S - SLOW_REACTION_S)


# ─────────────────────────────────────────────
# FATIGUE — smooth, and weighted towards acceleration
# ─────────────────────────────────────────────

#: Tired players keep most of their top speed...
FATIGUE_SPEED_FLOOR = 0.92
#: ...but lose a lot of acceleration. This asymmetry is deliberate and is the
#: opposite of a flat multiplier on both.
FATIGUE_ACCEL_FLOOR = 0.70


def fatigue_speed_multiplier(stamina: float) -> float:
    """Stamina 0-100 -> multiplier on top speed. Smooth, never zero."""
    return FATIGUE_SPEED_FLOOR + 0.08 * (_clamp01(stamina) / 100.0)


def fatigue_acceleration_multiplier(stamina: float) -> float:
    """Stamina 0-100 -> multiplier on acceleration. Harsher than speed."""
    return FATIGUE_ACCEL_FLOOR + 0.30 * (_clamp01(stamina) / 100.0)


# ─────────────────────────────────────────────
# THE PROFILE
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PhysicalProfile:
    """One player's physical capabilities, in SI units.

    Produced by :func:`calibrate` from a PLOFA ``PlayerDNA`` and the current
    stamina reading. Everything downstream in ``physics/`` consumes this and
    never touches a DNA score directly, which is what keeps the conversion in
    exactly one place.
    """

    name: str = ""
    position: str = ""
    #: top speed, m/s, after fatigue
    max_speed: float = 7.5
    #: acceleration, m/s^2, after fatigue
    acceleration: float = 1.92
    #: deceleration, m/s^2 (a multiple of acceleration)
    deceleration: float = 2.7
    #: base reaction delay, seconds, before situational adjustment
    reaction_time: float = 0.29
    #: 0-100, the stamina reading this profile was built at
    stamina: float = 100.0
    #: untaxed values, kept so a caller can reason about the tax itself
    raw_max_speed: float = 7.5
    raw_acceleration: float = 1.92
    #: how sharply this player turns. Derived from agility but capped so it
    #: can never make a slow player fast.
    agility: float = 0.6

    def with_stamina(self, stamina: float) -> "PhysicalProfile":
        """Return the same player at a different stamina reading.

        Immutability is deliberate: a profile is a snapshot of a player at an
        instant, and the physics layer must not hold a mutable object whose
        values change underfoot mid-trajectory.

        Constructed directly rather than by inverting the raw SI values back
        into 0-100 scores and re-calibrating. That round trip happened to be
        lossless, but it is fragile with no payoff, and the pre-fatigue maxima
        are already carried on the profile.
        """
        speed_mult = fatigue_speed_multiplier(stamina)
        accel_mult = fatigue_acceleration_multiplier(stamina)
        return PhysicalProfile(
            name=self.name,
            position=self.position,
            max_speed=self.raw_max_speed * speed_mult,
            acceleration=self.raw_acceleration * accel_mult,
            deceleration=self.raw_acceleration * accel_mult * DECEL_FACTOR,
            reaction_time=self.reaction_time,
            stamina=_clamp01(stamina),
            raw_max_speed=self.raw_max_speed,
            raw_acceleration=self.raw_acceleration,
            agility=self.agility,
        )

@dataclass
class Calibration:
    """Optional global adjustments applied on top of every profile.

    This is where a caller injects factors it already owns — notably weather,
    which PLOFA's ``weather_physics`` already models with ``rolling_decel_mult``
    and ``pitch_grip``. Keeping the hook here means the physics package stays
    free of weather imports while still honouring the conditions.
    """

    speed_multiplier: float = 1.0
    acceleration_multiplier: float = 1.0
    reaction_multiplier: float = 1.0
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        return (f"speed x{self.speed_multiplier:.3f}  "
                f"accel x{self.acceleration_multiplier:.3f}  "
                f"reaction x{self.reaction_multiplier:.3f}")


def calibrate_from_values(
    *,
    name: str = "",
    position: str = "",
    pace_score: float = 60.0,
    acceleration_score: float = 60.0,
    anticipation_score: Optional[float] = None,
    reaction_time: Optional[float] = None,
    agility: float = 60.0,
    stamina: float = 100.0,
    calibration: Optional[Calibration] = None,
) -> PhysicalProfile:
    """Build a :class:`PhysicalProfile` from raw 0-100 scores.

    This is the function the tests exercise directly, because it has no
    dependency on a live ``PlayerDNA`` — :func:`calibrate` is the thin adapter
    that pulls scores out of one.
    """
    cal = calibration or Calibration()

    raw_speed = speed_for_pace(pace_score) * cal.speed_multiplier
    raw_accel = acceleration_for(acceleration_score) * cal.acceleration_multiplier

    if reaction_time is None:
        base_reaction = reaction_for(
            60.0 if anticipation_score is None else anticipation_score)
    else:
        base_reaction = float(reaction_time)
    base_reaction *= cal.reaction_multiplier

    fatigue_speed = fatigue_speed_multiplier(stamina)
    fatigue_accel = fatigue_acceleration_multiplier(stamina)

    return PhysicalProfile(
        name=name,
        position=position,
        max_speed=raw_speed * fatigue_speed,
        acceleration=raw_accel * fatigue_accel,
        deceleration=raw_accel * fatigue_accel * DECEL_FACTOR,
        reaction_time=base_reaction,
        stamina=stamina,
        raw_max_speed=raw_speed,
        raw_acceleration=raw_accel,
        agility=_clamp01(agility) / 100.0,
    )


def calibrate(dna, stamina: float = 100.0,
              calibration: Optional[Calibration] = None) -> PhysicalProfile:
    """Adapt a live PLOFA ``PlayerDNA`` into a :class:`PhysicalProfile`.

    Reads only what the brief names: ``physical.pace``, ``physical.acceleration``,
    ``physical.agility`` and ``mental.anticipation``. Any missing domain falls
    back to the DNA defaults rather than raising, so an incomplete player object
    degrades to average instead of failing a match.
    """
    phys = getattr(dna, "physical", None)
    mental = getattr(dna, "mental", None)
    return calibrate_from_values(
        name=getattr(dna, "name", "") or "",
        position=getattr(dna, "position", "") or "",
        pace_score=getattr(phys, "pace", 60.0) if phys else 60.0,
        acceleration_score=(getattr(phys, "acceleration", 60.0)
                            if phys else 60.0),
        anticipation_score=(getattr(mental, "anticipation", 60.0)
                            if mental else 60.0),
        agility=(getattr(phys, "agility", 60.0) if phys else 60.0),
        stamina=stamina,
        calibration=calibration,
    )
