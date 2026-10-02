"""
physics/continuous_time.py — the simulated match clock.
======================================================

Why this exists when ``MatchState.match_clock_s`` already exists
--------------------------------------------------------------
It does, and it is the engine's authoritative clock. This module is NOT a
replacement for it. It exists because the physics layer needs three things
``match_clock_s`` does not offer:

* **Fractional arithmetic with explicit units.** A clock that is "a float that
  happens to be seconds" invites off-by-60 mistakes. :class:`ContinuousClock`
  is explicit and converts both ways.
* **Play time versus match time.** A goal consumes stoppage time. §19 requires
  that distinction be maintained, and folding it into a single float loses it.
  :class:`ContinuousClock` tracks both: the *play* clock that physics runs on
  and the *match* clock the scoreboard shows.
* **A testable guarantee that CPU time is irrelevant.** :meth:`advance` is the
  only way time moves, it takes an explicit delta, and nothing in this module
  can read a wall clock. A test can therefore assert that two runs which took
  different amounts of real time produce identical simulated timestamps.

The single rule
---------------
Time only ever moves because someone called :meth:`ContinuousClock.advance`
with a simulated delta. There is no ``time.sleep`` here, no ``datetime.now``,
and no import of ``time`` at all — which is asserted by test, because the
cheapest way to break the whole premise of this package is to reach for the
wall clock once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

#: A 90-minute match is 5,400 play-seconds; stoppage can add more. The cap is a
#: sanity bound, not a limit the simulation is expected to reach.
MAX_MATCH_SECONDS = 200 * 60.0


class MatchTime(NamedTuple):
    """A fractional match time, split for display without losing precision.

    ``MatchTime(2241, 21.735)`` is 37:21.735. ``second`` is a float on purpose:
    the engine's own ``TimelineEntry`` already carries ``second: float``, so
    this matches what downstream consumers expect.
    """

    minute: int
    second: float

    @property
    def total_seconds(self) -> float:
        return self.minute * 60.0 + self.second

    def __str__(self) -> str:
        # Millisecond precision, and correct for seconds >= 10: a naive
        # f"{second:06.3f}" renders 21.0 as "021.000" and produced timestamps
        # like "37:2100.000".
        whole = int(self.second)
        millis = int(round((self.second - whole) * 1000.0))
        if millis >= 1000:            # rounding carried, e.g. 21.9999
            whole += 1
            millis -= 1000
        return f"{self.minute}:{whole:02d}.{millis:03d}"


def seconds_to_match_time(seconds: float) -> MatchTime:
    """2241.735 -> MatchTime(minute=37, second=21.735)."""
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    minute = int(seconds // 60)
    return MatchTime(minute, seconds - minute * 60.0)


def match_time_to_seconds(minute: int, second: float = 0.0) -> float:
    """(37, 21.735) -> 2241.735."""
    if minute < 0:
        raise ValueError(f"minute must be non-negative, got {minute}")
    if not 0.0 <= second < 60.0:
        raise ValueError(f"second must be in [0, 60), got {second}")
    return minute * 60.0 + second


@dataclass
class ContinuousClock:
    """Fractional simulated time, with stoppage tracked separately.

    Two counters, deliberately:

    * ``play_seconds`` — time the ball is in play. Physics runs on this, so a
      goal celebration does not let a striker "run" during the applause.
    * ``match_seconds`` — what the scoreboard shows. Equals play time plus
      whatever stoppage has been added.

    Every mutation goes through :meth:`advance`, which is what makes
    "CPU execution time has zero influence" a structural property rather than
    a promise.
    """

    play_seconds: float = 0.0
    stoppage_seconds: float = 0.0

    # -- reading ---------------------------------------------------------
    @property
    def match_seconds(self) -> float:
        """Scoreboard time: play time plus accumulated stoppage."""
        return self.play_seconds + self.stoppage_seconds

    @property
    def match_time(self) -> MatchTime:
        return seconds_to_match_time(self.match_seconds)

    @property
    def play_time(self) -> MatchTime:
        return seconds_to_match_time(self.play_seconds)

    def __str__(self) -> str:
        return (f"play {self.play_time} | match {self.match_time} "
                f"| stoppage {self.stoppage_seconds:.1f}s")

    # -- moving ----------------------------------------------------------
    def advance(self, delta_seconds: float) -> float:
        """Move PLAY time forward by a simulated delta. Returns the new value.

        Rejects a negative delta: time does not run backwards inside a match,
        and silently allowing it would let a physics bug un-fire an event.
        """
        if delta_seconds < 0:
            raise ValueError(
                f"cannot advance the clock backwards by {delta_seconds}s; "
                f"a negative delta means an arrival time was computed wrong")
        if self.match_seconds + delta_seconds > MAX_MATCH_SECONDS:
            raise ValueError(
                f"match would run past {MAX_MATCH_SECONDS}s "
                f"({MAX_MATCH_SECONDS / 60:.0f} minutes)")
        self.play_seconds += delta_seconds
        return self.play_seconds

    def add_stoppage(self, seconds: float) -> float:
        """Charge stoppage — a goal, an injury, a substitution.

        Stoppage advances the SCOREBOARD but not the play clock, so no player
        moves while the ball is dead. That distinction is §19's whole point.
        """
        if seconds < 0:
            raise ValueError(f"stoppage must be non-negative, got {seconds}")
        self.stoppage_seconds += seconds
        return self.match_seconds

    def advance_to(self, target_play_seconds: float) -> float:
        """Jump forward to an absolute play time. Never backwards."""
        if target_play_seconds < self.play_seconds:
            raise ValueError(
                f"cannot rewind the play clock from {self.play_seconds:.3f} "
                f"to {target_play_seconds:.3f}")
        return self.advance(target_play_seconds - self.play_seconds)

    # -- the guarantee the whole package rests on -----------------------
    def advance_by_timeline(self, steps) -> "ContinuousClock":
        """Advance through ``(absolute_play_seconds, label)`` milestones.

        A convenience for building a readable event timeline. Because every
        step is an absolute play time, the result is independent of the order
        the CPU happens to execute the work in — which is exactly the property
        §20 determinism and §23 performance both depend on.
        """
        for absolute, _label in steps:
            self.advance_to(float(absolute))
        return self

    def snapshot(self) -> dict:
        return {
            "play_seconds": round(self.play_seconds, 6),
            "stoppage_seconds": round(self.stoppage_seconds, 6),
            "match_seconds": round(self.match_seconds, 6),
            "play_time": str(self.play_time),
            "match_time": str(self.match_time),
        }


def stoppage_for(event: str, rng: Optional[object] = None) -> float:
    """A conservative stoppage estimate for a dead-ball event.

    Deterministic given ``rng``: the caller supplies the seeded stream, so this
    never reaches for global randomness. There is deliberately no randomness
    here at all by default — a fixed table is more explainable and the
    stoppage number barely moves a season's outcome.
    """
    table = {
        "goal": 30.0,
        "penalty_awarded": 25.0,
        "injury": 45.0,
        "substitution": 20.0,
        "corner": 8.0,
        "throw_in": 4.0,
        "free_kick": 6.0,
        "offside": 3.0,
    }
    base = table.get(event, 0.0)
    if rng is None:
        return base
    # Optional jitter, drawn from the caller's seeded stream only.
    return base * (0.85 + 0.30 * float(rng.between(0, 1000)) / 1000.0)
