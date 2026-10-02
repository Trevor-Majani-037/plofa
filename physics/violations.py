"""
physics/violations.py — the no-teleportation guard.
====================================================

§15 calls this a hard invariant:

    "A player must NOT teleport from (x1, y1) to (x2, y2) with zero simulated
     time... Add tests that detect impossible movement speeds. Do not silently
     allow impossible movement."

This module is the runtime half of that. Tests are the other half. Both exist
because the failure mode is invisible if you only look at one of them: a test
tells you a violation happened, this tells you *where* and *when*, in a live
match, with the entity names attached.

Why a violation registry and not just an exception
--------------------------------------------------
A hard raise on the first violation would abort a match. That is the right
behaviour in a test and the wrong behaviour in a season, where one bad frame
should be *reported* and the match carried on. So:

* :func:`check_move` returns a :class:`PhysicsViolation` or ``None``.
* :class:`PhysicsWorld` collects violations and keeps counting.
* :class:`PhysicsViolationError` exists for callers that *do* want to abort —
  a test, or a calibration run that should fail loudly.

The distinction matters because the engine currently *has* position jumps and
knows about them (see the comment at ``match_engine.py:2715``). Rolling that
out has to be observable, or the first sign of it will be a season that quietly
changed shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: Absolute ceiling on any measured speed before it counts as a violation.
#: Matches ``player_motion.ABSOLUTE_SPEED_CEILING`` but is restated here so this
#: module can be used to audit movement produced *anywhere*, including by code
#: that never touched the kinematics.
SPEED_CEILING_MPS = 12.0

#: A move smaller than this is treated as noise rather than a teleport. One
#: centimetre of float drift over a tick is not a physics violation.
EPSILON_M = 1e-6


@dataclass(frozen=True)
class PhysicsViolation:
    """One impossible movement, with enough context to find its cause."""

    kind: str
    detail: str
    subject: str = ""
    start: Tuple[float, float] = (0.0, 0.0)
    end: Tuple[float, float] = (0.0, 0.0)
    dt: float = 0.0
    implied_speed: float = 0.0

    def __str__(self) -> str:
        who = f" [{self.subject}]" if self.subject else ""
        return (f"{self.kind}{who}: {self.detail} "
                f"({self.start[0]:.2f},{self.start[1]:.2f}) -> "
                f"({self.end[0]:.2f},{self.end[1]:.2f}) in {self.dt:.4f}s "
                f"= {self.implied_speed:.2f} m/s")


class PhysicsViolationError(RuntimeError):
    """Raised when a caller wants an impossible movement to abort the run."""

    def __init__(self, violation: PhysicsViolation) -> None:
        super().__init__(str(violation))
        self.violation = violation


def check_move(start: Tuple[float, float], end: Tuple[float, float],
               dt: float,
               subject: str = "",
               speed_ceiling: float = SPEED_CEILING_MPS) -> Optional[PhysicsViolation]:
    """Return a violation if this move was physically impossible, else ``None``.

    Two things are checked, and they are different failures:

    * **Teleportation** — a non-zero distance covered in zero (or negative)
      time. This is the §15 violation proper.
    * **Impossible speed** — a move that implies a speed no human can reach.
      This catches a miscalibrated profile or a bad integration, which would
      otherwise show up much later as a nonsensical position.

    A zero-length move in zero time is explicitly allowed: nothing happened,
    which is not a teleport.
    """
    if dt < 0:
        return PhysicsViolation(
            kind="negative_time",
            detail=f"movement given negative duration {dt:.4f}s",
            subject=subject, start=tuple(start), end=tuple(end), dt=dt)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)

    if dist <= EPSILON_M:
        return None                      # nothing moved; not a violation

    if dt <= EPSILON_M:
        return PhysicsViolation(
            kind="teleport",
            detail=f"covered {dist:.2f} m in {dt:.6f}s",
            subject=subject, start=tuple(start), end=tuple(end), dt=dt,
            implied_speed=math.inf)

    speed = dist / dt
    if speed > speed_ceiling:
        return PhysicsViolation(
            kind="impossible_speed",
            detail=(f"implied {speed:.2f} m/s exceeds the "
                    f"{speed_ceiling:.2f} m/s ceiling"),
            subject=subject, start=tuple(start), end=tuple(end), dt=dt,
            implied_speed=speed)
    return None


@dataclass
class ViolationLog:
    """Collects violations across a match without interrupting it."""

    entries: List[PhysicsViolation] = field(default_factory=list)
    #: how many were dropped after the cap, so the total is never understated
    dropped: int = 0
    cap: int = 200

    def record(self, violation: Optional[PhysicsViolation]) -> None:
        if violation is None:
            return
        if len(self.entries) < self.cap:
            self.entries.append(violation)
        else:
            self.dropped += 1

    def check(self, start, end, dt, subject: str = "",
              speed_ceiling: float = SPEED_CEILING_MPS) -> Optional[PhysicsViolation]:
        violation = check_move(start, end, dt, subject, speed_ceiling)
        self.record(violation)
        return violation

    @property
    def total(self) -> int:
        return len(self.entries) + self.dropped

    def by_kind(self, kind: str) -> List[PhysicsViolation]:
        return [v for v in self.entries if v.kind == kind]

    def clear(self) -> None:
        self.entries.clear()
        self.dropped = 0

    def summary(self) -> str:
        if not self.total:
            return "no physics violations"
        counts: dict = {}
        for v in self.entries:
            counts[v.kind] = counts.get(v.kind, 0) + 1
        parts = [f"{n}x {k}" for k, n in sorted(counts.items())]
        tail = f" (+{self.dropped} dropped)" if self.dropped else ""
        return f"{self.total} physics violations: {', '.join(parts)}{tail}"

    def raise_if_any(self) -> None:
        """Abort the run if anything was recorded. For tests and calibration."""
        if self.entries:
            raise PhysicsViolationError(self.entries[0])
