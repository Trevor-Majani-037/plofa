"""Episodic memory — what happened to this player, and how it feels now.

Ported from TOLAND's ``MemorySystem`` (D:\\TOLAND FOOTBALL FEDERATION\\
football_sim\\cognition\\memory.py).  Events are stored with a tag (e.g.
``foul_in_box``), a location, and an emotional severity that fades with a
``0.999`` per-second multiplier and is forgotten entirely once it drops
below ``0.1``.  Recall sums the surviving emotional weight of a tag — that
sum is what the ExperienceModel turns into temperament.
"""

from __future__ import annotations

from typing import Any, List, Optional


class MemoryEvent:
    """A single remembered moment."""

    __slots__ = ("tag", "location", "severity", "age")

    def __init__(self, tag: str, location: Optional[tuple] = None,
                 severity: float = 1.0):
        self.tag = tag
        self.location = location
        self.severity = severity
        self.age = 0.0  # seconds since the event


class MemorySystem:
    """Ordered, decaying recollection of episodic events."""

    # Multiply severity by this each simulated second of age.
    DECAY_PER_SECOND = 0.999
    # Any memory whose severity has faded below this is forgotten.
    FORGET_BELOW = 0.1

    def __init__(self) -> None:
        self.events: List[MemoryEvent] = []

    def store(self, tag: str, location: Optional[tuple] = None,
              severity: float = 1.0) -> None:
        self.events.append(MemoryEvent(tag, location, severity))

    def decay(self, dt: float = 1.0) -> None:
        """Age every memory by ``dt`` seconds; old ones fade and are
        forgotten once their emotional weight drops below the threshold."""
        decayed = []
        for e in self.events:
            e.age += dt
            e.severity *= (self.DECAY_PER_SECOND ** dt)
            if e.severity > self.FORGET_BELOW:
                decayed.append(e)
        self.events = decayed

    def recall(self, tag: str) -> float:
        """Total surviving emotional weight of a memory tag."""
        return sum(e.severity for e in self.events if e.tag == tag)

    def recall_all(self) -> dict:
        """Tag → total weight snapshot, for introspection / tests."""
        out: dict = {}
        for e in self.events:
            out[e.tag] = out.get(e.tag, 0.0) + e.severity
        return out

    def events_for(self, tag: str) -> List[Any]:
        return [e for e in self.events if e.tag == tag]

    def __len__(self) -> int:
        return len(self.events)