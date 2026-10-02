"""
PLOFA 26/27 — MANAGER MEMORY
====================================================
manager_memory.py

Structured episodic memory: the manager remembers what worked and what
didn't, indexed by the exact game situation at the time.  A memory key
is a 3-tuple (score_state, minute_bucket, posture_taken).

Memory is discounted over time (decay) and bounded (max_entries), so the
most-frequently-used memory lines persist while old/rare patterns fade.

Not wired into the engine yet.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

MEMORY_VERSION = 1


MINUTE_BUCKETS = ("0-15", "15-30", "30-45", "45-60", "60-75", "75-90")
SCORE_STATES = ("leading", "level", "trailing")
POSTURES = ("DEFEND", "BALANCED", "ATTACK")

_BUCKET_EOF: List[Tuple[int, int]] = [
    (1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90),
]


def _minute_bucket(minute: int) -> str:
    m = max(1, int(minute))
    for i, (lo, hi) in enumerate(_BUCKET_EOF):
        if m <= hi:
            return MINUTE_BUCKETS[i]
    return MINUTE_BUCKETS[-1]


def _score_state(score_diff: int) -> str:
    if score_diff > 0:
        return "leading"
    if score_diff < 0:
        return "trailing"
    return "level"


@dataclass
class OutcomeStats:
    """Accumulated result statistics for one (score_state, bucket, posture)."""
    wins: int = 0
    draws: int = 0
    losses: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "wins": self.wins, "draws": self.draws,
            "losses": self.losses, "xg_for": self.xg_for,
            "xg_against": self.xg_against, "count": self.count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OutcomeStats:
        return cls(
            wins=int(d.get("wins", 0)), draws=int(d.get("draws", 0)),
            losses=int(d.get("losses", 0)),
            xg_for=float(d.get("xg_for", 0.0)),
            xg_against=float(d.get("xg_against", 0.0)),
            count=int(d.get("count", 0)),
        )


class ManagerMemory:
    """Episodic memory store indexed by (score_state, minute_bucket, posture).

    Over time all entries decay by ``decay`` (multiplicatively on count), so
    fresh or frequently-used memories survive while stale ones drop off.
    """

    def __init__(self, max_entries: int = 100, decay: float = 0.97):
        self.max_entries = max_entries
        self.decay = decay
        self.table: Dict[Tuple, OutcomeStats] = {}

    # ── Key construction ────────────────────────────────────────

    @staticmethod
    def key_for(context) -> Tuple[str, str, str]:
        """Map a context dict/namespace into a memory key.

        Required attributes: ``score_diff`` (int), ``minute`` (int),
        ``posture_taken`` (str).
        """
        sd = int(getattr(context, "score_diff",
                         getattr(context, "goal_diff", 0)))
        minute = int(getattr(context, "minute", 1))
        posture = str(getattr(context, "posture_taken", "BALANCED"))
        return (_score_state(sd), _minute_bucket(minute), posture)

    # ── Recording ───────────────────────────────────────────────

    def record(self, context, posture_taken: str, result, team: str) -> None:
        """Record a completed match-decision pair into memory.

        ``result`` is a MatchResult (or duck-typed object) with
        ``home_goals``/``away_goals`` and ``config.home_team``.
        """
        key = self.key_for(context)

        # Derive win/draw/loss from the result.
        cfg = getattr(result, "config", None)
        is_home = True
        if cfg is not None:
            is_home = (getattr(cfg, "home_team", "") == team)
        gf = float(getattr(result, "home_goals", 0) or 0)
        ga = float(getattr(result, "away_goals", 0) or 0)
        if not is_home:
            gf, ga = ga, gf

        if gf > ga:
            outcome = "win"
        elif gf == ga:
            outcome = "draw"
        else:
            outcome = "loss"

        # XG handling: try to read numeric fields.
        xgf = 0.0
        xga = 0.0
        if is_home:
            xgf = float(getattr(result, "home_xg", 0) or 0)
            xga = float(getattr(result, "away_xg", 0) or 0)
        else:
            xgf = float(getattr(result, "away_xg", 0) or 0)
            xga = float(getattr(result, "home_xg", 0) or 0)

        if key not in self.table:
            self.table[key] = OutcomeStats()

        stats = self.table[key]
        if outcome == "win":
            stats.wins += 1
        elif outcome == "draw":
            stats.draws += 1
        else:
            stats.losses += 1
        stats.xg_for += xgf
        stats.xg_against += xga
        stats.count += 1

        # Decay ALL OTHER entries (the just-recorded key stays hot) so
        # unused memories fade between records.
        for k, v in self.table.items():
            if k != key:
                v.count = int(v.count * self.decay)

        # Strip dead entries (zero count AND no recorded outcomes).
        self.table = {k: v for k, v in self.table.items()
                      if v.count > 0 or v.wins or v.draws or v.losses}

        # Enforce max_entries: drop the lowest-count entry.
        if len(self.table) > self.max_entries:
            worst = min(self.table, key=lambda k: self.table[k].count)
            del self.table[worst]

    # ── Retrieval ───────────────────────────────────────────────

    def bias_for(self, context) -> np.ndarray:
        """Return (3,) bias vector for postures from memory.

        Each posture is scored by expected points (0..1) derived from
        historical wins/draws/losses under the same game-state conditions.
        Returns zeros if no data for any of the three postures.
        """
        base = self.key_for(context)
        biases = []
        for posture in POSTURES:
            key = (base[0], base[1], posture)
            s = self.table.get(key)
            if s is None or s.count == 0:
                biases.append(0.0)
            else:
                ep = (3.0 * s.wins + s.draws) / s.count   # 0..1
                biases.append(ep)
        biases = np.array(biases, dtype=np.float64)

        # If ALL postures are empty, return zeros (no memory).
        if not self.table:
            return np.zeros(3, dtype=np.float64)

        # Map from expected-points [0..1] to bias [-1..+1]
        # relative to the group (not absolute), so higher EP → +1.
        if biases.max() > 0:
            biases = biases / max(biases.max(), 1e-9)
            biases = 2.0 * biases - 1.0
        return biases

    def strength(self) -> float:
        """How much the manager trusts memory (0..1, saturated at 50 entries)."""
        total = sum(s.count for s in self.table.values())
        return min(1.0, total / 50.0)

    # ── Info ────────────────────────────────────────────────────

    def summarize(self) -> dict:
        return {
            "version": MEMORY_VERSION,
            "entries": len(self.table),
            "total_count": sum(s.count for s in self.table.values()),
            "strength": self.strength(),
            "decay": self.decay,
            "max_entries": self.max_entries,
        }

    # ── Persistence ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "version": MEMORY_VERSION,
            "max_entries": self.max_entries,
            "decay": self.decay,
            "table": {
                str(k): v.to_dict() for k, v in self.table.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> ManagerMemory:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        mem = cls(
            max_entries=int(raw.get("max_entries", 100)),
            decay=float(raw.get("decay", 0.97)),
        )
        for k_str, v in raw.get("table", {}).items():
            # Parse tuple key from string representation "(a, b, c)"
            key = ast.literal_eval(k_str)  # safe: always a 3-tuple of str
            mem.table[key] = OutcomeStats.from_dict(v)
        return mem

    def __repr__(self) -> str:
        return (f"ManagerMemory({len(self.table)} entries, "
                f"strength={self.strength():.2f})")