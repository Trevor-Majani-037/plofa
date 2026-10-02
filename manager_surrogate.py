"""
PLOFA 26/27 — MANAGER SURROGATE
====================================================
manager_surrogate.py

A learned expected-points table that converts "what the static manager did
in situation X" into a reward signal usable for evolving a ManagerBrain
(Phase 7).  It is a CURRICULUM, not the final signal: decisions (and
5-min checkpoints) are recorded from the STATIC ManagerProfile, then each
row is credited with the match outcome correlated with its team's final
result and xG delta.

State space: a 6-part bucket derived from the 12-float manager sensors (D1):

    score_state    3 values  (leading / level / trailing)   from s[0]
    minute_bucket  6 values  (0..5 covering 0..90')          from s[1]
    momentum_sign  2 values  (+ / -)                         from s[7]
    fatigue_hi     2 values  (F if fatigue >= 0.5)           from s[5]
    confidence_lo  2 values  (c if confidence < 0.4)         from s[6]
    recent_event   2 values  (E if recent goal impact)       from s[4]

= 3*6*2*2*2*2 = 288 possible buckets.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

POSTURE_LABELS = ("DEFEND", "BALANCED", "ATTACK")


def bucket(sensors: Union[np.ndarray, Sequence[float]]) -> str:
    """Return the 6-part bucket key for a 12-float manager sensor vector."""
    s = [float(v) for v in sensors]
    if len(s) < 12:
        raise ValueError(f"expected >= 12 sensors, got {len(s)}")

    score_state = "W" if s[0] > 0.0 else ("L" if s[0] < 0.0 else "D")
    minute_bucket = min(5, max(0, int(s[1] * 6.0)))
    momentum_sign = "+" if s[7] >= 0.5 else "-"
    fatigue_hi = "F" if s[5] >= 0.5 else "f"
    confidence_lo = "c" if s[6] < 0.4 else "C"
    recent_event = "E" if abs(s[4]) > 0.01 else "e"
    return f"{score_state}.{minute_bucket}.{momentum_sign}.{fatigue_hi}.{confidence_lo}.{recent_event}"


class ManagerSurrogate:
    """Per-(bucket, posture) expected-points table trained from collected
    (sensors, posture, outcome_points) rows."""

    def __init__(self):
        # table[(bucket, posture)] -> [sum_points, count]
        self._table: Dict[Tuple[str, str], List[float]] = {}
        # posture -> [sum_points, count]  (global fallback)
        self._posture_totals: Dict[str, List[float]] = {}
        self._total_count = 0

    # ── Training ────────────────────────────────────────────────

    def fit(self, rows: Sequence[Any]) -> None:
        """Aggregate rows.

        Each row is either (sensors, posture, outcome_points) or
        (sensors, posture, outcome_points, match_outcome); the extra
        match_outcome element is ignored here.
        """
        self._table = {}
        self._posture_totals = {}
        self._total_count = 0

        for row in rows:
            sensors, posture, points = row[0], row[1], float(row[2])
            key = self._safe_bucket(sensors)
            tkey = (key, posture)
            entry = self._table.setdefault(tkey, [0.0, 0.0])
            entry[0] += points
            entry[1] += 1.0

            pe = self._posture_totals.setdefault(posture, [0.0, 0.0])
            pe[0] += points
            pe[1] += 1.0
            self._total_count += 1

    @staticmethod
    def _safe_bucket(sensors) -> str:
        try:
            return bucket(sensors)
        except Exception:
            return "?.?.-.f.C.e"

    # ── Lookup ──────────────────────────────────────────────────

    def support(self, sensors: Union[np.ndarray, Sequence[float]],
                posture: str) -> int:
        return int(self._table.get((self._safe_bucket(sensors), posture), [0.0, 0.0])[1])

    def expected_points(self, sensors: Union[np.ndarray, Sequence[float]],
                        posture: str) -> float:
        """Mean outcome-points for (bucket, posture) with fallbacks:

        exact (bucket, posture) -> posture-only -> global -> 0.5.
        """
        key = (self._safe_bucket(sensors), posture)
        entry = self._table.get(key)
        if entry is not None and entry[1] > 0:
            return float(entry[0] / entry[1])

        pe = self._posture_totals.get(posture)
        if pe is not None and pe[1] > 0:
            return float(pe[0] / pe[1])

        return 0.5

    # ── Coverage / metadata ─────────────────────────────────────

    def coverage(self) -> Dict[str, int]:
        return {
            "buckets": len({k[0] for k in self._table}),
            "bucket_posture_pairs": len(self._table),
            "postures": len(self._posture_totals),
            "rows": self._total_count,
        }

    # ── Serialization ───────────────────────────────────────────

    def save(self, path: str) -> None:
        d = {
            "kind": "manager_surrogate",
            "version": 1,
            "table": {
                f"{b}||{p}": [s, c] for (b, p), (s, c) in self._table.items()
            },
            "posture_totals": {
                p: [s, c] for p, (s, c) in self._posture_totals.items()
            },
            "total_count": self._total_count,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ManagerSurrogate":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("kind") != "manager_surrogate":
            raise ValueError(f"expected kind='manager_surrogate', got {data.get('kind')!r}")
        obj = cls()
        obj._table = {
            tuple(k.split("||")): [float(s), float(c)]
            for k, (s, c) in data["table"].items()
        }
        obj._posture_totals = {
            p: [float(s), float(c)] for p, (s, c) in data["posture_totals"].items()
        }
        obj._total_count = int(data.get("total_count", 0))
        return obj