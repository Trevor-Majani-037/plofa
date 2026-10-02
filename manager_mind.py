"""
PLOFA 26/27 — MANAGER MIND
====================================================
manager_mind.py

The psychological filter layer that shapes BOTH perception and decisions.
The ManagerBrain is the raw "decision engine"; the ManagerMind is the
personality that colours what the brain sees (perception filters) and how
it acts (decision filters), plus event callbacks (goals, end of match),
pedagogy and selection preferences.

All attributes are floats in 0..1 (except stress_accumulator 0..1 and
history_posture_counts).

Not wired into the engine yet (Phase 10 toggle).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

MIND_VERSION = 1
POSTURE_LABELS = ("DEFEND", "BALANCED", "ATTACK")


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class ManagerMind:
    """The psychological filter layer of a manager.

    Static traits (seeded from manager DNA, evolved as floats) + stateful
    in-match attributes.  All methods are deterministic given the state.
    """

    # ── Static personality traits (0..1) ─────────────────────────
    pedagogy: float = 0.5             # develops young players
    eq: float = 0.5                   # emotional intelligence
    empathy: float = 0.5              # reads player state
    dogma: float = 0.5                # ideology vs pragmatism
    youth_trust: float = 0.5          # faith in younger players
    rotation_discipline: float = 0.5  # sticks to rotation plans
    pressure_baseline: float = 0.5    # composure under stress

    # ── State (in-match / across matches) ────────────────────────
    composure_current: float = 0.5
    stress_accumulator: float = 0.0
    history_posture_counts: Dict[str, int] = field(
        default_factory=lambda: {"DEFEND": 0, "BALANCED": 0, "ATTACK": 0})

    # ══════════════════════════════════════════════════════════════
    # PERCEPTION FILTERS — applied BEFORE the brain sees anything
    # ══════════════════════════════════════════════════════════════

    def filter_perception(self, sensors: np.ndarray, engine) -> np.ndarray:
        """Tint the raw 12-float sensor vector with the manager's bias.

        Returns a COPY — the caller's array is never mutated.
        """
        s = sensors.copy()
        # Paranoid manager: sees fatigue as worse than it is.
        if self.dogma > 0.6 and self.pressure_baseline < 0.4:
            s[5] = min(1.0, s[5] + 0.2)          # team_fatigue
            s[7] = max(0.0, s[7] - 0.15)         # momentum
        # Optimistic manager: sees fatigue as less severe.
        if self.eq > 0.7 and self.pressure_baseline > 0.7:
            s[5] = max(0.0, s[5] - 0.15)         # team_fatigue
            s[7] = min(1.0, s[7] + 0.1)          # momentum
        # Gut-feel manager: confidence reading sways with his own mood.
        if self.empathy > 0.7:
            mood_bias = (self.composure_current - 0.5) * 0.2
            s[6] = float(np.clip(s[6] + mood_bias, 0.0, 1.0))
        # Data-driven manager: trusts the reading literally (no change).
        return s

    # ══════════════════════════════════════════════════════════════
    # DECISION FILTERS — applied AFTER the brain outputs
    # ══════════════════════════════════════════════════════════════

    def filter_posture(self, raw_probs: np.ndarray, engine) -> np.ndarray:
        """Pull posture probabilities toward the manager's historical bias."""
        if self.dogma > 0.5:
            preferred = self._preferred_posture()
            pref_idx = list(POSTURE_LABELS).index(preferred)
            bias = np.zeros(3)
            bias[pref_idx] = (self.dogma - 0.5) * 0.6
            logits = np.log(np.clip(raw_probs, 1e-9, 1.0)) + bias
            z = logits - logits.max()
            return np.exp(z) / np.exp(z).sum()
        return raw_probs

    def filter_pressing(self, raw: float, engine) -> float:
        """Clamp pressing for paranoid+dogmatic managers (over-cautious)."""
        if self.dogma > 0.6 and self.pressure_baseline < 0.4:
            return float(np.clip(raw, 0.2, 0.7))
        return float(np.clip(raw, 0.0, 1.0))

    def filter_sub_urgency(self, raw: float, engine) -> float:
        """High empathy / youth_trust managers act faster on subs."""
        if self.empathy > 0.6:
            raw += 0.1
        if self.youth_trust > 0.7:
            raw += 0.05
        return float(np.clip(raw, 0.0, 1.0))

    def _preferred_posture(self) -> str:
        """Argmax of historical posture counts (deterministic tie-break)."""
        counts = self.history_posture_counts
        best = "BALANCED"
        best_count = -1
        for label in POSTURE_LABELS:           # DEFEND < BALANCED < ATTACK
            c = counts.get(label, 0)
            if c > best_count:
                best = label
                best_count = c
        return best

    def record_posture_taken(self, posture: str) -> None:
        """Record which posture the manager actually committed to."""
        if posture in self.history_posture_counts:
            self.history_posture_counts[posture] += 1
        else:
            self.history_posture_counts[posture] = 1

    # ══════════════════════════════════════════════════════════════
    # EVENT CALLBACKS — in-match psychology
    # ══════════════════════════════════════════════════════════════

    def on_goal_conceded(self, context=None) -> None:
        """EQ restores composure partially; conceding always stresses."""
        self.composure_current = _clip(
            self.composure_current + self.eq * 0.1 - (1.0 - self.eq) * 0.05)
        self.stress_accumulator = min(1.0, self.stress_accumulator + 0.15)

    def on_goal_scored(self, context=None) -> None:
        """Scoring calms: composure up a touch, stress down."""
        self.composure_current = _clip(self.composure_current + 0.05)
        self.stress_accumulator = max(0.0, self.stress_accumulator - 0.1)

    def end_of_match(self, result, team) -> None:
        """Update composure + stress from the final result.

        ``result`` may be a full MatchResult (has .config.home_team and
        goal fields) or any duck-typed object exposing goal counts.
        """
        is_home = True
        home_goal = away_goal = 0
        cfg = getattr(result, "config", None)
        if cfg is not None:
            is_home = (getattr(cfg, "home_team", None) == team)
        home_goal = float(getattr(result, "home_goals",
                                  getattr(result, "goals_for", 0)) or 0)
        away_goal = float(getattr(result, "away_goals",
                                  getattr(result, "goals_against", 0)) or 0)
        gf = home_goal if is_home else away_goal
        ga = away_goal if is_home else home_goal

        if gf > ga:                                   # win
            self.composure_current = _clip(self.composure_current + 0.08)
            self.stress_accumulator = max(0.0, self.stress_accumulator - 0.15)
        elif gf == ga:                                # draw
            self.composure_current = _clip(self.composure_current + 0.02)
            self.stress_accumulator = max(0.0, self.stress_accumulator - 0.05)
        else:                                         # loss
            # EQ cushions the psychological blow of a defeat.
            self.composure_current = _clip(
                self.composure_current - 0.08 + self.eq * 0.05)
            self.stress_accumulator = min(1.0, self.stress_accumulator + 0.15)

        # The match being over always allows some decompression.
        self.stress_accumulator = max(0.0, self.stress_accumulator - 0.05)

    def current_pressure_handling(self) -> float:
        """How well the manager is currently coping (composure w/ stress)."""
        return _clip(self.composure_current - 0.5 * self.stress_accumulator)

    # ══════════════════════════════════════════════════════════════
    # PEDAGOGY & SELECTION
    # ══════════════════════════════════════════════════════════════

    def apply_pedagogy(self, player, growth: float) -> float:
        """Young players develop faster under a pedagogical manager."""
        age = getattr(player.dna, "age", 25)
        if age >= 24:
            return growth
        return growth * (1.0 + self.pedagogy * 0.5)

    def select_starting_xi(self, available_players, match_importance) -> List[Any]:
        """Rank available players for the starting XI, best first.

        Score blends quality, current form and freshness, weighted by the
        manager's rotation discipline and youth trust.  High-importance
        matches weight quality/form more (play the best XI); low-importance
        matches weight freshness more (rotate).  Deterministic: ties break
        alphabetically, so the same inputs always produce the same order.
        """
        imp = _clip(float(match_importance), 0.0, 1.0)

        def _score(p):
            dna = getattr(p, "dna", None)
            form_state = getattr(dna, "form", None) if dna else None

            conf = getattr(form_state, "confidence", 50.0) or 50.0
            form = float(conf) / 100.0
            fatigue = float(getattr(form_state, "fatigue_level", 0.0) or 0.0) / 100.0
            quality = float(getattr(dna, "overall_rating", getattr(p, "rating", 60.0))
                            or 60.0) / 100.0
            age = float(getattr(dna, "age", 25) or 25)
            name = getattr(p, "name", "")

            # A fresher player is a better pick when rotation is valued.
            rotation_weight = self.rotation_discipline * (1.0 - 0.6 * imp)
            quality_weight = 0.40 + 0.30 * imp
            form_weight = 0.35
            freshness_weight = 0.25 * rotation_weight

            young_boost = 0.15 if age < 24 else 0.0

            score = (
                quality * quality_weight
                + form * form_weight
                + (1.0 - fatigue) * freshness_weight
                + young_boost * self.youth_trust
            )
            return -score, name     # sort best first, ties alphabetical

        return sorted(available_players, key=_score)

    # ══════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ══════════════════════════════════════════════════════════════

    def serialize(self) -> Dict[str, Any]:
        return {
            "kind": "manager_mind",
            "version": MIND_VERSION,
            "pedagogy": self.pedagogy,
            "eq": self.eq,
            "empathy": self.empathy,
            "dogma": self.dogma,
            "youth_trust": self.youth_trust,
            "rotation_discipline": self.rotation_discipline,
            "pressure_baseline": self.pressure_baseline,
            "composure_current": self.composure_current,
            "stress_accumulator": self.stress_accumulator,
            "history_posture_counts": dict(self.history_posture_counts),
        }

    @classmethod
    def deserialize(cls, data) -> "ManagerMind":
        kind = data.get("kind", "")
        if kind != "manager_mind":
            raise ValueError(f"expected kind='manager_mind', got {kind!r}")

        def _f(key: str, default: float) -> float:
            v = data.get(key, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        mind = cls(
            pedagogy=_f("pedagogy", 0.5),
            eq=_f("eq", 0.5),
            empathy=_f("empathy", 0.5),
            dogma=_f("dogma", 0.5),
            youth_trust=_f("youth_trust", 0.5),
            rotation_discipline=_f("rotation_discipline", 0.5),
            pressure_baseline=_f("pressure_baseline", 0.5),
            composure_current=_f("composure_current", 0.5),
            stress_accumulator=_f("stress_accumulator", 0.0),
        )
        hpc = data.get("history_posture_counts", {})
        if isinstance(hpc, dict):
            mind.history_posture_counts = {
                k: int(v) for k, v in hpc.items()
            }
        return mind

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ManagerMind":
        with open(path, "r", encoding="utf-8") as f:
            return cls.deserialize(json.load(f))

    def __repr__(self) -> str:
        return (f"ManagerMind(dogma={self.dogma:.2f}, eq={self.eq:.2f}, "
                f"empathy={self.empathy:.2f}, composure={self.composure_current:.2f})")