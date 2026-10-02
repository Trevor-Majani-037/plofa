"""
PLOFA 26/27 — TRAINING SYSTEM
===============================
training_system.py

Between-match football subsystem: a lightweight but FUNCTIONAL training
week that runs between matchdays (the "beyond the pitch" layer alongside
squad_manager's fatigue model, player_personality's professionalism, and
season_manager's SeasonState).

What it does for each player every matchday gap:
    • Consults personality (professionalism) — a 90-pro player turns up
      and sharpens; a 20-pro player coasts and can lose an edge.
    • Consults manager emphasis — risk-averse managers drill defense,
      high-risk managers drill attack, possession managers drill timing.
    • Gives young players (< 24) measurable attribute development in the
      focus areas; veterans hold level; nobody gets +10 overnight.
    • Nudges form/confidence toward a "training sharpness" band instead
      of just letting matches drive it.
    • Manages fatigue: professional players recover faster; heavy load
      adds a small training-lag effect (never an injury itself).
    • Commits ALL changes into SeasonState so they persist across
      matchdays and feed into next week's DNA (apply_pre_match).

Nothing here is cosmetic — every effect lands on numbers the match
engine already reads (DNA attributes, form.confidence, form.fatigue_level).

Usage:
    from training_system import TrainingSystem, TrainingFocus

    system = TrainingSystem()
    system.run_week(players, team_name="Hartwell City",
                    matchday=3, state=season_state, manager=home_mgr)
    print(system.report_text())

Integration: season_manager.run_matchday() calls run_week() for every
club after the fixtures finish, right before SeasonState.save().
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# FOCUS AREAS
# ─────────────────────────────────────────────

class TrainingFocus(Enum):
    ATTACK      = "attack"
    DEFENSE     = "defense"
    FITNESS     = "fitness"
    SET_PIECES  = "set_pieces"
    TACTICS     = "tactics"


# Focus -> the DNA attribute domains it develops.
_FOCUS_DOMAINS: Dict[TrainingFocus, List[str]] = {
    TrainingFocus.ATTACK:     ["technical", "mental"],
    TrainingFocus.DEFENSE:    ["defending", "mental"],
    TrainingFocus.FITNESS:    ["physical"],
    TrainingFocus.SET_PIECES: ["technical"],
    TrainingFocus.TACTICS:    ["mental", "passing"],
}

# Focus -> concrete attribute names inside those domains.
_FOCUS_ATTRIBUTES: Dict[TrainingFocus, List[str]] = {
    TrainingFocus.ATTACK:     ["finishing", "long_shots", "crossing",
                               "first_touch", "dribbling", "vision",
                               "positioning"],
    TrainingFocus.DEFENSE:    ["tackling", "marking", "interceptions",
                               "blocking", "clearances", "anticipation",
                               "concentration"],
    TrainingFocus.FITNESS:    ["stamina", "pace", "acceleration", "agility"],
    TrainingFocus.SET_PIECES: ["free_kick", "penalty_taking", "heading",
                               "crossing"],
    TrainingFocus.TACTICS:    ["decisions", "positioning", "anticipation",
                               "concentration", "geometric_awareness",
                               "through_balls", "switch_play"],
}


# ─────────────────────────────────────────────
# TRAINING RECORD
# ─────────────────────────────────────────────

@dataclass
class TrainingRecord:
    """What one player gained/lost from a single training week."""
    player: str
    team: str
    focus: str
    matchday: int
    effectiveness: float                 # 0..1 how much the work stuck
    confidence_delta: float              # rounds to 1-2 form points
    fatigue_delta: float                 # rounds to recovery/lag
    age: int = 24
    attribute_deltas: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# THE TRAINING SYSTEM
# ─────────────────────────────────────────────

class TrainingSystem:
    """Applies one training week per player, per matchday gap."""

    # ── manager emphasis → focus blend ─────────────────────────
    @staticmethod
    def _manager_focus(manager: Any) -> TrainingFocus:
        """Map a ManagerProfile's philosophy to a dominant training focus."""
        if manager is None:
            return TrainingFocus.TACTICS
        prefer_possession = getattr(manager, "prefer_possession", 0.5) or 0.5
        risk = getattr(manager, "risk_tolerance", 0.5) or 0.5
        if prefer_possession >= 0.65:
            return TrainingFocus.TACTICS
        if risk >= 0.65:
            return TrainingFocus.ATTACK
        if risk <= 0.35:
            return TrainingFocus.DEFENSE
        return TrainingFocus.FITNESS

    @staticmethod
    def _position_focus(player: Any) -> TrainingFocus:
        """A player's natural role rotates their positional focus."""
        pos = getattr(player, "position", None)
        if not pos and hasattr(player, "dna"):
            pos = getattr(player.dna, "position", None)
        pos = pos or ""
        if pos in ("ST", "CF", "LW", "RW", "CAM"):
            return TrainingFocus.ATTACK
        if pos in ("CB", "LB", "RB", "CDM"):
            return TrainingFocus.DEFENSE
        if pos == "GK":
            return TrainingFocus.SET_PIECES
        return TrainingFocus.TACTICS

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def _professionalism(player: Any) -> float:
        pers = getattr(player, "personality", None)
        if pers is not None:
            pro = getattr(pers, "professionalism", 50.0)
            if pro is not None:
                return float(pro)
        return 50.0

    @staticmethod
    def _age(player: Any) -> int:
        dna = getattr(player, "dna", None)
        if dna is not None:
            return int(getattr(dna, "age", 24))
        return 24

    @staticmethod
    def _fatigue_level(player: Any) -> float:
        dna = getattr(player, "dna", None)
        if dna is not None:
            form = getattr(dna, "form", None)
            if form is not None:
                return float(getattr(form, "fatigue_level", 0.0))
        return 0.0

    @staticmethod
    def _set_domain_attr(player: Any, domain: str, attr: str, value: float):
        obj = getattr(player, "dna", None)
        if obj is None:
            return
        grp = getattr(obj, domain, None)
        if grp is None:
            return
        if hasattr(grp, attr):
            setattr(grp, attr, max(0.0, min(100.0, value)))

    def _development_gain(self, player: Any, focus: TrainingFocus,
                          effectiveness: float) -> Dict[str, float]:
        """Age-scaled attribute development in the focus domains.
        U23 players develop; prime players hold; veterans gently decline
        in speed-adjacent physical areas. Always small (±0.2-1.4 per week)."""
        age = self._age(player)
        deltas: Dict[str, float] = {}

        if age >= 33:
            dev = -0.25 * effectiveness
        elif age >= 29:
            dev = 0.0
        elif age <= 21:
            dev = 1.2 * effectiveness * random.uniform(0.8, 1.2)
        elif age <= 24:
            dev = 0.7 * effectiveness * random.uniform(0.7, 1.2)
        else:
            dev = 0.35 * effectiveness * random.uniform(0.6, 1.2)

        if abs(dev) < 1e-9:
            return deltas

        grp_name = _FOCUS_DOMAINS[focus][0]
        obj = getattr(player, "dna", None)
        grp = getattr(obj, grp_name, None) if obj is not None else None
        if grp is None:
            return deltas

        for attr in _FOCUS_ATTRIBUTES[focus]:
            current = getattr(grp, attr, None)
            if current is None:
                continue
            gain = round(max(-1.0, min(1.5, dev * random.uniform(0.5, 1.4))), 2)
            if gain == 0.0:
                continue
            self._set_domain_attr(player, grp_name, attr, current + gain)
            deltas[attr] = gain
        return deltas

    # ── the core: one week ────────────────────────────────────
    def run_week(
        self,
        players: List[Any],
        team_name: str = "",
        matchday: int = 0,
        manager: Any = None,
        state: Any = None,
    ) -> List[TrainingRecord]:
        """
        Run one training week for a list of players (PlayerProfile or any
        object exposing .dna / .personality). If `state` (SeasonState) is
        provided, changes are committed into its per-player persisted dicts
        so they survive to next matchday.

        Returns a TrainingRecord per player (empty notes = player sat out).
        """
        records: List[TrainingRecord] = []
        team_blend = self._manager_focus(manager)

        for player in players:
            name = getattr(player, "name", "")
            if not name:
                continue

            focus = self._position_focus(player)
            if team_blend is not None:
                focus = team_blend  # team sessions dominate individual reps

            pro = self._professionalism(player)
            age = self._age(player)
            fatigue = self._fatigue_level(player)

            # Effectiveness: professionalism dominates, age & fatigue tax it.
            effectiveness = 0.55 + (pro / 100.0) * 0.45
            effectiveness -= max(0.0, (fatigue - 60.0) / 100.0) * 0.30
            effectiveness *= 1.0 - max(0.0, (age - 30) * 0.012)
            effectiveness = max(0.05, min(1.0, effectiveness))

            notes: List[str] = []
            if pro >= 75:
                notes.append("model professional on the training ground")
            elif pro <= 40:
                notes.append("coasted through drills")

            # Confidence drift toward a training-sharpness band.
            conf = getattr(player, "dna", None)
            cur_conf = float(getattr(getattr(conf, "form", None), "confidence", 50.0))
            sharp_target = 52.0 + (pro / 100.0) * 16.0          # ~52..68
            conf_delta = round((sharp_target - cur_conf) * 0.06 * effectiveness, 2)
            conf_delta = max(-2.0, min(2.0, conf_delta))
            if conf is not None and hasattr(conf, "form"):
                conf.form.confidence = max(0.0, min(100.0, cur_conf + conf_delta))

            # Fatigue: pros recover cleanly; heavy load leaves a small lag.
            if effectiveness >= 0.7:
                fatigue_delta = -round(2.0 + (pro / 100.0) * 4.0, 1)   # recover
            else:
                fatigue_delta = -round(1.0 + (pro / 100.0) * 2.5, 1)
            if conf is not None and hasattr(conf, "form"):
                conf.form.fatigue_level = max(
                    0.0, min(100.0, getattr(conf.form, "fatigue_level", 0.0) + fatigue_delta)
                )

            # Attribute development (young players only, always small).
            deltas = self._development_gain(player, focus, effectiveness)

            record = TrainingRecord(
                player=name, team=team_name, focus=focus.value,
                matchday=matchday, effectiveness=round(effectiveness, 3),
                confidence_delta=conf_delta,
                fatigue_delta=fatigue_delta,
                age=age,
                attribute_deltas=deltas,
                notes=notes,
            )

            # Commit to SeasonState so the changes persist.
            if state is not None:
                s = state.get_player_state(name)
                if "confidence" in s:
                    s["confidence"] = round(
                        max(0.0, min(100.0, float(s["confidence"]) + conf_delta)), 1
                    )
                if "fatigue_level" in s:
                    s["fatigue_level"] = round(
                        max(0.0, min(100.0, float(s["fatigue_level"]) + fatigue_delta)), 1
                    )
                state.players[name] = s

            records.append(record)

        return records

    def report_text(self, records: List[TrainingRecord],
                    team_name: str = "") -> str:
        """Human-readable weekly training report."""
        if not records:
            return f"  🏋️  {team_name}: no training data."
        lines = [f"  🏋️  {team_name} training week "
                 f"(matchday {records[0].matchday}):"]
        up = [r for r in records if any(r.attribute_deltas.values())]
        sharp = [r for r in records if r.confidence_delta >= 1.0]
        if up:
            for r in up:
                gained = ", ".join(
                    f"{k} +{v}" for k, v in r.attribute_deltas.items() if v > 0
                )
                lines.append(f"    ▸ {r.player} (aged {r.age}, "
                             f"{r.focus}): {gained or 'held level'}")
        if sharp:
            lines.append("    ▸ sharpened: " + ", ".join(r.player for r in sharp))
        if any(r.effectiveness < 0.3 for r in records):
            lines.append("    ⚠️  low application in this group")
        return "\n".join(lines)