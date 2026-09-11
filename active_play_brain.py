"""Bounded, player-specific intelligence for on-ball decisions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActiveActionDecision:
    action: str
    confidence: float
    reason: str
    perception_quality: float = 0.5
    evaluation_error: float = 0.0
    risk: float = 0.5


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _attribute(dna: Any, domain: str, name: str, default: float) -> float:
    return float(getattr(getattr(dna, domain, None), name, default))


class ActivePlayBrain:
    """Choose intent from a player's imperfect view and imperfect judgment."""

    @staticmethod
    def _memory(player: Any) -> dict[str, float]:
        memory = getattr(player, "_active_brain_memory", None)
        if memory is None:
            memory = {"success": 0.0, "failure": 0.0, "pressure_failures": 0.0}
            setattr(player, "_active_brain_memory", memory)
        return memory

    @staticmethod
    def _soul_name(player: Any) -> str:
        try:
            from player_soul import SoulApplicator
            soul = SoulApplicator.get_soul(player)
            return getattr(getattr(soul, "archetype", None), "name", "")
        except Exception:
            return ""

    @classmethod
    def decide(
        cls,
        player: Any,
        x: float,
        y: float,
        teammates: list[Any],
        defenders: list[Any],
        position_engine: Any,
        team_profile: Any,
        under_pressure: bool,
        attacks_right: bool,
        game_state: Any,
    ) -> ActiveActionDecision:
        dna = getattr(player, "dna", None)
        tendencies = getattr(dna, "tendencies", None)
        technical = getattr(dna, "technical", None)
        mental = getattr(dna, "mental", None)

        memory = cls._memory(player)
        vision = _clamp(_attribute(dna, "mental", "vision", 50.0) / 100.0)
        anticipation = _clamp(_attribute(dna, "mental", "anticipation", 50.0) / 100.0)
        decisions = _clamp(_attribute(dna, "mental", "decisions", 50.0) / 100.0)
        composure = _clamp(_attribute(dna, "mental", "composure", 50.0) / 100.0)
        concentration = _clamp(_attribute(dna, "mental", "concentration", 50.0) / 100.0)
        dribble = _clamp(float(getattr(tendencies, "attempts_dribble", 0.25)))
        control = _clamp(_attribute(dna, "technical", "ball_control", 50.0) / 100.0)

        # Players do not receive the whole defensive map. Unknown defenders
        # still affect execution in the existing physics layer.
        perception_quality = _clamp(0.35 * vision + 0.35 * anticipation + 0.30 * concentration)
        awareness_radius = 8.0 + 22.0 * perception_quality

        visible_defenders = []
        nearest_defender = None
        if position_engine is not None:
            for defender in defenders or []:
                if getattr(defender, "position", "") == "GK":
                    continue
                dx, dy = position_engine.get_position(defender.name)
                distance = math.hypot(dx - x, dy - y)
                if distance <= awareness_radius:
                    visible_defenders.append(defender)
                if nearest_defender is None or distance < nearest_defender:
                    nearest_defender = distance

        perceived_nearest = None
        if position_engine is not None and visible_defenders:
            perceived_nearest = min(
                math.hypot(*(
                    position_engine.get_position(defender.name)[i] - (x, y)[i]
                    for i in (0, 1)
                ))
                for defender in visible_defenders
            )
        space = 1.0 if perceived_nearest is None else _clamp((perceived_nearest - 1.5) / 7.5)
        final_third = x > 70.0 if attacks_right else x < 35.0
        own_half = x < 35.0 if attacks_right else x > 70.0

        carry_score = 0.18 + dribble * 0.48 + control * 0.18 + space * 0.24
        pass_score = 0.42 + decisions * 0.24 + composure * 0.12

        if final_third:
            carry_score += 0.10
        if own_half:
            pass_score += 0.12
        if under_pressure:
            carry_score -= 0.34
            pass_score += 0.16

        style = getattr(getattr(team_profile, "style", None), "value", "")
        if style in {"tiki_taka", "structured_possession", "possession"}:
            pass_score += 0.12
        elif style in {"fluid_counter", "route_one", "attacking", "ultra_attacking"}:
            carry_score += 0.08

        soul_name = cls._soul_name(player)
        if soul_name in {"ATTACKING_PROPHET", "WIDE_DESTROYER"}:
            carry_score += 0.10
        elif soul_name in {"MIDFIELD_PHILOSOPHER", "SWEEPER_SAGE"}:
            pass_score += 0.10

        state_value = getattr(game_state, "name", "LEVEL")
        if state_value in {"HOME_CHASE", "AWAY_CHASE"}:
            carry_score += 0.08
        elif state_value in {"HOME_CRUISE", "AWAY_CRUISE"}:
            pass_score += 0.10

        confidence_memory = _clamp(0.5 + (memory["success"] - memory["failure"]) * 0.03)
        carry_score += (confidence_memory - 0.5) * 0.10
        pass_score += (confidence_memory - 0.5) * 0.10
        if under_pressure:
            pass_score -= memory["pressure_failures"] * 0.02

        # Better decision-makers estimate options more accurately, but nobody
        # is noise-free. This is judgment error, separate from execution.
        evaluation_error = (1.0 - decisions) * 0.24 + (1.0 - composure) * 0.10
        carry_score += random.gauss(0.0, evaluation_error)
        pass_score += random.gauss(0.0, evaluation_error)

        carry_score = max(0.01, carry_score)
        pass_score = max(0.01, pass_score)
        total = carry_score + pass_score
        action = "CARRY" if random.random() < carry_score / total else "PASS"
        confidence = max(carry_score, pass_score) / total
        risk = _clamp(0.35 + (carry_score - pass_score) * 0.35)
        reason = (
            "open space and ball-carrying profile"
            if action == "CARRY"
            else "available support, pressure, and game control"
        )
        return ActiveActionDecision(
            action, round(confidence, 3), reason,
            round(perception_quality, 3), round(evaluation_error, 3), round(risk, 3)
        )

    @classmethod
    def record_outcome(cls, player: Any, decision: ActiveActionDecision,
                       success: bool, under_pressure: bool) -> None:
        """Update short-lived match memory without creating perfect learning."""
        memory = cls._memory(player)
        for key in memory:
            memory[key] *= 0.96
        if success:
            memory["success"] += 1.0
        else:
            memory["failure"] += 1.0
            if under_pressure:
                memory["pressure_failures"] += 1.0