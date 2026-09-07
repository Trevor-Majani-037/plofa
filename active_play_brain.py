"""Active on-ball action selection for possession touches.

This layer chooses intent. Existing event chains remain responsible for
executing the chosen carry or pass and resolving the physical outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActiveActionDecision:
    action: str
    confidence: float
    reason: str


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ActivePlayBrain:
    """Choose the carrier's next intent from the current match state."""

    @staticmethod
    def decide(
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

        dribble = _clamp(float(getattr(tendencies, "attempts_dribble", 0.25)))
        composure = _clamp(float(getattr(mental, "composure", 50.0)) / 100.0)
        decisions = _clamp(float(getattr(mental, "decisions", 50.0)) / 100.0)
        control = _clamp(float(getattr(technical, "ball_control", 50.0)) / 100.0)

        nearest_defender = None
        if position_engine is not None:
            for defender in defenders or []:
                if getattr(defender, "position", "") == "GK":
                    continue
                dx, dy = position_engine.get_position(defender.name)
                distance = math.hypot(dx - x, dy - y)
                if nearest_defender is None or distance < nearest_defender:
                    nearest_defender = distance

        space = 1.0 if nearest_defender is None else _clamp((nearest_defender - 1.5) / 7.5)
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

        state_value = getattr(game_state, "name", "LEVEL")
        if state_value in {"HOME_CHASE", "AWAY_CHASE"}:
            carry_score += 0.08
        elif state_value in {"HOME_CRUISE", "AWAY_CRUISE"}:
            pass_score += 0.10

        carry_score = max(0.01, carry_score)
        pass_score = max(0.01, pass_score)
        total = carry_score + pass_score
        action = "CARRY" if carry_score > pass_score else "PASS"
        confidence = max(carry_score, pass_score) / total
        reason = (
            "open space and ball-carrying profile"
            if action == "CARRY"
            else "available support, pressure, and game control"
        )
        return ActiveActionDecision(action, round(confidence, 3), reason)