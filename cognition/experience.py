"""Learned behaviour — memories become temperament.

Ported from TOLAND's ``ExperienceModel`` (D:\\TOLAND FOOTBALL FEDERATION\\
football_sim\\cognition\\experience.py) verbatim:

    risk_in_box      = max(0.4, 1.0 - foul_pain * 0.05)        fear from pain
    tackle_aggression= min(1.5, 1.0 + (clean_wins - lost_duels) * 0.03)
    confidence       = max(0.5, 1.0 + clean_wins*0.04 - lost_duels*0.06)

The bias dict is the player's *temperament*: it modifies how the neural
distribution is sampled (see cognition.mind.temper), so two players with
identical brains and bodies play differently purely because of what they
remember.
"""

from __future__ import annotations

from typing import Any


class ExperienceModel:
    """Temperament biases learned from episodic memory."""

    def __init__(self) -> None:
        # Start neutral — a blank slate plays exactly like the raw brain.
        self.bias = {
            "tackle_aggression": 1.0,
            "risk_in_box": 1.0,
            "confidence": 1.0,
        }

    def apply_memory(self, memory_system: Any) -> None:
        """Fold the current memory state into temperament.

        Called on meaningful events (and always at match end), not every
        tick — temperament should change slowly.
        """
        foul_pain = memory_system.recall("foul_in_box")
        lost_duels = memory_system.recall("lost_duel")
        clean_wins = memory_system.recall("clean_tackle_win")

        # Fear grows from pain.
        self.bias["risk_in_box"] = max(0.4, 1.0 - foul_pain * 0.05)

        # Aggression adapts to how the duels went.
        self.bias["tackle_aggression"] = min(
            1.5, 1.0 + (clean_wins - lost_duels) * 0.03
        )

        # Confidence is fragile.
        self.bias["confidence"] = max(
            0.5, 1.0 + (clean_wins * 0.04) - (lost_duels * 0.06)
        )

    def __repr__(self) -> str:
        b = self.bias
        return (f"ExperienceModel(risk_in_box={b['risk_in_box']:.3f}, "
                f"tackle_aggression={b['tackle_aggression']:.3f}, "
                f"confidence={b['confidence']:.3f})")