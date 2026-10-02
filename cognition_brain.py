"""CognitionDecisionBrain — the TOLAND mind driving the neural body.

A drop-in alternative to ``NeuralDecisionBrain.decide`` with the same
signature and the same ``PlayerDecision`` return.  It runs the SAME evolved
``FootballBrain`` forward pass, but:

    1. the scene fed to the sensors is the player's FOV-gated view
       (VisionSystem: only actors inside view radius / attention slots),
       instead of the omniscient match state — a player cannot pick a
       teammate he cannot see, and an unseen defender puts no pressure
       on the option;
    2. the forward-pass distribution is re-shaped by the player's
       temperament (ExperienceModel biases folded via PlayerMind.temper)
       before temperature sampling — same brain, same geometry, two
       players, two choices, because one remembers;
    3. since 2026-09-20 the distribution is then reasoner-corrected with
       the SAME consequence critic as the vanilla path (its expected-
       payoff read of the shared perceived head, this time the player's
       FOV-gated scene) and the coach's intent bias is applied, so an
       engaged mind reasons about consequences exactly like an engaged
       neural player — the person, the memory, the odds.

With no mind registered the decision is byte-identical to the vanilla
neural core (delegates straight to ``_decide_core``).
"""

from __future__ import annotations

from typing import Any, List

from football_brain import INTENT_LABELS
from brain_sensors import _fatigue_estimate
from brain_integration import (
    _resolve_brain, build_sensors, _build_decision,
    _decision_temperature, _sample_from_probs, _decide_core,
    _apply_consequence, _apply_coach_bias,
)
from cognition.mind import get_mind

# The cognition senses gate IS the player's subjective view — so the
# perception layer's own FOV cone, relevance top-k and positional noise
# are bypassed.  This avoids double-gating (cognition gate + perception
# gate) and makes the TOLAND senses the authoritative subjective filter.
from perception import PerceptionConfig
_IDENTICAL_PERCEPTION = PerceptionConfig(enabled=False)


class CognitionDecisionBrain:
    """Neural brain + TOLAND subjective layer (senses + temperament)."""

    @staticmethod
    def decide(
        player: Any,
        x: float,
        y: float,
        teammates: List[Any],
        defenders: List[Any],
        position_engine: Any,
        team_profile: Any,
        under_pressure: bool,
        attacks_right: bool,
        game_state: Any,
        minute: float = 45.0,
        soul: Any = None,
        record_trace: bool = False,
        coach_instructions: Any = None,
    ):
        mind = get_mind(getattr(player, "name", ""))
        if mind is None or not mind.engaged:
            return _decide_core(
                player, x, y, teammates, defenders, position_engine,
                team_profile, under_pressure, attacks_right, game_state,
                minute, soul, record_trace, coach_instructions,
            )

        brain = _resolve_brain(player)

        # 1) FOV gate — only what the player can actually see survives.
        vis_teammates, vis_defenders = mind.perceive(
            player, x, y, teammates, defenders, position_engine,
        )

        # 2) Sensors from the GATED scene (same pipeline as the core path,
        #    but with perception in identity mode — the cognition senses
        #    gate IS the player's view).
        sensors = build_sensors(
            brain, player, x, y, vis_teammates, vis_defenders,
            position_engine, team_profile, under_pressure, attacks_right,
            game_state, minute, perception_config=_IDENTICAL_PERCEPTION,
        )

        # 3) Forward pass through the evolved brain.
        probs = brain.forward(sensors)
        fatigue = _fatigue_estimate(player, minute)

        # 4) Temperament re-shapes the distribution before sampling.
        probs = mind.temper(
            probs, labels=INTENT_LABELS,
            under_pressure=under_pressure, fatigue=fatigue,
        )

        # 5) Coach's on-ball say (same bias as the vanilla path; a mind
        #    does not lose the manager's instruction).
        favored_flank = None
        if coach_instructions is not None:
            probs = _apply_coach_bias(
                probs, player,
                getattr(coach_instructions, "intent_bias", None))
            favored_flank = getattr(coach_instructions, "favored_flank", None)

        # 6) Consequence reasoning on the SAME shared perceptual head the
        #    mind's brain read (the FOV-gated scene).  Byte-identical when
        #    the seam is off, exactly like the vanilla path.
        import numpy as np
        base24 = np.asarray(sensors[:24], dtype=np.float64)
        probs, cons_trace = _apply_consequence(
            probs, base24, getattr(player, "position", ""), record_trace,
        )

        # 7) Sample exactly as the core would.
        temperature = _decision_temperature(player, under_pressure, fatigue)
        chosen_idx = _sample_from_probs(probs, temperature)
        chosen_prob = float(probs[chosen_idx])
        return _build_decision(
            brain, player, x, y, vis_teammates, vis_defenders,
            position_engine, attacks_right, fatigue, sensors, probs,
            chosen_idx, chosen_prob, minute, record_trace,
            favored_flank, cons_trace,
        )


# ─────────────────────────────────────────────────────────────
# MIND OBSERVER (memory → the match's event stream)
# ─────────────────────────────────────────────────────────────
# Wires into match_engine's _absorb_chain via the same module-level hook
# pattern as the team press controller.  Every absorbed event is dispatched
# to the minds of the players involved, so episodic memory fills from real
# match facts without any parallel simulation.

def _observe_event(event: Any) -> None:
    # MatchEvent.player / secondary_player are NAME STRINGS, not profiles.
    for actor in (
        getattr(event, "player", None),
        getattr(event, "secondary_player", None),
    ):
        if actor is None:
            continue
        name = actor if isinstance(actor, str) else getattr(actor, "name", None)
        if not name:
            continue
        mind = get_mind(name)
        if mind is not None:
            mind.observe(event)


def engage_mind_observer() -> None:
    """Feed every absorbed match event into the involved players' minds."""
    from match_engine import set_cognition_observer
    set_cognition_observer(_observe_event)


def disengage_mind_observer() -> None:
    from match_engine import set_cognition_observer
    set_cognition_observer(None)