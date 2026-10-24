"""THE SITUATIONAL DECISION PATH — attention, memory, recurrence and a plan.

WHERE IT FITS
-------------
``brain_integration.NeuralDecisionBrain.decide`` is the engine's single
on-ball entry point.  It routes:

    a SituationalBrain  ->  HERE (attention + recurrence + memory + plan)
    a FootballBrain      ->  the shipped path (cognition route, then core)

The route is chosen by the TYPE of the brain registered for the player,
so the shipped ``brains/`` set is untouched: nothing in this file runs
unless a player has been given a ``SituationalBrain`` explicitly.

THE ORDER OF OPERATIONS, AND WHY
--------------------------------
1. **FOV GATE** — if the player has an ENGAGED mind, the scene is gated by
   his senses first, exactly as the cognition path does.  A player cannot
   attend to a teammate he cannot see, and the attention set is built from
   the gated scene.
2. **TOKENS** — the entity set the attention reads.
3. **STATE / ROLE / MEMORY / PLAN** — the four blocks.
4. **RECURRENT FORWARD** — the GRU, carrying this player's hidden state.
5. **PLAN PROPOSAL** — the plan head names a phase; the team adopts it
   only with the hysteresis in ``player_state.TeamPlan.propose``.
6. **CONSEQUENCE + COACH + TEMPERATURE** — the same seams the shipped
   path uses, in the same order, so the decision pipeline downstream of
   the network is IDENTICAL.

THE PLAN IS ADVISORY, NOT AUTHORITATIVE
---------------------------------------
The plan head proposes; the team plan adopts with hysteresis.  The
engine's own possession-phase machine (``possession_phases``) is NOT
overridden — this is an input that conditions the players' decisions, not
a replacement for the phase engine.  Making it authoritative is a
separate, deliberate decision with its own gate.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from situational_brain import (
    HIDDEN_D, OUT_D, PLAN_OUT_D, SituationalBrain, _softmax,
    STATE_D, MEMORY_D, PLAN_D,
)


class SituationalDecisionBrain:
    """The four-capability decision path."""

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
        from brain_integration import (
            _apply_consequence, _apply_coach_bias, _build_decision,
            _decision_temperature, _fatigue_estimate, _resolve_brain,
            _sample_from_probs,
        )
        from perception import perceive
        from situation_tokens import entity_tokens
        from situational_context import assemble, situation_blocks

        brain = _resolve_brain(player)
        if not isinstance(brain, SituationalBrain):
            raise TypeError("SituationalDecisionBrain needs a SituationalBrain")

        # ── 1) FOV GATE ───────────────────────────────────────────
        vis_teammates, vis_defenders = teammates, defenders
        try:
            from cognition.mind import get_mind

            mind = get_mind(getattr(player, "name", "") or "")
            if mind is not None and mind.engaged:
                vis_teammates, vis_defenders = mind.perceive(
                    player, x, y, teammates, defenders, position_engine)
        except Exception:
            vis_teammates, vis_defenders = teammates, defenders

        # ── 2) TOKENS ─────────────────────────────────────────────
        tokens, token_mask = entity_tokens(
            player, x, y, vis_teammates, vis_defenders,
            position_engine, attacks_right)

        # ── 3) THE FOUR BLOCKS ────────────────────────────────────
        from brain_integration import _score_diff_of

        score_diff = _score_diff_of(game_state)
        base_vec, scene = perceive(
            player, x, y, vis_teammates, vis_defenders, position_engine,
            under_pressure, attacks_right, game_state, minute,
            team_possession=True, score_diff=score_diff,
            return_scene=True,
        )
        role, memory, plan = situation_blocks(
            player, x, y, scene["teammates"], scene["defenders"],
            scene["position_engine"], attacks_right, base_vec)
        # Adapt role width to the brain's expected input.  Old brains
        # (47-wide, 9 role features) and new brains (51-wide, 13 role
        # features) coexist: pad with zeros or truncate so the matmul
        # never fails.  A zero-padded feature is "no data" and the net
        # learns to ignore it; truncation drops features the brain never
        # saw, which is correct.
        expected_total = brain.w_q.shape[0]
        role_width = expected_total - STATE_D - MEMORY_D - PLAN_D
        role = np.asarray(role, dtype=np.float64).reshape(-1)
        if role.shape[0] < role_width:
            role = np.pad(role, (0, role_width - role.shape[0]))
        elif role.shape[0] > role_width:
            role = role[:role_width]
        situation = assemble(base_vec, role, memory, plan)

        # ── 4) THE RECURRENT FORWARD ──────────────────────────────
        store = getattr(position_engine, "neural_states", None)
        pstate = store.player(getattr(player, "name", "") or "") \
            if store is not None else None
        h_prev = pstate.ensure_hidden(HIDDEN_D) if pstate is not None \
            else np.zeros(HIDDEN_D, dtype=np.float64)
        logits, plan_logits, h_new, _cache = brain.forward(
            tokens, token_mask, base_vec, role, memory, plan, h_prev)
        if pstate is not None:
            pstate.hidden = h_new
        probs = _softmax(logits)

        # ── 5) THE PLAN PROPOSAL ──────────────────────────────────
        plan_trace = None
        if store is not None:
            from player_state import player_team

            team = player_team(player)
            if team:
                team_plan = store.team(team)
                pprobs = _softmax(plan_logits)
                pid = int(np.argmax(pprobs))
                conf = float(pprobs[pid])
                adopted = team_plan.propose(pid, conf)
                plan_trace = {
                    "proposed": pid,
                    "confidence": round(conf, 4),
                    "adopted": bool(adopted),
                    "team_plan": team_plan.plan_id,
                }

        # ── 6) THE SAME SEAMS AS THE SHIPPED PATH ─────────────────
        favored_flank: Optional[str] = None
        if coach_instructions is not None:
            probs = _apply_coach_bias(
                probs, player,
                getattr(coach_instructions, "intent_bias", None))
            favored_flank = getattr(coach_instructions, "favored_flank", None)
        fatigue = _fatigue_estimate(player, minute)

        cons_trace = None
        base24 = np.asarray(base_vec, dtype=np.float64)
        probs, cons_trace = _apply_consequence(
            probs, base24, getattr(player, "position", ""), record_trace,
        )

        temperature = _decision_temperature(player, under_pressure, fatigue)
        chosen_idx = _sample_from_probs(probs, temperature)
        chosen_prob = float(probs[chosen_idx])

        decision = _build_decision(
            brain, player, x, y, vis_teammates, vis_defenders,
            position_engine, attacks_right, fatigue, situation, probs,
            chosen_idx, chosen_prob, minute, record_trace, favored_flank,
            cons_trace,
        )
        if record_trace and decision.trace is not None:
            decision.trace["plan"] = plan_trace
            decision.trace["situation"] = {
                "state": [round(float(v), 4) for v in base_vec],
                "role": [round(float(v), 4) for v in role],
                "memory": [round(float(v), 4) for v in memory],
                "plan": [round(float(v), 4) for v in plan],
                "tokens": int(token_mask.sum()),
            }
        return decision
