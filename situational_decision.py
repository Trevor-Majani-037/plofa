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
1. **FOV GATE** — `perceive` applies the role cone (130-200 deg), the range
   scaling, the positional noise and the relevance top-k (GK 12, CB 10,
   CM 8, ST 6), and returns the lists it kept.  That is the ONE vision gate
   on this path, and it is the one the distillation corpus used.
2. **TOKENS** — the entity set the attention reads, built from the men that
   gate kept.  NOBODY ELSE: a token is a man the cone saw.
3. **STATE / ROLE / MEMORY / PLAN** — the four blocks, from the same scene.
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

import os
from typing import Any, Dict, List, Optional

import numpy as np

from situational_brain import (
    HIDDEN_D, EMBED_D, OUT_D, PLAN_OUT_D, SituationalBrain, _softmax,
    STATE_D, ROLE_D, MEMORY_D, PLAN_D, INTENT_LABELS,
)


def _resolve_mind(player: Any):
    """The player's ``PlayerMind``, or ``None``.

    One lookup, used twice: the FOV gate (§1) and the temperament seam (§4b).
    A failure to import the cognition package must degrade to "no mind", not
    to a dead decision, because the shipped path has to run in environments
    where ``cognition`` is absent.

    Returning ``None`` is also the byte-identical case: a player with no
    registered mind is unaffected, exactly as documented in ``cognition.mind``
    (*"a player with no registered mind is completely unaffected"*).
    """
    try:
        from cognition.mind import get_mind

        return get_mind(getattr(player, "name", "") or "")
    except Exception:
        return None


def _token_source() -> str:
    """Which men the attention set is built from.

    ``scene`` (the default) — the lists ``perceive`` kept: role cone, range,
    top-k.  This is the feed the distillation corpus used
    (``surrogate_collect._situational_blocks`` reads ``scene``), and therefore
    the feed the shipped weights were fitted for.

    ``legacy`` — the feed in force until 2026-11-16: the mind's 25 m / 2 + 2
    frame (at most four men, no angle test), or, with no mind registered, the
    RAW lists and no cone at all.  Neither matched the corpus.  Kept only so
    an A/B gate can run both arms from one tree; nothing depends on it.
    """
    value = os.environ.get("PLOFA_TOKEN_SOURCE", "scene").strip().lower()
    return "legacy" if value == "legacy" else "scene"


def _temper_armed() -> bool:
    """Armed only by an explicit environment variable, like the shot read-out.

    WHY THIS SWITCH EXISTS, AND WHY IT IS OFF
    -----------------------------------------
    Until 2026-11-16 this module was **not on the live route**: the situational
    pipeline had been duplicated inside ``brain_integration._maybe_situational``
    and that copy is what every match actually ran.  The copy had no temperament
    seam.  So the reconnection described below was written, tested, documented —
    and never applied to a single player.

    Fixing the routing therefore has a side effect that is NOT a routing change:
    the moment the live path becomes this one, temperament starts being applied
    to every player who has a registered mind.  ``match_probe`` registers none,
    so a gated experiment would never notice; ``auto_run_match`` registers one
    per player, so a season run would change behaviour without anybody asking.

    One change at a time.  The routing fix must be provably behaviour-neutral,
    and temperament gets its own gate and its own measurement.  Set
    ``PLOFA_TEMPER=1`` to arm it.
    """
    return os.environ.get("PLOFA_TEMPER", "") == "1"


def _apply_temper(probs: Any, mind: Any, under_pressure: bool,
                  fatigue: float, record_trace: bool = False):
    """Distort the intent distribution by the player's temperament.

    ``mind.temper`` was called at exactly one place — ``cognition_brain.py`` —
    which every shipped situational brain bypasses.  So a player's temperament
    was computed from his scars, saved into the season state, carried into the
    next match, and then **never applied to the player who was actually
    playing**.  The shipped brains read the memory as 8 input floats and were
    never tempered by it.  This is the reconnection.

    Contract
    --------
    * no mind, or a disengaged mind → ``probs`` returned untouched, so the
      path is byte-identical to the un-tempered one;
    * a shape change, a non-finite value or a zero-sum vector is **rejected**
      rather than trusted: a mis-sized vector would silently mislabel every
      intent downstream, because ``chosen_idx`` indexes ``INTENT_LABELS``
      positionally.  A silent mislabel is far worse than an ignored scar.
    * ``temper`` is allowed to decline — it returns its input when the biases
      are neutral — so "no change" is a normal outcome, not a failure.

    Returns ``(probs, trace)``.  ``trace`` is ``None`` unless a change was
    actually applied and ``record_trace`` was asked for.
    """
    if mind is None or not getattr(mind, "engaged", False):
        return probs, None
    before = np.asarray(probs, dtype=np.float64)
    try:
        after = np.asarray(
            mind.temper(before, INTENT_LABELS, bool(under_pressure), fatigue),
            dtype=np.float64)
    except Exception:
        return probs, None
    if after.shape != before.shape:
        return probs, None
    if not np.all(np.isfinite(after)) or float(after.sum()) <= 0.0:
        return probs, None
    trace = None
    if record_trace:
        bias = getattr(getattr(mind, "experience", None), "bias", None) or {}
        trace = {
            "risk": round(float(bias.get("risk_in_box", 1.0)), 4),
            "confidence": round(float(bias.get("confidence", 1.0)), 4),
            # total variation distance: how far the scar moved the decision
            "shift": round(float(np.abs(after - before).sum()) / 2.0, 4),
        }
    return after, trace


# ─────────────────────────────────────────────────────────────
# THE SHOT READ-OUT — loaded per position, off unless armed
# ─────────────────────────────────────────────────────────────

_SHOT_HEAD_CACHE: Dict[str, Any] = {}
_SHOT_HEAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "brains", "shot_heads")


def _shot_head_armed() -> bool:
    """Armed only by an explicit environment variable.

    Deliberately not a config file and not a default.  The artifact was fitted
    offline against synthetic states; until a real-match gate has run, arming it
    has to be a conscious act by whoever is running the experiment, not
    something that happens because a file exists on disk.
    """
    return os.environ.get("PLOFA_SHOT_HEAD", "") == "1"


def _shot_head_for(player: Any):
    """The head for this player's position, or None.  Cached per position."""
    position = str(getattr(player, "position", "") or "").strip()
    stem = position.rstrip("0123456789").upper()
    if not stem:
        return None
    if stem in _SHOT_HEAD_CACHE:
        return _SHOT_HEAD_CACHE[stem]
    head = None
    path = os.path.join(_SHOT_HEAD_DIR, f"{stem}.json")
    if os.path.exists(path):
        try:
            from shot_head import ShotHead
            head = ShotHead.load(path)
        except Exception:
            head = None          # a corrupt artifact must not stop a match
    _SHOT_HEAD_CACHE[stem] = head
    return head


def _apply_shot_head(player: Any, logits: Any, hidden: Any):
    """Add the read-out's bias to the SHOOT logit, if a head exists and is armed."""
    if not _shot_head_armed():
        return logits
    head = _shot_head_for(player)
    if head is None:
        return logits
    try:
        return head.apply_logits(logits, hidden)
    except Exception:
        return logits            # never let a sidecar break a match


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

        # ── 1) WHAT HE CAN ACTUALLY SEE ───────────────────────────
        # ONE vision gate: `perceive` applies the role cone (130-200 deg),
        # the range scaling, the positional noise and the relevance top-k
        # (GK 12, CB 10, CM 8, ST 6) and hands back the lists it kept.
        #
        # THE CORPUS IS THE ARBITER.  `surrogate_collect._situational_blocks`
        # builds the tokens from `scene` — i.e. from those kept lists —
        # measured mask sums: GK up to 12, CB 10, CM 8, ST 6, mean 4.76 men a
        # look.  Until 2026-11-16 this function built them from whatever the
        # FOV gate left instead: a mind's 25 m / 2 + 2 frame, so at most four
        # men — and 35 % of those tokens sat OUTSIDE the cone.  With no mind
        # registered it fed the RAW lists, with no cone at all.  Neither
        # fiction is what the weights were distilled on, so the attention now
        # reads the same pitch the training run read.
        #
        # `PLOFA_TOKEN_SOURCE=legacy` restores the old feed, for one A/B gate.
        mind = _resolve_mind(player)
        legacy = _token_source() == "legacy"

        vis_teammates, vis_defenders = teammates, defenders
        if legacy and mind is not None and mind.engaged:
            try:
                vis_teammates, vis_defenders = mind.perceive(
                    player, x, y, teammates, defenders, position_engine)
            except Exception:
                vis_teammates, vis_defenders = teammates, defenders

        from brain_integration import _player_score_diff

        # Player-frame, not home-frame (gate 8).  `attacks_right` is team
        # identity in this engine, so an away carrier reads his OWN goal
        # difference instead of the scoreboard's.
        score_diff = _player_score_diff(game_state, attacks_right)
        base_vec, scene = perceive(
            player, x, y, vis_teammates, vis_defenders, position_engine,
            under_pressure, attacks_right, game_state, minute,
            team_possession=True, score_diff=score_diff,
            return_scene=True,
        )
        seen_teammates = scene.get("teammates", vis_teammates)
        seen_defenders = scene.get("defenders", vis_defenders)
        seen_engine = scene.get("position_engine", position_engine)

        # Legacy keeps its old feed (gated lists + the TRUE engine); the
        # shipped path uses the perceived scene, noisy positions included,
        # because that is the world the corpus recorded.
        if legacy:
            feed_t, feed_d, feed_e = vis_teammates, vis_defenders, position_engine
        else:
            feed_t, feed_d, feed_e = seen_teammates, seen_defenders, seen_engine

        # ── 2) TOKENS — the men he can see, and only them ─────────
        tokens, token_mask = entity_tokens(
            player, x, y, feed_t, feed_d, feed_e, attacks_right)

        # ── 3) THE FOUR BLOCKS ────────────────────────────────────
        role, memory, plan = situation_blocks(
            player, x, y, seen_teammates, seen_defenders, seen_engine,
            attacks_right, base_vec)
        # The brain's forward_logits will handle role width adaptation.
        # Do NOT pad/truncate here — the brain knows its own expected input size.
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

        # ── 4a) THE SHOT READ-OUT — the geometry the head cannot read ──
        #
        # Measured 2026-11-16 (`tools/diag/_shothead_probe.py`): the shipped ST
        # brain's 32-d hidden state CARRIES shot viability (a linear probe on it
        # reaches AUC 0.95) and the output layer reads it BACKWARDS (AUC 0.28,
        # spearman -0.51).  That is why a striker with 109 shooting intents took
        # none of them: the information was there and the head inverted it.
        #
        # `shot_head.ShotHead` is a 33-parameter read-out on that same hidden
        # state, added to the SHOOT logit and to nothing else.  Because no other
        # logit moves, this can only ever change a decision where SHOOT becomes
        # or stops being the argmax — it cannot reorder the other nine intents
        # against each other.  That is the whole safety argument, and it is why
        # this sits before the softmax rather than inside the network.
        #
        # OFF BY DEFAULT.  The artifact is fitted offline and each position is
        # gated on its own numbers; a position whose brain does not encode shot
        # viability well enough has NO file and is untouched.  Set
        # PLOFA_SHOT_HEAD=1 to arm it.  Nothing in `brains/` changes.
        logits = _apply_shot_head(player, logits, h_new)
        probs = _softmax(logits)

        # ── 4b) TEMPERAMENT — the scar, applied ───────────────────
        # WHY HERE, AND WHY THIS IS THE WHOLE POINT.
        # ``mind.temper`` was called at exactly one place, ``cognition_brain.py``,
        # which every shipped situational brain bypasses.  So a player's
        # temperament was computed from his scars, saved into the season state,
        # carried into the next match — and then never applied to the player who
        # was actually playing.  The shipped brains READ the memory as 8 input
        # floats and were never TEMPERED by it.  This is the reconnection.
        #
        # It sits BEFORE the coach bias and BEFORE the consequence critic, on
        # purpose.  Temperament is the player's own disposition; a coach's order
        # is an external multiplier on top of it, and an order should not be
        # able to erase a scar (a frightened player told to shoot is still more
        # reluctant than a confident one, but he does shoot more than he would
        # have).  Putting temper last would let it undo an explicit instruction.
        #
        # The consequence critic converts probs to log-space afterwards, so this
        # distortion is preserved exactly rather than diluted.
        fatigue = _fatigue_estimate(player, minute)
        # The switch lives HERE, at the call site, not inside `_apply_temper`.
        # `_apply_temper` is the arithmetic and must stay testable on its own;
        # whether the live path is allowed to apply it is a policy question, and
        # policy belongs where the decision is made.
        if _temper_armed():
            probs, temper_trace = _apply_temper(
                probs, mind, under_pressure, fatigue, record_trace)
        else:
            temper_trace = None

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

        cons_trace = None
        base24 = np.asarray(base_vec, dtype=np.float64)
        probs, cons_trace = _apply_consequence(
            probs, base24, getattr(player, "position", ""), record_trace,
        )

        temperature = _decision_temperature(player, under_pressure, fatigue)
        chosen_idx = _sample_from_probs(probs, temperature)
        chosen_prob = float(probs[chosen_idx])

        decision = _build_decision(
            brain, player, x, y, feed_t, feed_d, feed_e,
            attacks_right, fatigue, situation, probs,
            chosen_idx, chosen_prob, minute, record_trace, favored_flank,
            cons_trace,
        )
        if record_trace and decision.trace is not None:
            decision.trace["plan"] = plan_trace
            decision.trace["temperament"] = temper_trace
            decision.trace["situation"] = {
                "state": [round(float(v), 4) for v in base_vec],
                "role": [round(float(v), 4) for v in role],
                "memory": [round(float(v), 4) for v in memory],
                "plan": [round(float(v), 4) for v in plan],
                "tokens": int(token_mask.sum()),
            }
        return decision
