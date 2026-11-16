"""The temperament seam — the scar reaches the pitch (2026-11-08).

THE BUG THIS FILE PINS
----------------------
``mind.temper()`` was fully built: it turns a player's accumulated scars into a
distorted intent distribution.  It was called at exactly one place,
``cognition_brain.py`` — the cognition path, which every shipped situational
brain bypasses because ``_maybe_situational`` returns first.

So the shipped brains had *half* the loop.  A player's temperament was computed
from his scars, saved into the season state, carried into the next match, read
as 8 input floats by the brain — and then never applied to the player who was
actually playing.  ``memory_tail`` let him *know* he had been burned; nothing
made him *behave* as though he had.

The reconnection is ``situational_decision._apply_temper``, called in
``decide()`` between the network's softmax and the coach/consequence seams.

WHY ONE TEST READS THE SOURCE
-----------------------------
Every test below would still pass if ``decide()`` stopped calling
``_apply_temper`` — the function would be perfect and unreachable, which is
precisely the failure being repaired.  ``test_the_live_path_calls_the_seam``
therefore asserts the call site exists, so the loop cannot come unwired again
in silence.
"""
import inspect

import numpy as np
import pytest

import situational_decision as sd
from situational_brain import INTENT_LABELS


class _Mind:
    """A mind whose only job is to return a given vector."""

    def __init__(self, out=None, engaged=True):
        self.engaged = engaged
        self._out = out
        self.calls = []

    def temper(self, probs, labels, under_pressure=False, fatigue=0.0):
        self.calls.append((labels, under_pressure, fatigue))
        if self._out is not None:
            return self._out
        return probs


def _probs():
    p = np.array([0.10, 0.10, 0.10, 0.05, 0.15, 0.10, 0.05, 0.15, 0.10, 0.10])
    return p / p.sum()


def _bias(risk=1.0, conf=1.0):
    """A REAL ``PlayerMind`` with its temperament forced to a given setting.

    Not a double: the point of these tests is that the production ``temper``
    — with its own risk/safe tables and its own renormalisation — is what
    reaches the decision.  A fake here would have tested the fake.
    """
    from cognition.mind import PlayerMind

    mind = PlayerMind()
    mind.experience.bias["risk_in_box"] = risk
    mind.experience.bias["confidence"] = conf
    return mind


# ── the identity cases: the path must stay byte-identical ────────────────

def test_no_mind_returns_the_vector_untouched():
    p = _probs()
    out, trace = sd._apply_temper(p, None, False, 0.0, record_trace=True)
    assert np.array_equal(out, p)
    assert trace is None


def test_disengaged_mind_is_ignored():
    p = _probs()
    mind = _Mind(out=np.full(10, 0.1), engaged=False)
    out, trace = sd._apply_temper(p, mind, False, 0.0, record_trace=True)
    assert np.array_equal(out, p)
    assert trace is None
    assert mind.calls == [], "a disengaged mind must not even be consulted"


def test_neutral_bias_leaves_the_distribution_alone():
    """``temper`` declines when the biases are neutral — that is not a failure."""
    p = _probs()
    out, _ = sd._apply_temper(p, _bias(1.0, 1.0), False, 0.0)
    assert np.allclose(out, p)


# ── the behaviour change: a scarred player really does differ ────────────

def test_fear_moves_mass_from_risky_to_safe_intents():
    """risk_in_box 0.4 is a player who has been burned in the box."""
    p = _probs()
    out, trace = sd._apply_temper(p, _bias(risk=0.4), False, 0.0,
                                  record_trace=True)

    i_shot = INTENT_LABELS.index("SHOOT")
    i_safe = INTENT_LABELS.index("SAFE_PASS")
    i_drib = INTENT_LABELS.index("DRIBBLE")

    assert out[i_shot] < p[i_shot], "a frightened player must shoot less"
    assert out[i_drib] < p[i_drib], "and dribble less"
    assert out[i_safe] > p[i_safe], "and lean into the safe pass"
    assert out.sum() == pytest.approx(1.0), "temper must renormalise"
    assert trace["risk"] == pytest.approx(0.4)
    assert trace["shift"] > 0.0, "a real distortion must be reported"


def test_confidence_sharpens_or_flattens():
    """confidence > 1 peaks the distribution; < 1 flattens it."""
    p = _probs()
    sharp, _ = sd._apply_temper(p, _bias(conf=2.0), False, 0.0)
    flat, _ = sd._apply_temper(p, _bias(conf=0.3), False, 0.0)

    def peak(q):
        return float(np.max(q))

    assert peak(sharp) > peak(p), "a confident player commits harder"
    assert peak(flat) < peak(p), "an indecisive player spreads it"
    assert sharp.sum() == pytest.approx(1.0)
    assert flat.sum() == pytest.approx(1.0)


def test_the_labels_handed_to_temper_are_the_logit_order():
    """A wrong label order would distort the wrong intents — silently."""
    mind = _Mind()
    sd._apply_temper(_probs(), mind, False, 0.0)
    labels = mind.calls[0][0]
    assert list(labels) == list(INTENT_LABELS)
    assert len(labels) == 10


# ── the guard rails: a bad vector must never reach ``chosen_idx`` ────────

def test_a_misshapen_vector_is_rejected_not_trusted():
    """``chosen_idx`` indexes INTENT_LABELS positionally: a short vector would
    mislabel every intent downstream, which is worse than an ignored scar."""
    p = _probs()
    out, trace = sd._apply_temper(p, _Mind(out=np.full(4, 0.25)), False, 0.0,
                                  record_trace=True)
    assert np.array_equal(out, p)
    assert trace is None


def test_a_non_finite_vector_is_rejected():
    p = _probs()
    bad = _probs()
    bad[0] = np.nan
    out, _ = sd._apply_temper(p, _Mind(out=bad), False, 0.0)
    assert np.array_equal(out, p)


def test_a_zero_sum_vector_is_rejected():
    p = _probs()
    out, _ = sd._apply_temper(p, _Mind(out=np.zeros(10)), False, 0.0)
    assert np.array_equal(out, p)


def test_a_raising_mind_cannot_kill_the_decision():
    class _Boom(_Mind):
        def temper(self, *a, **k):
            raise RuntimeError("cognition blew up")

    p = _probs()
    out, _ = sd._apply_temper(p, _Boom(), False, 0.0)
    assert np.array_equal(out, p)


# ── the call site itself: the whole point of the repair ──────────────────

def test_the_live_path_calls_the_seam():
    src = inspect.getsource(sd.SituationalDecisionBrain.decide)
    assert "_apply_temper(" in src, (
        "decide() no longer applies the temperament — the scar is unwired again"
    )
    assert "_resolve_mind(" in src, "decide() must resolve the mind once"


def test_temper_is_applied_before_the_coach_and_the_critic():
    """An order may bias a scar; it must not erase it.  And the critic works in
    log space, so it preserves whatever distortion it is handed."""
    src = inspect.getsource(sd.SituationalDecisionBrain.decide)
    i_temper = src.index("_apply_temper(")
    i_coach = src.index("_apply_coach_bias(")
    i_cons = src.index("_apply_consequence(")
    assert i_temper < i_coach < i_cons
