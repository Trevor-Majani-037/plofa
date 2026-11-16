"""The shot read-out: it must be inert by default, surgical when armed.

The mechanism was chosen BECAUSE it cannot touch the other nine intents.  That
claim is only worth anything if it is checked, so the central test here is not
"does the bias exist" but "is every non-SHOOT logit bit-identical".

The second theme is arming.  A sidecar file that silently changes match
behaviour because it happens to sit on disk is exactly the kind of invisible
coupling that produced a collapsed critic and a dead plan block.  Nothing here
may run unless `PLOFA_SHOT_HEAD=1`.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shot_head  # noqa: E402
from football_brain import INTENT_LABELS  # noqa: E402
from situational_brain import HIDDEN_D, OUTPUT_SIZE  # noqa: E402

SHOOT = INTENT_LABELS.index("SHOOT")


def _head(beta=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return shot_head.ShotHead("ST", rng.normal(0, 0.3, HIDDEN_D), 0.1, beta)


# ─────────────────────────────────────────────────────────────
# THE SAFETY ARGUMENT
# ─────────────────────────────────────────────────────────────

def test_only_the_shoot_logit_moves():
    """Every other logit must be bit-identical, not merely close."""
    head = _head()
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2.0, OUTPUT_SIZE)
    hidden = rng.normal(0, 1.0, HIDDEN_D)

    out = head.apply_logits(logits, hidden)

    assert out[SHOOT] == pytest.approx(
        logits[SHOOT] + head.bias(hidden), rel=0, abs=1e-12)
    others = [i for i in range(OUTPUT_SIZE) if i != SHOOT]
    assert np.array_equal(out[others], logits[others]), (
        "a non-SHOOT logit moved — the read-out is not surgical")


def test_the_bias_is_exactly_beta_times_the_score():
    head = _head(beta=3.0)
    rng = np.random.default_rng(2)
    hidden = rng.normal(0, 1.0, (7, HIDDEN_D))
    logits = np.zeros((7, OUTPUT_SIZE))

    out = head.apply_logits(logits, hidden)
    expected = 3.0 * (hidden @ head.w + head.b)
    assert np.allclose(out[:, SHOOT], expected, atol=1e-12)


def test_beta_zero_is_the_identity():
    head = _head(beta=0.0)
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 1.0, OUTPUT_SIZE)
    hidden = rng.normal(0, 1.0, HIDDEN_D)
    assert np.array_equal(head.apply_logits(logits, hidden), logits)


def test_the_bias_is_clamped_to_a_prior_and_never_a_veto():
    """A bias of -104 is a veto, not a prior.  Measured, then clamped.

    On the shipped ST brain at beta = 6 the raw bias ran from -103.7 to +15.4
    over real hidden states.  At -104 p_SHOOT is exactly 0.0 forever, which
    contradicts the whole design ("the brain may still decline a shot") and is
    brittle to any out-of-distribution hidden state.  The clamp is the guard,
    and it has to hold on absurd inputs too.
    """
    head = shot_head.ShotHead("ST", np.full(HIDDEN_D, 1000.0), 1000.0, 6.0)
    hidden = np.ones((5, HIDDEN_D))
    bias = head.bias(hidden)
    assert np.all(bias <= shot_head.BIAS_CLAMP)
    assert np.all(bias >= -shot_head.BIAS_CLAMP)

    negative = shot_head.ShotHead("ST", np.full(HIDDEN_D, -1000.0), -1000.0, 6.0)
    assert np.all(negative.bias(hidden) == -shot_head.BIAS_CLAMP)


def test_the_clamp_still_lets_a_stronger_intent_win():
    """Clamped, the head can still be outvoted — that is what makes it a prior."""
    head = shot_head.ShotHead("ST", np.zeros(HIDDEN_D), 1000.0, 6.0)
    hidden = np.zeros(HIDDEN_D)
    logits = np.zeros(OUTPUT_SIZE)
    logits[SHOOT] = -20.0                 # SHOOT is hopeless without the bias
    out = head.apply_logits(logits, hidden)
    assert out[SHOOT] == pytest.approx(-20.0 + shot_head.BIAS_CLAMP)
    assert out.argmax() != SHOOT, "the clamp turned a prior into a veto"


def test_a_positive_bias_can_only_ever_help_shoot_win():
    """Monotonicity: raising beta cannot lower p_SHOOT for a positive score.

    This is the property that makes the mechanism safe to sweep: the head is not
    allowed to be non-monotone in its own knob, because then "how much" would be
    a hidden second parameter.
    """
    rng = np.random.default_rng(4)
    logits = rng.normal(0, 1.0, (200, OUTPUT_SIZE))
    hidden = rng.normal(0, 1.0, (200, HIDDEN_D))
    score = hidden @ _head().w + _head().b
    keep = score > 0.5

    prev = None
    for beta in (0.5, 1.0, 2.0, 4.0):
        h = shot_head.ShotHead("ST", _head().w, _head().b, beta)
        lg = h.apply_logits(logits[keep], hidden[keep])
        lg -= lg.max(axis=1, keepdims=True)
        e = np.exp(lg)
        p = (e / e.sum(axis=1, keepdims=True))[:, SHOOT]
        if prev is not None:
            assert np.all(p >= prev - 1e-12), "p_SHOOT fell as beta rose"
        prev = p


# ─────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────

def test_round_trip_preserves_every_number(tmp_path):
    head = _head(beta=4.5)
    head.meta = {"holdout_auc": 0.9, "tier": "PROVISIONAL"}
    path = tmp_path / "ST.json"
    head.save(str(path))

    back = shot_head.ShotHead.load(str(path))
    assert np.array_equal(back.w, head.w)
    assert back.b == head.b
    assert back.beta == head.beta
    assert back.meta["tier"] == "PROVISIONAL"


def test_the_written_file_is_small_and_readable(tmp_path):
    """A sidecar nobody can inspect is a sidecar nobody will trust."""
    head = _head()
    path = tmp_path / "ST.json"
    head.save(str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert len(raw["w"]) == HIDDEN_D
    assert set(raw) >= {"position", "beta", "b", "w", "meta"}


# ─────────────────────────────────────────────────────────────
# THE ROLE GUARD
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("position", ["GK", "CB", "LB", "CM", "CDM"])
def test_roles_without_shoot_are_refused_without_training(position):
    """A keeper must not get a shot head, and must not pay for the attempt.

    The guard runs BEFORE the corpus is built, so a refusal is free.  Without it
    the fitter produced a PROVISIONAL head for the goalkeeper — arithmetically
    fine, football nonsense.
    """
    head, report = shot_head.fit(None, position, n_states=50, seed=1)
    assert head is None
    assert report["tier"] == "REFUSED"
    assert "no SHOOT in its vocabulary" in report["verdict"]


# ─────────────────────────────────────────────────────────────
# THE DECISION PATH
# ─────────────────────────────────────────────────────────────

def test_shot_head_is_inert_without_the_flag(monkeypatch):
    """Default behaviour is bit-identical to having no artifact at all."""
    import situational_decision as sd

    monkeypatch.delenv("PLOFA_SHOT_HEAD", raising=False)
    logits = np.array([0.1, 0.2, 0.3, 0.4, 5.0, 0.6, 0.7, 0.8, 0.9, 1.0])
    hidden = np.ones(HIDDEN_D)

    class P:
        position = "ST"

    assert np.array_equal(sd._apply_shot_head(P(), logits, hidden), logits)


def test_a_missing_artifact_means_no_change_even_when_armed(monkeypatch):
    import situational_decision as sd

    monkeypatch.setenv("PLOFA_SHOT_HEAD", "1")
    sd._SHOT_HEAD_CACHE.clear()

    class P:
        position = "NOPE"

    logits = np.arange(OUTPUT_SIZE, dtype=float)
    assert np.array_equal(
        sd._apply_shot_head(P(), logits, np.ones(HIDDEN_D)), logits)


def test_a_corrupt_artifact_cannot_break_a_match(monkeypatch, tmp_path):
    import situational_decision as sd

    monkeypatch.setenv("PLOFA_SHOT_HEAD", "1")
    monkeypatch.setattr(sd, "_SHOT_HEAD_DIR", str(tmp_path))
    (tmp_path / "ST.json").write_text("{ not json", encoding="utf-8")
    sd._SHOT_HEAD_CACHE.clear()

    class P:
        position = "ST"

    logits = np.arange(OUTPUT_SIZE, dtype=float)
    assert np.array_equal(
        sd._apply_shot_head(P(), logits, np.ones(HIDDEN_D)), logits)
