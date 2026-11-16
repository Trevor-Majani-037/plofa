"""A collapsed critic must never be able to ship.

THE BUG (found 2026-11-13)
--------------------------
``brains_trainer/critic_v5.json`` has been blended into every decision at
strength 0.3 since 2026-09-20.  Measured over the 12,297-decision corpus:

    first-layer units that never activate   30 of 32
    second-layer units that never activate  31 of 32
    Q range across every real state         0.0123, in a 1.30-wide band

So the "consequence correction" was a near-constant vector — an intent prior
wearing a neural network.  It nudged every decision the same way wherever it
happened on the pitch, and nothing in the build noticed for six weeks, because
**a dead network has an ordinary loss**: predicting the mean is exactly what
minimises weighted MSE once the gradient through every unit is zero.

That is why the guard is structural rather than loss-based.  These tests pin
each of the three checks against a real failure and a real success, so the
guard cannot be quietly weakened later.
"""
import numpy as np
import pytest

import critic_surrogate as cs
from football_brain import INTENT_LABELS

OUT = len(INTENT_LABELS)


def _dead_critic(means, stds):
    """A critic whose first layer can never activate — the v5 pathology.

    A large negative bias on every first-layer unit puts every pre-activation
    below zero for any input, so ReLU zeroes the layer and no gradient ever
    reaches it again.  This is what lr=1e-2 did to the real critic.
    """
    return cs.ValueCritic(
        np.zeros((cs.CRITIC_INPUT, cs.H1)), np.full(cs.H1, -1e6),
        np.zeros((cs.H1, cs.H2)), np.zeros(cs.H2),
        np.zeros((cs.H2, OUT)), np.zeros(OUT),
        means, stds,
    )


def _toy_data(n=400, seed=3):
    """States where the reward depends on a sensor — so a live net can learn.

    Two signals, both visible in the input: intent 0 pays more the closer the
    goal (slot 14 low), and every other intent pays more with more space
    (slot 5 high).  A network that only memorises the intent means cannot fit
    either, which is exactly what the prior comparison is designed to expose.
    """
    rng = np.random.default_rng(seed)
    S = rng.random((n, 24))
    I = rng.integers(0, OUT, n)
    y = np.where(I == 0, 1.20 - 0.60 * S[:, 14], 0.60 + 0.30 * S[:, 5])
    y = y + rng.normal(0, 0.01, n)
    w = np.ones(n)
    means = S.mean(axis=0)
    stds = S.std(axis=0) + 1e-9
    Z = (S - means) / stds
    X = np.concatenate([Z, np.zeros((n, 11))], axis=1)
    X[:, 24] = 1.0
    return X, I, y, w, means, stds


def test_an_all_dead_critic_fails_on_output_not_anatomy():
    """The v5 pathology end to end: no unit alive means no output range.

    Dead units are REPORTED as a warning rather than deciding the verdict —
    see ``test_many_dead_units_can_still_pass`` for the measured reason.  What
    actually condemns this critic is that a fully dark first layer produces a
    constant, and a constant cannot rank states.
    """
    X, I, y, w, means, stds = _toy_data()
    tr = np.arange(320)
    te = np.arange(320, 400)
    h = cs.critic_health(_dead_critic(means, stds), X, I, y, tr, te)
    assert h["dead_units_h1"] == cs.H1
    assert any("never activate" in w for w in h["warnings"])
    assert h["verdict"] == "FAIL"
    assert any("Q spans" in f for f in h["failures"])


def test_many_dead_units_can_still_pass():
    """A measured correction to this guard's first draft.

    The draft failed any critic with more than a quarter of a layer dark.  A
    2,000-state synthetic run then trained to holdout rank correlation 0.986
    while 23 of 32 second-layer units never activated — the surviving units
    carried the function.  Anatomy is a hint; performance is the verdict.  A
    guard that rejects a working critic is as harmful as one that ships a
    broken one, so this case is pinned.
    """
    X, I, y, w, means, stds = _toy_data(n=2000, seed=3)
    tr = np.arange(1500)
    te = np.arange(1500, 2000)
    c, _, _ = cs.train_critic_backprop(
        X[tr], I[tr], y[tr], w[tr], X[te], I[te], y[te], w[te],
        means, stds, epochs=150, lr=5e-3, seed=1, verbose=False)
    h = cs.critic_health(c, X, I, y, tr, te)
    assert h["dead_units_h2"] > cs.H2 // 4, "this case is only interesting if units died"
    assert h["verdict"] == "PASS", h["failures"]
    assert h["rho_holdout"] > 0.5


def test_flat_output_is_detected():
    """No dead units, but the output never moves — also a failure.

    This is the second half of the v5 pathology and the reason dead-unit
    counting alone is not enough: a net can keep every unit alive and still
    emit a constant if its output weights are tiny.
    """
    X, I, y, w, means, stds = _toy_data()
    c = cs.ValueCritic(
        np.ones((cs.CRITIC_INPUT, cs.H1)) * 0.01, np.zeros(cs.H1),
        np.ones((cs.H1, cs.H2)) * 0.01, np.zeros(cs.H2),
        np.zeros((cs.H2, OUT)), np.full(OUT, 0.5),
        means, stds,
    )
    h = cs.critic_health(c, X, I, y, np.arange(320), np.arange(320, 400))
    assert h["q_spread"] == pytest.approx(0.0, abs=1e-9)
    assert h["verdict"] == "FAIL"
    assert any("Q spans" in f for f in h["failures"])


def test_a_net_that_only_learns_the_intent_prior_is_rejected():
    """Beating the prior is the point of conditioning on the state.

    A critic that reproduces the intent means has learned nothing a 10-number
    table could not hold.  If it does not beat that table on held-out data,
    the network is decoration and the build must say so.
    """
    X, I, y, w, means, stds = _toy_data(seed=11)
    # Intent-mean predictor expressed as a critic: output weights pick up the
    # per-intent constant, inputs are ignored entirely.  The constants come from
    # TRAIN ONLY — the same information the guard's own baseline is allowed.
    # Fitting them on holdout too would leak, and a leaky baseline is how a
    # decorative critic would sneak past the check.
    prior = np.array([y[:320][I[:320] == k].mean() if (I[:320] == k).any()
                      else y[:320].mean() for k in range(OUT)])
    c = cs.ValueCritic(
        np.zeros((cs.CRITIC_INPUT, cs.H1)), np.ones(cs.H1),
        np.zeros((cs.H1, cs.H2)), np.ones(cs.H2),
        np.zeros((cs.H2, OUT)), prior.copy(),
        means, stds,
    )
    h = cs.critic_health(c, X, I, y, np.arange(320), np.arange(320, 400))
    assert h["rho_delta"] <= 0.0
    assert h["verdict"] == "FAIL"


def test_a_real_training_run_passes():
    """The whole point: at lr=5e-3 the recipe that died at 1e-2 is alive."""
    X, I, y, w, means, stds = _toy_data(n=800, seed=5)
    tr = np.arange(600)
    te = np.arange(600, 800)
    c, _, _ = cs.train_critic_backprop(
        X[tr], I[tr], y[tr], w[tr], X[te], I[te], y[te], w[te],
        means, stds, epochs=150, lr=5e-3, seed=1, verbose=False)
    h = cs.critic_health(c, X, I, y, tr, te)
    assert h["q_spread"] > 0.25 * (cs._REWARD_HI - cs._REWARD_LO)
    assert h["verdict"] == "PASS", h["failures"]


def test_build_refuses_to_write_a_collapsed_critic(tmp_path, monkeypatch):
    """The guard must stop the FILE, not just print a warning.

    The health computation itself is covered by the tests above; this one is
    about the build refusing to proceed.  Forcing a real collapse here would
    test the optimiser again and would make the test fragile to learning-rate
    changes, so the verdict is stubbed and the refusal path is what is pinned.
    """
    rng = np.random.default_rng(2)
    records = [{
        "sensors": list(rng.random(24)),
        "intent": INTENT_LABELS[int(rng.integers(0, OUT))],
        "position": "CM",
        "goals": 0, "xg": 0.0, "shots": 0, "turnovers": 0,
    } for _ in range(200)]

    monkeypatch.setattr(cs, "critic_health", lambda *a, **k: {
        "verdict": "FAIL", "failures": ["stubbed collapse"], "warnings": [],
        "dead_units_h1": 32, "dead_units_h2": 32, "q_spread": 0.0,
        "rho_holdout": 0.0, "rho_prior_holdout": 0.0, "rho_delta": 0.0,
    })
    out = tmp_path / "critic.json"
    with pytest.raises(SystemExit):
        cs.build_critic(records, str(out), str(tmp_path / "r.json"),
                        method="backprop", epochs=5, target_mode="mc", seed=1)
    assert not out.exists(), "a FAILED health check must not leave a file behind"


def test_shipped_v5_is_a_known_bad_artifact():
    """Regression anchor: the artifact that motivated the guard still fails it.

    If this ever starts passing, either the file was replaced (then update the
    test deliberately) or the guard was weakened (then fix the guard).
    """
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "brains_trainer", "critic_v5.json")
    if not os.path.exists(path):
        pytest.skip("critic_v5.json not present")
    recs = cs._load_dataset(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "brains_trainer", "corpus_consequences_20260920_pregoals.json"))
    if not recs:
        pytest.skip("no consequence dataset")
    X, I, y, w, means, stds = cs._prepare(recs)
    n = len(recs)
    hold = max(1, int(n * 0.2))
    order = np.arange(n)
    np.random.default_rng(123 + 999).shuffle(order)
    v5 = cs.ValueCritic.load(path)
    h = cs.critic_health(v5, X, I, y, order[hold:], order[:hold])
    assert h["verdict"] == "FAIL"
    assert h["dead_units_h1"] > cs.H1 // 2
