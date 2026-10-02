"""Manager Brains — Phase 7 evolution tests."""
import math
import warnings

import numpy as np
import pytest

from manager_brain import ManagerBrain, MANAGER_POSTURE_LABELS
from manager_evolution import synthetic_fitness, evolve
from manager_surrogate import ManagerSurrogate, bucket


def _sensors(score=0.0, minute=0.5, fatigue=0.3, conf=0.7,
             momentum=0.5, recent=0.0) -> np.ndarray:
    s = np.zeros(12, dtype=np.float64)
    s[0] = score
    s[1] = minute
    s[2] = 1.0
    s[3] = 0.5
    s[4] = recent
    s[5] = fatigue
    s[6] = conf
    s[7] = momentum
    s[8] = 0.5
    s[9] = 0.4
    s[10] = 0.5
    s[11] = 0.9
    return s


def _sample_rows():
    rows = [
        (_sensors(score=0.6, minute=0.5, fatigue=0.3, conf=0.7,
                  momentum=0.8, recent=0.0), "ATTACK", 0.9),
        (_sensors(score=0.6, minute=0.5, fatigue=0.3, conf=0.7,
                  momentum=0.8, recent=0.0), "ATTACK", 0.7),
        (_sensors(score=0.6, minute=0.5, fatigue=0.3, conf=0.7,
                  momentum=0.8, recent=0.0), "DEFEND", 0.4),
        (_sensors(score=-0.6, minute=0.9), "ATTACK", 0.6),
        (_sensors(score=0.0, minute=0.1), "BALANCED", 0.5),
    ]
    return [
        (sens, posture, points, "H:1-0")
        for sens, posture, points in rows
    ]


def test_surrogate_fit_and_lookup():
    rows = _sample_rows()
    sur = ManagerSurrogate()
    sur.fit(rows)

    sA = _sensors(score=0.6, minute=0.5, fatigue=0.3, conf=0.7,
                  momentum=0.8, recent=0.0)
    # W.3.+.f.C.e  -> (ATTACK, 0.9/0.7), (DEFEND, 0.4)
    assert bucket(sA) == "W.3.+.f.C.e"
    assert sur.support(sA, "ATTACK") == 2
    assert sur.support(sA, "DEFEND") == 1
    assert sur.expected_points(sA, "ATTACK") == pytest.approx(0.8)
    assert sur.expected_points(sA, "DEFEND") == pytest.approx(0.4)

    # Unseen posture in the bucket -> posture-only fallback -> 0.5 when
    # the posture itself was never seen anywhere.
    unseen = _sensors(score=-0.6, minute=0.9)  # L.4.-.f.C.e  unseen bucket
    assert sur.support(unseen, "BALANCED") == 0
    assert sur.expected_points(unseen, "BALANCED") == pytest.approx(0.5)

    # round-trip through JSON
    path = __import__("tempfile").mkdtemp()
    fp = path + "/surrogate.json"
    sur.save(fp)
    sur2 = ManagerSurrogate.load(fp)
    assert sur2.support(sA, "ATTACK") == 2
    assert sur2.expected_points(sA, "ATTACK") == pytest.approx(0.8)


def test_fitness_batch_matches_loop():
    rows = _sample_rows()
    sur = ManagerSurrogate()
    sur.fit(rows)
    brain = ManagerBrain.random(seed=11)

    vec = synthetic_fitness(brain, rows, surrogate=sur)

    # Reference loop: identical formula, single-sample forward passes.
    n = len(rows)
    rewards = []
    confidences = []
    counts = np.zeros(3)
    penalties = 0.0
    for sens, _, _points, _outcome in rows:
        p = brain.forward(np.asarray(sens, dtype=np.float64))["posture_probs"]
        idx = int(np.argmax(p))
        posture = MANAGER_POSTURE_LABELS[idx]
        counts[idx] += 1.0
        rewards.append(sur.expected_points(sens, posture))
        confidences.append(p[idx])
        sp = np.sort(p)[::-1]
        if len(sp) > 1 and (sp[0] - sp[1]) < 0.02:
            penalties += 0.1
    mr = float(np.mean(rewards))
    mc = float(np.mean(confidences))
    fit = mr * (0.5 + 0.5 * mc) - penalties / n
    shares = counts / counts.sum()
    max_share = float(shares.max())
    dom_pen = max(0.0, max_share - 0.60) * 1.5
    pos = shares[shares > 0]
    entropy = -float(np.sum(pos * np.log(pos + 1e-12)))
    bdiv = entropy / math.log(3)
    expected = max(0.0, min(1.0, (fit - dom_pen) * (1.0 + 0.5 * bdiv) / 1.5))

    assert vec == pytest.approx(expected, abs=1e-9)


def test_evolve_reduces_loss():
    rows = _sample_rows()
    surrogate = ManagerSurrogate()
    surrogate.fit(rows)

    baseline = synthetic_fitness(ManagerBrain.random(seed=1), rows, surrogate=surrogate)

    res = evolve(samples=rows, population_size=16, generations=8,
                 surrogate=surrogate, seed=7, verbose=False)
    assert len(res.generation) == 8
    assert len(res.mean) == 8

    # The tracked best beats a random (unevolved) individual on the SAME
    # samples and never degrades below the synthetic 0 floor.
    assert res.best_fitness >= baseline - 1e-9
    assert 0.0 <= res.best_fitness <= 1.0


def test_bucket_encoding_bounds():
    for _ in range(50):
        s = np.random.uniform(0.0, 1.0, 12)
        s[0] = np.random.uniform(-1.0, 1.0)
        s[4] = np.random.uniform(-1.0, 1.0)
        k = bucket(s)
        parts = k.split(".")
        assert parts[0] in ("W", "L", "D")
        assert 0 <= int(parts[1]) <= 5
        assert parts[2] in ("+", "-")
        assert parts[3] in ("F", "f")
        assert parts[4] in ("c", "C")
        assert parts[5] in ("E", "e")