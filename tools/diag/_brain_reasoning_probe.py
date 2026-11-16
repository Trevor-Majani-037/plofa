"""Does the brain reason, or does it memorise? Two measurements.

TEST A — CONDITIONAL ENTROPY, COARSE vs FINE
--------------------------------------------
The observatory reported "context-sensitive but noisy" at 41% predictability.
That number came from binning the situation into a 6-bit bucket
(P,FT,OH,S,C,G) and asking how often the modal intent inside a bucket was
chosen.  A deterministic function of a 24-d continuous input WILL look
unpredictable under a 6-bit key: the bin is coarser than the function.

So estimate H(intent | state) twice on the same samples:

    coarse   the 6-bit bucket the surrogate uses
    fine     k-nearest-neighbour entropy in the full standardised sensor
             space: for each sample, the intent distribution among its k
             nearest neighbours, entropy of that, averaged

Both are biased by finite samples (the fine one badly), so neither is read
in absolute terms.  Each is compared against its OWN shuffle null — intents
permuted across states — which has the same bias.  The difference
(I = H(I) - H(I|S)) is then interpretable, and the coarse/fine contrast is
the actual question.

TEST B — CAN IT EXTRAPOLATE INTO A SITUATION IT HAS NOT SEEN?
-------------------------------------------------------------
A policy that reasons is a FUNCTION of the situation: it interpolates and
extrapolates smoothly, so a held-out region of the pitch is predictable
from its neighbours.  A policy that memorises is a lookup: it has nothing
to say where it has no neighbours.

Two splits, same model:

    random   80/20 at random          -> the in-distribution ceiling
    spatial  train OUTSIDE a region,  -> extrapolation into unseen space
             test  INSIDE it

Reported against the majority-class baseline of the test set.  If spatial
accuracy collapses to the baseline while random holds up, the policy is
local, not a function of football.

WHAT THIS CAN AND CANNOT SHOW
-----------------------------
Failing the spatial test RULES OUT reasoning.  Passing it does not prove
reasoning: a smooth interpolator is not a reasoner.  The test is a
falsifier, and it is reported as one.

CAVEAT ON THE DATA
------------------
Reads `brains_trainer/corpus.raw.json`, which predates the 2026-10-31 goal
vision: slots 2-3 are duplicates of 0-1 there, so goal bearing and mouth
width are absent.  That makes Test A CONSERVATIVE (less information ->
higher measured conditional entropy) and Test B's features limited to the
ones that survive.  Re-run on a fresh corpus when one exists.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

CORPUS = "brains_trainer/corpus.raw.json"
OUT = "artifacts/brain_reasoning_probe.json"

#: Bucket the surrogate actually keys on (surrogate_collect.FitnessSurrogate.bucket)
def coarse_key(s: np.ndarray) -> str:
    return (f"P{int(s[16] > 0.5)}FT{int(s[12] > 0.5)}OH{int(s[13] > 0.5)}"
            f"S{int(s[9] > 0.4)}C{int(s[15] > 0.5)}G{int(s[14] < 0.5)}")


def entropy(counts) -> float:
    """Shannon entropy in bits of a count vector."""
    c = np.asarray([x for x in counts if x > 0], dtype=np.float64)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def cond_entropy_coarse(sensors: np.ndarray, intents: np.ndarray) -> float:
    """E[ H(intent | 6-bit bucket) ] weighted by bucket mass."""
    by: dict[str, list] = {}
    for s, i in zip(sensors, intents):
        by.setdefault(coarse_key(s), []).append(i)
    n = len(intents)
    total = 0.0
    for rows in by.values():
        total += (len(rows) / n) * entropy(Counter(rows).values())
    return total


def knn_indices(sensors: np.ndarray, k: int = 10) -> np.ndarray:
    """The k nearest neighbours of every row, self excluded, as indices.

    Computed ONCE and reused by every shuffle null: the neighbour structure
    is a property of the states, not of the labels, so permuting intents does
    not change who is near whom.  Recomputing it per permutation is 20x the
    work for the same answer.
    """
    X = np.asarray(sensors, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    sq = (Z * Z).sum(1)
    d = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.fill_diagonal(d, np.inf)
    return np.argpartition(d, k, axis=1)[:, :k]


def cond_entropy_from_idx(idx: np.ndarray, intents: np.ndarray) -> float:
    """E[ H(intent | k nearest neighbours) ] given a precomputed neighbour map."""
    _uniq, codes = np.unique(intents, return_inverse=True)
    codes = codes.astype(np.int64)
    C = int(codes.max()) + 1
    neigh = codes[idx]                       # (n, k)
    total = 0.0
    for row in neigh:
        cnt = np.bincount(row, minlength=C).astype(np.float64) + 1e-9
        p = cnt / cnt.sum()
        total += float(-(p * np.log2(p)).sum())
    return total / idx.shape[0]


def information(sensors, intents, k=10, coarse=True, seed=0, n_null=20) -> dict:
    """I(situation; intent) with its own shuffle null, for one estimator."""
    intents = np.asarray(intents)
    h_i = entropy(Counter(intents).values())
    if coarse:
        h_given = cond_entropy_coarse(sensors, intents)
        est = cond_entropy_coarse
    else:
        idx = knn_indices(sensors, k)
        h_given = cond_entropy_from_idx(idx, intents)
        est = lambda s, i: cond_entropy_from_idx(idx, i)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_null):
        perm = rng.permutation(intents)
        nulls.append(h_i - est(sensors, perm))
    nulls = np.asarray(nulls)
    return {
        "H_intent": round(h_i, 4),
        "H_given": round(h_given, 4),
        "I": round(h_i - h_given, 4),
        "null_mean": round(float(nulls.mean()), 4),
        "null_p95": round(float(np.percentile(nulls, 95)), 4),
        "p": round(float((nulls >= (h_i - h_given)).mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────
# TEST B — spatial extrapolation
# ─────────────────────────────────────────────────────────────────

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_logreg(X, y, C, iters=600, lr=0.5, l2=1e-3):
    """Multinomial logistic regression, plain gradient descent.

    Deliberately the simplest smooth model available: few parameters, strong
    inductive bias toward a global function.  If a smooth model cannot carry
    the policy across a region boundary, the policy is not smooth there.
    """
    n, d = X.shape
    W = np.zeros((d, C))
    b = np.zeros(C)
    Y = np.zeros((n, C))
    Y[np.arange(n), y] = 1.0
    for _ in range(iters):
        P = softmax(X @ W + b)
        G = (P - Y) / n
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def predict(X, W, b):
    return softmax(X @ W + b).argmax(1)


def standardise(train, test):
    mu, sd = train.mean(0), train.std(0)
    sd[sd < 1e-9] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def run_split(X, y, tr, te, C):
    Xtr, Xte = standardise(X[tr], X[te])
    W, b = fit_logreg(Xtr, y[tr], C)
    pred = predict(Xte, W, b)
    acc = float((pred == y[te]).mean())
    maj = Counter(y[tr]).most_common(1)[0][0]
    base = float((y[te] == maj).mean())
    return round(acc, 4), round(base, 4), len(tr), len(te)


def test_b(sensors, intents, positions):
    """Random 80/20 ceiling vs spatial holdout, per position."""
    results = {}
    regions = {
        "box_zone14": lambda x, y: np.logical_and(np.logical_and(x >= 70, x <= 90),
                                                  np.logical_and(y >= 24, y <= 44)),
        "def_third":  lambda x, y: x < 35,
        "wide_right": lambda x, y: np.logical_and(y > 48, x > 52),
    }
    for pos in sorted(set(positions)):
        m = positions == pos
        S, I = sensors[m], intents[m]
        if len(S) < 150:
            continue
        x, y = S[:, 0] * 105.0, S[:, 1] * 68.0
        feats = np.column_stack([x, y, S[:, 16], S[:, 4], S[:, 5], S[:, 6], S[:, 9]])
        _uniq, codes = np.unique(I, return_inverse=True)
        codes = codes.astype(np.int64)
        C = int(codes.max()) + 1
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(I))
        cut = int(0.8 * len(I))
        r_acc, r_base, _, _ = run_split(feats, codes, perm[:cut], perm[cut:], C)
        row = {"n": int(len(I)), "random_acc": r_acc, "random_base": r_base}
        for name, fn in regions.items():
            inside = fn(x, y)
            if inside.sum() < 40 or (~inside).sum() < 150:
                row[name] = None
                continue
            tr = np.where(~inside)[0]
            te = np.where(inside)[0]
            acc, base, ntr, nte = run_split(feats, codes, tr, te, C)
            row[name] = {"acc": acc, "base": base, "n_train": ntr, "n_test": nte}
        results[pos] = row
    return results


def main():
    data = json.load(open(CORPUS))
    sensors = np.array([r[0] for r in data], dtype=np.float64)
    intents = np.array([r[1] for r in data])
    positions = np.array([r[4] for r in data])

    # A nearest-neighbour entropy estimate is INVALIDATED by duplicate rows:
    # an exact copy of a state is its own nearest neighbour, contributing an
    # entropy of 0 and dragging the mean down.  Measured here: 732 of 4,933
    # rows are exact duplicates (max 2 copies), so dedupe before estimating.
    _, first = np.unique(np.round(sensors, 6), axis=0, return_index=True)
    first = np.sort(first)
    n_dupes = len(data) - len(first)
    sensors, intents, positions = sensors[first], intents[first], positions[first]
    print(f"corpus {len(data)} rows -> {len(sensors)} distinct states "
          f"({n_dupes} exact duplicates dropped)")
    print(f"{len(set(intents))} intents, {len(set(positions))} positions\n")

    print("TEST A — I(situation; intent), coarse key vs full sensor space")
    print(f"{'estimator':<10} {'H(I)':>7} {'H(I|S)':>8} {'I':>7} "
          f"{'null':>7} {'p95':>7} {'p':>6}")
    out = {"test_a": {}, "test_b": {}}
    for label, coarse in (("coarse", True), ("fine_k10", False)):
        r = information(sensors, intents, coarse=coarse)
        out["test_a"][label] = r
        print(f"{label:<10} {r['H_intent']:>7.3f} {r['H_given']:>8.3f} "
              f"{r['I']:>7.3f} {r['null_mean']:>7.3f} {r['null_p95']:>7.3f} "
              f"{r['p']:>6.2f}")

    print("\nTEST B — random 80/20 vs spatial holdout (logistic regression)")
    print(f"{'pos':<5} {'n':>5} {'random':>7} {'base':>6}   "
          f"{'zone14':>7} {'base':>6}   {'def3rd':>7} {'base':>6}")
    tb = test_b(sensors, intents, positions)
    out["test_b"] = tb
    for pos, r in sorted(tb.items()):
        def cell(k):
            v = r.get(k)
            return ("     -", "     -") if v is None else (f"{v['acc']:>7.3f}", f"{v['base']:>6.3f}")
        z = cell("box_zone14")
        d = cell("def_third")
        print(f"{pos:<5} {r['n']:>5} {r['random_acc']:>7.3f} "
              f"{r['random_base']:>6.3f}   {z[0]} {z[1]}   {d[0]} {d[1]}")

    import os
    os.makedirs("artifacts", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
