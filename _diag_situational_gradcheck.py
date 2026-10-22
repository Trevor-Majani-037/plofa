"""GRADIENT CHECK — is SituationalBrain.backward actually correct?

Run:  .venv\\Scripts\\python.exe _diag_situational_gradcheck.py [--tokens N] [--steps T]

WHY THIS FILE EXISTS
--------------------
``situational_brain.SituationalBrain.backward`` is a hand-written
backward pass through a GRU, an attention block and two heads, written in
numpy because the project has no gradient library.  A hand-written
backward pass that is WRONG does not crash — it produces a smaller loss
for the wrong reason, or it makes the loss go UP, and both look like "the
network is hard to train".

The only honest check is FINITE DIFFERENCES: perturb each parameter by
epsilon, recompute the loss, and compare the measured slope with the
analytic one.  They must agree.

It has already earned its keep twice:
  * the GRU's update gate had ``dh'/dz = n - h`` instead of ``h - n``;
  * the attention used the SHORT softmax Jacobian, which drops a term
    whenever the loss depends on alpha through a non-shift-invariant
    function (here: through the context vector).
Both produced a plausible-looking training run and a wrong answer.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from situational_brain import (
    EMBED_D, HIDDEN_D, MEMORY_D, OUT_D, PLAN_D, PLAN_OUT_D,
    ROLE_D, STATE_D, TOKEN_D, SituationalBrain,
)

EPS = 1e-6


def softmax_rows(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build(seed=7, n_steps=4, n_tok=3):
    rng = np.random.default_rng(seed)
    brain = SituationalBrain.random(seed=seed)
    tokens = rng.normal(size=(n_steps, n_tok, TOKEN_D))
    mask = np.ones((n_steps, n_tok))
    if n_tok > 1:
        mask[-1, -1] = 0.0        # a padded slot, so the mask is exercised
    state = rng.normal(size=(n_steps, STATE_D))
    role = rng.normal(size=(n_steps, ROLE_D))
    memory = rng.normal(size=(n_steps, MEMORY_D))
    plan = rng.normal(size=(n_steps, PLAN_D))
    h0 = np.zeros(HIDDEN_D)
    y = rng.integers(0, OUT_D, size=n_steps)
    py = rng.integers(0, PLAN_OUT_D, size=n_steps)
    return brain, tokens, mask, state, role, memory, plan, h0, y, py


def loss_of(brain, tokens, mask, state, role, memory, plan, h0, y, py):
    logits, plan_logits, _h, _c = brain.forward_sequence(
        tokens, mask, state, role, memory, plan, h0)
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    nll = -(z[np.arange(len(y)), y] - lse)
    zp = plan_logits - plan_logits.max(axis=1, keepdims=True)
    lpse = np.log(np.exp(zp).sum(axis=1))
    pnll = -(zp[np.arange(len(py)), py] - lpse)
    return float((nll + pnll).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    A = ap.parse_args()

    brain, tokens, mask, state, role, memory, plan, h0, y, py = build(
        seed=A.seed, n_steps=A.steps, n_tok=A.tokens)
    n_steps = A.steps

    logits, plan_logits, _h, caches = brain.forward_sequence(
        tokens, mask, state, role, memory, plan, h0)

    # The cross-entropy gradient w.r.t. the LOGITS is (probs - onehot), NOT
    # (logits - onehot).  Passing the raw logits is a probe bug that makes
    # every parameter look wrong at once.
    probs = softmax_rows(logits)
    dlogits = probs.copy()
    dlogits[np.arange(n_steps), y] -= 1.0
    pprobs = softmax_rows(plan_logits)
    dplan = pprobs.copy()
    dplan[np.arange(n_steps), py] -= 1.0
    grads = brain.backward(dlogits, dplan, caches)

    base = loss_of(brain, tokens, mask, state, role, memory, plan, h0, y, py)

    rng = np.random.default_rng(99)
    worst = 0.0
    worst_name = ""
    checked = 0
    per_tensor = {}
    for name in sorted(brain.w):
        arr = brain.w[name]
        flat = arr.reshape(-1)
        picks = rng.choice(len(flat), size=min(8, len(flat)), replace=False)
        tw = 0.0
        for idx in picks:
            original = float(flat[idx])
            flat[idx] = original + EPS
            lp = loss_of(brain, tokens, mask, state, role, memory, plan, h0, y, py)
            flat[idx] = original - EPS
            lm = loss_of(brain, tokens, mask, state, role, memory, plan, h0, y, py)
            flat[idx] = original
            numeric = (lp - lm) / (2.0 * EPS)
            analytic = float(grads[name].reshape(-1)[idx])
            denom = max(1e-8, abs(numeric) + abs(analytic))
            rel = abs(numeric - analytic) / denom
            checked += 1
            tw = max(tw, rel)
            if rel > worst:
                worst = rel
                worst_name = (f"{name}[{idx}] numeric={numeric:+.6e} "
                              f"analytic={analytic:+.6e}")
        per_tensor[name] = tw

    print(f"tokens={A.tokens} steps={A.steps} seed={A.seed}")
    print(f"loss = {base:.6f}")
    print(f"checked {checked} parameters across {len(brain.w)} tensors")
    print("\nworst relative error per tensor:")
    for name in sorted(per_tensor, key=lambda k: -per_tensor[k]):
        print(f"  {name:6s}  {per_tensor[name]:.3e}")
    print(f"\nworst overall = {worst:.3e}  ({worst_name})")
    if worst > 1e-4:
        print("\nBACKWARD IS WRONG")
        return 1
    print("\nBACKWARD MATCHES FINITE DIFFERENCES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
