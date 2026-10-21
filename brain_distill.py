"""DISTILLATION — train the net to imitate the surrogate's OWN policy.

WHY THIS EXISTS
---------------
The genetic algorithm kept collapsing onto the best CONSTANT intent, even
after two real fixes:

* the per-position reward SCALE (the balance: the reward's swing is now
  4-20x the largest guard instead of 5-20x smaller), and
* the corpus COVERAGE (the striker's final-third shot cell went from n = 0
  to n = 16 at +0.0148 — seven times the next intent).

And the diagnosis is not "the objective is the ceiling": **the surrogate's
own optimum IS the broad, football-shaped policy the census wants.**  The
per-bucket argmax of the striker's table is

    CARRY 24.0%  PROTECT_POSSESSION 19.7%  RECYCLE 18.3%  DRIBBLE 11.3%
    SHOOT 11.0%  SAFE_PASS 4.7%  PROGRESSIVE_PASS 4.3%  SWITCH 4.3%
    THROUGH_BALL 2.3%

— nine intents, a 24% maximum, and the striker SHOOTING 11% of the time —
and it is worth **+0.44 to +0.51** (scaled) over the best constant, i.e.
**+0.22 to +0.25 of fitness**, which beats the whole guard budget (0.19).

**So the objective is right and the SEARCH is the blocker.** The constant is
a LOCAL optimum a bias tweak reaches in one step; the useful
state-dependence is a specific function (a threshold rule over six bucket
features) that a blind search from a random initialisation does not reach —
and a random net's state-dependence is useless (its argmax is random), so
selection kills the state-dependent members and keeps the constant.

**WHAT THIS DOES INSTEAD.** It trains the net to imitate the table's policy
directly: for each state, the target is the intent the table prefers in
that situation, and the net is fitted to predict it (cross-entropy, numpy,
Adam). The result is a brain whose intent mix IS the table's policy —
generalised over the PERCEIVED scene and the ROLE block, which the six
coarse buckets throw away. The genetic algorithm can then refine it from
that start, with the reward term dominant and the table's argmax as the
starting point rather than a random net.

WHAT THIS IS NOT
----------------
It is NOT a new objective: the target comes from the same surrogate, on the
same corpus, with the same reward. It is a different way to reach the
optimum the objective already defines. **The acceptance gate is unchanged:
pass maps and the eye, never fitness — and the census still decides whether
a brain set may ship.**

WHAT THIS FILE PINS (in `tests/test_brain_distill.py`)
-------------------------------------------------------
That the targets are the table's argmax over the ROLE's vocabulary (and
never a forbidden action), that the training reduces the loss and raises the
agreement with the table, that the distilled brain's intent MIX matches the
table's argmax mix (the census-passing property), and that the
distillation is deterministic.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain_evolution import generate_state_corpus
from decision_brain import PlayerIntent
from football_brain import FootballBrain
from role_features import ROLE_INTENTS, V2_INPUT_D, _role_family

#: The intents each role may perform — the training TARGETS come from here,
#: so a role is never trained towards an action it cannot perform.
def _vocabulary(position: str) -> List[PlayerIntent]:
    fam = _role_family(position) or ""
    names = ROLE_INTENTS.get(fam, frozenset())
    return [i for i in PlayerIntent if i.value in names]


def _targets(surrogate, sensors: np.ndarray, position: str) -> np.ndarray:
    """The table's preferred intent for each state, as an index."""
    vocab = _vocabulary(position)
    if not vocab:
        raise ValueError(f"no role vocabulary for position {position!r}")
    out = np.zeros(len(sensors), dtype=np.int64)
    for i, row in enumerate(sensors):
        best = None
        best_v = None
        for intent in vocab:
            v = float(surrogate.expected_success(row[:24], intent, position))
            if best_v is None or v > best_v:
                best, best_v = intent, v
        out[i] = PlayerIntent.PASS.value if best is None else 0
        out[i] = [p.value for p in PlayerIntent].index(best.value)
    return out


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def distill_brain(
    position: str,
    surrogate: Any,
    n_states: int = 2000,
    seed: int = 123,
    epochs: int = 120,
    learning_rate: float = 0.02,
    input_size: Optional[int] = None,
    perception_config: Optional[Any] = None,
    goal_bias: float = 0.0,
    verbose: bool = False,
) -> Tuple[FootballBrain, Dict[str, Any]]:
    """Fit one position's brain to the surrogate's own policy.

    Returns the brain and a report (the agreement with the table, the loss
    curve, and the intent mix of the result).
    """
    fam = _role_family(position) or ""
    width = input_size if input_size is not None else V2_INPUT_D.get(fam, 24)
    corpus = generate_state_corpus(position, n_states, seed, goal_bias,
                                  input_size=width,
                                  perception_config=perception_config)
    y = _targets(surrogate, corpus, position)

    brain = FootballBrain.random(seed=seed, input_size=width)
    params = [brain.w1, brain.b1, brain.w2, brain.b2, brain.w3, brain.b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss_curve = []
    agree_curve = []
    n = len(corpus)
    for epoch in range(epochs):
        logits = brain.forward_logits(corpus)
        logp = _log_softmax(logits)
        loss = float(-logp[np.arange(n), y].mean())
        probs = np.exp(logp)
        agree = float((probs.argmax(axis=1) == y).mean())
        loss_curve.append(loss)
        agree_curve.append(agree)

        # dL/dlogits = probs - onehot
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        # back through the softmax is folded into dlogits by the log-softmax
        # gradient above; now through the layers
        h1 = np.maximum(0.0, corpus @ brain.w1 + brain.b1)
        h2 = np.maximum(0.0, h1 @ brain.w2 + brain.b2)
        gw3 = h2.T @ dlogits
        gb3 = dlogits.sum(axis=0)
        dh2 = dlogits @ brain.w3.T
        dh2[h2 <= 0.0] = 0.0
        gw2 = h1.T @ dh2
        gb2 = dh2.sum(axis=0)
        dh1 = dh2 @ brain.w2.T
        dh1[h1 <= 0.0] = 0.0
        gw1 = corpus.T @ dh1
        gb1 = dh1.sum(axis=0)
        grads = [gw1, gb1, gw2, gb2, gw3, gb3]

        step = epoch + 1
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * (g * g)
            mhat = m[i] / (1 - b1 ** step)
            vhat = v[i] / (1 - b2 ** step)
            p -= learning_rate * mhat / (np.sqrt(vhat) + eps)
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:>4}  loss {loss:.4f}  agree {agree:.3f}")

    final = brain.forward_logits(corpus).argmax(axis=1)
    agree = float((final == y).mean())
    labels = [p.value for p in PlayerIntent]
    mix: Dict[str, int] = {}
    for idx in final:
        mix[labels[int(idx)]] = mix.get(labels[int(idx)], 0) + 1
    # The SCHEMA lineage, exactly as `brain_evolution.evolve` stamps it: a
    # brain that does not declare its role features is a brain whose extra
    # inputs are INERT — and `FootballBrain.save` refuses to write an arch
    # the validator cannot reconcile.
    if width != 24:
        from brain_schema import brain_meta_dict
        from role_features import MENU_NAMES, V2_SHARED_D
        fam_names = list(MENU_NAMES.get(fam, ()))
        tail_width = width - V2_SHARED_D
        brain.meta = brain_meta_dict(
            training_method="distilled_surrogate",
            sensor_schema="v2_role_features",
            role_family=fam,
            role_features=(fam_names[:max(0, tail_width)] or None),
        )
    report = {
        "position": position, "width": width, "states": n, "epochs": epochs,
        "seed": seed, "loss_first": loss_curve[0], "loss_last": loss_curve[-1],
        "agreement_first": agree_curve[0], "agreement": agree,
        "mix": {k: round(v / n, 4) for k, v in sorted(mix.items())},
        "target_mix": {k: round(v / n, 4) for k, v in sorted(
            {labels[int(i)]: int((y == i).sum()) for i in set(y.tolist())}
            .items())},
    }
    return brain, report


def distill_all(
    positions: List[str],
    surrogate: Any,
    out_dir: str,
    n_states: int = 2000,
    seed: int = 123,
    epochs: int = 120,
    perception_config: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Distil every position and write the brains."""
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    reports = {}
    for pos in positions:
        if verbose:
            print(f"  distilling {pos} ...")
        brain, report = distill_brain(pos, surrogate, n_states=n_states,
                                     seed=seed, epochs=epochs,
                                     perception_config=perception_config,
                                     verbose=verbose)
        path = os.path.join(out_dir, f"{pos}.json")
        brain.save(path)
        reports[pos] = report
        if verbose:
            top = sorted(report["mix"].items(), key=lambda kv: -kv[1])[:4]
            print(f"    {pos}: agreement {report['agreement']:.3f}  "
                  f"mix " + ", ".join(f"{k} {v:.0%}" for k, v in top))
    with open(os.path.join(out_dir, "_distill_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    return reports


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Distil the surrogate's policy "
                                           "into per-position brains.")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--out", default="brains_distilled")
    p.add_argument("--positions", default=None)
    p.add_argument("--states", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    from critic_surrogate import load_any_surrogate

    sur = load_any_surrogate(args.surrogate)
    positions = ([s.strip().upper() for s in args.positions.split(",")]
                 if args.positions else
                 ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW",
                  "ST", "CF"])
    from perception import PerceptionConfig

    distill_all(positions, sur, args.out, n_states=args.states,
                seed=args.seed, epochs=args.epochs,
                perception_config=PerceptionConfig(enabled=True,
                                                  role_blocks=True,
                                                  seed=args.seed))
