"""WHERE DOES THE FITNESS COME FROM — the reward, or the GUARDS?

This replaces the stale `_diag_fitness_parts.py`, which reproduces the OLD
formula (`base * (0.5 + 0.5 * mean_conf)`) and no longer matches
`synthetic_fitness`.

THE CURRENT FORMULA
-------------------
    base    = 0.5 + 0.5 * clip(mean_reward, -1, 1)
    fitness = base
              + CONFIDENCE_GUARD_W * (mean_conf - 0.5)
              + DIVERSITY_GUARD_W * bdiv
              - penalties / n_states

Every guard has a FIXED weight; the reward's weight comes out of the
position's table.  The question this probe answers is the one that decides
a retrain: **did the GA take the reward, or did the guards decide?**

It reports, for a brain set:

    mean_reward / mean_conf / bdiv / penalty / base / fitness
        the decomposition of the brain's own fitness
    the BEST CONSTANT
        the fitness a brain that always played the single best-average
        intent would get, with the same confidence and the same penalty
    the ORACLE
        the fitness the state-dependent optimum would get, with the same
        guards

and then the verdict: if the collapsed brain's fitness BEATS the oracle's,
the guards decided and the reward was never the objective.  The probe
VERIFIES its own arithmetic against the real `synthetic_fitness` — a probe
that re-implements the thing it measures is evidence about nothing unless
the replication is checked.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_fitness_parts2.py \\
        brains_smoke_v6st_g80 brains/surrogate_pos_v6.json
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from brain_evolution import (  # noqa: E402
    CONFIDENCE_GUARD_W, DIVERSITY_GUARD_W, generate_state_corpus,
    synthetic_fitness,
)
from decision_brain import PlayerIntent  # noqa: E402
from football_brain import FootballBrain  # noqa: E402
from role_features import ROLE_INTENTS, V2_INPUT_D, _role_family  # noqa: E402
from surrogate_collect import FitnessSurrogate  # noqa: E402

POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]
N_STATES = 300
SEED = 123


def _corpus(pos):
    """The GA's own corpus: role width, perception on, seed 123."""
    from perception import PerceptionConfig
    fam = _role_family(pos) or ""
    width = V2_INPUT_D.get(fam, 24)
    cfg = PerceptionConfig(enabled=True, role_blocks=True, seed=SEED)
    return generate_state_corpus(pos, N_STATES, SEED, 0.0, input_size=width,
                                 perception_config=cfg)


def decompose(brain, position, corpus, surrogate):
    """Reproduce `synthetic_fitness` and return its parts."""
    from brain_evolution import _INTENT_BY_IDX
    rewards, confs = [], []
    counts = np.zeros(10, dtype=np.float64)
    penalties = 0.0
    for row in corpus:
        probs = brain.forward(row)
        idx = int(np.argmax(probs))
        counts[idx] += 1.0
        rewards.append(surrogate.expected_success(row[:24],
                                                  _INTENT_BY_IDX[idx], position))
        confs.append(float(probs[idx]))
        sp = np.sort(probs)[::-1]
        if len(sp) > 1 and (sp[0] - sp[1]) < 0.02:
            penalties += 0.1
    mean_reward = float(np.mean(rewards))
    mean_conf = float(np.mean(confs))
    shares = counts / counts.sum()
    shares = shares[shares > 0]
    entropy = -float(np.sum(shares * np.log(shares + 1e-12)))
    bdiv = entropy / math.log(10)
    base = 0.5 + 0.5 * max(-1.0, min(1.0, mean_reward))
    penalty_rate = penalties / len(corpus)
    fitness = (base + CONFIDENCE_GUARD_W * (mean_conf - 0.5)
               + DIVERSITY_GUARD_W * bdiv - penalty_rate)
    return {
        "mean_reward": mean_reward, "mean_conf": mean_conf, "bdiv": bdiv,
        "penalty": penalty_rate, "base": base, "fitness": fitness,
        "intents": int((counts > 0).sum()),
        "top_share": float(counts.max() / counts.sum()),
    }


def _table_extremes(sur, corpus, position):
    """The best constant and the per-state oracle over the corpus."""
    fam = _role_family(position) or ""
    allowed = [i for i in PlayerIntent if i.value in ROLE_INTENTS.get(fam, ())]
    sums = {i.value: 0.0 for i in allowed}
    oracle = 0.0
    n = 0
    for row in corpus:
        vals = []
        for intent in allowed:
            v = float(sur.expected_success(row[:24], intent, position))
            sums[intent.value] += v
            vals.append(v)
        if vals:
            oracle += max(vals)
            n += 1
    n = max(1, n)
    best_const = max(sums.values()) / n
    oracle_mean = oracle / n
    return best_const, oracle_mean, sums


def main():
    brains_dir = sys.argv[1] if len(sys.argv) > 1 else "brains_v6"
    sur_path = sys.argv[2] if len(sys.argv) > 2 else "brains/surrogate_pos_v6.json"
    args_positions = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    sur = FitnessSurrogate.load(sur_path)
    print(f"brains:    {brains_dir}")
    print(f"surrogate: {sur_path}")
    print(f"corpus:    the GA's own ({N_STATES} states, seed {SEED}, role width, "
          f"perception on)")
    print()
    print(f"{'pos':<5}{'intents':>8}{'top%':>7}{'reward':>10}{'conf':>7}"
          f"{'bdiv':>7}{'pen':>7}{'base':>9}{'FITNESS':>10}"
          f"{'const F':>10}{'oracle F':>10}{'verdict':>12}")
    print("-" * 110)
    for pos in POSITIONS:
        if args_positions and pos not in args_positions:
            continue
        path = os.path.join(brains_dir, f"{pos}.json")
        if not os.path.exists(path):
            print(f"{pos:<5} (missing)")
            continue
        corpus = _corpus(pos)
        brain = FootballBrain.load(path)
        try:
            d = decompose(brain, pos, corpus, sur)
            real = synthetic_fitness(brain, pos, batched_sensors=corpus,
                                      surrogate=sur)
        except ValueError as exc:
            print(f"{pos:<5} (width mismatch: {exc})")
            continue
        const, oracle, _sums = _table_extremes(sur, corpus, pos)
        # the constant and the oracle carry the SAME guards as the brain, so
        # the comparison is about the reward term alone
        const_f = (0.5 + 0.5 * max(-1.0, min(1.0, const))
                   + CONFIDENCE_GUARD_W * (d["mean_conf"] - 0.5)
                   + DIVERSITY_GUARD_W * d["bdiv"] - d["penalty"])
        oracle_f = (0.5 + 0.5 * max(-1.0, min(1.0, oracle))
                    + CONFIDENCE_GUARD_W * (d["mean_conf"] - 0.5)
                    + DIVERSITY_GUARD_W * d["bdiv"] - d["penalty"])
        verdict = "REWARD WON" if d["fitness"] >= oracle_f - 1e-9 else \
            ("GUARDS WON" if d["fitness"] >= const_f - 1e-9 else "BELOW CONST")
        print(f"{pos:<5}{d['intents']:>8}{100.0 * d['top_share']:>7.1f}"
              f"{d['mean_reward']:>10.4f}{d['mean_conf']:>7.3f}"
              f"{d['bdiv']:>7.3f}{d['penalty']:>7.3f}{d['base']:>9.4f}"
              f"{d['fitness']:>10.4f}{const_f:>10.4f}{oracle_f:>10.4f}"
              f"{verdict:>12}   (real {real:.4f})")


if __name__ == "__main__":
    sys.exit(main())
