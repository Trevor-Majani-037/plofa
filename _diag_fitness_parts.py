"""WHERE DOES THE FITNESS COME FROM — the reward, or the CONFIDENCE?

Why this exists
---------------
The retrain against the role-pooled surrogate produced collapses in four
positions (LB: TWO intents, 82% RECYCLE; CDM: 61% DRIBBLE; CM: 67%
SAFE_PASS; LW: 61% CARRY) and role-inappropriate actions in the GK (17%
SHOOT, 14% CROSS) and the CB (44% SHOOT) — while every FITNESS number was
unremarkable (0.60-0.67).

The suspicion is the fitness formula itself:

    base   = 0.5 + 0.5 * clip(mean_reward, -1, 1)
    fitness = base * (0.5 + 0.5 * mean_conf) - penalties / n_states

The threat reward's MEAN is about 0.005 and its SPREAD between intents is
about 0.001-0.02, so `base` barely moves.  The confidence term's range is
0.55-1.00 — a factor of two.  So the GA may be selecting for CONFIDENCE
(peakedness) rather than for the reward, and a peaked network is exactly a
collapsed intent mix.

This probe measures the two contributions SEPARATELY, and it VERIFIES its
own arithmetic against the real `synthetic_fitness` — a probe that
re-implements the thing it measures is evidence about nothing unless the
replication is checked.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_evolution import generate_state_corpus, synthetic_fitness  # noqa: E402
from football_brain import FootballBrain, INTENT_LABELS  # noqa: E402
from role_features import V2_INPUT_D  # noqa: E402
from surrogate_collect import FitnessSurrogate  # noqa: E402

FAMILY = {
    "GK": "GK", "CB": "CB", "LB": "FB", "RB": "FB", "CDM": "DM",
    "CM": "CM", "CAM": "AM", "LW": "WING", "RW": "WING", "ST": "ST", "CF": "ST",
}


def decompose(brain, position, corpus, surrogate):
    """Reproduce `synthetic_fitness` and return its parts."""
    rewards, confs = [], []
    for row in corpus:
        probs = brain.forward(row)
        idx = int(np.argmax(probs))
        rewards.append(surrogate.expected_success(row[:24],
                                                  _intent_of(idx), position))
        confs.append(float(probs[idx]))
    mean_reward = float(np.mean(rewards))
    mean_conf = float(np.mean(confs))
    base = 0.5 + 0.5 * max(-1.0, min(1.0, mean_reward))
    fitness = base * (0.5 + 0.5 * mean_conf)
    return mean_reward, mean_conf, base, fitness


def _intent_of(idx):
    from brain_evolution import _INTENT_BY_IDX
    return _INTENT_BY_IDX[idx]


def main():
    brains_dir = sys.argv[1] if len(sys.argv) > 1 else "brains_v4"
    ref_dir = sys.argv[2] if len(sys.argv) > 2 else "brains"
    sur_path = sys.argv[3] if len(sys.argv) > 3 else "brains/surrogate_pos_v4.json"
    sur = FitnessSurrogate.load(sur_path)
    positions = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]
    n_states = 300

    print(f"{'pos':<5} {'brain':<12} {'mean_reward':>12} {'mean_conf':>10} "
          f"{'base':>8} {'fitness':>8} {'conf spread':>12} {'rew spread':>11}")
    print("-" * 92)
    for pos in positions:
        path = os.path.join(brains_dir, f"{pos}.json")
        if not os.path.exists(path):
            continue
        width = V2_INPUT_D[FAMILY[pos]]
        corpus = generate_state_corpus(pos, n_states, seed=123, input_size=width)
        brain = FootballBrain.load(path)
        mr, mc, base, fit = decompose(brain, pos, corpus, sur)

        # the real function, to prove the decomposition reproduces it
        real = synthetic_fitness(brain, pos, batched_sensors=corpus,
                                  surrogate=sur)
        label = "new"
        # and the shipped brain, at ITS OWN width (the legacy path)
        rpath = os.path.join(ref_dir, f"{pos}.json")
        note = ""
        if os.path.exists(rpath):
            try:
                import json
                with open(rpath, encoding="utf-8") as f:
                    arch = json.load(f)["arch"][0]
                rcorpus = generate_state_corpus(pos, n_states, seed=123,
                                                input_size=arch)
                rb = FootballBrain.load(rpath)
                rmr, rmc, rbase, rfit = decompose(rb, pos, rcorpus, sur)
                note = (f"  | shipped: reward {rmr:+.5f} conf {rmc:.3f} "
                        f"fitness {rfit:.4f}")
            except Exception as exc:
                note = f"  | shipped: {exc}"

        # the two spreads: how much fitness the confidence term can move, and
        # how much the reward term can
        conf_spread = base * 0.5 * (1.0 - 0.1)
        rew_spread = 0.5 * 0.02
        print(f"{pos:<5} {label:<12} {mr:>+12.6f} {mc:>10.4f} {base:>8.5f} "
              f"{fit:>8.5f} {conf_spread:>12.5f} {rew_spread:>11.5f}"
              f"   (real {real:.5f}){note}")


if __name__ == "__main__":
    sys.exit(main())
