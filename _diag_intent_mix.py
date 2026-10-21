"""Does the retrained striker actually SHOOT?

The collapse check
------------------
T2's documented second collapsed optimum was "ST never shoots".  The retrain
could reproduce that in a way no fitness number would reveal, because the
fitness is the surrogate's own opinion of the brain.  So the check is not
the fitness — it is the INTENT MIX of the evolved brain, read directly.

And the comparison that means something is against the SHIPPED T1 brain on
the SAME synthetic states: "does the new striker shoot more often than the
one in production?"  A retrain that produces the same intent mix has changed
nothing; a retrain that produces a collapsed mix has broken something.

Usage:
    python _diag_intent_mix.py brains_smoke_st/ST.json brains/ST.json [ST]
"""
import collections
import sys

import numpy as np

from brain_evolution import generate_state_corpus
from brain_integration import NeuralDecisionBrain
from brain_sensors import extract_sensors  # noqa: F401  (import side effects)
from football_brain import INTENT_LABELS
from brain_evolution import INPUT_SIZE
from role_features import V2_INPUT_D


def load(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def brain_from_file(path):
    """Load a brain JSON with the project's OWN loader, so the arch, the
    role-feature declaration and the weights are read exactly as they are in
    a match."""
    from football_brain import FootballBrain
    return FootballBrain.load(path)


def role_family(position):
    fam = {
        "GK": "GK", "CB": "CB", "LB": "FB", "RB": "FB", "CDM": "DM",
        "CM": "DM", "CAM": "AM", "LW": "WING", "RW": "WING", "ST": "ST",
        "CF": "ST",
    }
    return fam.get(position, "DM")


def intent_mix(brain, corpus):
    counts = collections.Counter()
    rewards = []
    for row in corpus:
        idx, conf, _probs = brain.predict(row)
        counts[INTENT_LABELS[idx]] += 1
        rewards.append(conf)
    n = len(corpus)
    return counts, n


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    position = "ST"
    if paths[-1].isupper() and len(paths[-1]) <= 4 and paths[-1] in (
            "GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"):
        position = paths.pop()
    width = V2_INPUT_D[role_family(position)]

    corpus = generate_state_corpus(position, 2000, seed=999, input_size=width)
    print(f"corpus: {corpus.shape[0]} states, width {corpus.shape[1]} "
          f"({position})")

    for path in paths:
        brain = brain_from_file(path)
        counts, n = intent_mix(brain, corpus)
        print(f"\n=== {path} ===")
        for label, c in counts.most_common():
            print(f"    {label:<20} {c:>5}  ({100.0*c/n:>5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
