"""The COLLAPSE SWEEP: every position's intent mix, new brain vs shipped.

Why this exists
---------------
The first retrained striker chose CROSS 43% of the time and SHOOT 0% — a
four-intent collapse on an action the striker never performs, and its
fitness number (0.6487) was unremarkable.  **Only the intent mix showed
it.**  So every position has to be read the same way, against the shipped
brain on the SAME corpus, before any brain set is considered.

The reference is the shipped T1 brain for that position, so the comparison
is a controlled pair: same corpus, same states, two weights.

Usage:
    python _diag_intent_mix_all.py brains_v4 [reference_dir]
"""
import os
import sys
from collections import Counter

import numpy as np

from brain_evolution import generate_state_corpus
from football_brain import FootballBrain, INTENT_LABELS
from role_features import V2_INPUT_D

POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]

FAMILY = {
    "GK": "GK", "CB": "CB", "LB": "FB", "RB": "FB", "CDM": "DM",
    "CM": "CM", "CAM": "AM", "LW": "WING", "RW": "WING", "ST": "ST", "CF": "ST",
}


def mix(brain, corpus):
    counts = Counter()
    for row in corpus:
        idx, _conf, _probs = brain.predict(row)
        counts[INTENT_LABELS[idx]] += 1
    return counts


def main():
    new_dir = sys.argv[1]
    ref_dir = sys.argv[2] if len(sys.argv) > 2 else "brains"
    n_states = 2000
    seed = 999

    print(f"{'pos':<5} {'intents':>8} {'max share':>10} {'top intent':<20} "
          f"{'SHOOT':>7} {'CROSS':>7}  vs shipped")
    print("-" * 100)
    flagged = []
    for pos in POSITIONS:
        path = os.path.join(new_dir, f"{pos}.json")
        if not os.path.exists(path):
            print(f"{pos:<5} (missing)")
            continue
        width = V2_INPUT_D[FAMILY[pos]]
        corpus = generate_state_corpus(pos, n_states, seed=seed, input_size=width)
        brain = FootballBrain.load(path)
        counts = mix(brain, corpus)
        total = sum(counts.values())
        top, topn = counts.most_common(1)[0]
        intents = len(counts)

        ref_path = os.path.join(ref_dir, f"{pos}.json")
        note = ""
        if os.path.exists(ref_path):
            try:
                rcounts = mix(FootballBrain.load(ref_path), corpus)
                rtotal = sum(rcounts.values())
                rt, rtn = rcounts.most_common(1)[0]
                note = f"top {rt} {100.0*rtn/rtotal:.0f}%, {len(rcounts)} intents"
            except Exception as exc:  # a width mismatch is a real finding
                note = f"(reference unreadable: {exc})"

        shoot = 100.0 * counts.get("SHOOT", 0) / total
        cross = 100.0 * counts.get("CROSS", 0) / total
        print(f"{pos:<5} {intents:>8} {100.0*topn/total:>9.1f}% {top:<20} "
              f"{shoot:>6.1f}% {cross:>6.1f}%  {note}")

        # the flags: a collapsed mix, or a role-specific action the role
        # should not be performing at all.
        #
        # THE ROLE-INAPPROPRIATE RULE IS DERIVED, NOT WRITTEN BY HAND.  It
        # used to be a hand-written list of positions that "should not
        # shoot", and it listed the CAM — but the CAM IS the attacking
        # midfielder, `role_features.ROLE_INTENTS["AM"]` includes SHOOT and
        # `ROLE_ONLY_INTENTS["SHOOT"]` is {ST, AM, WING}: the AM family
        # performed SHOOT in the corpus, so a CAM shooting is
        # ROLE-APPROPRIATE and the rule was flagging good football.
        from role_features import ROLE_FAMILY, ROLE_ONLY_INTENTS
        fam = ROLE_FAMILY.get(pos.rstrip("0123456789"))
        for intent, allowed in ROLE_ONLY_INTENTS.items():
            share = 100.0 * counts.get(intent, 0) / total
            if fam is not None and fam not in allowed and share > 1.0:
                flagged.append(f"{pos}: {intent} at {share:.1f}% "
                               f"(a role-inappropriate action)")

        if intents <= 4:
            flagged.append(f"{pos}: only {intents} intents")
        if 100.0 * topn / total > 40.0:
            flagged.append(f"{pos}: {top} at {100.0*topn/total:.0f}%")

    print()
    if flagged:
        print("FLAGGED:")
        for f in flagged:
            print(f"    {f}")
    else:
        print("no collapse flagged: every position uses >4 intents, no action "
              "over 40%, and no role-specific action out of role.")


if __name__ == "__main__":
    sys.exit(main())
