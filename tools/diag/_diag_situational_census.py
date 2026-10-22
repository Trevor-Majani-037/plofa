"""THE SITUATIONAL CENSUS — every position's intent mix on HELD-OUT data.

Run:
    .venv\\Scripts\\python.exe tools\\diag\\_diag_situational_census.py \
        brains_v8 brains/surrogate_pos_v8_situational.sequences.json

WHY IT EXISTS, AND WHY IT IS NOT THE MLP CENSUS
------------------------------------------------
``_diag_intent_mix_all.py`` is the census for the MLP brains: it reads
each position's intent mix over a SYNTHETIC corpus and flags a collapse.
The situational brain cannot be read that way — it takes ENTITY TOKENS, a
role block, a memory block and a plan, none of which
``generate_state_corpus`` produces.

So this reads the REAL corpus, and — the point — it reads the HELD-OUT
SPLIT, using the SAME deterministic split the trainer used
(``situational_distill.split_sequences``).  A collapse measured on the
training set is still a collapse, but a brain that merely FAILED TO LEARN
looks different on data it has never seen, and the census has to be able
to tell those two apart.

THE RULES ARE COPIED, NOT REINVENTED: fewer than five intents, any single
intent over 40%, or a role-specific action the role should not perform
(derived from ``role_features.ROLE_ONLY_INTENTS``).  Two gates that
drift are one gate.

THE REFERENCE is the mix of the SURROGATE'S OWN policy over the same
held-out states — the table the brain is trained to imitate.  So the
comparison is controlled: the same states, two policies, one of them the
target.
"""
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "diag"))

import numpy as np

from situational_brain import HIDDEN_D, SituationalBrain
from situational_distill import (
    census_flags, load_sequences, split_sequences, _mix_over,
    _position_family,
)

POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]


def table_policy_mix(surrogate: Any, sequences: List[Any],
                     position: str) -> Dict[str, float]:
    """The SURROGATE's own policy over the same states.

    This is what the brain is TRAINED to imitate, so it is the honest
    reference for "is this mix collapsed?" — not the shipped MLP brain,
    which reads a different input layout entirely.
    """
    from decision_brain import PlayerIntent
    from role_features import ROLE_INTENTS, _role_family
    from surrogate_collect import FitnessSurrogate  # noqa: F401  (path)

    fam = _role_family(position) or ""
    vocab = [i for i in PlayerIntent if i.value in ROLE_INTENTS.get(fam, frozenset())]
    if not vocab:
        return {}
    counts: Counter = Counter()
    for seq in sequences:
        for row in seq.state:
            best = None
            best_v = None
            for intent in vocab:
                v = float(surrogate.expected_success(row[:24], intent, position))
                if best_v is None or v > best_v:
                    best, best_v = intent, v
            if best is not None:
                counts[best.value] += 1
    total = max(1, sum(counts.values()))
    return {k: v / total for k, v in counts.items()}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    brains_dir = sys.argv[1]
    corpus_path = sys.argv[2]
    surrogate_path = sys.argv[3] if len(sys.argv) > 3 else None

    all_seqs = load_sequences(corpus_path)
    by_pos: Dict[str, List[Any]] = {}
    for seq in all_seqs:
        by_pos.setdefault(seq.position or "", []).append(seq)

    surrogate = None
    if surrogate_path and os.path.exists(surrogate_path):
        from critic_surrogate import load_any_surrogate

        surrogate = load_any_surrogate(surrogate_path)

    print(f"{'pos':<5} {'seqs':>5} {'intents':>8} {'max share':>10} "
          f"{'top intent':<20} {'SHOOT':>7} {'CROSS':>7}  vs the table")
    print("-" * 104)
    flags: List[str] = []
    for pos in POSITIONS:
        path = os.path.join(brains_dir, f"{pos}.json")
        if not os.path.exists(path):
            print(f"{pos:<5} (missing)")
            continue
        seqs = by_pos.get(_position_family(pos), [])
        if not seqs:
            print(f"{pos:<5} (no sequences for family "
                  f"{_position_family(pos)})")
            continue
        _train, eval_ = split_sequences(seqs, seed=123, holdout=0.2)
        brain = SituationalBrain.load(path)
        mix = _mix_over(brain, eval_ or seqs)
        total = max(1e-9, sum(mix.values()))
        top = max(mix, key=lambda k: mix[k])
        shoot = mix.get("SHOOT", 0.0) / total
        cross = mix.get("CROSS", 0.0) / total

        note = ""
        if surrogate is not None:
            tmix = table_policy_mix(surrogate, eval_ or seqs, pos)
            if tmix:
                ttop = max(tmix, key=lambda k: tmix[k])
                note = (f"top {ttop} {tmix[ttop]:.0%}, "
                        f"{len(tmix)} intents")
        print(f"{pos:<5} {len(eval_ or seqs):>5} {len(mix):>8} "
              f"{mix[top]/total:>9.1%} {top:<20} {shoot:>6.1%} {cross:>6.1%}"
              f"  {note}")

        pos_flags = census_flags(pos, mix)
        flags.extend(pos_flags)

    print()
    if flags:
        print("FLAGGED:")
        for f in flags:
            print(f"    {f}")
    else:
        print("no collapse flagged: every position uses >4 intents, no action "
              "over 40%, and no role-specific action out of role.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
