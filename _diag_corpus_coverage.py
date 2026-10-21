"""THE COVERAGE: what the corpus actually contains, per position and intent.

Why this exists
---------------
Three holes in the corpus were measured, and all three were DATA holes:

  * the CM family performed NO through-ball in 4,011 real decisions;
  * the striker performed exactly ONE shot, so his shooting value rested on
    a single sample shrunk to a bucket mean;
  * and a shoot decision was worth EXACTLY 0.0, because the possession
    chain creates a ZERO-LENGTH marker carry and the real shot is fired by
    the attack chain.

The ε-exploration collection is meant to close the first two, and the
shot-aware join the third.  This probe reads the RAW corpus and reports, per
position, the sample count and the MEAN REWARD per intent — the ground
truth, before any shrinkage.

Usage:
    python _diag_corpus_coverage.py brains/surrogate_pos_v5.raw.json
"""
import collections
import json
import sys


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])

    per_pos = collections.defaultdict(lambda: collections.defaultdict(list))
    total = 0
    for row in pairs:
        pos = row.get("position", "")
        intent = row.get("intent", "")
        per_pos[pos][intent].append(float(row.get("reward", 0.0)))
        total += 1

    print(f"{path}: {total} pairs, {len(per_pos)} positions")
    print()
    print(f"{'pos':<6} {'intent':<20} {'n':>5} {'mean reward':>12}  {'min':>8} {'max':>8}")
    print("-" * 68)
    holes = []
    for pos in sorted(per_pos):
        for intent in sorted(per_pos[pos]):
            rs = per_pos[pos][intent]
            print(f"{pos:<6} {intent:<20} {len(rs):>5} "
                  f"{sum(rs)/len(rs):>+12.5f}  {min(rs):>+8.4f} {max(rs):>+8.4f}")
        # the holes this whole work was about
        if "THROUGH_BALL" not in per_pos[pos] and pos in ("CM", "CAM"):
            holes.append(f"{pos}: still NO through-ball sample")
        if "SHOOT" not in per_pos[pos] and pos in ("ST", "CF"):
            holes.append(f"{pos}: still NO shot sample")
        if per_pos[pos].get("SHOOT") and len(per_pos[pos]["SHOOT"]) < 5:
            holes.append(f"{pos}: only {len(per_pos[pos]['SHOOT'])} shot samples")
        if "CROSS" in per_pos[pos] and pos in ("ST", "CF", "CB", "GK", "CDM"):
            holes.append(f"{pos}: {len(per_pos[pos]['CROSS'])} CROSS samples "
                         f"(a role-inappropriate action)")
    print()
    if holes:
        print("HOLES:")
        for h in holes:
            print(f"    {h}")
    else:
        print("no holes: the through-ball and the shot are covered, and no "
              "role-specific action leaked across roles.")


if __name__ == "__main__":
    sys.exit(main())
