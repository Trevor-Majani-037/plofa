"""THE CORPUS'S PER-BUCKET COVERAGE — the gate on a new collection.

WHY THIS FILE EXISTS
--------------------
`_diag_corpus_coverage.py` answers "which intents does this position have
samples of?".  That was the question when the hole was "the striker
performed exactly one shot".  **The hole is now sharper: the striker's
shots exist, and they are all in the WRONG bucket.**

Measured on the v5 corpus (6 matches, ε-exploration 0.25): the striker's
clearest-chance bucket `P1FT1OH0S1C1G1` (final third, goal-close, central,
with space) held ~10 striker decisions and **NOT ONE SHOT**, so its top value
was PROTECT_POSSESSION +0.1068 — the largest positive value in the striker's
whole table.  The objective could not say "shoot" because it had never been
told, and the genetic algorithm duly learned "protect possession".

So the gate for any new collection is per BUCKET, not per intent:

  * the striker (and the second striker) must have a SHOOT cell in a
    final-third bucket, with n >= 3 so the shrinkage does not swallow it;
  * the midfielders must have a THROUGH_BALL cell in a final-third bucket,
    with n >= 3;
  * the wide roles must have a CROSS cell in a final-third bucket, n >= 3;
  * and a SHOOT cell whose mean reward is 0.0 exactly is a RED FLAG, not a
    neutral reading: that is the zero-length marker carry of a strike the
    geometry refused, and it would teach the table that shooting is worth
    nothing.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_corpus_bucket_coverage.py \\
        brains/surrogate_pos_v5.raw.json
"""
import collections
import json
import re
import sys

#: (position, intent) -> what the corpus must contain for the table to be
#: able to teach it.  The intent must appear in a FINAL-THIRD bucket.
REQUIRED = {
    ("ST", "SHOOT"), ("CF", "SHOOT"), ("CAM", "SHOOT"),
    ("LW", "CROSS"), ("RW", "CROSS"), ("LB", "CROSS"), ("RB", "CROSS"),
    ("CM", "THROUGH_BALL"), ("CDM", "THROUGH_BALL"),
}

MIN_N = 3


def _bucket(sensors):
    pressure = int(sensors[16] > 0.5)
    final_third = int(sensors[12] > 0.5)
    own_half = int(sensors[13] > 0.5)
    space = int(sensors[9] > 0.4)
    central = int(sensors[15] > 0.5)
    goal_close = int(sensors[14] < 0.5)
    return (f"P{pressure}FT{final_third}OH{own_half}S{space}"
            f"C{central}G{goal_close}")


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])

    cell = collections.defaultdict(lambda: collections.defaultdict(list))
    pos_n = collections.Counter()
    for row in pairs:
        pos = row.get("position", "")
        intent = row.get("intent", "")
        b = _bucket(row.get("sensors", [0.0] * 24))
        cell[(pos, b)][intent].append(float(row.get("reward", 0.0)))
        pos_n[pos] += 1

    print(f"{path}: {len(pairs)} pairs")
    print()
    print("THE REQUIRED CELLS (an intent the corpus must contain in a "
          "FINAL-THIRD bucket):")
    print(f"{'pos':<6}{'intent':<16}{'final-third buckets':>20}{'best n':>8}"
          f"{'mean reward':>13}  verdict")
    print("-" * 84)
    missing = []
    for (pos, intent) in sorted(REQUIRED):
        ft_buckets = [b for (p, b) in cell if p == pos and "FT1" in b]
        best_n = 0
        best_mean = 0.0
        best_bucket = ""
        for b in ft_buckets:
            vals = cell[(pos, b)].get(intent)
            if not vals:
                continue
            if len(vals) > best_n:
                best_n = len(vals)
                best_mean = sum(vals) / len(vals)
                best_bucket = b
        if best_n >= MIN_N:
            verdict = "OK"
            if abs(best_mean) < 1e-12:
                verdict = "RED FLAG: worth exactly 0.0"
                missing.append(f"{pos} {intent}: {best_bucket} worth exactly "
                               f"0.0 (a refused strike)")
        else:
            verdict = "MISSING"
            missing.append(f"{pos} {intent}: no final-third cell with n >= "
                           f"{MIN_N} (best n = {best_n})")
        print(f"{pos:<6}{intent:<16}{len(ft_buckets):>20}{best_n:>8}"
              f"{best_mean:>+13.5f}  {verdict}"
              + (f"   [{best_bucket}]" if best_bucket else ""))
    print("-" * 84)
    print()
    print("PER-POSITION FINAL-THIRD COVERAGE (the buckets that matter):")
    for pos in sorted(pos_n):
        ft = [b for (p, b) in cell if p == pos and "FT1" in b]
        if not ft:
            print(f"  {pos:<6} NO final-third decision at all "
                  f"({pos_n[pos]} samples)")
            continue
        intents = collections.Counter()
        for b in ft:
            for ik in cell[(pos, b)]:
                intents[ik] += len(cell[(pos, b)][ik])
        print(f"  {pos:<6} {len(ft):>3} final-third buckets, "
              f"{sum(intents.values()):>4} decisions: "
              + ", ".join(f"{k} {v}" for k, v in intents.most_common()))
    print()
    if missing:
        print("HOLES:")
        for h in missing:
            print(f"    {h}")
        print()
        print("A corpus with these holes cannot teach the table what to do in")
        print("the situations that matter.  Do NOT refit and evolve on it —")
        print("that produces the same collapse with a higher fitness number.")
    else:
        print("no holes: every required action is covered in a final-third "
              "bucket.")


if __name__ == "__main__":
    sys.exit(main())
