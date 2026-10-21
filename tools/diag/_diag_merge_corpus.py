"""MERGE two raw corpora and fit the result — free extra samples per cell.

WHY
---
The v7 corpus (12 matches, situation-aware exploration) has the coverage the
v5 corpus lacked: the striker's final-third SHOOT cell went from n = 0 to
n = 16, with a mean of +0.0125 — **seven times the next intent** in that
bucket.  But one cell is still thin: the SECOND striker's (the CF's) own
final-third shot cells, which sit at **n = 2 and worth exactly 0.0** (every
one of those shots was refused by the engine's shot gate, so each was
recorded as the zero-length marker carry).

The v5 corpus holds ~15 more final-third decisions for the CF, and ~31 more
for the striker, so merging is free extra data for every cell.  Whether it
HELPS is not obvious: more samples of the COMMON actions means less
shrinkage for them, which can move their values toward (or away from) the
shooting value.  **So the merge is measured, not assumed.**

WHAT THIS DOES
--------------
Merges N raw corpora into one, fits it, and prints, for the positions that
matter, the final-third cells the corpus census cares about — so the
comparison against the un-merged fit is a like-for-like read of the SAME
numbers the census prints.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_merge_corpus.py \\
        brains/surrogate_pos_v7.raw.json brains/surrogate_pos_v5.raw.json \\
        brains/surrogate_pos_v7merged.json
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from surrogate_collect import FitnessSurrogate, load_raw_corpus  # noqa: E402


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
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 1
    out_path = args[-1]
    paths = args[:-1]

    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        pairs = data.get("pairs", [])
        rows.extend(pairs)
        print(f"{p}: {len(pairs)} pairs")
    print(f"merged: {len(rows)} pairs")
    print()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"pairs": rows}, f)
    print(f"wrote {out_path}")
    print()

    sur = FitnessSurrogate().fit(load_raw_corpus(out_path))
    print("THE FINAL-THIRD CELLS THE CENSUS CARES ABOUT:")
    for pos, intents in (("ST", ("SHOOT",)), ("CF", ("SHOOT",)),
                          ("CAM", ("SHOOT",)), ("CM", ("THROUGH_BALL",)),
                          ("CDM", ("THROUGH_BALL",)),
                          ("LW", ("CROSS",)), ("RW", ("CROSS",)),
                          ("LB", ("CROSS",)), ("RB", ("CROSS",))):
        best = []
        for bucket, cells in sur.table.get(pos, {}).items():
            if "FT1" not in bucket:
                continue
            for ik in intents:
                if ik in cells:
                    n = sur.counts.get(pos, {}).get(bucket, {}).get(ik, 0)
                    best.append((n, cells[ik], bucket))
        best.sort(reverse=True)
        if not best:
            print(f"  {pos:<5} {intents[0]:<14} NO final-third cell")
            continue
        n, v, bucket = best[0]
        flag = "" if n >= 3 else "   <-- THIN"
        if abs(v) < 1e-12 and n >= 3:
            flag = "   <-- WORTH EXACTLY 0.0 (refused strikes)"
        print(f"  {pos:<5} {intents[0]:<14} n={n:<5} {v:+.5f}  [{bucket}]{flag}")
    print()
    print("the same table as the un-merged fit is printed by")
    print("_diag_position_table.py <surrogate> <pos>")


if __name__ == "__main__":
    sys.exit(main())
