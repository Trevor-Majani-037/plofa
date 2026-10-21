"""Dump the global ("") row for the rare intents, cell by cell.

Question: the 6-match v4 fit has SHOOT at +0.0056 in the POSITION tables
(joint best of all ten intents) but -0.0039 in the GLOBAL row.  The GA's
`_lookup` falls back to the global row whenever a position's bucket has no
cell for that intent, and for an intent with 14 cells that fallback is used
most of the time.  So the global row matters disproportionately for exactly
the intents the objective was built to favour.

This probe prints, for the global row: every bucket's cell for that intent,
its sample count, its fitted value, and the raw sum (count x value) so the
aggregate can be re-derived by hand rather than trusted.

Usage:
    python _diag_global_row.py brains/surrogate_pos_v4.json [INTENT ...]
"""
import json
import sys


def main():
    path = sys.argv[1]
    wanted = sys.argv[2:] or ["SHOOT", "CROSS", "THROUGH_BALL", "SAFE_PASS"]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    table = data.get("table", data)
    counts = data.get("counts", {})

    glob = table.get("", {})
    gcounts = counts.get("", {})

    for intent in wanted:
        print(f"\n=== GLOBAL row: {intent} ===")
        tot_c = 0
        tot_s = 0.0
        for bucket, intents in sorted(glob.items()):
            if intent not in intents:
                continue
            val = intents[intent]
            c = gcounts.get(bucket, {}).get(intent, 0)
            tot_c += c
            tot_s += c * val
            print(f"  {bucket:<22} n={c:<4} value={val:+.4f}  "
                  f"contribution={c * val:+.5f}")
        if tot_c:
            print(f"  TOTAL n={tot_c}  raw-sum={tot_s:+.5f}  "
                  f"weighted mean={tot_s / tot_c:+.5f}")
        else:
            print("  (no cells)")

    # And the same for the position tables, so the two can be compared.
    print("\n=== per-position cells for the same intents ===")
    for intent in wanted:
        print(f"\n  {intent}:")
        for pos, buckets in sorted(table.items()):
            if pos == "":
                continue
            for bucket, intents in sorted(buckets.items()):
                if intent not in intents:
                    continue
                c = counts.get(pos, {}).get(bucket, {}).get(intent, 0)
                print(f"    {pos:<4} {bucket:<22} n={c:<3} "
                      f"value={intents[intent]:+.4f}")


if __name__ == "__main__":
    main()
