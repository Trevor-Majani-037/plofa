"""Print one position's whole surrogate table, bucket by bucket.

Why this exists
---------------
The GA's fitness is the surrogate's value for the intent the brain CHOSE
(`synthetic_fitness` -> `surrogate.expected_success(sensors, intent,
position)`), so the question that decides a retrain is not "is this intent's
average positive" — it is **"in the buckets where this intent is available,
is it competitive with the alternatives?"**.

The aggregate over buckets hid that.  On the 6-match v4 fit the ST's own two
SHOOT cells read -0.0017 and -0.0051 (n=1 each, shrunk toward the bucket
mean) while the global fallback for the same buckets read +0.0243 / -0.0270.
`FitnessSurrogate._lookup` prefers the POSITION cell, so the ST's own numbers
are the ones the GA will use — and they must be compared against the other
intents in the SAME bucket, not against an average.

Usage:
    python _diag_position_table.py brains/surrogate_pos_v4.json ST [CM]
"""
import json
import sys


def main():
    path = sys.argv[1]
    positions = sys.argv[2:] or ["ST"]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    table = data.get("table", data)
    counts = data.get("counts", {})

    for pos in positions:
        buckets = table.get(pos, {})
        print(f"\n=== {pos} ({len(buckets)} buckets) ===")
        for bucket, intents in sorted(buckets.items()):
            print(f"  {bucket}")
            ordered = sorted(intents.items(), key=lambda kv: -kv[1])
            for intent, value in ordered:
                c = counts.get(pos, {}).get(bucket, {}).get(intent, 0)
                print(f"      {intent:<20} {value:+.4f}   n={c}")
        # the global row for the same buckets — the fallback the GA uses
        # when the position has no cell of its own for an intent
        print(f"  --- global row, same buckets ---")
        for bucket in sorted(buckets):
            g = table.get("", {}).get(bucket, {})
            if not g:
                continue
            ordered = sorted(g.items(), key=lambda kv: -kv[1])
            top = ", ".join(f"{i}={v:+.4f}" for i, v in ordered[:5])
            print(f"      {bucket:<22} {top}")


if __name__ == "__main__":
    main()
