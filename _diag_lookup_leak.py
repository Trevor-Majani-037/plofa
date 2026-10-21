"""What does the GA actually read for a given (bucket, intent, position)?

The question
------------
The retrained ST collapsed to CROSS 43% / SHOOT 0%.  The ST's own table has
NO CROSS cell in any bucket, so `FitnessSurrogate._lookup` falls back to the
GLOBAL row for that bucket — where CROSS is +0.0420, a number produced by
WINGERS crossing.

This probe asks `_lookup` directly, for each intent, and reports:
  * the position's own cell (if any),
  * the global row's cell for the same bucket (the fallback),
  * WHICH ONE `_lookup` RETURNS.

So it answers "is the value the GA reads borrowed from another role?" without
guessing.  It also reports the same lookup for the roles that legitimately
perform each intent, so the rule can be checked against real usage rather
than assumed.

Usage:
    python _diag_lookup_leak.py brains/surrogate_pos_v4.json ST [bucket]
"""
import json
import sys

import numpy as np

from surrogate_collect import FitnessSurrogate
from football_brain import INTENT_LABELS


def main():
    path = sys.argv[1]
    position = sys.argv[2] if len(sys.argv) > 2 else "ST"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sur = FitnessSurrogate()
    sur.table = data.get("table", {})
    sur.counts = data.get("counts", {})
    sur.prior = data.get("prior", 0.0)

    positions = sorted(p for p in sur.table if p != "")
    print(f"surrogate: {path}")
    print(f"positions: {positions}")
    print()

    # which intents does each position actually use anywhere?
    usage = {}
    for pos in positions:
        used = set()
        for bucket, intents in sur.table.get(pos, {}).items():
            used.update(intents.keys())
        usage[pos] = used
    print("=== intent usage by position (from the position's own table) ===")
    for pos in positions:
        print(f"  {pos:<4} {sorted(usage[pos])}")
    print()

    print("=== per bucket, for the focus position ===")
    for bucket in sorted(sur.table.get(position, {})):
        row = sur.table[position][bucket]
        grow = sur.table.get("", {}).get(bucket, {})
        print(f"  {bucket}")
        for ik in INTENT_LABELS:
            own = row.get(ik)
            glob = grow.get(ik)
            got = sur._lookup(bucket, ik, position)
            src = ("OWN" if own is not None else
                   ("GLOBAL-BORROWED" if glob is not None else "PRIOR"))
            got_s = "  None  " if got is None else f"{got:+.4f}"
            print(f"      {ik:<20} own={('%.4f' % own) if own is not None else '   --  '} "
                  f"global={('%.4f' % glob) if glob is not None else '   --  '} "
                  f"lookup={got_s}  <- {src}"
                  + ("   ** BORROWED FROM ANOTHER ROLE **"
                     if (own is None and glob is not None
                         and ik not in usage.get(position, set())) else ""))


if __name__ == "__main__":
    sys.exit(main())
