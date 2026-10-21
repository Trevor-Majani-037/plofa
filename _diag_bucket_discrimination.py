"""CAN THE SURROGATE TELL ITS INTENTS APART INSIDE A BUCKET?

Why this exists
---------------
The v5 retrain collapsed HARDER than v4 (8 of 11 positions to two or three
intents; the CF to 96% SWITCH).  The role-inappropriate actions were fixed by
the FORBIDDEN rule, so what is left is a plain collapse: the brain picks the
same one or two actions almost everywhere.

Two candidate causes, and they are very different:
  A. the FITNESS still rewards peakedness; or
  B. the SURROGATE has no signal about which intent to pick in a bucket, so
     the GA has nothing to follow and the network's own bias decides.

If (B), then no amount of fitness rebalancing will help, and the fix is in
the DATA: the cells must carry real differences.  This probe measures it.

For every position it reports, per bucket:
  * the winning intent and its margin over the runner-up;
  * how often each intent wins across the position's buckets.

If one intent wins most buckets by a real margin, the surrogate is telling
the brain what to do and the brain is RIGHT.  If the margins are noise, the
surrogate has no signal and the collapse is the GA's only option.

Usage:
    python _diag_bucket_discrimination.py brains/surrogate_pos_v5.json
"""
import collections
import json
import sys


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    table = data.get("table", {})
    counts = data.get("counts", {})

    print(f"{path}")
    print(f"{'pos':<5} {'buckets':>8} {'clear':>6} {'mean margin':>12}  "
          f"{'winner(s) across the buckets'}")
    print("-" * 96)
    for pos in sorted(p for p in table if p):
        buckets = table[pos]
        winners = collections.Counter()
        margins = []
        clear = 0
        for bucket, intents in buckets.items():
            if not intents:
                continue
            ordered = sorted(intents.items(), key=lambda kv: -kv[1])
            win, wv = ordered[0]
            winners[win] += 1
            if len(ordered) > 1:
                margin = wv - ordered[1][1]
                margins.append(margin)
                if margin > 0.005:
                    clear += 1
        n = len(margins)
        mean_margin = sum(margins) / n if n else 0.0
        top = ", ".join(f"{i}:{c}" for i, c in winners.most_common(4))
        print(f"{pos:<5} {n:>8} {clear:>6} {mean_margin:>12.5f}  {top}")

    # and the global row, which is the fallback for a position with no cell
    g = table.get("", {})
    winners = collections.Counter()
    margins = []
    for bucket, intents in g.items():
        ordered = sorted(intents.items(), key=lambda kv: -kv[1])
        winners[ordered[0][0]] += 1
        if len(ordered) > 1:
            margins.append(ordered[0][1] - ordered[1][1])
    print(f"\nGLOBAL  {len(margins):>8} "
          f"{sum(1 for m in margins if m > 0.005):>6} "
          f"{(sum(margins)/len(margins) if margins else 0):>12.5f}  "
          + ", ".join(f"{i}:{c}" for i, c in winners.most_common(5)))


if __name__ == "__main__":
    sys.exit(main())
