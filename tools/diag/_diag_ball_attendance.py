"""What does each kind of ball action look like, in FOOTBALL terms?

For every ball sample, ask: how far is the nearest body from the ball at that
instant? Split by the ball's `kind`, because the answer should be different:

  carry   a man is dribbling   -> the ball must be at his feet (~1-2 m)
  pass    the ball is in flight-> nobody should be on it (it is 15-25 m/s)
  move    a struck/lofted ball -> same
  fetch   a loose ball being retrieved -> someone is closing on it
  rest    a dead ball          -> the taker is standing over it

Usage:
    python tools/diag/_diag_ball_attendance.py [path/to/replay.json]
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict


STEP = 0.25


def series_at(samples, t):
    lo, hi, best = 0, len(samples) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            best = samples[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def pct(vals, q):
    if not vals:
        return float("nan")
    i = min(len(vals) - 1, int(q * (len(vals) - 1)))
    return vals[i]


def report(path: str) -> None:
    d = json.load(open(path, encoding="utf-8"))
    ball = sorted((float(b["t"]), float(b["x"]), float(b["y"]),
                   b.get("kind", "?")) for b in d["ball"])
    players = {n: sorted((float(a), float(b), float(c)) for a, b, c in rows)
               for n, rows in d["player_path"].items()}
    end_t = max(s[-1][0] for s in players.values())

    by_kind = defaultdict(list)
    t, i = 0.0, 0
    while t <= end_t:
        while i + 1 < len(ball) and ball[i + 1][0] <= t:
            i += 1
        b = ball[i]
        near = None
        for s in players.values():
            r = series_at(s, t)
            if r is None:
                continue
            dd = math.hypot(r[1] - b[1], r[2] - b[2])
            if near is None or dd < near:
                near = dd
        if near is not None:
            by_kind[b[3]].append(near)
        t += STEP

    print(f"file {path}")
    print(f"nearest body to the ball, by ball kind (metres)")
    print(f"{'kind':8s} {'n':>7s} {'p10':>7s} {'p50':>7s} {'p90':>7s} "
          f"{'max':>7s} {'<=2m':>7s}")
    order = ["carry", "pass", "move", "fetch", "rest", "restart",
             "event", "static"]
    for kind in order:
        vals = sorted(by_kind.get(kind) or [])
        if not vals:
            continue
        close = 100.0 * sum(1 for v in vals if v <= 2.0) / len(vals)
        print(f"{kind:8s} {len(vals):7d} {pct(vals,.1):7.1f} "
              f"{pct(vals,.5):7.1f} {pct(vals,.9):7.1f} "
              f"{max(vals):7.1f} {close:6.1f}%")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1
           else "plofa_broadcast/replay.json")
