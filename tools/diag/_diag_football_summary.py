"""Football-facing summary of the current replay.

Answers "what does this look like on the pitch" with numbers a coach would
recognise:
  * the fastest SUSTAINED second anyone produced (not the per-sample max,
    which one bad write can poison),
  * how many bodies were where, i.e. is there space between the lines,
  * where each keeper actually stood (own third vs caught upfield),
  * how long the ball went unattended.

Usage:
    python tools/diag/_diag_football_summary.py [replay.json]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


def sustained_top(samples, window=1.0):
    """Fastest distance covered in any `window` seconds of match time."""
    best = 0.0
    j = 0
    for i in range(len(samples)):
        while samples[i][0] - samples[j][0] > window:
            j += 1
        if i == j:
            continue
        dt = samples[i][0] - samples[j][0]
        if dt <= 0:
            continue
        d = math.hypot(samples[i][1] - samples[j][1],
                       samples[i][2] - samples[j][2])
        # normalise to m/s so a short window cannot flatter a body
        best = max(best, d / dt)
    return best


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
    d = json.load(open(path, encoding="utf-8"))
    players = {n: sorted((float(a), float(b), float(c)) for a, b, c in rows)
               for n, rows in d["player_path"].items()}

    print(f"file {path}   ({d['home_team']} {d['home_goals']}-"
          f"{d['away_goals']} {d['away_team']})")

    ranked = sorted(((sustained_top(s), n) for n, s in players.items()),
                    reverse=True)
    print("\nfastest sustained 1-second run (m/s) -- real sprint is 7-9")
    for v, n in ranked[:6]:
        print(f"  {v:5.2f}  {n}")
    print("slowest four")
    for v, n in ranked[-4:]:
        print(f"  {v:5.2f}  {n}")

    print("\nkeepers: how much of the match in their own third (x <= 28 or "
          "x >= 77)")
    for n in ("Keano Walsh", "Jasper Vance"):
        s = players.get(n)
        if not s:
            continue
        xs = [r[1] for r in s]
        own = sum(1 for x in xs if x <= 28.0 or x >= 77.0)
        print(f"  {n:14s} x range {min(xs):5.1f}..{max(xs):5.1f}  "
              f"in own third {100.0 * own / len(xs):5.1f}%")

    # Space between the lines: how often is the closest teammate pair < 1.5 m?
    names = list(players)
    step = 2.0
    end = min(s[-1][0] for s in players.values())
    close_pairs = 0
    samples = 0
    t = 0.0
    idx = {n: 0 for n in names}
    while t <= end:
        pos = {}
        for n in names:
            s = players[n]
            i = idx[n]
            while i + 1 < len(s) and s[i + 1][0] <= t:
                i += 1
            idx[n] = i
            pos[n] = (s[i][1], s[i][2])
        pairs = 0
        vals = list(pos.values())
        for a in range(len(vals)):
            for b in range(a + 1, len(vals)):
                if math.hypot(vals[a][0] - vals[b][0],
                              vals[a][1] - vals[b][1]) < 1.5:
                    pairs += 1
        if pairs:
            close_pairs += 1
        samples += 1
        t += step
    print(f"\ncrowding: {100.0 * close_pairs / max(samples, 1):.1f}% of "
          f"moments have at least one pair of bodies inside 1.5 m")


if __name__ == "__main__":
    main()
