"""Football-truth sprint audit: does anyone in the match actually sprint?

The engine's own `physics_sprint_count` reads zero for every player, which is
either (a) a dead detector, or (b) the truth. This counts sprints straight off
the recorded path, independent of the counter: a sprint is >= 7.0 m/s held for
>= 1.0 s (Di Salvo's threshold), measured as distance over a sliding 1 s
window.
"""
from __future__ import annotations

import json
import math
import sys

SPRINT = 7.0
HI = 8.5


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
    d = json.load(open(path, encoding="utf-8"))
    print(f"{d['home_team']} {d['home_goals']}-{d['away_goals']} {d['away_team']}")

    totals = {"sprints": 0, "hi": 0}
    rows = []
    for name, raw in d["player_path"].items():
        s = sorted((float(a), float(b), float(c)) for a, b, c in raw)
        runs = []
        j = 0
        best = 0.0
        for i in range(len(s)):
            while s[i][0] - s[j][0] > 1.0:
                j += 1
            dt = s[i][0] - s[j][0]
            if dt < 0.8:
                continue
            v = math.hypot(s[i][1] - s[j][1], s[i][2] - s[j][2]) / dt
            best = max(best, v)
            runs.append(v)
        # count distinct bouts: a new bout starts when the player has been
        # below the threshold for >= 5 s
        n_sprint = 0
        cooling = 0.0
        last_t = None
        hot = False
        for i, v in enumerate(runs):
            t = s[i][0]
            cooling += (t - last_t) if last_t is not None else 0.0
            last_t = t
            if v >= SPRINT:
                cooling = 0.0
                if not hot:
                    n_sprint += 1
                    hot = True
            else:
                if cooling >= 5.0:
                    hot = False
        totals["sprints"] += n_sprint
        rows.append((n_sprint, best, name))

    rows.sort(reverse=True)
    print(f"\n{'player':22s} {'bouts':>6s} {'best 1s':>8s}")
    for n, best, name in rows[:8]:
        print(f"  {name:20s} {n:6d} {best:8.2f}")
    print("  ...")
    for n, best, name in rows[-4:]:
        print(f"  {name:20s} {n:6d} {best:8.2f}")
    print(f"\ntotal sprint bouts off the path: {totals['sprints']}  "
          f"(a real 90 has ~300-500 across both teams)")
    print("engine's own counter: 0 for every player "
          "(physics_sprint_count)")


if __name__ == "__main__":
    main()
