"""Did any side field two goalkeepers, and did every sub actually play?

The 2026-11-07 defect: Luca Ferrini (CM) off at 80' for Joel Trent (GK);
Trent then covered 0.00 km in 19 minutes at a pinned x=11.9, because every
keeper code path scans the roster and breaks on the FIRST keeper.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
d = json.load(open(path, encoding="utf-8"))

bad = 0
for f in d["frames"]:
    for side in ("home", "away"):
        gks = [r["player"] for r in f[side] if r["position"] == "GK"]
        if len(gks) > 1:
            if not bad:
                print(f"minute {f['minute']}: {side} has {len(gks)} keepers: {gks}")
            bad += 1
print(f"frames with two keepers on one side: {bad}")

paths = d["player_path"]
print("\nsubstitute activity (travelled, path extent)")
for side in ("home", "away"):
    for s in d.get("subs", {}).get(side, []):
        n = s.get("player")
        p = paths.get(n)
        if not p:
            continue
        xs = [r[1] for r in p]
        ys = [r[2] for r in p]
        dist = sum(((p[i][1] - p[i - 1][1]) ** 2 + (p[i][2] - p[i - 1][2]) ** 2) ** 0.5
                   for i in range(1, len(p)))
        print(f"  {side:5s} {n:18s} {len(p):5d} samples  "
              f"{dist / 1000.0:5.2f} km  x {min(xs):5.1f}..{max(xs):5.1f}  "
              f"y {min(ys):5.1f}..{max(ys):5.1f}")

# any player standing perfectly still for a long spell?
print("\nplayers whose whole path spans < 2 m (should be none who played)")
for n, p in paths.items():
    xs = [r[1] for r in p]
    ys = [r[2] for r in p]
    span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    if span < 2.0 and len(p) > 120:
        print(f"  {n:20s} span {span:4.2f} m over {len(p)} samples")
