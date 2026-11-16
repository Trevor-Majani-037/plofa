"""Does the engine's booked distance match the path it drew?

`physics_distance_m` is what the engine says the player covered. The path is
what the viewer draws. If the booked number is larger, the engine is
crediting motion the body never made -- which is also why the booked bands
(walking 64%) disagree with the recorded ones (walking 48%, frozen 25%).
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
d = json.load(open(path, encoding="utf-8"))

booked = defaultdict(float)
for f in d["frames"]:
    for side in ("home", "away"):
        for r in f[side]:
            booked[r["player"]] += float(r.get("physics_distance_m") or 0.0)

drawn = {}
for name, rows in (d.get("player_path") or {}).items():
    s = sorted((float(r[0]), float(r[1]), float(r[2])) for r in rows)
    drawn[name] = sum(math.hypot(s[i][1] - s[i - 1][1], s[i][2] - s[i - 1][2])
                      for i in range(1, len(s)))

print(f"{'player':20s} {'booked km':>10s} {'drawn km':>9s} {'booked/drawn':>13s}")
print("-" * 58)
tb = td = 0.0
for name in sorted(drawn, key=lambda n: -booked.get(n, 0.0))[:16]:
    b = booked.get(name, 0.0)
    td_ = drawn[name]
    tb += b
    td += td_
    ratio = (b / td_) if td_ else float("nan")
    print(f"{name:20s} {b / 1000:10.2f} {td_ / 1000:9.2f} {ratio:13.3f}")
print("-" * 58)
print(f"{'TOTAL':20s} {tb / 1000:10.2f} {td / 1000:9.2f} {tb / td:13.3f}")
