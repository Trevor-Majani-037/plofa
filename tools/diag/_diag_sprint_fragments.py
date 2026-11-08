"""Sprint TIME versus sprint COUNT.

Di Salvo defines a sprint as >= 1 s above 7.0 m/s. The engine bands the time
correctly, but the bout counter needs the speed to hold above 7.0 after an EMA
(0.3 s) for a further 1.0 s. If the time exists but the count does not, the
high-speed movement is arriving in fragments shorter than a real sprint -- the
bodies are flitting through the top band, not running in it.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
d = json.load(open(path, encoding="utf-8"))

acc = defaultdict(lambda: {"t": 0.0, "n": 0.0, "hi": 0.0, "m": 0.0, "pos": ""})
for f in d["frames"]:
    for side in ("home", "away"):
        for r in f[side]:
            a = acc[r["player"]]
            a["pos"] = r["position"]
            a["t"] += r["physics_sprint_time_s"]
            a["n"] += r["physics_sprint_count"]
            a["hi"] += r["physics_high_speed_sprint_count"]
            a["m"] += r["physics_distance_m"]

print(f"{'player':20s} {'pos':4s} {'sprint s':>9s} {'count':>6s} "
      f"{'s/bout':>7s} {'km':>5s}")
print("-" * 60)
for name, a in sorted(acc.items(), key=lambda kv: -kv[1]["t"]):
    if a["t"] < 1.0:
        continue
    avg = (a["t"] / a["n"]) if a["n"] else float("nan")
    flag = ""
    if a["n"] == 0:
        flag = "  <- all fragments < 1 s"
    print(f"{name:20s} {a['pos']:4s} {a['t']:9.1f} {a['n']:6.0f} "
          f"{avg:7.2f} {a['m'] / 1000.0:5.2f}{flag}")

T = sum(a["t"] for a in acc.values())
N = sum(a["n"] for a in acc.values())
print("-" * 60)
print(f"total sprint time {T / 60.0:6.1f} min   counted bouts {N:5.0f}"
      f"   time per counted bout {T / N if N else 0:5.2f} s")
print("a real sprint bout lasts 2-4 s, so ~30-60 bouts per player "
      "at 2-3 min of sprint time")
