"""Per-match football summary of the six MD10 exports, from player_path."""
from __future__ import annotations

import glob
import json
import math
import os

for path in sorted(glob.glob("plofa_output/*MD10*/replay.json")):
    d = json.load(open(path, encoding="utf-8"))
    tot = {}
    gk = {}
    for name, rows in d["player_path"].items():
        s = sorted((float(r[0]), float(r[1]), float(r[2])) for r in rows)
        m = sum(math.hypot(s[i][1] - s[i - 1][1], s[i][2] - s[i - 1][2])
                for i in range(1, len(s)))
        tot[name] = m
    # keeper identification from the first frame
    for side in ("home", "away"):
        for r in d["frames"][0][side]:
            if r.get("position") == "GK":
                gk[r["player"]] = side
    top = sorted(tot.items(), key=lambda kv: -kv[1])[:3]
    gktxt = ", ".join(f"{n} {tot[n] / 1000:.2f}km" for n in gk)
    print(f"{d['home_team']:16s} {d['home_goals']}-{d['away_goals']} "
          f"{d['away_team']:16s} {d['duration_s'] / 60:4.0f}min")
    print("   top distance: " + ", ".join(f"{n} {m / 1000:.1f}km"
                                          for n, m in top))
    print(f"   keepers: {gktxt}   possession {d.get('home_possession')}")
