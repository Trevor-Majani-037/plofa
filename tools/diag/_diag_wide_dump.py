"""Dump ONE position_log frame, so the wide-channel probe stops guessing."""
import sys
sys.path.insert(0, ".")
import random
import numpy as np
random.seed(777)
np.random.seed(777)

from _diag_chance_coords import build_pair

pair = build_pair("Oxton", "Natrican")
res = pair.simulate()

f = res.position_log[40]
print("frame keys:", sorted(f.keys()))
print("minute:", f.get("minute"), "poss:", f.get("possession_team"))
for side in ("home", "away"):
    rows = f.get(side) or []
    print(f"\n--- {side}: {len(rows)} rows ---")
    for r in rows[:14]:
        print("   ", {k: r.get(k) for k in
                      ("player", "position", "home_x", "home_y",
                       "current_x", "current_y", "zone")})

print("\npair attrs:", [a for a in dir(pair) if not a.startswith("__")])
print("\nresult attrs:", [a for a in dir(res) if not a.startswith("_")][:40])