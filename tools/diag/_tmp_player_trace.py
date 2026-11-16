import json
import math
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
name = sys.argv[2] if len(sys.argv) > 2 else "Kwame Asante"
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 4640.0
t1 = float(sys.argv[4]) if len(sys.argv) > 4 else 4670.0

d = json.load(open(path, encoding="utf-8"))
rows = [r for r in d["player_path"][name] if t0 <= r[0] <= t1]
prev = None
for r in rows:
    step = "" if prev is None else f"{math.hypot(r[1]-prev[1], r[2]-prev[2]):6.1f}"
    print(f"{r[0]:9.2f}  {r[1]:6.1f} {r[2]:6.1f}  step {step}")
    prev = r
