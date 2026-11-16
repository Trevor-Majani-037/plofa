import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "tools/diag/_raw_player_path.json"
name = sys.argv[2] if len(sys.argv) > 2 else "Finn Larsson"
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 4522.0
t1 = float(sys.argv[4]) if len(sys.argv) > 4 else 4534.0

d = json.load(open(path, encoding="utf-8"))
for r in d[name]:
    if t0 <= r[0] <= t1:
        src = r[3] if len(r) > 3 else "?"
        print(f"{r[0]:9.2f} {r[1]:6.1f} {r[2]:6.1f}  {src}")
