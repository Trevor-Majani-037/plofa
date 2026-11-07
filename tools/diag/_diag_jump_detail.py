"""List the jumps >15 m in a replay with before/after coordinates and the
ball kind at that moment, so the source of the remaining snaps is identifiable.
"""
import json
import math
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
d = json.load(open(path, encoding="utf-8"))
ball = sorted((float(b["t"]), float(b["x"]), float(b["y"]), b.get("kind", ""))
              for b in d["ball"])
pp = {n: sorted((float(a), float(b), float(c)) for a, b, c in rows)
      for n, rows in d["player_path"].items()}


def ball_at(t):
    lo, hi, best = 0, len(ball) - 1, None
    while lo <= hi:
        m = (lo + hi) // 2
        if ball[m][0] <= t:
            best = ball[m]
            lo = m + 1
        else:
            hi = m - 1
    return best


rows = []
for n, s in pp.items():
    for a, b in zip(s, s[1:]):
        jump = math.hypot(b[1] - a[1], b[2] - a[2])
        if jump > 15.0:
            bb = ball_at(b[0])
            rows.append((b[0], n, jump, b[0] - a[0],
                         jump / max(1e-6, b[0] - a[0]),
                         a[1:], b[1:], bb[3] if bb else "?",
                         bb[1] if bb else 0.0, bb[2] if bb else 0.0))
rows.sort(key=lambda r: -r[4])
print(f"{len(rows)} jumps >15 m")
for t, n, j, dt, v, a, b, kind, bx, by in rows:
    print(f"{int(t)//60:02d}:{t%60:05.1f} {n[:18]:18s} {j:5.1f} m "
          f"dt {dt:4.2f} v {v:6.1f} {a[0]:5.1f},{a[1]:5.1f} -> "
          f"{b[0]:5.1f},{b[1]:5.1f} ball {kind:8s} ({bx:5.1f},{by:5.1f})")
