"""Print what is happening around a window of interest: ball kind, position,
nearest bodies, and which players are moving."""
import json
import math
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 5280.0
t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 5340.0

d = json.load(open(path, encoding="utf-8"))
ball = sorted((float(b["t"]), float(b["x"]), float(b["y"]), b.get("kind", ""))
              for b in d["ball"])
pp = {n: sorted((float(a), float(b), float(c)) for a, b, c in rows)
      for n, rows in d["player_path"].items()}


def last(samples, t):
    lo, hi, best = 0, len(samples) - 1, None
    while lo <= hi:
        m = (lo + hi) // 2
        if samples[m][0] <= t:
            best = samples[m]
            lo = m + 1
        else:
            hi = m - 1
    return best


prev = None
for T in [t0 + i * 0.5 for i in range(int((t1 - t0) / 0.5) + 1)]:
    b = last(ball, T)
    if b is None:
        continue
    pos = [(n,) + tuple(last(s, T)[1:]) for n, s in pp.items()
           if last(s, T) is not None]
    near = min(pos, key=lambda p: math.hypot(p[1] - b[1], p[2] - b[2]),
               default=None)
    gap = math.hypot(near[1] - b[1], near[2] - b[2]) if near else -1
    moved = "" if prev is None else f"{math.hypot(b[1]-prev[1], b[2]-prev[2]):4.1f}"
    print(f"{int(T)//60:02d}:{T%60:04.1f} ball {b[1]:5.1f},{b[2]:5.1f} {b[3]:8s}"
          f" step {moved:>4s} near {near[0][:16]:16s} {gap:5.1f} m")
    prev = b
