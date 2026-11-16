import json
from collections import Counter

d = json.load(open("plofa_broadcast/replay.json", encoding="utf-8"))
T = 105.0 / 3.0
ball = d["ball"]
c = Counter()
for row in ball:
    x = float(row["x"])
    c["own" if x < T else ("final" if x > 2 * T else "middle")] += 1
n = max(len(ball), 1)
print("ball thirds:", {k: round(100.0 * v / n, 1) for k, v in c.items()})

# how many bodies are in each third, on average
bodies = Counter()
frames = d["frames"]
for f in frames:
    for side in ("home", "away"):
        for r in f[side]:
            x = float(r["x"])
            bodies["own" if x < T else ("final" if x > 2 * T else "middle")] += 1
tot = sum(bodies.values()) or 1
print("body thirds:", {k: round(100.0 * v / tot, 1) for k, v in bodies.items()})

# average number of players within 10 m of the ball
import math
near = []
for f in frames:
    bx = f.get("ball_x")
    by = f.get("ball_y")
    if bx is None:
        continue
    k = 0
    for side in ("home", "away"):
        for r in f[side]:
            if math.hypot(r["x"] - bx, r["y"] - by) <= 10.0:
                k += 1
    near.append(k)
if near:
    print("bodies within 10 m of the ball (mean):", round(sum(near) / len(near), 2),
          "min", min(near), "max", max(near))
else:
    print("frames carry no ball_x; skipping proximity")
