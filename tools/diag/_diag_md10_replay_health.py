"""Are the exported MD10 replays actually clean?

The broadcast draws from `player_path` (0.5 s keyframes), NOT from
`frames` (1 minute). Measuring the per-minute frames is meaningless -- it
averages a sprint into 1.2 m/s. This reads the path the viewer interpolates.

A jump here is a teleport on screen. 25 m/s is the bound: the fastest human
sprinters sustain ~10 m/s and the engine's own pace ladder caps at 7.78.
"""
from __future__ import annotations

import glob
import json
import math
import os

LIMIT = 25.0


def main() -> None:
    print(f"{'match':38s} {'max m/s':>8s} {f'over {LIMIT:.0f}':>9s} "
          f"{'>15m steps':>11s} {'samples':>8s}")
    print("-" * 80)
    for path in sorted(glob.glob("plofa_output/*MD10*/replay.json")):
        d = json.load(open(path, encoding="utf-8"))
        pp = d.get("player_path") or {}
        worst = 0.0
        over = 0
        big_steps = 0
        samples = 0
        for name, rows in pp.items():
            s = sorted((float(r[0]), float(r[1]), float(r[2])) for r in rows)
            for i in range(1, len(s)):
                t0, x0, y0 = s[i - 1]
                t1, x1, y1 = s[i]
                dt = t1 - t0
                if dt <= 0:
                    continue
                dist = math.hypot(x1 - x0, y1 - y0)
                v = dist / dt
                samples += 1
                if v > worst:
                    worst = v
                if v > LIMIT:
                    over += 1
                if dist > 15.0:
                    big_steps += 1
        name = os.path.basename(os.path.dirname(path))
        print(f"{name:38s} {worst:8.1f} {over:9d} {big_steps:11d} {samples:8d}")


if __name__ == "__main__":
    main()
