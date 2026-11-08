"""How long does a high-speed effort actually LAST in the recorded path?

The bout counter now accumulates fragments, so a player only fails to score a
bout if his time above 7.0 m/s arrives in pieces separated by more than the
recovery window. This measures exactly that: every maximal episode at or above
7.0 m/s, and how many seconds it lasted.

A real effort lasts 2-4 s. If an engine player's high-speed time is a shower
of 0.1-0.3 s blips, the movement has no sprints in it -- the counter is right
and the locomotion is wrong.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

MPS = 7.0
GAP = 0.35          # bridge a single missed sample (0.2 s at 5 Hz)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "tools/diag/_raw_player_path.json"
    raw = json.load(open(path, encoding="utf-8"))

    print(f"{'player':20s} {'episodes':>8s} {'total s':>8s} "
          f"{'mean s':>7s} {'max s':>6s} {'>=1s':>5s}  duration histogram")
    print("-" * 96)
    for name in ("Bruno Reis", "Tariq El-Amin", "Tavish Crane",
                 "Finn Larsson", "Kwame Asante", "Luca Ferrini"):
        rows = raw.get(name)
        if not rows:
            continue
        samples = sorted((float(r[0]), float(r[1]), float(r[2])) for r in rows)
        # speed between consecutive samples
        eps = []
        cur = 0.0
        last_hot = None
        total = 0.0
        for i in range(1, len(samples)):
            t0, x0, y0 = samples[i - 1]
            t1, x1, y1 = samples[i]
            dt = t1 - t0
            if dt <= 0:
                continue
            v = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt
            if v >= MPS:
                total += dt
                if last_hot is not None and (t1 - last_hot) <= GAP:
                    cur += dt
                else:
                    if cur > 0:
                        eps.append(cur)
                    cur = dt
                last_hot = t1
        if cur > 0:
            eps.append(cur)
        if not eps:
            print(f"{name:20s} {'--':>8s}  no effort reached {MPS} m/s")
            continue
        hist = defaultdict(int)
        for d in eps:
            hist[min(4, int(d))] += 1
        h = "  ".join(f"{k}s:{hist[k]}" for k in sorted(hist))
        print(f"{name:20s} {len(eps):8d} {total:8.1f} "
              f"{total / len(eps):7.2f} {max(eps):6.2f} "
              f"{sum(1 for d in eps if d >= 1.0):5d}  {h}")


if __name__ == "__main__":
    main()
