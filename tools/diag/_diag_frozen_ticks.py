"""Do they actually STAND? Measured at the raw 0.2 s tick, no window.

The engine books band time from the pace it ORDERED. The viewer draws the
position it RECORDED. Those disagree, and the disagreement is the story:
a body inside its arrival band can be ordered to jog and still not move.

Two numbers per player:
  frozen  -- share of ticks with no displacement at all (< 2 cm)
  moving  -- share of non-frozen ticks, by achieved band
"""
from __future__ import annotations

import json
import sys
from collections import Counter

FROZEN = 0.02


def band(v: float) -> str:
    if v < 0.5:
        return "standing"
    if v < 2.0:
        return "walking"
    if v < 4.2:
        return "jogging"
    if v < 7.0:
        return "running"
    return "sprinting"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "tools/diag/_raw_player_path.json"
    raw = json.load(open(path, encoding="utf-8"))

    tot = Counter()
    frozen_s = 0.0
    total_s = 0.0
    rows_out = []
    for name, rows in raw.items():
        s = sorted((float(r[0]), float(r[1]), float(r[2])) for r in rows)
        c = Counter()
        fs = 0.0
        ts = 0.0
        for i in range(1, len(s)):
            t0, x0, y0 = s[i - 1]
            t1, x1, y1 = s[i]
            dt = t1 - t0
            if dt <= 0 or dt > 0.5:
                continue
            d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            ts += dt
            if d < FROZEN:
                fs += dt
            else:
                c[band(d / dt)] += dt
        total_s += ts
        frozen_s += fs
        for k, v in c.items():
            tot[k] += v
        if ts > 300.0:
            rows_out.append((name, ts, fs, c))

    for name, ts, fs, c in sorted(rows_out, key=lambda r: -r[1])[:30]:
        mv = ts - fs
        share = " ".join(f"{k[:4]} {100.0 * c[k] / ts:4.1f}%" for k in
                         ("standing", "walking", "jogging", "running", "sprinting"))
        print(f"{name:20s} frozen {100.0 * fs / ts:5.1f}%   {share}")

    print("-" * 88)
    print(f"ALL PLAYERS  frozen {100.0 * frozen_s / total_s:5.1f}% of "
          f"{total_s / 60.0:,.0f} player-min")
    for k in ("standing", "walking", "jogging", "running", "sprinting"):
        print(f"  {k:10s} {100.0 * tot[k] / total_s:5.1f}%")


if __name__ == "__main__":
    main()
