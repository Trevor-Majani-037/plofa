"""WHERE in the walk band are they, and which branch put them there?

The walk band is 0.5-2.0 m/s, and two thirds of the match sits in it. That
is only actionable if we know WHERE inside the band the mass is, because
each engine branch runs at a fixed multiple of a role constant:

    involvement gate, ball far side   jog * 0.30   -> 0.525 - 0.60 m/s
    block shuffle (line adjusting)    jog * 0.55   -> 0.96 - 1.10 m/s
    patrol orbit (settled outfield)   jog * 0.80   -> 1.40 - 1.60 m/s
    plain shape travel                jog          -> 1.75 - 2.00 m/s

So the histogram of ACHIEVED speed is a fingerprint: a spike at 1.5 says
the time is patrol, a spike at 0.57 says the involvement gate is holding
bodies at a crawl, a spike at 1.05 says block shuffle.

Speeds are measured over a 0.5 s window so single-sample jitter cannot
create a fake mode.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

WIN = 5          # samples; at 5 Hz this is 0.5 s


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "tools/diag/_raw_player_path.json"
    raw = json.load(open(path, encoding="utf-8"))

    hist = Counter()
    total_s = 0.0
    band = Counter()
    for name, rows in raw.items():
        s = sorted((float(r[0]), float(r[1]), float(r[2])) for r in rows)
        for i in range(WIN, len(s)):
            t0, x0, y0 = s[i - WIN]
            t1, x1, y1 = s[i]
            dt = t1 - t0
            if dt <= 0:
                continue
            v = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt
            total_s += dt / WIN
            if v < 0.5:
                band["standing"] += dt / WIN
            elif v < 2.0:
                band["walking"] += dt / WIN
                hist[round(v / 0.05) * 0.05] += dt / WIN
            elif v < 4.2:
                band["jogging"] += dt / WIN
            elif v < 7.0:
                band["running"] += dt / WIN
            else:
                band["sprinting"] += dt / WIN

    n = sum(band.values()) or 1.0
    print("bands, measured off the recorded path (0.5 s window)")
    for b in ("standing", "walking", "jogging", "running", "sprinting"):
        print(f"  {b:10s} {100.0 * band[b] / n:5.1f}%   {band[b] / 60.0:7.0f} player-min")

    print(f"\nWALK BAND ONLY -- where the mass sits ({band['walking'] / 60.0:.0f} player-min)")
    print(f"{'speed':>6s} {'% of walk':>9s} {'% of match':>10s}  bar")
    walk = band["walking"] or 1.0
    for v, sec in sorted(hist.items(), key=lambda kv: -kv[1])[:14]:
        print(f"{v:6.2f} {100.0 * sec / walk:8.1f}% {100.0 * sec / n:9.1f}%  "
              + "#" * int(60.0 * sec / walk))

    print("\nbranch fingerprints")
    for label, lo, hi in (
        ("involvement gate  jog*0.30", 0.48, 0.65),
        ("block shuffle     jog*0.55", 0.90, 1.16),
        ("patrol orbit      jog*0.80", 1.33, 1.66),
        ("plain travel      jog    ", 1.70, 2.00),
    ):
        sec = sum(s for v, s in hist.items() if lo <= v <= hi)
        print(f"  {label}  {100.0 * sec / walk:5.1f}% of walking  "
              f"({100.0 * sec / n:4.1f}% of match)")


if __name__ == "__main__":
    main()
