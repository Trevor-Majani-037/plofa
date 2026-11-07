"""Forensics: where do big position steps come from, in the RAW 5 Hz path?

`replay.json` is decimated to 2 Hz, which can turn a legal multi-second
set-piece run-in into what looks like one teleport when two writers cover
the same window. This reads the un-decimated dump written by
`pitch_replay._dump_raw_path` (PLOFA_TRACE_JUMPS=1) and, for every step
larger than `--min-m`, prints the surrounding samples with their source.

Usage:
    python tools/diag/_diag_raw_jumps.py [--min-m 8] [--top 25]
"""
from __future__ import annotations

import argparse
import json
import math


def load(path: str):
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for name, rows in d.items():
        out[name] = [(float(r[0]), float(r[1]), float(r[2]),
                      (r[3] if len(r) > 3 else "?")) for r in rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-m", type=float, default=8.0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--path", default="tools/diag/_raw_player_path.json")
    a = ap.parse_args()

    players = load(a.path)
    hits = []
    for name, rows in players.items():
        for i in range(1, len(rows)):
            t0, x0, y0, s0 = rows[i - 1]
            t1, x1, y1, s1 = rows[i]
            d = math.hypot(x1 - x0, y1 - y0)
            if d >= a.min_m:
                hits.append((d, t1 - t0, name, i, rows, s0, s1))
    hits.sort(key=lambda h: -h[0])

    print(f"raw samples: {sum(len(v) for v in players.values())} "
          f"across {len(players)} players")
    print(f"steps >= {a.min_m} m: {len(hits)}")
    print(f"{'t':>9s} {'dt':>5s} {'dist':>6s} {'v':>6s}  player / sources")
    for d, dt, name, i, rows, s0, s1 in hits[:a.top]:
        print(f"{rows[i][0]:9.1f} {dt:5.2f} {d:6.1f} "
              f"{d / max(dt, 1e-6):6.1f}  {name}  [{s0}->{s1}]")

    if hits:
        d, dt, name, i, rows, s0, s1 = hits[0]
        print(f"\ncontext for the worst step: {name}")
        lo = max(0, i - 8)
        for r in rows[lo:i + 8]:
            mark = "  <<<" if r is rows[i] else ""
            print(f"  {r[0]:9.2f} {r[1]:6.1f} {r[2]:6.1f}  {r[3]:9s}{mark}")


if __name__ == "__main__":
    main()
