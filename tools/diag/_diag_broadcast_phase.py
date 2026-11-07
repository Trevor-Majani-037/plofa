"""Diagnose the three broadcast symptoms the user reports:

1. The ball stands alone for ~20 s and nobody goes for it.
2. Players cluster / stand on the same spot.
3. The ball keeps playing as if nobody is on the pitch.

Reads a saved replay.json (player_path + ball path, both on the global
clock) and reports, in half-second samples:

  * nearest player to the ball (mean / p90 / max), and the worst runs where
    the ball travels while the nearest body is far away
  * dead-ball runs: ball displacement < 0.5 m over >= 5 s, with how far the
    nearest body is and how many bodies are inside 10 m
  * crowding: pairs of players closer than 1.5 m, and the longest run of
    samples where >= 2 pairs are stacked

Usage:
    python tools/diag/_diag_broadcast_phase.py [path/to/replay.json]
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict

STEP = 0.5          # sampling step, seconds
DEAD_MOVE_M = 0.5   # ball displacement that still counts as "standing"
DEAD_MIN_S = 5.0    # report dead-ball runs at least this long
FAR_M = 15.0        # ball "alone" if nearest body is farther than this
STACK_M = 1.5       # two players this close are stacked


def load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def series_at(samples, t):
    """Last sample at or before t (samples are (t, x, y))."""
    lo, hi = 0, len(samples) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            best = samples[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main(path: str) -> None:
    d = load(path)
    ball = [(float(b["t"]), float(b["x"]), float(b["y"]))
            for b in d["ball"]]
    ball.sort(key=lambda r: r[0])
    players = {name: sorted((float(a), float(b), float(c))
                            for a, b, c in rows)
               for name, rows in d["player_path"].items()}
    end_t = min(ball[-1][0], max(s[0] for s in
                                 (rows[-1] for rows in players.values())))

    # ── samples ────────────────────────────────────────────────────────
    rows = []
    t = 0.0
    cursor = {name: 0 for name in players}
    while t <= end_t:
        b = series_at(ball, t)
        if b is None:
            t += STEP
            continue
        bx, by = b[1], b[2]
        near = None
        near_name = ""
        pos = []
        for name, samples in players.items():
            s = series_at(samples, t)
            if s is None:
                continue
            pos.append((name, s[1], s[2]))
            dxy = math.hypot(s[1] - bx, s[2] - by)
            if near is None or dxy < near:
                near, near_name = dxy, name
        rows.append({"t": t, "bx": bx, "by": by, "near": near,
                     "near_name": near_name, "pos": pos})
        t += STEP

    near_all = [r["near"] for r in rows if r["near"] is not None]
    near_all.sort()

    def pct(q):
        if not near_all:
            return float("nan")
        i = min(len(near_all) - 1, int(q * (len(near_all) - 1)))
        return near_all[i]

    print("=" * 72)
    print(f"file            {path}")
    print(f"duration        {end_t / 60.0:.1f} min, {len(rows)} samples @ {STEP}s")
    print(f"ball samples    {len(ball)}   player rows {len(players)}")
    print("-" * 72)
    print("BALL-TO-NEAREST-BODY DISTANCE (m)")
    print(f"  mean {statistics.mean(near_all):5.1f}  p50 {pct(.5):5.1f}  "
          f"p90 {pct(.9):5.1f}  p99 {pct(.99):5.1f}  max {pct(1.0):5.1f}")
    alone = [r for r in rows if r["near"] is not None and r["near"] > FAR_M]
    print(f"  samples with ball > {FAR_M:.0f} m from every body: "
          f"{len(alone)} ({100.0 * len(alone) / max(1, len(rows)):.1f}%)")

    # worst contiguous "alone" runs
    runs = []
    cur = []
    for r in rows:
        if r["near"] is not None and r["near"] > FAR_M:
            cur.append(r)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    runs.sort(key=lambda run: -len(run))
    print(f"  worst 'ball alone' runs (>{FAR_M:.0f} m, >=1 s): {len(runs)}")
    for run in runs[:6]:
        span = run[-1]["t"] - run[0]["t"]
        moved = math.hypot(run[-1]["bx"] - run[0]["bx"],
                           run[-1]["by"] - run[0]["by"])
        print(f"    {int(run[0]['t']) // 60:02d}:{run[0]['t'] % 60:04.1f} "
              f"len {span + STEP:5.1f}s  ball moved {moved:5.1f} m  "
              f"mean gap {statistics.mean(x['near'] for x in run):5.1f} m")

    # ── dead ball ──────────────────────────────────────────────────────
    print("-" * 72)
    print("DEAD-BALL RUNS (ball moves < 0.5 m)")
    dead = []
    cur = [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if math.hypot(r["bx"] - prev["bx"], r["by"] - prev["by"]) <= 0.02:
            cur.append(r)
        else:
            dead.append(cur)
            cur = [r]
    dead.append(cur)
    long_dead = [c for c in dead if (c[-1]["t"] - c[0]["t"]) >= DEAD_MIN_S]
    total_dead = sum(c[-1]["t"] - c[0]["t"] for c in long_dead)
    print(f"  runs >= {DEAD_MIN_S:.0f}s: {len(long_dead)}   "
          f"total {total_dead / 60.0:.1f} min "
          f"({100.0 * total_dead / max(1.0, end_t):.1f}% of match)")
    long_dead.sort(key=lambda c: -(c[-1]["t"] - c[0]["t"]))
    for c in long_dead[:8]:
        span = c[-1]["t"] - c[0]["t"]
        near = statistics.mean(x["near"] for x in c)
        inside10 = statistics.mean(
            sum(1 for _, px, py in x["pos"]
                if math.hypot(px - x["bx"], py - x["by"]) <= 10.0) for x in c)
        print(f"    {int(c[0]['t']) // 60:02d}:{c[0]['t'] % 60:04.1f} "
              f"len {span:5.1f}s  near {near:5.1f} m  "
              f"bodies within 10 m {inside10:4.1f}  "
              f"ball ({c[0]['bx']:.0f},{c[0]['by']:.0f})")

    # ── crowding ──────────────────────────────────────────────────────
    print("-" * 72)
    print("CROWDING (pairs closer than 1.5 m)")
    per_sample = []
    for r in rows:
        pos = r["pos"]
        n = 0
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if math.hypot(pos[i][1] - pos[j][1],
                              pos[i][2] - pos[j][2]) < STACK_M:
                    n += 1
        per_sample.append((r["t"], n))
    stacked = [s for s in per_sample if s[1] >= 2]
    worst = sorted(per_sample, key=lambda s: -s[1])[:6]
    print(f"  samples with >=2 stacked pairs: {len(stacked)} "
          f"({100.0 * len(stacked) / max(1, len(per_sample)):.1f}%)")
    print(f"  peak stacked pairs {worst[0][1] if worst else 0}   "
          f"worst times: " + ", ".join(
              f"{int(t) // 60:02d}:{t % 60:04.1f} ({n})" for t, n in worst))

    # longest consecutive run of stacking
    runs, cur = [], []
    for t, n in per_sample:
        if n >= 2:
            cur.append((t, n))
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    runs.sort(key=lambda c: -len(c))
    print(f"  longest stacked runs:")
    for c in runs[:5]:
        print(f"    {int(c[0][0]) // 60:02d}:{c[0][0] % 60:04.1f} "
              f"len {len(c) * STEP:5.1f}s  peak {max(n for _, n in c)} pairs")

    # ── who is alone with the ball ────────────────────────────────────
    print("-" * 72)
    print("WHO THE BALL IS NEAREST TO (share of samples)")
    share = defaultdict(int)
    for r in rows:
        share[r["near_name"]] += 1
    for name, n in sorted(share.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {name:24s} {100.0 * n / max(1, len(rows)):5.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "plofa_broadcast/replay.json")
