"""Ball-vision diagnostic (2026-09-20): does the seam actually move presses?

Measures per press-read over a few matches (ball-vision ON, the default):
  1. STALENESS  - how often a press read sees a ball (seen) vs a stale
     "last known" estimate (not seen), and the mean/max error (m) of the
     perceived ball vs the true ball when stale.
  2. SIGMA      - expected position uncertainty (m) for visible and stale.
  3. DECISION MELT - how often the PERCEIVED ball flips p(press) by >0.05
     vs the SAME brain fed the TRUE ball, broken down by position.

If flips are rare (<~2%) the seam barely moves decisions and retraining is
unnecessary; if they're common, retrain the off-ball brains on perceived
corpora (the train/inference distribution mismatch is real).

Run:  python scripts/diagnose_ball_vision.py [--seeds 42,43,44]
"""
from __future__ import annotations

import os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception import get_perception_config, set_perception, PerceptionConfig
from pitch_replay import run_scratch_match
import ball_vision as bv
import offball_brain_wiring as obw

import math

ROWS = []        # (position, flip, dp)
BALL = []        # (pname, seen, sigma, err_m)


_real_off = obw.offball_press_prob
_real_ball = bv.perceive_ball


def _diag_ball(engine, pname, player, attacks_right, true_x, true_y, minute):
    px, py, seen, sigma = _real_ball(engine, pname, player, attacks_right,
                                     true_x, true_y, minute)
    err = float(math.hypot(px - true_x, py - true_y))
    BALL.append((pname, seen, sigma, err))
    return px, py, seen, sigma


def _diag_off(engine, pname, position, team, ball_x, ball_y, danger_t,
              minute, score_diff):
    if position not in ("LW", "RW", "CM"):
        return _real_off(engine, pname, position, team, ball_x, ball_y,
                         danger_t, minute, score_diff)
    p_perc = _real_off(engine, pname, position, team, ball_x, ball_y,
                       danger_t, minute, score_diff)
    old = get_perception_config()
    try:
        set_perception(PerceptionConfig(ball_vision=False))
        p_true = _real_off(engine, pname, position, team, ball_x, ball_y,
                           danger_t, minute, score_diff)
    finally:
        set_perception(old)
    if p_perc is None or p_true is None:
        return p_perc
    ROWS.append((position, abs(p_perc - p_true) > 0.05,
                 abs(p_perc - p_true)))
    return p_perc


def run(seeds):
    bv.perceive_ball = _diag_ball
    obw.offball_press_prob = _diag_off
    try:
        for seed in seeds:
            run_scratch_match(seed=seed, verbose=False)
    finally:
        bv.perceive_ball = _real_ball
        obw.offball_press_prob = _real_off


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    run(seeds)

    n = len(BALL)
    stale = [b for b in BALL if not b[1]]
    visible = [b for b in BALL if b[1]]
    print(f"press reads: {n}  |  {len(seeds)} matches")
    print(f"  stale reads: {len(stale)} ({round(100*len(stale)/max(n,1),1)}%)")
    if stale:
        errs = [b[3] for b in stale]
        sigs = [b[2] for b in stale]
        print(f"  stale mean err: {round(statistics.mean(errs),1)} m  "
              f"(max {round(max(errs),1)} m)")
        print(f"  stale mean sigma: {round(statistics.mean(sigs),1)} m")
    if visible:
        sigs = [b[2] for b in visible if b[2] > 1e-9]
        print(f"  visible mean sigma: "
              f"{round(statistics.mean(sigs),1) if sigs else 0} m")

    if ROWS:
        flips = [r for r in ROWS if r[1]]
        dps = [r[2] for r in ROWS]
        print(f"\np(press) dual-read: {len(ROWS)}")
        print(f"  flips >0.05: {len(flips)} ({round(100*len(flips)/len(ROWS),1)}%)")
        print(f"  mean |dp|: {round(statistics.mean(dps),3)}")
        for pos in ("LW", "RW", "CM"):
            sub = [r for r in ROWS if r[0] == pos]
            if not sub:
                continue
            f = sum(1 for r in sub if r[1])
            print(f"  {pos}: {len(sub)} reads, {f} flips "
                  f"({round(100*f/len(sub),1)}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())