"""Off-ball actor-honesty diagnostic (2026-09-20).

Measures how much the runner's perception gate actually MOVES the press
brains' decisions over a few matches (production config, actor-honesty ON):

  1. SHRINK - gated vision: how many of the 21 other players the runner
     actually "sees" each press read (vs the omniscient all).
  2. MELT   - how often p(press) with HONEST actors flips by >0.05 vs the
     SAME brain fed TRUE actor lists, broken down by position.

Ball-vision's ball slot was almost inert (0.4% flips, mean |dp| 0.004).
Actors SHOULD move more - defender proximity drives the off-ball vector.
If melt is large here, retraining the off-ball brains on perceived-actor
corpora is the justified next lever; if it's small too, the press brains
are low-influence and evolution is the lever instead.

Run:  python scripts/diagnose_offball_actor_perception.py [--seeds 42,43,44]
"""
from __future__ import annotations

import os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception import get_perception_config, set_perception, PerceptionConfig
from pitch_replay import run_scratch_match
import perception as pc
import offball_brain_wiring as obw

ROWS = []    # (position, p_honest, p_true)
SEEN = []    # (n_seen, n_total)

_real_gate = pc.perceive_offball_actors
_real_off = obw.offball_press_prob


def _diag_gate(player, x, y, teammates=None, defenders=None,
               position_engine=None, attacks_right=None,
               config=None, return_meta=False):
    tms, defs, eng = _real_gate(player, x, y, teammates=teammates,
                                defenders=defenders,
                                position_engine=position_engine,
                                attacks_right=attacks_right,
                                config=config,
                                return_meta=return_meta)
    total = len(teammates or []) + len(defenders or [])
    SEEN.append((len(tms) + len(defs), total))
    return tms, defs, eng


def _diag_off(engine, pname, position, team, ball_x, ball_y, danger_t,
              minute, score_diff):
    if position not in ("LW", "RW", "CM"):
        return _real_off(engine, pname, position, team, ball_x, ball_y,
                         danger_t, minute, score_diff)
    p_honest = _real_off(engine, pname, position, team, ball_x, ball_y,
                         danger_t, minute, score_diff)
    old = get_perception_config()
    try:
        # TRUE actor lists (no gate, no noise); ball-vision unchanged.
        set_perception(PerceptionConfig(offball_actor_perception=False))
        p_true = _real_off(engine, pname, position, team, ball_x, ball_y,
                           danger_t, minute, score_diff)
    finally:
        set_perception(old)
    if p_honest is None or p_true is None:
        return p_honest
    ROWS.append((position, p_honest, p_true))
    return p_honest


def run(seeds):
    pc.perceive_offball_actors = _diag_gate
    obw.offball_press_prob = _diag_off
    try:
        for seed in seeds:
            run_scratch_match(seed=seed, verbose=False)
    finally:
        pc.perceive_offball_actors = _real_gate
        obw.offball_press_prob = _real_off


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    run(seeds)

    if SEEN:
        tot = sum(s for _n, s in SEEN)
        n = len(SEEN)
        ratios = [s / max(t, 1) for s, t in SEEN]
        print(f"press reads: {n} | {len(seeds)} matches")
        print(f"  omniscient total actors: {tot}  -> "
              f"gated mean {round(statistics.mean(ratios),2)} "
              f"({round(100 * statistics.mean(ratios),1)}% of the 21)")
        ltd = sum(1 for s, t in SEEN if s < t)
        print(f"  reads where gating hid >= 1 actor: "
              f"{ltd} ({round(100 * ltd / max(n, 1),1)}%)")

    if ROWS:
        flips = [r for r in ROWS if abs(r[1] - r[2]) > 0.05]
        dps = [abs(r[1] - r[2]) for r in ROWS]
        print(f"\np(press) dual-read (honest vs true actors): {len(ROWS)}")
        print(f"  flips >0.05: {len(flips)} "
              f"({round(100 * len(flips) / len(ROWS),1)}%)")
        print(f"  mean |dp|: {round(statistics.mean(dps),3)}")
        for pos in ("LW", "RW", "CM"):
            sub = [r for r in ROWS if r[0] == pos]
            if not sub:
                continue
            f = sum(1 for r in sub if abs(r[1] - r[2]) > 0.05)
            d = [abs(r[1] - r[2]) for r in sub]
            print(f"  {pos}: {len(sub)} reads, {f} flips "
                  f"({round(100 * f / len(sub),1)}%), mean |dp| "
                  f"{round(statistics.mean(d),3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())