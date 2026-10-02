"""Off-ball movement-honesty diagnostic (2026-09-20).

The engine's movement integrator now runs the CHASE + INVOLVEMENT gates off
each player's PERSONAL perceived ball (team SHAPE keeps the true ball).
Two imprints, both sourced from the physics store (real GPS distance +
sprints) and live seam samples:

  1. FEED SANITY  - of every movement seam read, how often the player's
                    belief disagreed with the truth (|perceived - true|),
                    and how often that disagreement exceeded the CHASE
                    (12.5 m) + INVOLVEMENT (30 m) sandwiches -> phantom
                    windows (would-sprint at a ghost / would-miss the real).
  2. PHYSICS BODY - aggregate distance, sprint and high-speed-sprint counts
                    across the match (engine.physics_totals), OFF vs ON.

If honest movement is a live seam, OFF vs ON matches should show a physics
delta (sprint counts/quade shift) and a >0 feed disagreement rate.  If the
feed is a ghost again (near-zero disagreement), the movement integrator is
too coarse to hear honest perception either.

Run:  python scripts/diagnose_ball_vision_movement.py [--seeds 42,43,44]
"""
from __future__ import annotations

import os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception import (get_perception_config, set_perception,
                        PerceptionConfig)
from pitch_replay import run_scratch_match
import match_engine as me

CHASE_TRIGGER = 12.5
INVOLVE_GATE = 30.0

GLOBAL = {"reads": 0, "blind": 0, "ghost_total": 0.0,
          "phantom_windows": 0, "miss_windows": 0}

_real_movement = me.movement_ball


def _diag_movement(engine, pname, position, ar, tx, ty, minute):
    px, py = _real_movement(engine, pname, position, ar, tx, ty, minute)
    GLOBAL["reads"] += 1
    dx = px - tx
    dy = py - ty
    err = float((dx * dx + dy * dy) ** 0.5)
    if err > 1e-6:
        GLOBAL["blind"] += 1
        GLOBAL["ghost_total"] += err
    st = engine.position_engine.states.get(pname)
    if st is not None:
        cx, cy = st.current_x, st.current_y
        belief_dist = float(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)
        true_dist = float(((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5)
        if belief_dist < CHASE_TRIGGER and true_dist > CHASE_TRIGGER * 3:
            GLOBAL["phantom_windows"] += 1   # about to sprint at a ghost
        if true_dist < CHASE_TRIGGER and belief_dist > CHASE_TRIGGER * 3:
            GLOBAL["miss_windows"] += 1      # about to miss the real thing
    return px, py


def _run(seeds, movement_on):
    set_perception(PerceptionConfig(ball_vision=True,
                                    ball_vision_movement=movement_on))
    rows = []
    for seed in seeds:
        for n in list(GLOBAL):
            GLOBAL[n] = 0
        me.movement_ball = _diag_movement
        row = None
        try:
            r = run_scratch_match(seed=seed, verbose=False)
            tot = r.physics_totals()
            row = {
                "distance": sum(v["distance_m"] for v in tot.values()),
                "sprints": sum(v["sprint_count"] for v in tot.values()),
                "hi": sum(v["high_speed_sprint_count"] for v in tot.values()),
                "goals": r.home_goals + r.away_goals,
                "poss": r.home_possession_pct,
                "reads": GLOBAL["reads"], "blind": GLOBAL["blind"],
                "ghost": GLOBAL["ghost_total"],
                "phantom": GLOBAL["phantom_windows"],
                "miss": GLOBAL["miss_windows"],
            }
        finally:
            me.movement_ball = _real_movement
        if row is not None:
            rows.append(row)
    return rows


def _mean(rows, key, dec=1):
    return round(statistics.mean(r[key] for r in rows), dec), \
        round(statistics.pstdev(r[key] for r in rows), dec)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    off = _run(seeds, movement_on=False)
    on = _run(seeds, movement_on=True)

    feeds = [(r["reads"], r["blind"], r["ghost"], r["phantom"], r["miss"])
             for r in on]
    n_reads = sum(f[0] for f in feeds)
    n_blind = sum(f[1] for f in feeds)
    print(f"seam feed (movement ON, {len(seeds)} matches):")
    print(f"  reads: {n_reads} | blind reads "
          f"({n_blind} = {round(100 * n_blind / max(n_reads, 1), 1)}%)")
    g = sum(f[2] for f in feeds) / max(n_blind, 1)
    print(f"  mean |perceived - true| on blind reads: {round(g, 1)} m")
    ph = sum(f[3] for f in feeds); ms = sum(f[4] for f in feeds)
    print(f"  phantom-chase windows: {ph} | missed-close-down windows: {ms}")

    print(f"\nphysics body: {'metric':<10}{'OFF (mean)':<20}{'ON (mean)':<18}"
          f"{'delta':<8}")
    for key, label in (("distance", "km"),
                       ("sprints", "sprints"),
                       ("hi", "hi-speed"),
                       ("goals", "goals"),
                       ("poss", "poss%")):
        mo, so = _mean(off, key, dec=1)
        mn, sn = _mean(on, key, dec=1)
        if key == "distance":
            mo, mn = mo / 1000.0, mn / 1000.0
        print(f"{label:<10}{mo:<14}(sd {so:<4}){mn:<12}(sd {sn:<4})"
              f"{round(mn - mo, 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())