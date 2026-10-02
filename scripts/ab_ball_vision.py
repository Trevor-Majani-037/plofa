"""Ball-vision A/B (2026-09-20): 12 matches with ball-vision ON vs OFF.

Both batches run the SAME seeds through run_scratch_match so the only
difference is the perceived-ball seam.  The off-ball press brains (LW/RW/CM)
read a PERCEIVED ball when ball-vision is ON; OFF is today's omniscient feed.

Expectation if the seam works: possession barely moves, but build-up from a
turned-over / blind ball gets slightly worse (this run only reports the
headline match totals — goals, xG, possession, tempo).

Run:  python scripts/ab_ball_vision.py  |  (or with --verbose)
"""
from __future__ import annotations

import os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception import set_perception, PerceptionConfig
from pitch_replay import run_scratch_match

SEEDS = list(range(42, 48))   # 42..47 inclusive = 6 matches per arm


def _arm_config(on, seam):
    if seam == "actor":
        return PerceptionConfig(offball_actor_perception=on,
                                ball_vision=True)
    if seam == "movement":
        return PerceptionConfig(ball_vision_movement=on,
                                ball_vision=True)
    return PerceptionConfig(ball_vision=on, offball_actor_perception=True)


def _batch(on: bool, seam: str, seeds, out=None, tag="match"):
    set_perception(_arm_config(on, seam))
    rows = []
    for seed in seeds:
        r = run_scratch_match(seed=seed, verbose=False)
        rows.append({
            "seed": seed,
            "arm": "ON" if on else "OFF",
            "goals": r.home_goals + r.away_goals,
            "xg": r.home_xg + r.away_xg,
            "poss": r.home_possession_pct,
        })
        print(f"  [{tag} {seed} done: "
              f"{rows[-1]['goals']} goals, {rows[-1]['xg']:.3f} xG]")
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(f"{rows[-1]['arm']},{rows[-1]['seed']},"
                         f"{rows[-1]['goals']},{rows[-1]['xg']},"
                         f"{rows[-1]['poss']}\n")
    return rows


def _mean(rows, key):
    return round(statistics.mean(r[key] for r in rows), 3), \
        round(statistics.pstdev(r[key] for r in rows), 3)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", choices=("ball", "actor", "movement"),
                    default="ball",
                    help="which honesty seam to A/B (other stays ON)")
    ap.add_argument("--seeds", default="42,43,44,45,46,47",
                    help="comma-separated seeds for the batch")
    ap.add_argument("--batch", choices=("both", "off", "on"),
                    default="both",
                    help="run both arms, or just one (chunked runs)")
    ap.add_argument("--out",
                    help="append raw rows (arm,seed,goals,xg,poss) to a file")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.batch in ("both", "off"):
        off = _batch(False, args.seam, seeds, out=args.out, tag="OFF")
    else:
        off = []
    if args.batch in ("both", "on"):
        on = _batch(True, args.seam, seeds, out=args.out, tag="ON")
    else:
        on = []
    label = args.seam
    if args.seam == "ball":
        label = "ball-vision"
    elif args.seam == "actor":
        label = "actor-honesty"
    else:
        label = "movement-honesty"
    print(f"{label} A/B (same {len(seeds)} seeds, run_scratch_match) "
          f"[{args.batch}]")
    print(f"{'metric':<8} {'OFF (mean)':<20} {'ON (mean)':<20} {'delta':<10}")
    for key, label in (("goals", "goals"), ("xg", "xG"), ("poss", "poss%")):
        mo, so = _mean(off, key) if off else (0.0, 0.0)
        mn, sn = _mean(on, key) if on else (0.0, 0.0)
        delta = round(mn - mo, 3) if (off and on) else float("nan")
        print(f"{label:<8} {mo:<10}(sd {so:<7}) {mn:<10}(sd {sn:<7}) "
              f"{delta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())