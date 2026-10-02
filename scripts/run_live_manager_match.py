"""Run one scratch match with LIVE brain-managers.

Wires the evolved brain (manager_brains/v1/test_manager.json) + neutral
minds + per-club persisted memory (manager_brains/live/<club>/), flips
match_engine.USE_MANAGER_BRAIN for the match only, then saves each club's
mind + memory so the NEXT match starts with past experience.

Determinism: same result across processes under a fixed PYTHONHASHSEED
*provided the persisted live memory is identical* (memory evolves, so
back-to-back runs legitimately differ — that is the point).

Usage:
    $env:PYTHONHASHSEED="0"; py -3 scripts\\run_live_manager_match.py --seed 42
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import match_engine as _me
from pitch_replay import run_scratch_match


def main() -> int:
    ap = argparse.ArgumentParser(description="Live brain-manager scratch match")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--brain", default=None,
                    help="brain JSON path (default: manager_brains/v1/test_manager.json)")
    ap.add_argument("--no-persist", action="store_true",
                    help="do not save mind+memory afterwards")
    args = ap.parse_args()

    # The module default is ON since 2026-09-20 (live manager is the default
    # cerebrum); run_scratch_match pins the flag around the simulation and
    # restores it afterwards whatever it was before.
    _old = _me.USE_MANAGER_BRAIN
    assert _me.USE_MANAGER_BRAIN is True, "module default is now ON (Phase 10)"
    r = run_scratch_match(seed=args.seed, verbose=False,
                          use_manager_brain=True, brain_path=args.brain,
                          persist_memory=not args.no_persist)
    print(f"LIVE seed={args.seed}  {r.home_goals}-{r.away_goals}  "
          f"xG={round(r.home_xg, 2)}-{round(r.away_xg, 2)}  "
          f"poss={round(r.home_possession_pct, 1)}")
    print(f"  flag restored: {_me.USE_MANAGER_BRAIN is _old}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
