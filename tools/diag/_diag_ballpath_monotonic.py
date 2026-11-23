"""Which calibration seed breaks match_ball_path monotonicity, and why.

Replicates tests/test_calibration.py::_run_one EXACTLY (in-memory, no export,
no file writes) and prints the first timestamp inversion in the global ball
path. Env flags let the same code be run with the 2026-11-23 fixes disabled:
    PLOFA_LEGACY_PLAN_FEED=1   dead plan block
    PLOFA_LEGACY_CB_MICRO=1    flat 0.55 micro-carry rate

Run:
    python -X utf8 tools\\diag\\_diag_ballpath_monotonic.py 1 2 3 4 5 6
"""
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_match as R
from match_engine import MatchEngine, MatchConfig, MatchResult
from player_dna import SquadBuilder


def run_one(seed):
    random.seed(seed)
    home_squad = SquadBuilder.build(
        team_name=R.HOME_TEAM, starters=R.HOME_STARTERS, substitutes=R.HOME_SUBS,
        team_superstars=R.HOME_SUPERSTARS, set_piece_takers=R.HOME_SP_TAKERS,
    )
    away_squad = SquadBuilder.build(
        team_name=R.AWAY_TEAM, starters=R.AWAY_STARTERS, substitutes=R.AWAY_SUBS,
        team_superstars=R.AWAY_SUPERSTARS, set_piece_takers=R.AWAY_SP_TAKERS,
    )
    for p in (home_squad["starters"] + home_squad["substitutes"]
              + away_squad["starters"] + away_squad["substitutes"]):
        if p.name in R.SOUL_PLAYERS:
            p.dna.soul = R.SOUL_PLAYERS[p.name]

    cfg = MatchConfig(
        home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM, match_date=R.MATCH_DATE,
        matchday=R.MATCHDAY, season=R.SEASON, competition=R.COMPETITION,
        venue=R.VENUE, stadium_capacity=R.CAPACITY, referee=R.REFEREE,
        referee_strictness=R.STRICTNESS, is_derby=R.IS_DERBY,
    )
    eng = MatchEngine(cfg, R.HOME_STYLE, R.AWAY_STYLE)
    eng.set_squad(R.HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    eng.set_squad(R.AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    return eng.simulate()


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]
    legacy_plan = bool(os.environ.get("PLOFA_LEGACY_PLAN_FEED"))
    legacy_micro = bool(os.environ.get("PLOFA_LEGACY_CB_MICRO"))
    print(f"legacy_plan={legacy_plan} legacy_micro={legacy_micro} seeds={seeds}")
    for seed in seeds:
        res = run_one(seed)
        path = list(res.full_match_ball_path or [])
        bad = []
        for i in range(1, len(path)):
            if float(path[i]["t"]) < float(path[i - 1]["t"]):
                bad.append((i, path[i - 1]["t"], path[i]["t"],
                            path[i - 1].get("kind"), path[i].get("kind")))
            if len(bad) >= 3:
                break
        print(f"seed {seed}: samples={len(path)} inversions={len(bad)} "
              f"{bad[:2]}", flush=True)


if __name__ == "__main__":
    main()
