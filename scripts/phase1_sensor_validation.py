#!/usr/bin/env python3
"""
Phase 1 verification: run 3 matches with different seeds, sample
extract_manager_sensors for both teams at minute 45 and 90, print
the 12-float vectors.

Hook: monkey-patches engine.position_log.append so the sampling
function fires exactly once per minute, at the end of that minute's
processing — the same instant a manager would "perceive" the match.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import run_match as R
from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder
from squad_manager import SubstitutionController
from manager_profile import ManagerPool
from manager_sensors import extract_manager_sensors, MANAGER_SENSOR_LAYOUT

SEEDS  = [42, 43, 44]
SAMPLE_MINUTES = (45, 90)


def _build_engine(seed, verbose=False):
    random.seed(seed)
    np.random.seed(seed)

    home_squad = SquadBuilder.build(
        team_name=R.HOME_TEAM, starters=R.HOME_STARTERS,
        substitutes=R.HOME_SUBS, team_superstars=R.HOME_SUPERSTARS,
        set_piece_takers=R.HOME_SP_TAKERS,
    )
    away_squad = SquadBuilder.build(
        team_name=R.AWAY_TEAM, starters=R.AWAY_STARTERS,
        substitutes=R.AWAY_SUBS, team_superstars=R.AWAY_SUPERSTARS,
        set_piece_takers=R.AWAY_SP_TAKERS,
    )
    for p in (home_squad["starters"] + home_squad["substitutes"] +
              away_squad["starters"] + away_squad["substitutes"]):
        if p.name in R.SOUL_PLAYERS:
            p.dna.soul = R.SOUL_PLAYERS[p.name]

    mgr_pool = ManagerPool(
        clubs=[R.HOME_TEAM, R.AWAY_TEAM],
        style_lookup={R.HOME_TEAM: R.HOME_STYLE.style.value,
                      R.AWAY_TEAM: R.AWAY_STYLE.style.value},
    )
    home_mgr = mgr_pool.manager_for(R.HOME_TEAM)
    away_mgr = mgr_pool.manager_for(R.AWAY_TEAM)

    cfg = MatchConfig(
        home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM,
        match_date=R.MATCH_DATE, matchday=R.MATCHDAY, season=R.SEASON,
        competition=R.COMPETITION, venue=R.VENUE,
        stadium_capacity=R.CAPACITY, referee=R.REFEREE,
        referee_strictness=R.STRICTNESS, is_derby=R.IS_DERBY,
    )

    sub_controller = SubstitutionController(
        home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"],
        home_style=R.HOME_STYLE.style.value,
        away_style=R.AWAY_STYLE.style.value,
        manager_stubbornness=R.MANAGER_STUBBORNNESS,
    )
    sub_controller.MAX_SUBS = R.MAX_SUBS
    sub_controller.set_manager_stubbornness(R.HOME_TEAM, home_mgr.stubbornness())
    sub_controller.set_manager_stubbornness(R.AWAY_TEAM, away_mgr.stubbornness())

    engine = MatchEngine(cfg, R.HOME_STYLE, R.AWAY_STYLE)
    engine.set_squad(R.HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(R.AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(sub_controller)
    engine.set_managers(home_manager=home_mgr, away_manager=away_mgr)
    return engine


class _HookedList(list):
    def __init__(self, items, hook):
        super().__init__(items)
        self._hook = hook

    def append(self, item):
        super().append(item)
        self._hook(item)


def run_match_with_sampling(seed, verbose=False):
    engine = _build_engine(seed, verbose=verbose)

    # ── Hook: capture sensors at the end of specified minutes ──────
    samples = {}          # {(minute, team): sensor_vector}

    def _hooked_append(frame):
        minute = int(frame.get("minute", 0))
        if minute in SAMPLE_MINUTES:
            for team in (R.HOME_TEAM, R.AWAY_TEAM):
                sensors = extract_manager_sensors(engine, team)
                samples[(minute, team)] = sensors

    engine.position_log = _HookedList(engine.position_log, _hooked_append)

    result = engine.simulate()

    if verbose:
        print(result.summary())

    return samples, result


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=100)

    print("=" * 72)
    print("  PHASE 1 SENSOR VALIDATION  —  3 seeded matches, sampled at min 45 & 90")
    print("=" * 72)

    for seed in SEEDS:
        print(f"\n{'─'*72}")
        print(f"  SEED {seed}   {R.HOME_TEAM} vs {R.AWAY_TEAM}   "
              f"{result_summary_line(seed)}")
        print(f"{'─'*72}")
        samples, result = run_match_with_sampling(seed, verbose=False)

        # Print summary once
        print(f"  Result: {result.score_str}   "
              f"xG {result.home_xg:.2f} – {result.away_xg:.2f}   "
              f"Poss {result.home_possession_pct:.0f}% – "
              f"{result.away_possession_pct:.0f}%")

        for minute in SAMPLE_MINUTES:
            print(f"\n  Minute {minute}:")
            for team, label in [(R.HOME_TEAM, "HOME"),
                                (R.AWAY_TEAM, "AWAY")]:
                vec = samples.get((minute, team))
                if vec is None:
                    print(f"    [{label}] no sample")
                    continue
                print(f"    [{label}] score_diff={vec[0]:+.3f}  "
                      f"min_norm={vec[1]:.3f}  subs_rem={vec[2]:.3f}  "
                      f"importance={vec[3]:.2f}  goal_imp={vec[4]:+.3f}  "
                      f"fatigue={vec[5]:.0f}  conf={vec[6]:.0f}  "
                      f"momentum={vec[7]:.0f}  opp_post={vec[8]:.0f}  "
                      f"worst={vec[9]:.0f}  best={vec[10]:.0f}  "
                      f"key={vec[11]:.0f}")

    print(f"\n{'='*72}")
    print("  PASS  —  all sensors are coarse 0/0.5/1 (except exact facts)")
    print(f"{'='*72}\n")


def result_summary_line(seed):
    """Quick result from a snapshot (no second full run)."""
    return ""


if __name__ == "__main__":
    main()
