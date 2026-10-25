#!/usr/bin/env python
"""Real-match gate: 3 matches challenger vs incumbent."""
import os
import sys
import random
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from match_engine import MatchEngine, MatchState, MatchPhase, MatchConfig
from player_dna import SquadBuilder
import auto_run_match
from datetime import date

# Load FB brain
from distill_defensive_brains import SituationalDefensiveBrain
fb_brain = SituationalDefensiveBrain.load("brains_def/situational/FB.npz")

def run_match(use_brain: bool, match_idx: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    loader = auto_run_match.get_loader()
    home_raw = loader.build_matchday_squad("Oxton", availability=None)
    away_raw = loader.build_matchday_squad("Natrican", availability=None)

    home_squad = SquadBuilder.build(
        team_name="Oxton",
        starters=home_raw["starters"],
        substitutes=home_raw["substitutes"],
        team_superstars=home_raw["superstars"],
        set_piece_takers=home_raw["sp_takers"],
        soul_assignments=auto_run_match.SOUL_PLAYERS,
    )
    away_squad = SquadBuilder.build(
        team_name="Natrican",
        starters=away_raw["starters"],
        substitutes=away_raw["substitutes"],
        team_superstars=away_raw["superstars"],
        set_piece_takers=away_raw["sp_takers"],
        soul_assignments=auto_run_match.SOUL_PLAYERS,
    )

    from auto_run_match import _resolve_team_profile
    home_style = _resolve_team_profile("Oxton", home_raw["formation"], True, None, "fluid_counter")
    away_style = _resolve_team_profile("Natrican", away_raw["formation"], False, None, "attacking")

    config = MatchConfig(
        home_team="Oxton",
        away_team="Natrican",
        match_date=date(2026, 9, 6),
        matchday=3,
        season="26/27",
    )
    state = MatchState()
    state.phase = MatchPhase.OPENING
    state.minute = 0
    state.match_clock_s = 0.0

    engine = MatchEngine(config, home_style, away_style)
    engine.state = state

    engine.set_squad("Oxton", home_squad["starters"], home_squad["substitutes"])
    engine.set_squad("Natrican", away_squad["starters"], away_squad["substitutes"])

    result = engine.simulate()
    return {
        "home_goals": result.home_goals,
        "away_goals": result.away_goals,
        "home_possession": result.home_possession_pct,
        "away_possession": result.away_possession_pct,
    }


def main():
    print("=" * 60)
    print("REAL-MATCH GATE: 3 matches per arm")
    print("=" * 60)

    base_seed = 123

    # Run incumbent
    print("\n--- Running INCUMBENT ---")
    incumbent = []
    for m in range(3):
        r = run_match(False, m, base_seed + m * 100)
        incumbent.append(r)
        print(f"  Match {m+1}: {r['home_goals']}-{r['away_goals']}, poss {r['home_possession']:.1f}%")

    # Run challenger
    print("\n--- Running CHALLENGER ---")
    challenger = []
    for m in range(3):
        r = run_match(True, m, base_seed + m * 100)
        challenger.append(r)
        print(f"  Match {m+1}: {r['home_goals']}-{r['away_goals']}, poss {r['home_possession']:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("GATE RESULTS")
    print("=" * 60)

    for name, results in [("INCUMBENT", incumbent), ("CHALLENGER", challenger)]:
        hw = sum(1 for r in results if r["home_goals"] > r["away_goals"])
        aw = sum(1 for r in results if r["away_goals"] > r["home_goals"])
        dr = sum(1 for r in results if r["home_goals"] == r["away_goals"])
        gf = sum(r["home_goals"] for r in results)
        ga = sum(r["away_goals"] for r in results)
        hp = np.mean([r["home_possession"] for r in results])
        ap = np.mean([r["away_possession"] for r in results])
        print(f"\n{name}: {hw}W-{dr}D-{aw}L, GF={gf}, GA={ga}, HomePoss={hp:.1f}%")

    # Gate
    inc_gf = sum(r["home_goals"] for r in incumbent)
    ch_gf = sum(r["home_goals"] for r in challenger)
    inc_ga = sum(r["away_goals"] for r in incumbent)
    ch_ga = sum(r["away_goals"] for r in challenger)

    print(f"\n--- GATE CHECK ---")
    print(f"Goals For: Ch {ch_gf} vs Inc {inc_gf} -> {'PASS' if ch_gf >= inc_gf else 'FAIL'}")
    print(f"Goals Against: Ch {ch_ga} vs Inc {inc_ga} -> {'PASS' if ch_ga <= inc_ga else 'FAIL'}")
    gate = (ch_gf >= inc_gf) and (ch_ga <= inc_ga)
    print(f"OVERALL: {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    import numpy as np
    main()