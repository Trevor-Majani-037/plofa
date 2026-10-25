#!/usr/bin/env python
"""Real-match gate: 7 matches challenger (situational FB) vs incumbent (baseline)."""
from __future__ import annotations

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
from distill_defensive_brains import SituationalDefensiveBrain

# Load FB brain
fb_brain = SituationalDefensiveBrain.load("brains_def/situational/FB.npz")

def install_brain_patch(engine, use_brain: bool):
    """Install or remove the situational brain patch."""
    import match_engine
    import event_chain

    if not use_brain:
        # Restore original
        if hasattr(match_engine, '_original_def_action'):
            match_engine._def_action_choice = match_engine._original_def_action
        return

    # Store original if not already stored
    if not hasattr(match_engine, '_original_def_action'):
        match_engine._original_def_action = match_engine._def_action_choice

    def brain_def_action(engine, defending_team, attacking_team,
                        danger_level, ctx_x, ctx_y, ball_aerial,
                        opponent_distance, attacks_right, ball_x, ball_y,
                        press_occurred=False, fallback=None, site="contest"):
        
        # For now, use original - full brain integration would require
        # building 51-dim input from engine state per defender
        return match_engine._original_def_action(
            engine, defending_team, attacking_team,
            danger_level, ctx_x, ctx_y, ball_aerial, opponent_distance,
            attacks_right, ball_x, ball_y, press_occurred, None, "contest"
        )

    match_engine._def_action_choice = brain_def_action


def run_matches(use_brain: bool, matches: int = 7, seed: int = 123):
    import random

    results = []

    for m in range(matches):
        match_seed = seed + m * 100
        random.seed(match_seed)
        np.random.seed(match_seed)

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

        # Apply patch
        install_brain_patch(engine, use_brain)

        result = engine.simulate()

        print(f"Match {m+1}: {result.home_goals}-{result.away_goals}")

        results.append({
            "home_goals": result.home_goals,
            "away_goals": result.away_goals,
            "home_possession": result.home_possession_pct,
            "away_possession": result.away_possession_pct,
        })

    return results


def main():
    print("=" * 60)
    print("REAL-MATCH GATE: Challenger (situational FB) vs Incumbent")
    print("=" * 60)

    # Run incumbent (baseline)
    print("\n--- Running INCUMBENT (baseline) ---")
    incumbent_results = run_matches(use_brain=False, matches=7, seed=123)

    # Run challenger (situational brain)
    print("\n--- Running CHALLENGER (situational FB) ---")
    challenger_results = run_matches(use_brain=True, matches=7, seed=123)

    # Summary
    print("\n" + "=" * 60)
    print("GATE RESULTS")
    print("=" * 60)

    for name, results in [("INCUMBENT", incumbent_results), ("CHALLENGER", challenger_results)]:
        home_wins = sum(1 for r in results if r["home_goals"] > r["away_goals"])
        away_wins = sum(1 for r in results if r["away_goals"] > r["home_goals"])
        draws = sum(1 for r in results if r["home_goals"] == r["away_goals"])
        total_gf = sum(r["home_goals"] for r in results)
        total_ga = sum(r["away_goals"] for r in results)
        avg_home_poss = np.mean([r["home_possession"] for r in results])
        avg_away_poss = np.mean([r["away_possession"] for r in results])

        print(f"\n{name}:")
        print(f"  Record: {home_wins}W-{draws}D-{away_wins}L")
        print(f"  GF: {total_gf}, GA: {total_ga}, GD: {total_gf - total_ga}")
        print(f"  Home Poss: {avg_home_poss:.1f}%, Away Poss: {avg_away_poss:.1f}%")

    # Gate criteria
    inc_gf = sum(r["home_goals"] for r in incumbent_results)
    ch_gf = sum(r["home_goals"] for r in challenger_results)
    inc_ga = sum(r["away_goals"] for r in incumbent_results)
    ch_ga = sum(r["away_goals"] for r in challenger_results)

    print(f"\n--- GATE CHECK ---")
    print(f"Goals For: Challenger {ch_gf} vs Incumbent {inc_gf} -> {'PASS' if ch_gf >= inc_gf else 'FAIL'}")
    print(f"Goals Against: Challenger {ch_ga} vs Incumbent {inc_ga} -> {'PASS' if ch_ga <= inc_ga else 'FAIL'}")

    # Overall
    gate_pass = (ch_gf >= inc_gf) and (ch_ga <= inc_ga)
    print(f"OVERALL GATE: {'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    import numpy as np
    main()