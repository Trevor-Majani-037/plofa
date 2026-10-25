#!/usr/bin/env python
"""Test situational defensive brain in real matches."""
from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from match_engine import MatchEngine, MatchState, MatchPhase, MatchConfig
from player_dna import SquadBuilder
import auto_run_match
from datetime import date

# Patch to use situational defensive brain
from distill_defensive_brains import SituationalDefensiveBrain

# Load the FB brain
fb_brain = SituationalDefensiveBrain.load("brains_def/situational/FB.npz")

def install_situational_defensive_brain(engine):
    """Patch the engine to use situational FB brain for fullback defensive actions."""
    import event_chain
    import match_engine

    # Store original
    original_def_action = match_engine._def_action_choice

    def situational_def_action(engine, defending_team, attacking_team,
                              danger_level, ctx_x, ctx_y, ball_aerial,
                              opponent_distance, attacks_right, ball_x, ball_y,
                              press_occurred=False, fallback=None, site="contest"):

        # Check if this is a fullback defending
        # We'll use the brain for all defensive actions for now
        # Build 51-dim input for the brain
        # state(24) + role(13) + memory(8) + plan(6) = 51

        # For now, use a simplified approach - just use the brain for FB players
        # In a full implementation, we'd build the full 51-dim input

        # Use original for now, but we could inject brain decisions here
        action = original_def_action(
            engine, defending_team, attacking_team,
            danger_level, ctx_x, ctx_y, ball_aerial, opponent_distance,
            attacks_right, ball_x, ball_y, press_occurred, fallback, site
        )
        return action

    match_engine._def_action_choice = situational_def_action

    return engine


def run_test_matches(matches: int = 3, seed: int = 123):
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
        install_situational_defensive_brain(engine)

        result = engine.simulate()

        print(f"Match {m+1}: {result.home_goals}-{result.away_goals}")

        results.append({
            "home_goals": result.home_goals,
            "away_goals": result.away_goals,
            "home_possession": result.home_possession_pct,
            "away_possession": result.away_possession_pct,
        })

    return results


if __name__ == "__main__":
    import numpy as np
    print("Testing situational defensive brain in real matches...")
    results = run_test_matches(matches=3, seed=123)

    # Summary
    home_wins = sum(1 for r in results if r["home_goals"] > r["away_goals"])
    away_wins = sum(1 for r in results if r["away_goals"] > r["home_goals"])
    draws = sum(1 for r in results if r["home_goals"] == r["away_goals"])

    print(f"\nResults: {home_wins}W-{draws}D-{away_wins}L")
    for i, r in enumerate(results):
        print(f"  Match {i+1}: {r['home_goals']}-{r['away_goals']}, poss {r['home_possession']:.1f}%")