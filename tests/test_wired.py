#!/usr/bin/env python
"""Test match with wired FB brain."""
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Wire the brain first
import wire_brain

from match_engine import MatchEngine, MatchState, MatchPhase, MatchConfig
from player_dna import SquadBuilder
import auto_run_match
from datetime import date

random.seed(123)
np.random.seed(123)

loader = auto_run_match.get_loader()
home_raw = loader.build_matchday_squad("Oxton", availability=None)
away_raw = loader.build_matchday_squad("Natrican", availability=None)

from player_dna import SquadBuilder
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
print(f"Home: {result.home_goals}, Away: {result.away_goals}")
print(f"Home poss: {result.home_possession_pct:.1f}%")