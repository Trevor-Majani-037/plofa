#!/usr/bin/env python3
"""
Test to measure computational cost of simulating with "all brains active at 10Hz + physics resolution + action logic"
vs current hybrid approach. Adjust for more realistic footballer thinking times.
This is a TEST ONLY - revert after.
Must NOT use auto_run_match (would increment season games).
"""
import time
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from datetime import date
from squad_manager import SubstitutionController
from manager_profile import ManagerProfile

# Try to use the same teams as in run_match - Hartwell City vs Thornfield
HOME_TEAM = "Hartwell City"
AWAY_TEAM = "Thornfield"

HOME_STYLE = TeamProfile(
    name=HOME_TEAM,
    style=TeamStyle.ATTACKING,
    playing_style=PlayingStyle.HIGH_PRESS,
    intensity=Intensity.HIGH,
)

AWAY_STYLE = TeamProfile(
    name=AWAY_TEAM,
    style=TeamStyle.FLUID_COUNTER,
    playing_style=PlayingStyle.COUNTER,
    intensity=Intensity.MEDIUM,
)

config = MatchConfig(
    home_team=HOME_TEAM,
    away_team=AWAY_TEAM,
    match_date=date(2026, 10, 27),
    matchday=9,
    season="26/27",
    competition="PLOFA",
    venue="Test Stadium",
)

# Load from run_match squads
from run_match import (
    HOME_STARTERS, HOME_SUBS, AWAY_STARTERS, AWAY_SUBS,
    HOME_SUPERSTARS, AWAY_SUPERSTARS, HOME_SP_TAKERS, AWAY_SP_TAKERS,
)

home_squad = SquadBuilder.build(
    HOME_TEAM,
    starters=HOME_STARTERS,
    substitutes=HOME_SUBS,
    team_superstars=HOME_SUPERSTARS,
    set_piece_takers=HOME_SP_TAKERS,
)
away_squad = SquadBuilder.build(
    AWAY_TEAM,
    starters=AWAY_STARTERS,
    substitutes=AWAY_SUBS,
    team_superstars=AWAY_SUPERSTARS,
    set_piece_takers=AWAY_SP_TAKERS,
)

home_mgr = ManagerProfile(name="Test Home", club=HOME_TEAM)
away_mgr = ManagerProfile(name="Test Away", club=AWAY_TEAM)

sub_controller = SubstitutionController(
    home_team=HOME_TEAM,
    away_team=AWAY_TEAM,
    home_subs_bench=home_squad["substitutes"],
    away_subs_bench=away_squad["substitutes"],
    home_style=HOME_STYLE.style.value,
    away_style=AWAY_STYLE.style.value,
    manager_stubbornness=0.3,
)

engine = MatchEngine(config, HOME_STYLE, AWAY_STYLE)
engine.set_squad(HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
engine.set_squad(AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
engine.set_stamina_controller(sub_controller)
engine.set_managers(home_manager=home_mgr, away_manager=away_mgr)

print("Starting baseline match...")
start = time.time()
result = engine.simulate()
elapsed = time.time() - start
print(f"Baseline completed in {elapsed:.2f}s")
print(result.summary())
