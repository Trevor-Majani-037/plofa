"""Quick corner count verification test"""
import sys
from datetime import date
from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from roster_loader import get_loader

print("=" * 60)
print("CORNER COUNT TEST - Before and After Fix")
print("=" * 60)

# Setup minimal match
HOME_TEAM = "Triumpher"
AWAY_TEAM = "Seafcea"
MATCH_DATE = date(2027, 3, 28)

loader = get_loader()

# Quick squad setup
home_raw = loader.build_matchday_squad(HOME_TEAM, availability={})
away_raw = loader.build_matchday_squad(AWAY_TEAM, availability={})

home_squad = SquadBuilder.build(
    team_name=HOME_TEAM,
    starters=home_raw["starters"],
    substitutes=home_raw["substitutes"][:3],  # Fewer subs for speed
    team_superstars=home_raw["superstars"],
    set_piece_takers=home_raw["sp_takers"],
)
away_squad = SquadBuilder.build(
    team_name=AWAY_TEAM,
    starters=away_raw["starters"],
    substitutes=away_raw["substitutes"][:3],
    team_superstars=away_raw["superstars"],
    set_piece_takers=away_raw["sp_takers"],
)

HOME_STYLE = TeamProfile(
    name=HOME_TEAM,
    style=TeamStyle.ULTRA_ATTACKING,
    playing_style=PlayingStyle.POSSESSION,
    intensity=Intensity.VERY_HIGH,
)

AWAY_STYLE = TeamProfile(
    name=AWAY_TEAM,
    style=TeamStyle.WING_PLAY,
    playing_style=PlayingStyle.COUNTER,
    intensity=Intensity.LOW,
)

config = MatchConfig(
    home_team=HOME_TEAM,
    away_team=AWAY_TEAM,
    match_date=MATCH_DATE,
    matchday=24,
    season="26/27",
    competition="PLOFA",
    venue=f"{HOME_TEAM} Stadium",
    stadium_capacity=100_000,
    referee="Test Ref",
    referee_strictness=0.5,
    is_derby=False,
    weather="clear",
)

print(f"\nRunning: {HOME_TEAM} vs {AWAY_TEAM}")
print("This will take ~30 seconds...\n")

engine = MatchEngine(config, HOME_STYLE, AWAY_STYLE)
engine.set_squad(HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
engine.set_squad(AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])

result = engine.simulate()

# Count corners
from event_chain import EventType
corner_events = [e for e in result.timeline if e.event_type == EventType.CORNER_TAKEN]
home_corners = len([e for e in corner_events if e.team == HOME_TEAM])
away_corners = len([e for e in corner_events if e.team == AWAY_TEAM])
total_corners = home_corners + away_corners

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Score: {HOME_TEAM} {result.home_goals} - {result.away_goals} {AWAY_TEAM}")

# Analyze all relevant events
saves = [e for e in result.timeline if e.event_type == EventType.SAVE]
blocks = [e for e in result.timeline if e.event_type == EventType.BLOCK]
clearances = [e for e in result.timeline if e.event_type == EventType.CLEARANCE]
woodwork = [e for e in result.timeline if e.event_type == EventType.HIT_WOODWORK]
shots_on_target = [e for e in result.timeline if e.event_type == EventType.SHOT_ON_TARGET]

print(f"\nKey Event Counts:")
print(f"  Shots on target: {len(shots_on_target)}")
print(f"  Saves: {len(saves)}")
print(f"  Blocks (in DefensiveChain): {len(blocks)}")
print(f"  Clearances: {len(clearances)}")
print(f"  Woodwork: {len(woodwork)}")

# Analyze save metadata
print(f"\nSave Details:")
parries = 0
catches = 0
parry_corners = 0
for save in saves:
    if hasattr(save, 'metadata') and save.metadata:
        if save.metadata.get('parried'):
            parries += 1
        else:
            catches += 1
print(f"  Parries: {parries}")
print(f"  Catches: {catches}")

# Count corners
from event_chain import EventType
corner_events = [e for e in result.timeline if e.event_type == EventType.CORNER_TAKEN]
home_corners = len([e for e in corner_events if e.team == HOME_TEAM])
away_corners = len([e for e in corner_events if e.team == AWAY_TEAM])
total_corners = home_corners + away_corners

print(f"\nCorner Kicks:")
print(f"  {HOME_TEAM}: {home_corners}")
print(f"  {AWAY_TEAM}: {away_corners}")
print(f"  TOTAL: {total_corners}")
print(f"\nCorners per team: {total_corners/2:.1f}")
print(f"\n{'✅ REALISTIC' if total_corners >= 8 else '❌ TOO LOW'} (Real football: 8-11 per match)")

# Show corner source breakdown
corner_won_events = [e for e in result.timeline if e.event_type == EventType.CORNER_WON]
print(f"\nCorner Sources (n={len(corner_won_events)}):")
sources = {}
for e in corner_won_events:
    src = "unknown"
    if hasattr(e, 'metadata') and e.metadata:
        if e.metadata.get("from_save_deflection"):
            src = "GK save/parry"
        elif e.metadata.get("from_woodwork"):
            src = "Woodwork"
        elif e.metadata.get("from_shot_block"):
            src = "Shot block"
        else:
            src = "Other deflection"
    sources[src] = sources.get(src, 0) + 1

for src, count in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {src}: {count}")

# Look for failed clearances that should have gone for corners
print(f"\nClearing Analysis:")
failed_clears = [c for c in clearances if hasattr(c, 'metadata') and c.metadata and not c.metadata.get('effective', True)]
print(f"  Failed clearances: {len(failed_clears)}")
corner_from_clear = 0
for c in failed_clears:
    if c.metadata.get('failure_cause') == 'slice' or c.metadata.get('dest') == 'corner':
        corner_from_clear += 1
print(f"  Failed clearances marked for corner: {corner_from_clear}")

print("\n" + "=" * 60)
