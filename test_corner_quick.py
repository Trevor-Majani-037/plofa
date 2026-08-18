"""Quick corner count test - 3 matches average"""
import sys
from datetime import date
from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from roster_loader import get_loader
from event_chain import EventType

def run_test_match():
    """Run one quick test match and return corner count"""
    HOME_TEAM = "Triumpher"
    AWAY_TEAM = "Seafcea"
    
    loader = get_loader()
    
    home_raw = loader.build_matchday_squad(HOME_TEAM, availability={})
    away_raw = loader.build_matchday_squad(AWAY_TEAM, availability={})
    
    home_squad = SquadBuilder.build(
        team_name=HOME_TEAM,
        starters=home_raw["starters"],
        substitutes=home_raw["substitutes"][:2],
        team_superstars=home_raw["superstars"],
        set_piece_takers=home_raw["sp_takers"],
    )
    away_squad = SquadBuilder.build(
        team_name=AWAY_TEAM,
        starters=away_raw["starters"],
        substitutes=away_raw["substitutes"][:2],
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
        match_date=date(2027, 3, 28),
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
    
    engine = MatchEngine(config, HOME_STYLE, AWAY_STYLE)
    engine.set_squad(HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    
    result = engine.simulate()
    
    # Count key events
    saves = [e for e in result.timeline if e.event_type == EventType.SAVE]
    blocks = [e for e in result.timeline if e.event_type == EventType.BLOCK]
    clearances = [e for e in result.timeline if e.event_type == EventType.CLEARANCE]
    woodwork = [e for e in result.timeline if e.event_type == EventType.HIT_WOODWORK]
    shots_on_target = [e for e in result.timeline if e.event_type == EventType.SHOT_ON_TARGET]
    corner_events = [e for e in result.timeline if e.event_type == EventType.CORNER_TAKEN]
    corner_won_events = [e for e in result.timeline if e.event_type == EventType.CORNER_WON]
    
    # Analyze sources
    sources = {"saves": 0, "blocks": 0, "clearances": 0, "woodwork": 0, "other": 0}
    for e in corner_won_events:
        if hasattr(e, 'metadata') and e.metadata:
            if e.metadata.get("from_save_deflection"):
                sources["saves"] += 1
            elif e.metadata.get("from_woodwork"):
                sources["woodwork"] += 1
            elif e.metadata.get("from_shot_block") or e.metadata.get("from_defensive_block"):
                sources["blocks"] += 1
            elif e.metadata.get("from_failed_clearance"):
                sources["clearances"] += 1
            else:
                sources["other"] += 1
        else:
            sources["other"] += 1
    
    return {
        "score": f"{result.home_goals}-{result.away_goals}",
        "corners_total": len(corner_events),
        "corners_won": len(corner_won_events),
        "shots_on_target": len(shots_on_target),
        "saves": len(saves),
        "blocks": len(blocks),
        "clearances": len(clearances),
        "woodwork": len(woodwork),
        "sources": sources,
    }

print("=" * 70)
print("CORNER FIX VERIFICATION - Running 3 Test Matches")
print("=" * 70)

results = []
for i in range(3):
    print(f"\nMatch {i+1}/3...", end=" ", flush=True)
    r = run_test_match()
    results.append(r)
    print(f"✓ Score: {r['score']}, Corners: {r['corners_total']}")

print("\n" + "=" * 70)
print("AGGREGATE RESULTS")
print("=" * 70)

avg_corners = sum(r['corners_total'] for r in results) / len(results)
avg_shots = sum(r['shots_on_target'] for r in results) / len(results)
avg_saves = sum(r['saves'] for r in results) / len(results)
avg_blocks = sum(r['blocks'] for r in results) / len(results)

print(f"\nAverage per match:")
print(f"  Total corners: {avg_corners:.1f}")
print(f"  Shots on target: {avg_shots:.1f}")
print(f"  Saves: {avg_saves:.1f}")
print(f"  Defensive blocks: {avg_blocks:.1f}")

# Aggregate sources
total_sources = {"saves": 0, "blocks": 0, "clearances": 0, "woodwork": 0, "other": 0}
for r in results:
    for k, v in r['sources'].items():
        total_sources[k] += v

print(f"\nCorner sources (total across {len(results)} matches):")
for source, count in sorted(total_sources.items(), key=lambda x: -x[1]):
    if count > 0:
        print(f"  {source}: {count}")

print(f"\n{'='*70}")
if avg_corners >= 8:
    print("✅ SUCCESS: Corner counts are now REALISTIC (8-11 per match target)")
elif avg_corners >= 6:
    print("⚠️  IMPROVED: Corner counts better but still slightly low (target: 8-11)")
else:
    print("❌ NEEDS MORE WORK: Corner counts still too low (target: 8-11)")

print(f"{'='*70}\n")
