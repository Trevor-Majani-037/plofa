"""
Diagnostic: dump average player positions per half from a real match run
to see the clustering pattern.
"""
import random
from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder
from run_match import (
    HOME_TEAM, AWAY_TEAM, HOME_STARTERS, AWAY_STARTERS,
    HOME_SUBS, AWAY_SUBS, HOME_SUPERSTARS, AWAY_SUPERSTARS,
    HOME_SP_TAKERS, AWAY_SP_TAKERS,
    HOME_STYLE, AWAY_STYLE, MATCH_DATE, MATCHDAY, SEASON, COMPETITION,
    VENUE, CAPACITY, REFEREE, STRICTNESS, IS_DERBY,
)

random.seed(42)

home_squad = SquadBuilder.build(
    team_name=HOME_TEAM, starters=HOME_STARTERS, substitutes=HOME_SUBS,
    team_superstars=HOME_SUPERSTARS, set_piece_takers=HOME_SP_TAKERS,
)
away_squad = SquadBuilder.build(
    team_name=AWAY_TEAM, starters=AWAY_STARTERS, substitutes=AWAY_SUBS,
    team_superstars=AWAY_SUPERSTARS, set_piece_takers=AWAY_SP_TAKERS,
)

config = MatchConfig(
    home_team=HOME_TEAM, away_team=AWAY_TEAM, match_date=MATCH_DATE,
    matchday=MATCHDAY, season=SEASON, competition=COMPETITION,
    venue=VENUE, stadium_capacity=CAPACITY, referee=REFEREE,
    referee_strictness=STRICTNESS, is_derby=IS_DERBY,
)

engine = MatchEngine(config, HOME_STYLE, AWAY_STYLE)
engine.set_squad(HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
engine.set_squad(AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
engine.quiet = True
result = engine.simulate()

print(f"\nScore: {result.state.home_goals} - {result.state.away_goals}")
print(f"Total events: {len(result.timeline)}")
print(f"\n{'='*70}")
print("AVERAGE POSITIONS (per team, minutes 1-90) — from position_log")
print(f"{'='*70}")

for side in ("home", "away"):
    team_name = HOME_TEAM if side == "home" else AWAY_TEAM
    # Collect all snapshots for this team
    pos_sums = {}
    pos_counts = {}
    for frame in engine.position_log:
        for p in frame[side]:
            name = p["player"]
            pos_sums.setdefault(name, [0.0, 0.0])
            pos_counts.setdefault(name, 0)
            pos_sums[name][0] += p["x"]
            pos_sums[name][1] += p["y"]
            pos_counts[name] += 1

    print(f"\n--- {team_name} ---")
    print(f"  {'Player':<18} {'Pos':<5} {'Avg X':>7} {'Avg Y':>7}  {'Zone':<14}")
    print(f"  {'-'*62}")
    # Need positions; get from engine.position_engine snapshot
    snap_rows = {r["player"]: r for r in engine.position_engine.snapshot(team_name)}
    for name, counts in sorted(pos_counts.items(), key=lambda kv: -kv[1]):
        sx, sy = pos_sums[name]
        avg_x = sx / counts
        avg_y = sy / counts
        row = snap_rows.get(name, {})
        pos = row.get("position", "?")
        zone = engine.position_engine.zone_name(name)
        print(f"  {name:<18} {pos:<5} {avg_x:>7.1f} {avg_y:>7.1f}  {zone:<14}")

    # Also show home positions
    print(f"  {'-- home positions --':<40}")
    for r in engine.position_engine.snapshot(team_name):
        print(f"  {r['player']:<18} {r['position']:<5} home=({r['home_x']:.0f},{r['home_y']:.0f})  "
              f"last=({r['current_x']:.0f},{r['current_y']:.0f})  drift={r['drift_from_home']:.1f}m")

# Analyze clustering: at each minute, how spread out is each team?
print(f"\n{'='*70}")
print("SPREAD ANALYSIS (avg distance from team centroid, per 15-min block)")
print(f"{'='*70}")
for side in ("home", "away"):
    team_name = HOME_TEAM if side == "home" else AWAY_TEAM
    print(f"\n--- {team_name} ---")
    for block_start in range(1, 91, 15):
        block_end = block_start + 14
        block_frames = [f for f in engine.position_log if block_start <= f["minute"] <= block_end]
        if not block_frames:
            continue
        spreads = []
        for frame in block_frames:
            players = frame[side]
            if not players:
                continue
            cx = sum(p["x"] for p in players) / len(players)
            cy = sum(p["y"] for p in players) / len(players)
            spread = sum(
                ((p["x"] - cx) ** 2 + (p["y"] - cy) ** 2) ** 0.5
                for p in players
            ) / len(players)
            spreads.append(spread)
        avg_spread = sum(spreads) / len(spreads)
        avg_x = sum(
            sum(p["x"] for p in f[side]) / len(f[side]) if f[side] else 0
            for f in block_frames
        ) / len(block_frames)
        avg_y = sum(
            sum(p["y"] for p in f[side]) / len(f[side]) if f[side] else 0
            for f in block_frames
        ) / len(block_frames)
        print(f"  min {block_start:2d}-{block_end:2d}: avg spread={avg_spread:5.1f}m  "
              f"team centroid=({avg_x:5.1f},{avg_y:5.1f})")

# All 10 outfield players centroid per block
print(f"\n{'='*70}")
print("BALL POSITION TRACKING (per 15-min block)")
print(f"{'='*70}")
for block_start in range(1, 91, 15):
    block_end = block_start + 14
    block_events = [e for e in result.timeline if block_start <= e.minute <= block_end]
    if not block_events:
        continue
    bx = sum(e.location_x for e in block_events) / len(block_events)
    by = sum(e.location_y for e in block_events) / len(block_events)
    # Events in attacking third
    att_right = sum(1 for e in block_events if e.location_x > 70)
    att_left = sum(1 for e in block_events if e.location_x < 35)
    mid = sum(1 for e in block_events if 35 <= e.location_x <= 70)
    total = len(block_events)
    print(f"  min {block_start:2d}-{block_end:2d}: avg ball=({bx:5.1f},{by:5.1f})  "
          f"def_third={att_left/total*100:4.1f}%  mid={mid/total*100:4.1f}%  att_third={att_right/total*100:4.1f}%")