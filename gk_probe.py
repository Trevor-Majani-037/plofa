import sys
sys.path.insert(0, '.')
from datetime import date
from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from roster_loader import get_loader
from event_chain import EventType
from collections import Counter

HOME_TEAM = 'Triumpher'
AWAY_TEAM = 'Seafcea'
loader = get_loader()
home_raw = loader.build_matchday_squad(HOME_TEAM, availability={})
away_raw = loader.build_matchday_squad(AWAY_TEAM, availability={})
home_squad = SquadBuilder.build(team_name=HOME_TEAM, starters=home_raw['starters'], substitutes=home_raw['substitutes'][:2], team_superstars=home_raw['superstars'], set_piece_takers=home_raw['sp_takers'])
away_squad = SquadBuilder.build(team_name=AWAY_TEAM, starters=away_raw['starters'], substitutes=away_raw['substitutes'][:2], team_superstars=away_raw['superstars'], set_piece_takers=away_raw['sp_takers'])
HOME_STYLE = TeamProfile(name=HOME_TEAM, style=TeamStyle.ULTRA_ATTACKING, playing_style=PlayingStyle.POSSESSION, intensity=Intensity.VERY_HIGH)
AWAY_STYLE = TeamProfile(name=AWAY_TEAM, style=TeamStyle.WING_PLAY, playing_style=PlayingStyle.COUNTER, intensity=Intensity.LOW)
config = MatchConfig(home_team=HOME_TEAM, away_team=AWAY_TEAM, match_date=date(2027, 3, 28), matchday=24, season='26/27', competition='PLOFA', venue=HOME_TEAM + ' Stadium', stadium_capacity=100000, referee='Test Ref', referee_strictness=0.5, is_derby=False, weather='clear')
engine = MatchEngine(config, HOME_STYLE, AWAY_STYLE)
engine.set_squad(HOME_TEAM, home_squad['starters'], home_squad['substitutes'])
engine.set_squad(AWAY_TEAM, away_squad['starters'], away_squad['substitutes'])
print('simulating...', flush=True)
result = engine.simulate()
long_counts = Counter()
for e in result.timeline:
    if e.event_type == EventType.LONG_PASS:
        long_counts[e.outcome] += 1
sweeps = [e for e in result.timeline if e.event_type == EventType.SAVE and (e.metadata or {}).get('type') == 'gk_sweep']
print('long pass outcomes:', dict(long_counts), flush=True)
print('gk_sweeps:', len(sweeps), flush=True)
print('total saves:', sum(1 for e in result.timeline if e.event_type == EventType.SAVE), flush=True)
print('done', flush=True)