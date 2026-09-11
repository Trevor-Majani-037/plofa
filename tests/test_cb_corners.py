import sys
sys.stdout.reconfigure(encoding='utf-8')
import random
from datetime import date
from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from match_engine import EventType

HOME_STARTERS = [
    ('Keano Walsh', 'GK', ['sweeper_keeper'], 29),
    ('Darius Frost', 'LB', ['aggressive_fullback'], 24),
    ('Emeka Obi', 'CB', ['ball_playing_cb'], 27),
    ('Tavish Crane', 'CB', ['stopper_defender'], 30),
    ('Rico Alves', 'RB', ['overlapping_fullback'], 25),
    ('Mateo Sanz', 'CDM', ['anchor_man'], 28),
    ('Luca Ferrini', 'CM', ['box_box'], 26),
    ('Kofi Mensah', 'CAM', ['creator'], 24),
    ('Adri Vela', 'LW', ['dribbler'], 22),
    ('Dragan Novak', 'ST', ['clinical_finisher'], 29),
    ('Percy', 'RW', ['grand_dribbler'], 24),
]

random.seed(42)
home = SquadBuilder.build('Hartwell City', HOME_STARTERS)
away = SquadBuilder.build('Away', [('P0', 'GK', [], 25)] + [(f'P{i}', 'CB', [], 25) for i in range(1, 11)])
config = MatchConfig(home_team='Hartwell City', away_team='Away', match_date=date(2026, 8, 16), matchday=1)
hs = TeamProfile('Hartwell City', TeamStyle.ATTACKING, PlayingStyle.HIGH_PRESS, Intensity.HIGH)
as_ = TeamProfile('Away', TeamStyle.FLUID_COUNTER, PlayingStyle.COUNTER, Intensity.MEDIUM)
engine = MatchEngine(config, hs, as_)
engine.set_squad('Hartwell City', home['starters'], home['substitutes'])
engine.set_squad('Away', away['starters'], away['substitutes'])
result = engine.simulate()

corner_events = [e for e in result.timeline if e.event_type == EventType.CORNER_TAKEN]

print('Corners taken by:')
for e in corner_events[:15]:
    player = [p for p in home['starters'] + away['starters'] if p.name == e.player][0]
    pos = player.position
    print(f"  {e.minute}\" {e.player} ({pos})")

cb_corners = [e for e in corner_events if any(p.position == 'CB' and p.name == e.player for p in home['starters'] + away['starters'])]
print(f"\nCB corners: {len(cb_corners)}")
print(f"Total corners: {len(corner_events)}")

if cb_corners:
    print('FAIL: CBs are taking corners!')
else:
    print('PASS: No CBs taking corners')
