import random, sys, json, tempfile, os
sys.path.insert(0, '.')
from datetime import date
from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder, DNAFactory
from exporter import PLOFAExporter

# Build a squad with a world-class creator
HOME_STARTERS = [
    ('World Class GK', 'GK', ['sweeper_keeper'], 29),
    ('Defender 1', 'CB', ['ball_playing_cb'], 27),
    ('Defender 2', 'CB', ['stopper_defender'], 30),
    ('Defender 3', 'CB', ['ball_playing_cb'], 28),
    ('Defender 4', 'RB', ['overlapping_fullback'], 25),
    ('CDM', 'CDM', ['anchor_man'], 28),
    ('Box Box', 'CM', ['box_box'], 26),
    ('WORLD CLASS CREATOR', 'CAM', ['creator'], 24),  # This is our Ozil/De Bruyne
    ('LW', 'LW', ['dribbler'], 22),
    ('ST', 'ST', ['clinical_finisher'], 29),
    ('RW', 'RW', ['grand_dribbler'], 24),
]

def run_test(seed, style, intensity):
    random.seed(seed)
    home = SquadBuilder.build('Home', HOME_STARTERS)
    away = SquadBuilder.build('Away', (
        [('Away GK', 'GK', [], 25)] +
        [(f'P{i}', 'CB', [], 25) for i in range(10)]
    ))
    
    config = MatchConfig(home_team='Home', away_team='Away', match_date=date(2026, 8, 16), matchday=1)
    hs = TeamProfile('Home', style, PlayingStyle.HIGH_PRESS, intensity)
    as_ = TeamProfile('Away', TeamStyle.FLUID_COUNTER, PlayingStyle.COUNTER, Intensity.MEDIUM)
    engine = MatchEngine(config, hs, as_)
    engine.set_squad('Home', home['starters'], home['substitutes'])
    engine.set_squad('Away', away['starters'], away['substitutes'])
    result = engine.simulate()
    
    # Export
    tmpdir = tempfile.mkdtemp()
    exporter = PLOFAExporter(result, {'Home': home, 'Away': away})
    exporter.export_json(os.path.join(tmpdir, 'match.json'))
    with open(os.path.join(tmpdir, 'match.json'), encoding='utf-8-sig') as f:
        data = json.load(f)
    
    players = data.get('players', {})
    creator_stats = None
    for p in players.values():
        if 'WORLD CLASS CREATOR' in p.get('player', ''):
            creator_stats = p
            break
    
    if creator_stats:
        cc = creator_stats.get('chances_created', 0)
        bcc = creator_stats.get('big_chances_created', 0)
        total = cc + bcc
        print(f'{style.name} seed={seed}: creator chances_created={cc}, big={bcc}, TOTAL={total}')
    else:
        print(f'{style.name} seed={seed}: creator not found in players')

print('=== Testing world-class creator chance generation ===\n')
for style in [TeamStyle.TIKI_TAKA, TeamStyle.GEGENPRESSING, TeamStyle.ATTACKING]:
    for intensity in [Intensity.HIGH, Intensity.MEDIUM]:
        for seed in [42, 7, 99]:
            run_test(seed, style, intensity)
        print()
