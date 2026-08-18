"""Quick diagnostic for modern winger positioning."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from position_engine import PositionEngine
from player_dna import SquadBuilder

home = SquadBuilder.build('Home FC', [
    ('GK1','GK',[]), ('CB1','CB',[]), ('CB2','CB',[]),
    ('LB1','LB',[]), ('RB1','RB',[]), ('CDM1','CDM',[]),
    ('CM1','CM',[]), ('CAM1','CAM',[]),
    ('Home LW','LW',['dribbler','speedster']),
    ('ST1','ST',[]),
    ('Home RW','RW',['grand_dribbler','inverted']),
])

away = SquadBuilder.build('Away FC', [
    ('GK2','GK',[]), ('CB3','CB',[]), ('CB4','CB',[]),
    ('LB2','LB',[]), ('RB2','RB',[]), ('CDM2','CDM',[]),
    ('CM2','CM',[]), ('CAM2','CAM',[]),
    ('Away LW','LW',['dribbler','speedster']),
    ('ST2','ST',[]),
    ('Away RW','RW',['grand_dribbler','inverted']),
])

class FakeProfile:
    defensive_line=0.5; width=0.6; tempo=0.5; directness=0.5; press_intensity=0.5

pe = PositionEngine()
pe.initialize_team('Home FC', home['starters'], FakeProfile(), attacks_right=True)
pe.initialize_team('Away FC', away['starters'], FakeProfile(), attacks_right=False)

print('1. HOME WINGER HOMES (attacks right):')
for n in ['Home LW', 'Home RW']:
    s = pe.states[n]
    print(f'   {n}: home=({s.home_x:.0f}, {s.home_y:.0f})')

print('2. AWAY WINGER HOMES (attacks left):')
for n in ['Away LW', 'Away RW']:
    s = pe.states[n]
    print(f'   {n}: home=({s.home_x:.0f}, {s.home_y:.0f})')

# Simulate 10 min in possession for home
print('3. HOME 10min in possession (ball x=85):')
for m in range(1, 11):
    pe.drift_minute('Home FC', FakeProfile(), type('P',(),{'value':'second_open'})(),
                    minute=m, in_possession=True, ball_x=85.0, ball_y=30.0)
for n in ['Home LW', 'Home RW']:
    s = pe.states[n]
    z = s.zone
    print(f'   {n}: ({s.current_x:.1f}, {s.current_y:.1f}) zone={z} '
          f'in_att={s.current_x>70} wide={s.current_y<22 or s.current_y>46}')

# Away team attacking
print('4. AWAY 10min in possession (ball x=20):')
pe2 = PositionEngine()
pe2.initialize_team('Away FC', away['starters'], FakeProfile(), attacks_right=False)
for m in range(1, 11):
    pe2.drift_minute('Away FC', FakeProfile(), type('P',(),{'value':'second_open'})(),
                     minute=m, in_possession=True, ball_x=20.0, ball_y=40.0)
for n in ['Away LW', 'Away RW']:
    s = pe2.states[n]
    print(f'   {n}: ({s.current_x:.1f}, {s.current_y:.1f}) zone={s.zone} '
          f'in_att={s.current_x<35} wide={s.current_y<22 or s.current_y>46}')

print('DONE')