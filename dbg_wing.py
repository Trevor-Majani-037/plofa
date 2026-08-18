"""Debug home LW drift."""
from position_engine import PositionEngine, PlayerSpatialState
from player_dna import SquadBuilder

s = SquadBuilder.build('T', [
    ('G','GK',[]), ('C1','CB',[]), ('C2','CB',[]),
    ('L','LB',[]), ('R','RB',[]), ('M1','CDM',[]),
    ('M2','CM',[]), ('M3','CAM',[]),
    ('W1','LW',['dribbler']), ('S','ST',[]), ('W2','RW',['inverted']),
])

class P:
    defensive_line=0.5; width=0.6; tempo=0.5; directness=0.5; press_intensity=0.5

pe = PositionEngine()
pe.initialize_team('T', s['starters'], P())
w = pe.states['W1']
print('start home:', w.home_x, w.home_y, 'cur:', w.current_x, w.current_y)
print('awareness:', w.geometric_awareness)

# Simulate 5 min, print after each
for m in range(1, 6):
    # Inspect _attacker_space_run candidate selection manually
    from winger_behavior import WingerBehaviorEngine
    wp = pe.winger_registry.get('W1')
    print(f'  profile: flank_commitment={wp.flank_commitment:.2f} anchor_y={wp.touchline_anchor_y}')

    pe.drift_minute('T', P(), type('P',(),{'value':'second_open'})(),
                    minute=m, in_possession=True, ball_x=85, ball_y=30)
    print(f'm{m}: cur=({w.current_x:.1f}, {w.current_y:.1f}) '
          f'flank_pull_target={wp.touchline_anchor_y}')

print('final zone:', w.zone)