"""Guard tests for the checkpoint-23 wide-touchline behaviour:
1. defensive_block keeps full-backs/wingers on their own touchline (no collapse)
   while CBs still tuck toward the ball.
2. Out-of-possession touchline press pumps a pressing team's full-back forward
   along the flank, and is gated OFF below the press_intensity floor.
3. The (previously swapped) touchline anchor constants are now frame-correct.
"""
import random
from position_engine import PositionEngine
from player_dna import SquadBuilder
from winger_behavior import (
    WingerBehaviorEngine, LEFT_TOUCHLINE_ANCHOR_Y, RIGHT_TOUCHLINE_ANCHOR_Y,
)


def _profile(press_intensity):
    return type('Prof', (), {
        'defensive_line': 0.5,
        'width': 0.9,
        'tempo': 0.5,
        'directness': 0.5,
        'press_intensity': press_intensity,
    })


def _build(attacks_right=True, press_intensity=0.9):
    s = SquadBuilder.build('Test FC', [
        ('GK', 'GK', []), ('CB1', 'CB', []), ('CB2', 'CB', []),
        ('LB', 'LB', ['aggressive_fullback']), ('RB', 'RB', ['overlapping_fullback']),
        ('CDM', 'CDM', []), ('CM1', 'CM', []), ('CM2', 'CM', []),
        ('LW', 'LW', ['dribbler', 'speedster']), ('ST', 'ST', []),
        ('RW', 'RW', ['inverted']),
    ])
    pe = PositionEngine()
    pe.initialize_team('Test FC', s['starters'], _profile(press_intensity),
                       attacks_right=attacks_right)
    return pe, s


def test_defensive_block_preserves_fullback_touchline_but_cb_tucks():
    random.seed(3)
    pe, squad = _build(attacks_right=True)          # home defends own_goal_x=0
    lb = pe.states['LB']
    cb = pe.states['CB1']
    # Full-back on the left touchline; ball deep in own half on the RIGHT side.
    lb.current_x, lb.current_y = 40.0, 10.0
    cb.current_x, cb.current_y = 40.0, 34.0
    pe.defensive_block('Test FC', 18.0, 48.0, own_goal_x=0.0,
                       danger_level=80.0, pull_strength=1.0)
    # CB is the compact core: tucks hard toward the ball's side.
    assert cb.current_y > 40.0, f"CB should tuck toward ball, got y={cb.current_y:.1f}"
    # Full-back keeps the touchline: never dragged onto the ball's opposite flank.
    assert lb.current_y < 22.0, f"LB must hold its own touchline, got y={lb.current_y:.1f}"


def test_out_of_possession_touchline_press_gated_by_press_intensity():
    random.seed(5)
    pe_hi, squad = _build(attacks_right=True, press_intensity=0.9)
    pe_lo = PositionEngine()
    pe_lo.initialize_team('Test FC', squad['starters'], _profile(0.4),
                          attacks_right=True)
    lb_hi = pe_hi.states['LB']
    lb_lo = pe_lo.states['LB']

    phase = type('P', (), {'value': 'second_open'})()
    # Same live ball: high in the opponent's half, on the LB's flank (y small).
    for m in range(1, 4):
        pe_hi.drift_minute('Test FC', _profile(0.9), phase,
                           minute=m, in_possession=False,
                           ball_x=70.0, ball_y=12.0)
        pe_lo.drift_minute('Test FC', _profile(0.4), phase,
                           minute=m, in_possession=False,
                           ball_x=70.0, ball_y=12.0)
    # High press: full-back surges FORWARD up the flank to press.
    # Low press: stays deeper (no touchline press).
    assert lb_hi.current_x > lb_lo.current_x + 8.0, (
        f"High-press LB should push forward past low-press LB: "
        f"hi={lb_hi.current_x:.1f} lo={lb_lo.current_x:.1f}")
    assert lb_hi.current_y < 22.0, f"LB press must hold the touchline, got y={lb_hi.current_y:.1f}"


def test_touchline_anchor_constants_are_frame_correct():
    squad = SquadBuilder.build('T', [
        ('G2','GK',[]),('C1','CB',[]),('C2','CB',[]),
        ('L2','LB',[]),('R2','RB',[]),('M1','CDM',[]),
        ('M2','CM',[]),('M3','CAM',[]),
        ('W1','LW',[]),('S','ST',[]),('W2','RW',[]),
    ])
    lw = next(p for p in squad['starters'] if p.position == 'LW')
    rw = next(p for p in squad['starters'] if p.position == 'RW')
    plw = WingerBehaviorEngine.build_profile_from_dna(lw)
    prw = WingerBehaviorEngine.build_profile_from_dna(rw)
    # LW anchors the LEFT touchline (y=10), RW the RIGHT (y=58).
    assert LEFT_TOUCHLINE_ANCHOR_Y == 10.0 and RIGHT_TOUCHLINE_ANCHOR_Y == 58.0
    assert plw.touchline_anchor_y == 10.0, f"LW anchor wrong: {plw.touchline_anchor_y}"
    assert prw.touchline_anchor_y == 58.0, f"RW anchor wrong: {prw.touchline_anchor_y}"
    # A drifted LW carry should be pushed back LEFT (negative y bias).
    bias = WingerBehaviorEngine.carry_direction_bias(plw, 80.0, 50.0, True)
    assert bias < 0.0, f"LW carry bias should point to left touchline, got {bias:.2f}"