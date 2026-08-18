"""Guard tests for the Checkpoint-18 winger on-the-ball steering wiring:

1. carry_direction_bias pulls a drifted winger's carry back onto his flank
   (formation-corrected anchor — including mirrored attacking-left teams).
2. _winger_carry_steering resolves the FORMATION anchor (home_y), never the
   position-name anchor, and returns only valid drive modes.
3. Drive/cut instincts differentiate traditional vs inverted wingers, and the
   cut is gated on the flank channel / half-space openness.
"""
import random

from winger_behavior import (
    WingerBehaviorEngine, WingerSpatialProfile,
    LEFT_TOUCHLINE_ANCHOR_Y, RIGHT_TOUCHLINE_ANCHOR_Y,
)
from position_engine import PositionEngine
from player_dna import SquadBuilder
from event_chain import PossessionChain


def _profile(byline_instinct=0.55):
    return WingerSpatialProfile(
        flank="left",
        touchline_anchor_y=LEFT_TOUCHLINE_ANCHOR_Y,
        flank_commitment=0.85,
        byline_instinct=byline_instinct,
        cross_instinct=0.45,
    )


def _build():
    s = SquadBuilder.build('T', [
        ('G', 'GK', []), ('C1', 'CB', []), ('C2', 'CB', []),
        ('L', 'LB', []), ('R', 'RB', []), ('M1', 'CDM', []),
        ('M2', 'CM', []), ('M3', 'CAM', []),
        ('W1', 'LW', ['dribbler']), ('S', 'ST', []), ('W2', 'RW', ['inverted']),
    ])
    pe = PositionEngine()
    prof = type('P', (), {
        'defensive_line': 0.5, 'width': 0.6, 'tempo': 0.5,
        'directness': 0.5, 'press_intensity': 0.5,
    })()
    pe.initialize_team('T', s['starters'], prof, attacks_right=True)
    return pe, s


def test_carry_bias_pulls_drifted_winger_back_to_own_touchline():
    prof = _profile()
    # Drifted ~15m inside the left touchline (y=25 vs anchor 10).
    bias = WingerBehaviorEngine.carry_direction_bias(
        prof, 80.0, 25.0, True, anchor_y=LEFT_TOUCHLINE_ANCHOR_Y)
    assert bias < 0.0, f"left winger drifted inside must be pulled LEFT, got {bias}"
    # Inside the flank channel the carry is not forced sideways.
    bias_on = WingerBehaviorEngine.carry_direction_bias(
        prof, 80.0, 12.0, True, anchor_y=LEFT_TOUCHLINE_ANCHOR_Y)
    assert bias_on == 0.0


def test_carry_bias_mirrored_formation_uses_home_y_not_name():
    # Name-based profile says 'LW' anchors LEFT, but a mirrored formation puts
    # him on the RIGHT side (home_y ~58).
    prof = _profile()
    bias = WingerBehaviorEngine.carry_direction_bias(
        prof, 80.0, 45.0, True, anchor_y=RIGHT_TOUCHLINE_ANCHOR_Y)
    assert bias > 0.0, f"mirrored LW must be pulled to the RIGHT anchor, got {bias}"


def test_cut_inside_gated_to_own_formation_flank():
    inv = _profile(byline_instinct=0.20)
    # Mirrored LW (anchor 58). Standing centre (y=35) is NOT his flank → no cut.
    assert not WingerBehaviorEngine.should_cut_inside(
        inv, 82.0, 35.0, True, half_space_open=True, anchor_y=58.0)
    # On his own formation flank (y=60) the cut is a live option.
    random.seed(3)
    hits = sum(WingerBehaviorEngine.should_cut_inside(
        inv, 82.0, 60.0, True, half_space_open=True, anchor_y=58.0)
        for _ in range(50))
    assert hits > 10, f"inverted winger should cut from his own flank, got {hits}/50"


def test_drive_byline_mirrored_formation_works():
    trad = _profile(byline_instinct=0.85)
    random.seed(5)
    hits = sum(WingerBehaviorEngine.should_drive_byline(
        trad, 80.0, 60.0, True, anchor_y=58.0) for _ in range(50))
    assert hits > 15, f"traditional winger should drive from his own flank, got {hits}/50"


def test_traditional_drives_more_inverted_cuts_only_when_open():
    random.seed(11)
    trad = _profile(byline_instinct=0.85)
    inv = _profile(byline_instinct=0.20)
    n = 300
    drives_trad = sum(
        WingerBehaviorEngine.should_drive_byline(trad, 82.0, 12.0, True)
        for _ in range(n))
    drives_inv = sum(
        WingerBehaviorEngine.should_drive_byline(inv, 82.0, 12.0, True)
        for _ in range(n))
    assert drives_trad > drives_inv * 1.5, (
        f"traditional winger should drive far more than inverted: "
        f"{drives_trad} vs {drives_inv}")

    cuts_inv_open = sum(
        WingerBehaviorEngine.should_cut_inside(inv, 82.0, 12.0, True,
                                               half_space_open=True)
        for _ in range(n))
    cuts_inv_closed = sum(
        WingerBehaviorEngine.should_cut_inside(inv, 82.0, 12.0, True,
                                               half_space_open=False)
        for _ in range(n))
    cuts_trad_open = sum(
        WingerBehaviorEngine.should_cut_inside(trad, 82.0, 12.0, True,
                                               half_space_open=True)
        for _ in range(n))
    assert cuts_inv_open > cuts_trad_open, (
        f"inverted winger should cut more than traditional: "
        f"{cuts_inv_open} vs {cuts_trad_open}")
    assert cuts_inv_open > cuts_inv_closed, (
        f"open half-space must boost the cut: {cuts_inv_open} vs {cuts_inv_closed}")


def test_winger_carry_steering_resolves_formation_anchor_and_mode():
    pe, s = _build()
    w1 = next(p for p in s['starters'] if p.position == 'LW')
    mode, anchor, bias = PossessionChain._winger_carry_steering(
        w1, 82.0, 12.0, True, False, [], pe, commit_rolls=False)
    assert anchor == pe.states['W1'].home_y, (
        f"anchor must be the formation home_y, got {anchor}")
    assert bias == 0.0  # on his flank → no forced sideways carry

    random.seed(7)
    modes = set()
    for _ in range(30):
        m, a, b = PossessionChain._winger_carry_steering(
            w1, 82.0, 12.0, True, False, [], pe)
        assert m in (None, "byline", "cut_inside")
        assert a == pe.states['W1'].home_y
        modes.add(m)
    # A dribbler/inverted winger in the final third mixes drive, cut and pass.
    assert "cut_inside" in modes, f"inverted winger should sometimes cut: {modes}"


def test_attacking_crash_near_side_winger_stays_on_own_flank():
    pe, s = _build()
    lw = pe.states['W1']   # left winger, home_y ~10.8
    rw = pe.states['W2']   # right winger, home_y ~57.2
    random.seed(4)
    lw.current_x, lw.current_y = 80.0, 12.0
    rw.current_x, rw.current_y = 80.0, 58.0
    # Ball on the LEFT flank in the crossing zone: LW near-side, RW far-side.
    for _ in range(3):
        pe.attacking_crash('T', 82.0, 12.0, True, minute=1, intensity=1.0)
    # Near-side winger holds his own flank — never the penalty spot (y≈34).
    assert lw.current_y < 25.0, (
        f"near-side winger must stay on his flank, got y={lw.current_y:.1f}")
    # Far-side winger sprints to the back post (right side).
    assert rw.current_y > 40.0, (
        f"far-side winger must head to the back post, got y={rw.current_y:.1f}")
    # Both push toward the goal they attack.
    assert lw.current_x > 80.0 and rw.current_x > 80.0
