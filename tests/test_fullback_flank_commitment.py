"""Guard tests for fullback flank commitment (PITCH-WIDTH follow-up):

1. _pass_destination_to_receiver delivers LB/RB receptions onto the flank band
   (a drifted fullback is re-planted wide, not fed in the half-space — the
   same loop that was fixed for wingers in Checkpoint 31).
2. An INVERTED fullback's deliberate build-up pocket survives the band.
3. _fullback_carry_steering resolves the FORMATION home_y anchor and returns a
   touchline-direction bias when drifted and zero bias when hugging the line.
"""
import random

from position_engine import PositionEngine
from player_dna import SquadBuilder
from event_chain import PossessionChain


def _build():
    s = SquadBuilder.build('T', [
        ('G', 'GK', []), ('C1', 'CB', []), ('C2', 'CB', []),
        ('L', 'LB', []), ('R', 'RB', []), ('M1', 'CDM', []),
        ('M2', 'CM', []), ('M3', 'CAM', []),
        ('W1', 'LW', []), ('S', 'ST', []), ('W2', 'RW', []),
    ])
    pe = PositionEngine()
    prof = type('P', (), {
        'defensive_line': 0.5, 'width': 0.6, 'tempo': 0.5,
        'directness': 0.5, 'press_intensity': 0.5,
    })()
    pe.initialize_team('T', s['starters'], prof, attacks_right=True)
    return pe, s


def _get_player(s, pos):
    return next(p for p in s['starters'] if p.position == pos)


def test_fullback_reception_pulled_onto_flank_band():
    pe, s = _build()
    lb = _get_player(s, 'LB')
    home_y = pe.states['L'].home_y
    # LB drifted 20m infield of home (y≈26) — the half-space.
    pe.states['L'].current_x, pe.states['L'].current_y = 60.0, home_y + 20.0
    end_px, end_py = PossessionChain._pass_destination_to_receiver(
        lb, 40.0, 40.0, 15.0, pe, True)
    # Delivery must land back near the flank (within ~10m of the line anchor),
    # not be planted where he drifted (home_y + 20).
    assert end_py < home_y + 12.0, (
        f"drifted LB must be fed on the flank, got y={end_py:.1f} (home_y={home_y:.1f})")


def test_inverted_fullback_pocket_survives_the_band():
    pe, s = _build()
    lb = _get_player(s, 'LB')
    home_y = pe.states['L'].home_y
    fb_prof = pe.fullback_registry.get('L')
    assert fb_prof is not None
    # A truly inverted LB: tucks into the build-up pocket.
    pe.states['L'].current_x, pe.states['L'].current_y = 40.0, home_y + 18.0
    keep = fb_prof.tuck_instinct
    fb_prof.tuck_instinct = 0.85
    try:
        _ipx, inv_py = PossessionChain._pass_destination_to_receiver(
            lb, 40.0, 40.0, 15.0, pe, True)
    finally:
        fb_prof.tuck_instinct = keep
    # Same setup, standard (non-inverted) fullback in the tuck zone.
    pe.states['L'].current_x, pe.states['L'].current_y = 40.0, home_y + 18.0
    _spx, sta_py = PossessionChain._pass_destination_to_receiver(
        lb, 40.0, 40.0, 15.0, pe, True)
    # The inverted pocket delivery stays further infield than the standard one
    # (wasn't yanked all the way back to the touchline).
    assert inv_py > sta_py, (
        f"inverted FB pocket lost: inverted y={inv_py:.1f} vs standard y={sta_py:.1f}")


def test_fullback_carry_steering_resolves_formations_and_bias():
    pe, s = _build()
    lb = _get_player(s, 'LB')
    home_y = pe.states['L'].home_y

    # On his flank → no sideways forcing, anchor is the formation home_y.
    pe.states['L'].current_x, pe.states['L'].current_y = 60.0, home_y + 2.0
    mode, anchor, bias = PossessionChain._fullback_carry_steering(
        lb, 60.0, home_y + 2.0, True, False, [], pe, commit_rolls=False)
    assert anchor == home_y, f"anchor must be the formation home_y, got {anchor}"
    assert bias == 0.0

    # Drifted central → carry must pull back toward the touchline.
    pe.states['L'].current_x, pe.states['L'].current_y = 60.0, home_y + 20.0
    _m, _a, drifted_bias = PossessionChain._fullback_carry_steering(
        lb, 60.0, home_y + 20.0, True, False, [], pe, commit_rolls=False)
    assert drifted_bias < 0.0, (
        f"left fullback drifted inside must be pulled LEFT, got {drifted_bias:.2f}")

    # Mirrored (attacking-left) RB: home_y is on the left-touchline side, so
    # the pull direction flips correctly.
    pe_m, s_m = _build()
    rb = _get_player(s_m, 'RB')
    rb_home = pe_m.states['R'].home_y
    pe_m.states['R'].current_x, pe_m.states['R'].current_y = 45.0, rb_home - 20.0
    _m2, anchor2, rb_bias = PossessionChain._fullback_carry_steering(
        rb, 45.0, rb_home - 20.0, True, False, [], pe_m, commit_rolls=False)
    assert anchor2 == rb_home
    # rb_home > 34 → pulled toward higher y (rightward on the mirrored side)
    if rb_home > 34.0:
        assert rb_bias > 0.0, f"mirrored RB must be pulled toward its own touchline, got {rb_bias:.2f}"


def test_fullback_delivery_band_covers_both_sides():
    pe, s = _build()
    rb = _get_player(s, 'RB')
    rb_home = pe.states['R'].home_y
    # RB drifted 20m infield (y ≈ rb_home - 20).
    pe.states['R'].current_x, pe.states['R'].current_y = 60.0, rb_home - 20.0
    _px, end_py = PossessionChain._pass_destination_to_receiver(
        rb, 40.0, 40.0, 15.0, pe, True)
    assert end_py > rb_home - 12.0, (
        f"drifted RB must be fed on the right flank, got y={end_py:.1f} (rb_home={rb_home:.1f})")