"""Checkpoint 35 — PITCH-STRETCH RULE guard tests.

Wide roles (LW/RW/LB/RB) are the team's WIDTH PROVIDERS. When the middle of
the pitch is packed (live ball on the 24-44m spine) they hold/push to the
touchline channel — a RULE, not a preference — and every stretch movement
stays INSIDE the pitch bounds.

Contracts covered here:
1. pitch_spine_weight — 1.0 on the spine, 0.0 on the touchlines, linear
   taper through the 18/50m band edges.
2. wide_stretch_blend — only wide roles, scaled by the spine weight, 0 when
   the ball is already wide.
3. receive_option_quality — an IN-CHANNEL wide outlet under a central ball is
   floored to ~0.92 (a live, near-central outlet); a wide player who has
   left his channel gets NO floor (anisotropy + post discipline survive).
4. record_touch — a central touch snaps the wide player back toward his
   touchline channel (harder than a half-space touch) and always clamps
   inside the pitch; an on-flank touch still tracks exactly.
5. drift_minute — the flank-anchor pull is boosted by the spine weight so the
   wide player re-asserts the touchline while the middle is packed.
"""
from match_engine import TeamProfile, TeamStyle, PlayingStyle, Intensity, MatchPhase
from position_engine import (
    PositionEngine, pitch_spine_weight,
    WIDE_OUTLET_DIRECTION_FLOOR, STRETCH_FLOOR_LIFT,
    STRETCH_CLAMP_LOW, STRETCH_CLAMP_HIGH,
)
from player_dna import SquadBuilder


def _profile():
    return TeamProfile(
        name='Test FC', style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM,
    )


def _build_engine(attacks_right=True):
    """Full 4-3-3 so the LW/RW are registered in the winger registry."""
    squad = SquadBuilder.build('Test FC', [
        ('GK', 'GK', []), ('CB1', 'CB', []), ('CB2', 'CB', []),
        ('LB', 'LB', ['aggressive_fullback']), ('RB', 'RB', ['overlapping_fullback']),
        ('CDM', 'CDM', ['anchor_man']), ('CM1', 'CM', []), ('CM2', 'CM', []),
        ('LW', 'LW', ['dribbler', 'speedster', 'traditional_winger']),
        ('ST', 'ST', ['clinical_finisher']),
        ('RW', 'RW', ['grand_dribbler', 'inverted']),
    ])
    pe = PositionEngine()
    profile = TeamProfile(
        name='Test FC', style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM,
    )
    pe.initialize_team('Test FC', squad['starters'], profile,
                       attacks_right=attacks_right)
    return pe


def test_pitch_spine_weight_values():
    assert pitch_spine_weight(6.0) == 0.0
    assert pitch_spine_weight(17.0) == 0.0
    assert pitch_spine_weight(34.0) == 1.0
    assert pitch_spine_weight(44.0) == 1.0
    assert pitch_spine_weight(50.0) == 0.0
    assert pitch_spine_weight(60.0) == 0.0
    # Linear taper on the band edges (peak is exactly 1.0 at 24/44).
    assert 0.0 < pitch_spine_weight(21.0) < pitch_spine_weight(28.0) == 1.0
    assert 0.0 < pitch_spine_weight(47.0) < 1.0


def test_wide_stretch_blend_only_wide_roles_and_spine_scaled():
    pe = _build_engine(attacks_right=True)
    # LW in-channel: central ball -> active stretch; wide ball -> no stretch.
    assert 0.0 < pe.wide_stretch_blend('LW', 34.0) <= 0.30
    assert pe.wide_stretch_blend('LW', 6.0) == 0.0
    # Deep spine is the strongest stretch, band edge is weakest.
    assert pe.wide_stretch_blend('LW', 34.0) > pe.wide_stretch_blend('LW', 21.0) > 0.0
    # Full-backs are width providers too.
    assert pe.wide_stretch_blend('RB', 34.0) > 0.0
    # Non-wide central roles are never stretch-steered.
    assert pe.wide_stretch_blend('CB1', 34.0) == 0.0
    assert pe.wide_stretch_blend('CM1', 34.0) == 0.0
    assert pe.wide_stretch_blend('ST', 34.0) == 0.0


def test_receive_floor_makes_in_channel_wide_outlet_live_under_central_ball():
    pe = _build_engine(attacks_right=True)
    lw_state = pe.states['LW']
    home_y = lw_state.home_y
    # LW standing exactly on his flank channel.
    lw_state.current_y = home_y
    quality_in_channel = pe.receive_option_quality('LW', 60.0, 34.0)

    # A wide player who left his channel gets NO floor — the flank is empty.
    lw_state.current_y = 40.0
    quality_off_channel = pe.receive_option_quality('LW', 60.0, 34.0)

    # The rule: an in-channel wide outlet under a packed middle is a live,
    # near-central option; an empty flank is a terrible target.
    assert quality_in_channel > 0.60, \
        f"In-channel LW under central ball should be a live outlet, " \
        f"got {quality_in_channel:.3f}"
    assert quality_off_channel < quality_in_channel - 0.15, \
        f"Off-channel flank must NOT be preferred: in={quality_in_channel:.3f} " \
        f"off={quality_off_channel:.3f}"


def test_record_touch_snaps_central_touch_to_touchline_and_clamps_bounds():
    pe = _build_engine(attacks_right=True)
    lw_state = pe.states['LW']
    home_y = lw_state.home_y

    # Central touch (ball on the spine) — the winger was dragged into the
    # packed middle; the stretch duty snaps him back to the line...
    pe.record_touch('LW', 30.0, 34.0, minute=45)
    snapped_y = lw_state.current_y
    assert snapped_y < home_y + 10.0, \
        f"Central touch should rip the LW toward y={home_y:.1f}, got {snapped_y:.1f}"

    # ...and always stays inside the pitch bounds.
    pe.record_touch('LW', 30.0, 6.0, minute=46)
    assert lw_state.current_y == 6.0
    pe.record_touch('LW', 30.0, 34.0, minute=47)
    assert STRETCH_CLAMP_LOW - 0.01 <= lw_state.current_y <= STRETCH_CLAMP_HIGH + 0.01


def test_record_touch_on_own_flank_tracks_exactly():
    pe = _build_engine(attacks_right=True)
    lw_state = pe.states['LW']
    home_y = lw_state.home_y
    pe.record_touch('LW', 30.0, home_y, minute=45)
    assert abs(lw_state.current_y - home_y) < 0.1, \
        "On-flank touch must track exactly (no hold correction)"


def test_nonwide_central_touch_still_tracks_exactly():
    pe = _build_engine(attacks_right=True)
    pe.record_touch('ST', 30.0, 34.0, minute=45)
    assert abs(pe.states['ST'].current_y - 34.0) < 0.1, \
        "Non-wide roles must not be flank-steered on touch"


def test_drift_flank_pull_boosted_by_spine_weight():
    # Single-player team (out of possession) isolates the flank-anchor loop:
    # no wide runs, no line cohesion, no graph relaxation.
    def _one_engine():
        squad = SquadBuilder.build('Test FC', [
            ('GK', 'GK', []), ('CB1', 'CB', []), ('CB2', 'CB', []),
            ('LB', 'LB', []), ('RB', 'RB', []), ('CDM', 'CDM', []),
            ('CM1', 'CM', []), ('CM2', 'CM', []),
            ('LW', 'LW', ['dribbler', 'speedster']),
            ('ST', 'ST', []), ('RW', 'RW', []),
        ])
        pe = PositionEngine()
        pe.initialize_team('Test FC', squad['starters'], _profile(),
                           attacks_right=True)
        return pe

    ph = type('P', (), {'value': 'second_open'})()
    home_y = None

    def _drift(ball_y):
        nonlocal home_y
        pe = _one_engine()
        lw = pe.states['LW']
        home_y = lw.home_y
        lw.current_x, lw.current_y = 60.0, 30.0   # dragged 24m infield
        pe.drift_minute('Test FC', _profile(), ph, game_state_gd=0, minute=45,
            in_possession=False, ball_x=60.0, ball_y=ball_y)
        return lw.current_y

    y_packed = _drift(34.0)     # middle packed -> stretch duty on
    y_wide = _drift(6.0)        # ball wide -> no stretch duty

    assert y_packed < y_wide, \
        f"Central ball must pin the LW wider than a wide ball: " \
        f"packed={y_packed:.1f} wide={y_wide:.1f} (home_y={home_y:.1f})"
    assert STRETCH_CLAMP_LOW - 0.01 <= y_packed <= STRETCH_CLAMP_HIGH + 0.01