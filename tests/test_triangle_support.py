"""Checkpoint 36 — TRIANGLE SUPPORT RULE guard tests.

Midfielders must be ABLE to complete a passing triangle: the classic
third-man shape both on the wings (near-side CM joining the winger +
full-back cluster) and in the centre (CM pair split around the ball with
the CDM pivot behind). Before this checkpoint the live 10 Hz machine only
compacted midfields 7% toward the ball's y, so the near-side CM physically
could never reach a wide-support socket.

Contracts covered here:
1. WIDE ball on a flank -> near-side CM commits to a half-space support
   socket off the wide cluster; far-side CM gets only a subtle balance
   shift; CDM pivot slides toward the ball side.
2. CENTRAL ball -> the CM pair splits either side of the ball and the CDM
   drops deeper as the pivot (the central triangle).
3. The rule is IN-POSSESSION ONLY (out of possession = block/press duty).
4. Sockets stay INSIDE the pitch (half-space, never the touchline) and the
   steer is a pace-preserving TARGET steer, not a position jump.
5. Roles outside CDM/CM/CAM are never steered.
"""
from position_engine import (
    PositionEngine,
    TRI_HALF_SPACE_MAX, TRI_NEAR_ALPHA,
    TRI_FAR_ALPHA, TRI_PIVOT_ALPHA, TRI_CENTRAL_ALPHA,
)
from player_dna import SquadBuilder


def _build_engine(attacks_right=True):
    squad = SquadBuilder.build('Test FC', [
        ('GK', 'GK', []), ('CB1', 'CB', []), ('CB2', 'CB', []),
        ('LB', 'LB', []), ('RB', 'RB', []), ('CDM', 'CDM', ['anchor_man']),
        ('CM1', 'CM', []), ('CM2', 'CM', []),
        ('LW', 'LW', ['dribbler', 'speedster']),
        ('ST', 'ST', []), ('RW', 'RW', ['grand_dribbler']),
    ])
    pe = PositionEngine()
    pe.initialize_team('Test FC', squad['starters'],
                       None, attacks_right=attacks_right)
    return pe


def _names(pe, *positions):
    return [n for n in pe.team_rosters.get('Test FC', [])
            if pe.states[n].position in positions]


def test_wide_ball_near_cm_commits_to_flank_socket():
    pe = _build_engine(attacks_right=True)
    # Lineup order: the first CM is the left man, the second the right.
    cm_left, cm_right = _names(pe, 'CM')

    # Ball on the RIGHT flank -> the right CM is near and commits.
    tri = pe.midfielder_triangle_support('Test FC', 78.0, 58.0, True, True)
    a_r, x_r, y_r = tri[cm_right]
    assert abs(a_r - TRI_NEAR_ALPHA) < 1e-9
    # Near socket is on the ball side, inside the pitch, capped to half-space.
    assert 34.0 < y_r <= 34.0 + TRI_HALF_SPACE_MAX
    assert 0.0 <= x_r <= 105.0

    # Left CM is the far-side balance man here.
    a_l, x_l, y_l = tri[cm_left]
    assert a_l <= TRI_FAR_ALPHA + 1e-9

    # Mirror on the LEFT flank: the left CM becomes the near man.
    tri_l = pe.midfielder_triangle_support('Test FC', 78.0, 10.0, True, True)
    assert tri_l[cm_left][0] == TRI_NEAR_ALPHA
    assert tri_l[cm_left][2] < 34.0


def test_pivot_slides_and_far_cm_balances_on_wide_ball():
    pe = _build_engine(attacks_right=True)
    cdm = _names(pe, 'CDM')[0]
    cm_left, cm_right = _names(pe, 'CM')
    tri = pe.midfielder_triangle_support('Test FC', 78.0, 58.0, True, True)
    a, x, y = tri[cdm]
    assert abs(a - TRI_PIVOT_ALPHA) < 1e-9
    assert y > 34.0  # pivot slides toward the ball side, not away
    # Far-side CM balance shift is subtle, never the near-side commit.
    assert abs(tri[cm_left][0] - TRI_FAR_ALPHA) < 1e-9
    assert tri[cm_left][2] < 34.0  # drifts the other way to balance width


def test_central_ball_forms_central_triangle():
    pe = _build_engine(attacks_right=True)
    cdm = _names(pe, 'CDM')[0]
    cm1, cm2 = _names(pe, 'CM')
    tri = pe.midfielder_triangle_support('Test FC', 60.0, 34.0, True, True)
    y_cm1 = tri[cm1][2]
    y_cm2 = tri[cm2][2]
    # CM pair splits either side of the ball (left man down to 26, right up
    # to 42 — 8m off each side of the central ball line).
    assert abs(y_cm1 - 34.0) == abs(y_cm2 - 34.0) == 8.0
    assert y_cm1 != y_cm2
    # CDM drops DEEPER behind the play than either CM (own-goal side), so
    # the pivot is farther from the opposition goal.
    assert tri[cdm][1] < tri[cm1][1]
    assert tri[cdm][1] < tri[cm2][1]
    assert tri[cdm][0] == TRI_CENTRAL_ALPHA


def test_out_of_possession_no_triangle_steer():
    pe = _build_engine(attacks_right=True)
    assert pe.midfielder_triangle_support('Test FC', 78.0, 58.0, False, True) == {}


def test_non_midfielders_never_steered():
    pe = _build_engine(attacks_right=True)
    tri = pe.midfielder_triangle_support('Test FC', 78.0, 58.0, True, True)
    for n in tri:
        assert pe.states[n].position in ("CDM", "CM", "CAM")


def test_sockets_always_inside_pitch_and_target_steer_pace_safe():
    pe = _build_engine(attacks_right=True)
    for bx in (30.0, 60.0, 90.0):
        for by in (6.0, 22.0, 34.0, 46.0, 62.0):
            tri = pe.midfielder_triangle_support('Test FC', bx, by, True, True)
            for n, (alpha, sx, sy) in tri.items():
                assert 0.0 < alpha <= TRI_NEAR_ALPHA, "alpha bounded"
                assert 0.0 <= sy <= 68.0, "socket y inside the pitch"
                assert 0.0 <= sx <= 105.0, "socket x inside the pitch"


def test_initial_position_unchanged_by_steer_only():
    """The steer returns a TARGET (no position mutation) — realism gate."""
    pe = _build_engine(attacks_right=True)
    before = {n: (pe.states[n].current_x, pe.states[n].current_y)
              for n in pe.team_rosters['Test FC']}
    pe.midfielder_triangle_support('Test FC', 78.0, 58.0, True, True)
    after = {n: (pe.states[n].current_x, pe.states[n].current_y)
             for n in pe.team_rosters['Test FC']}
    assert before == after


def test_live_10hz_integrator_moves_near_cm_into_flank_socket():
    """REAL-machinery integration: drive MatchEngine._offball_run with a wide
    possession ball and confirm the near-side CM physically wanders toward the
    ball side at jog (the triangle actually materialises) while the far-side
    CM holds the balance side."""
    from datetime import date
    from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
    home = SquadBuilder.build('Home', [
        ('GK', 'GK', []), ('CB1', 'CB', []), ('CB2', 'CB', []),
        ('LB', 'LB', []), ('RB', 'RB', []), ('CDM', 'CDM', ['anchor_man']),
        ('CM1', 'CM', []), ('CM2', 'CM', []),
        ('LW', 'LW', ['dribbler', 'speedster']),
        ('ST', 'ST', []), ('RW', 'RW', ['grand_dribbler']),
    ])
    away = SquadBuilder.build('Away', [
        ('AGK', 'GK', []), ('A0', 'CB', []), ('A1', 'CB', []),
        ('A2', 'CB', []), ('A3', 'CB', []), ('A4', 'CB', []),
        ('A5', 'CM', []), ('A6', 'CM', []), ('A7', 'CM', []),
        ('A8', 'ST', []), ('A9', 'ST', []),
    ])
    config = MatchConfig(home_team='Home', away_team='Away',
                         match_date=date(2026, 8, 16), matchday=1)
    hp = TeamProfile('Home', TeamStyle.BALANCED, PlayingStyle.HIGH_PRESS, Intensity.HIGH)
    ap = TeamProfile('Away', TeamStyle.FLUID_COUNTER, PlayingStyle.COUNTER, Intensity.MEDIUM)
    engine = MatchEngine(config, hp, ap)
    engine.set_squad('Home', home['starters'])
    engine.set_squad('Away', away['starters'])

    # Wide possession on the RIGHT flank, attacking right.
    engine.state.last_ball_x = 78.0
    engine.state.last_ball_y = 58.0
    engine.state.possession_team = 'Home'
    engine.state.cross_active = False
    engine.state.match_clock_s = 0.0
    engine._on_ball_this_minute = set()
    engine._patrol = {}
    engine._chase_state = {}
    engine._minute_start_clock = 0.0
    pe = engine.position_engine
    engine._minute_start_snapshot = {
        n: (pe.states[n].current_x, pe.states[n].current_y)
        for n in pe.team_rosters['Home'] + pe.team_rosters['Away']
    }
    cms = [n for n in pe.team_rosters['Home'] if pe.states[n].position == 'CM']
    cm_left, cm_right = cms  # lineup order: left (home_y 24), right (44)
    assert pe.states[cm_left].home_y < pe.states[cm_right].home_y

    start_l = pe.states[cm_left].current_y
    start_r = pe.states[cm_right].current_y

    # 20 s of real 10 Hz jogging with the ball wide on the right.
    engine._offball_run(20.0, True)

    end_l = pe.states[cm_left].current_y
    end_r = pe.states[cm_right].current_y
    # Near-side (right) CM steps toward the flank vs his start; the far-side
    # (left) CM holds/balances the other way. All movement is jog-capped.
    assert end_r > start_r + 1.5, \
        f"Near CM receded from the wide cluster: {start_r:.1f} -> {end_r:.1f}"
    assert end_l < start_l + 1.0, \
        f"Far CM should not drift toward the ball side: {start_l:.1f} -> {end_l:.1f}"
    assert end_r > end_l, \
        f"Near CM must end up on the ball side of the far CM: " \
        f"near={end_r:.1f} far={end_l:.1f}"