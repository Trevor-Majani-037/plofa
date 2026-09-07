"""
PLOFA 26/27 — TACTICAL SHAPES VALIDATION (Feature #1)
=====================================================
Validates the in-match formation/shape stance layer:
  - Stance selection matches the TacticalAI posture lenses (chasing -> all-out
    chase shape, protecting -> see-it-out block, man-down -> compact bank).
  - PositionEngine applies stance deltas additively over the authored shape,
    mirrors them for teams attacking left, clamps to the pitch, is idempotent,
    and reverts to base when the stance returns to baseline.
  - EffectiveTactics carries the same stance the engine applies.
"""

import pytest

from position_engine import PositionEngine
from tactical_shapes import (
    FormationStance,
    formation_stance_for,
    stance_delta_roles,
)
from attack_patterns import AttackPattern
from match_engine import (
    TeamProfile, TeamStyle, PlayingStyle, Intensity, MatchState,
)


class _State:
    def __init__(self, gd=0, minute=1):
        self.goal_difference = gd
        self.minute = minute


class _FakePlayer:
    def __init__(self, name, position):
        self.name = name
        self.position = position


def _profile(style=TeamStyle.BALANCED):
    return TeamProfile(
        name="Test Team",
        style=style,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM,
    )


def _build_engine(team="Test FC", away=False):
    pe = PositionEngine()
    players = [p for p in ("LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST")]
    players = [_FakePlayer(p, p) for p in players]
    pe.initialize_team(team, players, _profile(), attacks_right=not away)
    return pe


# ── STANCE SELECTION ─────────────────────────

def test_trailing_by_two_late_choses_all_out_chase():
    assert formation_stance_for(_profile(), _State(gd=-2, minute=85),
                                "T", "T") == FormationStance.ALL_OUT_CHASE


def test_trailing_by_one_late_pushes():
    assert formation_stance_for(_profile(), _State(gd=-1, minute=75),
                                "T", "T") == FormationStance.PUSHING


def test_leading_by_two_late_sees_it_out():
    assert formation_stance_for(_profile(), _State(gd=2, minute=85),
                                "T", "T") == FormationStance.SEE_IT_OUT


def test_leading_by_one_late_protects():
    assert formation_stance_for(_profile(), _State(gd=1, minute=80),
                                "T", "T") == FormationStance.PROTECT_LEAD


def test_level_late_is_tense():
    assert formation_stance_for(_profile(), _State(gd=0, minute=85),
                                "T", "T") == FormationStance.TENSE_LEVEL


def test_early_game_stays_authoring_shape():
    assert formation_stance_for(_profile(), _State(gd=-1, minute=10),
                                "T", "T") == FormationStance.BASELINE


def test_scoreline_is_directional_for_away_team():
    # Away team's goal_difference is inverted by the selector.
    assert formation_stance_for(_profile(), _State(gd=2, minute=85),
                                "Away", "Home") == FormationStance.ALL_OUT_CHASE
    assert formation_stance_for(_profile(), _State(gd=-2, minute=85),
                                "Away", "Home") == FormationStance.SEE_IT_OUT


# ── DELTA TABLES ──────────────────────────────

def test_all_out_chase_pushes_roles_forward():
    d = stance_delta_roles(FormationStance.ALL_OUT_CHASE)
    assert d["LB"][0] > 10.0 and d["RB"][0] > 10.0   # full-backs climb late
    assert d["ST"][0] > 0.0 and d["CAM"][0] > 0.0    # striker/CAM advance
    assert d["ST"][1] >= 0.0


def test_see_it_out_drops_wingers_and_keeps_striker_isolated():
    d = stance_delta_roles(FormationStance.SEE_IT_OUT)
    assert d["LW"][0] <= -15.0                        # wingers drop to wide-mid
    assert abs(d["ST"][0]) < abs(d["LW"][0])         # striker is the out ball
    assert d["LW"][1] * d["RW"][1] < 0.0             # both tuck toward centre


def test_baseline_is_noop():
    assert stance_delta_roles(FormationStance.BASELINE) == {}


def test_man_down_compacts_wide_players_toward_centre():
    base = stance_delta_roles(FormationStance.PUSHING, own_red_cards=0)
    man = stance_delta_roles(FormationStance.PUSHING, own_red_cards=1)
    # LW rests low y; man-down must pull it toward the centre (+y).
    assert man["LW"][1] > base.get("LW", (0.0, 0.0))[1]
    assert man["RW"][1] < base.get("RW", (0.0, 0.0))[1]


def test_fatigue_sags_the_shape():
    base = stance_delta_roles(FormationStance.BASELINE)
    tired = stance_delta_roles(FormationStance.BASELINE, avg_stamina=55.0)
    assert tired.get("CM", (0.0, 0.0))[0] < 0.0
    assert base == {}


# ── POSITION ENGINE APPLICATION ────────────────

def test_apply_stance_moves_homes_from_authoring_base():
    pe = _build_engine()
    base = {n: pe.states[n].home_x for n in pe.team_rosters["Test FC"]}
    changed = pe.apply_formation_stance("Test FC", FormationStance.ALL_OUT_CHASE)
    assert changed is True
    assert pe.states["LB"].home_x > base["LB"]
    assert pe.states["CAM"].home_x > base["CAM"]
    # Idempotent: same call again changes nothing.
    assert pe.apply_formation_stance("Test FC", FormationStance.ALL_OUT_CHASE) is False


def test_apply_stance_mirrors_for_left_attacking_team():
    pe = _build_engine(away=True)
    base = {n: pe.states[n].home_x for n in pe.team_rosters["Test FC"]}
    pe.apply_formation_stance("Test FC", FormationStance.SEE_IT_OUT)
    # Away team attacks LEFT: "forward" = decreasing x, so SEE_IT_OUT (drop
    # back) must INCREASE x for their forwards relative to the authored base.
    assert pe.states["CAM"].home_x > base["CAM"]
    # And the LB (own-left flank, high physical y for the away team) tucks
    # toward centre under SEE_IT_OUT → y decreases.
    assert pe.states["LB"].home_y < base["LB"]


def test_apply_stance_clamps_inside_pitch():
    pe = _build_engine()
    pe.apply_formation_stance("Test FC", FormationStance.ALL_OUT_CHASE)
    for n, st in pe.states.items():
        assert 4.0 <= st.home_x <= 101.0
        assert 2.0 <= st.home_y <= 66.0


def test_pattern_can_be_stack_replaced_and_cleared():
    pe = _build_engine()
    pe.apply_formation_stance("Test FC", FormationStance.ALL_OUT_CHASE)
    pe.apply_attack_pattern("Test FC", AttackPattern.OVERLOAD_RIGHT)
    overloaded_lb = pe.states["LB"].home_x
    pe.apply_attack_pattern("Test FC", AttackPattern.NONE)
    assert abs(pe.states["LB"].home_x - overloaded_lb) > 0.1   # reverted
    assert pe.apply_attack_pattern("Test FC", AttackPattern.NONE) is False


# ── TACTICAL AI PROPAGATION ─────────────────────

def test_effective_tactics_carries_the_stance():
    from tactical_ai import TacticalAI, EffectiveTactics
    state = MatchState(home_goals=0, away_goals=2, minute=88)
    eff = TacticalAI.adjust(_profile(), state, "T", "T")
    assert isinstance(eff, EffectiveTactics)
    assert eff.stance == FormationStance.ALL_OUT_CHASE
    # Neutral early game -> baseline.
    state2 = MatchState(home_goals=0, away_goals=0, minute=5)
    eff2 = TacticalAI.adjust(_profile(), state2, "T", "T")
    assert eff2.stance == FormationStance.BASELINE