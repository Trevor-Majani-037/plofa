"""
PLOFA 26/27 — ATTACK PATTERN LIBRARY VALIDATION (Feature #2)
============================================================
Validates the team-specific attack pattern layer:
  - Chunk-stable pattern selection (a team commits to a ~5-minute block) that
    still rotates and honours the chasing/protecting game-state overrides.
  - Favoured-flank semantics in normalised attack direction (mirror-safe).
  - Pattern role deltas compose with stance deltas on the PositionEngine and
    revert to the authored shape when the pattern is cleared.
"""

import pytest

from attack_patterns import (
    AttackPattern,
    pattern_for,
    pattern_role_deltas,
    favored_flank,
    flank_bias_multiplier,
)
from position_engine import PositionEngine
from tactical_shapes import FormationStance


class _State:
    def __init__(self, gd=0, minute=1):
        self.goal_difference = gd
        self.minute = minute


class _FakePlayer:
    def __init__(self, name, position):
        self.name = name
        self.position = position


def _build_engine(team="Test FC", attacks_right=True):
    pe = PositionEngine()
    players = [_FakePlayer(p, p) for p in
               ("LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST")]
    pe.initialize_team(team, players, type("P", (), {}), attacks_right=attacks_right)
    return pe


# ── SELECTION ─────────────────────────────────

def test_pattern_is_chunk_stable():
    a = pattern_for("wing_play", _State(), "T", "H", minute=62)
    b = pattern_for("wing_play", _State(), "T", "H", minute=63)
    assert a == b            # same 5-minute chunk -> same commit
    assert a is not AttackPattern.NONE


def test_pattern_rotates_across_chunks():
    seen = {pattern_for("wing_play", _State(), "T", "H", m)
            for m in range(0, 30, 5)}
    assert len(seen) > 1     # not frozen on one pattern all half


def test_deterministic_given_same_chunk():
    a = pattern_for("tiki_taka", _State(), "T", "H", 10)
    b = pattern_for("tiki_taka", _State(), "T", "H", 10)
    assert a == b


def test_chasing_leans_direct_channels():
    for minute in range(0, 95, 5):
        pat = pattern_for("tiki_taka", _State(), "T", "H", minute,
                          chasing=True)
        assert pat == AttackPattern.DIRECT_CHANNELS  # chasing overrides identity


def test_style_pools_respected():
    # A route-one side never runs a box-midfield possession pattern.
    for minute in range(0, 95, 5):
        assert pattern_for("route_one", _State(), "T", "H", minute) \
            == AttackPattern.DIRECT_CHANNELS


# ── FLANK SEMANTICS ───────────────────────────

def test_favored_flank_mapping():
    assert favored_flank(AttackPattern.OVERLOAD_RIGHT) == "R"
    assert favored_flank(AttackPattern.OVERLOAD_LEFT) == "L"
    assert favored_flank(AttackPattern.BOX_MIDFIELD) is None
    assert favored_flank(None) is None


def test_flank_bias_is_attack_direction_normalised():
    # Attacking RIGHT: own-right = high y.
    assert flank_bias_multiplier(58.0, True, "R") > 1.0
    assert flank_bias_multiplier(10.0, True, "R") < 1.0
    # Attacking LEFT: own-right = LOW physical y (the mirror).
    assert flank_bias_multiplier(10.0, False, "R") > 1.0
    assert flank_bias_multiplier(58.0, False, "R") < 1.0
    # Neutral / unknown pattern never distorts the pass map.
    assert flank_bias_multiplier(30.0, True, None) == 1.0


# ── ROLE DELTAS ───────────────────────────────

def test_pattern_deltas_nonempty_for_identity_patterns():
    for pat in (AttackPattern.OVERLOAD_RIGHT, AttackPattern.OVERLOAD_LEFT,
                AttackPattern.BOX_MIDFIELD,
                AttackPattern.WING_ISOLATION_LEFT, AttackPattern.DIRECT_CHANNELS):
        assert pattern_role_deltas(pat)  # non-empty
    assert pattern_role_deltas(AttackPattern.NONE) == {}


def test_overloads_target_the_named_side():
    right = pattern_role_deltas(AttackPattern.OVERLOAD_RIGHT)
    left = pattern_role_deltas(AttackPattern.OVERLOAD_LEFT)
    # Ball-side fullback advances on both mirrors; far-side FB tucks toward
    # centre (opposite y signs); CAM leans onto the overloaded channel.
    assert right["RB"][0] > 0.0 and left["LB"][0] > 0.0
    assert right["LB"][1] > 0.0 and left["RB"][1] < 0.0
    assert right["CAM"][1] > 0.0 and left["CAM"][1] < 0.0


# ── POSITION ENGINE INTEGRATION ───────────────

def test_apply_pattern_shifts_receiving_shape():
    pe = _build_engine(attacks_right=True)
    base_lb = pe.states["LB"].home_y
    base_rb = pe.states["RB"].home_y
    pe.apply_attack_pattern("Test FC", AttackPattern.OVERLOAD_RIGHT)
    # Far-side LB tucks toward centre (dy > 0 for low-y LB), RB stays wide.
    assert pe.states["LB"].home_y > base_lb
    assert pe.states["RB"].home_y <= base_rb + 0.6


def test_pattern_and_stance_compose_and_revert():
    pe = _build_engine(attacks_right=True)
    base_cam_x = pe.states["CAM"].home_x
    base_cam_y = pe.states["CAM"].home_y
    pe.apply_formation_stance("Test FC", FormationStance.ALL_OUT_CHASE)
    pe.apply_attack_pattern("Test FC", AttackPattern.OVERLOAD_RIGHT)
    # Stance pushed the CAM forward; pattern shifts him to the overload side.
    assert pe.states["CAM"].home_x > base_cam_x
    assert pe.states["CAM"].home_y > base_cam_y + 2.0
    # Clearing the pattern returns the OVERLOAD lean while keeping the stance.
    pe.apply_attack_pattern("Test FC", AttackPattern.NONE)
    assert pe.states["CAM"].home_y == pytest.approx(base_cam_y)
    pe.apply_formation_stance("Test FC", FormationStance.BASELINE)
    assert pe.states["CAM"].home_x == pytest.approx(base_cam_x)