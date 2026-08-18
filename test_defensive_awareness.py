"""
PLOFA 26/27 — Defensive Awareness Tests (Checkpoint 9)
======================================================
test_defensive_awareness.py

Validates the Threat/Danger Intelligence feature end to end:

    1.  Monotonicity      — danger strictly rises as the ball approaches a team's
                           own goalpost xy, on BOTH halves (home defends x=0,
                           away defends x=105).
    2.  Central > wide    — a central ball is more dangerous than a wide one at
                           the same depth.
    3.  Zone ordering     — six_yard_box > inside_box > edge_of_box > outside_box > deep.
    4.  Zero-threat base  — the ball deep in a team's own half is a 0-danger
                           baseline (nothing changes for low-threat football).
    5.  Relief            — clearances lower the danger level (scaled by how far
                           the ball moved); scuffed clearances barely help; tackles/
                           interceptions/blocks apply WIN_RELIEF_FACTOR.
    6.  Headed vs foot    — DefensiveChain picks headed for an aerial ball, foot
                           otherwise; body_part honours preferred_foot; success
                           rate responds to aerial_dominance (headed) and
                           defending.clearances (foot).
    7.  Coordinated block — PositionEngine.defensive_block pulls the back four
                           goal-side and toward the ball's y; it is a strict
                           no-op below danger 25.
    8.  Preservation      — a full simulated match still runs, the MatchResult
                           carries the ThreatEngine, and reported goals_conceded
                           exactly matches the score.
"""

import random
from datetime import date
from types import SimpleNamespace

import pytest

from match_engine import (
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle,
    Intensity, EventType, MatchState, MatchPhase, GameState,
)
from event_chain import DefensiveChain, ChainDispatcher, ChainResult
from position_engine import PositionEngine
from player_dna import SquadBuilder
from threat_engine import (
    ThreatEngine, assess, danger_zone, band_of,
    danger_after_clearance, WIN_RELIEF_FACTOR,
    calculate_relative_ball_angle, defender_facing_point,
    orientation_zone, clearance_failure_multiplier,
    clearance_foot_for_angle, clearance_target_vector,
    apply_width_bias, own_goal_probability,
)


# ============================================================================
# Helpers
# ============================================================================

def _fake_event(event_type, team, bx, by, body_part=None, outcome=True,
                from_x=None, from_y=None):
    """A duck-typed timeline event the ThreatEngine can consume."""
    return SimpleNamespace(
        event_type=event_type,
        team=team,
        end_x=bx,
        end_y=by,
        location_x=bx if from_x is None else from_x,
        location_y=by if from_y is None else from_y,
        body_part=body_part,
        outcome=outcome,
    )


_ROLES = [
    ("GK", ["sweeper_keeper"]),
    ("CB", ["stopper_defender"]),
    ("CB", ["ball_playing_cb"]),
    ("LB", ["aggressive_fullback"]),
    ("RB", ["overlapping_fullback"]),
    ("CDM", ["anchor_man"]),
    ("CM", ["engine"]),
    ("CM", ["box_box"]),
    ("CAM", ["creator"]),
    ("LW", ["winger"]),
    ("ST", ["fox_in_box"]),
]


def _make_squad(team_name: str) -> list:
    """A deterministic 11-man squad built the way run_match.py builds them."""
    starters = [
        (f"{team_name[:3]} {pos} {i}", pos, specialties, 26)
        for i, (pos, specialties) in enumerate(_ROLES)
    ]
    squad = SquadBuilder.build(team_name, starters)
    return squad["starters"]


def _make_cb(team_name: str):
    """A single CB (with its mutable DNA) for clearance-attribute tests."""
    squad = _make_squad(team_name)
    return next(p for p in squad if p.position == "CB")


# ============================================================================
# 1. MONOTONICITY — danger rises as the ball approaches the defended goal
# ============================================================================

@pytest.mark.parametrize("own_goal_x", [0.0, 105.0])
def test_danger_is_monotonic_toward_own_goal(own_goal_x):
    """A team's danger must never fall as the ball moves closer to the
    goalpost xy it defends, all else equal. Sampled within the danger
    radius (<70m) along the central axis so every point is live."""
    approach = [60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0] if own_goal_x == 0.0 \
        else [45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0]
    levels = [assess(x, 34.0, own_goal_x).level for x in approach]
    for farther, nearer in zip(levels, levels[1:]):
        assert nearer >= farther, (
            f"danger must not fall as ball approaches: {nearer} !>= {farther}")
    assert levels[-1] > levels[0], "danger must be strictly higher near the goal"
    assert 0.0 <= levels[-1] <= 100.0


# ============================================================================
# 2. CENTRAL vs WIDE — same depth, central is worse
# ============================================================================

@pytest.mark.parametrize("own_goal_x", [0.0, 105.0])
def test_central_ball_more_dangerous_than_wide(own_goal_x):
    depth = 80.0 if own_goal_x == 105.0 else 25.0  # inside each team's danger radius
    central = assess(depth, 34.0, own_goal_x)
    wide = assess(depth, 64.0, own_goal_x)
    assert central.level > wide.level


# ============================================================================
# 3. ZONE ORDERING — the six-yard box is the most dangerous place
# ============================================================================

def test_zone_danger_ordering():
    own_goal_x = 105.0  # away defending
    levels = {}
    for zone, bx in [("six_yard_box", 101.0), ("inside_box", 92.0),
                     ("edge_of_box", 78.0), ("outside_box", 55.0),
                     ("deep", 15.0)]:
        assert danger_zone(bx, own_goal_x) == zone, f"zone misclassified for x={bx}"
        levels[zone] = assess(bx, 34.0, own_goal_x).level

    order = ["six_yard_box", "inside_box", "edge_of_box", "outside_box", "deep"]
    for hi, lo in zip(order, order[1:]):
        assert levels[hi] > levels[lo], f"{hi} ({levels[hi]}) !> {lo} ({levels[lo]})"


# ============================================================================
# 4. ZERO-THREAT BASELINE — own-half ball is a 0-danger starting point
# ============================================================================

@pytest.mark.parametrize("own_goal_x", [0.0, 105.0])
def test_zero_threat_baseline(own_goal_x):
    """A ball 70+m from the defended goalpost xy is a 0-danger baseline —
    the calm start state nothing about the defender should change for."""
    far_x = 75.0 if own_goal_x == 0.0 else 30.0
    a = assess(far_x, 34.0, own_goal_x)
    assert a.level == 0.0
    assert band_of(a.level) == "LOW"


# ============================================================================
# 5. RELIEF — clearances and defensive wins lower live danger
# ============================================================================

def test_clearance_relief_scales_with_distance():
    own_goal_x = 105.0
    base = 90.0
    hoof = danger_after_clearance(base, own_goal_x, 99.0, 34.0, 55.0, 34.0, True)
    short = danger_after_clearance(base, own_goal_x, 99.0, 34.0, 90.0, 34.0, True)
    assert hoof < short < base
    assert hoof < base * 0.5  # a big hoof cuts danger by more than half


def test_scuffed_clearance_barely_relieves():
    own_goal_x = 105.0
    base = 90.0
    scuffed = danger_after_clearance(base, own_goal_x, 99.0, 34.0, 95.0, 34.0, False)
    assert base * 0.90 < scuffed < base  # tiny drop, ball still in the box


def test_defensive_win_applies_flat_relief():
    te = ThreatEngine("Home FC", "Away FC")
    # Put the ball right at away's box (away defends x=105).
    te.observe_event(_fake_event(EventType.PASS, "Home FC", 95.0, 34.0), 30)
    before = te.danger_at("Away FC")
    assert before > 60.0
    # Interception wins it back -> flat relief applied to the danger at the
    # new (retreating) ball position.
    te.observe_event(_fake_event(EventType.INTERCEPTION, "Away FC", 78.0, 30.0), 31)
    expected = assess(78.0, 30.0, 105.0, approach=0.0).level * WIN_RELIEF_FACTOR
    assert te.danger_at("Away FC") == pytest.approx(expected, rel=0.01)
    assert te.danger_at("Away FC") < before


def test_threat_engine_counts_and_relieves_headed_clearance():
    te = ThreatEngine("Home FC", "Away FC")
    te.observe_event(_fake_event(EventType.CROSS_SUCCESS, "Home FC", 99.0, 34.0), 40)
    before = te.danger_at("Away FC")
    assert te.clearances["headed"]["Away FC"] == 0
    # A headed clearance away from goal.
    te.observe_event(
        _fake_event(EventType.CLEARANCE, "Away FC", 70.0, 20.0,
                    body_part="head", from_x=99.0, from_y=34.0), 41)
    assert te.clearances["headed"]["Away FC"] == 1
    assert te.clearances["foot"]["Away FC"] == 0
    assert te.danger_at("Away FC") < before * 0.5


def test_goal_peak_and_kickoff_reset():
    te = ThreatEngine("Home FC", "Away FC")
    te.observe_event(_fake_event(EventType.PASS, "Home FC", 95.0, 34.0), 40)
    te.on_goal("Away FC", 41)
    assert te.danger_at("Away FC") == 100.0   # the threat was realised
    assert te.goals_conceded["Away FC"] == 1
    te.on_kickoff(42)
    assert te.danger_at("Away FC") == 0.0     # ball back at centre circle
    assert te.danger_at("Home FC") == 0.0


def test_threat_engine_counts_foot_clearance():
    te = ThreatEngine("Home FC", "Away FC")
    te.observe_event(_fake_event(EventType.PASS, "Home FC", 96.0, 34.0), 50)
    te.observe_event(
        _fake_event(EventType.CLEARANCE, "Away FC", 62.0, 40.0,
                    body_part="right_foot", from_x=96.0, from_y=34.0), 51)
    assert te.clearances["foot"]["Away FC"] == 1
    assert te.clearances["headed"]["Away FC"] == 0


# ============================================================================
# 6. HEADED vs FOOT — DefensiveChain clearance mechanics
# ============================================================================

def test_clearance_kind_follows_aerials():
    assert DefensiveChain._clearance_kind(True) == "headed"
    assert DefensiveChain._clearance_kind(False) == "foot"


def test_clearance_body_part_honours_preferred_foot():
    right = _make_cb("Right FC")
    right.dna.preferred_foot = "right"
    assert DefensiveChain._clearance_body_part(right, "foot") == "right_foot"
    left = _make_cb("Left FC")
    left.dna.preferred_foot = "left"
    assert DefensiveChain._clearance_body_part(left, "foot") == "left_foot"
    assert DefensiveChain._clearance_body_part(left, "headed") == "head"


def test_clearance_success_responds_to_attributes():
    # High aerial dominance -> better headed clearances.
    high = _make_cb("High FC")
    low = _make_cb("Low FC")
    high.dna.physical.jumping = 99.0
    high.dna.technical.heading = 99.0
    high.dna.mental.bravery = 99.0
    low.dna.physical.jumping = 10.0
    low.dna.technical.heading = 10.0
    low.dna.mental.bravery = 10.0
    high.dna.defending.clearances = 95.0
    low.dna.defending.clearances = 10.0
    high.dna.defending.marking = 90.0
    low.dna.defending.marking = 10.0
    high.dna.mental.composure = 95.0
    low.dna.mental.composure = 10.0
    high.dna.mental.anticipation = 95.0
    low.dna.mental.anticipation = 10.0

    hr = DefensiveChain._clearance_success_rate(high, "headed", danger_level=0.0)
    lr = DefensiveChain._clearance_success_rate(low, "headed", danger_level=0.0)
    assert hr > lr + 0.15

    fr = DefensiveChain._clearance_success_rate(high, "foot", danger_level=0.0)
    fl = DefensiveChain._clearance_success_rate(low, "foot", danger_level=0.0)
    assert fr > fl + 0.15


def test_clearance_panic_in_critical_danger():
    d = _make_cb("Danger FC")
    calm = DefensiveChain._clearance_success_rate(d, "foot", danger_level=0.0)
    panicked = DefensiveChain._clearance_success_rate(d, "foot", danger_level=100.0)
    assert panicked < calm


# ============================================================================
# 7. COORDINATED BLOCK — PositionEngine.defensive_block
# ============================================================================

def _make_block_engine():
    profile = TeamProfile(name="Away FC", style=TeamStyle.BALANCED,
                          playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
    pe = PositionEngine()
    squad = _make_squad("Away FC")
    pe.initialize_team("Away FC", squad, profile, attacks_right=False)
    return pe, squad


def test_defensive_block_pulls_back_four_goal_side_and_toward_ball():
    pe, squad = _make_block_engine()
    cbs = [p for p in squad if p.position == "CB"]
    before = {p.name: pe.get_position(p.name) for p in cbs}

    # Ball at away's box (away defends x=105 via attacks_right=False -> own_goal_x passed explicitly).
    pe.defensive_block("Away FC", 88.0, 20.0, own_goal_x=105.0, danger_level=85.0,
                       pull_strength=1.0)

    for p in cbs:
        bx, by = before[p.name]
        ax, ay = pe.get_position(p.name)
        assert ax > bx, f"{p.name} did not move goal-side: {bx} -> {ax}"
        assert abs(ay - 20.0) < abs(by - 20.0), (
            f"{p.name} did not shift toward ball y: {by} -> {ay}")


def test_defensive_block_noop_below_danger_25():
    pe, squad = _make_block_engine()
    cbs = [p for p in squad if p.position == "CB"]
    before = {p.name: pe.get_position(p.name) for p in cbs}

    pe.defensive_block("Away FC", 88.0, 20.0, own_goal_x=105.0, danger_level=10.0,
                       pull_strength=1.0)

    for p in cbs:
        assert pe.get_position(p.name) == before[p.name], "block must be a no-op below danger 25"


# ============================================================================
# 8. PRESERVATION — a full match still runs and carries the ThreatEngine
# ============================================================================

def test_full_match_runs_with_threat_engine():
    random.seed(7)
    config = MatchConfig(home_team="Home FC", away_team="Away FC", match_date=date.today())
    home_profile = TeamProfile(name="Home FC", style=TeamStyle.ATTACKING,
                               playing_style=PlayingStyle.HIGH_PRESS, intensity=Intensity.HIGH)
    away_profile = TeamProfile(name="Away FC", style=TeamStyle.BALANCED,
                               playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)

    engine = MatchEngine(config, home_profile, away_profile)
    engine.set_squad("Home FC", _make_squad("Home FC"))
    engine.set_squad("Away FC", _make_squad("Away FC"))
    result = engine.simulate()

    # MatchResult carries the ThreatEngine.
    threat = getattr(result, "threat", None)
    assert threat is not None, "MatchResult must carry the ThreatEngine"

    # Report exposes both teams with the full awareness summary.
    report = threat.report()
    assert set(report.keys()) == {"Home FC", "Away FC"}
    for team in ("Home FC", "Away FC"):
        row = report[team]
        assert "peak_danger" in row and "danger_timeline" in row
        assert "clearances_headed" in row and "clearances_foot" in row
        assert 0.0 <= row["peak_danger"] <= 100.0
        assert len(row["danger_timeline"]) > 10

    # Reported goals conceded exactly match the score.
    total = (report["Home FC"]["goals_conceded"] + report["Away FC"]["goals_conceded"])
    assert total == len(result.goals)
    # Bound relaxed from 5 -> 8 for the attacking-matrix feature: the matrix
    # makes attacks more dangerous, so per-team concessions can exceed the old
    # pre-feature ceiling (A/B: mean 7.4 goals/match vs 3.7 baseline).
    for team in ("Home FC", "Away FC"):
        assert 0 <= report[team]["goals_conceded"] <= 8


# ============================================================================
# 9. BODY ORIENTATION & BIOMECHANICS — the spatial clearance layer
# ============================================================================

def test_relative_ball_angle_cardinal_directions():
    """0° = in front, ±90° = flanks, ±180° = behind the defender."""
    d_x, d_y, f_x, f_y = 0.0, 0.0, 10.0, 0.0   # facing east
    assert calculate_relative_ball_angle(d_x, d_y, f_x, f_y, 5.0, 0.0) == pytest.approx(0.0, abs=0.01)
    assert calculate_relative_ball_angle(d_x, d_y, f_x, f_y, 0.0, 5.0) == pytest.approx(90.0, abs=0.01)
    assert calculate_relative_ball_angle(d_x, d_y, f_x, f_y, 0.0, -5.0) == pytest.approx(-90.0, abs=0.01)
    assert abs(abs(calculate_relative_ball_angle(d_x, d_y, f_x, f_y, -5.0, 0.0)) - 180.0) < 0.01


def test_relative_ball_angle_wraps_without_jumping():
    """atan2 normalization keeps the angle in (-180, +180] across the wrap."""
    ang = calculate_relative_ball_angle(0.0, 0.0, 0.0, 10.0, 5.0, 0.0)
    assert ang == pytest.approx(-90.0, abs=0.01)   # east ball vs north-facing defender
    ang2 = calculate_relative_ball_angle(0.0, 0.0, -10.0, 0.0, 5.0, 0.0)
    assert abs(abs(ang2) - 180.0) < 0.01


def test_defender_facing_point_geometry():
    # Away defends x=105. Ball (96) behind the defender (90) -> beaten -> faces own goal.
    f = defender_facing_point(90.0, 34.0, 96.0, 34.0, 105.0)
    assert f == (105.0, 34.0)
    # Defender (88) still goal-side of the ball (82) -> faces the ball.
    assert defender_facing_point(88.0, 34.0, 82.0, 34.0, 105.0) == (82.0, 34.0)
    # Mirror for home defending x=0.
    fh = defender_facing_point(10.0, 34.0, 6.0, 34.0, 0.0)
    assert fh == (0.0, 34.0)


def test_orientation_zone_thresholds():
    assert orientation_zone(0.0) == "optimal"
    assert orientation_zone(30.0) == "optimal"     # inclusive boundary
    assert orientation_zone(30.1) == "flank"
    assert orientation_zone(90.0) == "flank"       # inclusive boundary
    assert orientation_zone(90.1) == "blind"
    assert orientation_zone(180.0) == "blind"
    assert orientation_zone(-45.0) == "flank"
    assert orientation_zone(-150.0) == "blind"


def test_clearance_failure_multiplier_scales():
    assert clearance_failure_multiplier(0.0) == pytest.approx(1.0)
    assert clearance_failure_multiplier(60.0) == pytest.approx(1.15)     # flank
    assert clearance_failure_multiplier(150.0) == pytest.approx(1.40)    # blind
    contested = clearance_failure_multiplier(0.0, contested_distance=0.5)
    assert contested == pytest.approx(1.30)
    tired = clearance_failure_multiplier(0.0, stamina=20.0)
    assert tired == pytest.approx(1.21)
    worst = clearance_failure_multiplier(170.0, contested_distance=0.3, stamina=15.0)
    assert worst > 2.0   # blind + contested + fatigued stacks hard


def test_clearance_foot_for_angle():
    assert clearance_foot_for_angle(0.0, "right") == "right_foot"
    assert clearance_foot_for_angle(0.0, "left") == "left_foot"
    assert clearance_foot_for_angle(60.0, "right") == "left_foot"    # ball on left flank
    assert clearance_foot_for_angle(-60.0, "left") == "right_foot"   # ball on right flank
    assert clearance_foot_for_angle(29.0, "left") == "left_foot"     # dead zone keeps preferred


def test_clearance_target_vector_points_away():
    vx, vy = clearance_target_vector(96.0, 34.0, 105.0)
    assert vx < 0                      # away from away's own goal (x=105)
    vx2, vy2 = clearance_target_vector(9.0, 40.0, 0.0)
    assert vx2 > 0                     # away from home's own goal (x=0)


def test_width_bias_avoids_zone_14():
    # Central landing 30m from the defended goal line -> Zone-14 nudge wide.
    assert abs(apply_width_bias(75.0, 40.0, 105.0) - 34.0) >= 8.0
    # A wide ball is untouched.
    assert apply_width_bias(75.0, 55.0, 105.0) == 55.0
    # A landing outside the 22-40m band is untouched.
    assert apply_width_bias(60.0, 40.0, 105.0) == 40.0


def test_own_goal_probability_peaks_in_panic():
    calm = own_goal_probability(0.0, contested_distance=3.0, stamina=95.0, danger_level=0.0)
    panic = own_goal_probability(175.0, contested_distance=0.2, stamina=15.0, danger_level=100.0)
    assert 0.0 < calm < 0.01
    assert panic > calm * 5
    assert own_goal_probability(180.0, 0.1, 10.0, 100.0) <= 0.20   # clamped cap


def test_clearance_kind_uses_ball_height():
    assert DefensiveChain._clearance_kind(True, 0.5) == "foot"     # low ball beats aerial flag
    assert DefensiveChain._clearance_kind(False, 1.5) == "headed"  # high ball beats low flag
    assert DefensiveChain._clearance_kind(True) == "headed"        # no height -> aerial flag
    assert DefensiveChain._clearance_kind(False) == "foot"


def _failed_clearances(result) -> int:
    n = 0
    for e in result.events:
        if e.event_type == EventType.CLEARANCE and not e.outcome:
            n += 1
        if e.event_type == EventType.OWN_GOAL:
            n += 1
    return n


def _defense_state():
    state = MatchState(minute=70, home_goals=0, away_goals=0)
    state.phase = MatchPhase.PEAK_INTENSITY
    return state


def test_blind_clearance_fails_more_often_than_optimal():
    random.seed(11)
    state = _defense_state()
    home = _make_squad("Home FC")
    away = _make_squad("Away FC")

    fails_optimal = 0
    fails_blind = 0
    for _ in range(300):
        # Optimal: goal-side defender faces the ball, fresh legs, free of pressure.
        r = DefensiveChain.generate(
            70, "Away FC", "Home FC", away, home, state, "clearance",
            context_x=96.0, context_y=34.0,
            danger_level=90.0, ball_aerial=False, own_goal_x=105.0,
            stamina=95.0, opponent_distance=4.0,
        )
        fails_optimal += _failed_clearances(r)

        # Blind: beaten defender facing their own goal, ball dropped over the
        # shoulder, striker breathing down their neck, dead legs.
        r2 = DefensiveChain.generate(
            70, "Away FC", "Home FC", away, home, state, "clearance",
            context_x=88.0, context_y=40.0,
            danger_level=95.0, ball_aerial=True, own_goal_x=105.0,
            defender_facing_x=105.0, defender_facing_y=34.0,
            opponent_distance=0.4, stamina=30.0,
        )
        fails_blind += _failed_clearances(r2)

    assert fails_blind > fails_optimal


def test_own_goal_on_critical_clearance_failure(monkeypatch):
    import event_chain as ec
    monkeypatch.setattr(ec, "own_goal_probability", lambda *a, **k: 1.0)
    monkeypatch.setattr(ec, "random", random.Random(42))

    state = _defense_state()
    state.minute = 88
    home = _make_squad("Home FC")
    away = _make_squad("Away FC")

    seen = False
    for _ in range(300):
        r = DefensiveChain.generate(
            88, "Away FC", "Home FC", away, home, state, "clearance",
            context_x=90.0, context_y=34.0,
            danger_level=95.0, ball_aerial=True, own_goal_x=105.0,
            defender_facing_x=105.0, defender_facing_y=34.0,
            opponent_distance=0.3, stamina=25.0,
        )
        if r.own_goal:
            seen = True
            assert r.goal_scored is True
            assert r.goal_team == "Home FC"          # attacking team credited
            assert r.goal_scorer != ""               # defender's name recorded
            assert r.possession_lost is True
            og = [e for e in r.events if e.event_type == EventType.OWN_GOAL]
            assert len(og) == 1
            assert og[0].team == "Away FC"           # own goal logged for the defender's side
            meta = r.events[-1].metadata
            assert meta.get("orientation_zone") in ("optimal", "flank", "blind")
            assert "relative_angle" in meta
            break
    assert seen, "forced own-goal path never fired"


def test_clearance_metadata_carries_biomechanics():
    state = _defense_state()
    home = _make_squad("Home FC")
    away = _make_squad("Away FC")
    r = DefensiveChain.generate(
        70, "Away FC", "Home FC", away, home, state, "clearance",
        context_x=95.0, context_y=34.0,
        danger_level=88.0, ball_aerial=True, own_goal_x=105.0,
        stamina=72.0, opponent_distance=0.8,
    )
    clearance = next(e for e in r.events if e.event_type == EventType.CLEARANCE)
    meta = clearance.metadata
    assert meta["clearance_type"] == "headed"
    assert "relative_angle" in meta and isinstance(meta["relative_angle"], (int, float))
    assert meta["orientation_zone"] in ("optimal", "flank", "blind")
    assert meta["stamina"] == 72.0
    assert meta["contested"] == 0.8
    assert clearance.body_part == "head"


def test_absorb_chain_counts_own_goal_on_scoreboard():
    """An OWN_GOAL produced by a critical clearance failure must register on
    the scoreboard for the attacking team and in the goals list."""
    from datetime import date
    config = MatchConfig(home_team="Home FC", away_team="Away FC", match_date=date.today())
    home_profile = TeamProfile(name="Home FC", style=TeamStyle.BALANCED,
                               playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
    away_profile = TeamProfile(name="Away FC", style=TeamStyle.BALANCED,
                               playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
    engine = MatchEngine(config, home_profile, away_profile)
    engine.set_squad("Home FC", _make_squad("Home FC"))
    engine.set_squad("Away FC", _make_squad("Away FC"))

    cr = ChainResult()
    cr.goal_scored = True
    cr.goal_team = "Home FC"          # attacking team is credited
    cr.goal_scorer = "OG Player"
    cr.own_goal = True
    cr.events.append(DefensiveChain.make_event(
        70, EventType.OWN_GOAL, "Away FC", "OG Player",
        MatchPhase.PEAK_INTENSITY, GameState.LEVEL,
        location_x=96.0, location_y=34.0,
    ))

    engine._absorb_chain(cr, 70)
    assert engine.state.home_goals == 1
    assert engine.state.away_goals == 0
    assert len(engine.goals) == 1
    assert engine.goals[0].event_type == EventType.OWN_GOAL
    # The threat was realised for the conceding side.
    assert engine.threat.goals_conceded["Away FC"] == 1
