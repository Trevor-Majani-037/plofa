"""
PLOFA 26/27 — ATTACKING MATRIX TESTS (Checkpoint 10)
====================================================
Property tests for the High-Density Attacking Matrix module:

    Geometry properties
      P1  shot_score is monotonic as the carrier approaches the goal
      P2  pressure tax lowers shot_score
      P3  a central shooting position beats an equally-distant wide one
      P4  composure softens the pressure tax
      P5  lane_clearance: blocked < partial < clear, monotonic in distance
      P6  strategic_value: free+advanced option ranks above marked+deep one

    Decision properties
      P7  elite close-range chance resolves SHOOT
      P8  counter prefers a KEY_PASS to the far runner, build-up recycles
      P9  low_block panic shot vs balanced recycle from the same state
      P10  finishing quality flips the low-block decision
      P11  no options at all resolves RECYCLE_PASS

    Integration / preservation
      P12  scenario_for + network_zone + shooting_angle_degrees helpers
      P13  PossessionChain per-touch SHOOT hands off hand-off fields
      P14  no position engine => matrix falls back, chain unchanged
      P15  full-match smoke: matrix metadata appears, sim stays sane

All geometry is deterministic; randomness is only seeded where the
chain/match engines are involved.
"""

import copy
import math
import random
from datetime import date

import pytest

from match_engine import (
    MatchEngine, MatchConfig, MatchState, TeamProfile, TeamStyle,
    PlayingStyle, Intensity, EventType,
)
from event_chain import ChainDispatcher
from position_engine import PositionEngine
from player_dna import PlayerProfile, SquadBuilder, DNAFactory
from attacking_matrix import (
    AttackingDecision, AttackingMatrix,
    shot_score, lane_clearance, strategic_value,
    nearest_defender_dist, network_zone, scenario_for,
    shooting_angle_degrees, REFERENCE_ANGLE,
)

# ── FIXTURES / HELPERS ───────────────────────────────────────────

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


def _make_squad(team_name: str):
    starters = [
        (f"{team_name[:3]} {pos} {i}", pos, specs, 26)
        for i, (pos, specs) in enumerate(_ROLES)
    ]
    return SquadBuilder.build(team_name, starters)["starters"]


def _make_engine(att_squad, def_squad, att_team="Att FC", def_team="Def FC",
                 style=TeamStyle.BALANCED):
    pe = PositionEngine()
    prof = TeamProfile(att_team, style, PlayingStyle.MIXED, Intensity.MEDIUM)
    dprof = TeamProfile(def_team, TeamStyle.BALANCED, PlayingStyle.MIXED, Intensity.MEDIUM)
    pe.initialize_team(att_team, att_squad, prof, attacks_right=True)
    pe.initialize_team(def_team, def_squad, dprof, attacks_right=False)
    return pe, prof


def _place(pe, squad, mapping):
    """mapping: player name -> (x, y)"""
    # Exact-placement helper: bypass record_touch's Checkpoint 21d wide-role
    # flank-hold so a placed wide player stays exactly at the coordinate.
    for p in squad:
        if p.name in mapping:
            s = pe.states.get(p.name)
            if s is not None:
                s.current_x, s.current_y = mapping[p.name]


def _player(squad, position):
    return next(p for p in squad if p.position == position)


def _outfield_teammates(squad, carrier):
    return [p for p in squad if p.name != carrier.name and p.position != "GK"]


def _defender(squad):
    return _player(squad, "CB")


# ── P1: MONOTONIC SHOT SCORE ─────────────────────────────────────

def test_shot_score_monotonic_in_proximity():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "ST")
    cb = _defender(d)
    _place(pe, d, {cb.name: (10, 34)})  # far away -> no pressure tax

    scores = []
    for x in [60, 70, 80, 90, 96]:
        _place(pe, att, {carrier.name: (x, 34)})
        scores.append(shot_score(x, 34, [cb], pe, attacks_right=True, dna=carrier.dna))

    for a, b in zip(scores, scores[1:]):
        assert b >= a, f"shot score fell: {a} -> {b}"
    assert scores[-1] > scores[0]
    assert scores[-1] > 0.6  # close, central, no pressure -> dominant
    assert scores[0] < 0.3  # midfield distance -> heavily decayed


# ── P2: PRESSURE TAX ─────────────────────────────────────────────

def test_shot_score_pressure_tax():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "ST")
    cb = _defender(d)

    _place(pe, att, {carrier.name: (90, 34)})

    # Pressed: defender on top of the carrier
    _place(pe, d, {cb.name: (90, 34)})
    pressed = shot_score(90, 34, [cb], pe, attacks_right=True, dna=carrier.dna)

    # Free: same geometry, defender 10m away
    _place(pe, d, {cb.name: (100, 34)})
    free = shot_score(90, 34, [cb], pe, attacks_right=True, dna=carrier.dna)

    assert pressed < free
    assert free == pytest.approx(
        shot_score(90, 34, [], pe, attacks_right=True, dna=carrier.dna)
    )


# ── P3: CENTRAL BEATS WIDE AT EQUAL DISTANCE ─────────────────────

def test_shot_score_central_beats_wide():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "ST")

    # Both points are exactly 20m from the goal centre (105, 34)
    central_dist = math.hypot(105 - 85, 34 - 34)
    wide_dist = math.hypot(105 - 89, 34 - 46)
    assert central_dist == pytest.approx(wide_dist, abs=1e-9)

    central = shot_score(85, 34, [], pe, attacks_right=True, dna=carrier.dna)
    wide = shot_score(89, 46, [], pe, attacks_right=True, dna=carrier.dna)
    assert central > wide


# ── P4: COMPOSURE SOFTENS PRESSURE TAX ───────────────────────────

def test_shot_score_composure_sensitivity():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "ST")
    cb = _defender(d)

    _place(pe, att, {carrier.name: (90, 34)})
    _place(pe, d, {cb.name: (90, 33.5)})  # 0.5m -> hard pressure

    calm = copy.deepcopy(carrier.dna)
    calm.mental.composure = 95
    nervous = copy.deepcopy(carrier.dna)
    nervous.mental.composure = 5

    calm_score = shot_score(90, 34, [cb], pe, attacks_right=True, dna=calm)
    nervous_score = shot_score(90, 34, [cb], pe, attacks_right=True, dna=nervous)

    # Higher composure -> smaller pressure penalty -> higher score
    assert calm_score > nervous_score
    # And both are still below the unpressed baseline for the same player
    baseline = shot_score(90, 34, [], pe, attacks_right=True, dna=calm)
    assert calm_score < baseline


# ── P5: LANE CLEARANCE ───────────────────────────────────────────

def test_lane_clearance_blocked_partial_clear():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    cb = _defender(d)

    corridor = (70, 34, 90, 34)  # carrier -> teammate along the central line

    # Defender standing exactly on the corridor -> fully blocked
    _place(pe, d, {cb.name: (80, 34)})
    assert lane_clearance(*corridor, [cb], pe) == pytest.approx(0.0)

    # Defender 14m off the corridor -> fully clear
    _place(pe, d, {cb.name: (70, 20)})
    assert lane_clearance(*corridor, [cb], pe) == pytest.approx(1.0)

    # Defender 1.8m off -> partial (linear ramp between 1.2m and 3m)
    _place(pe, d, {cb.name: (80, 32.2)})
    assert lane_clearance(*corridor, [cb], pe) == pytest.approx(1.0 / 3.0, abs=0.05)


def test_lane_clearance_monotonic():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    cb = _defender(d)

    corridor = (70, 34, 90, 34)
    _place(pe, d, {cb.name: (80, 32.5)})  # 1.5m off the line
    near = lane_clearance(*corridor, [cb], pe)
    _place(pe, d, {cb.name: (80, 31.5)})  # 2.5m off the line
    far = lane_clearance(*corridor, [cb], pe)
    assert far > near
    assert 0.0 < near < 1.0
    assert 0.0 < far < 1.0


# ── P6: STRATEGIC VALUE ORDERING ─────────────────────────────────

def test_strategic_value_free_advanced_beats_marked_deep():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "CM")
    cb = _defender(d)

    _place(pe, att, {carrier.name: (70, 34)})

    # Free, advanced runner with a marker 10m away
    _place(pe, d, {cb.name: (85, 44)})
    free_advanced = strategic_value(
        70, 34, 95, 34, [cb], pe, attacks_right=True
    )

    # Marked, deep teammate with a defender ON the passing lane
    _place(pe, d, {cb.name: (50, 34)})
    marked_deep = strategic_value(
        70, 34, 50, 34, [cb], pe, attacks_right=True
    )

    assert free_advanced > marked_deep
    assert free_advanced > 0.8
    assert marked_deep == pytest.approx(0.0)


# ── P7: ELITE SHOT ───────────────────────────────────────────────

def test_decision_elite_shot_inside_box():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "ST")

    _place(pe, att, {carrier.name: (95, 34)})
    # Everyone else sits within 15m -> no far options to distract
    for i, tm in enumerate(_outfield_teammates(att, carrier)):
        _place(pe, att, {tm.name: (90 - (i % 3), 32 + (i % 5))})

    dec = AttackingMatrix.decide(
        carrier, _outfield_teammates(att, carrier), [],
        95, 34, position_engine=pe, attacks_right=True,
        team_profile=prof,
    )
    assert dec.action == "SHOOT"
    assert dec.shot_score >= 0.5


# ── P8: COUNTER vs BUILD-UP ──────────────────────────────────────

def test_counter_prefers_far_key_pass_build_up_recycles():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "CM")
    runner = _player(att, "ST")
    cb = _defender(d)

    _place(pe, att, {carrier.name: (40, 34)})
    for tm in _outfield_teammates(att, carrier):
        _place(pe, att, {tm.name: (28, 34)})
    _place(pe, att, {runner.name: (56, 34)})  # must be placed AFTER the bulk loop
    # Defender 2.28m off the through-ball corridor -> lane 0.60
    _place(pe, d, {cb.name: (48, 31.72)})

    lane = lane_clearance(40, 34, 56, 34, [cb], pe)
    assert lane == pytest.approx(0.60, abs=0.03)
    val = strategic_value(40, 34, 56, 34, [cb], pe)
    assert val == pytest.approx(0.516, abs=0.03)

    teammates = _outfield_teammates(att, carrier)

    counter_dec = AttackingMatrix.decide(
        carrier, teammates, [cb], 40, 34,
        position_engine=pe, attacks_right=True,
        scenario="counter",
    )
    assert counter_dec.action == "KEY_PASS"
    assert counter_dec.target is not None
    assert counter_dec.target.name == runner.name

    build_up_dec = AttackingMatrix.decide(
        carrier, teammates, [cb], 40, 34,
        position_engine=pe, attacks_right=True,
        scenario="build_up",
    )
    assert build_up_dec.action == "RECYCLE_PASS"
    assert build_up_dec.target is None or build_up_dec.target.name != runner.name


# ── P9: LOW-BLOCK PANIC vs BALANCED ──────────────────────────────

def test_low_block_panic_shot_balanced_recycles():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "CM")
    carrier.dna.technical.finishing = 55
    carrier.dna.technical.long_shots = 55

    _place(pe, att, {carrier.name: (94, 34)})
    for tm in _outfield_teammates(att, carrier):
        _place(pe, att, {tm.name: (50, 34)})  # far, no close options

    teammates = _outfield_teammates(att, carrier)

    low_block_dec = AttackingMatrix.decide(
        carrier, teammates, [], 94, 34,
        position_engine=pe, attacks_right=True,
        scenario="low_block",
    )
    assert low_block_dec.action == "SHOOT"

    balanced_dec = AttackingMatrix.decide(
        carrier, teammates, [], 94, 34,
        position_engine=pe, attacks_right=True,
        scenario="balanced",
    )
    assert balanced_dec.action == "RECYCLE_PASS"


# ── P10: FINISHING FLIPS THE LOW-BLOCK DECISION ──────────────────

def test_finishing_quality_flips_low_block_decision():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "ST")
    teammates = _outfield_teammates(att, carrier)

    _place(pe, att, {carrier.name: (94, 34)})
    for tm in teammates:
        _place(pe, att, {tm.name: (70, 34)})  # far options only

    elite = copy.deepcopy(carrier.dna)
    elite.technical.finishing = 90
    elite.technical.long_shots = 90
    elite.mental.decisions = 80
    elite.tendencies.plays_safe = 0.20
    elite.form.confidence = 80

    poor = copy.deepcopy(carrier.dna)
    poor.technical.finishing = 20
    poor.technical.long_shots = 20
    poor.mental.decisions = 50
    poor.tendencies.plays_safe = 0.70
    poor.form.confidence = 40

    elite_dec = AttackingMatrix.decide(
        carrier, teammates, [], 94, 34,
        position_engine=pe, attacks_right=True,
        scenario="low_block", dna=elite,
    )
    poor_dec = AttackingMatrix.decide(
        carrier, teammates, [], 94, 34,
        position_engine=pe, attacks_right=True,
        scenario="low_block", dna=poor,
    )

    assert elite_dec.action == "SHOOT"
    assert poor_dec.action == "RECYCLE_PASS"
    assert elite_dec.shot_score > poor_dec.shot_score


# ── P11: NO OPTIONS -> RECYCLE ───────────────────────────────────

def test_decision_recycle_when_everything_blocked():
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    carrier = _player(att, "CB")
    teammates = _outfield_teammates(att, carrier)

    _place(pe, att, {carrier.name: (30, 34)})
    spots = [(35, 34), (40, 34), (45, 34)]
    defenders = []
    for i, tm in enumerate(teammates):
        sx, sy = spots[i % len(spots)]
        _place(pe, att, {tm.name: (sx, sy)})
    # Every passing lane is blocked by an OUTFIELD defender standing on it
    outfield_def = [p for p in d if p.position != "GK"]
    for i, spot in enumerate(spots):
        _place(pe, d, {outfield_def[i].name: spot})
        defenders.append(outfield_def[i])

    dec = AttackingMatrix.decide(
        carrier, teammates, defenders, 30, 34,
        position_engine=pe, attacks_right=True,
        scenario="balanced",
    )
    assert dec.action == "RECYCLE_PASS"


# ── P12: SCENARIO / NETWORK / ANGLE HELPERS ──────────────────────

def test_scenario_for_mapping():
    def prof(style):
        return TeamProfile("X", style, PlayingStyle.MIXED, Intensity.MEDIUM)

    assert scenario_for(prof(TeamStyle.FLUID_COUNTER)) == "counter"
    assert scenario_for(prof(TeamStyle.ROUTE_ONE)) == "counter"
    assert scenario_for(prof(TeamStyle.PARK_THE_BUS)) == "low_block"
    assert scenario_for(prof(TeamStyle.ULTRA_DEFENSIVE)) == "low_block"
    assert scenario_for(prof(TeamStyle.DEFENSIVE)) == "low_block"
    assert scenario_for(prof(TeamStyle.TIKI_TAKA)) == "build_up"
    assert scenario_for(prof(TeamStyle.STRUCTURED_POSSESSION)) == "build_up"
    assert scenario_for(prof(TeamStyle.VERTICAL_TIKI_TAKA)) == "build_up"
    assert scenario_for(prof(TeamStyle.BALANCED)) == "balanced"
    assert scenario_for(prof(TeamStyle.ATTACKING)) == "balanced"


def test_network_zone():
    assert network_zone(5.0) == "close"
    assert network_zone(14.9) == "close"
    assert network_zone(15.0) == "far"
    assert network_zone(40.0) == "far"


def test_shooting_angle_degrees():
    assert shooting_angle_degrees(85, 34) == pytest.approx(0.0)
    assert shooting_angle_degrees(85, 54) == pytest.approx(45.0, abs=0.5)
    assert shooting_angle_degrees(96, 34) == pytest.approx(0.0)
    # Mirror behaviour for a team attacking left
    assert shooting_angle_degrees(20, 34, attacks_right=False) == pytest.approx(0.0)


def test_reference_angle_constant():
    assert REFERENCE_ANGLE == pytest.approx(2 * math.atan2(3.66, 11.0))


# ── P13: POSSESSION CHAIN SHOOT HAND-OFF ─────────────────────────

def test_possession_chain_shoot_handoff():
    random.seed(11)
    att = _make_squad("Att FC")
    d = _make_squad("Def FC")
    pe, prof = _make_engine(att, d)
    state = MatchState()

    st = _player(att, "ST")
    _place(pe, att, {st.name: (98, 34)})
    for tm in _outfield_teammates(att, st):
        _place(pe, att, {tm.name: (60, 34)})

    res = ChainDispatcher.possession(
        30, "Att FC", att, prof, state, 3,
        defending_players=[], position_engine=pe,
        context_x=98.0, context_y=34.0, attacks_right=True,
    )

    assert res.shoot_decision is True
    assert res.shoot_player
    # The ST may be passed the ball a touch or two before the SHOOT resolves;
    # the hand-off must reflect THAT touch (deep in the attacking box).
    assert res.shoot_x > 94.0
    assert 0.0 <= res.shoot_y <= 68.0
    assert res.shot_taken is True
    assert not res.possession_lost
    # The shot itself is NOT generated here — it hands off to AttackChain
    assert not any(e.event_type == EventType.SHOT_ATTEMPT for e in res.events)


# ── P14: NO POSITION ENGINE -> FALLBACK ──────────────────────────

def test_decide_fallback_without_position_engine():
    att = _make_squad("Att FC")
    carrier = _player(att, "ST")
    teammates = _outfield_teammates(att, carrier)

    dec = AttackingMatrix.decide(
        carrier, teammates, [], 95, 34,
        position_engine=None, attacks_right=True,
        scenario="balanced",
    )
    assert dec.action == "RECYCLE_PASS"
    assert dec.target is None
    assert dec.fallback is True


def test_fullback_prefers_same_flank_wing_receiver():
    att = _make_squad("Att FC")
    pe, prof = _make_engine(att, _make_squad("Def FC"))
    carrier = next(p for p in att if p.position == "LB")
    lw = next(p for p in att if p.position == "LW")
    rw = PlayerProfile(DNAFactory.create_archetype("traditional_winger", name="Right Wing"), team_name="Att FC")
    rw.dna.position = "RW"
    pe.register_substitute("Att FC", rw, prof)

    _place(pe, att, {
        carrier.name: (40.0, 10.0),
        lw.name: (30.0, 10.0),
    })
    pe.record_touch(rw.name, 30.0, 58.0, 0)

    opts = AttackingMatrix._build_options(
        carrier, [lw, rw], [], 40.0, 10.0, pe, True
    )
    lw_opt = next(o for o in opts if o.target.name == lw.name)
    rw_opt = next(o for o in opts if o.target.name == rw.name)

    assert lw_opt.value > rw_opt.value, (
        f"LB should prefer same-flank LW over opposite-flank RW when both are available, got {lw_opt.value:.3f} <= {rw_opt.value:.3f}"
    )


def test_possession_chain_unchanged_without_position_engine():
    random.seed(12)
    att = _make_squad("Att FC")
    prof = TeamProfile("Att FC", TeamStyle.BALANCED, PlayingStyle.MIXED, Intensity.MEDIUM)
    state = MatchState()

    res = ChainDispatcher.possession(
        30, "Att FC", att, prof, state, 4,
        defending_players=[], context_x=50.0, context_y=34.0,
    )

    assert res.shoot_decision is False
    assert res.shot_taken is False
    assert len(res.events) > 0
    assert not any(e.event_type == EventType.SHOT_ATTEMPT for e in res.events)


# ── P15: FULL MATCH SMOKE TEST ───────────────────────────────────

def test_back_to_keeper_intense_pressure_and_no_forward_open():
    random.seed(7)
    att = _make_squad('Att FC'); d = _make_squad('Def FC')
    pe, prof = _make_engine(att, d)
    carrier = _player(att, 'CB')
    # Mirror the chain's decide() call: teammates INCLUDING the keeper, who
    # is the release-valve target. (Outfield-only lists can never reach it.)
    tm = [p for p in att if p.name != carrier.name]

    _place(pe, att, {carrier.name: (30, 34)})
    _place(pe, d, {_player(d, 'LB').name: (30.5, 34)})
    gk = [p for p in att if p.position == 'GK'][0]
    _place(pe, att, {gk.name: (8, 34)})
    # The defending ST sits at (17, 34) in the default shape, directly on the
    # release lane to the keeper — that screens the outlet and correctly
    # blocks it. The scenario is "keeper unmarked", so move him wide.
    _place(pe, d, {_player(d, 'ST').name: (17, 62)})

    dec = AttackingMatrix.decide(carrier, tm, d[:], 30, 34,
                                 position_engine=pe, attacks_right=True,
                                 scenario='balanced')
    assert dec.action == "RECYCLE_PASS", f"Expected RECYCLE_PASS to GK under pressure, got {dec.action}"
    assert dec.target is not None, "Decision target should be non-None"
    assert dec.target.position == "GK", f"Decision target should be GK, got {dec.target.position}"
    assert "back_to_keeper" in dec.reason.lower(), f"Reason should mention back_to_keeper, got {dec.reason}"
