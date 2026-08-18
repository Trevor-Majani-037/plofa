"""
PLOFA 26/27 — BALL-CENTRIC ELLIPTICAL WEIGHTING TESTS (Checkpoint 20)
====================================================================
Property tests for the ball-centric elliptical weighting model:

    Pure geometry (position_engine.ball_centric_ellipse_weight)
      P1  anisotropy: ahead > behind > side at equal displacement
      P2  ellipse centre sits AHEAD of the ball (forward bias)
      P3  attacking-left mirrors attacking-right
      P4  weight stays within [0, 1] across the whole pitch

    PositionEngine integration
      P5  ball_centric_weight honours the team's attack direction
      P6  per-role sigma: an ST stays a live option further ahead than a CB
      P7  unknown player -> neutral weight (1.0), nothing breaks

    Receiver selection integration
      P8  _pick_receiver favours the receiver AHEAD of the ball over an
          equally-distant receiver BEHIND it (seeded, statistical)
      P9  the floor keeps a fully off-ellipse receiver a real option

    AttackingMatrix integration
      P10  _build_options penalises a lateral receiver vs a central one at
           equal depth (progress/depth/freedom/lane all equal — only the
           ellipse term differs)
"""

import math
import random

import pytest

from match_engine import (
    TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from event_chain import PossessionChain
from position_engine import (
    PositionEngine, ball_centric_ellipse_weight,
    ELLIPSE_SIGMA_ALONG, ELLIPSE_SIGMA_ACROSS,
    ELLIPSE_FORWARD_SHIFT, ELLIPSE_COMPOSE_FLOOR,
)
from player_dna import PlayerProfile, SquadBuilder
from attacking_matrix import AttackingMatrix

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


def _make_engine(att_squad, def_squad, att_team="Att FC", def_team="Def FC"):
    pe = PositionEngine()
    prof = TeamProfile(att_team, TeamStyle.BALANCED, PlayingStyle.MIXED, Intensity.MEDIUM)
    dprof = TeamProfile(def_team, TeamStyle.BALANCED, PlayingStyle.MIXED, Intensity.MEDIUM)
    pe.initialize_team(att_team, att_squad, prof, attacks_right=True)
    pe.initialize_team(def_team, def_squad, dprof, attacks_right=False)
    return pe, prof


def _place(pe, squad, mapping):
    # Exact-placement helper for geometry tests: bypass record_touch's
    # Checkpoint 21d wide-role flank-hold so a placed wide player stays
    # exactly at the requested coordinate (record_touch now deliberately
    # pulls LW/RW/LB/RB toward home_y).
    for p in squad:
        if p.name in mapping:
            s = pe.states.get(p.name)
            if s is not None:
                s.current_x, s.current_y = mapping[p.name]


def _player(squad, position):
    return next(p for p in squad if p.position == position)


# ── P1: ANISOTROPY ───────────────────────────────────────────────

def test_ellipse_anisotropy_ahead_behind_side():
    # All three receivers are the same euclidean distance (25m) from the ball.
    ahead = ball_centric_ellipse_weight(50, 34, 75, 34, True)
    behind = ball_centric_ellipse_weight(50, 34, 25, 34, True)
    side = ball_centric_ellipse_weight(50, 34, 50, 59, True)

    assert ahead > behind > side
    assert ahead > 0.5            # a 25m forward runner is a live option
    assert side < 0.1             # 25m lateral is not


# ── P2: FORWARD BIAS ─────────────────────────────────────────────

def test_ellipse_centre_is_ahead_of_the_ball():
    # The ellipse's peak sits ELLIPSE_FORWARD_SHIFT metres ahead of the ball.
    peak_x = 50 + ELLIPSE_FORWARD_SHIFT
    at_peak = ball_centric_ellipse_weight(50, 34, peak_x, 34, True)
    at_ball = ball_centric_ellipse_weight(50, 34, 50, 34, True)
    assert at_peak == pytest.approx(1.0, abs=1e-9)
    assert at_peak > at_ball


# ── P3: ATTACKING-LEFT MIRROR ────────────────────────────────────

def test_ellipse_mirrors_for_attacking_left():
    # Same geometry, mirrored: "ahead" becomes -x for an away team.
    r = ball_centric_ellipse_weight(50, 34, 75, 34, True)
    l = ball_centric_ellipse_weight(50, 34, 25, 34, False)
    assert r == pytest.approx(l, abs=1e-9)


# ── P4: BOUNDED WEIGHT ───────────────────────────────────────────

def test_ellipse_stays_in_unit_range_across_pitch():
    for bx, by, px, py in [
        (0, 34, 105, 34), (105, 68, 0, 0), (52.5, 34, 30, 60), (20, 8, 99, 2),
    ]:
        w = ball_centric_ellipse_weight(bx, by, px, py, True)
        assert 0.0 < w <= 1.0


# ── P5: ATTACK-DIRECTION HONOURED ────────────────────────────────

def test_position_engine_honours_attack_direction():
    att = _make_squad("Att FC")
    deff = _make_squad("Def FC")
    pe, _ = _make_engine(att, deff)

    st_att = _player(att, "ST")      # attacks right (goal at x=105)
    st_def = _player(deff, "ST")     # attacks left  (goal at x=0)

    # Away team's home positions are mirrored, so place both ahead of the ball.
    _place(pe, att, {st_att.name: (75, 34)})
    _place(pe, deff, {st_def.name: (25, 34)})

    w_att = pe.ball_centric_weight(st_att.name, 50, 34)
    w_def = pe.ball_centric_weight(st_def.name, 50, 34)
    assert w_att == pytest.approx(w_def, abs=1e-9)
    assert w_att > 0.5  # both are genuinely "ahead" of the ball


# ── P6: PER-ROLE SIGMA ───────────────────────────────────────────

def test_striker_stays_live_longer_than_centre_back():
    assert ELLIPSE_SIGMA_ALONG["ST"] > ELLIPSE_SIGMA_ALONG["CB"]

    st_w = ball_centric_ellipse_weight(
        50, 34, 86, 34, True, sigma_along=ELLIPSE_SIGMA_ALONG["ST"])
    cb_w = ball_centric_ellipse_weight(
        50, 34, 86, 34, True, sigma_along=ELLIPSE_SIGMA_ALONG["CB"])
    assert st_w > cb_w


# ── P7: UNKNOWN PLAYER IS NEUTRAL ────────────────────────────────

def test_unknown_player_is_neutral():
    att = _make_squad("Att FC")
    deff = _make_squad("Def FC")
    pe, _ = _make_engine(att, deff)
    assert pe.ball_centric_weight("Nobody Here", 50, 34) == 1.0


# ── P8: RECEIVER PICK FAVOURS THE AHEAD PLAYER ───────────────────

def test_receiver_pick_favours_ahead_of_ball():
    random.seed(2026)
    att = _make_squad("Att FC")
    deff = _make_squad("Def FC")
    pe, prof = _make_engine(att, deff)

    st = _player(att, "ST")
    cb = _player(att, "CB")

    # Ball at halfway. ST 30m AHEAD (toward 105); a CB 30m BEHIND.
    _place(pe, att, {st.name: (85, 34), cb.name: (25, 34)})
    passer = _player(att, "CM")

    counts = {"ahead": 0, "behind": 0}
    for _ in range(3000):
        pick = PossessionChain._pick_receiver(
            att, passer, 55, prof,
            position_engine=pe, y=34, def_players=[],
            attacks_right=True,
        )
        if pick is None:
            continue
        if pick.name == st.name:
            counts["ahead"] += 1
        elif pick.name == cb.name:
            counts["behind"] += 1

    # Checkpoint 21 note — the old guard demanded >3x ahead. The anti-cluster
    # rework (receive_option_quality) deliberately keeps BEHIND players alive
    # as recycle options (the ball must be able to leave the central clump
    # sideways and backwards), so forward dominance is now ~2.4x. The guard
    # still asserts the pass clearly favours the ahead runner and fails if a
    # regression ever makes behind receivers the dominant pick.
    assert counts["ahead"] > counts["behind"] * 2.0, counts


# ── P9: FLOOR KEEPS OFF-ELLIPSE RECEIVERS ALIVE ──────────────────

def test_floor_keeps_off_ellipse_receiver_alive():
    # A receiver completely off the ellipse retains at least the floor
    # fraction of their value — lateral/backward support is never zeroed.
    off = ball_centric_ellipse_weight(50, 34, 50, 65, True, sigma_across=4.0)
    assert off < 0.01
    composed = ELLIPSE_COMPOSE_FLOOR + (1.0 - ELLIPSE_COMPOSE_FLOOR) * off
    assert composed >= ELLIPSE_COMPOSE_FLOOR
    assert composed > 0.0


# ── P10: MATRIX OPTIONS PENALISE LATERAL AT EQUAL DEPTH ──────────

def test_matrix_build_options_penalise_lateral_receiver():
    att = _make_squad("Att FC")
    deff = _make_squad("Def FC")
    pe, _ = _make_engine(att, deff)

    carrier = _player(att, "ST")
    cbs = [p for p in att if p.position == "CB"][:2]

    # Same depth ahead of the ball; one central, one 25m off laterally.
    _place(pe, att, {
        carrier.name: (55, 34),
        cbs[0].name: (80, 34),   # central, on the ellipse
        cbs[1].name: (80, 59),   # same x-depth, far lateral
    })
    teammates = [p for p in att if p.position != "GK" and p.name != carrier.name]

    opts = AttackingMatrix._build_options(
        carrier, teammates, [], 55, 34, pe, True, team_profile=None,
    )
    by_name = {o.target.name: o for o in opts}
    central = by_name[cbs[0].name]
    lateral = by_name[cbs[1].name]

    # progress/freedom/depth/lane are identical (same tx, no defenders) —
    # only the ball-centric ellipse differs, so central must rank higher.
    assert central.progress == pytest.approx(lateral.progress, abs=1e-9)
    assert central.lane == lateral.lane == 1.0
    assert central.freedom == lateral.freedom == 1.0
    assert central.value > lateral.value
    assert lateral.value > 0.0  # floor keeps the wide option positive
