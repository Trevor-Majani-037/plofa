"""
PLOFA 26/27 — SET-PIECE ATTACKING ROUTINES (Feature #3)
========================================================
Guards the dead-ball routine layer:
  - chunk-stable corner / free-kick routine selection that rotates across the
    half and honours the chasing / protecting game-state commits,
  - squad-aerial steering (no leapers -> ground routines; jumps -> post pools),
  - routine-zone receiver weighting for the corner delivery,
  - delivery parameter mapping (target zone / swing / height bias / crowd),
  - end-to-end chain behaviour: a TRAINED_CROSS in the direct half-circle is
    worked into the box as a corner, and a direct routine strikes the wall.
"""

from datetime import date

import pytest

import set_piece_routines as spr
from set_piece_routines import (
    SetPieceRoutine,
    corner_routine_for,
    freekick_routine_for,
    corner_delivery,
    aerial_presence,
    receiver_weight,
)
from position_engine import PositionEngine


class _State:
    def __init__(self, gd=0, minute=30):
        self.goal_difference = gd
        self.minute = minute


# ── ROUTINE SELECTION ─────────────────────────

def test_corner_routine_is_chunk_stable():
    a = corner_routine_for("route_one", _State(), "T", "H", minute=62)
    b = corner_routine_for("route_one", _State(), "T", "H", minute=63)
    assert a is b  # same 5-minute block -> same committed routine
    assert a.is_corner


def test_corner_routine_rotates_across_chunks():
    seen = {corner_routine_for("wing_play", _State(), "T", "H", m)
            for m in range(0, 40, 5)}
    assert len(seen) > 1


def test_corner_routine_deterministic_same_chunk():
    assert corner_routine_for("tiki_taka", _State(), "T", "H", 10) \
        == corner_routine_for("tiki_taka", _State(), "T", "H", 10)


def test_chasing_commits_six_yard_pile():
    for m in range(0, 95, 5):
        assert corner_routine_for("tiki_taka", _State(), "T", "H", m,
                                  chasing=True) == SetPieceRoutine.SIX_YARD_PILE


def test_protecting_commits_short_corner():
    for m in range(0, 95, 5):
        assert corner_routine_for("route_one", _State(), "T", "H", m,
                                  protecting=True) == SetPieceRoutine.SHORT_CORNER


def test_possession_identity_leads_short_corner():
    # A tiki-taka side's pool is dominated by the short corner / worked ball,
    # so it lands there often but still retains its other schemes.
    from collections import Counter
    c = Counter(corner_routine_for("tiki_taka", _State(), "T", "H", m)
                for m in range(0, 90, 5))
    assert c[SetPieceRoutine.SHORT_CORNER] >= 3


def test_no_aerial_squad_stays_on_the_ground():
    for m in range(0, 95, 5):
        r = corner_routine_for("route_one", _State(), "T", "H", m,
                               aerial_score=0.20)
        assert r in (SetPieceRoutine.SHORT_CORNER,
                     SetPieceRoutine.PENALTY_SPOT_CROWD,
                     SetPieceRoutine.FAR_POST_OUTSWING)


def test_freekick_always_crossed_outside_direct_range():
    for m in range(0, 95, 5):
        assert freekick_routine_for("route_one", _State(), "T", "H", m,
                                    direct_range=False, taker_free_kick=0.9) \
            == SetPieceRoutine.TRAINED_CROSS


def test_chasing_freekick_commits_direct():
    for m in range(0, 95, 5):
        assert freekick_routine_for("tiki_taka", _State(), "T", "H", m,
                                    direct_range=True, taker_free_kick=0.5,
                                    chasing=True) == SetPieceRoutine.DIRECT_ATTEMPT


def test_protecting_freekick_commits_cross():
    for m in range(0, 95, 5):
        assert freekick_routine_for("route_one", _State(), "T", "H", m,
                                    direct_range=True, taker_free_kick=0.5,
                                    protecting=True) == SetPieceRoutine.TRAINED_CROSS


def test_weak_taker_leans_the_box_not_the_shot():
    from collections import Counter
    c = Counter(freekick_routine_for("route_one", _State(), "T", "H", m,
                                     direct_range=True, taker_free_kick=0.20)
                for m in range(0, 95, 5))
    # The no-strike pool is box-weighted: the cross clearly dominates.
    assert c[SetPieceRoutine.TRAINED_CROSS] >= \
        c[SetPieceRoutine.DIRECT_ATTEMPT] * 2


# ── SQUAD AERIAL PRESENCE ─────────────────────

def _fake_player(name, position, jumping=60.0, heading=60.0,
                 composure=60.0, ball_control=60.0, pace=60.0,
                 finishing=60.0):
    from player_dna import PlayerDNA
    dna = PlayerDNA(name=name, position=position)
    dna.physical.jumping = jumping
    dna.technical.heading = heading
    dna.mental.composure = composure
    dna.technical.ball_control = ball_control
    dna.technical.finishing = finishing
    dna.physical.pace = pace
    p = type("P", (), {})()
    p.name = name
    p.position = position
    p.dna = dna
    return p


def test_aerial_presence_scales_with_jumping():
    weak = [_fake_player("a", "CB", jumping=45, heading=50),
            _fake_player("b", "ST", jumping=45, heading=50)]
    strong = [_fake_player("a", "CB", jumping=95, heading=90),
              _fake_player("b", "ST", jumping=95, heading=90)]
    assert aerial_presence(strong) > aerial_presence(weak)


def test_aerial_presence_empty_is_neutral():
    assert aerial_presence([]) == 0.5


# ── RECEIVER WEIGHT ───────────────────────────

def test_receiver_weight_prefers_leapers_for_posts():
    leaper = _fake_player("j", "CB", jumping=95, heading=90,
                          composure=50, ball_control=50)
    finisher = _fake_player("f", "ST", jumping=50, heading=50,
                            composure=95, ball_control=50)
    assert receiver_weight("near", leaper) > receiver_weight("near", finisher)
    assert receiver_weight("far", leaper) > receiver_weight("far", finisher)


def test_receiver_weight_prefers_finisher_for_penalty_spot():
    leaper = _fake_player("j", "CB", jumping=95, heading=90,
                          composure=50, ball_control=50)
    finisher = _fake_player("f", "ST", jumping=50, heading=50,
                            composure=95, ball_control=50)
    assert receiver_weight("penalty", finisher) > receiver_weight("penalty", leaper)


def test_receiver_weight_baseline_zone_is_aerial():
    leaper = _fake_player("j", "CB", jumping=95, heading=90)
    small = _fake_player("s", "CM", jumping=40, heading=40)
    assert receiver_weight(None, leaper) > receiver_weight(None, small)


# ── DELIVERY PARAMETERS ───────────────────────

def test_corner_delivery_maps_zones_and_commitment():
    d = corner_delivery(SetPieceRoutine.SIX_YARD_PILE)
    assert d["target_zone"] == "six"
    assert d["crowd"] == 1.0
    assert corner_delivery(SetPieceRoutine.SHORT_CORNER)["target_zone"] == "edge"
    # Baseline (None) returns neutral defaults and never a committed pile.
    base = corner_delivery(None)
    assert base["crowd"] == 0.0
    assert base["target_zone"] is None


def test_corner_delivery_record_unchanged_by_default():
    # Repeated calls must not mutate the module map.
    a = corner_delivery(SetPieceRoutine.NEAR_POST_FLICKON)
    a["crowd"] = 99.0
    b = corner_delivery(SetPieceRoutine.NEAR_POST_FLICKON)
    assert b["crowd"] == 0.45


# ── END-TO-END CHAIN ──────────────────────────
# A TRAINED_CROSS in the direct half-circle must be worked into the box
# (FREEKICK_CROSS, then the corner delivery path), never struck the wall.

def _build_engine():
    from match_engine import (
        MatchConfig, MatchEngine, TeamProfile, TeamStyle, PlayingStyle, Intensity,
    )
    from player_dna import SquadBuilder
    home = SquadBuilder.build("Home", [
        ("GK", "GK", []), ("CB1", "CB", []), ("CB2", "CB", []), ("LB", "LB", []),
        ("RB", "RB", []), ("CDM", "CDM", ["anchor_man"]), ("CM1", "CM", []),
        ("CM2", "CM", []), ("LW", "LW", ["dribbler"]), ("ST", "ST", []),
        ("RW", "RW", ["grand_dribbler"]),
    ])
    away = SquadBuilder.build("Away", [
        ("AGK", "GK", []), ("A0", "CB", []), ("A1", "CB", []), ("A2", "CB", []),
        ("A3", "CB", []), ("A4", "CB", []), ("A5", "CM", []), ("A6", "CM", []),
        ("A7", "CM", []), ("A8", "ST", []), ("A9", "ST", []),
    ])
    cfg = MatchConfig(home_team="Home", away_team="Away",
                      match_date=date(2026, 8, 16), matchday=1)
    hp = TeamProfile("Home", TeamStyle.ROUTE_ONE, PlayingStyle.DIRECT, Intensity.HIGH)
    ap = TeamProfile("Away", TeamStyle.TIKI_TAKA, PlayingStyle.POSSESSION, Intensity.MEDIUM)
    eng = MatchEngine(cfg, hp, ap)
    eng.set_squad("Home", home["starters"])
    eng.set_squad("Away", away["starters"])
    return eng


def test_trained_cross_in_direct_range_works_the_box():
    from match_engine import SituationType
    from event_chain import ChainDispatcher
    eng = _build_engine()
    eng.state.last_ball_x = 84.0
    eng.state.last_ball_y = 34.0
    res = ChainDispatcher.set_piece(
        30, "Home", "Away",
        eng.active_players["Home"], eng.active_players["Away"],
        eng.state, SituationType.DIRECT_FREEKICK,
        attacks_right=True, position_engine=eng.position_engine,
        context_x=84.0, context_y=34.0,
        routine=SetPieceRoutine.TRAINED_CROSS,
    )
    types = [e.event_type.name for e in res.events]
    assert "FREEKICK_CROSS" in types       # worked into the box
    assert "FREEKICK_DIRECT" not in types  # never struck the wall
    assert "AERIAL_DUEL" in types          # dead-ball aerial delivery


def test_direct_routine_strikes_goal():
    from match_engine import SituationType
    from event_chain import ChainDispatcher
    eng = _build_engine()
    for attempt in range(8):
        res = ChainDispatcher.set_piece(
            30, "Home", "Away",
            eng.active_players["Home"], eng.active_players["Away"],
            eng.state, SituationType.DIRECT_FREEKICK,
            attacks_right=True, position_engine=eng.position_engine,
            context_x=84.0, context_y=34.0,
            routine=SetPieceRoutine.DIRECT_ATTEMPT,
        )
        types = [e.event_type.name for e in res.events]
        assert "FREEKICK_DIRECT" in types, types
        assert "FREEKICK_CROSS" not in types, types
        assert "GOAL" in types or "SAVE" in types or "SHOT_OFF_TARGET" in types
        break


def test_corner_routine_emits_metadata():
    from match_engine import SituationType
    from event_chain import ChainDispatcher
    eng = _build_engine()
    res = ChainDispatcher.set_piece(
        30, "Home", "Away",
        eng.active_players["Home"], eng.active_players["Away"],
        eng.state, SituationType.CORNER,
        attacks_right=True, position_engine=eng.position_engine,
        routine=SetPieceRoutine.SIX_YARD_PILE,
    )
    taken = next((e for e in res.events if e.event_type.name == "CORNER_TAKEN"), None)
    assert taken is not None
    assert taken.metadata
    assert taken.metadata.get("routine") == "six_yard_pile"
    assert taken.metadata.get("crowd") == 1.0


def test_matchengine_sp_routine_never_penalty():
    from match_engine import SituationType
    eng = _build_engine()
    assert eng._sp_routine("Home", SituationType.PENALTY, 30,
                           eng.active_players["Home"]) is None


def test_matchengine_sp_routine_corner_is_committed():
    from match_engine import SituationType
    eng = _build_engine()
    r = eng._sp_routine("Home", SituationType.CORNER, 30,
                        eng.active_players["Home"])
    assert r is None or r.is_corner