"""
CHANCE PROVENANCE (2026-10-02)
=============================
The user asked whether key passes and assists carry true tracked coordinates.
Measured: they did not, and the reason was not the one expected. These tests
pin the four things that were actually wrong, so each cannot silently return:

  1. `CHANCE_CREATED` had a start point drawn from the global football RNG
     (`x - random.uniform(5, 20)`), matching no pass that player ever made,
     and an end forced onto the shot's taken location.
  2. `record_touch(creator, x - 8, y)` planted the assist-giver 8 m behind the
     shooter AND wrote it into the position engine, so every later position
     read inherited the fiction.
  3. Set-piece deliveries (`CORNER_TAKEN`, `FREEKICK_CROSS`) carried no
     destination at all — so the key passes the honest ledger finds were
     exactly the ones with no endpoint.
  4. A crossed free kick was struck from a CORNER ARC: `delivery_x/delivery_y`
     was computed and documented as the Law 11 origin, then the flight ignored
     it and launched from `corner_x, random.choice([1.0, 67.0])`.

Plus the frame bug found while measuring, which is why a corner's end point is
asserted to sit in the ATTACKING half: `ChainDispatcher.set_piece` defaults
`attacks_right=True` and the corner call site in `match_engine.py` omitted it,
so every away-team corner was resolved at its own end. 6 of 36 shots in a
two-match sample were taken on the shooting team's own half; all were the away
side. Now 0 of 42.

These are deliberately unit/single-chain tests, not full-match assertions. A
full match takes ~90-150 s here and the match is not reproducible from
`random.seed` (module-level brain/mind caches survive `simulate()`), so a
full-match test here would assert on noise.
"""

import ast
import math
import os
from datetime import date

import pytest

from event_chain import BaseChain, ChainDispatcher
from match_engine import (
    EventType, MatchConfig, MatchEngine, MatchState, SituationType,
    TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from player_dna import SquadBuilder
from position_engine import PositionEngine
from set_piece_routines import SetPieceRoutine


# ── helpers ──────────────────────────────────────────────────────

def _build_engine(home_name="Home"):
    home = SquadBuilder.build(home_name, [
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
    cfg = MatchConfig(home_team=home_name, away_team="Away",
                      match_date=date(2026, 8, 16), matchday=1)
    hp = TeamProfile(home_name, TeamStyle.ROUTE_ONE, PlayingStyle.DIRECT,
                     Intensity.HIGH)
    ap = TeamProfile("Away", TeamStyle.TIKI_TAKA, PlayingStyle.POSSESSION,
                     Intensity.MEDIUM)
    eng = MatchEngine(cfg, hp, ap)
    eng.set_squad(home_name, home["starters"])
    eng.set_squad("Away", away["starters"])
    return eng


def _corner(eng, att="Home", attacks_right=True):
    return ChainDispatcher.set_piece(
        30, att, "Away" if att == "Home" else "Home",
        eng.active_players[att], eng.active_players["Away" if att == "Home" else "Home"],
        eng.state, SituationType.CORNER,
        attacks_right=attacks_right, position_engine=eng.position_engine,
        routine=SetPieceRoutine.SIX_YARD_PILE,
    )


def _crossed_fk(eng, att="Home", attacks_right=True, fk_x=70.0, fk_y=12.0):
    return ChainDispatcher.set_piece(
        30, att, "Away" if att == "Home" else "Home",
        eng.active_players[att], eng.active_players["Away" if att == "Home" else "Home"],
        eng.state, SituationType.DIRECT_FREEKICK,
        attacks_right=attacks_right, position_engine=eng.position_engine,
        context_x=fk_x, context_y=fk_y,
        routine=SetPieceRoutine.TRAINED_CROSS,
    )


def _first(res, *names):
    want = {n for n in names}
    for e in res.events:
        if e.event_type.name in want:
            return e
    return None


# ── 1. PositionEngine.tracked_position ───────────────────────────

def test_tracked_position_is_none_for_an_untracked_player():
    """The reason the new accessor exists.

    `get_position` answers (50.0, 34.0) — the centre spot — for a player it has
    no state for. That is a coordinate which reads as real, so any caller
    about to WRITE a coordinate into an event must not use it. This is the same
    trap that made `_diag_wall_shape.py` measure four players all sitting on
    the centre spot.
    """
    eng = _build_engine()
    pe = eng.position_engine
    assert pe.get_position("Nobody At All") == (50.0, 34.0)   # the trap
    assert pe.tracked_position("Nobody At All") is None       # the fix


def test_tracked_position_returns_the_live_position_of_a_known_player():
    eng = _build_engine()
    pe = eng.position_engine
    for mp in eng.active_players["Home"]:
        at = pe.tracked_position(mp.name)
        assert at is not None, mp.name
        assert pe.get_position(mp.name) == at, mp.name


# ── 2. Set-piece deliveries carry a real destination ─────────────

def test_corner_delivery_has_a_real_destination():
    """Was 0/N. Emitted without one and stamped after `resolve_aerial`, because
    the aimed target is not known until the flight is built and the contact
    point is not known until the duel resolves."""
    eng = _build_engine()
    taken = _first(_corner(eng), "CORNER_TAKEN")
    assert taken is not None
    assert taken.end_x is not None and taken.end_y is not None, \
        "a corner with no destination is a zero-length 'key pass'"
    md = taken.metadata
    assert md.get("delivery_end_source") in ("contact", "aimed_target")
    # the stamped numbers and the event fields must agree
    assert md.get("delivery_end_x") == pytest.approx(taken.end_x, abs=0.02)
    assert md.get("delivery_end_y") == pytest.approx(taken.end_y, abs=0.02)


def test_corner_delivery_lands_in_the_attacking_box():
    eng = _build_engine()
    taken = _first(_corner(eng), "CORNER_TAKEN")
    d = abs(105.0 - taken.end_x)
    assert 3.0 <= d <= 25.0, f"corner delivered {d:.1f} m from the goal it attacks"


def test_away_team_corner_lands_at_the_away_end():
    """The frame bug. Attacking LEFT must put the delivery at low x."""
    eng = _build_engine()
    taken = _first(_corner(eng, att="Away", attacks_right=False), "CORNER_TAKEN")
    d = abs(0.0 - taken.end_x)
    assert 3.0 <= d <= 25.0, \
        f"away corner delivered {d:.1f} m from the goal it attacks"


def test_corner_names_the_player_it_was_aimed_at():
    eng = _build_engine()
    taken = _first(_corner(eng), "CORNER_TAKEN")
    assert taken.secondary_player, "no receiver, so the ledger cannot link it"
    names = {mp.name for mp in eng.active_players["Home"]}
    assert taken.secondary_player in names
    assert taken.metadata.get("delivery_target") in names


def test_crossed_freekick_has_a_real_destination():
    eng = _build_engine()
    res = _crossed_fk(eng)
    cross = _first(res, "FREEKICK_CROSS")
    assert cross is not None, [e.event_type.name for e in res.events]
    assert cross.end_x is not None and cross.end_y is not None
    assert cross.metadata.get("delivery_end_source") in ("contact", "aimed_target")


def test_crossed_freekick_is_struck_from_the_foul_spot():
    """The two-origins bug. `delivery_origin` is the Law 11 origin; the flight
    used to launch from `corner_x, random.choice([1.0, 67.0])` instead, so a
    crossed free kick was offside-judged from the foul spot and struck from a
    random corner flag."""
    from event_chain import SetPieceChain
    eng = _build_engine()
    fk_x, fk_y = 72.0, 9.0
    sub = SetPieceChain._corner_chain(
        30, "Home", "Away",
        eng.active_players["Home"], eng.active_players["Away"],
        eng.state, attacks_right=True, position_engine=eng.position_engine,
        routine=SetPieceRoutine.TRAINED_CROSS,
        delivery_origin=(fk_x, fk_y), is_corner=False,
    )
    taken = _first(sub, "CORNER_TAKEN")
    assert taken is not None
    assert (taken.location_x, taken.location_y) == (fk_x, fk_y), \
        "the ball was struck from a corner arc, not from the foul"


# ── 3. clamp_attack_x ────────────────────────────────────────────

@pytest.mark.parametrize("x", [0.0, 3.0, 15.0, 50.0, 85.0, 90.0, 102.0, 120.0])
def test_clamp_attack_x_is_exactly_the_old_clamp_when_attacking_right(x):
    """Attacking right must be byte-identical to the hard-coded clamp it
    replaced, or every calibration figure taken on that path is void."""
    old = max(85.0, min(102.0, x))
    assert BaseChain.clamp_attack_x(x, 85.0, 102.0, True) == pytest.approx(old)


@pytest.mark.parametrize("x", [0.0, 3.0, 15.0, 50.0, 85.0, 90.0, 102.0])
def test_clamp_attack_x_mirrors_the_band_when_attacking_left(x):
    mirrored = max(85.0, min(102.0, 105.0 - x))
    assert BaseChain.clamp_attack_x(x, 85.0, 102.0, False) \
        == pytest.approx(105.0 - mirrored)


def test_clamp_attack_x_never_leaves_the_band_either_way():
    for ar in (True, False):
        for x in range(-40, 146, 5):
            got = BaseChain.clamp_attack_x(float(x), 85.0, 100.0, ar)
            lo, hi = (85.0, 100.0) if ar else (5.0, 20.0)
            assert lo <= got <= hi, (x, ar, got)


# ── 4. every set_piece call site passes attacks_right ────────────

def test_every_set_piece_call_site_in_match_engine_passes_attacks_right():
    """`ChainDispatcher.set_piece` declares `attacks_right: bool = True`, so a
    call site that omits it silently resolves the set piece at the wrong end.
    The corner site did exactly that for the away team. This walks the AST
    rather than grepping text, so it cannot be satisfied by a mention in a
    comment or a call to something else of the same name."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "match_engine.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", None)
        if name != "set_piece":
            continue
        kwargs = {kw.arg for kw in node.keywords if kw.arg}
        sites.append((node.lineno, "attacks_right" in kwargs))
    assert sites, "no set_piece call sites found — did the dispatcher move?"
    missing = [ln for ln, ok in sites if not ok]
    assert not missing, (
        f"match_engine.py calls set_piece without attacks_right at line(s) "
        f"{missing}; the default is True, so those set pieces are resolved "
        f"at the wrong end of the pitch"
    )


# ── 5. CHANCE_CREATED provenance ─────────────────────────────────

def test_chance_created_origin_is_tracked_not_drawn():
    """`origin_known` / `origin_source` must exist, and a live chain must
    report a tracked origin. If `origin_known` ever goes false in a match
    where the whole XI is registered, the fabrication is back.

    Names are supplied here because the creator is now the REAL passer with no
    fallback: with no passer there is no creator and therefore no
    CHANCE_CREATED event at all. This test used to call the chain with no names
    and get an event anyway, because the creator was a weighted draw that
    always returned somebody. Supplying the passer is what makes it exercise
    the mechanism the assertion is actually about.
    """
    from event_chain import AttackChain
    eng = _build_engine()
    eng.position_engine.record_touch("LW", 58.0, 18.0, 30)
    res = ChainDispatcher.attack(
        30, "Home", "Away",
        eng.active_players["Home"], eng.active_players["Away"],
        eng.home_profile, eng.away_profile, eng.state,
        SituationType.OPEN_PLAY,
        position_engine=eng.position_engine,
        context_x=92.0, context_y=30.0, attacks_right=True,
        shooter_name="ST", assister_name="LW",
    )
    ch = _first(res, "CHANCE_CREATED", "BIG_CHANCE_CREATED")
    assert ch is not None, [e.event_type.name for e in res.events]
    assert ch.player == "LW", "the creator must be the named passer"
    assert ch.secondary_player == "ST", "the shooter must stay the receiver"
    md = ch.metadata
    assert md.get("origin_known") is True
    assert md.get("origin_source") == "tracked_position"
    # the origin is the creator's real position, so it is a plausible pitch
    # coordinate of a real man rather than an offset off the shot
    assert -5.0 <= ch.location_x <= 110.0
    assert -5.0 <= ch.location_y <= 73.0
    # and the pair is not the old fabrication signature
    assert not (5.0 <= math.hypot(ch.end_x - ch.location_x,
                                  ch.end_y - ch.location_y) <= 20.0
                and md.get("origin_source") != "tracked_position")


def test_unassisted_strike_emits_no_chance_creation_event():
    """The companion to the above, and the guard on the relationship between
    chances created, shot assists and assists.

    With no passer there is no creator and no CHANCE_CREATED event. Crediting
    the shooter instead would make the engine disagree with
    `ChanceCreationLedger._find_setup_pass`, which returns None for a dribble.
    See `tests/test_chance_truth.py` for the full argument.

    This was the SECOND instance of the same fixture defect: it called the
    chain once with no seeding and asserted a strike came out, but
    `AttackChain` is PROBABILITY-GATED, so the outcome depends on whatever ran
    before. Here the single draw produced `['HIT_WOODWORK']` and the assertion
    failed — and it is instructive *why* it failed:

      * `MatchEvent.is_shot` is a PROPERTY over a fixed event-type list
        (`match_engine.py:616`) that contains SHOT_ON_TARGET, SHOT_OFF_TARGET,
        SHOT_BLOCKED, GOAL, PENALTY_SCORED, PENALTY_MISSED — and NOT
        HIT_WOODWORK. A ball on target that hits the frame is therefore not
        `is_shot`.
      * That is a *deliberate and internally consistent* modelling choice, not
        an oversight: the exporter's "Total Shots" is
        `shots_on_target + shots_off_target + shots_blocked` (exporter.py:1758,
        3058, 4183) and it tracks woodwork as its own `hit_woodwork` column
        (exporter.py:954). So the two agree, and real football — where a
        woodwork strike IS a shot on target — is where the disagreement with
        Opta lives. That is a CALIBRATION question, filed below, and it is not
        silently changed here because it would move every shot number in the
        project.

    So the assertion is "a strike was RECORDED", which is what the test is
    actually about, and it sweeps seeds because the outcome is a draw.
    """
    import random as _r
    strikes = 0
    for seed in range(40):
        _r.seed(5200 + seed)
        eng = _build_engine()
        res = ChainDispatcher.attack(
            30, "Home", "Away",
            eng.active_players["Home"], eng.active_players["Away"],
            eng.home_profile, eng.away_profile, eng.state,
            SituationType.OPEN_PLAY,
            position_engine=eng.position_engine,
            context_x=92.0, context_y=30.0, attacks_right=True,
            shooter_name="ST", assister_name="",
        )
        assert _first(res, "CHANCE_CREATED", "BIG_CHANCE_CREATED") is None, (
            f"seed {seed}: an unassisted strike emitted a chance-creation "
            f"event, which would contradict the ledger's None-for-a-dribble")
        # and the strike itself must still be recorded
        recorded = [e for e in res.events
                    if e.is_shot or e.event_type == EventType.HIT_WOODWORK]
        strikes += len(recorded)
        for e in recorded:
            assert e.player == "ST", (
                f"seed {seed}: the named shooter must be the man who strikes "
                f"it, got {e.player!r}")
    assert strikes > 0, (
        "40 seeds produced no strike at all, so this test proved nothing — the "
        "assertions inside the loop were never reached")
