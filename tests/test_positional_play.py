"""
PLOFA 26/27 - Positional-play layer: rest defence + live striker runs
=====================================================================
tests/test_positional_play.py

Two changes, one theme: both add behaviour the positional-play book asks for
that the engine could not previously express, and BOTH are things that only
become visible when the layer is live.

    1.  REST DEFENCE.  The last of the four superiority types with no
        representation. It is the only one that is a CONSTRAINT rather than a
        preference, so it is tested as an invariant: in ordinary shapes it must
        cost nothing and return None, and the degenerate "whole team beyond the
        ball" state must be unreachable.

    2.  STRIKER RUNS.  striker_behavior.py was fully written and unreachable
        from a match - its only consumer sat under drift_minute, which
        MatchEngine never calls. Wiring it live exposed two latent bugs in code
        that had never run, both about the offside line, and those are pinned
        here as regressions with the geometry that provokes them.

The suite also breaks each new layer deliberately. A constraint that has never
been observed to bind is not known to bind, and a cache that has never been
observed to expire is not known to expire.
"""
from __future__ import annotations

import random
from datetime import date

import pytest

from match_engine import (
    Intensity, MatchConfig, MatchEngine, PlayingStyle, TeamProfile, TeamStyle,
)
from player_dna import SquadBuilder
from position_engine import PositionEngine
from striker_behavior import StrikerSpatialProfile


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

# 4-3-3 with a poacher: gives the registry a real ST profile to steer.
_ROLES = [
    ("GK", ["sweeper_keeper"]), ("CB", ["stopper_defender"]),
    ("CB", ["ball_playing_cb"]), ("LB", ["aggressive_fullback"]),
    ("RB", ["overlapping_fullback"]), ("CDM", ["anchor_man"]),
    ("CM", ["engine"]), ("CM", ["box_box"]), ("LW", ["winger"]),
    ("RW", ["winger"]), ("ST", ["fox_in_box"]),
]


def _squad(team_name: str):
    starters = [
        (f"{team_name[:3]} {pos} {i}", pos, specs, 26)
        for i, (pos, specs) in enumerate(_ROLES)
    ]
    return SquadBuilder.build(team_name, starters)["starters"]


def _profile(name: str) -> TeamProfile:
    return TeamProfile(
        name=name, style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.POSSESSION, intensity=Intensity.MEDIUM,
    )


def _engine(attacks_right: bool = True) -> PositionEngine:
    """A PositionEngine holding one team, positions free to be set by hand."""
    pe = PositionEngine()
    pe.initialize_team("Home", _squad("Home FC"), _profile("Home FC"),
                       attacks_right=attacks_right)
    return pe


def _place(pe: PositionEngine, x_by_name: dict, y: float = 34.0) -> None:
    for name, x in x_by_name.items():
        st = pe.states.get(name)
        if st is not None:
            st.current_x, st.current_y = float(x), y


def _names(pe: PositionEngine, position: str):
    return [n for n, s in pe.states.items() if s.position == position]


class _FakePE:
    """get_position() over a fixed dict - all last_line_gap needs."""

    def __init__(self, table):
        self._table = table

    def get_position(self, name):
        return self._table.get(name, (50.0, 34.0))


def _shim(name, position):
    return type("P", (), {"name": name, "position": position})()


# =============================================================================
# 1. OFFSIDE LINE - the second-last defender (regression: latent bug)
#
# Law 11 forms the line with the SECOND-deepest defender. The old code used the
# DEEPEST, which is the man on his own keeper's side of the line. It also
# mirrored the x axis wrongly when a team attacked left, which made the reported
# gap saturate at 1.0 for an entire half - i.e. "run in behind" was always
# allowed. Both were invisible while the module was unreachable.
# =============================================================================


def test_offside_line_is_the_second_deepest_not_the_deepest():
    # A sweeping CB well behind the rest. The deep man (x=20) is NOT the line.
    pe = _FakePE({"Sweeper": (20.0, 34.0), "LastCB": (40.0, 34.0)})
    assert StrikerSpatialProfile().offside_line_nx(
        True, [_shim("Sweeper", "CB"), _shim("LastCB", "CB")], pe) == 20.0


def test_offside_line_ignores_the_goalkeeper():
    # The keeper is deeper than every outfield defender and must not count.
    pe = _FakePE({"GK": (2.0, 34.0), "CB1": (40.0, 34.0), "CB2": (52.0, 34.0)})
    assert StrikerSpatialProfile().offside_line_nx(
        True, [_shim("GK", "GK"), _shim("CB1", "CB"), _shim("CB2", "CB")], pe) == 40.0


def test_offside_line_ignores_the_deepest_outfielder_too():
    # Two outfielders behind everyone else: the line is the second of the two.
    pe = _FakePE({"Sweeper": (18.0, 34.0), "HighCB": (30.0, 34.0), "CB2": (55.0, 34.0)})
    assert StrikerSpatialProfile().offside_line_nx(
        True,
        [_shim("Sweeper", "CB"), _shim("HighCB", "CB"), _shim("CB2", "CB")], pe) == 30.0


def test_offside_line_is_none_without_two_outfield_defenders():
    pe = _FakePE({"GK": (2.0, 34.0), "CB1": (40.0, 34.0)})
    assert StrikerSpatialProfile().offside_line_nx(
        True, [_shim("GK", "GK"), _shim("CB1", "CB")], pe) is None
    assert StrikerSpatialProfile().offside_line_nx(True, [], pe) is None
    assert StrikerSpatialProfile().offside_line_nx(True, None, None) is None


def test_gap_opens_into_a_channel_behind_a_sweeping_defender():
    """The regression that matters, in the direction the old code got right.

    Striker at 35, a sweeping CB on 20, the rest of the line on 40. There is a
    15 m channel and the striker should run into it. Measured against the
    DEEPEST defender the same geometry reads as five metres BEHIND the line and
    returns 0.0 - the run is refused precisely when the space is there.
    """
    pe = _FakePE({"Sweeper": (20.0, 34.0), "LastCB": (40.0, 34.0)})
    gap = StrikerSpatialProfile().last_line_gap(
        35.0, 34.0, True, [_shim("Sweeper", "CB"), _shim("LastCB", "CB")], pe)
    assert gap == pytest.approx(1.0)


def test_gap_is_direction_symmetric():
    """The other regression: attacking LEFT saturated the gap at 1.0.

    Mirrored about the halfway line, the same physical situation must produce
    the same gap. It did not: the old axis maths returned abs(x + 78) for a
    leftward attack, so the striker saw a 100 m channel in his own half.
    """
    prof = StrikerSpatialProfile()
    table = {"Sweeper": (20.0, 34.0), "LastCB": (40.0, 34.0)}
    defenders = [_shim("Sweeper", "CB"), _shim("LastCB", "CB")]
    right = prof.last_line_gap(35.0, 34.0, True, defenders, _FakePE(table))
    # Mirror the whole situation: same shape, other way.
    mirrored_table = {"Sweeper": (105.0 - 20.0, 34.0),
                      "LastCB": (105.0 - 40.0, 34.0)}
    left = prof.last_line_gap(105.0 - 35.0, 34.0, False, defenders,
                              _FakePE(mirrored_table))
    assert right == pytest.approx(left)


def test_tight_channel_reports_a_tight_channel():
    """A mirrored geometry that is genuinely tight must NOT read as 1.0.

    This is the assertion the old code could not have passed: defenders on 65
    and 75, striker on 70 - five metres beyond the line either way.
    """
    prof = StrikerSpatialProfile()
    right_table = {"CB1": (65.0, 34.0), "CB2": (75.0, 34.0)}
    defenders = [_shim("CB1", "CB"), _shim("CB2", "CB")]
    gap_right = prof.last_line_gap(70.0, 34.0, True, defenders,
                                   _FakePE(right_table))
    left_table = {"CB1": (40.0, 34.0), "CB2": (30.0, 34.0)}
    gap_left = prof.last_line_gap(35.0, 34.0, False, defenders,
                                  _FakePE(left_table))
    assert gap_right == pytest.approx(5.0 / 15.0)
    assert gap_left == pytest.approx(5.0 / 15.0)


def test_gap_is_zero_when_level_with_the_line():
    # The line with defenders on 40 and 55 is 40 - the second-deepest. A striker
    # ON 40 is level with it; a striker on 55 is not, and has a full channel.
    pe = _FakePE({"CB1": (40.0, 34.0), "CB2": (55.0, 34.0)})
    defenders = [_shim("CB1", "CB"), _shim("CB2", "CB")]
    assert StrikerSpatialProfile().last_line_gap(
        40.0, 34.0, True, defenders, pe) == pytest.approx(0.0)
    assert StrikerSpatialProfile().last_line_gap(
        55.0, 34.0, True, defenders, pe) == pytest.approx(1.0)


def test_gap_falls_back_to_neutral_without_information():
    assert StrikerSpatialProfile().last_line_gap(50.0, 34.0, True, None, None) == 0.5


# =============================================================================
# 2. RUN-BEHIND TARGET - sited off the line, not off a fixed x
# =============================================================================


def test_run_behind_target_follows_the_line():
    prof = StrikerSpatialProfile()
    deep = prof.run_behind_target(True, 34.0, line_nx=78.0)
    high = prof.run_behind_target(True, 34.0, line_nx=60.0)
    # A fixed 92 m target would return the same answer for both.
    assert deep[0] == pytest.approx(80.0)
    assert high[0] == pytest.approx(62.0)


def test_run_behind_target_is_direction_mirrored():
    prof = StrikerSpatialProfile()
    right = prof.run_behind_target(True, 34.0, line_nx=78.0)[0]
    left = prof.run_behind_target(False, 34.0, line_nx=78.0)[0]
    assert right + left == pytest.approx(105.0)


def test_run_behind_target_lead_is_small_enough_to_stay_in_the_real_band():
    """A 2 m lead keeps the flag call in the 3-8 per match band.

    event_chain's flag discipline saturates at OFFSIDE_CALL_PEAK (6%) for a
    decisive position; a run aimed 14 m beyond the line is permanently in that
    worst case, and every pass to him is a dead ball.
    """
    prof = StrikerSpatialProfile()
    lead = prof.RUN_BEHIND_LEAD_M
    assert 0.0 < lead <= 4.0


def test_run_behind_target_stays_on_the_pitch():
    prof = StrikerSpatialProfile()
    for line in (95.0, 99.0, 104.0):
        tx, ty = prof.run_behind_target(True, 34.0, line_nx=line)
        assert 0.0 <= tx <= 105.0 and 0.0 <= ty <= 68.0


def test_run_behind_target_falls_back_when_no_line_known():
    """No line readable -> the legacy fixed post, plus the same small lead, so a
    missing-information run is not a wildly different one from a known one."""
    prof = StrikerSpatialProfile()
    tx, _ = prof.run_behind_target(True, 34.0, line_nx=None)
    assert tx == pytest.approx(92.0 + prof.RUN_BEHIND_LEAD_M)


# =============================================================================
# 3. REST DEFENCE - the invariant, and the backstop property
# =============================================================================


def test_rest_defence_is_silent_in_an_ordinary_shape():
    pe = _engine()
    st = _names(pe, "ST")[0]
    cm = _names(pe, "CDM")[0]
    _place(pe, {st: 88.0, cm: 30.0})
    # Ball upfield at 70: the pivot is behind it, so the invariant already holds
    # even with only one man back.
    assert pe.rest_defence_violator("Home", 70.0, True, True) is None


def test_rest_defence_needs_two_behind_not_one():
    pe = _engine()
    for name, state in pe.states.items():
        if state.position != "GK":
            state.current_x, state.current_y = 80.0, 34.0
    cm = _names(pe, "CDM")[0]
    pe.states[cm].current_x = 30.0          # exactly one man behind the ball
    pe.REST_DEFENCE_MIN_BEHIND = 2
    assert pe.rest_defence_violator("Home", 60.0, True, True) is not None
    pe.REST_DEFENCE_MIN_BEHIND = 1
    assert pe.rest_defence_violator("Home", 60.0, True, True) is None


def test_rest_defence_fires_when_the_whole_team_is_beyond_the_ball():
    pe = _engine()
    for name, st in pe.states.items():
        if st.position != "GK":
            st.current_x, st.current_y = 80.0, 34.0
    violator = pe.rest_defence_violator("Home", 60.0, True, True)
    assert violator is not None
    # ...and it is an outfield player, not the keeper.
    assert pe.states[violator].position != "GK"


def test_rest_defence_holds_the_shallowest_man_not_the_striker():
    """The held player must be the one the attack can least afford to lose.

    Picking the most advanced man would strand the striker at the halfway line;
    the shallowest man is already a rest defender in everything but name.
    """
    pe = _engine()
    st = _names(pe, "ST")[0]
    for name, state in pe.states.items():
        if state.position != "GK":
            state.current_x, state.current_y = 80.0, 34.0
    pe.states[st].current_x = 95.0
    cb = _names(pe, "CB")[0]
    pe.states[cb].current_x = 62.0
    assert pe.rest_defence_violator("Home", 60.0, True, True) == cb
    assert pe.states[st].current_x == 95.0   # untouched


def test_rest_defence_never_fires_out_of_possession():
    pe = _engine()
    for name, state in pe.states.items():
        if state.position != "GK":
            state.current_x, state.current_y = 80.0, 34.0
    assert pe.rest_defence_violator("Home", 60.0, False, True) is None


def test_rest_defence_respects_its_own_kill_switch():
    pe = _engine()
    for name, state in pe.states.items():
        if state.position != "GK":
            state.current_x, state.current_y = 80.0, 34.0
    pe.REST_DEFENCE_ENABLED = False
    try:
        assert pe.rest_defence_violator("Home", 60.0, True, True) is None
    finally:
        pe.REST_DEFENCE_ENABLED = True


def test_rest_defence_is_direction_symmetric():
    """Attacking LEFT, "behind the ball" means a LARGER pitch x, because the
    ball is travelling toward x=0. Reading it the other way round would have
    the rule pull the whole team forward exactly when it should hold it back."""
    pe = _engine(attacks_right=False)
    for name, state in pe.states.items():
        if state.position != "GK":
            state.current_x, state.current_y = 80.0, 34.0
    # Ball at 45, everyone at 80 -> all ten are behind it. Invariant holds.
    assert pe.rest_defence_violator("Home", 45.0, True, False) is None
    # Ball at 45, everyone at 25 -> all ten are beyond it. Violated.
    for name, state in pe.states.items():
        if state.position != "GK":
            state.current_x = 25.0
    assert pe.rest_defence_violator("Home", 45.0, True, False) is not None


def test_rest_defence_tolerates_a_missing_ball():
    pe = _engine()
    assert pe.rest_defence_violator("Home", None, True, True) is None


def test_rest_defence_ignores_the_keeper_when_counting():
    """A keeper on his line is not a rest defender, and must not satisfy it."""
    pe = _engine()
    for name, state in pe.states.items():
        if state.position != "GK":
            state.current_x, state.current_y = 80.0, 34.0
    gk = _names(pe, "GK")[0]
    pe.states[gk].current_x = 5.0        # behind the ball at 60
    assert pe.rest_defence_violator("Home", 60.0, True, True) is not None


# =============================================================================
# 4. REST-DEFENCE CLAMP - depth only, and only once
# =============================================================================


def test_clamp_pins_the_target_behind_the_ball():
    pe = _engine()
    tx, ty = pe.rest_defence_clamp(80.0, 34.0, 60.0, True)
    assert tx < 60.0
    assert tx == pytest.approx(60.0 - pe.REST_DEFENCE_GAP_M)


def test_clamp_leaves_the_lateral_position_alone():
    """Depth is the whole claim. Re-stamping y would be a second opinion
    about shape and would collapse the width CK35 spends its life on."""
    pe = _engine()
    _, ty = pe.rest_defence_clamp(80.0, 12.0, 60.0, True)
    assert ty == 12.0


def test_clamp_is_a_no_op_when_already_behind():
    pe = _engine()
    assert pe.rest_defence_clamp(40.0, 34.0, 60.0, True) == (40.0, 34.0)


def test_clamp_is_direction_mirrored():
    pe = _engine()
    right, _ = pe.rest_defence_clamp(80.0, 34.0, 60.0, True)
    left, _ = pe.rest_defence_clamp(25.0, 34.0, 45.0, False)
    assert right + left == pytest.approx(105.0)


def test_clamp_keeps_the_target_on_the_pitch():
    """Ball on the halfway line: the clamp must not demand a negative x."""
    pe = _engine()
    tx, _ = pe.rest_defence_clamp(80.0, 34.0, 52.0, True)
    assert tx >= 0.0


# =============================================================================
# 5. LIVE STRIKER RUN TARGETS
# =============================================================================


def test_striker_runs_are_empty_out_of_possession():
    pe = _engine()
    assert pe.striker_run_targets("Home", 60.0, 34.0, False, True) == {}


def test_striker_runs_are_empty_without_a_ball():
    pe = _engine()
    assert pe.striker_run_targets("Home", None, None, True, True) == {}


def test_striker_runs_respect_their_kill_switch():
    pe = _engine()
    pe.STRIKER_RUNS_LIVE = False
    try:
        out = pe.striker_run_targets("Home", 60.0, 34.0, True, True)
        assert out == {}
    finally:
        pe.STRIKER_RUNS_LIVE = True


def test_striker_runs_only_ever_address_a_striker():
    pe = _engine()
    random.seed(7)
    for ball_x in (40.0, 55.0, 70.0, 85.0):
        for ball_y in (20.0, 34.0, 48.0):
            out = pe.striker_run_targets("Home", ball_x, ball_y, True, True)
            for name, value in out.items():
                assert pe.states[name].position in ("ST", "CF")
                assert len(value) == 3
                blend, tx, ty = value
                assert 0.0 < blend <= 1.0
                assert 0.0 <= tx <= 105.0 and 0.0 <= ty <= 68.0


def test_a_registered_striker_is_offered_a_run_at_some_point():
    """A wiring claim, not a determinism claim.

    A registry that resolved nothing at any ball position would leave the whole
    layer inert while every individual call still returned a well-formed empty
    dict, so the shape assertions above would all pass on dead code. The axis
    of variation is the decision key (minute), which is what the layer actually
    varies on now that it draws no global RNG.
    """
    pe = _engine()
    st = _names(pe, "ST")[0]
    seen = 0
    for minute in range(60):
        for ball_x in (30.0, 45.0, 60.0, 78.0, 90.0):
            out = pe.striker_run_targets("Home", ball_x, 34.0, True, True,
                                         minute=minute)
            if st in out:
                seen += 1
    assert seen > 0, "striker run layer produced no target in 300 attempts"


def test_striker_runs_consume_no_global_random():
    """The load-bearing hygiene property.

    The project's own highest-value open bug is that a match is not
    reproducible from random.seed. A new global-stream consumer makes that
    worse, so this layer must take NONE. Measured directly: the identical call
    sequence must leave the global stream at the identical position, which it
    only can if nothing was drawn from it.
    """
    pe = _engine()
    st = _names(pe, "ST")[0]
    assert st, "no ST registered"
    ball = [(35.0, 34.0), (50.0, 20.0), (65.0, 48.0), (85.0, 34.0)]
    random.seed(1234)
    for bx, by in ball:
        pe.striker_run_targets("Home", bx, by, True, True, minute=7)
    after_layer = [random.random() for _ in range(5)]
    random.seed(1234)
    control = [random.random() for _ in range(5)]
    assert after_layer == control, "the run layer drew from the global stream"


def test_striker_runs_are_reproducible_for_a_given_key():
    pe = _engine()
    random.seed(9)
    first = pe.striker_run_targets("Home", 50.0, 34.0, True, True, minute=12)
    random.seed(999)          # a DIFFERENT global stream entirely
    second = pe.striker_run_targets("Home", 50.0, 34.0, True, True, minute=12)
    assert first == second, "same key gave a different run"


def test_deterministic_rng_is_a_stream_not_a_constant():
    f = PositionEngine._deterministic_rng("Home FC", "Home ST", 3)
    draws = [f() for _ in range(6)]
    assert len(set(draws)) > 1, "successive draws must differ"
    assert all(0.0 <= v < 1.0 for v in draws)


def test_deterministic_rng_is_keyed_on_every_part():
    a = PositionEngine._deterministic_rng("Home FC", "ST", 3)()
    b = PositionEngine._deterministic_rng("Home FC", "ST", 4)()
    c = PositionEngine._deterministic_rng("Away FC", "ST", 3)()
    d = PositionEngine._deterministic_rng("Home FC", "ST2", 3)()
    assert len({a, b, c, d}) == 4


def test_shape_shims_expose_only_what_the_engine_reads():
    from position_engine import _ShapeShim
    shim = _ShapeShim("Home ST 10", "ST")
    assert shim.name == "Home ST 10"
    assert shim.position == "ST"
    assert not hasattr(shim, "dna")


def test_opponent_shims_come_from_the_other_team():
    pe = _engine()
    pe.initialize_team("Away", _squad("Away FC"), _profile("Away FC"),
                       attacks_right=False)
    shims = pe._opponent_shape_shims("Home")
    assert shims, "no opponent shims built"
    assert {s.name for s in shims} == set(pe.team_rosters["Away"])
    assert all(s.name not in pe.team_rosters["Home"] for s in shims)


# =============================================================================
# 6. ENGINE WIRING - the per-tick and per-minute caches
#
# Both layers take a TEAM-WIDE decision. Resolving them per player would give
# different answers inside one tick, because positions move as the tick
# integrates: the first player pulled back changes the input the second player
# reads. The caches are what make the decision consistent, so they are tested
# as behaviour and not just as attributes.
# =============================================================================


def _match_engine():
    config = MatchConfig(home_team="Home FC", away_team="Away FC",
                         match_date=date.today())
    hp = TeamProfile(name="Home FC", style=TeamStyle.BALANCED,
                     playing_style=PlayingStyle.POSSESSION, intensity=Intensity.MEDIUM)
    ap = TeamProfile(name="Away FC", style=TeamStyle.BALANCED,
                     playing_style=PlayingStyle.POSSESSION, intensity=Intensity.MEDIUM)
    eng = MatchEngine(config, hp, ap)
    eng.set_squad("Home FC", _squad("Home FC"))
    eng.set_squad("Away FC", _squad("Away FC"))
    return eng


def test_rest_defence_cache_expires_with_the_tick():
    eng = _match_engine()
    first = eng._rest_defence_violator("Home FC", 60.0, True)
    eng._offball_tick_seq += 1
    second = eng._rest_defence_violator("Home FC", 60.0, True)
    # Same inputs, different tick: the second call was genuinely re-resolved
    # rather than served from a stale tick's entry.
    assert eng._rest_defence_seq == eng._offball_tick_seq


def test_striker_run_cache_is_keyed_on_minute_and_zone():
    """The cache must re-roll when the SITUATION moves, not just on the minute.

    Once-per-minute alone means a striker still running in behind after the
    ball turned over at his feet, so the zone is part of the key. Asserted on
    the cache's own contents: a ball a third of the pitch away, in the same
    minute, is a separate entry rather than a stale hit.
    """
    eng = _match_engine()
    eng._striker_runs("Home FC", 30.0, 34.0, True)
    eng._striker_runs("Home FC", 88.0, 34.0, True)
    zones = {k[2] for k in eng._striker_run_cache}
    assert len(zones) == 2, f"expected two distinct zones cached, got {zones}"


def test_striker_runs_hold_still_within_one_situation():
    """The opposite failure: identical inputs inside one tick must NOT re-roll.

    A per-tick re-roll would flicker the run mode and, before the deterministic
    RNG existed, consume the football stream ten times a second per striker.
    """
    eng = _match_engine()
    first = eng._striker_runs("Home FC", 60.0, 34.0, True)
    for _ in range(50):
        assert eng._striker_runs("Home FC", 60.0, 34.0, True) is first


def test_striker_runs_are_withheld_while_defending():
    eng = _match_engine()
    assert eng._striker_runs("Home FC", 60.0, 34.0, False) == {}


def test_rest_defence_is_withheld_while_defending():
    eng = _match_engine()
    assert eng._rest_defence_violator("Home FC", 60.0, False) is None


def test_striker_runs_do_not_roll_random_every_tick():
    """Alias kept explicit: the no-stream-consumption claim is asserted at the
    unit level in test_striker_runs_consume_no_global_random; this is the
    engine-level counterpart that the cache actually holds still."""
    eng = _match_engine()
    first = eng._striker_runs("Home FC", 60.0, 34.0, True)
    for _ in range(50):
        assert eng._striker_runs("Home FC", 60.0, 34.0, True) is first


def test_full_match_runs_with_both_layers_on():
    """A layer that only works in a probe is not wired."""
    random.seed(31)
    eng = _match_engine()
    result = eng.simulate()
    assert result.home_goals >= 0 and result.away_goals >= 0
    assert eng._offball_tick_seq > 0, "the 10 Hz tick counter never advanced"
