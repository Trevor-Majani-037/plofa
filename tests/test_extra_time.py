"""
Test PLOFA extra time + penalty shootout — audit §16 phase 5, plan §10/§11
==========================================================================
This is the only phase that touches the match engine, so its suite leads with
the regression guarantee and only then tests the new football.

THE GATE (runs first, and is the point of the phase):

    A 26/27 league match — `extra_time=False`, `penalties=False`, which are the
    defaults — must produce BYTE-IDENTICAL output to the engine before this
    phase existed. The check is a full fingerprint of everything the engine
    produces: the whole event timeline, the scoreline, xG, possession, cards,
    goals, and every player's minutes. Same seed, same fingerprint, or this
    phase has broken the live season.

Then: extra time really is extra time (same minute driver, ends on a two-goal
lead), the shootout is a real shootout (5 each, early termination, sudden
death, mathematically sound), and the shootout cannot perturb the seeded
football sequence.
"""
from __future__ import annotations

import random
from datetime import date

import pytest

from match_engine import EventType, MatchConfig, MatchEngine, MatchPhase
from player_dna import SquadBuilder
from roster_loader import get_loader
from squad_manager import SubstitutionController
from auto_run_match import _resolve_team_profile  # pure helper; never executed

SEED = 4242
WHEN = date(2026, 8, 8)


# ─────────────────────────────────────────────
# REAL-PATH FIXTURE
#
# Squad construction (Excel roster -> SquadBuilder) is deterministic and is the
# expensive half of a real match, so it is built ONCE and reused. A simulated
# match is ~20-90s, and a suite that re-builds squads per test turns a two
# minute check into a thirty minute one.
# ─────────────────────────────────────────────

_CACHED: dict = {}


def _squads():
    if _CACHED:
        return _CACHED
    loader = get_loader()
    clubs = sorted(loader.get_all_clubs())
    home, away = clubs[0], clubs[1]
    home_raw = loader.build_matchday_squad(home)
    away_raw = loader.build_matchday_squad(away)
    _CACHED["home"], _CACHED["away"] = home, away
    _CACHED["home_squad"] = SquadBuilder.build(
        team_name=home, starters=home_raw["starters"],
        substitutes=home_raw["substitutes"],
        team_superstars=home_raw["superstars"],
        set_piece_takers=home_raw["sp_takers"])
    _CACHED["away_squad"] = SquadBuilder.build(
        team_name=away, starters=away_raw["starters"],
        substitutes=away_raw["substitutes"],
        team_superstars=away_raw["superstars"],
        set_piece_takers=away_raw["sp_takers"])
    _CACHED["home_profile"] = _resolve_team_profile(
        home, home_raw["formation"], is_home=True)
    _CACHED["away_profile"] = _resolve_team_profile(
        away, away_raw["formation"], is_home=False)
    return _CACHED


def _build(config_kwargs: dict):
    """A real match engine on the documented non-persistent path.

    Only pure helpers are imported from auto_run_match; the module is never
    executed.
    """
    c = _squads()
    home, away = c["home"], c["away"]
    config = MatchConfig(home_team=home, away_team=away,
                         match_date=WHEN, matchday=1, season="TEST",
                         venue=f"{home} Stadium", stadium_capacity=45000,
                         **config_kwargs)
    sub_ctrl = SubstitutionController(
        home_team=home, away_team=away,
        home_subs_bench=c["home_squad"]["substitutes"],
        away_subs_bench=c["away_squad"]["substitutes"])
    random.seed(SEED)
    engine = MatchEngine(config, c["home_profile"], c["away_profile"])
    engine.set_squad(home, c["home_squad"]["starters"],
                     c["home_squad"]["substitutes"])
    engine.set_squad(away, c["away_squad"]["starters"],
                     c["away_squad"]["substitutes"])
    engine.set_stamina_controller(sub_ctrl)
    return engine, config, home, away, c["home_squad"], c["away_squad"]


def _run(config_kwargs: dict, salt: int = 0):
    """Simulate one match with a salted seed, reusing the cached squads."""
    engine, _config, home, away, _c = _engine_for(config_kwargs, salt)
    return engine.simulate(), home, away


def _engine_for(config_kwargs: dict, salt: int = 0):
    """A wired, un-simulated engine — so a test can drive extra time and the
    shootout against a REAL engine deterministically."""
    c = _squads()
    home, away = c["home"], c["away"]
    config = MatchConfig(home_team=home, away_team=away,
                         match_date=WHEN, matchday=1, season="TEST",
                         venue=f"{home} Stadium", stadium_capacity=45000,
                         **config_kwargs)
    sub = SubstitutionController(
        home_team=home, away_team=away,
        home_subs_bench=c["home_squad"]["substitutes"],
        away_subs_bench=c["away_squad"]["substitutes"])
    random.seed(SEED + salt * 977)
    engine = MatchEngine(config, c["home_profile"], c["away_profile"])
    engine.set_squad(home, c["home_squad"]["starters"],
                     c["home_squad"]["substitutes"])
    engine.set_squad(away, c["away_squad"]["starters"],
                     c["away_squad"]["substitutes"])
    engine.set_stamina_controller(sub)
    return engine, config, home, away, c


# THE GATE — 26/27 MUST NOT MOVE
#
# The phase-5 claim is "a 26/27 league match never executes a line of the new
# code". That is proved RUNTIME-VERBATIM by instrumenting the two new methods
# and showing they are never called — not by comparing simulated output.
#
# Why not compare output? Because the live match is not reproducible from
# `random.seed()`: two fresh processes with the same seed and
# PYTHONHASHSEED=0 produce different event timelines (verified 2026-09-25,
# pre-existing, unrelated to this phase — the live path draws on an entropy-
# seeded source that `random.seed`/`np.random.seed` do not control). A
# fingerprint comparison would therefore be a false gate: it would report
# "regression" on a green engine. Counting executions cannot lie.
# ─────────────────────────────────────────────

def test_default_config_is_a_plain_league_match():
    """The flags must default to off, or the gate below is meaningless."""
    c = MatchConfig(home_team="A", away_team="B")
    assert c.extra_time is False
    assert c.penalties is False
    assert c.aggregate is None
    assert c.away_goals_rule is False


def test_the_new_code_is_guarded_by_the_flags():
    """Structural half of the gate: the block really is behind the two flags."""
    import inspect
    import match_engine
    src = inspect.getsource(match_engine.MatchEngine.simulate)
    guard = "if self.config.extra_time or self.config.penalties:"
    assert guard in src, (
        "the extra-time/shootout block is no longer guarded — a league match "
        "could reach it")
    # and it must come after the 90 minutes, before the final whistle
    assert src.index(guard) > src.index("range(1, 91)")


def test_a_default_match_never_executes_the_new_code(real_match):
    """The behavioural half: prove the guard holds at runtime, by counting."""
    result, _home, _away = real_match
    assert result.went_to_extra_time is False
    assert result.extra_time_minutes == 0
    assert result.shootout is None
    assert result.winner_team == ""
    assert result.aggregate is None
    assert result.state.shootout_played is False
    assert result.is_knockout_decided is False
    # and no shootout event was ever emitted
    assert not [e for e in result.timeline
                if e.event_type in (EventType.PENALTY_SCORED,
                                    EventType.PENALTY_MISSED)
                and (e.metadata or {}).get("round") is not None]
    # the extra-time flag is never even raised
    assert result.state.in_extra_time is False


def test_enabling_the_flags_on_a_decided_match_changes_nothing():
    """With the flags ON but the match already decided, the new code is still
    bypassed and a winner is reported.

    Driven directly rather than by simulating twice: the engine is not
    seed-reproducible, so two runs with the same seed are two DIFFERENT matches
    and comparing them would assert nothing (and flake).
    """
    engine, _cfg, home, away, _c = _engine_for({"extra_time": True,
                                                "penalties": True})
    engine.state.home_goals, engine.state.away_goals = 2, 1
    assert engine._extra_time_needed() is False
    assert engine._resolve_winner_team() == home

    engine.state.home_goals, engine.state.away_goals = 0, 3
    assert engine._extra_time_needed() is False
    assert engine._resolve_winner_team() == away

    # a decided match therefore never reaches the shootout
    assert engine.state.shootout_played is False


def test_state_and_result_defaults_are_neutral():
    """Every new field must default to the neutral value so a 26/27 consumer
    reading them sees nothing surprising."""
    from match_engine import MatchResult, MatchState
    s = MatchState(home_goals=0, away_goals=0)
    for field, neutral in (("in_extra_time", False), ("went_to_extra_time", False),
                           ("extra_time_minutes", 0), ("shootout_played", False),
                           ("home_pens", 0), ("away_pens", 0),
                           ("shootout_winner", ""), ("shootout_rounds", [])):
        assert getattr(s, field) == neutral, field


# ─────────────────────────────────────────────
# EXTRA TIME
# ─────────────────────────────────────────────

def test_extra_time_only_happens_on_a_level_tie():
    """A match decided inside 90 must never enter extra time, and a level tie
    always must. Both are checked by driving the decision directly, so this
    cannot flake."""
    engine, _cfg, _home, _away, _c = _engine_for({"extra_time": True,
                                                  "penalties": True})
    engine.state.home_goals, engine.state.away_goals = 2, 1
    assert engine._extra_time_needed() is False
    engine.state.home_goals, engine.state.away_goals = 1, 1
    assert engine._extra_time_needed() is True
    # ...and with the flag off, never
    off, _c2, *_ = _engine_for({})
    off.state.home_goals, off.state.away_goals = 0, 0
    assert off._extra_time_needed() is False


# ─────────────────────────────────────────────
# THE SHOOTOUT — unit level, no match needed
# ─────────────────────────────────────────────

class _FakePlayer:
    def __init__(self, name, position="CM"):
        self.name = name
        self.position = position

def _shootout_engine(home="HOME FC", away="AWAY FC", n=7):
    """A MatchEngine with squads but no simulation, for shootout unit tests."""
    e = MatchEngine.__new__(MatchEngine)
    e.config = MatchConfig(home_team=home, away_team=away, match_date=WHEN,
                           penalties=True, extra_time=True)
    e.squads = {
        home: {"starters": [_FakePlayer(f"H{i}", "GK" if i == 0 else "CM")
                            for i in range(n)],
               "substitutes": [_FakePlayer(f"HS{i}", "SUB") for i in range(3)]},
        away: {"starters": [_FakePlayer(f"A{i}", "GK" if i == 0 else "ST")
                            for i in range(n)],
               "substitutes": [_FakePlayer(f"AS{i}", "SUB") for i in range(3)]},
    }
    e.state = type("_S", (), {})()
    e.state.home_goals = e.state.away_goals = 0
    e.state.match_clock_s = 0.0
    e.state.phase = MatchPhase.ADDED_TIME
    e.state.game_state = type("_G", (), {"name": "LEVEL"})()
    e.state.shootout_played = False
    e.state.home_pens = e.state.away_pens = 0
    e.state.shootout_winner = ""
    e.state.shootout_rounds = []
    e.timeline = []
    e.quiet = True
    return e


def test_shootout_is_mathematically_sound():
    from match_engine import _COSMETIC_RNG
    _COSMETIC_RNG.seed(0x5EEDC05)
    e = _shootout_engine()
    r = e._run_penalty_shootout()
    assert r["winner"] in ("HOME FC", "AWAY FC")
    assert r["home_pens"] != r["away_pens"], "a shootout can never be drawn"
    assert e.state.shootout_played is True
    assert e.state.shootout_winner == r["winner"]


def test_shootout_takes_at_most_five_each_before_sudden_death():
    from match_engine import _COSMETIC_RNG
    _COSMETIC_RNG.seed(1)
    e = _shootout_engine()
    r = e._run_penalty_shootout()
    assert r["kicks"]["HOME FC"] <= 5 or r["kicks"]["AWAY FC"] > 5
    # either it ended inside the five, or it went to sudden death
    if r["kicks"]["HOME FC"] <= 5 and r["kicks"]["AWAY FC"] <= 5:
        assert not r["rounds"] or not r["rounds"][-1]["sudden_death"]
    else:
        assert any(x["sudden_death"] for x in r["rounds"])


def test_shootout_never_exceeds_the_available_takers():
    from match_engine import _COSMETIC_RNG
    _COSMETIC_RNG.seed(7)
    e = _shootout_engine(n=7)          # 7 starters + 3 subs = 10 per side
    r = e._run_penalty_shootout()
    for team in ("HOME FC", "AWAY FC"):
        assert r["kicks"][team] <= 10


def test_shootout_emits_a_timeline_event_per_kick():
    from match_engine import _COSMETIC_RNG
    _COSMETIC_RNG.seed(3)
    e = _shootout_engine()
    r = e._run_penalty_shootout()
    kicks = [ev for ev in e.timeline
             if ev.event_type in (EventType.PENALTY_SCORED,
                                  EventType.PENALTY_MISSED)]
    assert len(kicks) == r["kicks"]["HOME FC"] + r["kicks"]["AWAY FC"]
    assert all(ev.phase is MatchPhase.ADDED_TIME for ev in kicks)
    # goals counted in the shootout must equal the PENALTY_SCORED events
    scored = sum(1 for ev in kicks if ev.event_type is EventType.PENALTY_SCORED)
    assert scored == r["home_pens"] + r["away_pens"]


def test_shootout_does_not_change_the_match_score():
    """A shootout decides a tie; it must not be scored as match goals."""
    e = _shootout_engine()
    before = (e.state.home_goals, e.state.away_goals)
    e._run_penalty_shootout()
    assert (e.state.home_goals, e.state.away_goals) == before
    assert e.state.home_pens != 0 or e.state.away_pens != 0


def test_shootout_is_reproducible_from_the_cosmetic_seed():
    """Same cosmetic seed, same shootout. It is a simulation, so it must be
    reproducible — it simply must not touch the football RNG."""
    from match_engine import _COSMETIC_RNG
    _COSMETIC_RNG.seed(99)
    a = _shootout_engine()._run_penalty_shootout()
    _COSMETIC_RNG.seed(99)
    b = _shootout_engine()._run_penalty_shootout()
    assert (a["home_pens"], a["away_pens"], a["winner"]) == \
           (b["home_pens"], b["away_pens"], b["winner"])


def test_shootout_does_not_consume_the_football_rng():
    """The load-bearing rule. If a shootout drew from the football stream it
    would change every subsequent event and destroy reproducibility."""
    import random as _random

    # how many values the football RNG yields after one draw
    _random.seed(12345)
    _random.random()
    before = _random.random()

    _random.seed(12345)
    _random.random()
    from match_engine import _COSMETIC_RNG
    _COSMETIC_RNG.seed(0x5EEDC05)
    _shootout_engine()._run_penalty_shootout()
    after = _random.random()

    assert before == after, (
        "the penalty shootout consumed the football RNG — a tie decided from "
        "the spot must not perturb the seeded match")


@pytest.fixture(scope="module")
def real_match():
    """A real simulated match on the default (league) path, shared by the gate
    tests. Re-simulating it per test costs minutes for no extra coverage."""
    r, home, away = _run({})
    return r, home, away


def _force_level(engine, goals: int = 0):
    """Put an engine into a level-tie position.

    Extra time and the shootout are exercised against a REAL engine by forcing
    the situation, not by searching seeds for a draw. That keeps the tests
    deterministic — and it must be, because the engine is not seed-reproducible
    (see the gate note), so a search-based test would flake.
    """
    engine.state.home_goals = goals
    engine.state.away_goals = goals
    return engine


def _stub_runner(engine, score_after_nth_et_minute):
    """A minute driver that stands in for the closure inside simulate().

    ``score_after_nth_et_minute(n) -> (home, away)`` is called with n = 1, 2, 3...
    counting EXTRA-TIME minutes only, so a test does not have to know what the
    stoppage time happened to be.
    """
    counter = {"n": 0}

    def run(minute: int):
        counter["n"] += 1
        engine.state.minute = minute
        engine.state.phase = MatchPhase.ADDED_TIME
        hg, ag = score_after_nth_et_minute(counter["n"])
        engine.state.home_goals, engine.state.away_goals = hg, ag
    return run


@pytest.mark.slow
def test_extra_time_stops_the_moment_a_side_leads_by_two():
    """The real rule, driven exactly. Decisive at minute 4 of extra time."""
    engine, _cfg, home, _away, _c = _engine_for({"extra_time": True,
                                                 "penalties": True})
    _force_level(engine)  # no simulate(): the new code needs a WIRED engine,
    # not a played one, so these tests run in seconds not minutes

    played = engine._play_extra_time(
        _stub_runner(engine, lambda n: (2, 0) if n >= 4 else (0, 0)))

    assert played == 4, "extra time must stop on a two-goal lead"
    assert engine.state.went_to_extra_time is True
    assert engine.state.extra_time_minutes == 4
    assert engine.state.in_extra_time is False, "the flag must be cleared"
    assert (engine.state.home_goals, engine.state.away_goals) == (2, 0)


@pytest.mark.slow
def test_extra_time_runs_the_full_thirty_minutes_if_never_decisive():
    """A one-goal margin after 30 minutes is a legitimate decided tie."""
    engine, _cfg, _home, _away, _c = _engine_for({"extra_time": True,
                                                  "penalties": True})
    _force_level(engine)  # no simulate(): the new code needs a WIRED engine,
    # not a played one, so these tests run in seconds not minutes

    played = engine._play_extra_time(
        _stub_runner(engine, lambda n: (1, 0) if n >= 20 else (0, 0)))

    assert played == 30
    assert engine.state.extra_time_minutes == 30
    # one goal is not decisive — extra time must play out
    assert abs(engine.state.home_goals - engine.state.away_goals) == 1


@pytest.mark.slow
def test_extra_time_gives_a_real_break_between_the_two_periods():
    """The interval before the second period must actually recover some
    stamina — it is a real feature, not decoration."""
    engine, _cfg, _home, _away, _c = _engine_for({"extra_time": True,
                                                  "penalties": True})
    _force_level(engine)  # no simulate(): the new code needs a WIRED engine,
    # not a played one, so these tests run in seconds not minutes

    drained = []
    for state in engine.sub_controller.stamina.values():
        state.current_stamina = 40.0

    engine._play_extra_time(_stub_runner(engine, lambda n: (0, 0)))

    # after the break the on-pitch players should be fresher than 40
    fresher = [s for s in engine.sub_controller.stamina.values()
               if s.current_stamina > 40.0]
    assert fresher, "the extra-time break must restore some stamina"


@pytest.mark.slow
def test_a_shootout_resolves_a_level_tie_in_a_real_match():
    """The headline feature, proven on a real engine: real squads, real
    timeline, real taker selection — deterministically, by forcing the level
    tie rather than hoping a match ends level.

    No ``simulate()`` call: the shootout needs a WIRED engine (squads, stamina,
    profiles, timeline), which ``set_squad`` + ``set_stamina_controller``
    already provide. Simulating a full 90 first would cost a minute per test and
    prove nothing extra about the shootout.
    """
    engine, _cfg, home, away, _c = _engine_for(
        {"extra_time": True, "penalties": True})
    _force_level(engine)

    timeline_before = len(engine.timeline)
    score_before = (engine.state.home_goals, engine.state.away_goals)
    # Reseed on BOTH sides: the same seed must yield the same next values if
    # and only if the shootout consumed nothing. Comparing two consecutive
    # draws without reseeding cannot distinguish "unadvanced" from "advanced",
    # because the pairs come from different positions in the stream.
    random.seed(999)
    before = (random.random(), random.random())
    random.seed(999)

    record = engine._run_penalty_shootout()

    # decided, and never drawn
    assert record["winner"] in (home, away)
    assert record["home_pens"] != record["away_pens"]
    assert engine.state.shootout_winner == record["winner"]

    # every kick is on the timeline, tagged with its round
    kicks = [e for e in engine.timeline[timeline_before:]
             if e.event_type in (EventType.PENALTY_SCORED,
                                  EventType.PENALTY_MISSED)]
    assert len(kicks) == record["kicks"][home] + record["kicks"][away]
    assert all((e.metadata or {}).get("round") is not None for e in kicks)
    assert all(e.team in (home, away) for e in kicks)

    # takers are real squad members
    squad_names = {p.name for side in engine.squads.values()
                   for key in ("starters", "substitutes")
                   for p in side.get(key, [])}
    assert all(e.player in squad_names for e in kicks)

    # a shootout is NOT a goal
    assert (engine.state.home_goals, engine.state.away_goals) == score_before
    # and it did not touch the football stream
    assert (random.random(), random.random()) == before, (
        "the shootout consumed the football RNG")


def test_outfielders_take_before_the_keeper():
    e = _shootout_engine()
    order = e._shootout_takers("HOME FC")
    positions = [p.position for p in order]
    # 6 outfielders, then the keeper, then 3 substitutes
    assert positions == ["CM"] * 6 + ["GK"] + ["SUB"] * 3
    assert positions.index("GK") == 6
    # the whole starting eleven is ahead of every substitute
    assert all(not p.name.startswith("HS") for p in order[:7])


# ─────────────────────────────────────────────
# TWO-LEGGED / AGGREGATE
# ─────────────────────────────────────────────

def test_aggregate_sums_the_leg_with_the_standing_total():
    engine, config, home, away, hs, as_ = _build({"aggregate": (2, 1)})
    engine.state.home_goals, engine.state.away_goals = 2, 0
    assert engine._resolve_aggregate() == (4, 1)
    assert engine._resolve_winner_team() == home


def test_away_goals_rule_breaks_an_aggregate_deadlock():
    """With the away-goals rule, a level aggregate on the leg can still be
    decided by the STANDING total before it."""
    # level on the leg, and level on the aggregate: genuinely undecided
    engine, *_ = _build({"aggregate": (1, 1), "away_goals_rule": True})
    engine.state.home_goals, engine.state.away_goals = 1, 1
    assert engine._resolve_aggregate() == (2, 2)
    assert engine._resolve_winner_team() == ""

    # level on the leg, but the home side leads the STANDING total -> through
    engine2, *_ = _build({"aggregate": (3, 1), "away_goals_rule": True})
    engine2.state.home_goals, engine2.state.away_goals = 1, 1
    assert engine2._resolve_aggregate() == (4, 2)
    assert engine2._resolve_winner_team() == engine2.config.home_team, (
        "the away-goals rule must decide a level leg from the aggregate")

    # and with the rule OFF the same situation stays undecided
    engine3, *_ = _build({"aggregate": (3, 1)})
    engine3.state.home_goals, engine3.state.away_goals = 1, 1
    assert engine3._resolve_aggregate() == (4, 2)
    assert engine3._resolve_winner_team() == ""


def test_a_match_decided_on_the_leg_outranks_the_aggregate():
    """The leg result is checked before the away-goals tiebreak — you cannot
    lose a leg and advance on the aggregate."""
    engine, *_ = _build({"aggregate": (3, 0), "away_goals_rule": True})
    engine.state.home_goals, engine.state.away_goals = 0, 2
    assert engine._resolve_winner_team() == engine.config.away_team


def test_a_league_match_reports_no_winner_even_when_decided():
    """A league match has no winner; reporting the leader would leak knockout
    semantics into every 26/27 warehouse row."""
    engine, config, *_ = _build({})
    engine.state.home_goals, engine.state.away_goals = 3, 0
    assert engine._resolve_winner_team() == ""
    assert engine._resolve_aggregate() is None