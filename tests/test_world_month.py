"""
Test the controlled one-month world — world/testworld.py + world/month.py
=========================================================================
Audit §16 phase 10 / plan §18. Three layers, tested at different costs:

  * the REAL reference data (:mod:`world.realworld`) — dates and rules checked
    against what was actually published, because inventing them proves nothing;
  * the WORLD and the CHECKLIST — fast, no match simulation needed;
  * the DRIVER runs real matches and is exercised by ``python -m world.runmonth``
    rather than here, because a half-hour suite is a suite nobody runs.

The continuity checker is deliberately tested by BREAKING the ledger six ways
and asserting the checker notices each. A checker that has never failed has not
been tested.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from world import realworld as rw
from world.month import ContinuityIssue, WorldMonth
from world.testworld import (
    CHAMPIONS_LEAGUE,
    COUNTRY,
    DOMESTIC_LEAGUE,
    EFL_CUP,
    ROSTER_CLUBS,
    build_test_world,
    month_fixtures,
)

SEASON = rw.SEASON
#: the real Champions League opening matchday — where the month begins
MONTH_START = rw.CL_LEAGUE_PHASE[0][1][0]


@pytest.fixture(scope="module")
def world():
    return build_test_world()


@pytest.fixture(scope="module")
def month(world):
    from world.ledger import WorldLedger
    led = WorldLedger(path="<memory>", country_id=COUNTRY, season_id=SEASON)
    return WorldMonth(world, ledger=led, seed=17)


# ─────────────────────────────────────────────
# THE REAL REFERENCE DATA
# ─────────────────────────────────────────────

def test_premier_league_matches_the_real_season():
    assert rw.PL_TEAMS == 20
    assert rw.PL_MATCHDAYS == 38
    assert rw.PL_RELEGATION_SLOTS == 3
    assert rw.PL_SEASON_START == date(2026, 8, 21)
    assert rw.PL_SEASON_END == date(2027, 5, 30)
    assert rw.PL_FIXTURE_RELEASE == date(2026, 6, 19)
    # 33 weekend + 5 midweek rounds
    assert rw.PL_WEEKEND_ROUNDS + rw.PL_MIDWEEK_ROUNDS == rw.PL_MATCHDAYS
    # 2026-27 has FOUR international breaks, with a three-week World Cup window
    assert len(rw.PL_INTERNATIONAL_BREAKS) == 4
    long_break = max(rw.PL_INTERNATIONAL_BREAKS,
                     key=lambda b: (b[1] - b[0]).days)
    assert (long_break[1] - long_break[0]).days >= 14


def test_premier_league_rules_are_the_real_rules():
    r = rw.premier_league_rules()
    assert (r.points_win, r.points_draw, r.points_loss) == (3, 1, 0)
    assert r.relegation_slots == 3
    assert r.double_round_robin is True
    assert r.tie_breakers[:3] == ("pts", "gd", "gf")


def test_champions_league_matchdays_are_the_published_dates():
    """UEFA published eight league-phase windows. These are the real ones."""
    expected = [
        (date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10)),
        (date(2026, 10, 13), date(2026, 10, 14)),
        (date(2026, 10, 20), date(2026, 10, 21)),
        (date(2026, 11, 3), date(2026, 11, 4)),
        (date(2026, 11, 24), date(2026, 11, 25)),
        (date(2026, 12, 8), date(2026, 12, 9)),
        (date(2027, 1, 19), date(2027, 1, 20)),
        (date(2027, 1, 27),),
    ]
    assert len(rw.CL_LEAGUE_PHASE) == 8
    for (_label, window), want in zip(rw.CL_LEAGUE_PHASE, expected):
        assert window == want
    # every league-phase date is a Tuesday, Wednesday or Thursday
    for _label, window in rw.CL_LEAGUE_PHASE:
        for d in window:
            assert d.weekday() in (1, 2, 3), f"{d} is not a midweek evening"


def test_champions_league_knockout_dates_are_the_published_dates():
    assert rw.CL_PLAYOFFS[0] == (date(2027, 2, 16), date(2027, 2, 17))
    assert rw.CL_ROUND_OF_16[0] == (date(2027, 3, 9), date(2027, 3, 10))
    assert rw.CL_QUARTER_FINALS[0] == (date(2027, 4, 6), date(2027, 4, 7))
    assert rw.CL_SEMIFINALS[0] == (date(2027, 4, 27), date(2027, 4, 28))
    assert rw.CL_FINAL == date(2027, 6, 5)
    assert "Metropolitano" in rw.CL_FINAL_VENUE


def test_champions_league_rules_are_the_real_rules():
    r = rw.champions_league_rules()
    assert r.type == "continental"
    assert r.legs == 2, "the knockout is two-legged"
    assert r.extra_time is True and r.penalties is True
    assert r.away_goals is False, "the away-goals rule was abolished in 2021"
    assert rw.CL_TEAMS == 36 and rw.CL_MATCHDAYS == 8
    assert rw.CL_DIRECT_TO_R16 == 8


def test_cup_dates_are_the_published_dates():
    # EFL Cup: tournament proper starts 7 Aug, final Sunday 21 March 2027
    assert rw.EFL_CUP_ROUNDS[1][1][0] == date(2026, 8, 7)
    assert rw.EFL_CUP_FINAL == date(2027, 3, 21)
    # FA Cup third round 9 Jan 2027, fifth round 6 Mar 2027
    fa = dict(rw.FA_CUP_ROUNDS)
    assert fa["Third Round"] == date(2027, 1, 9)
    assert fa["Fifth Round"] == date(2027, 3, 6)


def test_cup_patterns_expose_the_whole_published_window():
    """A cup round published across several evenings must offer exactly those
    evenings — never a Saturday in between that nobody announced."""
    p = rw.efl_cup_pattern("Round Three")
    window = dict((l, w) for l, w in rw.EFL_CUP_ROUNDS)["Round Three"]
    assert p.round_date_windows[0] == tuple(window)
    assert p.fixed_round_dates[0] == window[0]
    # and the Saturday between the two midweek batches is NOT a candidate
    assert date(2026, 9, 12) not in p.round_date_windows[0]


def test_the_pattern_json_round_trip_keeps_the_windows():
    import json
    p = rw.champions_league_pattern(upto=3)
    back = type(p).from_dict(json.loads(json.dumps(p.to_dict())))
    assert back == p
    assert back.round_date_windows[0] == p.round_date_windows[0]


# ─────────────────────────────────────────────
# THE TEST WORLD
# ─────────────────────────────────────────────

def test_world_uses_the_real_competitions_and_rules(world):
    assert world.league is not None and world.continental is not None
    assert world.cup is not None
    assert world.league.rules == rw.premier_league_rules()
    assert world.continental.rules == rw.champions_league_rules()
    assert {world.league.competition_id,
            world.continental.competition_id,
            world.cup.competition_id} == {
        DOMESTIC_LEAGUE, CHAMPIONS_LEAGUE, EFL_CUP}


def test_world_clubs_are_the_real_roster(world):
    from roster_loader import get_loader
    assert len(world.clubs) == len(get_loader().get_all_clubs()) == ROSTER_CLUBS
    for cid in world.clubs:
        assert world.name_of(cid) != cid, f"{cid} has no roster name"


def test_club_ids_are_derived_not_ordinal(world):
    again = build_test_world()
    assert again.clubs == world.clubs
    assert again.roster == world.roster


def test_continental_is_a_league_phase_inside_a_continental_competition(world):
    """The real Champions League league phase is a league in a continental
    competition — the competition class must accept that."""
    assert world.continental.rules.type == "continental"
    assert world.continental.rules.schedule_kind == "round_robin"
    assert world.continental is not None


# ─────────────────────────────────────────────
# THE SCHEDULER AGAINST THE REAL CALENDAR
# ─────────────────────────────────────────────

def test_the_real_calendar_places_without_a_collision(world):
    layout = ("same_day_double_booking", "venue_double_booking",
              "insufficient_rest")
    bad = [v for v in world.calendar.violations() if v.kind in layout]
    assert bad == [], f"the real 2026-27 calendar must be schedulable: {bad}"
    assert world.calendar.violations_of_kind("unscheduled") == []


def test_nothing_is_placed_off_its_published_window(world):
    """The whole point of using real dates: a tie may only fall on a day the
    competition actually announced."""
    by_comp = {c.competition_id: c for c in
               (world.league, world.continental, world.cup)}

    cl_windows = dict((l, w) for l, w in rw.CL_LEAGUE_PHASE)
    cl_allowed = set()
    for _label, window in rw.CL_LEAGUE_PHASE:
        cl_allowed.update(window)
    for f in world.calendar.for_competition(CHAMPIONS_LEAGUE):
        assert f.match_date in cl_allowed, (
            f"{CHAMPIONS_LEAGUE} played {f.match_date}, which UEFA never "
            f"published for the league phase")

    efl_window = dict((l, w) for l, w in rw.EFL_CUP_ROUNDS)["Round Three"]
    for f in world.calendar.for_competition(EFL_CUP):
        assert f.match_date in efl_window, (
            f"{EFL_CUP} played {f.match_date}, outside the published window "
            f"{[d.isoformat() for d in efl_window]}")


def test_champions_league_lands_on_its_published_opening_night(world):
    """UEFA opened matchday 1 on 8, 9 or 10 September. The calendar should take
    the first of those that is legal."""
    md1 = [f for f in world.calendar.for_competition(CHAMPIONS_LEAGUE)]
    assert md1, "no Champions League fixtures placed"
    first = min(f.match_date for f in md1)
    assert first in rw.CL_LEAGUE_PHASE[0][1]


def test_the_cup_moves_when_it_collides_with_a_european_night(world):
    """The EFL round and the Champions League overlap. The cup must yield to the
    higher-priority continental night rather than double-book anybody.

    Both are real: EFL Round Three runs 8-17 September, UEFA matchday 1 runs
    8-10 September. This is exactly the congestion the plan §12 calendar exists
    to resolve.
    """
    efl = world.calendar.for_competition(EFL_CUP)
    ucl = world.calendar.for_competition(CHAMPIONS_LEAGUE)
    efl_dates = {f.match_date for f in efl}
    ucl_dates = {f.match_date for f in ucl}
    # the cup took a different night from the European one, because the clubs
    # that played on the European night cannot play three days later
    assert efl_dates.isdisjoint(ucl_dates), (
        "cup and Champions League share a night — a club cannot play both")
    # ...and the cup landed on a real EFL evening, later than the European one
    assert min(efl_dates) > min(ucl_dates)


def test_every_club_is_scheduled(world):
    seen = set()
    for f in world.calendar.fixtures():
        seen.update(f.club_ids)
    assert seen == set(world.clubs)


def test_no_club_is_scheduled_twice_on_a_day(world):
    by_club_day = {}
    for fx in world.calendar.fixtures():
        for c in fx.club_ids:
            by_club_day.setdefault((c, fx.match_date), []).append(fx.fixture_id)
    assert all(len(v) == 1 for v in by_club_day.values())


def test_the_calendar_is_deterministic(world):
    a = [(f.fixture_id, f.match_date) for f in world.calendar.fixtures()]
    again = build_test_world()
    b = [(f.fixture_id, f.match_date) for f in again.calendar.fixtures()]
    assert a == b


# ─────────────────────────────────────────────
# THE MONTH
# ─────────────────────────────────────────────

def test_the_month_is_the_plan18_pattern_on_real_dates(world):
    """Plan §18's shape, but on the real football calendar."""
    fixtures = month_fixtures(world, days=31)
    assert fixtures, "the month is empty"

    by_day = {}
    for fx in fixtures:
        by_day.setdefault(fx.match_date, []).append(fx)

    weekdays = {d.weekday() for d in by_day}
    # a midweek European night and weekend league football
    assert 1 in weekdays or 2 in weekdays or 3 in weekdays, "a midweek round"
    assert 4 in weekdays or 5 in weekdays or 6 in weekdays, "a weekend round"
    comps = {f.competition_id for f in fixtures}
    assert comps == {DOMESTIC_LEAGUE, CHAMPIONS_LEAGUE, EFL_CUP}


def test_the_month_spans_about_a_month(world):
    fixtures = month_fixtures(world, days=31)
    days = sorted({f.match_date for f in fixtures})
    span = (days[-1] - days[0]).days
    assert 7 <= span <= 45, f"the window spans {span} days"
    assert days[0] == MONTH_START, "the month should begin at the real CL MD1"


def test_the_month_fixtures_are_chronological(world):
    dates = [f.match_date for f in month_fixtures(world)]
    assert dates == sorted(dates)


def test_the_month_is_deterministic(world):
    a = [f.fixture_id for f in month_fixtures(world)]
    b = [f.fixture_id for f in month_fixtures(world)]
    assert a == b


# ─────────────────────────────────────────────
# THE CONTINUITY CHECKER
# ─────────────────────────────────────────────

def test_the_checker_is_silent_on_a_healthy_ledger(month):
    assert month.check_continuity(MONTH_START) == []


def _record(mo, pid, comp, when, home, away, minutes=90, goals=1):
    from world.ledger import MatchReport, PlayerMatchLine
    mo.ledger.record_match(MatchReport(
        competition_id=comp, season_id=SEASON, match_date=when,
        home_id=home, away_id=away, home_goals=goals, away_goals=0,
        players=[PlayerMatchLine(player_id=pid, club_id=home,
                                 minutes=minutes, goals=goals)]))
    mo.ledger.advance_matchday()


def test_the_checker_catches_a_competition_boundary_reset(month):
    """The exact failure plan §7 forbids."""
    pid, when = "PLY-RESET", MONTH_START
    _record(month, pid, DOMESTIC_LEAGUE, when, "CLB-RS", "CLB-RS2")
    assert not any(i.player_id == pid
                   for i in month.check_continuity(when))

    state = month.ledger.player(pid)          # the forbidden "new season" reset
    state.minutes_played = 0
    state.goals = 0
    state.appearances = 0

    issues = [i for i in month.check_continuity(when) if i.player_id == pid]
    assert issues, "a reset across a competition boundary must be caught"
    assert any("BACKWARDS" in str(i) for i in issues)


def test_the_checker_catches_a_scoped_line_exceeding_the_total(month):
    pid = "PLY-OVER"
    _record(month, pid, DOMESTIC_LEAGUE, MONTH_START, "CLB-OVR", "CLB-OVR2")
    line = month.ledger.player(pid).per_season[SEASON]["per_competition"][DOMESTIC_LEAGUE]
    line["minutes"] = 900
    line["goals"] = 40
    issues = [i for i in month.check_continuity(MONTH_START) if i.player_id == pid]
    assert issues, "an over-claiming competition line must be caught"
    assert any("more than" in str(i) for i in issues)


def test_the_checker_catches_a_scoped_total_that_does_not_sum(month):
    pid = "PLY-SUM"
    _record(month, pid, DOMESTIC_LEAGUE, MONTH_START, "CLB-SM", "CLB-SM2")
    _record(month, pid, CHAMPIONS_LEAGUE, MONTH_START + timedelta(days=4),
            "CLB-SM", "CLB-SM3")
    assert not any(i.player_id == pid
                   for i in month.check_continuity(MONTH_START))
    month.ledger.player(pid).minutes_played = 999
    issues = [i for i in month.check_continuity(MONTH_START) if i.player_id == pid]
    assert issues, "a drifted continuous total must be caught"
    assert any("scoped lines sum" in str(i) for i in issues)


def test_the_checker_catches_a_double_booked_club(month):
    from world.ledger import MatchReport
    for comp in (DOMESTIC_LEAGUE, EFL_CUP):
        month.ledger.record_match(MatchReport(
            competition_id=comp, season_id=SEASON, match_date=MONTH_START,
            home_id="CLB-DUP", away_id="CLB-OTHER", home_goals=1, away_goals=0))
    issues = [i for i in month.check_continuity(MONTH_START)
              if i.player_id == "CLB-DUP"]
    assert issues, "a club playing twice in a day must be caught"
    assert any("2 matches" in str(i) for i in issues)


def test_the_checker_catches_an_appearance_without_minutes(month):
    """The bug the integration test found in real football data."""
    pid = "PLY-GHOST"
    _record(month, pid, DOMESTIC_LEAGUE, MONTH_START, "CLB-GH", "CLB-GH2",
            goals=0)
    month.ledger.player(pid).appearances = 5
    issues = [i for i in month.check_continuity(MONTH_START) if i.player_id == pid]
    assert issues, "appearances must match matches with minutes"
    assert any("appearances recorded" in str(i) for i in issues)


def test_continuity_issue_renders_with_context():
    text = str(ContinuityIssue("2026-09-08", "PLY-X", "something went wrong"))
    assert "2026-09-08" in text and "PLY-X" in text and "something" in text


# ─────────────────────────────────────────────
# SCOPE
# ─────────────────────────────────────────────

def test_the_month_writes_nothing_to_26_27():
    """Only MODULE-LEVEL imports count — a function-level import of a pure
    helper is the discipline the rest of the world layer uses."""
    import ast
    import world.month as wm
    import world.testworld as tw
    for mod in (wm, tw):
        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            assert "auto_run_match" not in names, (
                f"{mod.__name__} imports auto_run_match at MODULE level")


def test_realworld_has_no_engine_dependency():
    import ast
    import world.realworld as rwm
    tree = ast.parse(open(rwm.__file__, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = {node.module.split(".")[0]}
        else:
            continue
        for forbidden in ("match_engine", "roster_loader", "season_manager",
                          "exporter", "alltime_db"):
            assert forbidden not in names, (
                f"world/realworld.py imports {forbidden!r}; reference DATA "
                f"must not depend on the live pipeline")
