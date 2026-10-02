"""
Test PLOFA WORLD calendar — world/calendar.py
==============================================
Phase 4 (PLOFA_WORLD_LAYER_AUDIT.md §16 row 4, plan §12).

Plan §12 calls the calendar "one of the most important systems" and the audit's
§18 risk 3 states the rule this file exists to enforce:

    "The calendar's invariant must be enforced by test, not convention."

So the centre of this suite is adversarial: fixtures that *want* to collide are
handed to the calendar on purpose, and the invariant is asserted afterwards.
A green suite here means the invariant holds against an adversary, not merely
in the happy path.

Also covered: determinism (no PYTHONHASHSEED dependence), priority ordering,
rest windows, venue exclusivity, postponement, and the guarantee that the
calendar describes matches without ever running one.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta

import pytest

from world.calendar import (
    Calendar,
    CalendarConflictError,
    CompetitionEntry,
    PlannedFixture,
    SchedulePattern,
    Violation,
    cup_midweek,
    league_saturday,
)
from world.calendar import (
    SAT, SUN, TUE, WED,
    continental_midweek,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = date(2026, 8, 8)          # a Saturday


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

CLUBS = [f"CLB-{i:02d}" for i in range(8)]


def _calendar(**kw) -> Calendar:
    kw.setdefault("country_id", "CTR-TOL")
    kw.setdefault("season_id", "26/27")
    kw.setdefault("start", START)
    return Calendar(**kw)


def _double_round_robin(teams) -> list:
    """A full home-and-away schedule as PlannedFixtures, matchday-tagged.

    Mirrors the circle method of season_manager.FixtureList.round_robin so the
    adversarial tests feed the calendar exactly the shape production will.
    """
    teams = list(teams)
    n = len(teams)
    rounds, fixed, rest = [], teams[0], teams[1:]
    for _ in range(n - 1):
        rounds.append(list(zip([fixed] + rest[: n // 2 - 1], reversed(rest[n // 2 - 1:]))))
        rest = [rest[-1]] + rest[:-1]
    out = []
    md = 1
    for rnd in rounds:
        for h, a in rnd:
            out.append(PlannedFixture(h, a, round_name=f"MD {md}",
                                      stage_name="League", matchday=md))
        md += 1
    for i, rnd in enumerate(rounds):
        for h, a in rnd:
            out.append(PlannedFixture(a, h, round_name=f"MD {md}",
                                      stage_name="League", matchday=md))
        md += 1
    return out


# ─────────────────────────────────────────────
# SCHEDULE PATTERN — DATA VALIDATION
# ─────────────────────────────────────────────

def test_pattern_rejects_impossible_values():
    with pytest.raises(ValueError):
        SchedulePattern(name="x", weekdays=(9,))
    with pytest.raises(ValueError):
        SchedulePattern(name="x", kickoffs=())
    with pytest.raises(ValueError):
        SchedulePattern(name="x", round_interval_days=0)
    with pytest.raises(ValueError):
        SchedulePattern(name="x", min_rest_days=-1)


def test_pattern_json_round_trip():
    p = cup_midweek()
    assert SchedulePattern.from_dict(p.to_dict()) == p
    # and through a real file round trip
    assert SchedulePattern.from_dict(json.loads(json.dumps(p.to_dict()))) == p


def test_preset_priorities_encode_football_hierarchy():
    """Continental > domestic cup > league. A cup tie must never lose its
    Saturday to a league fixture simply because the league was built first."""
    assert continental_midweek().priority > cup_midweek().priority > league_saturday().priority


# ─────────────────────────────────────────────
# THE HARD INVARIANT — NO CLUB PLAYS TWICE IN A DAY
# ─────────────────────────────────────────────

def test_league_alone_never_double_books_a_club():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(),
                        _double_round_robin(CLUBS))
    cal.build()
    assert cal.violations_of_kind("same_day_double_booking") == []
    for club in CLUBS:
        dates = [f.match_date for f in cal.for_club(club)]
        assert len(dates) == len(set(dates)), f"{club} double-booked"


def test_three_competitions_on_one_club_set_never_double_book():
    """The plan §12 scenario exactly: Saturday league, Wednesday continental,
    Tuesday cup — all eight clubs in all three at once."""
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(),
                        _double_round_robin(CLUBS))
    cal.add_competition("CMP-CUP", "cup", cup_midweek(), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="Round of 8", stage_name="Knockout"),
        PlannedFixture(CLUBS[2], CLUBS[3], round_name="Round of 8", stage_name="Knockout"),
        PlannedFixture(CLUBS[4], CLUBS[5], round_name="Round of 8", stage_name="Knockout"),
        PlannedFixture(CLUBS[6], CLUBS[7], round_name="Round of 8", stage_name="Knockout"),
    ])
    cal.add_competition("CMP-EUCL", "continental", continental_midweek(), [
        PlannedFixture(CLUBS[0], CLUBS[2], round_name="MD 1", stage_name="League phase"),
        PlannedFixture(CLUBS[1], CLUBS[3], round_name="MD 1", stage_name="League phase"),
    ])
    cal.build()

    assert cal.violations_of_kind("same_day_double_booking") == []
    for club in CLUBS:
        dates = [f.match_date for f in cal.for_club(club)]
        assert len(dates) == len(set(dates)), (
            f"{club} was double-booked: "
            f"{[d.isoformat() for d in dates]}"
        )


def test_adversarial_pile_on_one_club_is_resolved_not_violated():
    """Six competitions all demanding the same club on the same day.

    The calendar must spread them across legal dates. If it cannot, it must
    report an unscheduled fixture — it must never emit a double booking.
    """
    cal = _calendar()
    # six one-off matches, every one of them involving CLB-00
    for i in range(6):
        cal.add_competition(
            f"CMP-{i}", "cup",
            cup_midweek(weekdays=(TUE, WED, SAT), name=f"c{i}"),
            [PlannedFixture(CLUBS[0], CLUBS[1 + i], round_name="R1")],
        )
    cal.build()

    assert cal.violations_of_kind("same_day_double_booking") == []
    dates = [f.match_date for f in cal.for_club(CLUBS[0])]
    assert len(dates) == len(set(dates))
    # a home and away fixture for the same pairing on the same day is the
    # classic self-inflicted wound
    for fx in cal.fixtures():
        assert fx.home_id != fx.away_id


def test_every_planned_fixture_lands_on_a_pattern_weekday():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(),
                        _double_round_robin(CLUBS))
    cal.add_competition("CMP-EUCL", "continental", continental_midweek(), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="MD 1"),
    ])
    cal.build()
    for fx in cal.for_competition("CMP-D1"):
        assert fx.match_date.weekday() == SAT
    for fx in cal.for_competition("CMP-EUCL"):
        assert fx.match_date.weekday() == WED


def test_minimum_rest_is_respected_across_competitions():
    cal = _calendar(default_min_rest_days=4)
    cal.add_competition("CMP-D1", "league", league_saturday(min_rest_days=4),
                        _double_round_robin(CLUBS[:4]))
    cal.add_competition("CMP-EUCL", "continental",
                        continental_midweek(min_rest_days=4), [
                            PlannedFixture(CLUBS[0], CLUBS[1], round_name="MD 1"),
                            PlannedFixture(CLUBS[2], CLUBS[3], round_name="MD 1"),
                        ])
    cal.build()
    assert cal.violations_of_kind("insufficient_rest") == []
    for club in CLUBS[:4]:
        fx = cal.for_club(club)
        for a, b in zip(fx, fx[1:]):
            assert (b.match_date - a.match_date).days >= 4, (
                f"{club}: only {(b.match_date - a.match_date).days} days rest"
            )


def test_a_stricter_competition_floor_governs_its_own_gaps():
    """A competition may demand MORE rest than the calendar default, never less.

    Both the placer and the independent validator must resolve that floor the
    same way, otherwise the check verifies a weaker rule than the placer
    enforced and a real rest violation passes unnoticed.
    """
    cal = _calendar(default_min_rest_days=1)
    cal.add_competition("CMP-D1", "league",
                        league_saturday(min_rest_days=1, round_interval_days=14),
                        _double_round_robin(CLUBS))
    # a cup demanding 10 days between its own ties
    cal.add_competition("CMP-CUP", "cup",
                        cup_midweek(weekdays=(SAT,), min_rest_days=10), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="R1"),
        PlannedFixture(CLUBS[0], CLUBS[2], round_name="R2"),
    ])
    cal.build()

    assert cal.violations_of_kind("insufficient_rest") == []
    cup_fx = sorted(cal.for_competition("CMP-CUP"), key=lambda f: f.match_date)
    gap = (cup_fx[1].match_date - cup_fx[0].match_date).days
    assert gap >= 10, f"cup floor of 10 days not honoured (got {gap})"
    # ...and the league, whose own floor is 1, was not dragged up to 10
    league_gaps = [
        (b.match_date - a.match_date).days
        for a, b in zip(cal.for_club(CLUBS[4]), cal.for_club(CLUBS[4])[1:])
    ]
    assert min(league_gaps) < 10, "the stricter cup floor leaked into the league"


def test_validate_catches_a_rest_floor_the_default_would_allow():
    """Prove the validator is genuinely independent: plant a 2-day gap that the
    default rest floor permits but the cup's stricter floor does not."""
    cal = _calendar(default_min_rest_days=1)
    cal.add_competition("CMP-D1", "league",
                        league_saturday(min_rest_days=1, round_interval_days=14),
                        _double_round_robin(CLUBS))
    cal.add_competition("CMP-CUP", "cup",
                        cup_midweek(weekdays=(SAT,), min_rest_days=10), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="R1"),
        PlannedFixture(CLUBS[0], CLUBS[2], round_name="R2"),
    ])
    cal.build()
    before = len(cal.violations_of_kind("insufficient_rest"))
    assert before == 0

    # reach past the placer and force a short gap directly on the fixture
    a, b = sorted(cal.for_competition("CMP-CUP"), key=lambda f: f.match_date)
    b.match_date = a.match_date + timedelta(days=2)
    rest = [v for v in cal.validate() if v.kind == "insufficient_rest"]
    assert rest, "validator missed a rest violation the default floor allows"
    assert "needs 10" in rest[0].detail


def test_a_venue_is_never_double_booked():
    """A shared stadium is a real constraint — two clubs, one ground, one day."""
    cal = _calendar(venue_resolver=lambda c: "National Stadium" if c in ("CLB-00", "CLB-01") else f"{c} Stadium")
    cal.add_competition("CMP-D1", "league", league_saturday(weekdays=(SAT, SUN)),
                        [
                            PlannedFixture("CLB-00", "CLB-04", round_name="R1", matchday=1),
                            PlannedFixture("CLB-01", "CLB-05", round_name="R1", matchday=1),
                            PlannedFixture("CLB-00", "CLB-06", round_name="R1", matchday=1),
                            PlannedFixture("CLB-01", "CLB-07", round_name="R1", matchday=1),
                        ])
    cal.build()
    assert cal.violations_of_kind("venue_double_booking") == []
    seen = {}
    for fx in cal.fixtures():
        key = (fx.venue, fx.match_date)
        assert key not in seen, f"{fx.venue} hosts two matches on {key[1]}"
        seen[key] = fx.fixture_id


def test_neutral_venue_overrides_the_home_club_ground():
    cal = _calendar()
    cal.add_competition("CMP-CUP", "cup", cup_midweek(), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="Final",
                       neutral_venue="Wembley Equivalent"),
    ])
    cal.build()
    assert cal.fixtures()[0].venue == "Wembley Equivalent"


# ─────────────────────────────────────────────
# DETERMINISM
# ─────────────────────────────────────────────

def test_build_is_deterministic_within_a_process():
    def make():
        cal = _calendar()
        cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS))
        cal.add_competition("CMP-CUP", "cup", cup_midweek(), [
            PlannedFixture(CLUBS[i], CLUBS[i + 1], round_name="R1")
            for i in range(0, 6, 2)
        ])
        cal.build()
        return [(f.fixture_id, f.match_date.isoformat(), f.kickoff, f.venue)
                for f in cal.fixtures()]

    assert make() == make()


def test_build_is_deterministic_across_processes():
    """Guards the PYTHONHASHSEED trap of README §8.

    Two fresh interpreters with DIFFERENT hash seeds must produce byte-identical
    calendars. If placement ever leaks dict/set iteration order, this fails.
    """
    script = (
        "import sys, json;"
        "sys.path.insert(0, %r);"
        "from datetime import date;"
        "from world.calendar import Calendar, PlannedFixture, league_saturday, cup_midweek;"
        "c = Calendar('CTR-TOL','26/27',date(2026,8,8));"
        "c.add_competition('CMP-D1','league',league_saturday(),"
        "  [PlannedFixture('A','B',matchday=1),PlannedFixture('C','D',matchday=1),"
        "   PlannedFixture('A','C',matchday=2),PlannedFixture('B','D',matchday=2)]);"
        "c.add_competition('CMP-CUP','cup',cup_midweek(),"
        "  [PlannedFixture('A','D',round_name='R1'),PlannedFixture('B','C',round_name='R1')]);"
        "c.build();"
        "print(json.dumps([[f.fixture_id,f.match_date.isoformat(),f.kickoff]"
        "  for f in c.fixtures()]))" % REPO_ROOT
    )
    outs = []
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run([sys.executable, "-c", script], env=env,
                             capture_output=True, text=True, cwd=REPO_ROOT)
        assert res.returncode == 0, res.stderr
        outs.append(res.stdout.strip())
    assert outs[0] == outs[1], (
        "calendar placement leaked hash-order dependence:\n"
        f"  seed 0:    {outs[0]}\n  seed 12345: {outs[1]}"
    )


def test_build_is_idempotent():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS))
    first = [(f.fixture_id, f.match_date) for f in cal.build()]
    second = [(f.fixture_id, f.match_date) for f in cal.build()]
    assert first == second


def test_fixture_ids_are_unique_and_stable():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS))
    cal.build()
    ids = [f.fixture_id for f in cal.fixtures()]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("FX-26/27-") for i in ids)


# ─────────────────────────────────────────────
# PRIORITY
# ─────────────────────────────────────────────

def test_higher_priority_competition_keeps_its_preferred_day():
    """Both want the same Saturday; the cup must win and the league move."""
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(priority=100), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="MD 1", matchday=1),
    ])
    cal.add_competition("CMP-CUP", "cup",
                        cup_midweek(weekdays=(SAT,), priority=200), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="Final"),
    ])
    cal.build()
    cup = cal.for_competition("CMP-CUP")[0]
    league = cal.for_competition("CMP-D1")[0]
    assert cup.match_date == START                       # the Saturday it wanted
    assert league.match_date != START                   # the league gave way
    assert league.match_date.weekday() == SAT            # but stayed a Saturday


def test_league_added_first_still_loses_the_date():
    """Ordering of ``add_competition`` must not change the outcome — priority
    is data, not insertion order."""
    cal = _calendar()
    cal.add_competition("CMP-CUP", "cup",
                        cup_midweek(weekdays=(SAT,), priority=200), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="Final"),
    ])
    cal.add_competition("CMP-D1", "league", league_saturday(priority=100), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="MD 1", matchday=1),
    ])
    cal.build()
    assert cal.for_competition("CMP-CUP")[0].match_date == START
    assert cal.for_competition("CMP-D1")[0].match_date != START


# ─────────────────────────────────────────────
# UNPLACEABLE FIXTURES ARE REPORTED, NEVER DROPPED
# ─────────────────────────────────────────────

def test_unplaceable_fixture_is_reported_not_silently_dropped():
    """A club cannot play 40 matches on the only allowed weekday inside a
    fortnight-long horizon. The calendar must say so."""
    cal = _calendar(horizon_days=14)
    cal.add_competition("CMP-CUP", "cup", cup_midweek(weekdays=(SAT,)), [
        PlannedFixture(CLUBS[0], CLUBS[1 + (i % 4)], round_name=f"R{i}")
        for i in range(40)
    ])
    cal.build()
    unscheduled = cal.violations_of_kind("unscheduled")
    assert unscheduled, "a 40-match pile-up cannot fit — it must be reported"
    assert cal.violations_of_kind("same_day_double_booking") == []
    # everything that WAS placed is still legal
    for club in {c for f in cal.fixtures() for c in f.club_ids}:
        dates = [f.match_date for f in cal.for_club(club)]
        assert len(dates) == len(set(dates))


def test_assert_clean_raises_on_a_broken_calendar():
    cal = _calendar(horizon_days=10)
    cal.add_competition("CMP-CUP", "cup", cup_midweek(weekdays=(SAT,)), [
        PlannedFixture(CLUBS[0], CLUBS[1 + (i % 4)], round_name=f"R{i}")
        for i in range(20)
    ])
    cal.build()
    with pytest.raises(CalendarConflictError):
        cal.assert_clean()


def test_a_fit_season_asserts_clean():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS))
    cal.build()
    assert cal.assert_clean() is cal


# ─────────────────────────────────────────────
# POSTPONEMENT
# ─────────────────────────────────────────────

def test_reschedule_moves_a_fixture_and_records_where_it_came_from():
    """A calendar with genuine slack: the league plays alternate Saturdays, so
    the off-week is free and a fixture can legitimately be pushed into it."""
    cal = _calendar(default_min_rest_days=1)
    cal.add_competition("CMP-D1", "league",
                        league_saturday(min_rest_days=1, round_interval_days=14),
                        _double_round_robin(CLUBS))
    cal.build()
    fx = cal.for_club(CLUBS[0])[0]
    before = fx.match_date
    target = before + timedelta(days=7)          # the empty week
    cal.reschedule(fx.fixture_id, target, reason="flooded pitch")
    assert fx.match_date == target
    assert fx.postponed_from == before.isoformat()
    assert fx.postpone_count == 1
    assert cal.assert_clean() is cal


def test_reschedule_refuses_to_create_a_double_booking():
    """A postponement that breaks the hard invariant must be rejected outright,
    not applied with a warning."""
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS))
    cal.add_competition("CMP-CUP", "cup", cup_midweek(weekdays=(SAT,)), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="Final"),
    ])
    cal.build()
    cup_final = cal.for_competition("CMP-CUP")[0]
    before = cup_final.match_date
    with pytest.raises(CalendarConflictError):
        cal.reschedule(cup_final.fixture_id, before + timedelta(days=7),
                       reason="deliberate collision")
    # nothing moved
    assert cal.fixture(cup_final.fixture_id).match_date == before
    assert cup_final.postpone_count == 0
    assert cal.assert_clean() is cal


def test_reschedule_unknown_fixture_raises():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS[:4]))
    cal.build()
    with pytest.raises(KeyError):
        cal.reschedule("FX-nope", date(2026, 9, 1))


def test_postpone_finds_the_first_legal_slot_and_skips_illegal_dates():
    """A fully-packed weekly league has no free Saturday at all, so a naive
    'roll to the next pattern weekday' postponement could never succeed.

    The calendar must search forward for the first date that satisfies every
    invariant, and report how far back the match actually had to go.
    """
    cal = _calendar()
    cal.add_competition("CMP-D1", "league",
                        league_saturday(weekdays=(SAT, SUN), min_rest_days=1),
                        _double_round_robin(CLUBS))
    cal.build()
    fx = cal.for_club(CLUBS[0])[0]
    before = fx.match_date

    cal.postpone(fx.fixture_id, days=1)

    assert fx.match_date > before
    assert fx.match_date.weekday() in (SAT, SUN)   # still a legal pattern day
    assert fx.postponed_from == before.isoformat()
    assert fx.postpone_count == 1
    # the whole calendar is still legal afterwards
    assert cal.assert_clean() is cal


def test_postpone_gives_up_loudly_rather_than_double_booking():
    """A calendar with no slack at all: postponement must fail, not cheat."""
    cal = _calendar(horizon_days=30, default_min_rest_days=1)
    cal.add_competition("CMP-CUP", "cup",
                        cup_midweek(weekdays=(SAT,), min_rest_days=1), [
        PlannedFixture(CLUBS[0], CLUBS[1 + (i % 4)], round_name=f"R{i}")
        for i in range(24)
    ])
    cal.build()
    fx = cal.for_club(CLUBS[0])[0]
    with pytest.raises(CalendarConflictError):
        cal.postpone(fx.fixture_id, days=1, search_days=5)
    # giving up loudly must not have corrupted the layout
    assert cal.violations_of_kind("same_day_double_booking") == []
    assert cal.violations_of_kind("venue_double_booking") == []
    assert cal.violations_of_kind("insufficient_rest") == []


def test_postpone_rejects_a_non_positive_delay():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS[:4]))
    cal.build()
    with pytest.raises(ValueError):
        cal.postpone(cal.fixtures()[0].fixture_id, days=0)


# ─────────────────────────────────────────────
# QUERIES & REPORTING
# ─────────────────────────────────────────────

def test_matchdays_groups_a_competition_into_dated_rounds():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS))
    cal.build()
    rounds = cal.matchdays("CMP-D1")
    assert len(rounds) == 14                       # 8 teams, double round robin
    dates = [d for d, _ in rounds]
    assert dates == sorted(dates)                  # chronological
    for _d, fxs in rounds:
        assert len(fxs) == 4                       # 8 clubs → 4 matches a round
        # no club appears twice within a round, by construction
        clubs = [c for f in fxs for c in f.club_ids]
        assert len(clubs) == len(set(clubs))


def test_kickoffs_are_spread_across_a_matchday():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league",
                        league_saturday(kickoffs=("13:00", "15:00", "17:30", "20:00")),
                        _double_round_robin(CLUBS))
    cal.build()
    first_round = cal.matchdays("CMP-D1")[0][1]
    assert len({f.kickoff for f in first_round}) == 4


def test_congestion_reports_busy_spells_without_failing():
    """Congestion is a fact about a season, not an error — it must be visible
    (plan §12 'fixture congestion') but never a violation.

    The realistic congested week: Saturday league plus a Tuesday cup tie plus a
    Wednesday continental match, with rest floors relaxed so the calendar is
    allowed to compress them (which is exactly when congestion becomes real).
    """
    cal = _calendar(default_min_rest_days=1)
    cal.add_competition("CMP-D1", "league",
                        league_saturday(min_rest_days=1), [
        PlannedFixture(CLUBS[0], CLUBS[1], round_name="MD 1", matchday=1),
        PlannedFixture(CLUBS[2], CLUBS[3], round_name="MD 1", matchday=1),
    ])
    cal.add_competition("CMP-CUP", "cup",
                        cup_midweek(weekdays=(SAT, TUE), min_rest_days=1), [
        PlannedFixture(CLUBS[0], CLUBS[2], round_name="R1"),
    ])
    cal.add_competition("CMP-EUCL", "continental",
                        continental_midweek(weekdays=(SAT, TUE, WED), min_rest_days=1), [
        PlannedFixture(CLUBS[0], CLUBS[3], round_name="MD 1"),
    ])
    cal.build()

    busy = [n for _d, n in cal.congestion(CLUBS[0], window_days=7)]
    assert max(busy) >= 2, "a league+cup+continental week must read as congested"
    # congestion is reported, never raised
    assert cal.violations_of_kind("insufficient_rest") == []
    assert cal.violations_of_kind("same_day_double_booking") == []


def test_on_and_for_competition_queries_agree_with_the_fixture_list():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS))
    cal.build()
    total = sum(len(cal.on(f.match_date)) for f in cal.fixtures()[:1])
    assert total == len(cal.on(cal.fixtures()[0].match_date))
    assert len(cal.for_competition("CMP-D1")) == len(cal.fixtures())


# ─────────────────────────────────────────────
# SERIALISATION
# ─────────────────────────────────────────────

def test_calendar_json_round_trips():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS))
    cal.build()
    blob = json.dumps(cal.to_dict())
    back = json.loads(blob)
    assert back["country_id"] == "CTR-TOL"
    assert back["season_id"] == "26/27"
    assert len(back["fixtures"]) == len(cal.fixtures())
    assert back["fixtures"][0]["match_date"] == cal.fixtures()[0].match_date.isoformat()


def test_world_fixture_from_dict_round_trips():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS[:4]))
    cal.build()
    from world.calendar import WorldFixture
    for fx in cal.fixtures():
        assert WorldFixture.from_dict(fx.to_dict()).to_dict() == fx.to_dict()


def test_summary_is_readable_and_counts_competitions():
    cal = _calendar()
    cal.add_competition("CMP-D1", "league", league_saturday(), _double_round_robin(CLUBS))
    cal.build()
    text = cal.summary()
    assert "CTR-TOL" in text and "26/27" in text
    assert "CMP-D1" in text
    assert "0 violation" in text


# ─────────────────────────────────────────────
# ARCHITECTURE INVARIANT
# ─────────────────────────────────────────────

def test_calendar_does_not_import_the_match_engine():
    """Plan §10: the calendar decides *when*, never *what*. It must not reach
    for the engine — the only hand-off is MatchContextFragment at call time."""
    import world.calendar as wcal
    for line in open(wcal.__file__, encoding="utf-8"):
        if line.startswith(("import ", "from ")):
            assert "match_engine" not in line
            assert "season_manager" not in line


def test_calendar_module_has_no_module_level_26_27_imports():
    import world.calendar as wcal
    for line in open(wcal.__file__, encoding="utf-8"):
        if line.startswith(("import ", "from ")):
            assert "auto_run_match" not in line, (
                "the calendar must never touch the production runner"
            )


# ─────────────────────────────────────────────
# INTEGRATION WITH THE COMPETITION LAYER
# ─────────────────────────────────────────────

def test_calendar_consumes_a_real_league_competition():
    """End-to-end: LeagueCompetition → calendar → dated, legal fixtures."""
    from world.competition import LeagueCompetition, league_rules

    lg = LeagueCompetition(
        competition_id="CMP-D1", name="Toland First Division",
        country_id="CTR-TOL", season_id="26/27",
        rules=league_rules(), participant_ids=CLUBS, seed=7,
    )
    lg.make_fixtures(START, days_between=7)

    cal = _calendar()
    cal.add_league(lg, league_saturday())
    cal.build()

    assert len(cal.fixtures()) == len(lg._fixtures.fixtures)
    assert cal.assert_clean() is cal
    for f in cal.fixtures():
        assert f.match_date.weekday() == SAT


def test_calendar_consumes_a_real_knockout_round():
    from world.competition import KnockoutCompetition, cup_rules

    ko = KnockoutCompetition(
        competition_id="CMP-CUP", name="Toland Cup",
        country_id="CTR-TOL", season_id="26/27",
        rules=cup_rules(), participant_ids=CLUBS, seed=7,
    )
    ko.draw_round("Quarter-final")

    cal = _calendar()
    cal.add_knockout_round(ko, cup_midweek(), "Quarter-final")
    cal.build()

    assert len(cal.fixtures()) == 4
    assert all(f.competition_type == "cup" for f in cal.fixtures())
    assert cal.assert_clean() is cal


def test_add_league_without_fixtures_raises_clearly():
    from world.competition import LeagueCompetition, league_rules
    lg = LeagueCompetition("CMP-D1", "X", "CTR-TOL", "26/27",
                           league_rules(), CLUBS[:4], seed=1)
    with pytest.raises(ValueError, match="make_fixtures"):
        _calendar().add_league(lg, league_saturday())


def test_add_knockout_round_before_drawing_raises_clearly():
    from world.competition import KnockoutCompetition, cup_rules
    ko = KnockoutCompetition("CMP-CUP", "X", "CTR-TOL", "26/27",
                             cup_rules(), CLUBS[:4], seed=1)
    with pytest.raises(ValueError, match="drawn"):
        _calendar().add_knockout_round(ko, cup_midweek(), "Semi-final")


def test_fixture_carries_competition_context_for_the_engine():
    """The calendar's last job: hand the engine a description of the match."""
    from world.competition import KnockoutCompetition, cup_rules

    ko = KnockoutCompetition("CMP-CUP", "Toland Cup", "CTR-TOL", "26/27",
                             cup_rules(), CLUBS[:4], seed=7)
    ko.draw_round("Semi-final")
    cal = _calendar()
    cal.add_knockout_round(ko, cup_midweek(), "Semi-final")
    cal.build()

    fx = cal.fixtures()[0]
    ctx = ko.context_for(stage_name=fx.stage_name, round_name=fx.round_name)
    assert ctx.competition_id == "CMP-CUP"
    assert ctx.competition_type == "cup"
    assert ctx.penalties is True
