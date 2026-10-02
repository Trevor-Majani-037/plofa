"""
PLOFA WORLD — real football rules and fixture dates.
===================================================
world/realworld.py  ·  Real-world reference data for 2026-27.

Everything here is **real**: the published Champions League matchday dates, the
Premier League season shape, and the domestic cup round dates, with the actual
rules each competition plays by. It exists because a test world built on invented
rules only proves the plumbing works — it says nothing about whether the rules
are right, and the whole point of a football simulator is that the rules match
the sport.

Sources (retrieved 2026-09-26):
  * UEFA, "2026/27 Champions League: Teams, dates, draws, format, final"
  * Premier League, "Dates for 2026/27 Premier League season confirmed"
  * Wikipedia, 2026-27 Premier League / Champions League / EFL Cup / FA Cup
  * EFL, Carabao Cup round dates
  * The FA, Emirates FA Cup round dates

Two things this data deliberately does NOT do: it does not invent a club list,
and it does not guess a date that was not published. Where a competition plays
on a two-day window (the Champions League, "8-10 September"), both candidate
dates are supplied and the calendar takes the first that is legal.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Tuple

from world.calendar import FRI, SAT, SUN, THU, TUE, WED, SchedulePattern
from world.competition import CompetitionRules, cup_rules, league_rules

SEASON = "2026/27"

# ─────────────────────────────────────────────
# PREMIER LEAGUE 2026-27
# ─────────────────────────────────────────────

PL_TEAMS = 20
PL_MATCHDAYS = 38
PL_SEASON_START = date(2026, 8, 21)      # opening match (Friday)
PL_MAIN_START = date(2026, 8, 22)        # the Saturday everyone else starts
PL_SEASON_END = date(2027, 5, 30)
PL_FIXTURE_RELEASE = date(2026, 6, 19)
PL_RELEGATION_SLOTS = 3
PL_WEEKEND_ROUNDS = 33
PL_MIDWEEK_ROUNDS = 5
#: 2026-27 runs on FOUR international breaks, not five, with a three-week
#: window over the FIFA World Cup off-season. That is why the opening and final
#: matchweeks were both pushed back a week.
PL_INTERNATIONAL_BREAKS: Tuple[Tuple[date, date], ...] = (
    (date(2026, 9, 26), date(2026, 10, 17)),   # 3-week, World Cup
    (date(2026, 11, 21), date(2026, 11, 30)),
    (date(2027, 1, 23), date(2027, 2, 1)),
    (date(2027, 3, 27), date(2027, 4, 3)),
)


def premier_league_rules() -> CompetitionRules:
    """3 points for a win, 1 for a draw, bottom three relegated."""
    return CompetitionRules(
        type="league",
        schedule_kind="round_robin",
        points_win=3, points_draw=1, points_loss=0,
        tie_breakers=("pts", "gd", "gf"),
        double_round_robin=True,
        relegation_slots=PL_RELEGATION_SLOTS,
        promotion_slots=0,
        substitutions_max=5,          # 2026-27: five substitutions permitted
        max_bench=9,
        suspension_type="yellow_accum",
        suspension_threshold=5,
        suspension_window=6,
    )


def premier_league_pattern() -> SchedulePattern:
    """Weekend rounds, with a midweek option for the five congested rounds."""
    return SchedulePattern(
        name="premier_league",
        weekdays=(FRI, SAT, SUN),
        kickoffs=("12:30", "15:00", "17:30"),
        round_interval_days=7,
        min_rest_days=3,
        priority=100,
    )


# ─────────────────────────────────────────────
# UEFA CHAMPIONS LEAGUE 2026-27
# ─────────────────────────────────────────────

CL_TEAMS = 36
CL_MATCHDAYS = 8

#: The published league-phase windows. Each matchday may fall on any of its
#: listed dates, so the calendar is given the real options rather than one
#: invented Saturday.
CL_LEAGUE_PHASE: Tuple[Tuple[str, Tuple[date, ...]], ...] = (
    ("Matchday 1", (date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10))),
    ("Matchday 2", (date(2026, 10, 13), date(2026, 10, 14))),
    ("Matchday 3", (date(2026, 10, 20), date(2026, 10, 21))),
    ("Matchday 4", (date(2026, 11, 3), date(2026, 11, 4))),
    ("Matchday 5", (date(2026, 11, 24), date(2026, 11, 25))),
    ("Matchday 6", (date(2026, 12, 8), date(2026, 12, 9))),
    ("Matchday 7", (date(2027, 1, 19), date(2027, 1, 20))),
    ("Matchday 8", (date(2027, 1, 27),)),
)

#: Knockout phase, two legs each until the final.
CL_PLAYOFFS: Tuple[Tuple[date, date], ...] = (
    (date(2027, 2, 16), date(2027, 2, 17)),
    (date(2027, 2, 23), date(2027, 2, 24)),
)
CL_ROUND_OF_16: Tuple[Tuple[date, date], ...] = (
    (date(2027, 3, 9), date(2027, 3, 10)),
    (date(2027, 3, 16), date(2027, 3, 17)),
)
CL_QUARTER_FINALS: Tuple[Tuple[date, date], ...] = (
    (date(2027, 4, 6), date(2027, 4, 7)),
    (date(2027, 4, 13), date(2027, 4, 14)),
)
CL_SEMIFINALS: Tuple[Tuple[date, date], ...] = (
    (date(2027, 4, 27), date(2027, 4, 28)),
    (date(2027, 5, 4), date(2027, 5, 5)),
)
CL_FINAL = date(2027, 6, 5)
CL_FINAL_VENUE = "Estadio Metropolitano, Madrid"

#: Top 8 go straight to the round of 16; 9th-24th contest the play-offs;
#: 25th-36th are eliminated. This is the 2024-25-onwards league-phase format.
CL_DIRECT_TO_R16 = 8
CL_PLAYOFF_SPOTS = 16
CL_ELIMINATED_FROM = 25


def champions_league_rules() -> CompetitionRules:
    """League phase: standard 3/1/0, then knockout with two legs and penalties.

    The away-goals rule was abolished in 2021 and is NOT enabled here.
    """
    return CompetitionRules(
        type="continental",
        schedule_kind="round_robin",
        points_win=3, points_draw=1, points_loss=0,
        tie_breakers=("pts", "gd", "gf", "wins", "draws"),
        double_round_robin=False,
        extra_time=True,
        penalties=True,
        aggregate=True,
        away_goals=False,          # abolished 2021
        legs=2,
        substitutions_max=5,
        max_bench=9,
        suspension_type="yellow_accum",
        suspension_threshold=5,
        suspension_window=6,
    )


def champions_league_pattern(
    upto: int = 0, *, slip_days: int = 1, priority: int = 300,
) -> SchedulePattern:
    """The real published league-phase dates.

    ``upto`` truncates the calendar to the first N matchdays — the honest way to
    run a short test, because a real 2026-27 Champions League is eight matchdays
    spread over five months and no one wants to simulate all of it to check the
    plumbing.

    The first date of each published window is the round's target, and
    ``slip_days`` lets a congested round move to the next real day of the SAME
    window — UEFA genuinely plays a matchday across two or three evenings
    (matchday 1 is 8, 9 OR 10 September), and forcing every tie onto the first
    date would invent collisions the real competition does not have.
    """
    rounds = CL_LEAGUE_PHASE[:upto] if upto else CL_LEAGUE_PHASE
    dates = tuple(window[0] for _label, window in rounds)
    windows = tuple(tuple(window) for _label, window in rounds)
    return SchedulePattern(
        name="champions_league",
        weekdays=(TUE, WED, THU),
        kickoffs=("18:45", "21:00"),
        min_rest_days=4,
        priority=priority,
        fixed_round_dates=dates,
        round_date_windows=windows,
        slip_days=slip_days,
    )


def champions_league_matchday_dates(matchday: int) -> Tuple[date, ...]:
    """The real date window for a league-phase matchday (1-indexed)."""
    return CL_LEAGUE_PHASE[matchday - 1][1]


# ─────────────────────────────────────────────
# EFL CUP 2026-27 (Carabao Cup)
# ─────────────────────────────────────────────

EFL_CUP_ROUNDS: Tuple[Tuple[str, Tuple[date, ...]], ...] = (
    ("Preliminary round", (date(2026, 8, 1), date(2026, 8, 2))),
    ("Round One", (date(2026, 8, 7), date(2026, 8, 8), date(2026, 8, 9))),
    ("Round Two", (date(2026, 8, 24), date(2026, 8, 25), date(2026, 8, 26),
                   date(2026, 8, 27))),
    ("Round Three", (date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10),
                     date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17))),
    ("Round Four", (date(2026, 10, 26), date(2026, 10, 27), date(2026, 10, 28),
                    date(2026, 10, 29))),
)
EFL_CUP_FINAL = date(2027, 3, 21)       # Sunday, Manchester City defending


def efl_cup_rules() -> CompetitionRules:
    """Single-leg knockout, extra time and penalties from the fourth round."""
    return CompetitionRules(
        type="cup",
        schedule_kind="knockout",
        extra_time=True,
        penalties=True,
        aggregate=False,
        away_goals=False,
        legs=1,
        double_round_robin=False,
        substitutions_max=5,
        max_bench=9,
        suspension_type="yellow_accum",
        suspension_threshold=5,
        suspension_window=6,
    )


def efl_cup_pattern(round_name: str = "Round Four", *,
                    priority: int = 200) -> SchedulePattern:
    """The real published dates for one EFL Cup round."""
    for label, window in EFL_CUP_ROUNDS:
        if label.lower() == round_name.lower():
            return SchedulePattern(
                name=f"efl_cup_{label.lower().replace(' ', '_')}",
                weekdays=(TUE, WED, THU),
                kickoffs=("19:45",),
                min_rest_days=3,
                priority=priority,
                fixed_round_dates=(window[0],),
                # the WHOLE published window, not a day-count slip: a tie may
                # fall on any evening the EFL announced and on no other
                round_date_windows=(tuple(window),),
            )
    raise KeyError(f"unknown EFL Cup round {round_name!r}; known: "
                   f"{', '.join(l for l, _ in EFL_CUP_ROUNDS)}")


# ─────────────────────────────────────────────
# FA CUP 2026-27
# ─────────────────────────────────────────────

FA_CUP_ROUNDS: Tuple[Tuple[str, date], ...] = (
    ("Extra Preliminary Round", date(2026, 8, 8)),
    ("Preliminary Round", date(2026, 8, 22)),
    ("First Round Qualifying", date(2026, 9, 5)),
    ("Second Round Qualifying", date(2026, 9, 19)),
    ("Third Round Qualifying", date(2026, 10, 3)),
    ("Fourth Round Qualifying", date(2026, 10, 17)),
    ("First Round", date(2026, 10, 10)),
    ("Second Round", date(2026, 12, 5)),
    ("Third Round", date(2027, 1, 9)),
    ("Fourth Round", date(2027, 2, 13)),
    ("Fifth Round", date(2027, 3, 6)),
)


def fa_cup_rules() -> CompetitionRules:
    return cup_rules(substitutions_max=5, max_bench=9,
                     away_goals=False, aggregate=False)


def fa_cup_pattern(round_name: str, *, priority: int = 200) -> SchedulePattern:
    """The real published date for one FA Cup round."""
    for label, when in FA_CUP_ROUNDS:
        if label.lower() == round_name.lower():
            return SchedulePattern(
                name=f"fa_cup_{label.lower().replace(' ', '_')}",
                weekdays=(SAT,),
                kickoffs=("15:00",),
                min_rest_days=4,
                priority=priority,
                fixed_round_dates=(when,),
                slip_days=4,      # replays and TV moves are normal
            )
    raise KeyError(f"unknown FA Cup round {round_name!r}; known: "
                   f"{', '.join(l for l, _ in FA_CUP_ROUNDS)}")


# ─────────────────────────────────────────────
# COMMUNITY SHIELD 2026-27
# ─────────────────────────────────────────────

COMMUNITY_SHIELD = date(2026, 8, 16)    # Sunday


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def season_summary() -> str:
    lines = [
        f"PLOFA real-world reference data — {SEASON}",
        "  Premier League : "
        f"{PL_MATCHDAYS} matchdays, {PL_TEAMS} clubs, "
        f"{PL_RELEGATION_SLOTS} relegated",
        f"    season {PL_SEASON_START} -> {PL_SEASON_END} "
        f"({PL_WEEKEND_ROUNDS} weekend + {PL_MIDWEEK_ROUNDS} midweek rounds)",
        f"    fixture release {PL_FIXTURE_RELEASE}, "
        f"{len(PL_INTERNATIONAL_BREAKS)} international breaks",
        f"  Champions League: {CL_TEAMS} clubs, {CL_MATCHDAYS} league-phase "
        f"matchdays, top {CL_DIRECT_TO_R16} direct to the R16",
        f"    league phase {CL_LEAGUE_PHASE[0][1][0]} -> "
        f"{CL_LEAGUE_PHASE[-1][1][0]}",
        f"    knockout play-offs {CL_PLAYOFFS[0][0]}, final {CL_FINAL} "
        f"({CL_FINAL_VENUE})",
        f"  EFL Cup         : {len(EFL_CUP_ROUNDS)} rounds + final "
        f"{EFL_CUP_FINAL}",
        f"  FA Cup          : {len(FA_CUP_ROUNDS)} rounds, "
        f"{FA_CUP_ROUNDS[-1][1]} fifth round",
        f"  Community Shield: {COMMUNITY_SHIELD}",
    ]
    return "\n".join(lines)
