"""
PLOFA WORLD — the controlled test world.
========================================
world/testworld.py  ·  Audit §16 phase 10 / plan §17.

Plan §17: *"Create a small controlled test environment... 2 countries, 1–2
divisions each, 8–12 clubs, realistic squads, domestic league, domestic cup,
Champions League."*

Built on REAL rules and REAL dates
----------------------------------
An earlier version of this world invented its own kickoff times, rest periods
and matchday spacing. That proves the plumbing works and nothing else — it says
nothing about whether the rules are right, and matching real football is the
entire purpose of the simulator. This version takes its rules and its fixture
dates from :mod:`world.realworld`, which holds the published 2026-27 Premier
League, Champions League, EFL Cup and FA Cup calendars.

So the test now asks a genuinely interesting question: **can the scheduler
honour the real football calendar?** Real 2026-27 is congested — the Champions
League plays on midweek dates that sit in the middle of Premier League rounds,
and a cup round spans several evenings. If the world layer can place a real
month of real fixtures without a single collision, that is worth knowing.

The one compromise, stated plainly
----------------------------------
The competition structure, rules and dates are **real**. The clubs are the 18
real squads from the 26/27 Excel roster, which is fewer than a real Premier
League's 20 and far fewer than the Champions League's 36. A test world with
invented players could not run a real match, and running a real match is the
point — so the map is the fiction, the football is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from world import realworld as rw
from world.calendar import (
    Calendar,
    SchedulePattern,
    continental_midweek,
    cup_midweek,
    league_saturday,
)
from world.competition import (
    CompetitionType,
    KnockoutCompetition,
    LeagueCompetition,
)
from world.ids import mint_id

# ─────────────────────────────────────────────
# THE TEST WORLD
# ─────────────────────────────────────────────

COUNTRY = "CTR-ENG"
COUNTRY_NAME = "England"
DOMESTIC_LEAGUE = "PL-2026-27"
CHAMPIONS_LEAGUE = "UCL-2026-27"
EFL_CUP = "EFL-CUP-2026-27"
FA_CUP = "FA-CUP-2026-27"

#: The 18 real clubs in the 26/27 roster. A real Premier League has 20; the
#: shortfall is stated rather than padded with invented clubs.
ROSTER_CLUBS = 18


@dataclass
class TestWorld:
    """A built test world: clubs, competitions and a calendar over them."""

    clubs: List[str] = field(default_factory=list)
    #: canonical club id -> the real roster name behind it. The calendar keys on
    #: IDs (plan §23) but ``roster_loader`` and the match engine speak names.
    roster: Dict[str, str] = field(default_factory=dict)
    league: Optional[LeagueCompetition] = None
    cup: Optional[KnockoutCompetition] = None
    continental: Optional[LeagueCompetition] = None
    calendar: Optional[Calendar] = None

    @property
    def all_clubs(self) -> List[str]:
        return list(self.clubs)

    def name_of(self, club_id: str) -> str:
        return self.roster.get(club_id, club_id)

    def competition_ids(self) -> List[str]:
        return [c.competition_id for c in (self.league, self.cup, self.continental)
                if c is not None]

    def summary(self) -> str:
        lines = [
            "PLOFA test world - REAL 2026-27 rules and dates",
            f"  {len(self.clubs)} real clubs from the 26/27 roster "
            f"(a real Premier League has {rw.PL_TEAMS})",
            f"  {DOMESTIC_LEAGUE:<20} {rw.PL_MATCHDAYS} matchdays, "
            f"{rw.PL_RELEGATION_SLOTS} relegated",
            f"  {CHAMPIONS_LEAGUE:<20} league phase, {rw.CL_MATCHDAYS} "
            f"matchdays, top {rw.CL_DIRECT_TO_R16} direct to the R16",
            f"  {EFL_CUP:<20} knockout, final {rw.EFL_CUP_FINAL}",
        ]
        if self.calendar is not None:
            lines.append(
                f"  calendar: {len(self.calendar.fixtures())} fixture(s), "
                f"{len(self.calendar.violations())} violation(s)")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# BUILDING
# ─────────────────────────────────────────────

def _club_ids(names: Sequence[str]) -> List[str]:
    """Canonical IDs for the real clubs, qualified by country so an identically
    named club in another country stays a different identity (plan §23)."""
    return [mint_id("club", f"{COUNTRY_NAME}::{n}") for n in names]


def build_test_world(
    start: Optional[date] = None,
    *,
    seed: int = 17,
    league_matchdays: int = 4,
    continental_matchdays: int = 1,
    cup_round: str = "Round Three",
    horizon_days: int = 200,
) -> TestWorld:
    """Assemble the test world from the live roster and the REAL calendar.

    ``start`` defaults to the real 2026-27 Premier League opening date.
    ``league_matchdays`` / ``continental_matchdays`` truncate each competition
    to a few real rounds so the month is a month and not a season.
    """
    from roster_loader import get_loader

    names = sorted(get_loader().get_all_clubs())
    if len(names) < 8:
        raise ValueError(
            f"the test world needs at least 8 clubs, roster has {len(names)}")

    world = TestWorld(clubs=_club_ids(names))
    world.roster = {cid: n for cid, n in zip(world.clubs, names)}
    when = start or rw.PL_MAIN_START

    # ── domestic league: REAL Premier League rules ──
    world.league = LeagueCompetition(
        competition_id=DOMESTIC_LEAGUE, name="Premier League 2026-27",
        country_id=COUNTRY, season_id=rw.SEASON,
        rules=rw.premier_league_rules(), participant_ids=world.clubs,
        seed=seed)
    world.league.make_fixtures(when, days_between=7)

    # ── Champions League: REAL rules and REAL published matchday dates ──
    world.continental = LeagueCompetition(
        competition_id=CHAMPIONS_LEAGUE, name="UEFA Champions League 2026-27",
        country_id="CTR-UEFA", season_id=rw.SEASON,
        rules=rw.champions_league_rules(), participant_ids=world.clubs,
        seed=seed)
    # anchor the CL on its REAL first matchday, not on the league's opening day
    world.continental.make_fixtures(rw.CL_LEAGUE_PHASE[0][1][0],
                                    days_between=7)

    # ── EFL Cup: REAL rules and REAL round dates ──
    world.cup = KnockoutCompetition(
        competition_id=EFL_CUP, name="Carabao Cup 2026-27",
        country_id=COUNTRY, season_id=rw.SEASON,
        rules=rw.efl_cup_rules(), participant_ids=world.clubs, seed=seed)
    # the opening round of the bracket, whatever its real name is (18 clubs pad
    # to 32 slots, so this is the "Round of 32")
    opening_round = world.cup.round_names()[0]
    world.cup.draw_round(opening_round)

    # keep the whole world inside roughly one month
    _truncate(world.league, league_matchdays)
    _truncate(world.continental, continental_matchdays)

    calendar = Calendar(country_id=COUNTRY, season_id=rw.SEASON, start=when,
                        horizon_days=horizon_days)
    calendar.add_league(world.league, rw.premier_league_pattern())
    calendar.add_league(world.continental,
                        rw.champions_league_pattern(upto=continental_matchdays))
    calendar.add_knockout_round(world.cup, rw.efl_cup_pattern(cup_round),
                                opening_round, stage_name="Knockout")
    calendar.build()
    world.calendar = calendar
    return world


def _truncate(competition, matchdays: int) -> None:
    """Cut a league down to its first ``matchdays`` real rounds, in place.

    A full 18-club double round robin is 34 matchdays. Left in the calendar it
    saturates every midweek slot, so the cup tie has nowhere legal to go until
    the league season ends — which pushes the "one-month" test months apart. The
    window, not the rules, is what makes this a month.
    """
    from season_manager import FixtureList

    fl = competition._fixtures
    if fl is None or not fl.fixtures:
        return
    keep = [f for f in fl.fixtures if f.matchday <= matchdays]
    competition._fixtures = FixtureList(keep)


# ─────────────────────────────────────────────
# THE REAL MONTH
# ─────────────────────────────────────────────

def month_fixtures(world: TestWorld, *, days: int = 31) -> List:
    """The controlled one-month window (plan §18), taken from the REAL calendar.

    Plan §18's shape, per club:

        weekend  domestic league
        midweek  Champions League
        weekend  domestic league
        midweek  domestic cup

    The window is a real date range starting at the Champions League's published
    opening matchday, so what gets played is what UEFA and the Premier League
    actually scheduled — not a synthetic approximation of it.
    """
    calendar = world.calendar
    start = rw.CL_LEAGUE_PHASE[0][1][0]
    from datetime import timedelta
    end = start + timedelta(days=days)

    out = [f for f in calendar.fixtures() if start <= f.match_date <= end]
    out.sort(key=lambda f: (f.match_date, f.kickoff, f.fixture_id))
    return out
