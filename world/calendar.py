"""
PLOFA WORLD — multi-competition calendar.
=========================================
world/calendar.py  ·  Phase 4 of the World Football plan.

One country/season, many competitions, one hard scheduling invariant. Plan §12:

    "The world must support clubs participating in multiple competitions
     simultaneously... The system must not accidentally schedule the same team
     to play two matches at the same time."

This module owns *when* a match happens, never *what happens in it*. It plans
fixtures and nothing else — the football still runs through the one
``MatchEngine`` (plan §10). Its output is a dated, conflict-free list of
:class:`WorldFixture`, each of which can hand a
:class:`~world.competition.MatchContextFragment` to the engine.

The invariants it guarantees
----------------------------
1. **No club plays twice on one calendar day.** This is the hard invariant and
   is enforced by construction in :meth:`Calendar.build`, then re-proved
   independently by :meth:`Calendar.validate` (audit §18 risk 3: an invariant
   enforced by convention is not enforced at all).
2. **Minimum rest is respected.** Default 3 days between any two of a club's
   fixtures, overridable per competition.
3. **One venue, one match, one day.** A stadium is a resource like a squad.
4. **Contested dates go to the higher-priority competition.** Cup and continental
   football cannot be squeezed out by the league simply because the league
   fixture list was generated first.
5. **Fully deterministic.** Placement order is a total order over stable keys,
   so a season calendar is reproducible from its inputs with no dependence on
   dict/set iteration order (the ``PYTHONHASHSEED`` trap of README §8) and no
   RNG at all.

Scope guardrails, same as the rest of the package:
  * no engine import at module load; the engine is reached only at call time
  * no writes to authoritative 26/27 ledgers
  * postponement/rescheduling is expressed, never silently applied
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ─────────────────────────────────────────────
# WEEKDAYS
# ─────────────────────────────────────────────
# date.weekday(): Monday=0 … Sunday=6. Named here so patterns read as English
# in world data files instead of as magic integers.
MON, TUE, WED, THU, FRI, SAT, SUN = range(7)
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

DEFAULT_MIN_REST_DAYS = 3
DEFAULT_HORIZON_DAYS = 400


# ─────────────────────────────────────────────
# VIOLATIONS
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Violation:
    """A broken invariant, reported rather than raised.

    A season that cannot be scheduled perfectly is a normal state of the world
    (a postponed cup tie, a congested festive period). The calendar's job is to
    *tell you exactly where*, so the caller decides what to do about it.
    """

    kind: str
    detail: str
    fixture_ids: Tuple[str, ...] = ()
    club_ids: Tuple[str, ...] = ()

    def __str__(self) -> str:
        who = f" [{', '.join(self.club_ids)}]" if self.club_ids else ""
        return f"{self.kind}: {self.detail}{who}"


# ─────────────────────────────────────────────
# SCHEDULE PATTERN (data)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class SchedulePattern:
    """How one competition wants to be scheduled. Pure data — plan §7.

    ``weekdays``      which days a round may land on (first match wins if the
                      pattern allows several)
    ``kickoffs``      kickoff times, assigned round-robin within a matchday
    ``round_interval_days``  spacing between consecutive rounds of this
                      competition (a Saturday league is 7; a cup is "whenever")
    ``min_rest_days``  rest floor for a club inside *this* competition
    ``priority``      who wins a contested date — higher wins
    """

    name: str
    weekdays: Tuple[int, ...] = (SAT,)
    kickoffs: Tuple[str, ...] = ("15:00",)
    round_interval_days: int = 7
    min_rest_days: int = DEFAULT_MIN_REST_DAYS
    priority: int = 100
    #: REAL fixture dates, one per round. When supplied, round N is placed on
    #: ``fixed_round_dates[N - 1]`` and ``weekdays`` / ``round_interval_days``
    #: are ignored. This is what lets a real published calendar drive the
    #: scheduler instead of a synthetic approximation — the Champions League
    #: plays on fixed midweek dates spread over irregular gaps, which no
    #: weekday-plus-interval pattern can express.
    #:
    #: A round that cannot take its exact date may slip up to ``slip_days``
    #: forward, because fixtures really do move. Anything beyond that is
    #: reported as ``unscheduled`` rather than quietly rescheduled somewhere
    #: arbitrary.
    fixed_round_dates: Tuple[date, ...] = ()
    slip_days: int = 3
    #: The FULL published date window per round, when one exists. Real
    #: competitions do not publish a single date per round — UEFA publishes
    #: "8-10 September" and the EFL publishes a Tuesday/Wednesday/Thursday
    #: batch. ``fixed_round_dates`` holds each round's first day; this holds
    #: every day that round may legally fall on.
    #:
    #: It matters: a day-count slip invents dates the competition never
    #: announced. A cup round published for 8, 9, 10, 15, 16, 17 September must
    #: not drift to the Saturday in between, because no tie is ever played then.
    round_date_windows: Tuple[Tuple[date, ...], ...] = ()

    def __post_init__(self) -> None:
        for wd in self.weekdays:
            if not 0 <= wd <= 6:
                raise ValueError(f"weekday must be 0-6, got {wd}")
        if not self.kickoffs:
            raise ValueError("a schedule pattern needs at least one kickoff")
        if self.round_interval_days < 1:
            raise ValueError("round_interval_days must be >= 1")
        if self.min_rest_days < 0:
            raise ValueError("min_rest_days must be >= 0")
        if self.slip_days < 0:
            raise ValueError("slip_days must be >= 0")
        for d in self.fixed_round_dates:
            if not isinstance(d, date):
                raise TypeError(
                    f"fixed_round_dates must hold dates, got {type(d).__name__}")
        for i, window in enumerate(self.round_date_windows):
            if not window:
                raise ValueError(f"round {i + 1} has an empty date window")
            for d in window:
                if not isinstance(d, date):
                    raise TypeError(
                        f"round_date_windows must hold dates, got "
                        f"{type(d).__name__}")
        if (self.fixed_round_dates and self.round_date_windows
                and len(self.fixed_round_dates) != len(self.round_date_windows)):
            raise ValueError(
                f"fixed_round_dates has {len(self.fixed_round_dates)} entries "
                f"but round_date_windows has {len(self.round_date_windows)}")

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["weekdays"] = list(self.weekdays)
        d["kickoffs"] = list(self.kickoffs)
        d["fixed_round_dates"] = [d_.isoformat() for d_ in self.fixed_round_dates]
        d["round_date_windows"] = [[w.isoformat() for w in window]
                                   for window in self.round_date_windows]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SchedulePattern":
        data = dict(d)
        data["weekdays"] = tuple(data.get("weekdays", (SAT,)))
        data["kickoffs"] = tuple(data.get("kickoffs", ("15:00",)))
        data["fixed_round_dates"] = tuple(
            date.fromisoformat(s) for s in data.get("fixed_round_dates", ()))
        data["round_date_windows"] = tuple(
            tuple(date.fromisoformat(x) for x in window)
            for window in data.get("round_date_windows", ()))
        return cls(**{k: v for k, v in data.items()
                      if k in _fields(cls)})


def league_saturday(**overrides: Any) -> SchedulePattern:
    """A weekend league: Saturday 15:00, one round a week."""
    base = dict(name="league_saturday", weekdays=(SAT,), kickoffs=("15:00",),
                round_interval_days=7, min_rest_days=3, priority=100)
    base.update(overrides)
    return SchedulePattern(**base)


def cup_midweek(**overrides: Any) -> SchedulePattern:
    """A domestic cup: midweek, late kickoff, outranks the league for a date."""
    base = dict(name="cup_midweek", weekdays=(TUE, WED), kickoffs=("19:45",),
                round_interval_days=7, min_rest_days=4, priority=200)
    base.update(overrides)
    return SchedulePattern(**base)


def continental_midweek(**overrides: Any) -> SchedulePattern:
    """Continental football: Wednesday, highest priority of all."""
    base = dict(name="continental_midweek", weekdays=(WED,), kickoffs=("20:00",),
                round_interval_days=7, min_rest_days=4, priority=300)
    base.update(overrides)
    return SchedulePattern(**base)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PlannedFixture:
    """A fixture that wants a date but does not have one yet.

    Produced by a competition (``LeagueCompetition`` fixtures or
    ``KnockoutCompetition`` ties) and handed to the calendar, which is the only
    thing that decides when it is played.
    """

    home_id: str
    away_id: str
    round_name: str = ""
    stage_name: str = ""
    leg: Optional[str] = None
    matchday: int = 0
    neutral_venue: str = ""        # set for a final played at a neutral ground

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class WorldFixture:
    """A scheduled fixture — a planned fixture plus its resolved slot."""

    fixture_id: str
    competition_id: str
    competition_type: str
    planned: PlannedFixture
    match_date: date
    kickoff: str
    venue: str
    priority: int = 100
    stage_name: str = ""
    round_name: str = ""
    leg: Optional[str] = None
    postponed_from: Optional[str] = None      # ISO date, if moved
    postpone_count: int = 0

    @property
    def home_id(self) -> str:
        return self.planned.home_id

    @property
    def away_id(self) -> str:
        return self.planned.away_id

    @property
    def club_ids(self) -> Tuple[str, str]:
        return (self.home_id, self.away_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "competition_id": self.competition_id,
            "competition_type": self.competition_type,
            "home_id": self.home_id,
            "away_id": self.away_id,
            "match_date": self.match_date.isoformat(),
            "kickoff": self.kickoff,
            "venue": self.venue,
            "priority": self.priority,
            "stage_name": self.stage_name or self.planned.stage_name,
            "round_name": self.round_name or self.planned.round_name,
            "leg": self.leg,
            "matchday": self.planned.matchday,
            "postponed_from": self.postponed_from,
            "postpone_count": self.postpone_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorldFixture":
        data = dict(d)
        planned = PlannedFixture(
            home_id=data.pop("home_id"),
            away_id=data.pop("away_id"),
            round_name=data.pop("round_name", ""),
            stage_name=data.pop("stage_name", ""),
            leg=data.pop("leg", None),
            matchday=data.pop("matchday", 0),
            neutral_venue=data.pop("neutral_venue", ""),
        )
        data["planned"] = planned
        data["match_date"] = date.fromisoformat(data["match_date"])
        return cls(**{k: v for k, v in data.items() if k in _fields(cls)})


# ─────────────────────────────────────────────
# COMPETITION ENTRY
# ─────────────────────────────────────────────

@dataclass
class CompetitionEntry:
    """One competition's contribution to the calendar."""

    competition_id: str
    competition_type: str
    pattern: SchedulePattern
    fixtures: List[PlannedFixture] = field(default_factory=list)
    competition: Any = None        # optional back-reference (never serialised)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "competition_id": self.competition_id,
            "competition_type": self.competition_type,
            "pattern": self.pattern.to_dict(),
            "fixtures": [f.to_dict() for f in self.fixtures],
        }


# ─────────────────────────────────────────────
# CALENDAR
# ─────────────────────────────────────────────

class Calendar:
    """A country/season calendar assembled from many competitions.

    Usage::

        cal = Calendar(country_id="CTR-TOL", season_id="26/27",
                       start=date(2026, 8, 8))
        cal.add_competition(league_entry)
        cal.add_competition(cup_entry)
        cal.build()
        assert not cal.violations_of_kind("same_day_double_booking")
    """

    def __init__(
        self,
        country_id: str,
        season_id: str,
        start: date,
        *,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        default_min_rest_days: int = DEFAULT_MIN_REST_DAYS,
        venue_resolver: Optional[Callable[[str], str]] = None,
    ) -> None:
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        self.country_id = country_id
        self.season_id = season_id
        self.start = start
        self.horizon_days = horizon_days
        self.default_min_rest_days = default_min_rest_days
        self._venue = venue_resolver or (lambda club_id: f"{club_id} Stadium")
        self._entries: List[CompetitionEntry] = []
        self._fixtures: List[WorldFixture] = []
        self._by_id: Dict[str, WorldFixture] = {}
        self._violations: List[Violation] = []
        #: rounds that could not take their published date and moved. Kept
        #: separate from violations: a moved fixture is legal football, and
        #: burying it in the violation list would make a real congested
        #: calendar look broken.
        self._slipped: List[Violation] = []

    # -- assembly -------------------------------------------------------

    def add_competition(
        self,
        competition_id: str,
        competition_type: str,
        pattern: SchedulePattern,
        fixtures: Iterable[PlannedFixture],
        *,
        competition: Any = None,
    ) -> "CompetitionEntry":
        entry = CompetitionEntry(
            competition_id=competition_id,
            competition_type=competition_type,
            pattern=pattern,
            fixtures=list(fixtures),
            competition=competition,
        )
        self._entries.append(entry)
        return entry

    def add_league(self, competition: Any, pattern: SchedulePattern) -> CompetitionEntry:
        """Add a :class:`~world.competition.LeagueCompetition` (fixtures first)."""
        if not competition._fixtures or not competition._fixtures.fixtures:
            raise ValueError(
                f"league {competition.competition_id} has no fixtures; "
                f"call make_fixtures() before adding it to a calendar"
            )
        return self.add_competition(
            competition.competition_id,
            competition.rules.type,
            pattern,
            [
                PlannedFixture(
                    home_id=f.home_team, away_id=f.away_team,
                    round_name=f"MD {f.matchday}", stage_name="League",
                    matchday=f.matchday,
                )
                for f in competition._fixtures.fixtures
            ],
            competition=competition,
        )

    def add_knockout_round(
        self,
        competition: Any,
        pattern: SchedulePattern,
        round_name: str,
        *,
        stage_name: str = "Knockout",
    ) -> CompetitionEntry:
        """Add one drawn round of a :class:`~world.competition.KnockoutCompetition`."""
        ties = competition.ties_in(round_name)
        if not ties:
            raise ValueError(f"round {round_name!r} has not been drawn")
        return self.add_competition(
            competition.competition_id,
            competition.rules.type,
            pattern,
            [
                PlannedFixture(
                    home_id=t.home_id, away_id=t.away_id,
                    round_name=round_name, stage_name=stage_name, leg=t.leg,
                )
                for t in ties
            ],
            competition=competition,
        )

    # -- placement ------------------------------------------------------

    def _eligible_dates(self, pattern: SchedulePattern, anchor: date) -> List[date]:
        """Every date in the horizon on which this pattern may stage a round."""
        out: List[date] = []
        d = anchor
        end = self.start + timedelta(days=self.horizon_days)
        while d <= end:
            if d.weekday() in pattern.weekdays:
                out.append(d)
            d += timedelta(days=1)
        return out

    def _round_anchor(self, pattern: SchedulePattern, matchday: int) -> date:
        """Where round ``matchday`` wants to sit in the season.

        If the pattern carries REAL published fixture dates, round N's anchor is
        its exact date — that is the whole point of ``fixed_round_dates``.
        Otherwise the anchor only *biases* placement, so a later round is used
        when the preferred week is already taken by higher-priority
        competition. That keeps a generated calendar readable without ever
        letting readability break an invariant.
        """
        if pattern.fixed_round_dates:
            if matchday <= 0:
                return pattern.fixed_round_dates[0]
            if matchday <= len(pattern.fixed_round_dates):
                return pattern.fixed_round_dates[matchday - 1]
            # more rounds were planned than dates were published: fall back to
            # the last published date and let the slip window carry it
            return pattern.fixed_round_dates[-1]
        if matchday <= 0:
            return self.start
        return self.start + timedelta(days=pattern.round_interval_days * (matchday - 1))

    def _candidates(self, pattern: SchedulePattern, matchday: int,
                    anchor: date) -> List[date]:
        """Dates this round may legally fall on, best first.

        With a published window, ONLY those dates are candidates. A real
        competition announces a set of days ("8-10 September", or a
        Tuesday/Wednesday/Thursday batch) and no tie is ever played on any
        other. Allowing an arbitrary day-count slip from the anchor invents
        fixture dates that were never announced — and silently, which is the
        worst way to be wrong about a published calendar.

        When there is no window, the anchor is extended by ``slip_days`` as a
        generic fallback.
        """
        end = self.start + timedelta(days=self.horizon_days)
        if pattern.round_date_windows:
            if 1 <= matchday <= len(pattern.round_date_windows):
                window = pattern.round_date_windows[matchday - 1]
            else:
                window = pattern.round_date_windows[-1]
            return [d for d in sorted(window) if d <= end]
        if pattern.fixed_round_dates:
            out: List[date] = []
            for offset in range(pattern.slip_days + 1):
                d = anchor + timedelta(days=offset)
                if d <= end:
                    out.append(d)
            return out
        return [d for d in self._eligible_dates(pattern, anchor)
                if d >= max(anchor, self.start)]

    def _club_dates(self, club_id: str) -> Dict[date, List[WorldFixture]]:
        out: Dict[date, List[WorldFixture]] = {}
        for fx in self._fixtures:
            if club_id in fx.club_ids:
                out.setdefault(fx.match_date, []).append(fx)
        return out

    def _rest_ok(self, club_id: str, when: date, competition_id: str) -> bool:
        """Can ``club_id`` play on ``when`` without breaching any rest floor?

        The floor for a gap is the strictest claim on that club: BOTH the
        competition being placed and the competition of the fixture it would sit
        next to. A cup that demands 10 days must be honoured even when the
        fixture crowding in beside it belongs to a lenient league — otherwise
        priority ordering silently defeats the stricter rule.
        """
        incoming = self._min_rest_for(competition_id)
        for other_date, others in self._club_dates(club_id).items():
            floor = incoming
            for fx in others:
                floor = max(floor, self._min_rest_for(fx.competition_id))
            if abs((when - other_date).days) < floor:
                return False
        return True

    def _venue_free(self, venue: str, when: date) -> bool:
        return not any(
            fx.venue == venue and fx.match_date == when for fx in self._fixtures
        )

    def _order_key(self, entry: CompetitionEntry, planned: PlannedFixture) -> Tuple:
        """Total order for placement. Stable across runs and processes.

        Deliberately NOT a dict/set iteration order: priority first (so the
        cup and continental football claim their dates before the league),
        then round, then a lexicographic tiebreak. Equal keys cannot occur
        because a competition never plans the same pairing twice in a round.
        """
        return (
            -entry.pattern.priority,
            entry.competition_id,
            planned.matchday,
            planned.round_name,
            planned.home_id,
            planned.away_id,
        )

    def _kickoff_for(self, pattern: SchedulePattern, seq: int) -> str:
        return pattern.kickoffs[seq % len(pattern.kickoffs)]

    def build(self) -> List[WorldFixture]:
        """Place every planned fixture. Deterministic and idempotent.

        Placement is greedy in priority order over a finite horizon. A fixture
        that cannot be placed is reported as a violation and left unscheduled —
        never silently dropped and never allowed to violate an invariant.
        """
        self._fixtures = []
        self._by_id = {}
        self._violations = []
        self._slipped = []

        work: List[Tuple[Tuple, CompetitionEntry, PlannedFixture]] = []
        for entry in self._entries:
            for planned in entry.fixtures:
                work.append((self._order_key(entry, planned), entry, planned))
        work.sort(key=lambda item: item[0])

        seq_by_date: Dict[Tuple[date, str], int] = {}
        counter = 0
        for _key, entry, planned in work:
            pattern = entry.pattern
            anchor = self._round_anchor(pattern, planned.matchday)
            venue = planned.neutral_venue or self._venue(planned.home_id)

            candidates = self._candidates(pattern, planned.matchday, anchor)
            chosen: Optional[date] = None
            slipped = False
            for when in candidates:
                if not self._venue_free(venue, when):
                    continue
                if any(not self._rest_ok(c, when, entry.competition_id)
                       for c in planned_clubs(planned)):
                    continue
                chosen = when
                slipped = (when != anchor)
                break

            if chosen is None:
                detail = (f"{entry.competition_id} {planned.round_name}: "
                          f"{planned.home_id} v {planned.away_id} found no "
                          f"legal slot")
                if pattern.fixed_round_dates:
                    detail += (f" on or near its published date "
                               f"{anchor.isoformat()}")
                else:
                    detail += f" in {pattern.name} within {self.horizon_days} days"
                self._violations.append(Violation(
                    kind="unscheduled", detail=detail,
                    club_ids=planned_clubs(planned)))
                continue
            if slipped:
                # A round that moved off its published date is legal — fixtures
                # really do shift — but it is information, not a violation, so
                # it is reported rather than silently accepted.
                self._slipped.append(Violation(
                    kind="slipped",
                    detail=(f"{entry.competition_id} {planned.round_name}: "
                            f"published {anchor.isoformat()}, played "
                            f"{chosen.isoformat()}"),
                    fixture_ids=(), club_ids=planned_clubs(planned)))

            slot = (chosen, entry.competition_id)
            seq = seq_by_date.get(slot, 0)
            seq_by_date[slot] = seq + 1
            counter += 1
            fx = WorldFixture(
                fixture_id=f"FX-{self.season_id}-{counter:04d}",
                competition_id=entry.competition_id,
                competition_type=entry.competition_type,
                planned=planned,
                match_date=chosen,
                kickoff=self._kickoff_for(pattern, seq),
                venue=venue,
                priority=pattern.priority,
                stage_name=planned.stage_name,
                round_name=planned.round_name,
                leg=planned.leg,
            )
            self._fixtures.append(fx)
            self._by_id[fx.fixture_id] = fx

        self._fixtures.sort(key=lambda f: (f.match_date, f.kickoff, f.fixture_id))
        self._violations.extend(self.validate())
        return list(self._fixtures)

    # -- invariant checking ---------------------------------------------

    def validate(self) -> List[Violation]:
        """Re-derive every invariant from the placed fixtures alone.

        Deliberately written *independently* of ``build`` so it can catch a bug
        in the placement logic itself. An invariant checked only by the code
        that enforces it is not checked.
        """
        out: List[Violation] = []

        by_club_day: Dict[Tuple[str, date], List[str]] = {}
        by_venue_day: Dict[Tuple[str, date], List[str]] = {}
        by_club: Dict[str, List[WorldFixture]] = {}

        for fx in self._fixtures:
            for club in fx.club_ids:
                by_club_day.setdefault((club, fx.match_date), []).append(fx.fixture_id)
                by_club.setdefault(club, []).append(fx)
            by_venue_day.setdefault((fx.venue, fx.match_date), []).append(fx.fixture_id)

        for (club, when), ids in sorted(by_club_day.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if len(ids) > 1:
                out.append(Violation(
                    kind="same_day_double_booking",
                    detail=f"{club} has {len(ids)} fixtures on "
                           f"{when.isoformat()} ({WEEKDAY_NAMES[when.weekday()]})",
                    fixture_ids=tuple(sorted(ids)),
                    club_ids=(club,),
                ))

        for (venue, when), ids in sorted(by_venue_day.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if len(ids) > 1:
                out.append(Violation(
                    kind="venue_double_booking",
                    detail=f"venue '{venue}' hosts {len(ids)} matches on "
                           f"{when.isoformat()}",
                    fixture_ids=tuple(sorted(ids)),
                ))

        for club, fxs in sorted(by_club.items()):
            fxs = sorted(fxs, key=lambda f: f.match_date)
            for a, b in zip(fxs, fxs[1:]):
                gap = (b.match_date - a.match_date).days
                # the stricter of the two competitions governs this gap
                floor = max(self._min_rest_for(a.competition_id),
                            self._min_rest_for(b.competition_id))
                if gap < floor:
                    out.append(Violation(
                        kind="insufficient_rest",
                        detail=f"{club}: {gap} day(s) between "
                               f"{a.match_date.isoformat()} and {b.match_date.isoformat()} "
                               f"(needs {floor})",
                        fixture_ids=(a.fixture_id, b.fixture_id),
                        club_ids=(club,),
                    ))
        return out

    def violations_of_kind(self, kind: str) -> List[Violation]:
        return [v for v in self._violations if v.kind == kind]

    def slipped(self) -> List[Violation]:
        """Rounds that moved off their published date. Legal, but worth seeing."""
        return list(self._slipped)

    def assert_clean(self) -> "Calendar":
        """Raise if any invariant is broken. For proof scripts and CI."""
        if self._violations:
            raise CalendarConflictError(
                "calendar violates its invariants:\n  "
                + "\n  ".join(str(v) for v in self._violations)
            )
        return self

    # -- postponement / rescheduling ------------------------------------

    def _violations_if_moved(self, fx: WorldFixture, new_date: date) -> List[Violation]:
        """What would break if ``fx`` were moved to ``new_date``? Mutates and
        restores ``fx`` so the trial is invisible."""
        previous = fx.match_date
        if previous == new_date:
            return []
        fx.match_date = new_date
        try:
            return self.validate()
        finally:
            fx.match_date = previous

    def reschedule(
        self,
        fixture_id: str,
        new_date: date,
        *,
        reason: str = "",
    ) -> WorldFixture:
        """Move an already-scheduled fixture to a specific date.

        The move is accepted only if it keeps every invariant; if it cannot, the
        fixture stays where it is and a violation is raised. A postponed match
        that quietly creates a double booking is worse than one that never
        moved, so this refuses rather than warns.
        """
        fx = self._by_id.get(fixture_id)
        if fx is None:
            raise KeyError(f"unknown fixture '{fixture_id}'")

        broken = self._violations_if_moved(fx, new_date)
        if broken:
            raise CalendarConflictError(
                f"cannot move {fixture_id} to {new_date.isoformat()}"
                + (f" ({reason})" if reason else "")
                + ":\n  " + "\n  ".join(str(v) for v in broken)
            )

        fx.postponed_from = fx.match_date.isoformat()
        fx.postpone_count += 1
        fx.match_date = new_date
        self._refresh_violations()
        return fx

    def postpone(
        self,
        fixture_id: str,
        *,
        days: int = 7,
        search_days: int = 120,
    ) -> WorldFixture:
        """Push a fixture back and let the calendar find the first legal slot.

        Plan §12 requires postponement to be a first-class operation. Merely
        rolling to the next matching weekday is not enough: in a league where
        every club plays every single week, *every* candidate date collides and
        a naive postponement can never succeed. So this searches forward from
        the requested delay for the first date that satisfies every invariant,
        and raises only when the whole search window is exhausted.
        """
        if days < 1:
            raise ValueError("days must be >= 1")
        if search_days < 1:
            raise ValueError("search_days must be >= 1")
        fx = self._by_id.get(fixture_id)
        if fx is None:
            raise KeyError(f"unknown fixture '{fixture_id}'")

        entry = self._entry_for(fx.competition_id)
        weekdays = entry.pattern.weekdays if entry else (fx.match_date.weekday(),)

        start = fx.match_date + timedelta(days=days)
        for offset in range(search_days + 1):
            when = start + timedelta(days=offset)
            if when.weekday() not in weekdays:
                continue
            if not self._violations_if_moved(fx, when):
                return self.reschedule(
                    fixture_id, when, reason=f"postponed {days}d"
                )
        raise CalendarConflictError(
            f"no legal slot for {fixture_id} between "
            f"{start.isoformat()} and {(start + timedelta(days=search_days)).isoformat()}"
        )

    def _refresh_violations(self) -> None:
        """Recompute invariant violations, keeping unscheduled reports.

        ``unscheduled`` is a record of the build, not of the current layout, so
        it survives a reschedule that may have freed a slot.
        """
        self._violations = [v for v in self._violations if v.kind == "unscheduled"]
        self._violations.extend(self.validate())
        self._fixtures.sort(key=lambda f: (f.match_date, f.kickoff, f.fixture_id))

    def _entry_for(self, competition_id: str) -> Optional[CompetitionEntry]:
        for entry in self._entries:
            if entry.competition_id == competition_id:
                return entry
        return None

    def _min_rest_for(self, competition_id: str) -> int:
        """The rest floor that applies to a competition's fixtures.

        A competition may demand MORE rest than the calendar default (a cup
        wants longer recoveries than a league), never less. Both ``build`` and
        ``validate`` must resolve the floor the same way — otherwise the
        independent check verifies a weaker rule than the placer enforced, and
        a genuine rest violation could pass unnoticed.
        """
        entry = self._entry_for(competition_id)
        if entry is None:
            return self.default_min_rest_days
        return max(entry.pattern.min_rest_days, self.default_min_rest_days)

    # -- queries ---------------------------------------------------------

    def fixtures(self) -> List[WorldFixture]:
        return list(self._fixtures)

    def fixture(self, fixture_id: str) -> WorldFixture:
        return self._by_id[fixture_id]

    def on(self, when: date) -> List[WorldFixture]:
        return [f for f in self._fixtures if f.match_date == when]

    def for_club(self, club_id: str) -> List[WorldFixture]:
        return sorted((f for f in self._fixtures if club_id in f.club_ids),
                      key=lambda f: f.match_date)

    def for_competition(self, competition_id: str) -> List[WorldFixture]:
        return [f for f in self._fixtures if f.competition_id == competition_id]

    def matchdays(self, competition_id: str) -> List[Tuple[date, List[WorldFixture]]]:
        """Fixtures grouped into dated rounds, earliest first."""
        by_date: Dict[date, List[WorldFixture]] = {}
        for f in self.for_competition(competition_id):
            by_date.setdefault(f.match_date, []).append(f)
        return [(d, sorted(by_date[d], key=lambda f: (f.kickoff, f.home_id)))
                for d in sorted(by_date)]

    def congestion(self, club_id: str, window_days: int = 14) -> List[Tuple[date, int]]:
        """Dates on which a club plays more than once inside ``window_days``.

        Not a violation — congestion is a real feature of a season and the
        thing a manager needs to see — but it must be *visible* (plan §12).
        """
        dates = sorted({f.match_date for f in self.for_club(club_id)})
        return [(d, sum(1 for x in dates if 0 <= (x - d).days < window_days))
                for d in dates]

    def violations(self) -> List[Violation]:
        return list(self._violations)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "country_id": self.country_id,
            "season_id": self.season_id,
            "start": self.start.isoformat(),
            "horizon_days": self.horizon_days,
            "default_min_rest_days": self.default_min_rest_days,
            "competitions": [e.to_dict() for e in self._entries],
            "fixtures": [f.to_dict() for f in self._fixtures],
        }

    def summary(self) -> str:
        # Deliberately ASCII-only: this is a report a human reads in whatever
        # console they happen to have open, and the default Windows console is
        # cp1252, where box-drawing glyphs arrive as '?'.
        lines = [
            f"PLOFA calendar - {self.country_id} {self.season_id}",
            f"  {len(self._entries)} competition(s), "
            f"{len(self._fixtures)} fixture(s), "
            f"{len(self._violations)} violation(s)",
        ]
        for entry in self._entries:
            n = len(self.for_competition(entry.competition_id))
            lines.append(
                f"  * {entry.competition_id:<22} {entry.pattern.name:<22} "
                f"{n:>4} fixture(s)  priority={entry.pattern.priority}"
            )
        if self._fixtures:
            span = (self._fixtures[-1].match_date
                    - self._fixtures[0].match_date).days
            lines.append(
                f"  {self._fixtures[0].match_date} -> "
                f"{self._fixtures[-1].match_date}  ({span} days)"
            )
        for v in self._violations[:10]:
            lines.append(f"  ! {v}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# ERRORS
# ─────────────────────────────────────────────

class CalendarConflictError(RuntimeError):
    """A scheduling request that cannot be honoured without breaking an
    invariant. Raised by ``assert_clean`` and ``reschedule``."""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def planned_clubs(planned: PlannedFixture) -> Tuple[str, str]:
    return (planned.home_id, planned.away_id)


def _fields(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}
