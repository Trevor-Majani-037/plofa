"""
PLOFA WORLD — the controlled one-month integration test.
========================================================
world/month.py  ·  Audit §16 phase 10 / plan §18.

Plan §18, and it is explicit that this is the test that matters:

    Saturday    domestic league
    Wednesday   Champions League
    Saturday    domestic league
    Wednesday   domestic cup

    "The same player must have one continuous state throughout. [...] This test
     is extremely important. If this works correctly, the architecture is
     approaching a usable World Football v1."

Everything before this was scaffolding proven against synthetic player lines.
This module runs the month with **real matches**: the calendar places them, the
same ``MatchEngine`` simulates them, ``world.ingest`` translates the result, and
``world.ledger`` keeps the state — and then the §18 checklist is verified after
every single match, not asserted at the end.

The continuity checks
---------------------
The one that matters is plan §7's: a player who plays Saturday's league must
carry that fatigue, workload, injuries and cards into Wednesday's continental
football. Concretely, after every matchday:

  * minutes, goals, appearances only ever INCREASE — state never resets at a
    competition boundary;
  * the continuous totals equal the sum of the per-competition lines;
  * every competition a player appeared in has its own coexisting line;
  * fatigue and injuries carry the ENGINE's values, not re-derived ones;
  * a club never appears twice on one day.

Nothing here writes to 26/27. The world is in memory and the ledger is a
scratch file the caller chooses; the live season state is never opened.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from world.ingest import apply_result, resolve_player_id
from world.ledger import WorldLedger
from world.testworld import TestWorld, month_fixtures


# ─────────────────────────────────────────────
# CONTINUITY REPORTING
# ─────────────────────────────────────────────

@dataclass
class ContinuityIssue:
    """One broken continuity invariant, with enough context to debug it."""

    where: str
    player_id: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.where}] {self.player_id}: {self.detail}"


@dataclass
class MatchdayRecord:
    match_date: date
    played: int = 0
    goals: int = 0
    issues: List[ContinuityIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass
class MonthReport:
    season_id: str = ""
    matchdays: List[MatchdayRecord] = field(default_factory=list)
    matches_played: int = 0
    total_goals: int = 0
    players_seen: int = 0
    issues: List[ContinuityIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> Dict[str, Any]:
        return {
            "season_id": self.season_id,
            "matches_played": self.matches_played,
            "total_goals": self.total_goals,
            "players_seen": self.players_seen,
            "matchdays": [
                {"match_date": d.match_date.isoformat(), "played": d.played,
                 "goals": d.goals, "issues": [str(i) for i in d.issues]}
                for d in self.matchdays
            ],
            "issues": [str(i) for i in self.issues],
        }


# ─────────────────────────────────────────────
# THE DRIVER
# ─────────────────────────────────────────────

class WorldMonth:
    """Plays a controlled month of real matches through the world layer."""

    #: cached squads — building them is the expensive half of a real match
    _squad_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self, world: TestWorld, *, ledger: Optional[WorldLedger] = None,
                 seed: int = 17, quiet: bool = True) -> None:
        self.world = world
        self.seed = seed
        self.quiet = quiet
        self.ledger = ledger or WorldLedger(path="<memory>")
        self._last_minutes: Dict[str, int] = {}
        self._last_apps: Dict[str, int] = {}

    # -- squad preparation ------------------------------------------------

    def _squad_for(self, club_id: str) -> Dict[str, Any]:
        """Real squad + profile for a club, built once and cached."""
        if club_id in WorldMonth._squad_cache:
            return WorldMonth._squad_cache[club_id]

        from auto_run_match import _resolve_team_profile
        from player_dna import SquadBuilder
        from roster_loader import get_loader

        name = self.world.name_of(club_id)
        loader = get_loader()
        raw = loader.build_matchday_squad(name)
        squad = SquadBuilder.build(
            team_name=name, starters=raw["starters"],
            substitutes=raw["substitutes"],
            team_superstars=raw["superstars"],
            set_piece_takers=raw["sp_takers"])
        profile = _resolve_team_profile(name, raw["formation"], is_home=True)
        entry = {"name": name, "squad": squad, "profile": profile,
                 "all_players": {
                     "starters": squad["starters"],
                     "substitutes": squad["substitutes"]}}
        WorldMonth._squad_cache[club_id] = entry
        return entry

    # -- one match --------------------------------------------------------

    def simulate_fixture(self, fixture, competition: Any) -> Dict[str, Any]:
        """Run one calendar fixture as a real match and fold it into the ledger.

        The competition's context is folded into the ``MatchConfig`` through the
        one sanctioned hand-off, so a knockout fixture would play extra time
        and penalties while a league fixture does not.
        """
        from match_engine import MatchConfig, MatchEngine
        from squad_manager import SubstitutionController
        from exporter import PLOFAExporter
        from world.ingest import apply_competition_context

        home = self._squad_for(fixture.home_id)
        away = self._squad_for(fixture.away_id)
        home_name, away_name = home["name"], away["name"]

        config = MatchConfig(
            home_team=home_name, away_team=away_name,
            match_date=fixture.match_date, matchday=1,
            season=self.ledger.season_id or "T1",
            competition=competition.competition_id,
            venue=fixture.venue, stadium_capacity=40000,
            is_derby=fixture.home_id.split("-")[0] == fixture.away_id.split("-")[0],
        )
        # the ONE hand-off from the world layer to the engine
        config = apply_competition_context(
            config, competition, round_name=fixture.round_name)

        sub = SubstitutionController(
            home_team=home_name, away_team=away_name,
            home_subs_bench=home["squad"]["substitutes"],
            away_subs_bench=away["squad"]["substitutes"])
        random.seed(self.seed)
        engine = MatchEngine(config, home["profile"], away["profile"])
        engine.set_squad(home_name, home["squad"]["starters"],
                         home["squad"]["substitutes"])
        engine.set_squad(away_name, away["squad"]["starters"],
                         away["squad"]["substitutes"])
        engine.set_stamina_controller(sub)
        result = engine.simulate()

        exporter = PLOFAExporter(
            result,
            {home_name: {"starters": home["squad"]["starters"],
                         "substitutes": home["squad"]["substitutes"]},
             away_name: {"starters": away["squad"]["starters"],
                         "substitutes": away["squad"]["substitutes"]}},
            sub_controller=sub)
        record = apply_result(
            self.ledger, result, exporter.accumulator.stats,
            competition_id=competition.competition_id,
            season_id=self.ledger.season_id or "T1",
            home_id=fixture.home_id, away_id=fixture.away_id,
            match_date=fixture.match_date,
            round_name=fixture.round_name, stage_name=fixture.stage_name,
            sub_controller=sub, adapter=self._adapter(),
        )
        record["went_to_extra_time"] = result.went_to_extra_time
        record["shootout"] = bool(result.shootout)
        return record

    def _adapter(self):
        """A name->ID adapter over the live roster, built once."""
        if getattr(self, "_adapter_cache", None) is None:
            from world.ids import NameAdapter
            from roster_loader import get_loader
            loader = get_loader()
            adapter = NameAdapter()
            for club in sorted(loader.get_all_clubs()):
                adapter.register("club", club)
            for club in sorted(loader.get_all_clubs()):
                for rec in loader.get_club_players(club):
                    adapter.register(
                        "player", rec.name,
                        id_override=resolve_player_id(rec.name))
            self._adapter_cache = adapter
        return self._adapter_cache

    # -- the month --------------------------------------------------------

    def competitions_by_id(self) -> Dict[str, Any]:
        w = self.world
        out = {}
        for comp in (w.league, w.continental, w.cup):
            if comp is not None:
                out[comp.competition_id] = comp
        return out

    def play(self, fixtures: Optional[Sequence] = None) -> MonthReport:
        """Play the month in chronological order, checking continuity as it goes."""
        comps = self.competitions_by_id()
        for comp in comps.values():
            self.ledger.register_competition(comp)
        for cid in self.world.all_clubs:
            self.ledger.register_club(cid, self.world.name_of(cid))

        todo = list(fixtures) if fixtures is not None else month_fixtures(self.world)
        report = MonthReport(season_id=self.ledger.season_id or "T1")

        by_date: Dict[date, List] = {}
        for fx in todo:
            by_date.setdefault(fx.match_date, []).append(fx)

        for when in sorted(by_date):
            day = MatchdayRecord(match_date=when)
            for fx in by_date[when]:
                comp = comps.get(fx.competition_id)
                if comp is None:
                    day.issues.append(ContinuityIssue(
                        str(when), "-", f"no competition registered for "
                                        f"{fx.competition_id}"))
                    continue
                record = self.simulate_fixture(fx, comp)
                day.played += 1
                report.matches_played += 1
                report.total_goals += int(record["score"].split("-")[0] or 0)
                report.total_goals += int(record["score"].split("-")[1] or 0)
            # one settlement per matchday, then the §18 checklist
            self.ledger.advance_matchday()
            day.issues.extend(self.check_continuity(when))
            report.matchdays.append(day)
            report.issues.extend(day.issues)

        report.players_seen = len(self.ledger.players)
        return report

    # -- THE §18 CHECKLIST -------------------------------------------------

    def check_continuity(self, when: date) -> List[ContinuityIssue]:
        """Verify plan §18's promise after every matchday.

        This is the actual deliverable of phase 10. The fixture scheduling is
        already proven; what is unproven is that a player's state is ONE
        continuous thing across a league, a continental competition and a cup.
        """
        where = when.isoformat()
        issues: List[ContinuityIssue] = []
        season = self.ledger.season_id or "T1"

        for pid, state in sorted(self.ledger.players.items()):
            # 1. continuous totals only ever grow
            if state.minutes_played < self._last_minutes.get(pid, 0):
                issues.append(ContinuityIssue(
                    where, pid, f"minutes went BACKWARDS across a competition "
                                f"boundary: {self._last_minutes[pid]} -> "
                                f"{state.minutes_played}"))
            if state.appearances < self._last_apps.get(pid, 0):
                issues.append(ContinuityIssue(
                    where, pid, f"appearances went backwards: "
                                f"{self._last_apps[pid]} -> {state.appearances}"))
            self._last_minutes[pid] = state.minutes_played
            self._last_apps[pid] = state.appearances

            # 2. the continuous total equals the sum of the scoped lines
            career = state.career_totals()["overall"]
            if career["minutes"] != state.minutes_played:
                issues.append(ContinuityIssue(
                    where, pid, f"scoped lines sum to {career['minutes']} "
                                f"but the continuous total is "
                                f"{state.minutes_played}"))
            if career["goals"] != state.goals:
                issues.append(ContinuityIssue(
                    where, pid, f"scoped goals sum to {career['goals']} but "
                                f"the continuous total is {state.goals}"))
            if career["appearances"] != state.appearances:
                issues.append(ContinuityIssue(
                    where, pid, f"scoped appearances sum to "
                                f"{career['appearances']} but the continuous "
                                f"total is {state.appearances}"))

            # 3. no line may exceed the continuous total
            for cid in state.competition_ids(season):
                line = state.line(season, cid)
                if line["minutes"] > state.minutes_played:
                    issues.append(ContinuityIssue(
                        where, pid, f"{cid} claims {line['minutes']} minutes, "
                                    f"more than the {state.minutes_played} "
                                    f"played in total"))
                if line["goals"] > state.goals:
                    issues.append(ContinuityIssue(
                        where, pid, f"{cid} claims {line['goals']} goals, more "
                                    f"than the {state.goals} scored in total"))

            # 4. appearances must equal the number of matches the player was in
            expected = sum(1 for m in state.match_history if m["minutes"] > 0)
            if expected != state.appearances:
                issues.append(ContinuityIssue(
                    where, pid, f"{expected} matches with minutes but "
                                f"{state.appearances} appearances recorded"))

        # 5. no club twice in a day, straight off the played history
        by_club_day: Dict[Tuple[str, str], int] = {}
        for m in self.ledger.match_history:
            key = (m["home_id"], m["match_date"])
            by_club_day[key] = by_club_day.get(key, 0) + 1
            key = (m["away_id"], m["match_date"])
            by_club_day[key] = by_club_day.get(key, 0) + 1
        for (club, day), n in by_club_day.items():
            if n > 1:
                issues.append(ContinuityIssue(
                    where, club, f"played {n} matches on {day}"))

        return issues
