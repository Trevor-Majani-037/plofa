"""
PLOFA WORLD — the world-facing state ledger.
===========================================
world/ledger.py  ·  Phase 6 of the World Football plan.

Plan §7 is the reason this module exists:

    "A player's state must not reset simply because he moves from the league
     to a cup or Champions League. [...] The player entering the Champions
     League must remember what happened on Saturday."

and plan §18 is the test that proves it:

    Saturday domestic league → Wednesday Champions League → Saturday league →
    Wednesday domestic cup, with ONE continuous state per player throughout.

The distinction this ledger is built around
------------------------------------------
Two kinds of state, and conflating them is the bug this design prevents:

**CARRY-ACROSS state** is global to the person and survives every boundary —
fatigue, injuries, suspensions, cards, confidence, form, accumulated minutes
and workload, development. Plan §7 lists them; they live once per player, at
the top level, and no competition transition may reset them.

**SCOPED state** belongs to one competition — appearances, goals, assists, xG,
and the rest of a season's statistical line. Plan §8 requires a player's league
and continental rows to *coexist*, so these are keyed
``(season_id, competition_id)`` and only ever aggregate on request.

What this module deliberately does NOT do
-----------------------------------------
It does not model fatigue, injury probability or development. Those belong to
``squad_manager`` / ``training_system``, and plan §13 is explicit that existing
logic is reused rather than duplicated. The ledger's job is **scoping and
continuity** — deciding where a piece of state lives and making sure it is
carried — not re-deriving physics. Callers push engine-computed deltas in via
:meth:`WorldLedger.apply_post_match`; the ledger guarantees they are stored
globally and attributed to the right competition.

The 26/27 dual-write rule (audit §14)
-------------------------------------
The live ``season_state.json`` stays authoritative for the PLOFA league and is
**never written by this module**. A second competition runs in its own
competition-scoped ledger file that only the world layer touches. The guard is
enforced in code — see :meth:`WorldLedger._assert_writable` — not by
convention, because a stray write to the live season ledger is unrecoverable.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from world.ids import canonical_key

SCHEMA_VERSION = 1
DEFAULT_LEDGER_PATH = "world_state.json"

#: Files the world layer must never write. The live 26/27 season ledger is
#: authoritative for the PLOFA league (audit §14) and a stray write to it
#: corrupts a season that cannot be replayed.
PROTECTED_PATHS = frozenset({
    "season_state.json",
    "season_stats.json",
    "manager_state.json",
    "referee_state.json",
    "alltime.db",
})


class LedgerWriteError(RuntimeError):
    """An attempt to write through the world layer into a protected 26/27 file."""


# ─────────────────────────────────────────────
# POST-MATCH INPUT
# ─────────────────────────────────────────────

@dataclass
class PlayerMatchLine:
    """One player's line in one match, as the caller reports it.

    The ledger never computes these; the engine does. Everything is optional
    except the identity, so a caller can report a partial line and the ledger
    still records what it was told.
    """

    player_id: str
    club_id: str = ""
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    shots: int = 0
    xg: float = 0.0
    yellow_cards: int = 0
    red_card: bool = False
    # engine-computed carry-across state AFTER this match
    fatigue: Optional[float] = None
    fitness: Optional[float] = None
    confidence: Optional[float] = None
    form: Optional[str] = None
    injury: Optional[str] = None
    injury_minute: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlayerMatchLine":
        return cls(**{k: v for k, v in d.items() if k in _fields(cls)})


@dataclass
class MatchReport:
    """A played match, in the shape the ledger consumes.

    Deliberately independent of ``MatchResult`` — the ledger is duck-typed
    against whatever the caller has, so the world layer stays independent of the
    match engine's internals (plan §3, plan §30).
    """

    competition_id: str
    season_id: str
    match_date: date
    home_id: str
    away_id: str
    home_goals: int = 0
    away_goals: int = 0
    round_name: str = ""
    stage_name: str = ""
    is_neutral: bool = False
    players: List[PlayerMatchLine] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["match_date"] = self.match_date.isoformat()
        d["players"] = [p.to_dict() for p in self.players]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MatchReport":
        data = dict(d)
        data["match_date"] = date.fromisoformat(data["match_date"])
        data["players"] = [PlayerMatchLine.from_dict(p) for p in data.get("players", ())]
        return cls(**{k: v for k, v in data.items() if k in _fields(cls)})


# ─────────────────────────────────────────────
# PLAYER STATE
# ─────────────────────────────────────────────

#: Plan §7's carry-across list, as ledger fields. Asserted against
#: :data:`PlayerState.CARRY_ACROSS` so the two can never drift apart.
CARRY_ACROSS_FIELDS = (
    "minutes_played",
    "appearances",
    "fatigue",
    "fitness",
    "confidence",
    "form",
    "workload",
    "goals",
    "assists",
    "shots",
    "xg",
    "yellow_cards",
    "red_cards",
)

#: The statistical line a competition owns. Everything here is scoped by
#: ``(season_id, competition_id)`` rather than carried.
SCOPED_STAT_FIELDS = (
    "appearances",
    "minutes",
    "goals",
    "assists",
    "shots",
    "xg",
    "yellow_cards",
    "red_cards",
)

#: Which continuous field each scoped statistic accumulates into. The names
#: differ deliberately — a season line reads "minutes", the continuous state
#: reads "minutes_played" — so the relationship is declared rather than inferred
#: from a name match. Asserted by the suite so the two lists cannot drift.
SCOPED_TO_CARRY_ACROSS = {
    "appearances": "appearances",
    "minutes": "minutes_played",
    "goals": "goals",
    "assists": "assists",
    "shots": "shots",
    "xg": "xg",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
}


@dataclass
class PlayerState:
    """One person's football life, globally, plus their per-competition lines.

    The split is the whole point:

    * the top-level scalars are CARRY-ACROSS — one continuous state that no
      competition transition may reset (plan §7);
    * ``per_season[season]["per_competition"][competition]`` holds the SCOPED
      statistical lines that plan §8 requires to coexist.
    """

    player_id: str
    club_id: str = ""
    # ── carry-across (global, never reset by a competition boundary) ──
    minutes_played: int = 0
    appearances: int = 0
    fatigue: float = 0.0
    fitness: float = 1.0
    confidence: float = 0.5
    form: str = ""
    workload: float = 0.0
    goals: int = 0
    assists: int = 0
    shots: int = 0
    xg: float = 0.0
    yellow_cards: int = 0
    red_cards: int = 0
    # ── events (global, dated) ──
    injuries: List[Dict[str, Any]] = field(default_factory=list)
    suspensions: List[Dict[str, Any]] = field(default_factory=list)
    yellow_card_events: List[Dict[str, Any]] = field(default_factory=list)
    #: How many matches this player has appeared in, ever. The suspension
    #: accumulation window is counted in MATCHES (not days), so every card
    #: event is stamped with the index at which it was earned.
    match_index: int = 0
    # ── scoped statistics ──
    per_season: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    # ── audit trail ──
    match_history: List[Dict[str, Any]] = field(default_factory=list)

    CARRY_ACROSS = CARRY_ACROSS_FIELDS

    # -- scoped line access ---------------------------------------------

    def line(self, season_id: str, competition_id: str) -> Dict[str, Any]:
        """The player's line in one competition of one season.

        Returns a fresh, fully-populated zeroed dict so a caller can never
        mutate the stored record by accident, and so a caller never has to
        guard against a missing key.
        """
        stored = (self.per_season.get(season_id, {})
                             .get("per_competition", {})
                             .get(competition_id))
        if stored is None:
            return {"competition_id": competition_id, "season_id": season_id,
                    **{f: (0.0 if f == "xg" else 0) for f in SCOPED_STAT_FIELDS}}
        return dict(stored)

    def competition_ids(self, season_id: str) -> List[str]:
        return sorted(self.per_season.get(season_id, {})
                                    .get("per_competition", {}))

    # -- career aggregation ---------------------------------------------

    def career_totals(self) -> Dict[str, Any]:
        """Every competition summed, per season and overall.

        This is the plan §8 promise: league and continental rows coexist AND
        career totals are a query over them, never a separately maintained
        counter that can drift.
        """
        per_season: Dict[str, Dict[str, Any]] = {}
        overall = {"appearances": 0, "minutes": 0, "goals": 0, "assists": 0,
                   "shots": 0, "xg": 0.0, "yellow_cards": 0, "red_cards": 0}
        for season_id, body in self.per_season.items():
            totals = {"appearances": 0, "minutes": 0, "goals": 0, "assists": 0,
                      "shots": 0, "xg": 0.0, "yellow_cards": 0, "red_cards": 0}
            for line in body.get("per_competition", {}).values():
                for f in SCOPED_STAT_FIELDS:
                    totals[f] = totals.get(f, 0) + line.get(f, 0)
                    overall[f] = overall.get(f, 0) + line.get(f, 0)
            per_season[season_id] = totals
        return {"per_season": per_season, "overall": overall,
                "competitions": sorted(
                    c for b in self.per_season.values()
                    for c in b.get("per_competition", {}))}

    # -- availability (plan §13) ----------------------------------------

    def settle(self) -> Dict[str, int]:
        """Advance this player by one match: heal and serve suspensions.

        Without this an injury or a suspension never expires and the player is
        unavailable FOREVER — the single most damaging way a ledger like this
        can be wrong, because it silently ends a player's season. Callers run
        it once per completed matchday via
        :meth:`WorldLedger.advance_matchday`.
        """
        healed = 0
        served = 0
        for bucket, counter in ((self.injuries, "injuries"),
                                (self.suspensions, "suspensions")):
            keep = []
            for event in bucket:
                if int(event.get("matches_remaining", 0)) > 0:
                    event["matches_remaining"] = int(event["matches_remaining"]) - 1
                    if counter == "injuries":
                        healed += 1
                    else:
                        served += 1
                if int(event.get("matches_remaining", 0)) > 0:
                    keep.append(event)
            setattr(self, counter, keep)
        return {"healed": healed, "served": served}

    def availability(self, on: date, suspension_rule: Optional[Dict[str, Any]] = None
                     ) -> Tuple[bool, str]:
        """Is this player available on ``on``? (plan §13)

        Suspension is resolved against the *competition's own* rule, because a
        five-yellow ban in the league need not apply in a cup whose accumulation
        window differs. That is why the rule is passed in rather than stored
        globally — it is competition data (audit §4 rows 13–14).
        """
        for inj in self.injuries:
            if _event_covers(inj, on):
                return False, f"injured: {inj.get('injury_type', 'unknown')}"

        if suspension_rule:
            window = int(suspension_rule.get("window", 6))
            threshold = int(suspension_rule.get("threshold", 5))
            # the window is counted in MATCHES, not days
            recent = sum(
                1 for ev in self.yellow_card_events
                if int(ev.get("match_index", 0)) > self.match_index - window
            )
            if recent >= threshold:
                return False, (
                    f"suspended: {recent} yellows in the last {window} matches "
                    f"({suspension_rule.get('type', 'rule')})"
                )

        for susp in self.suspensions:
            if int(susp.get("matches_remaining", 0)) > 0:
                return False, f"suspended: {susp.get('reason', 'rule')}"
        return True, "available"

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlayerState":
        return cls(**{k: v for k, v in d.items() if k in _fields(cls)})


# ─────────────────────────────────────────────
# CLUB / COMPETITION STATE
# ─────────────────────────────────────────────

@dataclass
class ClubState:
    club_id: str
    name: str = ""
    honours: List[str] = field(default_factory=list)
    per_season: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClubState":
        return cls(**{k: v for k, v in d.items() if k in _fields(cls)})


@dataclass
class CompetitionState:
    competition_id: str
    name: str = ""
    type: str = "league"
    country_id: str = ""
    season_id: str = ""
    rules: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    # competition-owned progression (standings rows, bracket) stays with the
    # competition object; the ledger only records what happened.

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompetitionState":
        return cls(**{k: v for k, v in d.items() if k in _fields(cls)})


# ─────────────────────────────────────────────
# THE LEDGER
# ─────────────────────────────────────────────

class WorldLedger:
    """The world-facing state store (audit §14).

    One file holds everything the world knows: competitions, the global player
    state, club honours and history. It is a **superset** of the v1
    ``SeasonState`` concept — everything ``SeasonState`` tracks for one league
    lives here too, plus the competition dimension and stable IDs.
    """

    def __init__(self, path: str = DEFAULT_LEDGER_PATH, *,
                 country_id: str = "", season_id: str = "") -> None:
        self._assert_writable(path)
        self.path = path
        self.country_id = country_id
        self.season_id = season_id
        self.players: Dict[str, PlayerState] = {}
        self.clubs: Dict[str, ClubState] = {}
        self.competitions: Dict[str, CompetitionState] = {}
        self.seasons: Dict[str, Dict[str, Any]] = {}
        self.match_history: List[Dict[str, Any]] = []
        self._dedupe_index: set = set()

    # -- the write guard (audit §14) -------------------------------------

    @staticmethod
    def _assert_writable(path: str) -> None:
        """Refuse to point the world ledger at a protected 26/27 file.

        Enforced in code because a stray write to ``season_state.json`` cannot
        be undone — a played fixture cannot be replayed (README §2).
        """
        name = Path(path).name.casefold()
        if name in PROTECTED_PATHS:
            raise LedgerWriteError(
                f"refusing to use '{path}' as a world ledger: it is a protected "
                f"26/27 file. The live season ledger stays authoritative for "
                f"the PLOFA league (audit §14); give the world layer its own "
                f"competition-scoped file."
            )

    # -- registration ----------------------------------------------------

    def register_competition(self, competition: Any) -> CompetitionState:
        """Register a competition from a ``world.competition.Competition``."""
        state = CompetitionState(
            competition_id=competition.competition_id,
            name=competition.name,
            type=competition.rules.type,
            country_id=competition.country_id,
            season_id=competition.season_id,
            rules=competition.rules.to_dict(),
        )
        self.competitions[state.competition_id] = state
        self.seasons.setdefault(state.season_id, {}).setdefault(
            "competitions", {})[state.competition_id] = {
                "name": state.name, "type": state.type,
            }
        return state

    def register_club(self, club_id: str, name: str = "") -> ClubState:
        club = self.clubs.get(club_id)
        if club is None:
            club = ClubState(club_id=club_id, name=name or club_id)
            self.clubs[club_id] = club
        elif name and not club.name:
            club.name = name
        return club

    def register_player(self, player_id: str, club_id: str = "") -> PlayerState:
        player = self.players.get(player_id)
        if player is None:
            player = PlayerState(player_id=player_id, club_id=club_id)
            self.players[player_id] = player
        elif club_id and not player.club_id:
            player.club_id = club_id
        return player

    # -- recording a match -----------------------------------------------

    def record_match(self, report: MatchReport) -> Dict[str, PlayerState]:
        """Record one played match. Idempotent per
        ``(season, competition, date, home, away)``.

        Returns the player states that were touched, so a caller can render
        them without a second lookup.
        """
        key = (report.season_id, report.competition_id, report.match_date,
               report.home_id, report.away_id)
        if key in self._dedupe_index:
            return {}
        self._dedupe_index.add(key)

        for club_id in (report.home_id, report.away_id):
            self.register_club(club_id)

        comp = self.competitions.setdefault(
            report.competition_id,
            CompetitionState(competition_id=report.competition_id,
                             season_id=report.season_id),
        )
        comp.results.append({
            "season_id": report.season_id,
            "match_date": report.match_date.isoformat(),
            "home_id": report.home_id,
            "away_id": report.away_id,
            "home_goals": report.home_goals,
            "away_goals": report.away_goals,
            "round_name": report.round_name,
            "stage_name": report.stage_name,
            "is_neutral": report.is_neutral,
        })

        self.match_history.append({
            "competition_id": report.competition_id,
            "season_id": report.season_id,
            "match_date": report.match_date.isoformat(),
            "home_id": report.home_id,
            "away_id": report.away_id,
            "score": f"{report.home_goals}-{report.away_goals}",
        })

        touched: Dict[str, PlayerState] = {}
        for line in report.players:
            state = touched.get(line.player_id)
            if state is None:
                state = self.apply_post_match(report, line)
                touched[line.player_id] = state
        return touched

    def apply_post_match(self, report: MatchReport,
                         line: PlayerMatchLine) -> PlayerState:
        """Fold one player's match line into their continuous state.

        The two-way split of §7 happens here and nowhere else:

        * the **carry-across** scalars are updated once, globally;
        * the **scoped** statistical line is updated under
          ``(season_id, competition_id)``.

        Engine-computed values (fatigue, fitness, confidence) are *taken from
        the caller*, never re-derived — plan §13 forbids duplicating the
        existing squad/fitness model.
        """
        state = self.register_player(line.player_id, line.club_id)

        # An appearance is a MATCH PLAYED, not a squad-list entry. The exporter
        # emits a stat line for every named player including unused
        # substitutes (minutes 0), and counting those would inflate every
        # appearance total in the warehouse — found by the phase-10 integration
        # test, which flagged "0 matches with minutes but 1 appearance".
        played = line.minutes > 0

        # ── carry-across: one continuous state, no competition boundary ──
        state.minutes_played += line.minutes
        if played:
            state.appearances += 1
        state.workload += line.minutes
        state.goals += line.goals
        state.assists += line.assists
        state.shots += line.shots
        state.xg += line.xg
        state.yellow_cards += line.yellow_cards
        state.red_cards += 1 if line.red_card else 0
        if line.fatigue is not None:
            state.fatigue = float(line.fatigue)
        if line.fitness is not None:
            state.fitness = float(line.fitness)
        if line.confidence is not None:
            state.confidence = float(line.confidence)
        if line.form is not None:
            state.form = line.form

        # ── events ──
        if line.injury:
            state.injuries.append({
                "date": report.match_date.isoformat(),
                "injury_type": line.injury,
                "minute": line.injury_minute,
                "competition_id": report.competition_id,
                "matches_remaining": _default_injury_matches(line.injury),
            })
        # Every yellow is stamped, including the first. A five-in-six rule is
        # evaluated by COUNTING cards across a window, so a card that is not
        # recorded is a card the accumulation rule can never see.
        for _ in range(line.yellow_cards):
            state.yellow_card_events.append({
                "date": report.match_date.isoformat(),
                "competition_id": report.competition_id,
                "match_index": state.match_index,
            })
        if line.red_card:
            state.suspensions.append({
                "date": report.match_date.isoformat(),
                "reason": "red_card",
                "matches_remaining": 1,
                "competition_id": report.competition_id,
            })
        elif line.yellow_cards >= 2:
            state.suspensions.append({
                "date": report.match_date.isoformat(),
                "reason": "second_yellow",
                "matches_remaining": 1,
                "competition_id": report.competition_id,
            })
        state.match_index += 1

        # ── scoped: this competition's own statistical line ──
        season = state.per_season.setdefault(report.season_id, {})
        per_comp = season.setdefault("per_competition", {})
        line_stats = per_comp.setdefault(
            report.competition_id,
            {"competition_id": report.competition_id,
             "season_id": report.season_id,
             **{f: (0.0 if f == "xg" else 0) for f in SCOPED_STAT_FIELDS}},
        )
        line_stats["appearances"] = line_stats.get("appearances", 0) + (1 if played else 0)
        line_stats["minutes"] = line_stats.get("minutes", 0) + line.minutes
        line_stats["goals"] = line_stats.get("goals", 0) + line.goals
        line_stats["assists"] = line_stats.get("assists", 0) + line.assists
        line_stats["shots"] = line_stats.get("shots", 0) + line.shots
        line_stats["xg"] = line_stats.get("xg", 0.0) + line.xg
        line_stats["yellow_cards"] = line_stats.get("yellow_cards", 0) + line.yellow_cards
        line_stats["red_cards"] = line_stats.get("red_cards", 0) + (1 if line.red_card else 0)

        state.match_history.append({
            "competition_id": report.competition_id,
            "season_id": report.season_id,
            "match_date": report.match_date.isoformat(),
            "minutes": line.minutes,
            "played": played,
            "goals": line.goals,
            "assists": line.assists,
            "home_id": report.home_id,
            "away_id": report.away_id,
        })
        return state

    def record_matchday(self, reports: Iterable[MatchReport]) -> Dict[str, PlayerState]:
        """Record a whole matchday, then advance the world by one match.

        The advance is what makes injuries and suspensions expire. Recording
        without advancing leaves a player unavailable forever, so the two steps
        are offered together as the safe default.
        """
        touched: Dict[str, PlayerState] = {}
        for report in reports:
            touched.update(self.record_match(report))
        self.advance_matchday()
        return touched

    def advance_matchday(self) -> Dict[str, Dict[str, int]]:
        """Settle every player by one match (heal injuries, serve suspensions)."""
        return {pid: state.settle() for pid, state in self.players.items()}

    # -- reading ---------------------------------------------------------

    def player(self, player_id: str) -> Optional[PlayerState]:
        return self.players.get(player_id)

    def club(self, club_id: str) -> Optional[ClubState]:
        return self.clubs.get(club_id)

    def competition(self, competition_id: str) -> Optional[CompetitionState]:
        return self.competitions.get(competition_id)

    def season_line(self, player_id: str, season_id: str,
                    competition_id: str) -> Dict[str, Any]:
        state = self.players.get(player_id)
        if state is None:
            return {"competition_id": competition_id, "season_id": season_id,
                    **{f: (0.0 if f == "xg" else 0) for f in SCOPED_STAT_FIELDS}}
        return state.line(season_id, competition_id)

    def career(self, player_id: str) -> Dict[str, Any]:
        state = self.players.get(player_id)
        if state is None:
            return {"per_season": {}, "overall": {}, "competitions": []}
        return state.career_totals()

    def suspension_rule(self, competition_id: str) -> Optional[Dict[str, Any]]:
        comp = self.competitions.get(competition_id)
        if comp is None or not comp.rules:
            return None
        return {
            "type": comp.rules.get("suspension_type", "yellow_accum"),
            "threshold": comp.rules.get("suspension_threshold", 5),
            "window": comp.rules.get("suspension_window", 6),
        }

    def availability(self, player_id: str, on: date,
                     competition_id: str = "") -> Tuple[bool, str]:
        """Plan §13 availability, resolved against the competition's own rules."""
        state = self.players.get(player_id)
        if state is None:
            return False, "unknown player"
        return state.availability(on, self.suspension_rule(competition_id))

    def played_in(self, player_id: str, competition_id: str,
                  season_id: str = "") -> List[Dict[str, Any]]:
        state = self.players.get(player_id)
        if state is None:
            return []
        return [m for m in state.match_history
                if m["competition_id"] == competition_id
                and (not season_id or m["season_id"] == season_id)]

    # -- reporting -------------------------------------------------------

    def cross_competition_line(self, player_id: str, season_id: str
                               ) -> Dict[str, Any]:
        """The plan §18 view: one player, every competition, side by side.

        This is the shape that makes continuity visible — a single row per
        competition next to a single continuous set of carry-across values.
        """
        state = self.players.get(player_id)
        if state is None:
            return {}
        rows = {cid: state.line(season_id, cid)
                for cid in state.competition_ids(season_id)}
        return {
            "player_id": player_id,
            "club_id": state.club_id,
            "continuous": {
                "minutes_played": state.minutes_played,
                "appearances": state.appearances,
                "fatigue": state.fatigue,
                "fitness": state.fitness,
                "confidence": state.confidence,
                "form": state.form,
                "workload": state.workload,
                "injuries": len(state.injuries),
                "suspensions": len(state.suspensions),
            },
            "per_competition": rows,
            "career": state.career_totals()["overall"],
        }

    def summary(self) -> str:
        lines = [
            f"PLOFA world ledger - {self.path}",
            f"  {len(self.competitions)} competition(s), "
            f"{len(self.clubs)} club(s), {len(self.players)} player(s), "
            f"{len(self.match_history)} match(es)",
        ]
        for season_id in sorted(self.seasons):
            comps = self.seasons[season_id].get("competitions", {})
            lines.append(f"  season {season_id}: "
                         f"{len(comps)} competition(s) "
                         f"({', '.join(sorted(comps)) or '-'})")
        return "\n".join(lines)

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "country_id": self.country_id,
            "season_id": self.season_id,
            "seasons": self.seasons,
            "competitions": {k: v.to_dict() for k, v in self.competitions.items()},
            "clubs": {k: v.to_dict() for k, v in self.clubs.items()},
            "players": {k: v.to_dict() for k, v in self.players.items()},
            "match_history": self.match_history,
        }

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path
        self._assert_writable(target)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str) -> "WorldLedger":
        cls._assert_writable(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ledger = cls(path=path, country_id=data.get("country_id", ""),
                     season_id=data.get("season_id", ""))
        ledger.seasons = data.get("seasons", {})
        ledger.competitions = {
            k: CompetitionState.from_dict(v)
            for k, v in data.get("competitions", {}).items()}
        ledger.clubs = {k: ClubState.from_dict(v)
                        for k, v in data.get("clubs", {}).items()}
        ledger.players = {k: PlayerState.from_dict(v)
                          for k, v in data.get("players", {}).items()}
        ledger.match_history = data.get("match_history", [])
        ledger._rebuild_dedupe_index()
        return ledger

    def _rebuild_dedupe_index(self) -> None:
        self._dedupe_index = {
            (m.get("season_id"), m.get("competition_id"),
             _parse_date(m.get("match_date")), m.get("home_id"), m.get("away_id"))
            for m in self.match_history
        }


# ─────────────────────────────────────────────
# 26/27 READ-ONLY ADAPTER (audit §14)
# ─────────────────────────────────────────────

def read_plofa_season_state(path: str = "season_state.json") -> Dict[str, Any]:
    """Read the live 26/27 season ledger WITHOUT writing to it.

    Audit §14's dual-write rule: the live file stays authoritative for the PLOFA
    league, and the world ledger only *reads* it when a request spans
    competitions. This function is deliberately read-only and returns a plain
    dict — it hands out copies, so a caller cannot mutate live state by accident
    either.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "season": data.get("season", ""),
        "standings": dict(data.get("standings", {})),
        "player_count": len(data.get("players", {})),
        "matchday": data.get("matchday"),
        "fixture_ledger": dict(data.get("fixture_ledger", {})),
    }


def import_plofa_season_state(ledger: WorldLedger, path: str = "season_state.json",
                              *, competition_id: str = "CMP-PLOFA",
                              season_id: str = "") -> WorldLedger:
    """Seed a world ledger from the live 26/27 state, read-only.

    Only the season-level facts travel (identity, the table, how far the season
    has run). Per-player form/fatigue is deliberately NOT copied: the live
    ``SeasonState`` remains the authority for the PLOFA league, and duplicating
    it would create two sources of truth for the same player.
    """
    live = read_plofa_season_state(path)
    season = season_id or live["season"]
    ledger.country_id = ledger.country_id or "CTR-PLOFA"
    ledger.season_id = ledger.season_id or season
    ledger.register_competition(_SyntheticCompetition(
        competition_id, "PLOFA 26/27", "league", "CTR-PLOFA", season))
    for club_name in live["standings"]:
        ledger.register_club(canonical_key(club_name), club_name)
    return ledger


class _SyntheticCompetition:
    """Minimal competition-shaped object for the read-only import path.

    Avoids importing ``world.competition`` here so the adapter stays usable
    even if the competition module is mid-refactor; ``register_competition``
    only reads these five attributes.
    """

    def __init__(self, competition_id, name, type_, country_id, season_id):
        self.competition_id = competition_id
        self.name = name
        self.rules = _SyntheticRules(type_)
        self.country_id = country_id
        self.season_id = season_id


class _SyntheticRules:
    def __init__(self, type_):
        self.type = type_

    def to_dict(self):
        return {"type": self.type}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

#: Rough recovery length per injury class, in matches. This is *bookkeeping for
#: availability only* — the injury model itself belongs to ``squad_manager``.
_INJURY_MATCHES = {
    "knock": 1, "bruise": 1, "fatigue": 1,
    "sprain": 3, "muscle_strain": 3,
    "fracture": 8, "tear": 10, "acl": 24,
}


def _default_injury_matches(injury_type: str) -> int:
    return _INJURY_MATCHES.get(str(injury_type).lower(), 4)


def _parse_date(value: Any) -> Optional[date]:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _event_covers(event: Dict[str, Any], on: date) -> bool:
    started = _parse_date(event.get("date"))
    if started is None or on < started:
        return False
    return int(event.get("matches_remaining", 0)) > 0


def _fields(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}
