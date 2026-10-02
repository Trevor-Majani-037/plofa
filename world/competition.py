"""
PLOFA WORLD — generic competition framework.
============================================
world/competition.py  ·  Phases 2–3 of the World Football plan.

A competition is DATA + progression, never engine. This module provides:

* ``CompetitionRules`` — a frozen block of competition-specific rule data
  (points, tie-breakers, extra time, penalties, substitutions, promotion/
  relegation, qualification). No rule is hard-coded into the framework.
* ``MatchContextFragment`` — the additive ``MatchConfig`` payload a
  competition hands to the match engine (plan §4). ``apply()`` duck-types
  against the PLOFA ``MatchConfig`` so the engine sees competition context
  without importing this module, and defaults preserve 26/27 behaviour.
* ``LeagueCompetition``  — round-robin league: standings + fixtures,
  reusing the existing ``season_manager`` types (plan §30) via lazy imports.
* ``KnockoutCompetition`` — deterministic single-leg knockout with optional
  penalty-shootout resolution (plan §9, §11). Aggregate/legs arrive with
  the calendar phase.

Phase 1–3 scope guardrails enforced here:
  * no engine import at module load (duck typing only)
  * no writes to authoritative 26/27 state
  * no calendar, no continental competition, no data population
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ─────────────────────────────────────────────
# COMPETITION TYPE
# ─────────────────────────────────────────────

class CompetitionType:
    LEAGUE = "league"
    CUP = "cup"
    CONTINENTAL = "continental"


# ─────────────────────────────────────────────
# COMPETITION RULES (data, not code)
# ─────────────────────────────────────────────

def _json_safe(value: Any) -> Any:
    """Recursively coerce tuples/sets to lists so ``to_dict`` output survives a
    real JSON round trip unchanged.

    A shallow ``list()`` is not enough: ``qualification_out`` holds dicts whose
    own values may be tuples, and a world data file that reads back subtly
    different from what was written is a silent-corruption bug waiting to
    surface months later in a ledger.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    return value


def _tuples(value: Any) -> Any:
    """Inverse of :func:`_json_safe` — recursively restore tuples.

    JSON has one array type, so a file round trip cannot tell a list from a
    tuple. The canonical *in-memory* form is therefore fixed as "tuples" by this
    function, which is what makes ``from_dict(json.loads(json.dumps(x))) == x``
    hold exactly. Without it a world data file written and re-read silently
    yields rules that no longer compare equal to the ones that produced it.
    """
    if isinstance(value, dict):
        return {k: _tuples(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_tuples(v) for v in value)
    return value


def _require_rules(rules: Any) -> "CompetitionRules":
    """Validate a rules payload before anything reads off it.

    Subclasses check ``rules.type`` in their own ``__init__`` *before* calling
    ``super().__init__``, so the type check has to happen here — otherwise a
    caller passing a plain dict gets an ``AttributeError`` from deep inside a
    constructor instead of the clear ``TypeError`` the API promises.
    """
    if not isinstance(rules, CompetitionRules):
        raise TypeError(
            "rules must be a CompetitionRules instance, got "
            f"{type(rules).__name__}"
        )
    return rules


@dataclass(frozen=True)
class CompetitionRules:
    """Every rule a competition needs. Fields are defaults; a competition
    configures itself by supplying the fields that differ."""

    type: str = CompetitionType.LEAGUE
    schedule_kind: str = "round_robin"          # "round_robin" | "knockout"
    points_win: int = 3
    points_draw: int = 1
    points_loss: int = 0
    tie_breakers: Tuple[str, ...] = ("pts", "gd", "gf")
    double_round_robin: bool = True
    rounds: Optional[int] = None                # explicit round count
    extra_time: bool = False
    penalties: bool = False
    aggregate: bool = False
    away_goals: bool = False
    legs: int = 1
    substitutions_max: int = 3                  # PLOFA standard default
    max_bench: int = 7
    suspension_type: str = "yellow_accum"       # competition-scoped §13
    suspension_threshold: int = 5
    suspension_window: int = 6
    relegation_slots: int = 0
    promotion_slots: int = 0
    qualification_out: Tuple[Dict[str, Any], ...] = ()   # [(competition_id, slots)]

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompetitionRules":
        data = dict(d)
        data["tie_breakers"] = _tuples(data.get("tie_breakers", ()))
        data["qualification_out"] = _tuples(data.get("qualification_out", ()))
        return cls(**{k: v for k, v in data.items() if k in cls_field_names()})


def league_rules(**overrides: Any) -> "CompetitionRules":
    """Convenience constructor for league rules (data lives in fields)."""
    base = dict(
        type=CompetitionType.LEAGUE,
        schedule_kind="round_robin",
        extra_time=False,
        penalties=False,
    )
    base.update(overrides)
    return CompetitionRules(**base)


def cup_rules(**overrides: Any) -> "CompetitionRules":
    """Convenience constructor for knockout cup rules (data lives in fields)."""
    base = dict(
        type=CompetitionType.CUP,
        schedule_kind="knockout",
        extra_time=True,
        penalties=True,
        double_round_robin=False,
    )
    base.update(overrides)
    return CompetitionRules(**base)


# ─────────────────────────────────────────────
# MATCH CONTEXT FRAGMENT
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class MatchContextFragment:
    """The additive MatchConfig payload for one match.

    Field names intentionally mirror the additive fields that Phases 2–3
    added to the PLOFA ``MatchConfig`` so ``apply()`` is a plain field-by-
    field replace. Everything defaults to today's 26/27 neutral behaviour.
    """

    competition_id: Optional[str] = None
    competition_type: Optional[str] = None
    stage_name: Optional[str] = None
    round_name: Optional[str] = None
    leg: Optional[str] = None
    extra_time: bool = False
    penalties: bool = False
    aggregate: Optional[Tuple[int, int]] = None
    away_goals_rule: bool = False
    importance: float = 1.0
    substitution_rules: Optional[Dict[str, Any]] = None

    def apply(self, config: Any) -> Any:
        """Return a NEW config carrying this fragment's known fields.

        Duck-typed against the PLOFA MatchConfig: only fields that exist on
        the datum are replaced, and ``None`` values are never applied. The
        input config is never mutated.
        """
        known = {f.name for f in dataclasses.fields(config)}
        updates = {}
        for name, value in dataclasses.asdict(self).items():
            if name in known and value is not None:
                updates[name] = value
        if not updates:
            return config
        return dataclasses.replace(config, **updates)


# ─────────────────────────────────────────────
# COMPETITION BASE
# ─────────────────────────────────────────────

class Competition:
    """Common identity shape for a competition."""

    def __init__(
        self,
        competition_id: str,
        name: str,
        country_id: str,
        season_id: str,
        rules: CompetitionRules,
        participant_ids: Sequence[str],
        seed: int = 0,
    ) -> None:
        if not isinstance(rules, CompetitionRules):
            raise TypeError("rules must be a CompetitionRules instance")
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError("participant_ids must be unique")
        self.competition_id = competition_id
        self.name = name
        self.country_id = country_id
        self.season_id = season_id
        self.rules = rules
        self.participant_ids = list(participant_ids)
        self.seed = seed

    # -- context --------------------------------------------------------

    def context_for(
        self,
        *,
        stage_name: Optional[str] = None,
        round_name: Optional[str] = None,
        leg: Optional[str] = None,
        aggregate: Optional[Tuple[int, int]] = None,
    ) -> MatchContextFragment:
        return MatchContextFragment(
            competition_id=self.competition_id,
            competition_type=self.rules.type,
            stage_name=stage_name or "League",
            round_name=round_name or "League",
            leg=leg,
            aggregate=aggregate,
            extra_time=self.rules.extra_time,
            penalties=self.rules.penalties,
            away_goals_rule=self.rules.away_goals,
            importance=1.0,
            substitution_rules={
                "max": self.rules.substitutions_max,
                "bench": self.rules.max_bench,
            },
        )

    # -- identity -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "competition_id": self.competition_id,
            "name": self.name,
            "country_id": self.country_id,
            "season_id": self.season_id,
            "seed": self.seed,
            "rules": self.rules.to_dict(),
            "participant_ids": list(self.participant_ids),
        }


# ─────────────────────────────────────────────
# LEAGUE
# ─────────────────────────────────────────────

class LeagueCompetition(Competition):
    """Round-robin league. Standings/fixtures reuse the existing
    ``season_manager.LeagueTable`` / ``FixtureList`` (plan §30 reuse)."""

    def __init__(
        self,
        competition_id: str,
        name: str,
        country_id: str,
        season_id: str,
        rules: CompetitionRules,
        participant_ids: Sequence[str],
        seed: int = 0,
    ) -> None:
        _require_rules(rules)
        # Validate the SCHEDULE FORMAT, not the `type` label. The real
        # Champions League league phase is a league played inside a continental
        # competition, so `type="continental"` must be accepted here — what
        # this class actually implements is round-robin scheduling.
        if rules.schedule_kind != "round_robin":
            raise ValueError(
                f"a league needs schedule_kind='round_robin', got "
                f"{rules.schedule_kind!r} (type {rules.type!r})")
        super().__init__(competition_id, name, country_id, season_id, rules,
                         participant_ids, seed)
        self._fixtures: Any = None          # season_manager.FixtureList
        self._table: Any = None             # season_manager.LeagueTable
        self._results: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    # -- setup ----------------------------------------------------------

    def _ensure_reused(self) -> None:
        """Lazily bring in the existing PLOFA league machinery."""
        from season_manager import FixtureList, LeagueTable  # lazy
        if self._fixtures is None:
            self._fixtures = FixtureList()
        if self._table is None:
            self._table = LeagueTable(list(self.participant_ids))

    def make_fixtures(self, start_date: date, *, days_between: int = 7) -> Any:
        """Generate the round-robin fixture list (double by default)."""
        from season_manager import FixtureList  # lazy

        fl = FixtureList.round_robin(
            list(self.participant_ids),
            start_date,
            days_between=days_between,
            double_round=self.rules.double_round_robin,
        )
        if self.rules.rounds is not None:
            by_md: Dict[int, List[Any]] = {}
            for f in fl.fixtures:
                by_md.setdefault(f.matchday, []).append(f)
            kept: List[Any] = []
            for md in sorted(by_md)[: self.rules.rounds]:
                kept.extend(by_md[md])
            fl = FixtureList(kept)
        self._fixtures = fl
        return fl

    def apply_result(
        self,
        home_team: str,
        away_team: str,
        matchday: int,
        home_goals: int,
        away_goals: int,
    ) -> None:
        """Record a league result into the shared table + fixtures."""
        self._ensure_reused()
        if self._table is None:
            raise RuntimeError("league table not initialised")
        self._table.add_result(home_team, away_team, home_goals, away_goals)
        self._results[(home_team, away_team, matchday)] = {
            "home_goals": home_goals,
            "away_goals": away_goals,
        }
        if self._fixtures is not None and self._fixtures.fixtures:
            self._fixtures.mark_played(home_team, away_team, matchday,
                                       home_goals, away_goals)

    def standings(self) -> List[Any]:
        """Sorted league table (list of season_manager.TeamRecord)."""
        self._ensure_reused()
        if self._table is None:
            raise RuntimeError("league table not initialised")
        return self._table.standings()

    def matchdays(self) -> int:
        if self._fixtures is None or not self._fixtures.fixtures:
            return 0
        return max(f.matchday for f in self._fixtures.fixtures)

    @property
    def results(self) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
        return dict(self._results)


# ─────────────────────────────────────────────
# KNOCKOUT CUP
# ─────────────────────────────────────────────

@dataclass
class KnockoutTie:
    round_name: str
    home_id: str
    away_id: str
    leg: Optional[str] = None
    played: bool = False
    home_goals: int = 0
    away_goals: int = 0
    home_penalties: Optional[int] = None
    away_penalties: Optional[int] = None
    winner_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


#: Round names for every bracket size a real competition actually uses. 32 and
#: 64 matter: the FA Cup third round is 32 clubs and continental knockouts can
#: be 64, and a knockout that raises KeyError on a real field size is a toy.
_ROUND_NAMES_BY_SIZE = {
    2: ("Final",),
    4: ("Semi-final", "Final"),
    8: ("Quarter-final", "Semi-final", "Final"),
    16: ("Round of 16", "Quarter-final", "Semi-final", "Final"),
    32: ("Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"),
    64: ("Round of 64", "Round of 32", "Round of 16", "Quarter-final",
         "Semi-final", "Final"),
}


class KnockoutCompetition(Competition):
    """Deterministic single-leg knockout with optional penalty resolution.

    Participants are padded with byes to the nearest power of two. A draw
    is reproducible: it re-seeds a dedicated RNG from ``(seed, round_index)``
    so results never depend on registration order of who applied what.
    """

    def __init__(
        self,
        competition_id: str,
        name: str,
        country_id: str,
        season_id: str,
        rules: CompetitionRules,
        participant_ids: Sequence[str],
        seed: int = 0,
    ) -> None:
        _require_rules(rules)
        # As with the league, this validates the FORMAT implemented, not the
        # label: a continental knockout is still a knockout.
        if rules.schedule_kind != "knockout":
            raise ValueError(
                f"a knockout needs schedule_kind='knockout', got "
                f"{rules.schedule_kind!r} (type {rules.type!r})")
        if rules.aggregate or rules.legs != 1:
            raise NotImplementedError(
                "Phase 1–3 scope: single-leg knockout only; aggregate/legs "
                "arrive with the calendar phase."
            )
        super().__init__(competition_id, name, country_id, season_id, rules,
                         participant_ids, seed)
        n = len(self.participant_ids)
        m = max(2, 1 << (n - 1).bit_length())
        self._slot_count = m
        self._round_names: Tuple[str, ...] = _ROUND_NAMES_BY_SIZE[m]
        # entrants per round (None = bye)
        self._entrants: Dict[str, List[Optional[str]]] = {
            rn: [] for rn in self._round_names
        }
        self._entrants[self._round_names[0]] = (
            list(self.participant_ids) + [None] * (m - n)
        )
        self._ties: Dict[str, List[KnockoutTie]] = {rn: [] for rn in self._round_names}
        self._round_index = {rn: i for i, rn in enumerate(self._round_names)}

    # -- structure ------------------------------------------------------

    def round_names(self) -> Tuple[str, ...]:
        return self._round_names

    def round_is_drawable(self, round_name: str) -> bool:
        """A round is drawable when no current-round ties exist yet."""
        return round_name in self._ties and not self._ties[round_name]

    # -- progression ----------------------------------------------------

    def draw_round(self, round_name: str) -> List[KnockoutTie]:
        """Deterministically draw one knockout round. Returns the ties."""
        if round_name not in self._round_index:
            raise ValueError(
                f"unknown round '{round_name}'; rounds: {', '.join(self._round_names)}"
            )
        if self._ties[round_name]:
            raise ValueError(f"round '{round_name}' is already drawn")

        idx = self._round_index[round_name]
        pool = [p for p in self._entrants[round_name] if p is not None]

        rng = random.Random(f"{self.seed}:{idx}")
        entries = pool[:]
        rng.shuffle(entries)

        # odd count → deterministic bye for the last entry of the shuffle
        if len(entries) % 2 == 1:
            bye = entries.pop()
            self._advance(round_name, bye)

        ties = []
        for i in range(0, len(entries) - 1, 2):
            tie = KnockoutTie(
                round_name=round_name, home_id=entries[i], away_id=entries[i + 1],
            )
            ties.append(tie)
        self._ties[round_name] = ties
        return ties

    def _advance(self, round_name: str, winner: str) -> None:
        idx = self._round_index[round_name]
        if idx + 1 < len(self._round_names):
            self._entrants[self._round_names[idx + 1]].append(winner)

    def apply_result(
        self,
        tie: KnockoutTie,
        home_goals: int,
        away_goals: int,
        *,
        home_penalties: Optional[int] = None,
        away_penalties: Optional[int] = None,
    ) -> None:
        """Resolve one tie and advance the winner."""
        if tie.round_name not in self._round_names:
            raise ValueError(f"tie belongs to unknown round '{tie.round_name}'")
        if tie.played:
            raise ValueError("tie is already resolved")
        if home_goals < 0 or away_goals < 0:
            raise ValueError("goal counts must be non-negative")

        tie.home_goals = int(home_goals)
        tie.away_goals = int(away_goals)
        tie.home_penalties = home_penalties
        tie.away_penalties = away_penalties
        tie.played = True

        if home_goals > away_goals:
            winner = tie.home_id
        elif away_goals > home_goals:
            winner = tie.away_id
        elif (self.rules.penalties and home_penalties is not None
              and away_penalties is not None):
            if home_penalties == away_penalties:
                raise ValueError("penalty shootout cannot be drawn")
            winner = tie.home_id if home_penalties > away_penalties else tie.away_id
        else:
            raise ValueError(
                f"level score {home_goals}-{away_goals} without a penalty "
                f"shootout (competition rules: penalties={self.rules.penalties})"
            )

        tie.winner_id = winner
        self._advance(tie.round_name, winner)

    # -- outcome --------------------------------------------------------

    def ties_in(self, round_name: str) -> List[KnockoutTie]:
        return list(self._ties.get(round_name, ()))

    def entrants_in(self, round_name: str) -> List[str]:
        return [p for p in self._entrants.get(round_name, ()) if p is not None]

    def winner(self) -> Optional[str]:
        """Winner of the final round once decided; None until then."""
        last = self._round_names[-1]
        ties = self._ties.get(last, ())
        if not ties or not ties[0].played:
            return None
        return ties[0].winner_id


# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────

def cls_field_names() -> set:
    return {f.name for f in dataclasses.fields(CompetitionRules)}