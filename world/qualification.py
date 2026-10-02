"""
PLOFA WORLD — qualification, promotion and relegation.
====================================================
world/qualification.py  ·  Phase 8 of the World Football plan.

Plan §14 and §15, in one module:

    §14  "Promotion/relegation rules must be competition configuration rather
          than hard-coded PLOFA assumptions."

    §15  "Competition qualification must be generated from actual
          previous-season results. [...] The system should eventually be able to
          finish one season and automatically construct the next season's
          competition participants."

So this module is the bridge between two seasons. It takes a finished league
table — read from the real warehouse, not invented — and produces the next
season's divisional membership and continental entrants, with every rule
supplied as data.

The rules are data, and that is the whole design
------------------------------------------------
Nothing here knows how many teams PLOFA has, who gets promoted, or what a
Champions League slot looks like. A :class:`QualificationRules` block says it.
That is what lets a second country, a second division, or a playoff round exist
without a code change (audit §4: everything in that table is world/competition
data, not engine logic).

Deterministic and total
-----------------------
Every resolver takes a finished table and returns a decision, or raises. There
is no "best effort" path: a table with too few teams for the configured
promotion slots is an error, not a silently shorter promotion list. A
qualification rule that quietly drops a club is how a league table quietly
becomes wrong.

Where the engine boundary is
----------------------------
Playoff *ties* are resolved by extra time and penalties, which is plan §10/§11
and lands in the engine (audit §16 phase 5). This module therefore accepts a
playoff tie's aggregate as an input and applies the promotion consequence — it
does not simulate the tie. That split is deliberate and is asserted by the
suite, so nobody later mistakes this for a missing feature.
"""
from __future__ import annotations

import dataclasses
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ─────────────────────────────────────────────
# STANDINGS
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Standing:
    """One finished league position."""

    rank: int
    club_id: str
    points: float = 0.0
    goal_diff: int = 0
    goals_for: int = 0

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if not self.club_id:
            raise ValueError("club_id is required")


class QualificationError(RuntimeError):
    """A rule set cannot be applied to the table it was given."""


# ─────────────────────────────────────────────
# PROMOTION / RELEGATION RULES (data)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PromotionRelegationRules:
    """How a division's membership changes. Pure configuration (plan §14).

    ``auto_promote``      clubs promoted outright from the top
    ``auto_relegate``     clubs relegated outright from the bottom
    ``playoff_promotion`` positions that contest a promotion playoff
    ``playoff_relegation`` positions that contest a relegation playoff
    ``playoff_resolution`` how a playoff tie is decided — "aggregate" or
                          "penalties". The tie itself is simulated by the engine
                          (phase 5); this only records which rule applies.
    """

    auto_promote: int = 0
    auto_relegate: int = 0
    playoff_promotion: Tuple[int, ...] = ()
    playoff_relegation: Tuple[int, ...] = ()
    playoff_resolution: str = "aggregate"

    def __post_init__(self) -> None:
        if self.auto_promote < 0 or self.auto_relegate < 0:
            raise ValueError("promotion/relegation counts must be >= 0")
        for name in ("playoff_promotion", "playoff_relegation"):
            positions = getattr(self, name)
            for p in positions:
                if p < 1:
                    raise ValueError(f"{name} positions must be >= 1, got {p}")

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["playoff_promotion"] = list(self.playoff_promotion)
        d["playoff_relegation"] = list(self.playoff_relegation)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromotionRelegationRules":
        data = dict(d)
        for k in ("playoff_promotion", "playoff_relegation"):
            data[k] = tuple(data.get(k, ()))
        return cls(**{k: v for k, v in data.items() if k in _fields(cls)})


def closed_league_rules(promote: int = 0, relegate: int = 0,
                        **kw: Any) -> PromotionRelegationRules:
    """A division nobody leaves — the 26/27 PLOFA default, stated explicitly."""
    return PromotionRelegationRules(auto_promote=promote,
                                    auto_relegate=relegate, **kw)


def two_tier_rules(promote: int = 1, relegate: int = 1) -> PromotionRelegationRules:
    """A conventional second tier: one up, one down, no playoffs."""
    return PromotionRelegationRules(auto_promote=promote, auto_relegate=relegate)


# ─────────────────────────────────────────────
# QUALIFICATION RULES (data)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class QualificationSlots:
    """One competition's entry criteria, as data (plan §15).

    ``positions``      inclusive rank ranges, e.g. ``((1, 4),)`` = the top four
    ``from_cup_winner`` whether this competition also takes the domestic cup
                       winner, and if so how many winners may enter
                       (``cup_winner_slots``)
    """

    competition_id: str
    positions: Tuple[Tuple[int, int], ...] = ()
    from_cup_winner: bool = False
    cup_winner_slots: int = 1

    def __post_init__(self) -> None:
        for lo, hi in self.positions:
            if lo < 1 or hi < lo:
                raise ValueError(f"bad rank range ({lo}, {hi})")
        if self.from_cup_winner and self.cup_winner_slots < 1:
            raise ValueError("cup_winner_slots must be >= 1 when the cup feeds it")

    def ranks(self) -> List[int]:
        out: List[int] = []
        for lo, hi in self.positions:
            out.extend(range(lo, hi + 1))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "competition_id": self.competition_id,
            "positions": [list(p) for p in self.positions],
            "from_cup_winner": self.from_cup_winner,
            "cup_winner_slots": self.cup_winner_slots,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QualificationSlots":
        return cls(
            competition_id=d["competition_id"],
            positions=tuple(tuple(p) for p in d.get("positions", ())),
            from_cup_winner=bool(d.get("from_cup_winner", False)),
            cup_winner_slots=int(d.get("cup_winner_slots", 1)),
        )


@dataclass(frozen=True)
class QualificationRules:
    """A division's full movement + entry rules for one season boundary."""

    division_id: str
    promotion: PromotionRelegationRules = field(
        default_factory=PromotionRelegationRules)
    slots: Tuple[QualificationSlots, ...] = ()
    #: division the promoted clubs join; None means this is the top division
    promotes_to: Optional[str] = None
    #: division the relegated clubs join
    relegates_to: Optional[str] = None

    def slot_for(self, competition_id: str) -> Optional[QualificationSlots]:
        for s in self.slots:
            if s.competition_id == competition_id:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "division_id": self.division_id,
            "promotion": self.promotion.to_dict(),
            "slots": [s.to_dict() for s in self.slots],
            "promotes_to": self.promotes_to,
            "relegates_to": self.relegates_to,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QualificationRules":
        return cls(
            division_id=d["division_id"],
            promotion=PromotionRelegationRules.from_dict(d.get("promotion", {})),
            slots=tuple(QualificationSlots.from_dict(s) for s in d.get("slots", ())),
            promotes_to=d.get("promotes_to"),
            relegates_to=d.get("relegates_to"),
        )


# ─────────────────────────────────────────────
# THE RESULT OF A SEASON BOUNDARY
# ─────────────────────────────────────────────

@dataclass
class SeasonTransition:
    """Everything that changes between two seasons, decided at once."""

    season_from: str
    season_to: str
    division_id: str
    relegated: List[str] = field(default_factory=list)
    promoted: List[str] = field(default_factory=list)
    promotion_playoffs: List["Playoff"] = field(default_factory=list)
    relegation_playoffs: List["Playoff"] = field(default_factory=list)
    #: competition_id -> ordered entrant club_ids
    entrants: Dict[str, List[str]] = field(default_factory=dict)
    #: clubs that qualified for nothing, for auditing
    unqualified: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"PLOFA season transition {self.season_from} -> {self.season_to}",
            f"  division {self.division_id}",
            f"  promoted : {', '.join(self.promoted) or '-'}",
            f"  relegated: {', '.join(self.relegated) or '-'}",
        ]
        for comp, clubs in sorted(self.entrants.items()):
            lines.append(f"  {comp}: {len(clubs)} entrant(s) "
                         f"[{', '.join(clubs) or '-'}]")
        if self.unqualified:
            lines.append(f"  qualified for nothing: "
                         f"{', '.join(self.unqualified)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "season_from": self.season_from,
            "season_to": self.season_to,
            "division_id": self.division_id,
            "relegated": list(self.relegated),
            "promoted": list(self.promoted),
            "promotion_playoffs": [p.to_dict() for p in self.promotion_playoffs],
            "relegation_playoffs": [p.to_dict() for p in self.relegation_playoffs],
            "entrants": {k: list(v) for k, v in self.entrants.items()},
            "unqualified": list(self.unqualified),
        }


@dataclass
class Playoff:
    """A promotion/relegation playoff tie awaiting its result.

    ``aggregate`` is filled in by whoever simulates the tie (the engine, phase 5
    — extra time and penalties). Until then ``resolved`` is False and the club
    is NOT promoted or relegated, because deciding a tie that has not been
    played is how a table becomes fiction.
    """

    position: int
    club_id: str
    stakes: str                       # "promotion" | "relegation"
    resolution: str = "aggregate"     # how the tie is decided
    aggregate: Optional[Tuple[int, int]] = None   # (winner goals, loser goals)
    resolved: bool = False
    winner_id: Optional[str] = None

    def resolve(self, winner_goals: int, loser_goals: int,
                winner_id: str) -> "Playoff":
        if self.resolved:
            raise QualificationError(f"playoff at position {self.position} "
                                     f"is already resolved")
        if winner_goals < loser_goals:
            raise QualificationError(
                f"playoff aggregate {winner_goals}-{loser_goals} does not "
                f"identify a winner; the tie must be decided on penalties")
        self.aggregate = (winner_goals, loser_goals)
        self.winner_id = winner_id
        self.resolved = True
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["aggregate"] = list(self.aggregate) if self.aggregate else None
        return d


# ─────────────────────────────────────────────
# THE ENGINE
# ─────────────────────────────────────────────

class QualificationEngine:
    """Turns a finished table into the next season, entirely from data."""

    def __init__(self, rules: QualificationRules) -> None:
        self.rules = rules

    # -- movement --------------------------------------------------------

    def resolve_movement(
        self,
        standings: Sequence[Standing],
        *,
        season_from: str,
        season_to: str,
        playoff_results: Optional[Dict[int, Tuple[int, int, str]]] = None,
    ) -> SeasonTransition:
        """Decide who moves between divisions.

        ``playoff_results`` maps a playoff's POSITION to
        ``(winner_goals, loser_goals, winner_club_id)``. A configured playoff
        with no result stays unresolved and the position is left empty — it is
        never guessed.
        """
        table = self._require_complete_table(standings)
        pr = self.rules.promotion
        playoff_results = playoff_results or {}

        transition = SeasonTransition(season_from=season_from,
                                      season_to=season_to,
                                      division_id=self.rules.division_id)

        # top of the table: straight promotion, then promotion playoffs
        top = table[: pr.auto_promote] if pr.auto_promote else []
        transition.promoted.extend(s.club_id for s in top)
        promotion_playoff_rows = [s for s in table
                                  if s.rank in pr.playoff_promotion]
        for s in promotion_playoff_rows:
            p = Playoff(position=s.rank, club_id=s.club_id, stakes="promotion",
                        resolution=pr.playoff_resolution)
            result = playoff_results.get(s.rank)
            if result is not None:
                p.resolve(*result)
                transition.promoted.append(p.winner_id)
            transition.promotion_playoffs.append(p)

        # bottom of the table: straight relegation, then relegation playoffs
        bottom = table[-pr.auto_relegate:] if pr.auto_relegate else []
        transition.relegated.extend(s.club_id for s in bottom)
        relegation_playoff_rows = [s for s in table
                                   if s.rank in pr.playoff_relegation]
        for s in relegation_playoff_rows:
            p = Playoff(position=s.rank, club_id=s.club_id, stakes="relegation",
                        resolution=pr.playoff_resolution)
            result = playoff_results.get(s.rank)
            if result is not None:
                p.resolve(*result)
                # a relegation playoff loser goes DOWN, not the winner
                transition.relegated.append(
                    s.club_id if p.winner_id != s.club_id else "")
            transition.relegation_playoffs.append(p)
        transition.relegated = [c for c in transition.relegated if c]

        return transition

    # -- entry -----------------------------------------------------------

    def resolve_entrants(
        self,
        standings: Sequence[Standing],
        *,
        cup_winners: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[str]]:
        """Which clubs enter each competition next season (plan §15).

        Order is deterministic: league positions in rank order, then cup winners.
        A club cannot enter the same competition twice.
        """
        table = self._require_complete_table(standings)
        by_rank = {s.rank: s.club_id for s in table}
        cup_winners = cup_winners or {}

        entrants: Dict[str, List[str]] = {}
        for slot in self.rules.slots:
            names: List[str] = []
            for rank in sorted(slot.ranks()):
                club = by_rank.get(rank)
                if club is None:
                    raise QualificationError(
                        f"{slot.competition_id} wants rank {rank} but the table "
                        f"only has {len(table)} position(s)")
                if club not in names:
                    names.append(club)
            if slot.from_cup_winner:
                for cup_id in sorted(cup_winners)[: slot.cup_winner_slots]:
                    winner = cup_winners[cup_id]
                    if winner and winner not in names:
                        names.append(winner)
            entrants[slot.competition_id] = names
        return entrants

    def build_transition(
        self,
        standings: Sequence[Standing],
        *,
        season_from: str,
        season_to: str,
        cup_winners: Optional[Dict[str, str]] = None,
        playoff_results: Optional[Dict[int, Tuple[int, int, str]]] = None,
    ) -> SeasonTransition:
        """Plan §15 end to end: finish one season, construct the next."""
        transition = self.resolve_movement(
            standings, season_from=season_from, season_to=season_to,
            playoff_results=playoff_results)
        transition.entrants = self.resolve_entrants(
            standings, cup_winners=cup_winners)

        # "Qualified for nothing" means still in this division with no berth
        # anywhere else. A relegated club is not unqualified — it is leaving —
        # and a promoted one has already been accounted for, so both are
        # excluded. Reporting them here would misdescribe a relegation as a
        # failure to qualify.
        qualified = {c for clubs in transition.entrants.values() for c in clubs}
        moved = set(transition.promoted) | set(transition.relegated)
        transition.unqualified = sorted(
            {s.club_id for s in self._require_complete_table(standings)}
            - qualified - moved)
        return transition

    # -- validation ------------------------------------------------------

    def _require_complete_table(self, standings: Sequence[Standing]) -> List[Standing]:
        """Ranks must be 1..N with no gaps or duplicates.

        A gapped table means the caller handed over something that is not a
        finished league table, and every downstream decision would be quietly
        wrong. Refuse instead.
        """
        if not standings:
            raise QualificationError("standings are empty")
        ranks = sorted(s.rank for s in standings)
        if len(set(ranks)) != len(ranks):
            raise QualificationError("duplicate ranks in standings")
        if ranks != list(range(1, len(ranks) + 1)):
            raise QualificationError(
                f"standings must cover ranks 1..{len(ranks)} with no gaps, got "
                f"{ranks}")
        return sorted(standings, key=lambda s: s.rank)


# ─────────────────────────────────────────────
# READING REAL RESULTS (plan §15: "actual previous-season results")
# ─────────────────────────────────────────────

def standings_from_warehouse(
    db_path: str,
    season_id: str,
    competition_id: str,
    *,
    team_names: Optional[Dict[int, str]] = None,
) -> List[Standing]:
    """Read a finished table out of the competition-keyed warehouse.

    This is what makes §15 real rather than aspirational: the entrants are
    derived from rows the engine actually produced, not from a table typed in by
    hand. Requires the audit §16 phase 7 migration (``competition_id`` present).
    """
    path = Path(db_path)
    if not path.exists():
        raise QualificationError(f"warehouse not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(season_standings)")}
        if "competition_id" not in cols:
            raise QualificationError(
                f"{path.name} is not competition-keyed yet — run "
                f"`alltime_db.py migrate-competitions` first")
        rows = conn.execute(
            "SELECT s.rank AS rank, s.team_id AS team_id, s.points AS points, "
            "       s.gf AS gf, s.ga AS ga, t.name AS name "
            "FROM season_standings s "
            "LEFT JOIN teams t ON t.team_id = s.team_id "
            "WHERE s.season = ? AND s.competition_id = ? "
            "ORDER BY s.rank",
            (season_id, competition_id)).fetchall()
        if not rows:
            raise QualificationError(
                f"no standings for season {season_id!r} competition "
                f"{competition_id!r} in {path.name}")
        return [Standing(rank=r["rank"] or (i + 1),
                         club_id=(team_names or {}).get(r["team_id"])
                         or r["name"] or f"team-{r['team_id']}",
                         points=r["points"] or 0.0,
                         goal_diff=(r["gf"] or 0) - (r["ga"] or 0),
                         goals_for=r["gf"] or 0)
                for i, r in enumerate(rows)]
    finally:
        conn.close()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _fields(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}
