"""
PLOFA WORLD — typed world schemas.
===================================
world/model.py  ·  Phase 1 of the World Football plan.

Typed, JSON-serialisable schemas for world entities (plan §6): country,
city, stadium, club, player, manager, referee. These are the *source-data
contracts* the world layer consumes; the live 26/27 engine continues to
use its own PlayerProfile/DNA types — `Player.engine_source` is the
reversible link back to that existing engine identity (plan §25).

Historical correctness (plan §6): `Stadium.capacity_by_season` maps a
season to its capacity so historical seasons never inherit today's
capacity. New entries default to the current season.

Every entity references others strictly by stable ID — never by name.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────
# KINDS (must mirror world.ids.ID_PREFIXES keys)
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# PRIMITIVES
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PitchDims:
    length_m: float = 105.0
    width_m: float = 68.0


def _as_dict(obj: Any) -> Dict[str, Any]:
    d = asdict(obj)
    for key, value in list(d.items()):
        if isinstance(value, date):
            d[key] = value.isoformat()
    return d


# ─────────────────────────────────────────────
# ENTITIES
# ─────────────────────────────────────────────

@dataclass
class Country:
    country_id: str
    name: str
    continent: str = ""
    association: str = ""
    league_structure: List[str] = field(default_factory=list)
    cup_structure: List[str] = field(default_factory=list)
    coefficient: float = 0.0
    city_ids: List[str] = field(default_factory=list)
    stadium_ids: List[str] = field(default_factory=list)
    club_ids: List[str] = field(default_factory=list)
    referee_ids: List[str] = field(default_factory=list)
    manager_ids: List[str] = field(default_factory=list)
    player_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Country":
        return cls(**{k: d[k] for k in d if k in cls_fields(cls)})


@dataclass
class City:
    city_id: str
    name: str
    country_id: str
    population: Optional[int] = None
    region: str = ""
    football_culture: str = ""
    climate: str = "clear"

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "City":
        return cls(**{k: d[k] for k in d if k in cls_fields(cls)})


@dataclass
class Stadium:
    stadium_id: str
    name: str
    city_id: str
    country_id: str
    surface: str = "grass"
    pitch: PitchDims = field(default_factory=PitchDims)
    capacity_by_season: Dict[str, int] = field(default_factory=dict)
    atmosphere: Dict[str, float] = field(default_factory=dict)

    def capacity_for(self, season: str) -> Optional[int]:
        """Capacity for a season; None when unknown (never fabricate)."""
        return self.capacity_by_season.get(season)

    def set_capacity(self, season: str, capacity: int) -> None:
        self.capacity_by_season[season] = int(capacity)

    def to_dict(self) -> Dict[str, Any]:
        d = _as_dict(self)
        d["pitch"] = asdict(self.pitch)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stadium":
        data = dict(d)
        if isinstance(data.get("pitch"), dict):
            data["pitch"] = PitchDims(**data["pitch"])
        return cls(**{k: data[k] for k in data if k in cls_fields(cls)})


@dataclass
class Club:
    club_id: str
    name: str
    country_id: str
    city_id: str = ""
    stadium_id: str = ""
    division_id: str = ""
    colors: Dict[str, str] = field(default_factory=dict)
    style: Dict[str, float] = field(default_factory=dict)
    rivalry_ids: List[str] = field(default_factory=list)
    honours: List[str] = field(default_factory=list)
    historical_records: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Club":
        return cls(**{k: d[k] for k in d if k in cls_fields(cls)})


@dataclass
class Player:
    player_id: str
    canonical_name: str
    country_id: str
    club_id: str = ""
    aliases: List[str] = field(default_factory=list)
    position: str = ""
    nationality: str = ""
    dob_iso: Optional[str] = None
    # reversible link to the existing engine identity (plan §25) — the
    # live engine stays authoritative; world never fabricates DNA.
    engine_source: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Player":
        return cls(**{k: d[k] for k in d if k in cls_fields(cls)})


@dataclass
class Manager:
    manager_id: str
    canonical_name: str
    country_id: str
    club_id: str = ""
    aliases: List[str] = field(default_factory=list)
    style: Dict[str, float] = field(default_factory=dict)
    engine_source: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Manager":
        return cls(**{k: d[k] for k in d if k in cls_fields(cls)})


@dataclass
class Referee:
    referee_id: str
    canonical_name: str
    country_id: str
    aliases: List[str] = field(default_factory=list)
    strictness: Optional[float] = None
    engine_source: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Referee":
        return cls(**{k: d[k] for k in d if k in cls_fields(cls)})


# ─────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────

def cls_fields(cls) -> set:
    import dataclasses
    return {f.name for f in dataclasses.fields(cls)}


def dump_json(entity, path: str) -> None:
    """Write any world entity to JSON (deterministic key order)."""
    d = entity.to_dict()
    Path(path).write_text(
        json.dumps(d, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_json(path: str):
    """Load a world entity record (["type": ...] must be a dict with an
    entity dict; the concrete schema is chosen by the caller)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data