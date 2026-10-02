"""
Test PLOFA WORLD typed schemas — world/model.py
=================================================
`world/model.py` shipped on 2026-09-23 with no suite at all. These are the
*source-data contracts* every other world module reads, so their two hard
promises need proving:

  * every entity references others strictly by stable ID, never by name
  * a schema survives a real JSON file round trip unchanged

Plus the two rules that quietly corrupt a football world if they are wrong:
per-season stadium capacity (plan §6 "historical capacity must remain
historically correct") and never fabricating a value that is not known.

Run from the repo root with PYTHONPATH=.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from world.model import (
    City,
    Club,
    Country,
    Manager,
    PitchDims,
    Player,
    Referee,
    Stadium,
    dump_json,
    load_json,
)


# ─────────────────────────────────────────────
# PRIMITIVES
# ─────────────────────────────────────────────

def test_pitch_dims_defaults_are_a_standard_football_pitch():
    p = PitchDims()
    assert (p.length_m, p.width_m) == (105.0, 68.0)


def test_pitch_dims_are_frozen():
    p = PitchDims()
    with pytest.raises(Exception):
        p.length_m = 120.0  # type: ignore[misc]


# ─────────────────────────────────────────────
# ROUND TRIP
# ─────────────────────────────────────────────

_ALL_ENTITIES = [
    Country(country_id="CTR-TOL", name="Toland", continent="Europa",
            association="Toland Football Federation",
            league_structure=["TOL-D1"], cup_structure=["TOL-CUP"],
            coefficient=12.5, city_ids=["CTY-1"], club_ids=["CLB-1"]),
    City(city_id="CTY-1", name="Hartwell", country_id="CTR-TOL",
        population=84000, region="North", climate="rain"),
    Stadium(stadium_id="STA-1", name="Hartwell Park", city_id="CTY-1",
            country_id="CTR-TOL", surface="grass",
            capacity_by_season={"25/26": 52000, "26/27": 54500},
            atmosphere={"base": 0.72}),
    Club(club_id="CLB-1", name="Hartwell City", country_id="CTR-TOL",
        city_id="CTY-1", stadium_id="STA-1", division_id="TOL-D1",
        colors={"home": "#003087"}, style={"press": 0.81},
        rivalry_ids=["CLB-2"], honours=["TOL-CUP 25/26"]),
    Player(player_id="PLY-1", canonical_name="Percy Osei", country_id="CTR-TOL",
           club_id="CLB-1", aliases=["Percy"], position="ST",
           dob_iso="1997-04-12", nationality="Tolandian",
           engine_source={"source": "PLOFA-2026-2027.xlsx", "row": 42}),
    Manager(manager_id="MGR-1", canonical_name="A. Rowe", country_id="CTR-TOL",
            club_id="CLB-1", style={"risk_tolerance": 0.6}),
    Referee(referee_id="REF-1", canonical_name="M. Adeyemi",
            country_id="CTR-TOL", strictness=0.55),
]


@pytest.mark.parametrize("entity", _ALL_ENTITIES, ids=lambda e: type(e).__name__)
def test_entity_json_round_trip_is_value_preserving(entity):
    """A world data file that reads back different from what was written is
    silent corruption, so assert the exact contract through real JSON."""
    back = type(entity).from_dict(json.loads(json.dumps(entity.to_dict())))
    assert back == entity


@pytest.mark.parametrize("entity", _ALL_ENTITIES, ids=lambda e: type(e).__name__)
def test_entity_dict_is_deterministic_in_declaration_order(entity):
    """``to_dict`` keeps dataclass declaration order (id first, then name —
    readable in a REPL), and must return that order identically every time."""
    import dataclasses
    d = entity.to_dict()
    assert list(d) == [f.name for f in dataclasses.fields(type(entity))]
    assert list(entity.to_dict()) == list(d)
    assert json.dumps(d) == json.dumps(entity.to_dict())


@pytest.mark.parametrize("entity", _ALL_ENTITIES, ids=lambda e: type(e).__name__)
def test_written_world_file_is_key_sorted(entity, tmp_path):
    """On disk the file is key-sorted, so a diff of two world data files shows
    real changes only."""
    path = tmp_path / f"{type(entity).__name__}.json"
    dump_json(entity, str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert list(raw) == sorted(raw)


@pytest.mark.parametrize("entity", _ALL_ENTITIES, ids=lambda e: type(e).__name__)
def test_from_dict_ignores_unknown_keys(entity):
    """Forward compatibility: a newer world file must still load."""
    data = dict(entity.to_dict())
    data["a_field_from_the_future"] = {"anything": 1}
    assert type(entity).from_dict(data) == entity


def test_dump_and_load_json_round_trip(tmp_path):
    stadium = Stadium(stadium_id="STA-1", name="Hartwell Park",
                      city_id="CTY-1", country_id="CTR-TOL",
                      capacity_by_season={"26/27": 54500})
    path = tmp_path / "stadium.json"
    dump_json(stadium, str(path))
    assert Stadium.from_dict(load_json(str(path))) == stadium


def test_dump_json_is_utf8_clean(tmp_path):
    """Club and player names carry diacritics; a world file must not mangle them."""
    club = Club(club_id="CLB-9", name="Roy Steupý United",
                country_id="CTR-TOL")
    path = tmp_path / "club.json"
    dump_json(club, str(path))
    assert Club.from_dict(load_json(str(path))).name == "Roy Steupý United"


# ─────────────────────────────────────────────
# ID-ONLY REFERENCES (the core contract)
# ─────────────────────────────────────────────

def _declared_fields(cls):
    import dataclasses
    return {f.name for f in dataclasses.fields(cls)}


# The reference graph, declared explicitly. If someone adds a `home_club` or a
# `manager_name` field, this fails — which is the whole point.
_EXPECTED_REFERENCES = {
    Country: {"country_id",
              "city_ids", "stadium_ids", "club_ids",
              "referee_ids", "manager_ids", "player_ids"},
    City:    {"city_id", "country_id"},
    Stadium: {"stadium_id", "city_id", "country_id"},
    Club:    {"club_id", "country_id", "city_id", "stadium_id",
              "division_id", "rivalry_ids"},
    Player:  {"player_id", "country_id", "club_id"},
    Manager: {"manager_id", "country_id", "club_id"},
    Referee: {"referee_id", "country_id"},
}

# An entity's OWN display name is legitimate. What must never exist is a
# reference to ANOTHER entity by name (a `home_club_name`, a `manager_name`) —
# that is what makes identity mutable. Declared per class so adding a stray
# name field anywhere fails the suite.
_OWN_NAME_FIELD = {
    Country: "name",
    City: "name",
    Stadium: "name",
    Club: "name",
    Player: "canonical_name",
    Manager: "canonical_name",
    Referee: "canonical_name",
}


@pytest.mark.parametrize("cls", list(_EXPECTED_REFERENCES), ids=lambda c: c.__name__)
def test_reference_fields_are_exactly_the_declared_id_graph(cls):
    """Every cross-entity reference is a stable ID — singular ``*_id`` or
    plural ``*_ids`` — and nothing else. plan §23: names are never primary
    identity, and a name creeping into a reference slot would silently make
    identity mutable again."""
    fields = _declared_fields(cls)
    references = {f for f in fields
                  if f.endswith("_id") or f.endswith("_ids")}
    assert references == _EXPECTED_REFERENCES[cls], (
        f"{cls.__name__} reference fields drifted from the declared ID graph"
    )


@pytest.mark.parametrize("cls", list(_EXPECTED_REFERENCES), ids=lambda c: c.__name__)
def test_the_only_name_field_is_the_entitys_own_display_name(cls):
    """An entity may carry its own display name; it may never carry a
    name-based *reference* to something else."""
    allowed = _OWN_NAME_FIELD[cls]
    for field in _declared_fields(cls):
        if "name" in field and not field.endswith("_ids"):
            assert field == allowed, (
                f"{cls.__name__}.{field} is a name field that is not the "
                f"entity's own display name ('{allowed}') — world entities "
                f"must reference each other by ID only"
            )


@pytest.mark.parametrize("cls", list(_EXPECTED_REFERENCES), ids=lambda c: c.__name__)
def test_the_own_display_name_is_required_not_optional(cls):
    """Without a display name an entity cannot be shown to a human, and the
    ledger would fall back to printing a raw ID."""
    import dataclasses
    field = next(f for f in dataclasses.fields(cls) if f.name == _OWN_NAME_FIELD[cls])
    assert field.default is dataclasses.MISSING, (
        f"{cls.__name__}.{field.name} must be required"
    )


@pytest.mark.parametrize("cls", list(_EXPECTED_REFERENCES), ids=lambda c: c.__name__)
def test_entity_id_field_matches_its_own_id_prefix(cls):
    """Every entity carries its own identity, and the field name agrees with
    the kind in ``world.ids.ID_PREFIXES`` (``club`` -> ``club_id``)."""
    prefixes = {
        "Country": "country", "City": "city", "Stadium": "stadium",
        "Club": "club", "Player": "player", "Manager": "manager",
        "Referee": "referee",
    }
    kind = prefixes[cls.__name__]
    assert f"{kind}_id" in _declared_fields(cls)


def test_country_reference_lists_are_independent_per_instance():
    """Mutable dataclass defaults are a classic shared-state bug: two countries
    must not share one list."""
    a = Country(country_id="CTR-A", name="A")
    b = Country(country_id="CTR-B", name="B")
    a.club_ids.append("CLB-1")
    assert b.club_ids == []


# ─────────────────────────────────────────────
# HISTORICAL STADIUM CAPACITY (plan §6)
# ─────────────────────────────────────────────

def test_capacity_is_per_season_not_a_single_number():
    s = Stadium(stadium_id="STA-1", name="Hartwell Park",
                city_id="CTY-1", country_id="CTR-TOL",
                capacity_by_season={"24/25": 48000, "25/26": 52000,
                                    "26/27": 54500})
    assert s.capacity_for("24/25") == 48000
    assert s.capacity_for("26/27") == 54500


def test_unknown_season_capacity_is_none_never_fabricated():
    """A historical season must never inherit today's capacity by accident."""
    s = Stadium(stadium_id="STA-1", name="Hartwell Park",
                city_id="CTY-1", country_id="CTR-TOL",
                capacity_by_season={"26/27": 54500})
    assert s.capacity_for("99/00") is None
    assert s.capacity_for("25/26") is None


def test_set_capacity_is_season_scoped():
    s = Stadium(stadium_id="STA-1", name="Hartwell Park",
                city_id="CTY-1", country_id="CTR-TOL",
                capacity_by_season={"26/27": 54500})
    s.set_capacity("27/28", 56000)
    assert s.capacity_for("27/28") == 56000
    assert s.capacity_for("26/27") == 54500


def test_nested_pitch_survives_the_round_trip_as_a_dataclass():
    s = Stadium(stadium_id="STA-1", name="Hartwell Park",
                city_id="CTY-1", country_id="CTR-TOL",
                pitch=PitchDims(110.0, 70.0))
    back = Stadium.from_dict(json.loads(json.dumps(s.to_dict())))
    assert isinstance(back.pitch, PitchDims)
    assert (back.pitch.length_m, back.pitch.width_m) == (110.0, 70.0)


# ─────────────────────────────────────────────
# THE ENGINE LINK (plan §25)
# ─────────────────────────────────────────────

def test_player_keeps_a_reversible_link_to_the_live_engine_identity():
    """plan §25: the live engine stays authoritative and the world never
    fabricates DNA. The link must be carried as data, not resolved here."""
    p = Player(player_id="PLY-1", canonical_name="Percy Osei",
               country_id="CTR-TOL", club_id="CLB-1", position="ST",
               engine_source={"source": "PLOFA-2026-2027.xlsx", "row": 42})
    assert p.engine_source["row"] == 42
    assert not hasattr(p, "dna"), "the world must not invent player DNA"
    assert Player.from_dict(p.to_dict()).engine_source == p.engine_source


def test_aliases_survive_so_the_name_adapter_can_be_seeded():
    p = Player(player_id="PLY-1", canonical_name="Rayan Victor James",
               country_id="CTR-TOL",
               aliases=["Victor James", "R. V. James"])
    back = Player.from_dict(p.to_dict())
    assert back.aliases == ["Victor James", "R. V. James"]


def test_referee_strictness_is_optional_until_it_is_known():
    r = Referee(referee_id="REF-1", canonical_name="M. Adeyemi",
                country_id="CTR-TOL")
    assert r.strictness is None
    r2 = Referee(referee_id="REF-2", canonical_name="K. Silva",
                 country_id="CTR-TOL", strictness=0.8)
    assert Referee.from_dict(r2.to_dict()).strictness == 0.8
