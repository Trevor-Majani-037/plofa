"""
Test PLOFA WORLD identity foundation — world/ids.py
====================================================
Proves the Phase-1 identity contract from PLOFA_WORLD_LAYER_AUDIT.md §23:

  * deterministic, order-independent ID minting (never ordinal)
  * bidirectional name ↔ ID reversibility
  * canonical-key normalisation (case/whitespace, diacritics preserved)
  * loud collision detection — no silent merging
  * real PLOFA 26/27 roster resolves to stable, reversible IDs

The v1 scope is about EXISTING PLOFA identities — so the definitive
assertion is: every club and player we index today already has a stable,
collision-free ID, and every name round-trips through it.
"""
from __future__ import annotations

import random
from typing import List

import pytest

from world.ids import (
    ID_KINDS,
    ID_PREFIXES,
    NameAdapter,
    IdentityCollisionError,
    canonical_key,
    mint_id,
    build_plofa_adapter,
)

# ─────────────────────────────────────────────
# canonical_key
# ─────────────────────────────────────────────

def test_canonical_key_is_case_and_whitespace_insensitive():
    assert canonical_key("Hartwell City") == canonical_key("hartwell  city")
    assert canonical_key("  HARTWELL CITY ") == canonical_key("Hartwell City")
    assert canonical_key("Costa Smeralda") == canonical_key("costa smeralda")


def test_canonical_key_preserves_diacritics():
    # Diacritics are identity-relevant, never stripped (plan §7).
    assert canonical_key("Roy Steupý") != canonical_key("Roy Steupy")


def test_canonical_key_rejects_empty_and_non_string():
    with pytest.raises(ValueError):
        canonical_key("   ")
    with pytest.raises(TypeError):
        canonical_key(42)


# ─────────────────────────────────────────────
# mint_id
# ─────────────────────────────────────────────

def test_mint_id_is_deterministic():
    assert mint_id("club", "PLOFA Panthers") == mint_id("club", "PLOFA Panthers")


def test_mint_id_is_order_independent():
    a = mint_id("club", "Hartwell City")
    b = mint_id("club", "Port Colborne United")
    # re-mint in reverse order must give the SAME ids (no ordinal counter)
    assert mint_id("club", "Port Colborne United") == b
    assert mint_id("club", "Hartwell City") == a


def test_mint_id_has_correct_grammar():
    eid = mint_id("player", "Some Name")
    prefix, _, suffix = eid.partition("-")
    assert prefix == ID_PREFIXES["player"]
    assert len(suffix) == 6
    assert all(c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in suffix)


def test_mint_id_unknown_kind_raises():
    with pytest.raises(ValueError):
        mint_id("dinosaur", "Rex")


def test_canonicalizes_equivalent_keys_to_same_id():
    assert mint_id("player", "Alan Shearer") == mint_id("player", "ALAN   SHEARER")


def test_id_prefixes_are_unique_per_kind():
    assert len(set(ID_PREFIXES.values())) == len(ID_PREFIXES)
    assert set(ID_PREFIXES) == set(ID_KINDS)


# ─────────────────────────────────────────────
# NameAdapter
# ─────────────────────────────────────────────

def test_register_and_resolve_round_trip():
    a = NameAdapter()
    pid = a.register("player", "Alan Shearer", aliases=["Shearer", "Big Al"])
    assert a.to_id("player", "Alan Shearer") == pid
    assert a.to_id("player", "ALAN SHEARER") == pid
    assert a.to_id("player", "Big Al") == pid
    assert a.to_name("player", pid) == "Alan Shearer"
    assert a.canonical_key_for("player", pid) == canonical_key("Alan Shearer")


def test_same_name_different_kinds_are_distinct():
    a = NameAdapter()
    club = a.register("club", "Rovers")
    city = a.register("city", "Rovers")
    assert club != city


def test_two_names_resolving_to_same_id_collides_loudly():
    """A *different* identity may never land on an ID that is already owned.

    The same key re-registered is idempotent (see the next test); a genuine
    collision is one where a second name claims an ID the registry already
    attributes to a different canonical key.
    """
    a = NameAdapter()
    a.register("player", "Alan Shearer")
    with pytest.raises(IdentityCollisionError):
        a.register("player", "Someone Else", id_override=a.to_id("player", "Alan Shearer"))


def test_register_is_idempotent():
    a = NameAdapter()
    pid = a.register("player", "Alan Shearer")
    assert a.register("player", "Alan Shearer") == pid
    assert a.register("player", "Alan Shearer") == pid


def test_alias_conflict_raises_loudly():
    a = NameAdapter()
    a.register("player", "Alan Shearer")
    with pytest.raises(IdentityCollisionError):
        a.register("player", "Someone Else", aliases=["Alan Shearer"])


def test_id_override_pins_renamed_identity():
    a = NameAdapter()
    old = a.register("player", "Alan Shearer")
    a.register("player", "Westside Heroes :: Alan Shearer", id_override="CLB-XXXXXX-2")
    assert a.to_id("player", "Alan Shearer") == old
    assert a.to_id("player", "Westside Heroes :: Alan Shearer") != old


def test_id_override_cannot_hijack_an_owned_id():
    """A pinned ID is where a silent merge would do the most damage, so
    ownership is enforced even when the caller supplies the ID explicitly."""
    a = NameAdapter()
    pid = a.register("player", "Alan Shearer")
    with pytest.raises(IdentityCollisionError):
        a.register("player", "Totally Different Person", id_override=pid)
    # the original identity is untouched
    assert a.to_name("player", pid) == "Alan Shearer"
    assert a.to_id("player", "Totally Different Person") is None


def test_recannonicalise_moves_a_name_without_changing_the_id():
    """A rename is a fact about one person, not an identity merge — and any
    name already written to a world ledger must stay resolvable."""
    a = NameAdapter()
    pid = a.register("player", "Victor James")
    assert a.recannonicalise("player", pid, "Rayan Victor James") == pid
    assert a.to_id("player", "Rayan Victor James") == pid
    assert a.to_name("player", pid) == "Rayan Victor James"
    # the old spelling still resolves — the alltime_db.alias retro rule
    assert a.to_id("player", "Victor James") == pid
    assert a.validate() == 1


def test_recannonicalise_can_drop_the_old_spelling():
    a = NameAdapter()
    pid = a.register("player", "Victor James")
    a.recannonicalise("player", pid, "Rayan Victor James", keep_old_name_as_alias=False)
    assert a.to_id("player", "Victor James") is None
    assert a.to_id("player", "Rayan Victor James") == pid


def test_recannonicalise_rejects_merging_two_people():
    a = NameAdapter()
    keep = a.register("player", "Victor James")
    taken = a.register("player", "Someone Else")
    with pytest.raises(IdentityCollisionError):
        a.recannonicalise("player", taken, "Victor James")
    assert a.to_id("player", "Someone Else") == taken
    assert a.to_id("player", "Victor James") == keep


def test_recannonicalise_unknown_id_raises():
    a = NameAdapter()
    with pytest.raises(KeyError):
        a.recannonicalise("player", "PLY-000000", "Nobody")


def test_validate_passes_and_counts():
    a = NameAdapter()
    a.register("club", "Rovers")
    a.register("club", "City", aliases=["South City"])
    assert a.validate() == 2


# ─────────────────────────────────────────────
# REAL PLOFA 26/27 ROSTER → WORLD IDS
# ─────────────────────────────────────────────
#
# Ground truth for the 26/27 source of truth (PLOFA-2026-2027.xlsx), verified
# against roster_loader on 2026-09-25:
#     18 clubs   ·   433 players   ·   every player name unique roster-wide
# These counts are pinned deliberately: a club or player silently vanishing
# from the world index is a data-loss bug, so the floor is asserted on every run.
#
# `auto_run_match.TEAM_CATALOG` also lists "Hartwell City" and "Thornfield
# United", but those are TEST teams whose squads are entered by hand in the
# scratch runner `run_match.py` (confirmed by Trevor, 2026-09-25). They are not
# part of the football world, so the world index correctly reports 18 and the
# production path correctly refuses them.

EXPECTED_CLUBS = 18
EXPECTED_PLAYERS = 433


def _roster_counts() -> tuple:
    from roster_loader import RosterLoader
    loader = RosterLoader()
    clubs = sorted(loader.get_all_clubs())
    return len(clubs), sum(len(list(loader.get_club_players(c))) for c in clubs)


def test_world_index_covers_the_entire_roster():
    """No club and no player may be missing from the world identity index."""
    adapter = build_plofa_adapter(include_players=True)
    assert len(adapter.entities("club")) == EXPECTED_CLUBS
    assert len(adapter.entities("player")) == EXPECTED_PLAYERS


def test_world_index_matches_the_roster_loader_exactly():
    """The index is derived from the roster, so it must track it exactly.

    Guards the other direction too: an index larger than the roster would mean
    the adapter invented identities.
    """
    adapter = build_plofa_adapter(include_players=True)
    n_clubs, n_players = _roster_counts()
    assert len(adapter.entities("club")) == n_clubs
    assert len(adapter.entities("player")) == n_players


def test_plofa_roster_produces_canonically_formed_ids():
    adapter = build_plofa_adapter(include_players=True)
    for kind, prefix in (("club", ID_PREFIXES["club"]),
                         ("player", ID_PREFIXES["player"])):
        for eid, _name in adapter.entities(kind):
            p, _, s = eid.partition("-")
            assert p == prefix
            assert len(s) == 6
            assert all(c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in s)


def test_plofa_roster_ids_are_collision_free():
    adapter = build_plofa_adapter(include_players=True)
    club_ids = [eid for eid, _ in adapter.entities("club")]
    player_ids = [eid for eid, _ in adapter.entities("player")]
    assert len(club_ids) == len(set(club_ids))
    assert len(player_ids) == len(set(player_ids))


def test_club_resolves_deterministically_across_adapters():
    """A real club resolves, and resolves the same way in a fresh adapter."""
    adapter = build_plofa_adapter(include_players=True)
    name = "Justice"
    eid = adapter.to_id("club", name)
    assert eid is not None
    assert eid.startswith("CLB-")
    assert adapter.to_name("club", eid) == name
    assert build_plofa_adapter(include_players=True).to_id("club", name) == eid


def test_plofa_entities_are_equal_across_two_adapters():
    """The world ↔ data interface must be stable no matter how many times
    or in what order we build the adapter (plan §23 reversibility + §7)."""
    a = build_plofa_adapter(include_players=True)
    b = build_plofa_adapter(include_players=True)
    assert a.entities("club") == b.entities("club")
    assert a.entities("player") == b.entities("player")


def test_club_qualified_key_keeps_same_name_players_distinct():
    """A player identity is club-qualified, so it must never collapse onto a
    bare name that a different club also uses.

    The 26/27 roster happens to have unique names throughout, so the adapter
    registers players by bare name while deriving the ID from the
    club-qualified key. The invariant is asserted structurally rather than by
    hoping a duplicate shows up: every player ID must equal the digest of its
    club-qualified key, never of its bare name.
    """
    from world.ids import mint_id, player_key
    from roster_loader import RosterLoader

    loader = RosterLoader()
    clubs = sorted(loader.get_all_clubs())
    seen = 0
    for club in clubs:
        for rec in loader.get_club_players(club):
            eid = mint_id("player", player_key(club, rec.name))
            assert eid != mint_id("player", rec.name), (
                f"{club}'s {rec.name} would collapse onto a bare-name identity"
            )
            seen += 1
    assert seen == EXPECTED_PLAYERS


def test_validate_reverses_every_registered_name():
    """The integrity gate every world dataset must pass before use."""
    adapter = build_plofa_adapter(include_players=True)
    assert adapter.validate() == EXPECTED_CLUBS + EXPECTED_PLAYERS


def test_plofa_ids_sample_is_stable():
    """Guard against accidental digest-algorithm drift.

    These digests are FROZEN. If the canonical-key rule or the digest changes
    they flip, and every ID already written into any world ledger silently
    becomes unresolvable. Recompute deliberately (plan §23) — never as a side
    effect of touching `world/ids.py`.
    """
    adapter = build_plofa_adapter(include_players=True)
    frozen = {
        "Avada Zenith": "CLB-PVZC6Y",
        "Justice": "CLB-11KEJ5",
        "Tryox City": "CLB-ALKT80",
    }
    for name, expected in frozen.items():
        assert adapter.to_id("club", name) == expected, (
            f"club id for {name!r} drifted from {expected} — recompute "
            f"deliberately or world ledgers will not resolve"
        )
