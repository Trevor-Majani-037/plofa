"""
PLOFA WORLD — stable canonical IDs and the reversible name↔ID adapter.
=======================================================================
world/ids.py  ·  Phase 1 of the World Football plan.

Establishes the identity foundation everything else depends on:

* ``mint_id(kind, key)``  — deterministic, insertion-order-independent ID.
  Equivalent keys (case/whitespace variants) always mint one ID, and the
  same key always mints the same ID regardless of registry build order.
* ``NameAdapter``         — bidirectional name↔ID index over a registered
  dataset (display names + aliases map to IDs, IDs map back to a canonical
  display name, with loud collision detection).

Design contract (world_layer audit §11 / §23):
  * IDs are DERIVED from a canonical key, never sequential. An ordinal
    scheme would silently renumber every entity when one is inserted ahead
    of an earlier one.
  * ``canonical_key`` is case/whitespace insensitive, so "Hartwell City",
    "hartwell  city" and "  HARTWELL CITY " are one identity.
  * Diacritics are PRESERVED: a real identity never loses distinguishing
    marks here. Same-person variants (e.g. "Roy Steupy" vs "Roy Steupý")
    are linked through the alias table, exactly like the existing
    alltime_db.alias layer.
  * A digest collision raises loudly instead of silently merging two
    distinct identities.

Entity ID grammar:  ``KIND-PREFIX-6CHARS`` e.g. ``CLB-8F3Q2Z`` for a club
(6-character base36 of crc32 over the canonical key; 2^32 space).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ─────────────────────────────────────────────
# ID GRAMMAR
# ─────────────────────────────────────────────

ID_PREFIXES = {
    "country": "CTR",
    "city": "CTY",
    "stadium": "STA",
    "club": "CLB",
    "player": "PLY",
    "manager": "MGR",
    "referee": "REF",
    "competition": "CMP",
}
ID_KINDS: Tuple[str, ...] = tuple(ID_PREFIXES)

_DIGEST_WIDTH = 6
_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class IdentityCollisionError(ValueError):
    """Two distinct canonical keys resolved to the same ID for one kind."""


# ─────────────────────────────────────────────
# CANONICAL KEYS
# ─────────────────────────────────────────────

def canonical_key(name: str) -> str:
    """Normalise a display name into the canonical identity key.

    Case- and whitespace-insensitive; diacritics are preserved (see the
    module docstring).
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}")
    if not name.strip():
        raise ValueError("name must be non-empty")
    return " ".join(name.casefold().split())


def _base36(n: int, width: int) -> str:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return "0" * width
    chars = []
    while n > 0:
        chars.append(_BASE36[n % 36])
        n //= 36
    while len(chars) < width:
        chars.append("0")
    return "".join(reversed(chars[-width:]))


def mint_id(kind: str, key: str) -> str:
    """Return the canonical ID for a key of the given kind.

    The key is normalised through :func:`canonical_key` first, so equivalent
    spellings always mint one ID::

        mint_id("club", "Hartwell City") == mint_id("club", "  hartwell  city ")

    This makes ``mint_id`` safe to call directly — a caller cannot forget to
    canonicalise and silently mint a second identity for a person it already
    has. It is a no-op for keys that are already canonical, so IDs minted
    before this normalisation was introduced are unchanged.
    """
    if kind not in ID_PREFIXES:
        raise ValueError(
            f"unknown kind '{kind}'; known kinds: {', '.join(ID_KINDS)}"
        )
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be a non-empty string")
    canonical = canonical_key(key)
    digest = _base36(zlib.crc32(canonical.encode("utf-8")), _DIGEST_WIDTH)
    return f"{ID_PREFIXES[kind]}-{digest}"


# ─────────────────────────────────────────────
# NAME ↔ ID ADAPTER
# ─────────────────────────────────────────────

@dataclass
class _Entity:
    kind: str
    id: str
    canonical_key: str
    canonical_name: str
    names: Set[str] = field(default_factory=set)


class NameAdapter:
    """Bidirectional name↔ID index over a registered world dataset.

    * ``register`` attaches display names / aliases to an entity ID. The ID
      is derived from the first name's canonical key unless ``id_override``
      pins it (the mechanism for renames and same-person aliases).
    * ``to_id`` is a pure function of the normalised name/key.
    * ``to_name`` is a reverse lookup over the registry — reversible for
      every registered entity.
    * ``register`` is idempotent for the same key and raises
      ``IdentityCollisionError`` on a real digest collision or when a name
      tries to map to two different IDs.
    """

    def __init__(self) -> None:
        self._entities: Dict[str, Dict[str, _Entity]] = {k: {} for k in ID_KINDS}
        self._by_key: Dict[str, Dict[str, str]] = {k: {} for k in ID_KINDS}

    # -- registration --------------------------------------------------

    def register(
        self,
        kind: str,
        display_name: str,
        aliases: Iterable[str] = (),
        id_override: Optional[str] = None,
    ) -> str:
        """Register an entity under its display name (+ aliases).

        Returns the resolved canonical ID. Re-registering the same name
        (or an alias of an existing identity) is a no-op; attaching a
        *different* canonical key to an already-owned ID raises
        ``IdentityCollisionError`` — including when ``id_override`` is used,
        because a pinned ID is exactly where a silent merge would do the most
        damage. To deliberately re-canonicalise an existing identity (a real
        rename, e.g. "Victor James" → "Rayan Victor James") use
        :meth:`recannonicalise`, which states the intent in the call site.
        """
        if kind not in ID_PREFIXES:
            raise ValueError(
                f"unknown kind '{kind}'; known kinds: {', '.join(ID_KINDS)}"
            )
        key = canonical_key(display_name)
        eid = id_override if id_override is not None else mint_id(kind, key)

        ent = self._entities[kind].get(eid)
        if ent is None:
            ent = _Entity(
                kind=kind, id=eid, canonical_key=key, canonical_name=display_name,
            )
            self._entities[kind][eid] = ent
        elif ent.canonical_key != key:
            raise IdentityCollisionError(
                f"{kind} '{display_name}' (key '{key}') resolves to {eid} already "
                f"owned by '{ent.canonical_name}' (key '{ent.canonical_key}'); "
                f"use recannonicalise() if this is an intended rename"
            )

        for name in (display_name, *tuple(aliases)):
            nk = canonical_key(name)
            existing = self._by_key[kind].get(nk)
            if existing is not None and existing != eid:
                raise IdentityCollisionError(
                    f"{kind} name '{name}' already maps to {existing}; "
                    f"cannot also map to {eid}"
                )
            self._by_key[kind][nk] = eid
            ent.names.add(name)
        return eid

    def recannonicalise(
        self,
        kind: str,
        id: str,
        new_display_name: str,
        *,
        keep_old_name_as_alias: bool = True,
    ) -> str:
        """Deliberately move an identity to a new canonical name, same ID.

        This is the ONLY way an existing entity's canonical key changes — a
        rename is a real-world fact about one person, not an identity merge, so
        it must be stated explicitly rather than smuggled in through
        ``register(id_override=...)``.

        The previous name keeps resolving to the ID (as an alias) by default, so
        anything already written to a world ledger under the old spelling stays
        resolvable — the same retro-compatibility rule as ``alltime_db.alias``.
        """
        ent = self._entities.get(kind, {}).get(id)
        if ent is None:
            raise KeyError(f"no {kind} registered with id '{id}'")

        new_key = canonical_key(new_display_name)
        if new_key == ent.canonical_key:
            return id

        owner = self._by_key[kind].get(new_key)
        if owner is not None and owner != id:
            raise IdentityCollisionError(
                f"{kind} name '{new_display_name}' already belongs to {owner}; "
                f"cannot re-canonicalise {id} onto it"
            )

        # re-point the index at the new key, then adopt it
        if keep_old_name_as_alias:
            self._by_key[kind][ent.canonical_key] = id
        else:
            self._by_key[kind].pop(ent.canonical_key, None)
        ent.names.discard(ent.canonical_name)
        ent.canonical_key = new_key
        ent.canonical_name = new_display_name
        ent.names.add(new_display_name)
        self._by_key[kind][new_key] = id
        return id

    # -- lookups -------------------------------------------------------

    def to_id(self, kind: str, name: str) -> Optional[str]:
        """Resolve a display name / alias to its canonical ID."""
        if kind not in ID_PREFIXES:
            return None
        return self._by_key[kind].get(canonical_key(name))

    def to_name(self, kind: str, id: str) -> Optional[str]:
        """Resolve a canonical ID back to its canonical display name."""
        ent = self._entities.get(kind, {}).get(id)
        return ent.canonical_name if ent is not None else None

    def canonical_key_for(self, kind: str, id: str) -> Optional[str]:
        ent = self._entities.get(kind, {}).get(id)
        return ent.canonical_key if ent is not None else None

    def names(self, kind: str, id: str) -> List[str]:
        """All known names (display + aliases) for an ID, sorted."""
        ent = self._entities.get(kind, {}).get(id)
        if ent is None:
            return []
        return sorted(ent.names)

    def entities(self, kind: str) -> List[Tuple[str, str]]:
        """[(id, canonical_name), ...] for a kind, sorted by id."""
        if kind not in ID_PREFIXES:
            return []
        return sorted(
            (eid, ent.canonical_name) for eid, ent in self._entities[kind].items()
        )

    def total(self) -> int:
        return sum(len(v) for v in self._entities.values())

    # -- integrity -----------------------------------------------------

    def validate(self) -> int:
        """Assert every registered name reverses back to its own ID.

        Returns the number of entities registered. Raise on inconsistency.
        """
        n = 0
        for kind in ID_KINDS:
            for eid, ent in self._entities[kind].items():
                n += 1
                if self._by_key[kind].get(ent.canonical_key) != eid:
                    raise IdentityCollisionError(
                        f"{kind} {eid}: canonical key '{ent.canonical_key}' "
                        f"does not reverse to its own ID"
                    )
                for name in ent.names:
                    if self.to_id(kind, name) != eid:
                        raise IdentityCollisionError(
                            f"{kind} {eid}: name '{name}' does not reverse to its ID"
                        )
        return n


# ─────────────────────────────────────────────
# ROSTER ADAPTER (PLOFA 26/27, read-only)
# ─────────────────────────────────────────────

def player_key(club: str, player_name: str) -> str:
    """Canonical key for a player: club-qualified to keep same-named
    players in different clubs distinct (plan §23: names are never primary
    identity, and a bare name is ambiguous)."""
    return canonical_key(f"{club} :: {player_name}")


def build_plofa_adapter(loader=None, include_players: bool = True) -> NameAdapter:
    """Build a NameAdapter over the live PLOFA 26/27 roster.

    Club IDs derive from the club name; player IDs derive from the club-
    qualified key so two players of the same name in different clubs stay
    distinct. A bare player name is indexed only when it is unambiguous in
    the roster; ambiguous bare names resolve via ``"Club :: Name"``.

    Read-only provenance: this reads ``PLOFA-2026-2027.xlsx`` and never
    writes to any authoritative 26/27 ledger.
    """
    from roster_loader import RosterLoader  # lazy: heavy dependency

    loader = loader if loader is not None else RosterLoader()
    clubs = sorted(loader.get_all_clubs())

    # which bare player names are ambiguous?
    name_clubs: Dict[str, Set[str]] = {}
    if include_players:
        for club in clubs:
            for rec in loader.get_club_players(club):
                name_clubs.setdefault(canonical_key(rec.name), set()).add(club)

    adapter = NameAdapter()
    for club in clubs:
        adapter.register("club", club)
        if not include_players:
            continue
        for rec in loader.get_club_players(club):
            identity_key = player_key(club, rec.name)
            eid = mint_id("player", identity_key)
            if len(name_clubs.get(canonical_key(rec.name), ())) == 1:
                adapter.register("player", rec.name, id_override=eid)
            else:
                adapter.register("player", f"{club} :: {rec.name}", id_override=eid)
    return adapter