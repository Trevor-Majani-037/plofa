"""
PLOFA WORLD — the squad composer.
=================================
world/squads.py  ·  Turns a real archived club into a squad the engine can play.

Why this is small
----------------
The heavy lifting already exists. :class:`player_dna.DNAFactory` takes a name,
a position and a list of SPECIALTIES, looks up an archetype (26 of them), and
builds all six attribute domains. :class:`player_dna.SquadBuilder` takes the
same tuples and produces the ``PlayerProfile`` objects the match engine reads.
``roster_loader.FORMATION_SLOTS`` already defines the 11-slot layout for nine
formations.

So the composer's whole job is to answer four questions per club:

  * **Who plays?**      the formation decides the eleven slots
  * **What are they?**  the club's recorded tactics decide their specialties
  * **How good are they?** the recorded budget class decides quality and depth
  * **What are they called?** a name pool, authored in the world's style

The specialties are the real lever, and it is worth being explicit about why.
``DNAFactory`` does not accept an archetype — it *derives* one from
position + specialties, taking the first specialty it recognises in
``SPECIALTY_ARCHETYPE_MAP``. So choosing ``target_man`` is how you get a
``target_man``; you never name the archetype directly. League and club identity
therefore arrive as data, which is the property that makes adding a sixth league
a config change rather than a code change.

Names: authored, and said so
----------------------------
The archive has no player-name corpus. Its ``Manager_1997`` column is dirty —
alongside real names like ``Mikel Brant`` and ``Eliza Brim`` it contains
``Chief Justice``, ``Commodore United``, ``Sir Reginald Avada`` and ``Fisher L.
Town``, i.e. role titles and club names. It is not a name source.

So :data:`FIRST_NAMES` / :data:`SURNAMES` are authored, in the style of the real
names the archive does contain (Anglo given names; surnames built from the
world's own place-name stems — Denborough, Natic, Natric, Pearl, Brim). They
are synthetic, and :func:`name_provenance` exists so nothing downstream can
mistake them for history. 128 x 176 gives 22,528 combinations, so no club ever
sees a duplicate.

Determinism
-----------
Seeded from the club's canonical ID via :func:`world.ids.canonical_key` and
crc32 — never ``hash()``, whose value for a string depends on PYTHONHASHSEED.
The same club therefore regenerates byte-identically on any machine.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from world.archive import ArchiveClub
from world.ids import canonical_key

# ─────────────────────────────────────────────
# NAME POOL — authored, not archived
# ─────────────────────────────────────────────

FIRST_NAMES: Tuple[str, ...] = (
    "Mikel", "Robert", "Hank", "Leo", "Victor", "Maria", "Eliza", "Nikolai",
    "Aldric", "Bastien", "Corvin", "Dmitri", "Edwin", "Ferran", "Gareth",
    "Halvor", "Ivo", "Jarek", "Kaspar", "Lennart", "Marek", "Nils", "Oskar",
    "Piotr", "Quill", "Rasmus", "Sander", "Tobias", "Ulrik", "Viggo",
    "Wendell", "Yorick", "Zoltan", "Anselm", "Bram", "Casper", "Dominik",
    "Emil", "Fabian", "Gunnar", "Henrik", "Ilmar", "Jonas", "Kasper", "Lauri",
    "Matthias", "Nikolaj", "Oskari", "Petri", "Risto", "Sten", "Timo",
    "Urho", "Valter", "Aksel", "Bertil", "Cato", "Dag", "Eero", "Frans",
    "Gert", "Hjalmar", "Ingvar", "Jarl", "Kjell", "Ludvig", "Mikkel", "Nils-Otto",
    "Otto", "Roope", "Sigurd", "Torsten", "Uno", "Ville", "Aatu", "Eetu",
    "Veikko", "Sakari", "Onni", "Kalle", "Jere", "Miro", "Niko", "Otso",
    "Tuomas", "Veeti", "Arttu", "Eeli", "Niilo", "Santeri", "Vili", "Aapo",
    "Elias", "Benjamin", "Theodor", "Anton", "Marcus", "Julian", "Rafael",
    "Sergio", "Pablo", "Iker", "Nuno", "Tiago", "Rui", "Andres", "Diego",
    "Emilio", "Gonzalo", "Luca", "Marco", "Matteo", "Andrea", "Davide",
    "Simone", "Federico", "Filippo", "Youssef", "Karim", "Amadou", "Sekou",
)

SURNAMES: Tuple[str, ...] = (
    "Brant", "Feld", "Harrow", "Denborough", "Natic", "Pearl", "Brim", "Natric",
    "Alder", "Bexley", "Corran", "Draycott", "Ellersby", "Fenwick", "Garrick",
    "Halstead", "Ingram", "Jarvis", "Kettering", "Lockwood", "Merrick",
    "Norwood", "Oakhurst", "Pemberton", "Quarrington", "Rushton", "Sedgwick",
    "Thornleigh", "Underhill", "Vance", "Wexford", "Yarrow", "Zennor",
    "Ashcombe", "Blackwood", "Crowther", "Dunmore", "Eastwick", "Fairbrook",
    "Grimsby", "Hollowell", "Ilminster", "Jarnac", "Kirkwall", "Langmere",
    "Marchmont", "Nayland", "Orsby", "Prestwich", "Rackham", "Stallworth",
    "Thornbury", "Uppingham", "Verwood", "Willingham", "Yelverton", "Zouche",
    "Arden", "Bexhill", "Carrington", "Dalbeattie", "Everleigh", "Fenby",
    "Gorleston", "Havering", "Inglewood", "Jesmond", "Kelsall", "Lumsdale",
    "Maddox", "Newcombe", "Ockbrook", "Padstow", "Quenby", "Ryehaven",
    "Scarborough", "Tanfield", "Uppington", "Vellacott", "Warkworth", "Yealand",
    "Ainsworth", "Bramwell", "Cholmondeley", "Dalgleish", "Eskdale", "Fitzalan",
    "Grenville", "Hawkridge", "Ingoldsby", "Jedburgh", "Kilbride", "Lorrimer",
    "Mowbray", "Netherton", "Oglethorpe", "Pemberly", "Rothwell", "Sinclair",
    "Trewhitt", "Ulverston", "Vanderhyde", "Wolstenholme", "Yaxleigh", "Zennaway",
    "Atherstone", "Bickerstaffe", "Chaldon", "Danby", "Elmswell", "Fordington",
    "Gisburn", "Hartsmere", "Inkberrow", "Jesmonde", "Kirkby", "Longton",
    "Maidstone", "Northallerton", "Oundle", "Pudsey", "Rothbury", "Snaith",
    "Toddington", "Upwell", "Wymeswold", "Yaxley", "Zeals",
    "Alderton", "Bransby", "Chillenden", "Dorchester", "Elsenham", "Faversham",
    "Goudhurst", "Harlesden", "Icklesham", "Kirkstall", "Lechlade", "Maldon",
    "Nunney", "Ockley", "Pulney", "Rye", "Shere", "Tillingham", "Upleby",
    "Wickham", "Yenston", "Zealcot",)

#: The real given names the archive does contain. Kept so
#: :func:`name_provenance` can state exactly which names are historical.
ARCHIVED_GIVEN_NAMES: Tuple[str, ...] = (
    "Mikel", "Robert", "Hank", "Leo", "Victor", "Maria", "Eliza", "Nikolai",
)


def name_provenance() -> Dict[str, object]:
    """State plainly which names are historical and which are authored."""
    return {
        "archived_given_names": list(ARCHIVED_GIVEN_NAMES),
        "authored_first_names": len(FIRST_NAMES),
        "authored_surnames": len(SURNAMES),
        "combinations": len(FIRST_NAMES) * len(SURNAMES),
        "note": "Names are AUTHORED in the style of the real names the 1997 "
                "register contains. They are not archived player data.",
    }


# ─────────────────────────────────────────────
# TACTICS → SPECIALTIES
# ─────────────────────────────────────────────
# DNAFactory derives the archetype from the first recognised specialty, so
# these pools ARE the identity control. Keys are position groups; a club's
# recorded tactics select the pool.
SPECIALTIES_BY_TACTIC: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "direct": {
        "GK": ("distribution_gk", "sweeper_keeper"),
        "DEF": ("stopper_defender", "no_nonsense_cb", "aerial_threat"),
        "MID": ("box_box", "engine", "ball_progressor"),
        "ATT": ("target_man", "aerial_threat", "clinical_finisher"),
        "WING": ("crosser", "pressing_forward"),
    },
    "attacking": {
        "GK": ("sweeper_keeper", "distribution_gk"),
        "DEF": ("ball_playing_cb", "overlapping_fullback"),
        "MID": ("press_breaker", "box_box", "ball_progressor"),
        "ATT": ("clinical_finisher", "fox_in_box", "pressing_forward"),
        "WING": ("pressing_forward", "dribbler"),
    },
    "physical": {
        "GK": ("sweeper_keeper", "distribution_gk"),
        "DEF": ("stopper_defender", "aerial_threat", "no_nonsense_cb"),
        "MID": ("interceptor", "anchor_man", "box_box"),
        "ATT": ("target_man", "aerial_threat"),
        "WING": ("pressing_forward", "crosser"),
    },
    "technical": {
        "GK": ("distribution_gk", "sweeper_keeper"),
        "DEF": ("ball_playing_cb", "sweeper_cb"),
        "MID": ("playmaker", "dl_playmaker", "creator"),
        "ATT": ("clinical_finisher", "cold_blooded"),
        "WING": ("dribbler", "inverted"),
    },
    "counter-attack": {
        "GK": ("sweeper_keeper", "distribution_gk"),
        "DEF": ("no_nonsense_cb", "stopper_defender"),
        "MID": ("ball_progressor", "box_box", "late_runner"),
        "ATT": ("speedster", "cold_blooded", "fox_in_box"),
        "WING": ("speedster", "pressing_forward"),
    },
    "possession": {
        "GK": ("distribution_gk", "sweeper_keeper"),
        "DEF": ("ball_playing_cb", "sweeper_cb"),
        "MID": ("playmaker", "dl_playmaker", "ball_progressor"),
        "ATT": ("deep_lying_forward", "clinical_finisher"),
        "WING": ("inverted", "creator"),
    },
    "fluid": {
        "GK": ("distribution_gk",),
        "DEF": ("ball_playing_cb", "underlapping_fullback"),
        "MID": ("ball_progressor", "inverted"),
        "ATT": ("late_runner", "clinical_finisher"),
        "WING": ("dribbler", "inverted", "grand_dribbler"),
    },
    "creative": {
        "GK": ("distribution_gk",),
        "DEF": ("ball_playing_cb",),
        "MID": ("creator", "playmaker", "sup_vision"),
        "ATT": ("clinical_finisher", "cold_blooded"),
        "WING": ("dribbler", "grand_dribbler", "creator"),
    },
    "defensive": {
        "GK": ("sweeper_keeper", "distribution_gk"),
        "DEF": ("no_nonsense_cb", "stopper_defender", "sweeper"),
        "MID": ("anchor_man", "interceptor"),
        "ATT": ("target_man", "late_runner"),
        "WING": ("crosser",),
    },
    "traditional": {
        "GK": ("sweeper_keeper", "distribution_gk"),
        "DEF": ("stopper_defender", "overlapping_fullback"),
        "MID": ("box_box", "engine"),
        "ATT": ("target_man", "aerial_threat"),
        "WING": ("crosser",),
    },
    "tactical": {
        "GK": ("distribution_gk", "sweeper_keeper"),
        "DEF": ("ball_playing_cb", "stopper_defender"),
        "MID": ("playmaker", "interceptor", "ball_progressor"),
        "ATT": ("clinical_finisher", "poacher"),
        "WING": ("inverted", "creator"),
    },
    "disciplined": {
        "GK": ("sweeper_keeper", "distribution_gk"),
        "DEF": ("no_nonsense_cb", "stopper_defender"),
        "MID": ("anchor_man", "interceptor"),
        "ATT": ("target_man", "late_runner"),
        "WING": ("crosser",),
    },
}

#: Only the specialties DNAFactory actually knows, used to validate a pool.
KNOWN_SPECIALTIES: Tuple[str, ...] = (
    "aerial_threat", "aggressive_fullback", "anchor_man", "ball_playing_cb",
    "ball_progressor", "ball_winner", "box_box", "clinical_finisher",
    "cold_blooded", "creator", "crosser", "deep_lying_forward",
    "distribution_gk", "dl_playmaker", "dribbler", "engine", "fox_in_box",
    "grand_box_to_box", "grand_creator", "grand_dribbler", "interceptor",
    "inverted", "late_runner", "no_nonsense_cb", "overlapping_fullback",
    "playmaker", "poacher", "press_breaker", "pressing_forward", "regista",
    "speedster", "stopper_defender", "sup_vision", "sweeper", "sweeper_cb",
    "sweeper_keeper", "target_man", "two_footed", "underlapping_fullback",
)

#: The rare tier. Only the strongest clubs get these, and rarely — that is what
#: makes a "grand creator" mean something when you see one in a match report.
GRAND_SPECIALTIES: Tuple[str, ...] = (
    "grand_box_to_box", "grand_creator", "grand_dribbler",
)

DEF_GROUP = frozenset({"CB", "LB", "RB"})
MID_GROUP = frozenset({"CDM", "CM", "CAM"})
WING_GROUP = frozenset({"LW", "RW"})
ATT_GROUP = frozenset({"ST", "CF"})


def position_group(position: str) -> str:
    if position == "GK":
        return "GK"
    if position in DEF_GROUP:
        return "DEF"
    if position in WING_GROUP:
        return "WING"
    if position in ATT_GROUP:
        return "ATT"
    return "MID"


# ─────────────────────────────────────────────
# QUALITY
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class QualityBand:
    """How strong a club's squad is, derived from the recorded budget class.

    ``scale`` exists because ``DNAFactory`` has no quality input: it derives
    every attribute from the ARCHETYPE, and archetypes encode *roles* (a
    target man is a target man at every level) rather than quality. Without an
    explicit scale a Lower-division club fields exactly the same players as an
    Elite one — which was observed: a Lower club's best player out-rated an
    Elite club's best, 82 to 80, because both drew from the same archetypes at
    similar ages.

    Club strength is therefore applied on top of the generated DNA by
    :func:`apply_club_quality`. It is a calibration step, and it is the only
    place in the world layer that adjusts an engine-produced value.
    """
    name: str
    #: multiplier applied to every generated attribute
    scale: float
    #: the age band a player is at their best in
    prime_ages: Tuple[int, int]
    #: probability a squad contains a "grand" specialty player
    grand_chance: float
    #: probability a player is two-footed
    two_footed_chance: float


#: The register's budget classes are heavily skewed (83 Lower, 9 Mid, 5 Elite,
#: 3 Upper), which is a realistic football economy: a deep pyramid of modest
#: clubs and a handful of giants. "100" was a spreadsheet totals artefact and
#: is rejected.
#:
#: The scales are a first pass, not a calibration. They set the ORDERING and a
#: plausible spread (~10% between tiers) and are the thing to tune once there
#: is a real season to compare against.
QUALITY_BANDS: Dict[str, QualityBand] = {
    "Upper": QualityBand("Upper", 1.07, (25, 30), 0.55, 0.34),
    "Elite": QualityBand("Elite", 1.03, (24, 30), 0.28, 0.26),
    "Mid":   QualityBand("Mid", 0.97, (23, 29), 0.08, 0.18),
    "Lower": QualityBand("Lower", 0.90, (22, 29), 0.02, 0.11),
}
DEFAULT_BAND = QUALITY_BANDS["Lower"]


def quality_band(budget_class: str) -> QualityBand:
    return QUALITY_BANDS.get(budget_class.strip(), DEFAULT_BAND)


#: The DNA attribute containers that :func:`apply_club_quality` scales. Named
#: explicitly so a new domain added to PlayerDNA is a loud omission here rather
#: than a silently unscaled attribute.
SCALED_DOMAINS: Tuple[str, ...] = (
    "physical", "technical", "mental", "passing", "defending", "gk_attrs",
)


def apply_club_quality(profiles, band: QualityBand):
    """Scale a built squad's attributes to the club's budget class, in place.

    Returns the same objects so it composes: ``apply_club_quality(
    SquadBuilder.build(...), band)``.

    Every numeric field of every scaled domain is multiplied by ``band.scale``.
    Non-numeric fields (labels, flags) are left alone, and the value is clamped
    to a sane 1..100 range because these attributes are read by the sensor layer
    as percentages.
    """
    for profile in profiles:
        dna = getattr(profile, "dna", None)
        if dna is None:
            continue
        for domain_name in SCALED_DOMAINS:
            domain = getattr(dna, domain_name, None)
            if domain is None:
                continue
            for field_name in getattr(domain, "__dataclass_fields__", {}):
                current = getattr(domain, field_name, None)
                if isinstance(current, bool) or not isinstance(current, (int, float)):
                    continue
                setattr(domain, field_name,
                        max(1.0, min(100.0, float(current) * band.scale)))
    return profiles


# ─────────────────────────────────────────────
# THE COMPOSED SQUAD
# ─────────────────────────────────────────────

@dataclass
class ComposedSquad:
    """A squad, ready to hand to ``SquadBuilder.build``."""
    club_id: str
    club_name: str
    formation: str
    tactics: Tuple[str, ...]
    budget_class: str
    starters: List[Tuple] = field(default_factory=list)
    substitutes: List[Tuple] = field(default_factory=list)
    superstars: List[str] = field(default_factory=list)
    set_piece_takers: List[str] = field(default_factory=list)
    seed: int = 0

    @property
    def squad_size(self) -> int:
        return len(self.starters) + len(self.substitutes)

    def to_dict(self) -> Dict[str, object]:
        return {
            "club_id": self.club_id,
            "club_name": self.club_name,
            "formation": self.formation,
            "tactics": list(self.tactics),
            "budget_class": self.budget_class,
            "starters": [list(t) for t in self.starters],
            "substitutes": [list(t) for t in self.substitutes],
            "superstars": list(self.superstars),
            "set_piece_takers": list(self.set_piece_takers),
            "seed": self.seed,
            "squad_size": self.squad_size,
        }


# ─────────────────────────────────────────────
# DETERMINISM
# ─────────────────────────────────────────────

def club_seed(club: ArchiveClub, salt: str = "") -> int:
    """A stable seed for a club, independent of PYTHONHASHSEED.

    Uses ``canonical_key`` + crc32 rather than ``hash()``: the existing draws at
    ball_vision.py:123 use ``hash()`` and therefore change with
    PYTHONHASHSEED, which is exactly the trap this avoids.
    """
    ident = f"{canonical_key(club.name)}|{club.source_id}|{salt}"
    return zlib.crc32(ident.encode("utf-8")) & 0xFFFFFFFF


class _Stream:
    """A tiny deterministic draw source (no numpy, no globals, no entropy)."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF or 1

    def _next(self) -> int:
        # xorshift32 — deterministic, dependency-free, good enough for squad
        # composition (this is not a cryptographic or statistical stream).
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._state = x or 1
        return self._state

    def below(self, n: int) -> int:
        return self._next() % n if n > 0 else 0

    def between(self, lo: int, hi: int) -> int:
        return lo + self.below(max(0, hi - lo + 1))

    def pick(self, seq: Sequence):
        return seq[self.below(len(seq))]

    def chance(self, p: float) -> bool:
        return (self._next() % 10_000) < int(p * 10_000)

    def sample(self, seq: Sequence, k: int) -> list:
        pool = list(seq)
        out = []
        for _ in range(min(k, len(pool))):
            out.append(pool.pop(self.below(len(pool))))
        return out


# ─────────────────────────────────────────────
# COMPOSING
# ─────────────────────────────────────────────

#: Fallback tactics when a club's recorded style is not in the table above.
FALLBACK_TACTIC = "tactical"   # must be a real key in SPECIALTIES_BY_TACTIC

#: Tactics recorded in the archive that have no pool yet, mapped to the nearest
#: thing we do model, so no club silently loses its identity.
TACTIC_ALIASES: Dict[str, str] = {
    "balanced": "tactical",
    "gritty": "direct",
    "counter-attacking": "counter-attack",
    "youth-focused": "fluid",
    "organized": "disciplined",
    "aggressive": "attacking",
    "fair play": "disciplined",
    "amateur spirit": "disciplined",
}


def normalise_tactics(tactics: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for raw in tactics:
        t = TACTIC_ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if t in SPECIALTIES_BY_TACTIC and t not in out:
            out.append(t)
    return tuple(out) or (FALLBACK_TACTIC,)


def _specialty_pool(tactics: Sequence[str], group: str, stream: _Stream,
                    used: Optional[set] = None) -> Tuple[str, ...]:
    """Pick this player's specialty from their club's tactical vocabulary.

    Prefers a specialty the squad does not already have. Without that, a
    three-player pool hands the same trait to a centre-back, a left-back and a
    right-back, and an XI of near-identical players is the signature of a
    generator that is not actually generating variety. The preference is soft:
    once a pool is exhausted it repeats rather than failing.
    """
    for tactic in tactics:
        pool = SPECIALTIES_BY_TACTIC.get(tactic, {}).get(group)
        if not pool:
            continue
        if used:
            fresh = [s for s in pool if s not in used]
            if fresh:
                return (stream.pick(fresh),)
        return (stream.pick(pool),)
    fallback = SPECIALTIES_BY_TACTIC["tactical"].get(group, ())
    return (stream.pick(fallback),) if fallback else ()


def _age(band: QualityBand, stream: _Stream) -> int:
    lo, hi = band.prime_ages
    if stream.chance(0.22):                      # a few veterans
        return stream.between(max(hi, 30), 35)
    if stream.chance(0.18):                      # a few youngsters
        return stream.between(18, max(18, lo - 1))
    return stream.between(lo, hi)


def _foot(stream: _Stream, band: QualityBand) -> str:
    if stream.chance(band.two_footed_chance):
        return "both"
    return "left" if stream.chance(0.26) else "right"


def _nationality(club: ArchiveClub) -> str:
    """A real nationality, rather than the hardcoded ``"Tolandian"``.

    ``roster_loader._to_tuple`` passes ``"Tolandian"`` for every player in the
    live season regardless of who they are. A composed squad at least varies it
    by the club's home city.
    """
    city = (club.city or "").strip()
    if not city:
        return "Tolandian"
    root = city.split()[0].rstrip("'")
    return f"{root}ian" if len(root) > 3 else "Tolandian"


def _unique_name(stream: _Stream, used: set) -> str:
    for _ in range(400):
        name = f"{stream.pick(FIRST_NAMES)} {stream.pick(SURNAMES)}"
        if name not in used:
            used.add(name)
            return name
    # Deterministic fallback rather than an infinite loop: qualify with a
    # numeral so the squad is still legal and still reproducible.
    base = f"{stream.pick(FIRST_NAMES)} {stream.pick(SURNAMES)}"
    n = 2
    while f"{base} {n}" in used:
        n += 1
    name = f"{base} {n}"
    used.add(name)
    return name


def compose_squad(
    club: ArchiveClub,
    *,
    bench_size: int = 9,
    formation: Optional[str] = None,
) -> ComposedSquad:
    """Build a full, deterministic squad for one real archived club.

    The output is exactly what :meth:`player_dna.SquadBuilder.build` expects:
    starters as ``(name, pos, specialties, age, nationality, foot)`` and
    substitutes with the position's customary substitution minute appended.
    """
    from roster_loader import FORMATION_SLOTS, SUB_TIMING

    shape = formation or club.formation or "4-4-2"
    if shape not in FORMATION_SLOTS:
        shape = "4-4-2"
    slots = FORMATION_SLOTS[shape]

    tactics = normalise_tactics(club.tactics)
    band = quality_band(club.budget_class)
    seed = club_seed(club)
    stream = _Stream(seed)
    used_names: set = set()
    used_specs: set = set()
    nationality = _nationality(club)

    squad = ComposedSquad(
        club_id=club.club_id,
        club_name=club.name,
        formation=shape,
        tactics=tactics,
        budget_class=club.budget_class or "Lower",
        seed=seed,
    )

    # At most ONE "grand" specialty per squad, and only for a strong club. A
    # grand player should be a note in the match report, not a squad-wide
    # habit, so the allowance is consumed the first time it is spent.
    grand_state = {"left": 1 if stream.chance(band.grand_chance) else 0}
    grand_pool = [s for t in tactics
                  for s in SPECIALTIES_BY_TACTIC.get(t, {}).get("ATT", ())
                  + SPECIALTIES_BY_TACTIC.get(t, {}).get("MID", ())
                  + SPECIALTIES_BY_TACTIC.get(t, {}).get("WING", ())
                  if s in GRAND_SPECIALTIES]

    def make(position: str) -> Tuple:
        group = position_group(position)
        if grand_state["left"] and grand_pool and group in ("ATT", "MID", "WING"):
            specs = [stream.pick(grand_pool)]
            grand_state["left"] = 0
        else:
            specs = list(_specialty_pool(tactics, group, stream, used_specs))
            used_specs.update(specs)
        if stream.chance(0.10):
            specs.append("two_footed")
        return (
            _unique_name(stream, used_names),
            position,
            tuple(specs),
            _age(band, stream),
            nationality,
            _foot(stream, band),
        )

    def make_sub(position: str) -> Tuple:
        base = make(position)
        minute = SUB_TIMING.get(position)
        return base if minute is None else base + (minute,)

    # ── generate a POOL, then SELECT the XI from it ──────────────────────
    # Picking eleven players and ageing each one independently produced 19-
    # and 18-year-olds starting for an Elite club while two strikers were 31
    # and 35. Real squads do not do that: a club carries a squad, and the best
    # available players start. So build a pool, score every player with the
    # engine's OWN age curve, and fill each formation slot from the best
    # eligible candidate. Young players then land on the bench, which is where
    # a 19-year-old belongs.
    def score(player: Tuple) -> float:
        from player_dna import DNAFactory
        base = float(DNAFactory._get_age_multiplier(int(player[3])))
        if any(s in GRAND_SPECIALTIES for s in player[2]):
            base *= 1.12
        return base

    #: How many candidates to generate per position. Comfortably above the
    #: number of slots that position fills, so selection has real choices.
    POOL_SIZES = {"GK": 3, "CB": 6, "LB": 3, "RB": 3, "CDM": 4, "CM": 7,
                  "CAM": 4, "LW": 4, "RW": 4, "ST": 6, "CF": 4}

    candidates: Dict[str, List[Tuple]] = {}
    for pos in dict.fromkeys(slots):
        candidates[pos] = [make(pos) for _ in range(POOL_SIZES.get(pos, 4))]
        candidates[pos].sort(key=score, reverse=True)

    def _select_xi(shape_slots: Sequence[str]) -> List[Tuple]:
        taken: set = set()
        picked: List[Tuple] = []
        for pos in shape_slots:
            for cand in candidates.get(pos, ()):
                if cand[0] in taken:
                    continue
                taken.add(cand[0])
                picked.append(cand)
                break
            else:  # pragma: no cover - POOL_SIZES always exceeds demand
                picked.append(make(pos))
        return picked

    squad.starters = _select_xi(slots)

    # The bench is the rest of the pool, best first, mirroring the XI's shape.
    bench_shape = ["GK", "CB", "CB", "LB", "RB", "CM", "CM", "ST", "ST"]
    squad.substitutes = [make_sub(pos) for pos in bench_shape[:bench_size]]

    squad.superstars = [t[0] for t in squad.starters
                        if any(s in GRAND_SPECIALTIES for s in t[2])]
    squad.set_piece_takers = [
        t[0] for t in squad.starters
        if t[1] in ("ST", "CF", "CAM", "LW", "RW")
    ][:3]
    return squad
