"""
PLOFA WORLD — the club register, imported from the real PLOFA archives.
=====================================================================
world/archive.py  ·  Audit §16 follow-on.

Why this exists
---------------
The world layer had no clubs to schedule. Every league, cup, calendar and
table it could build had nothing to put in it, because the only clubs that
existed were the 18 in the live 26/27 roster.

Then it turned out PLOFA already *has* a history, on disk:

  REGISTER   D:\\TOLAND FOOTBALL FEDERATION\\PLOFA-HISTORY---.xlsx
             101 clubs, each with city, district, founding year, ground,
             capacity, pitch size, kit colours, playing style, sponsor and a
             budget class. Plus a 1996 local cup and a registration timeline.

  MATCHES    ...\\PLOFA-2025-2026.COM\\FULL PLOFA HISTORY ATTEMPT.xlsx
             8,296 league matches over 27 seasons (2000/01 - 2026/27),
             3,450 cup matches, cup winners by season, title winners, and an
             all-time table of 129 clubs.

So this is an IMPORTER, not a generator. The clubs, their cities, their
grounds, their styles and twenty-five years of honours already exist.

Identity: the source is the authority
--------------------------------------
A survey of the two archives found 173 raw club spellings and 13 groups that
looked like spelling drift — ``Natrican`` / ``Natrican Town`` / ``FC Natrican``,
``Claw`` / ``Claw FC``, ``Lige-8`` / ``Lige 8``, ``Pearl FC`` / ``Pearl Town``.

They are NOT drift. The 1997 register gives every club its own ``Club_ID`` and
there are 101 of them with no duplicates, so the source has already decided:
these are 101 distinct clubs. The match archive then proves it — ``Madzu
United`` and ``Madzu City`` once played each other, and a club cannot play
itself.

And the names explain why: ``Natrican Town``, ``Natrican Railway``,
``Natrican College``, ``Natrican Wednesday``, ``Avada Mechanics``,
``Uditon Miners``, ``Port Virginia FC`` — an industrial-town football
ecosystem, where works sides and place-name variants are genuinely different
clubs.

**This module therefore never merges on a fuzzy key.** Doing so would silently
delete real clubs. :func:`world.ids.canonical_key` normalises case and
whitespace only — deliberately not stripping ``FC``/``United``/``Town`` — and
over this register it produces exactly zero collisions, which is the proof
that the conservative rule is the correct one.

Rename candidates between the 1997 register and the 27-season match archive
ARE reported (:func:`rename_candidates`) but never applied automatically: a
rename is a historical claim and should be a human decision with provenance.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from world.ids import NameAdapter, canonical_key, mint_id

# ─────────────────────────────────────────────
# SOURCE LOCATIONS
# ─────────────────────────────────────────────
# The archives live outside the repository, beside the rest of PLOFA's working
# files. Overridable so a test (or a future checkout elsewhere) can point at a
# copy; nothing here ever WRITES to them.
REGISTER_XLSX = os.environ.get(
    "PLOFA_ARCHIVE_REGISTER",
    r"D:\TOLAND FOOTBALL FEDERATION\PLOFA-HISTORY---.xlsx",
)
MATCHES_XLSX = os.environ.get(
    "PLOFA_ARCHIVE_MATCHES",
    r"D:\TOLAND FOOTBALL FEDERATION\PLOFA-2025-2026.COM"
    r"\FULL PLOFA HISTORY ATTEMPT.xlsx",
)

#: Fields the 1997 register gives us. Kept explicit so a schema change in the
#: spreadsheet surfaces as a loud KeyError rather than a silently empty club.
REGISTER_FIELDS: Tuple[str, ...] = (
    "Club_ID", "Club_Name", "City", "District", "Foundation_Year",
    "Manager_1997", "Home_Ground", "Capacity", "Surface", "Pitch_Size",
    "Kit_Colors", "Style", "Sponsor", "Budget_Class",
)

#: "Physical, direct, 4-4-2" -> tactics ("physical", "direct"), formation "4-4-2"
#:
#: The formation pattern must take the LONGEST run of digit-groups. A naive
#: ``\b(\d-\d-\d)\b`` matches "4-2-3" inside "4-2-3-1", because the hyphen is a
#: non-word character and therefore a word boundary — which silently turned
#: every 4-2-3-1 in the archive into a 4-2-3.
_FORMATION = re.compile(r"\d{1,2}-\d{1,2}-\d{1,2}(?:-\d{1,2})?")
_TACTIC = re.compile(r"^[a-z][a-z \-']*$")

#: The three competitions the archive already has, with 25+ seasons of winners.
#: Loaded, never invented — inventing them would overwrite PLOFA's own history.
KNOWN_CUP_COMPETITIONS: Tuple[str, ...] = (
    "TFF Cup", "Carabao Cup", "Community Shield",
)

#: The register carries a trailing ``Total`` row (``Club_ID='Total'``,
#: ``Club_Name='100'``, ``Budget_Class='100'``) that is a spreadsheet artefact,
#: not a club. Club IDs are CL001-style, so anything else is rejected loudly
#: rather than becoming a 128th club called "100".
_REGISTER_ID = re.compile(r"^CL\d+$")


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _int_or_none(value) -> Optional[int]:
    text = _clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def parse_style(style: str) -> Tuple[Tuple[str, ...], str]:
    """Split a register ``Style`` cell into tactics and a formation.

    ``"Physical, direct, 4-4-2"`` -> ``(("physical", "direct"), "4-4-2")``

    The tactics become archetype-selection hints downstream, and the formation
    is what the engine already uses, so keeping them apart matters.
    """
    text = _clean(style)
    if not text:
        return (), ""
    matches = _FORMATION.findall(text)
    # longest wins, so 4-2-3-1 beats the 4-2-3 prefix of itself
    formation = max(matches, key=len) if matches else ""
    parts = [p.strip().lower() for p in text.split(",")]
    tactics = tuple(
        p for p in parts
        if p and p != formation.lower() and _TACTIC.match(p)
    )
    return tactics, formation


# ─────────────────────────────────────────────
# THE CLUB
# ─────────────────────────────────────────────

@dataclass
class ArchiveClub:
    """One real PLOFA club, with everything the archives know about it."""

    club_id: str                       # canonical CLB-xxxxxx from world.ids
    source_id: str                     # the register's own Club_ID, e.g. CL005
    name: str                          # display name
    observed_names: Tuple[str, ...] = ()   # every raw spelling seen anywhere

    # register detail (present for the 101 registered clubs)
    city: str = ""
    district: str = ""
    founded: Optional[int] = None
    ground: str = ""
    capacity: Optional[int] = None
    surface: str = ""
    pitch_size: str = ""
    kit_colors: str = ""
    sponsor: str = ""
    budget_class: str = ""
    tactics: Tuple[str, ...] = ()
    formation: str = ""
    manager_1997: str = ""

    # match-archive history
    league_apps: int = 0
    first_seen: str = ""
    last_seen: str = ""
    seasons: Tuple[str, ...] = ()

    provenance: Tuple[str, ...] = ()
    in_live_roster: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "club_id": self.club_id,
            "source_id": self.source_id,
            "name": self.name,
            "observed_names": list(self.observed_names),
            "city": self.city,
            "district": self.district,
            "founded": self.founded,
            "ground": self.ground,
            "capacity": self.capacity,
            "surface": self.surface,
            "pitch_size": self.pitch_size,
            "kit_colors": self.kit_colors,
            "sponsor": self.sponsor,
            "budget_class": self.budget_class,
            "tactics": list(self.tactics),
            "formation": self.formation,
            "manager_1997": self.manager_1997,
            "league_apps": self.league_apps,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "seasons": list(self.seasons),
            "provenance": list(self.provenance),
            "in_live_roster": self.in_live_roster,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "ArchiveClub":
        data = dict(d)
        for key in ("observed_names", "tactics", "seasons", "provenance"):
            data[key] = tuple(data.get(key) or ())
        return cls(**data)  # type: ignore[arg-type]


@dataclass
class ClubRegister:
    """The full import: clubs keyed by canonical ID, plus what was rejected."""

    clubs: Dict[str, ArchiveClub] = field(default_factory=dict)
    adapter: Optional[NameAdapter] = None
    #: raw spelling -> canonical ID, for every spelling we ever saw
    spelling_index: Dict[str, str] = field(default_factory=dict)
    #: rename candidates found between the register and the match archive
    rename_candidates: List[Dict[str, object]] = field(default_factory=list)
    #: everything that looked like drift but was kept separate, with the reason
    distinct_but_similar: List[Dict[str, object]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.clubs)

    def __iter__(self):
        return iter(self.clubs.values())

    def get(self, name: str) -> Optional[ArchiveClub]:
        cid = self.spelling_index.get(canonical_key(name))
        return self.clubs.get(cid) if cid else None

    def live_clubs(self) -> List[ArchiveClub]:
        return [c for c in self.clubs.values() if c.in_live_roster]

    def played_top_flight(self) -> List[ArchiveClub]:
        return [c for c in self.clubs.values() if c.league_apps > 0]

    def registered_only(self) -> List[ArchiveClub]:
        return [c for c in self.clubs.values() if c.league_apps == 0]

    def by_id(self, club_id: str) -> Optional[ArchiveClub]:
        return self.clubs.get(club_id)

    def to_dict(self) -> Dict[str, object]:
        return {
            "clubs": [c.to_dict() for c in sorted(
                self.clubs.values(), key=lambda c: c.name)],
            "rename_candidates": self.rename_candidates,
            "distinct_but_similar": self.distinct_but_similar,
            "notes": self.notes,
        }


# ─────────────────────────────────────────────
# READING THE ARCHIVES
# ─────────────────────────────────────────────

def _sheet_rows(path: str, sheet: str) -> List[Dict[str, object]]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if sheet not in wb.sheetnames:
        raise KeyError(f"{os.path.basename(path)} has no sheet {sheet!r}; "
                       f"has {wb.sheetnames}")
    ws = wb[sheet]
    it = ws.iter_rows(values_only=True)
    header = [_clean(h) for h in next(it)]
    out: List[Dict[str, object]] = []
    for raw in it:
        if not any(c is not None and _clean(c) for c in raw):
            continue
        out.append(dict(zip(header, raw)))
    return out


def _season_is_real(value) -> bool:
    """The archive has a stray ``Sum`` row and a ``2026/2426`` label.

    Rather than trusting either, a season is real only if it looks like
    ``YYYY/YYYY``. Anything else is recorded in the register's notes instead of
    silently becoming a club's debut season.
    """
    text = _clean(value)
    return bool(re.fullmatch(r"\d{4}/\d{4}", text))


def load_match_archive(path: str = MATCHES_XLSX) -> Dict[str, object]:
    """Per-club league history from the 8,296-row match archive."""
    rows = _sheet_rows(path, "PLOFA MAIN MATCHES")
    apps: Dict[str, int] = {}
    first: Dict[str, str] = {}
    last: Dict[str, str] = {}
    seasons: Dict[str, set] = {}
    pairs: Dict[frozenset, int] = {}

    for row in rows:
        home, away = _clean(row.get("Home")), _clean(row.get("Away"))
        if not home or not away:
            continue
        season = _clean(row.get("Season"))
        date = row.get("Date")
        stamp = ""
        if date is not None:
            stamp = date.date().isoformat() if hasattr(date, "date") else _clean(date)
        for name in (home, away):
            apps[name] = apps.get(name, 0) + 1
            if stamp:
                first[name] = min(first.get(name, stamp), stamp)
                last[name] = max(last.get(name, ""), stamp)
            if _season_is_real(season):
                seasons.setdefault(name, set()).add(season)
        pairs[frozenset((home, away))] = pairs.get(frozenset((home, away)), 0) + 1

    return {
        "apps": apps,
        "first": first,
        "last": last,
        "seasons": {k: tuple(sorted(v)) for k, v in seasons.items()},
        "pairs": pairs,
    }


def load_cup_archive(path: str = MATCHES_XLSX) -> Dict[str, object]:
    """Which cups exist, and who won them, by season."""
    comps: Dict[str, int] = {}
    winners: Dict[Tuple[str, str], str] = {}
    rows = _sheet_rows(path, "CUP MATCHES")
    for row in rows:
        comp = _clean(row.get("Competition"))
        if not comp:
            continue          # the sheet has a blank trailing row
        comps[comp] = comps.get(comp, 0) + 1
        season, winner = _clean(row.get("Season")), _clean(row.get("Winner"))
        if season and winner:
            winners[(season, comp)] = winner

    by_comp: Dict[str, Dict[str, str]] = {}
    for (season, comp), winner in sorted(winners.items()):
        by_comp.setdefault(comp, {})[season] = winner
    return {"competitions": comps, "winners": by_comp}


# ─────────────────────────────────────────────
# BUILDING THE REGISTER
# ─────────────────────────────────────────────

def build_club_register(
    register_path: str = REGISTER_XLSX,
    matches_path: str = MATCHES_XLSX,
    *,
    live_clubs: Optional[Iterable[str]] = None,
) -> ClubRegister:
    """Import both archives into one canonical, auditable club register.

    The policy, stated once so it is auditable:

    * The 1997 register's ``Club_ID`` is authoritative — 101 distinct clubs.
    * ``canonical_key`` (case/whitespace only) is used to attach match-archive
      history to a register club. It never merges two register clubs.
    * A match-archive spelling with no register entry becomes its own club,
      flagged ``MATCHES_ONLY``, because inventing a merge would delete it.
    * Rename candidates are REPORTED, never applied.
    """
    reg = ClubRegister()
    reg.adapter = NameAdapter()
    live = {canonical_key(n) for n in (live_clubs or ())}

    history = load_match_archive(matches_path)
    apps: Dict[str, int] = history["apps"]                    # type: ignore[assignment]
    first: Dict[str, str] = history["first"]                  # type: ignore[assignment]
    last: Dict[str, str] = history["last"]                    # type: ignore[assignment]
    seasons: Dict[str, Tuple[str, ...]] = history["seasons"]  # type: ignore[assignment]
    pairs: Dict[frozenset, int] = history["pairs"]           # type: ignore[assignment]

    rows = _sheet_rows(register_path, "CLUB-DETAILS-1997")
    missing = [f for f in REGISTER_FIELDS
               if rows and f not in rows[0]]
    if missing:
        raise KeyError(f"CLUB-DETAILS-1997 is missing expected columns: "
                       f"{missing}")

    seen_source_ids = set()
    rejected: List[str] = []
    for row in rows:
        source_id = _clean(row.get("Club_ID"))
        name = _clean(row.get("Club_Name"))
        if not source_id or not name:
            continue
        if not _REGISTER_ID.match(source_id):
            # e.g. the spreadsheet's own "Total" row
            rejected.append(f"{source_id!r}/{name!r}")
            continue
        if source_id in seen_source_ids:
            raise ValueError(f"register repeats Club_ID {source_id!r}")
        seen_source_ids.add(source_id)

        tactics, formation = parse_style(_clean(row.get("Style")))
        key = canonical_key(name)
        club = ArchiveClub(
            club_id=mint_id("club", name),
            source_id=source_id,
            name=name,
            observed_names=(name,),
            city=_clean(row.get("City")),
            district=_clean(row.get("District")),
            founded=_int_or_none(row.get("Foundation_Year")),
            ground=_clean(row.get("Home_Ground")),
            capacity=_int_or_none(row.get("Capacity")),
            surface=_clean(row.get("Surface")),
            pitch_size=_clean(row.get("Pitch_Size")),
            kit_colors=_clean(row.get("Kit_Colors")),
            sponsor=_clean(row.get("Sponsor")),
            budget_class=_clean(row.get("Budget_Class")),
            tactics=tactics,
            formation=formation,
            manager_1997=_clean(row.get("Manager_1997")),
            provenance=(f"register:{source_id}",),
            in_live_roster=key in live,
        )
        _attach_history(club, apps, first, last, seasons)
        reg.adapter.register("club", name)
        reg.clubs[club.club_id] = club
        reg.spelling_index[key] = club.club_id

    # match-archive clubs the register never listed
    for spelling in sorted(apps):
        key = canonical_key(spelling)
        if key in reg.spelling_index:
            continue
        club = ArchiveClub(
            club_id=mint_id("club", spelling),
            source_id="",
            name=spelling,
            observed_names=(spelling,),
            provenance=("matches-only",),
            in_live_roster=key in live,
        )
        _attach_history(club, apps, first, last, seasons)
        reg.adapter.register("club", spelling)
        reg.clubs[club.club_id] = club
        reg.spelling_index[key] = club.club_id

    reg.rename_candidates = rename_candidates(reg)
    reg.distinct_but_similar = distinct_but_similar(reg)
    if rejected:
        reg.notes.append(
            f"rejected {len(rejected)} non-club register row(s): "
            f"{', '.join(rejected)}")
    return reg


def _attach_history(club: ArchiveClub, apps, first, last, seasons) -> None:
    """Fold the match archive's record for this exact spelling into the club."""
    name = club.name
    club.league_apps = int(apps.get(name, 0))
    club.first_seen = first.get(name, "")
    club.last_seen = last.get(name, "")
    club.seasons = tuple(seasons.get(name, ()))
    if club.league_apps:
        club.provenance = club.provenance + ("matches",)


def rename_candidates(reg: ClubRegister, pairs=None) -> List[Dict[str, object]]:
    """The two populations a rename question lives between — reported, not guessed.

    An earlier version of this proposed renames by shared name stem and emitted
    **123 suggestions, every one of them wrong**: ``Avada City FC`` was offered
    as the predecessor of ``Red Avada``, ``Crimson Avada``, ``Peak Avada`` and
    nine others, purely because they share the token "avada". A city with a
    dozen clubs is not a rename signal.

    Deciding which registered clubs folded, which were renamed, and which
    simply never got promoted needs a claim the data does not support: the
    match archive carries no city, founding year or register ID, so there is
    nothing to join on. So this reports the two populations and leaves the
    judgement to a human, rather than dressing a guess up as a finding.
    """
    never_played = sorted(
        ({"name": c.name, "source_id": c.source_id, "city": c.city,
          "founded": c.founded, "budget_class": c.budget_class}
         for c in reg.registered_only()),
        key=lambda r: (str(r["city"]), str(r["name"])),
    )
    matches_only = sorted(
        ({"name": c.name, "club_id": c.club_id, "league_apps": c.league_apps,
          "first_seen": c.first_seen, "last_seen": c.last_seen}
         for c in reg.clubs.values() if not c.source_id and c.league_apps),
        key=lambda r: -int(r["league_apps"]),      # type: ignore[arg-type]
    )
    out: List[Dict[str, object]] = [
        {"kind": "registered_never_played", "count": len(never_played),
         "clubs": never_played,
         "note": "in the 1997 register, absent from 27 seasons of league "
                 "football: folded, renamed, or never promoted"},
        {"kind": "played_but_unregistered", "count": len(matches_only),
         "clubs": matches_only,
         "note": "played league football but absent from the 1997 register: "
                 "founded later, or a rename of a registered club"},
    ]
    return out


def distinct_but_similar(reg: ClubRegister) -> List[Dict[str, object]]:
    """Clubs that look like spelling drift but are provably separate.

    This is the record of a merge we deliberately did NOT make. A survey
    proposed 13 such groups; the evidence rejected all of them, and this is
    where that decision is written down so it is reviewable rather than
    folklore.
    """
    out: List[Dict[str, object]] = []
    clubs = sorted(reg.clubs.values(), key=lambda c: c.name)
    for i, a in enumerate(clubs):
        for b in clubs[i + 1:]:
            ka, kb = canonical_key(a.name), canonical_key(b.name)
            if ka == kb:
                out.append({
                    "a": a.name, "b": b.name,
                    "reason": "same canonical key but distinct source IDs",
                    "a_source_id": a.source_id, "b_source_id": b.source_id,
                })
    return out
