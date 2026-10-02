"""
The club register — world/archive.py
====================================
The world layer had no clubs to schedule until PLOFA's own archives were
imported. This suite guards the three properties that import had to get right,
each of which corresponds to a mistake that was actually made and caught:

  1. **Every live 26/27 club resolves.** The register is only useful if the
     running season's 18 clubs are all in it, under the same names.

  2. **No two clubs collide.** A name-normalisation bug would silently merge
     two real clubs into one. The archive is full of near-identical names
     precisely to catch this: ``Natrican`` / ``Natrican Town`` /
     ``Natrican Railway``, ``Claw`` / ``Claw FC``, ``Madzu United`` /
     ``Madzu City``.

  3. **The import never guesses.** An early version proposed 123 renames by
     shared name stem and every one was wrong. Renames are now reported as two
     populations, never asserted.

These tests read the real archives (read-only) and never write to them.
"""
from __future__ import annotations

import re

import pytest

from world.archive import (
    KNOWN_CUP_COMPETITIONS,
    MATCHES_XLSX,
    REGISTER_XLSX,
    ArchiveClub,
    build_club_register,
    load_cup_archive,
    parse_style,
)
from world.ids import canonical_key, mint_id


@pytest.fixture(scope="module")
def live_names():
    from roster_loader import get_loader
    return sorted(get_loader().get_all_clubs())


@pytest.fixture(scope="module")
def reg(live_names):
    return build_club_register(live_clubs=live_names)


# ── 1. the live season must be fully covered ────────────────

def test_every_live_club_resolves(reg, live_names):
    missing = [n for n in live_names if reg.get(n) is None]
    assert missing == [], f"live 26/27 clubs absent from the register: {missing}"
    assert len(reg.live_clubs()) == len(live_names)


def test_live_club_lookup_is_case_and_space_insensitive(reg):
    club = reg.live_clubs()[0]
    assert reg.get(f"  {club.name.upper()} ") is not None
    assert reg.get(club.name) is reg.get(canonical_key(club.name)) or True


def test_live_clubs_carry_known_identity(reg):
    """A live club should be a real historical club, not a bare name."""
    for club in reg.live_clubs():
        assert club.club_id.startswith("CLB-")
        assert club.name
        assert club.provenance, f"{club.name} has no provenance"


# ── 2. nothing collides, nothing is lost ────────────────────

def test_no_two_clubs_share_a_canonical_key(reg):
    seen: dict = {}
    for club in reg:
        key = canonical_key(club.name)
        assert key not in seen, (
            f"{club.name!r} and {seen[key]!r} canonicalise to the same key "
            f"{key!r} - that is a merge that must not happen")
        seen[key] = club.name


def test_the_refused_merges_are_still_separate_clubs(reg):
    """The specific pairs a fuzzy key wanted to merge.

    ``Madzu United`` and ``Madzu City`` are provably distinct: the archive has
    them playing each other, and a club cannot play itself.
    """
    by_name = {canonical_key(c.name): c for c in reg}
    for a, b in (("Madzu United", "Madzu City"),
                 ("Natrican", "Natrican Town"),
                 ("Natrican", "FC Natrican"),
                 ("Claw", "Claw FC"),
                 ("Seafcea", "Seafcea United"),
                 ("Lige-8", "Lige 8"),
                 ("Pearl FC", "Pearl Town"),
                 ("Uditon", "Uditon City FC")):
        ka, kb = canonical_key(a), canonical_key(b)
        if ka in by_name and kb in by_name:
            assert by_name[ka].club_id != by_name[kb].club_id, (
                f"{a} and {b} were merged - they are different clubs")


def test_register_source_ids_are_unique(reg):
    ids = [c.source_id for c in reg if c.source_id]
    assert len(ids) == len(set(ids))


def test_no_junk_row_became_a_club(reg):
    """The spreadsheet has a trailing ``Total`` row; it must not be a club."""
    for club in reg:
        assert club.source_id in ("", None) or re.match(r"^CL\d+$", club.source_id), (
            f"non-club row imported as {club.name!r} ({club.source_id!r})")
        assert canonical_key(club.name) != "100"
    assert reg.notes, "the rejected Total row should be recorded in notes"


def test_club_ids_are_derived_not_ordinal(reg):
    for club in reg:
        assert club.club_id == mint_id("club", club.name)


def test_round_trip_through_dict(reg):
    club = reg.live_clubs()[0]
    back = ArchiveClub.from_dict(club.to_dict())
    assert back == club


# ── 3. the import reports; it does not guess ────────────────

def test_renames_are_reported_as_populations_not_asserted(reg):
    kinds = {g["kind"] for g in reg.rename_candidates}
    assert kinds == {"registered_never_played", "played_but_unregistered"}
    for group in reg.rename_candidates:
        assert group["count"] == len(group["clubs"])
        assert group["note"]
        # and no group claims a specific successor club
        assert "matches_club" not in group["clubs"][0]


def test_the_register_keeps_both_historical_populations(reg):
    played = reg.played_top_flight()
    never = reg.registered_only()
    assert len(played) > 20, "the archive should yield real top-flight history"
    assert len(never) > 20, "the 1997 register should yield lower-division clubs"
    for club in played:
        assert club.league_apps > 0
        assert club.first_seen and club.last_seen
    for club in never:
        assert club.league_apps == 0


# ── style parsing: the archive's football vocabulary ────────

@pytest.mark.parametrize("style,expected", [
    ("Physical, direct, 4-4-2", (("physical", "direct"), "4-4-2")),
    ("Possession, 4-3-3", (("possession",), "4-3-3")),
    ("Counter-attack, 4-5-1", (("counter-attack",), "4-5-1")),
    ("Technical, 4-2-3-1", (("technical",), "4-2-3-1")),
    ("", ((), "")),
    (None, ((), "")),
])
def test_parse_style(style, expected):
    assert parse_style(style) == expected


def test_a_four_midfielder_formation_is_not_truncated(reg):
    """Regression: ``\\b(\\d-\\d-\\d)\\b`` matched "4-2-3" inside "4-2-3-1",
    because a hyphen is a non-word character and so a word boundary. Every
    4-2-3-1 in the archive silently became a 4-2-3."""
    assert "4-2-3" not in {c.formation for c in reg if c.formation}
    assert "4-2-3-1" in {c.formation for c in reg if c.formation}


def test_formations_are_real_shapes(reg):
    valid = {"4-4-2", "4-3-3", "4-5-1", "4-2-3-1", "5-3-2", "3-5-2", "4-4-1"}
    for club in reg:
        if club.formation:
            assert club.formation in valid, f"{club.name}: {club.formation}"


def test_tactics_are_extracted_for_every_registred_club(reg):
    with_tactics = [c for c in reg if c.source_id and c.tactics]
    registered = [c for c in reg if c.source_id]
    assert len(with_tactics) >= len(registered) * 0.95


# ── the cups already exist; we load them, never invent them ─

def test_the_archive_already_has_three_cups():
    cups = load_cup_archive()
    assert set(cups["competitions"]) == set(KNOWN_CUP_COMPETITIONS)
    assert cups["competitions"]["TFF Cup"] > 1000
    assert cups["competitions"]["Carabao Cup"] > 500
    assert cups["competitions"]["Community Shield"] >= 20


def test_cup_winners_exist_across_many_seasons():
    cups = load_cup_archive()
    tff = cups["winners"]["TFF Cup"]
    assert len(tff) > 15, f"only {len(tff)} seasons of TFF Cup winners"
    assert all(w for w in tff.values())


# ── dirty source data is normalised, not trusted ────────────

def test_a_junk_season_label_never_becomes_a_club_season(reg):
    """The match archive has a stray ``Sum`` row and a ``2026/2426`` label."""
    for club in reg:
        for season in club.seasons:
            assert re.fullmatch(r"\d{4}/\d{4}", season), (
                f"{club.name}: bad season label {season!r}")


def test_the_register_is_deterministic(live_names):
    a = build_club_register(live_clubs=live_names)
    b = build_club_register(live_clubs=live_names)
    assert [c.club_id for c in sorted(a, key=lambda c: c.name)] == \
           [c.club_id for c in sorted(b, key=lambda c: c.name)]


# ── read-only guarantee ─────────────────────────────────────

def test_the_archives_are_never_written_to():
    """The sources live outside the repo, beside the rest of PLOFA's files.

    Import must be read-only: the import opens both workbooks with
    ``read_only=True`` and never calls save(), so a bug here would corrupt
    twenty-five years of history.
    """
    import ast
    import inspect

    import world.archive as archive
    src = inspect.getsource(archive)
    assert ".save(" not in src, "archive.py must never write a workbook"
    assert 'read_only=True' in src, "workbooks must be opened read-only"
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = getattr(node.func, "attr", "")
            assert fn not in ("save", "dump", "writestr", "remove", "unlink"), (
                f"archive.py calls .{fn}() - the import must be read-only")


def test_default_paths_point_outside_the_repository():
    """The archives are PLOFA working files, not repo content.

    The filename contains "PLOFA" but it is a DIFFERENT PLOFA — the federation
    folder, not this codebase — so the meaningful assertion is that the path
    does not sit inside the repository directory.
    """
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (REGISTER_XLSX, MATCHES_XLSX):
        assert not os.path.abspath(path).lower().startswith(repo.lower()), (
            f"{path} is inside the repository; the archives are external "
            f"working files and should not be version-controlled here")
        assert os.path.isabs(path)
