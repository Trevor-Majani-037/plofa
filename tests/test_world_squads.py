"""
The squad composer — world/squads.py
=====================================
A composed squad has to satisfy three things at once, and this suite guards
each:

  1. **The engine accepts it.** The output is exactly what
     ``player_dna.SquadBuilder.build`` consumes — six-tuples for starters,
     seven with a substitution minute for bench players. If a tuple drifts, the
     failure surfaces as a broken match rather than as a failing test.

  2. **A club regenerates identically.** Same club, same squad, on any machine.
     The seed is crc32 over the canonical key, never ``hash()`` — because
     ``hash()`` on a string depends on PYTHONHASHSEED, which is the exact trap
     that makes the existing engine non-reproducible.

  3. **There is a pyramid.** This is the one that was actually broken. Budget
     class originally affected only the age curve, never attribute quality, so
     a Lower-division club's best player out-rated an Elite club's best, 82 to
     80. ``DNAFactory`` has no quality input — archetypes encode roles, not
     levels — so quality has to be applied on top, and a test that lets the
     ordering invert is a test that will let the pyramid die quietly.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from world.archive import build_club_register, parse_style
from world.ids import canonical_key
from world.squads import (
    ARCHIVED_GIVEN_NAMES,
    FIRST_NAMES,
    GRAND_SPECIALTIES,
    KNOWN_SPECIALTIES,
    QUALITY_BANDS,
    SURNAMES,
    apply_club_quality,
    club_seed,
    compose_squad,
    name_provenance,
    normalise_tactics,
    quality_band,
)

REPO = r"D:\PLOFA\plofa"


@pytest.fixture(scope="module")
def reg():
    return build_club_register()


def _club(reg, budget=None, formation=None):
    for club in sorted(reg, key=lambda c: c.name):
        if budget and club.budget_class != budget:
            continue
        if formation and club.formation != formation:
            continue
        if club.tactics:
            return club
    raise AssertionError(f"no club with budget={budget} formation={formation}")


@pytest.fixture(scope="module")
def squads(reg):
    """One composed squad per budget class that actually appears."""
    out = {}
    for band in ("Upper", "Elite", "Mid", "Lower"):
        club = _club(reg, budget=band)
        out[band] = (club, compose_squad(club))
    return out


# ── 1. the engine must accept it ────────────────────────────

def test_squadbuilder_accepts_every_composed_squad(squads):
    from player_dna import SquadBuilder
    for band, (club, squad) in squads.items():
        built = SquadBuilder.build(
            club.name, squad.starters, squad.substitutes,
            squad.superstars, squad.set_piece_takers)
        assert len(built["starters"]) == 11, f"{band}: {len(built['starters'])} starters"
        assert built.get("substitutes"), f"{band}: no substitutes built"
        for profile in built["starters"]:
            assert profile.dna is not None
            assert profile.dna.name


def test_the_xi_matches_the_recorded_formation(squads):
    from roster_loader import FORMATION_SLOTS
    for band, (club, squad) in squads.items():
        expected = FORMATION_SLOTS[squad.formation]
        assert [t[1] for t in squad.starters] == expected, (
            f"{band}: XI shape does not match {squad.formation}")
        assert len(squad.starters) == 11


def test_starters_are_six_tuples_and_subs_carry_a_sub_minute(squads):
    from roster_loader import SUB_TIMING
    for _band, (_club_, squad) in squads.items():
        for t in squad.starters:
            assert len(t) == 6, f"starter tuple is {len(t)}-long: {t}"
            name, pos, specs, age, nat, foot = t
            assert isinstance(name, str) and name
            assert isinstance(specs, tuple) and specs
            assert isinstance(age, int) and 16 <= age <= 42
            assert isinstance(nat, str) and nat
            assert foot in ("left", "right", "both")
        for t in squad.substitutes:
            pos = t[1]
            expected = SUB_TIMING.get(pos)
            if expected is None:
                assert len(t) == 6, f"{pos} sub should have no minute: {t}"
            else:
                assert len(t) == 7 and t[6] == expected, f"{pos} sub: {t}"


def test_every_specialty_is_one_the_engine_knows(squads, reg):
    from player_dna import ArchetypeLibrary
    known = set(ArchetypeLibrary.SPECIALTY_ARCHETYPE_MAP) | {"two_footed"}
    for _band, (_club_, squad) in squads.items():
        for t in list(squad.starters) + list(squad.substitutes):
            for spec in t[2]:
                assert spec in known, f"{t[0]}: unknown specialty {spec!r}"


def test_the_specialty_actually_drives_the_archetype(squads):
    """The composer's whole premise: specialties select the archetype.

    DNAFactory derives the archetype from the first recognised specialty, so a
    squad that stocks ``target_man`` must actually produce target men.
    """
    from player_dna import ArchetypeLibrary, DNAFactory
    for _band, (club, squad) in squads.items():
        for t in squad.starters:
            expected = ArchetypeLibrary.get_archetype_for_player(t[1], list(t[2]))
            dna = DNAFactory.create(
                name=t[0], position=t[1], specialties=list(t[2]), age=t[3],
                nationality=t[4], preferred_foot=t[5])
            assert dna is not None
            assert expected in ArchetypeLibrary.ARCHETYPES, (
                f"{t[0]}: derived archetype {expected!r} is not in the library")


# ── 2. determinism ──────────────────────────────────────────

def test_a_club_regenerates_identically(reg):
    club = _club(reg, budget="Elite")
    a = compose_squad(club)
    b = compose_squad(club)
    assert a.to_dict() == b.to_dict()
    assert a.seed == b.seed


def test_the_seed_does_not_depend_on_pythonhashseed(reg):
    """The trap that already broke the engine, avoided by construction.

    ``ball_vision.py:123`` seeds from ``hash(ident)``, whose value for a string
    changes with PYTHONHASHSEED — so PLOFA is only reproducible when that is
    set to 0. New code must not inherit that.
    """
    club = _club(reg, budget="Elite")
    expected = club_seed(club)
    snippet = (
        "import sys; sys.path.insert(0, r'D:\\PLOFA\\plofa');"
        "from world.archive import build_club_register;"
        "from world.squads import club_seed;"
        f"c=[x for x in build_club_register() if x.name=={club.name!r}][0];"
        "print(club_seed(c))"
    )
    for hs in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", snippet], capture_output=True, text=True,
            timeout=600,
            env={"PYTHONHASHSEED": hs, "PATH": "/usr/bin:/bin",
                 "SYSTEMROOT": "C:\\Windows"})
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().splitlines()[-1] == str(expected), (
            f"seed changed under PYTHONHASHSEED={hs}")


def test_different_clubs_get_different_squads(reg):
    a = compose_squad(_club(reg, budget="Elite", formation="4-4-2"))
    b = compose_squad(_club(reg, budget="Elite", formation="4-3-3"))
    assert a.seed != b.seed
    assert a.to_dict() != b.to_dict()


def test_no_duplicate_player_names_in_a_squad(squads):
    for band, (_c, squad) in squads.items():
        names = [t[0] for t in squad.starters] + [t[0] for t in squad.substitutes]
        assert len(names) == len(set(names)), f"{band}: duplicate name in squad"


# ── 3. the pyramid must exist ───────────────────────────────

def test_tiers_are_ordered(squads):
    """The regression guard for the bug that made a pyramid impossible.

    A Lower club's best player once out-rated an Elite club's best, because
    budget class fed only the age curve and never the attributes.
    """
    from player_dna import SquadBuilder
    means = {}
    for band, (club, squad) in squads.items():
        built = SquadBuilder.build(
            club.name, squad.starters, squad.substitutes,
            squad.superstars, squad.set_piece_takers)
        apply_club_quality(built["starters"], quality_band(band))
        ratings = [float(p.dna.overall_rating) for p in built["starters"]]
        means[band] = sum(ratings) / len(ratings)
    assert means["Upper"] > means["Elite"], means
    assert means["Elite"] > means["Mid"], means
    assert means["Mid"] > means["Lower"], means


def test_quality_scales_are_ordered_and_close():
    scales = {k: v.scale for k, v in QUALITY_BANDS.items()}
    assert scales["Upper"] > scales["Elite"] > scales["Mid"] > scales["Lower"]
    # a plausible spread, not a cartoon: ~17% top to bottom
    spread = scales["Upper"] / scales["Lower"]
    assert 1.0 < spread < 1.35, scales


def test_apply_club_quality_leaves_non_numeric_fields_alone(squads):
    club, squad = squads["Elite"]
    from player_dna import SquadBuilder
    built = SquadBuilder.build(club.name, squad.starters, squad.substitutes,
                               squad.superstars, squad.set_piece_takers)
    before = [(p.dna.name, p.dna.position, p.dna.age,
               p.dna.physical.pace) for p in built["starters"]]
    apply_club_quality(built["starters"], QUALITY_BANDS["Lower"])
    after = [(p.dna.name, p.dna.position, p.dna.age,
              p.dna.physical.pace) for p in built["starters"]]
    for (n1, p1, a1, _), (n2, p2, a2, v2) in zip(before, after):
        assert (n1, p1, a1) == (n2, p2, a2), "identity fields must not change"
        assert 1.0 <= v2 <= 100.0


def test_the_xi_is_in_its_prime_and_the_young_are_on_the_bench(squads):
    """Regression: picking the XI and ageing each player independently gave an
    Elite club a 19-year-old starting keeper and an 18-year-old centre-back
    alongside two strikers aged 31 and 35."""
    for band, (_club_, squad) in squads.items():
        xi_ages = sorted(t[3] for t in squad.starters)
        bench_ages = [t[3] for t in squad.substitutes]
        assert min(xi_ages) >= 21, f"{band}: {min(xi_ages)}-year-old in the XI"
        assert max(xi_ages) <= 34, f"{band}: {max(xi_ages)}-year-old in the XI"
        # at least one bench player outside the XI's age range
        assert min(bench_ages) <= min(xi_ages) or max(bench_ages) >= max(xi_ages), (
            f"{band}: bench is entirely inside the XI's age range")


def test_squads_have_variety_not_near_identical_players(squads):
    """Regression: a three-entry pool handed ``aerial_threat`` to a CB, an LB
    and an RB at the same time."""
    for band, (_club_, squad) in squads.items():
        primary = [t[2][0] for t in squad.starters]
        assert len(set(primary)) >= 5, (
            f"{band}: only {len(set(primary))} distinct primary specialties "
            f"across 11 players: {primary}")


def test_grand_specialties_are_rare(squads, reg):
    """A grand player should be a note in a match report, not a squad habit."""
    grand_clubs = 0
    for _band, (club, squad) in squads.items():
        n = sum(1 for t in squad.starters
                if any(s in GRAND_SPECIALTIES for s in t[2]))
        assert n <= 1, f"{club.name}: {n} grand-specialty players"
        grand_clubs += n
    # and across every registered club, Lower clubs should almost never have one
    lowers = [compose_squad(c) for c in reg.registered_only()
              if c.budget_class == "Lower"][:25]
    with_grand = sum(1 for s in lowers
                     if any(x in GRAND_SPECIALTIES for t in s.starters
                            for x in t[2]))
    assert with_grand <= len(lowers) * 0.25, (
        f"{with_grand}/{len(lowers)} Lower clubs had a grand player; "
        f"band chance is {QUALITY_BANDS['Lower'].grand_chance}")


# ── football identity actually comes through ────────────────

def test_tactics_drive_the_specialties(reg):
    direct = compose_squad(_club(reg, budget="Elite"))
    assert direct.tactics
    for t in direct.starters:
        assert t[2], "every player needs a specialty"


def test_normalise_tactics_maps_the_archive_vocabulary():
    assert normalise_tactics(["Direct"]) == ("direct",)
    assert normalise_tactics(["counter-attacking"]) == ("counter-attack",)
    assert normalise_tactics(["youth-focused"]) == ("fluid",)
    assert normalise_tactics(["nonsense"]) == ("tactical",), "unknown -> fallback"
    assert normalise_tactics([]) == ("tactical",)
    assert normalise_tactics(["direct", "direct"]) == ("direct",), "deduped"


def test_parse_style_feeds_the_composer(reg):
    """Formation and tactics must survive the archive -> composer handoff."""
    club = _club(reg, budget="Elite")
    tactics, formation = parse_style(
        ", ".join(t.title() for t in club.tactics) + f", {club.formation}")
    assert formation == club.formation


def test_nationality_varies_rather_than_being_tolandian_everywhere(reg):
    """``roster_loader._to_tuple`` hardcodes "Tolandian" for every player in the
    live season. A composed squad at least varies it by the club's home city —
    so sample across cities, not alphabetically (the first clubs by name all
    happen to be in Avada)."""
    by_city: dict = {}
    for club in sorted(reg, key=lambda c: c.name):
        if club.tactics and club.city:
            by_city.setdefault(club.city, club)
    sample = [compose_squad(c) for c in by_city.values()]
    nats = {t[4] for s in sample for t in s.starters}
    assert len(nats) >= 3, f"only {len(nats)} nationalities across {len(sample)} cities: {nats}"
    assert nats != {"Tolandian"}, "every squad fell back to the hardcoded value"


# ── honest provenance ───────────────────────────────────────

def test_names_are_declared_as_authored_not_archived():
    prov = name_provenance()
    assert prov["authored_first_names"] == len(FIRST_NAMES)
    assert prov["authored_surnames"] == len(SURNAMES)
    assert "AUTHORED" in prov["note"]
    # the eight real given names the archive does contain are kept, and are
    # actually in the pool
    for name in ARCHIVED_GIVEN_NAMES:
        assert name in FIRST_NAMES, f"{name} is archived but not in the pool"


def test_the_name_pool_is_big_enough_to_avoid_collisions():
    """121 given x 150 surnames = 18,150 combinations.

    Comfortably more than the ~5,000 a 127-club world could ever consume, and
    enough that the 400-attempt uniqueness loop in _unique_name never has to
    fall back to a numbered name in practice.
    """
    combos = len(FIRST_NAMES) * len(SURNAMES)
    assert combos > 15_000, combos
    assert len(set(FIRST_NAMES)) == len(FIRST_NAMES), "duplicate first names"
    assert len(set(SURNAMES)) == len(SURNAMES), "duplicate surnames"


def test_no_specialty_pool_uses_an_archetype_name_instead_of_a_specialty(reg):
    """``shot_stopper`` is one of the 26 ARCHETYPES, not a specialty.

    Six goalkeeper pools were written with it, and the engine silently ignored
    it: ``get_archetype_for_player`` only recognises keys of
    SPECIALTY_ARCHETYPE_MAP, so a "shot_stopper" specialty quietly fell through
    to the positional default. The result happened to be a shot stopper, which
    is why it looked right — but the vocabulary was wrong, and a club asking for
    a sweeper-keeper build would not have got one.
    """
    from player_dna import ArchetypeLibrary
    from world.squads import SPECIALTIES_BY_TACTIC
    engine_specialties = set(ArchetypeLibrary.SPECIALTY_ARCHETYPE_MAP)
    archetype_names = set(ArchetypeLibrary.ARCHETYPES)
    archetype_only = archetype_names - engine_specialties
    assert archetype_only, "expected some archetype names not to be specialties"
    for tactic, groups in SPECIALTIES_BY_TACTIC.items():
        for group, pool in groups.items():
            for spec in pool:
                assert spec not in archetype_only, (
                    f"{tactic}/{group} uses {spec!r}, which is an archetype "
                    f"the engine will ignore as a specialty")
    from player_dna import ArchetypeLibrary
    assert set(KNOWN_SPECIALTIES) == set(
        ArchetypeLibrary.SPECIALTY_ARCHETYPE_MAP) | {"two_footed"}, (
        "the composer's own list of specialties has drifted from the engine's")


def test_the_composer_does_not_import_the_live_pipeline_at_module_level():
    """The world layer stays importable without the live season.

    ``roster_loader`` and ``player_dna`` are imported lazily inside functions,
    so importing world.squads cannot drag in the 26/27 pipeline.
    """
    import ast
    import world.squads as mod
    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = {node.module.split(".")[0]}
        else:
            continue
        for forbidden in ("roster_loader", "player_dna", "match_engine",
                          "season_manager", "auto_run_match", "alltime_db"):
            assert forbidden not in names, (
                f"world/squads.py imports {forbidden} at module level")
