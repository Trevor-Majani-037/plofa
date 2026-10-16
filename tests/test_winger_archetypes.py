"""
PLOFA 26/27 — NEW WINGER ARCHETYPES
====================================

The winger role is no longer just traditional_winger + inverted_winger.
Real football runs more types, and the DNA library now reflects them:

* inside_forward     — wide goal threat drifting centrally (Vinicius, Sané)
* wide_playmaker     — dictates from the flank, switches play (Beckham, Di María)
* flair_dribbler     — beats men for fun, flair with a purpose (Neymar, Doku)

Plus the existing traditional_winger / inverted_winger / pressing_winger.

These tests pin:
  * the archetype templates (bands, tendencies, positions)
  * the specialty -> archetype resolution (grand_dribbler / dribbler now
    resolve to flair_dribbler, NOT inverted_winger)
  * the WingerSpatialProfile each archetype builds (isolation thirst,
    byline instinct, cross instinct, flank commitment)
  * flair_dribbler owns the HIGHEST dribbling band in the game — a flair
    dribbler should lead the dribbling stats.
"""
from types import SimpleNamespace

import pytest

from player_dna import DNAFactory, ArchetypeLibrary
from winger_behavior import (
    WingerBehaviorEngine, WingerSpatialProfile,
    LEFT_TOUCHLINE_ANCHOR_Y, RIGHT_TOUCHLINE_ANCHOR_Y,
)


def _player(name, position, specialties, age=24):
    dna = DNAFactory.create(name, position, list(specialties), age=age,
                            preferred_foot="right")
    return SimpleNamespace(dna=dna, position=position)


# ══════════════════════════════════════════════════════════════════════
# ARCHETYPE TEMPLATES
# ══════════════════════════════════════════════════════════════════════


def test_new_winger_archetypes_exist():
    for key in ("inside_forward", "wide_playmaker", "flair_dribbler"):
        assert key in ArchetypeLibrary.ARCHETYPES, f"missing archetype {key}"
        arch = ArchetypeLibrary.ARCHETYPES[key]
        assert "LW" in arch["positions"] and "RW" in arch["positions"]


def test_flair_dribbler_owns_the_top_dribbling_band():
    """A flair dribbler must have the highest dribbling ceiling of any
    winger archetype — this is what lets Worth lead the dribbling stats."""
    winger_archs = [k for k, a in ArchetypeLibrary.ARCHETYPES.items()
                    if "LW" in a.get("positions", []) or
                    "RW" in a.get("positions", [])]
    tops = {}
    for key in winger_archs:
        arch = ArchetypeLibrary.ARCHETYPES[key]
        band = arch.get("technical.dribbling")
        if band:
            tops[key] = band[1]
    flair_top = tops["flair_dribbler"]
    assert flair_top >= 90, f"flair_dribbler dribbling top {flair_top} too low"
    for key, top in tops.items():
        if key != "flair_dribbler":
            assert flair_top >= top, (
                f"{key} dribbling top {top} exceeds flair_dribbler {flair_top}")


def _tend(arch, key):
    return arch.get(f"tendencies.{key}")


# ══════════════════════════════════════════════════════════════════════
# ARCHETYPE TEMPLATES
# ══════════════════════════════════════════════════════════════════════


def test_new_winger_archetypes_exist():
    for key in ("inside_forward", "wide_playmaker", "flair_dribbler"):
        assert key in ArchetypeLibrary.ARCHETYPES, f"missing archetype {key}"
        arch = ArchetypeLibrary.ARCHETYPES[key]
        assert "LW" in arch["positions"] and "RW" in arch["positions"]


def test_flair_dribbler_owns_the_top_dribbling_band():
    """A flair dribbler must have the highest dribbling ceiling of any
    winger archetype — this is what lets Worth lead the dribbling stats."""
    winger_archs = [k for k, a in ArchetypeLibrary.ARCHETYPES.items()
                    if "LW" in a.get("positions", []) or
                    "RW" in a.get("positions", [])]
    tops = {}
    for key in winger_archs:
        arch = ArchetypeLibrary.ARCHETYPES[key]
        band = arch.get("technical.dribbling")
        if band:
            tops[key] = band[1]
    flair_top = tops["flair_dribbler"]
    assert flair_top >= 90, f"flair_dribbler dribbling top {flair_top} too low"
    for key, top in tops.items():
        if key != "flair_dribbler":
            assert flair_top >= top, (
                f"{key} dribbling top {top} exceeds flair_dribbler {flair_top}")


def test_inside_forward_is_a_central_drifting_goal_threat():
    arch = ArchetypeLibrary.ARCHETYPES["inside_forward"]
    assert _tend(arch, "cuts_inside") >= 0.50
    assert _tend(arch, "makes_runs_behind") >= 0.50
    assert _tend(arch, "hugs_touchline") <= 0.60
    assert _tend(arch, "crosses_from_wide") <= 0.35
    assert "technical.finishing" in arch


def test_wide_playmaker_dictates_from_wide():
    arch = ArchetypeLibrary.ARCHETYPES["wide_playmaker"]
    assert _tend(arch, "crosses_from_wide") >= 0.50
    assert _tend(arch, "plays_through_ball") >= 0.15
    assert _tend(arch, "hugs_touchline") >= 0.55
    # not a speedster: pace band top no higher than traditional winger
    trad = ArchetypeLibrary.ARCHETYPES["traditional_winger"]
    assert arch.get("physical.pace", (0, 99))[1] <= trad.get("physical.pace", (0, 99))[1]


def test_flair_dribbler_is_a_1v1_chaos_player():
    arch = ArchetypeLibrary.ARCHETYPES["flair_dribbler"]
    assert _tend(arch, "attempts_dribble") >= 0.65
    assert _tend(arch, "attacks_fullback_1v1") >= 0.70
    assert _tend(arch, "crosses_from_wide") <= 0.40   # not a crosser
    assert _tend(arch, "dives") > 0   # flair players go to ground


# ══════════════════════════════════════════════════════════════════════
# SPECIALTY -> ARCHETYPE RESOLUTION
# ══════════════════════════════════════════════════════════════════════


def test_dribbling_specialties_resolve_to_flair_dribbler():
    """grand_dribbler and dribbler must map to flair_dribbler at LW/RW,
    NOT inverted_winger. A Neymar-type dribbler is not a Salah/Robben
    wide destroyer."""
    for spec in ("grand_dribbler", "dribbler"):
        key, chosen = ArchetypeLibrary.selecting_specialty("LW", [spec])
        assert key == "flair_dribbler", f"{spec} -> {key} at LW"
        assert chosen == spec
        key, chosen = ArchetypeLibrary.selecting_specialty("RW", [spec])
        assert key == "flair_dribbler", f"{spec} -> {key} at RW"


def test_inverted_still_resolves_to_inverted_winger():
    key, _ = ArchetypeLibrary.selecting_specialty("RW", ["inverted"])
    assert key == "inverted_winger"


# ══════════════════════════════════════════════════════════════════════
# WINGER SPATIAL PROFILE (geometry that reaches the pitch)
# ══════════════════════════════════════════════════════════════════════


def test_flair_dribbler_profile_is_high_isolation_low_cross():
    p = _player("Flair Probe", "LW", ["grand_dribbler", "nutmeg_king"])
    profile = WingerBehaviorEngine.build_profile_from_dna(p)
    assert isinstance(profile, WingerSpatialProfile)
    assert profile.isolation_thirst >= 0.80, profile.isolation_thirst
    assert profile.cross_instinct <= 0.40, profile.cross_instinct
    assert profile.flank == "left"
    assert abs(profile.touchline_anchor_y - LEFT_TOUCHLINE_ANCHOR_Y) < 1e-9


def test_inside_forward_profile_cuts_inside_and_arrives_late():
    p = _player("Inside Probe", "RW", ["speedster", "creator"])
    profile = WingerBehaviorEngine.build_profile_from_dna(p)
    # inside_forward's signature: low byline instinct (cuts inside),
    # high box entry (late runs), narrow flank commitment
    assert profile.byline_instinct <= 0.60 or profile.box_entry_instinct >= 0.55


def test_archetype_override_applies_for_each_new_type():
    """Directly exercise the archetype -> profile override branches so each
    new archetype's geometry signature is pinned (non-vacuity: each branch
    must actually move a value)."""

    def override(archetype_key):
        dna = DNAFactory.create("Probe", "LW", [], age=24)
        dna.archetype = archetype_key
        p = SimpleNamespace(dna=dna, position="LW")
        return WingerBehaviorEngine.build_profile_from_dna(p)

    flair = override("flair_dribbler")
    assert flair.isolation_thirst >= 0.85
    assert flair.cross_instinct <= 0.35

    inside = override("inside_forward")
    assert inside.byline_instinct <= 0.40
    assert inside.box_entry_instinct >= 0.70

    wide = override("wide_playmaker")
    assert wide.cross_instinct >= 0.65
    assert wide.flank_commitment >= 0.90
