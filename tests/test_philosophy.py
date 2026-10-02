"""
Unit tests for philosophy layer (Checkpoint 30).
Covers: archetype registry, blend, Lever A/B/C, None-safety,
EffectiveTactics patience/starve propagation.
"""
import sys
BASE = r"C:\Users\Trevor Majani\Downloads\plofa_checkpoint6\plofa"
sys.path.insert(0, BASE)

from match_engine import TeamStyle, PlayingStyle, Intensity, TeamProfile, MatchState
from philosophy import (ClubPhilosophy, ARCHETYPES, TOTAL_FOOTBALL,
                        POSITIONAL_DOMINANCE, LOW_BLOCK, MID_TABLE_PRAGMATISM,
                        resolve_philosophy, list_archetypes)


def _raw_profile(style="tiki_taka", style_enum=None):
    style_map = {
        "tiki_taka": TeamStyle.TIKI_TAKA, "balanced": TeamStyle.BALANCED,
        "park_the_bus": TeamStyle.PARK_THE_BUS, "gegenpressing": TeamStyle.GEGENPRESSING,
    }
    return TeamProfile(
        name="Test", style=style_map.get(style, TeamStyle.BALANCED),
        playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM,
    )


def test_archetype_count():
    arches = list_archetypes()
    assert len(arches) >= 8, f"Expected >=8 archetypes, got {len(arches)}"
    print("  PASS: test_archetype_count")


def test_none_safe():
    p = _raw_profile("tiki_taka")
    assert p.philosophy is None
    assert p.hunger == 0.0
    assert p.patience == 0.0
    assert p.starve == 0.0
    assert p.has_identity is False
    orig_poss = p.possession_target
    p._apply_philosophy()
    assert p.possession_target == orig_poss, "possession_target changed with None philosophy"
    assert p.hunger == 0.0 and p.has_identity is False
    print("  PASS: test_none_safe")


def test_blend_moves_knobs():
    p = _raw_profile("balanced")
    p.philosophy = POSITIONAL_DOMINANCE
    p._apply_philosophy()
    assert p.has_identity is True
    assert p.hunger > 0.8
    assert p.patience > 0.8
    assert p.starve > 0.4
    assert p.possession_target > 65, f"Expected possession_target >65, got {p.possession_target}"
    assert p.shots_per_sequence < 0.08, f"Expected shots <0.08, got {p.shots_per_sequence}"
    print("  PASS: test_blend_moves_knobs")


def test_low_block_flattens():
    p = _raw_profile("park_the_bus")
    p.philosophy = LOW_BLOCK
    p._apply_philosophy()
    assert p.hunger < 0.2
    assert p.press_intensity < 0.3
    assert p.defensive_line < 0.15
    assert p.directness > 0.4, f"Mourinho directness {p.directness}"
    print("  PASS: test_low_block_flattens")


def test_lever_b_starve_squashes():
    from match_engine import PossessionEngine
    p = _raw_profile("park_the_bus")
    state = MatchState()
    base_n = PossessionEngine.sequence_length(p, state)
    high_starve = PossessionEngine.sequence_length(p, state, oppressor_starve=0.9)
    assert high_starve <= base_n, f"Starve should shorten: {high_starve} > {base_n}"
    print("  PASS: test_lever_b_starve_squashes")


def test_lever_c_patience_stretches():
    from match_engine import PossessionEngine
    p = _raw_profile("tiki_taka")
    p.philosophy = POSITIONAL_DOMINANCE
    p._apply_philosophy()
    state = MatchState()
    no_pat = PossessionEngine.sequence_length(p, state)
    # create profile with zero patience for comparison
    p2 = _raw_profile("tiki_taka")
    with_pat = PossessionEngine.sequence_length(p2, state)
    # with_pat (0 patience) should generally be <= no_pat (0.88 patience)
    # Allow +-2 noise
    assert no_pat >= with_pat - 2, f"Patience should stretch: {no_pat} < {with_pat - 2}"
    print("  PASS: test_lever_c_patience_stretches")


def test_effective_tactics_has_fields():
    from tactical_ai import EffectiveTactics
    eff = EffectiveTactics(
        style=TeamStyle.BALANCED, press_intensity=0.5,
        tempo=0.5, directness=0.5, defensive_line=0.5,
        shots_per_sequence=0.1, big_chance_ratio=0.3,
        press_success_rate=0.25, possession_target=50.0,
        patience=0.85, starve=0.55,
    )
    assert eff.patience == 0.85
    assert eff.starve == 0.55
    # default construction (old-style) still works with defaults
    eff2 = EffectiveTactics(
        style=TeamStyle.BALANCED, press_intensity=0.5,
        tempo=0.5, directness=0.5, defensive_line=0.5,
        shots_per_sequence=0.1, big_chance_ratio=0.3,
        press_success_rate=0.25, possession_target=50.0,
    )
    assert eff2.patience == 0.0
    assert eff2.starve == 0.0
    print("  PASS: test_effective_tactics_has_fields")


def test_resolve_philosophy():
    phi = resolve_philosophy("Barcelona", manager_label="Possession")
    assert phi is not None
    assert phi.name == "PositionalDominance"
    phi2 = resolve_philosophy("MourinhoFC", club_override={"archetype": "LowBlock", "magnitude": 0.9})
    assert phi2 is not None
    assert phi2.name == "LowBlock"
    assert abs(phi2.magnitude - 0.9) < 0.001
    assert resolve_philosophy("NobodyFC") is None
    print("  PASS: test_resolve_philosophy")


if __name__ == "__main__":
    print("test_philosophy")
    test_archetype_count()
    test_none_safe()
    test_blend_moves_knobs()
    test_low_block_flattens()
    test_lever_b_starve_squashes()
    test_lever_c_patience_stretches()
    test_effective_tactics_has_fields()
    test_resolve_philosophy()
    print("ALL PASS")
