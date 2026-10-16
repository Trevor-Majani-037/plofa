"""
PLOFA 26/27 — ANTI-SOULS AND THE FLAIR_DRIBBLER SOUL
=====================================================

Positive souls lift a player toward greatness. Anti-souls are the dark
mirror: they drag probabilities DOWN (< 1.0) instead of up.

* FLAIR_DRIBBLER — the Neymar type. Beats men for fun, flair with a purpose.
  NOT a wide destroyer (that is Salah / Robben).
* ANTI_* — one dark twin for every positive soul (The Ghost, The Liability,
  The Passenger, The Squanderer, The Misser, The Traitor, The Static,
  The Blindspot, The Sieve), plus THE_HOTHEAD and THE_FRAGILE.

These tests pin:
  * FLAIR_DRIBBLER's multipliers (dribble success well above 1.0)
  * every anti-soul multiplies at least one key probability downward
  * SoulApplicator._apply_to_probability drags mult < 1.0 down (anti-soul)
    and lifts mult > 1.0 up (positive soul) — bidirectional
  * anti-souls have G < 1.00 so the attribute-ceiling lift never activates:
    they drag via multipliers, not via lifted bands
  * Zachery Worth is slotted as FLAIR_DRIBBLER, not WIDE_DESTROYER — a
    Neymar-type dribbler, not a Salah/Robben wide destroyer
"""
import ast
from pathlib import Path

import pytest

from player_soul import (
    PlayerSoul, GreatnessPillars, SoulArchetype, SoulLibrary,
    SoulApplicator,
)
from player_soul import _SOUL_ATTRIBUTE_CEILINGS


# ── helpers ──────────────────────────────────────────────────────────


def _soul(arch, pillars=(0.60, 0.60, 0.60)):
    """A weak soul: G < 1.00, so no attribute-ceiling lift activates."""
    h, t, l = pillars
    return PlayerSoul("Probe", archetype=arch,
                      pillars=GreatnessPillars(hardwork=h, talent=t, luck=l))


# ══════════════════════════════════════════════════════════════════════
# FLAIR_DRIBBLER — the Neymar type
# ═══════════════════════════════════════════════════════════════════════


def test_flair_dribbler_is_a_positive_soul():
    prof = SoulLibrary.get(SoulArchetype.FLAIR_DRIBBLER)
    assert prof.label == "Flair Dribbler"
    assert "Neymar" in prof.description
    assert "NOT a wide destroyer" in prof.description
    # Neymar type = beats men, not Salah/Robben single-channel cut
    assert prof.dribble_success_mult > 1.45
    assert prof.carry_frequency_mult > 1.25
    assert "LW" in prof.positions and "RW" in prof.positions


def test_flair_dribbler_declares_dribbling_ceilings():
    ceilings = _SOUL_ATTRIBUTE_CEILINGS[SoulArchetype.FLAIR_DRIBBLER]
    assert ceilings["technical.dribbling"] >= 95.0


# ══════════════════════════════════════════════════════════════════════
# ANTI-SOULS MULTIPLY DOWNWARD
# ═══════════════════════════════════════════════════════════════════════


_ANTI_SOULS = [
    SoulArchetype.ANTI_ATTACKING_PROPHET,
    SoulArchetype.ANTI_DEFENSIVE_PURIST,
    SoulArchetype.ANTI_MIDFIELD_PHILOSOPHER,
    SoulArchetype.ANTI_CREATIVE_ORACLE,
    SoulArchetype.ANTI_GOALSCORING_SAVANT,
    SoulArchetype.ANTI_PRESSING_EVANGELIST,
    SoulArchetype.ANTI_WIDE_DESTROYER,
    SoulArchetype.ANTI_SWEEPER_SAGE,
    SoulArchetype.ANTI_WALL,
    SoulArchetype.RED_MIST,
    SoulArchetype.GLASS,
]


def test_every_anti_soul_declares_a_profile():
    for arch in _ANTI_SOULS:
        prof = SoulLibrary.get(arch)
        assert prof is not None, f"no profile for {arch.name}"
        assert prof.label, f"{arch.name} has no label"


def test_every_anti_soul_multiplies_at_least_one_probability_down():
    """Each anti-soul must have at least one event multiplier < 1.0 —
    otherwise it is a positive soul wearing a dark name."""
    prob_keys = [
        # event probabilities (multiplied directly or via _apply_to_probability)
        "shot_quality_mult", "shot_frequency_mult", "dribble_success_mult",
        "press_success_mult", "tackle_success_mult", "interception_mult",
        "carry_frequency_mult", "pass_accuracy_mult", "big_chance_conversion",
        # psychological state boosts — < 1.0 drags the final multiplier down
        "pressure_resistance", "late_game_boost", "added_time_boost",
        "losing_state_boost",
    ]
    for arch in _ANTI_SOULS:
        prof = SoulLibrary.get(arch)
        lows = [k for k in prob_keys if getattr(prof, k, 1.0) < 1.0]
        assert lows, f"{arch.name} is not an anti-soul: nothing multiplies down"


def test_anti_souls_have_attribute_ceilings():
    """Required by the non-vacuity test (every soul declares ceilings).
    They never activate because anti-souls have G < 1.00."""
    for arch in _ANTI_SOULS:
        assert arch in _SOUL_ATTRIBUTE_CEILINGS, (
            f"{arch.name} declares no attribute ceilings")
        assert _SOUL_ATTRIBUTE_CEILINGS[arch], (
            f"{arch.name} ceilings are empty")


# ══════════════════════════════════════════════════════════════════════
# BIDIRECTIONAL PROBABILITY APPLICATION
# ═══════════════════════════════════════════════════════════════════════


def test_apply_to_probability_lifts_positive_soul():
    base, ceiling = 0.65, 0.88
    # mult > 1.0 lifts toward ceiling
    up = SoulApplicator._apply_to_probability(base, 1.50, ceiling)
    assert base < up <= ceiling


def test_apply_to_probability_drags_anti_soul_down():
    base, ceiling = 0.65, 0.88
    # mult < 1.0 drags downward
    down = SoulApplicator._apply_to_probability(base, 0.70, ceiling)
    assert 0.0 <= down < base


def test_apply_to_probability_identity_at_one():
    base = 0.65
    assert SoulApplicator._apply_to_probability(base, 1.0, 0.88) == base


def test_anti_soul_g_is_below_lift_floor():
    """Anti-souls are weak by definition: G < 1.00, so no attribute lift."""
    for arch in _ANTI_SOULS:
        g = _soul(arch).greatness_coefficient
        assert g < 1.00, f"{arch.name} G={g} is not below 1.00"


# ══════════════════════════════════════════════════════════════════════
# WORTH RE-SLOT: FLAIR_DRIBBLER, NOT WIDE_DESTROYER
# ═══════════════════════════════════════════════════════════════════════


def test_worth_is_flair_dribbler_not_wide_destroyer():
    """AST guard: Zachery Worth's soul entry in auto_run_match.py must be
    FLAIR_DRIBBLER (Neymar type), not WIDE_DESTROYER (Salah/Robben).
    Reading the source rather than importing the heavy runner."""
    path = Path(__file__).resolve().parents[1] / "auto_run_match.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == "Zachery Worth"):
                    found = True
                    # the value is a PlayerSoul(...) call; check its archetype kwarg
                    for kw in getattr(v, "keywords", []):
                        if kw.arg == "archetype":
                            val = kw.value
                            is_flair = (isinstance(val, ast.Attribute)
                                        and val.attr == "FLAIR_DRIBBLER")
                            is_wide = (isinstance(val, ast.Attribute)
                                       and val.attr == "WIDE_DESTROYER")
                            assert is_flair and not is_wide, (
                                f"Worth archetype is {ast.dump(val)}, "
                                f"expected FLAIR_DRIBBLER")
    assert found, "Zachery Worth not found in SOUL_PLAYERS"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
