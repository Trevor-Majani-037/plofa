"""Soul players: greatness must EMERGE from soulness.

Before 2026-10-14 the soul never touched the DNA at all. It was four
outcome multipliers bolted onto an ordinary character sheet, and the
pillars (hardwork/talent/luck) were computed, printed and exported while
reaching NOTHING on the pitch:

    weakest pillars  G=0.7000 QUALITY      -> modify_shot_quality 0.1000
    strongest pillars G=1.5543 TRANSCENDENT -> modify_shot_quality 0.1000   IDENTICAL

These tests pin the replacement design: the soul's archetype declares an
ATTRIBUTE CEILING, G decides how far toward it the band is lifted, and the
TALENT pillar decides where in that band the player sits.

Run _diag_soul_dna.py for the readable table this file asserts against.
"""
import dataclasses

import pytest

from player_dna import (DNAFactory, SquadBuilder, ArchetypeLibrary,
                         SOUL_LIFT_G_FLOOR, SOUL_LIFT_G_CEIL,
                         SOUL_TALENT_SPAN, SOUL_JITTER)
from player_soul import (PlayerSoul, GreatnessPillars, SoulArchetype, SoulLibrary,
                         _SOUL_ATTRIBUTE_CEILINGS)
import player_dna as pd

SPECS = {
    "ST": ["shooter", "creator"],
    "RW": ["shooter", "creator", "big_game_player"],
    "CM": ["box_box"], "CDM": ["anchor_man"], "CB": ["stopper_defender"],
}
ATTACK_FIELDS = ["technical.finishing", "mental.composure", "mental.vision",
                 "mental.positioning", "technical.dribbling"]


def make_soul(name, arch, pillars=(0.99, 0.95, 0.92)):
    hw, ta, lk = pillars
    return PlayerSoul(name, archetype=getattr(SoulArchetype, arch),
                      pillars=GreatnessPillars(hardwork=hw, talent=ta, luck=lk))


def dna_of(name, pos, soul=None, age=28, specs=None):
    return DNAFactory.create(name, pos, specs or SPECS[pos], age=age,
                             preferred_foot="left", soul=soul)


def get(dna, dotted):
    dom, fld = dotted.split(".")
    return float(getattr(getattr(dna, dom), fld))


# ── NON-VACUITY: the ceilings must name real attributes ──────────────────
# A typo'd ceiling key is silently ignored, which is the project's standing
# pathology (a mechanism that exists, is wired to nothing, and looks fine).
# `defending.pressuring` was exactly this and was caught by the check, not by
# any behavioural test.

def test_every_ceiling_key_names_a_real_attribute_field():
    valid = set()
    for cls in (pd.PhysicalAttributes, pd.TechnicalAttributes, pd.MentalAttributes,
                pd.PassingAttributes, pd.DefendingAttributes,
                pd.GoalkeeperAttributes):
        valid |= {f.name for f in dataclasses.fields(cls)}
    bad = [(a.name, k) for a, p in SoulLibrary.PROFILES.items()
           for k in p.attribute_ceilings if k.split(".", 1)[1] not in valid]
    assert bad == [], f"ceiling keys that match no attribute field (silently ignored): {bad}"


def test_every_soul_declares_at_least_one_ceiling():
    """A soul with no ceilings is the old bug wearing a new name."""
    for arch, profile in SoulLibrary.PROFILES.items():
        assert profile.attribute_ceilings, f"{arch.name} declares no attribute ceilings"


def test_every_ceiling_is_a_sane_number():
    """Sanity, not a no-op check.

    The first version of this asserted `ceiling > band top` for every
    archetype, which flagged 69 ceilings and was WRONG. Where the ceiling
    equals the archetype's band top the ceiling simply does not move, but
    SOUL_FLOOR_LIFT still raises the band's FLOOR, so the player still moves
    up — that is not a dead ceiling, it is a soul that cannot exceed an
    archetype already at the top of the scale. `poacher` finishing tops out
    at 95 and a soul ceiling of 95 is correct, not inert.
    """
    for arch, ceilings in _SOUL_ATTRIBUTE_CEILINGS.items():
        for key, val in ceilings.items():
            assert 30.0 <= val <= 99.0, f"{arch.name}.{key} = {val} is out of range"


def test_every_declared_ceiling_actually_moves_the_player_up():
    """Non-vacuity, measured rather than reasoned about.

    Each soul's PRIMARY attribute must genuinely rise when lifted. This is
    the check that catches a ceiling declared against an attribute the
    builder never calls — the `defending.pressuring` class of defect, where
    the key looks right and does nothing.
    """
    primary = {
        "GOALSCORING_SAVANT": "technical.finishing",
        "ATTACKING_PROPHET": "technical.finishing",
        "CREATIVE_ORACLE": "mental.vision",
        "MIDFIELD_PHILOSOPHER": "mental.vision",
        "SWEEPER_SAGE": "defending.interceptions",
        "DEFENSIVE_PURIST": "defending.tackling",
        "WALL": "defending.marking",
        "WIDE_DESTROYER": "technical.dribbling",
        "FLAIR_DRIBBLER": "technical.dribbling",
        "PRESSING_EVANGELIST": "mental.work_rate",
    }
    pos_for = {"GOALSCORING_SAVANT": "ST", "ATTACKING_PROPHET": "RW",
               "CREATIVE_ORACLE": "RW", "MIDFIELD_PHILOSOPHER": "CM",
               "SWEEPER_SAGE": "CDM", "DEFENSIVE_PURIST": "CB",
               "WALL": "CB", "WIDE_DESTROYER": "RW",
               "FLAIR_DRIBBLER": "RW",
               "PRESSING_EVANGELIST": "CM"}
    for arch, field in primary.items():
        assert field in SoulLibrary.PROFILES[
            getattr(SoulArchetype, arch)].attribute_ceilings, (
            f"{arch} does not declare its own primary attribute {field}")
        pos = pos_for[arch]
        ups = []
        for i in range(20):
            nm = f"PrimaryProbe {arch[:4]} {i:02d}"
            plain = dna_of(nm, pos, soul=None)
            souled = dna_of(nm, pos, soul=make_soul(nm, arch))
            ups.append(get(souled, field) - get(plain, field))
        assert min(ups) > 0.0, (
            f"{arch}.{field}: in at least one case the soul did not raise it "
            f"(min delta {min(ups):+.1f})")


# ── THE SOUL MUST ACTUALLY CHANGE THE CHARACTER SHEET ────────────────────

def test_a_soul_is_better_than_an_ordinary_player_of_the_same_position():
    """Across many names, on the attributes each soul is actually about.

    Checks each soul's PRIMARY attribute rather than one field for all of
    them: the first version asserted mental.positioning for all five
    soul/position pairs, and CREATIVE_ORACLE does not declare positioning
    because a creator is not defined by it. That was the test being wrong,
    not the ceiling table.
    """
    cases = [("ST", "GOALSCORING_SAVANT", "technical.finishing"),
             ("CB", "DEFENSIVE_PURIST", "defending.tackling"),
             ("CDM", "SWEEPER_SAGE", "defending.interceptions"),
             ("CM", "MIDFIELD_PHILOSOPHER", "mental.vision"),
             ("RW", "CREATIVE_ORACLE", "mental.vision")]
    for pos, arch, field in cases:
        lifts = []
        for i in range(25):
            nm = f"SoulProbe {i:02d}"
            plain = dna_of(nm, pos, soul=None)
            souled = dna_of(nm, pos, soul=make_soul(nm, arch))
            lifts.append(get(souled, field) - get(plain, field))
        assert min(lifts) > 0.0, (
            f"{arch} at {pos}: a soul did not beat an ordinary player on "
            f"{field} in at least one case (min delta {min(lifts):+.1f})")


def test_a_soul_is_better_on_the_attributes_its_soul_is_about():
    """Rice must be a better sweeper, not a better crosser."""
    rice = [dna_of(f"RiceProbe {i:02d}", "CDM", soul=make_soul(f"RiceProbe {i:02d}", "SWEEPER_SAGE"))
            for i in range(10)]
    plain = [dna_of(f"RiceProbe {i:02d}", "CDM") for i in range(10)]
    for field in ("defending.interceptions", "mental.positioning", "mental.anticipation"):
        s = min(get(d, field) for d in rice)
        p = max(get(d, field) for d in plain)
        assert s > p, f"SWEEPER_SAGE soul does not dominate {field}: soul min {s} vs plain max {p}"


def test_a_non_soul_attribute_is_not_lifted():
    """A soul lifts what its soul is ABOUT and leaves the rest alone."""
    lifted = SoulLibrary.PROFILES[SoulArchetype.SWEEPER_SAGE].attribute_ceilings
    assert "technical.crossing" not in lifted
    for i in range(10):
        nm = f"CrossProbe {i:02d}"
        plain = dna_of(nm, "CDM", soul=None)
        souled = dna_of(nm, "CDM", soul=make_soul(nm, "SWEEPER_SAGE"))
        assert get(souled, "technical.crossing") == get(plain, "technical.crossing"), (
            "an attribute the soul does not declare was still lifted")


# ── THE PILLARS. This is the part that was provably dead. ────────────────

def test_pillars_change_the_outcome_monotonically():
    """G=0.70 QUALITY and G=1.55 TRANSCENDENT must NOT be identical.

    This is the exact measurement that proved the old system dead.
    """
    seen = {}
    for label, pil in (("weak", (0.70, 0.70, 0.70)), ("mid", (0.85, 0.80, 0.80)),
                       ("percy", (0.99, 0.90, 0.99)), ("max", (0.99, 0.99, 0.99))):
        nm = f"PillarProbe {label}"
        s = make_soul(nm, "ATTACKING_PROPHET", pil)
        d = dna_of(nm, "RW", soul=s)
        seen[label] = (s.greatness_coefficient, get(d, "technical.finishing"),
                       get(d, "mental.composure"))
    assert len({v[1:] for v in seen.values()}) == len(seen), (
        f"pillars produce identical DNA: {seen}")

    # Monotonicity applies only where the lift is actually active (G > 1.00).
    # Below SOUL_LIFT_G_FLOOR a soul is deliberately indistinguishable from an
    # ordinary player, so `weak` and `mid` land wherever their own names hash
    # and the two are NOT ordered. Asserting they were would be asserting the
    # opposite of the design — the first version of this test did exactly that
    # and failed on [57.4, 51.1, 80.5, 92.0].
    unlifted = [seen[k][1] for k in ("weak", "mid")]
    lifted = [seen[k][1] for k in ("percy", "max")]
    assert lifted[1] > lifted[0], (
        f"the strongest soul is not the best finisher: {lifted}")
    assert min(lifted) > max(unlifted), (
        f"a lifted soul did not beat an unlifted one: lifted {lifted} "
        f"vs unlifted {unlifted}")
    assert all(v < 61.0 for v in unlifted), (
        f"an unlifted (QUALITY/ELITE) soul escaped the ordinary finishing "
        f"band: {unlifted}")


def test_talent_alone_moves_the_level():
    """The pillar, not the seed, decides where in the band a soul sits."""
    lo, hi = [], []
    for i in range(20):
        a = dna_of(f"T{i:02d}", "RW", soul=make_soul(f"T{i:02d}", "ATTACKING_PROPHET", (0.99, 0.72, 0.90)))
        b = dna_of(f"T{i:02d}", "RW", soul=make_soul(f"T{i:02d}", "ATTACKING_PROPHET", (0.99, 0.99, 0.90)))
        lo.append(get(a, "technical.finishing"))
        hi.append(get(b, "technical.finishing"))
    assert min(hi) > max(lo), (
        f"talent 0.99 is not reliably above talent 0.72: "
        f"max(low)={max(lo):.1f} vs min(high)={min(hi):.1f}")


def test_the_seed_only_moves_a_soul_by_a_bounded_jitter():
    """Guards SOUL_JITTER. If this grows, the pillar stops being the level."""
    vals = [get(dna_of(f"J{i:03d}", "RW", soul=make_soul(f"J{i:03d}", "ATTACKING_PROPHET")),
                 "technical.finishing") for i in range(200)]
    band_span = 200  # wide sample; assert the spread is far below a full band
    assert (max(vals) - min(vals)) < 30.0, (
        f"soul finishing spread {max(vals)-min(vals):.1f} is too wide — the seed "
        "is deciding the level again, not the pillar")


def test_a_quality_soul_gets_no_lift_at_all():
    """G below the floor means an ordinary character sheet."""
    for i in range(10):
        nm = f"QualityProbe {i:02d}"
        plain = dna_of(nm, "RW", soul=None)
        weak = dna_of(nm, "RW", soul=make_soul(nm, "ATTACKING_PROPHET", (0.70, 0.70, 0.70)))
        for f in ATTACK_FIELDS:
            assert get(weak, f) == get(plain, f), (
                f"a G={make_soul(nm,'ATTACKING_PROPHET',(0.70,0.70,0.70)).greatness_coefficient} "
                f"soul lifted {f}; QUALITY tier must be indistinguishable from ordinary")


# ── ORDINARY PLAYERS MUST NOT FEEL ANY OF IT ─────────────────────────────

def test_a_player_with_no_soul_is_untouched_by_the_soul_mechanism():
    """No soul -> no lift, no talent bias, no ceiling lookup."""
    saved = {}
    from player_dna import _DNA_SOUL, _DNA_ARCHETYPE_SPEC
    DNAFactory.create("Plain", "RW", SPECS["RW"], age=28)
    saved["soul"] = _DNA_SOUL["soul"]
    saved["spec"] = _DNA_ARCHETYPE_SPEC["spec"]
    assert saved["soul"] is None


def _xi(starters):
    """SquadBuilder insists on exactly 11 starters and exactly 1 GK."""
    pad = [(f"Pad {i:02d}", "CM", ["box_box"], 26) for i in range(10)]
    xi = list(starters) + pad[:10 - len(starters)]
    if not any(p == "GK" for _, p, *_ in xi):
        xi = xi[:9] + [("Keeper Pad", "GK", ["shot_stopper"], 26)] + xi[9:10]
    return xi


def test_passing_an_unrelated_soul_dict_changes_nothing():
    """SquadBuilder threading must not leak one player's soul into another."""
    plain = SquadBuilder.build("T", _xi([("Bystander", "RW", ["shooter", "creator"], 28)]))
    with_soul = SquadBuilder.build(
        "T", _xi([("Bystander", "RW", ["shooter", "creator"], 28)]),
        soul_assignments={"Soul Owner": make_soul("Soul Owner", "GOALSCORING_SAVANT")})
    a, b = plain["starters"][0].dna, with_soul["starters"][0].dna
    for f in ATTACK_FIELDS:
        assert get(a, f) == get(b, f), f"Bystander's {f} changed because someone else has a soul"


def test_squadbuilder_threads_the_soul_into_the_dna():
    """The wiring player_soul.WIRING_GUIDE described and nobody did."""
    soul = make_soul("Soul Owner", "GOALSCORING_SAVANT")
    sq = SquadBuilder.build("T", _xi([("Soul Owner", "ST", ["shooter", "creator"], 28)]),
                            soul_assignments={"Soul Owner": soul})
    assert sq["starters"][0].dna.soul is soul
    assert get(sq["starters"][0].dna, "technical.finishing") > get(
        dna_of("Soul Owner", "ST"), "technical.finishing")


def test_every_runner_passes_the_soul_registry():
    """AST guard: every SquadBuilder.build site must pass soul_assignments.

    The soul is attached to dna.soul by a LATER pass in both runners, so
    without this kwarg the build is byte-identical to having no soul at all
    and every soul silently reverts to an ordinary character sheet. No
    behavioural test can see it, because the later attachment still succeeds.

    This walks BOTH files. An earlier version of this guard checked
    auto_run_match.py alone and would have missed run_match.py entirely —
    the same one-file hole the shooter_name guard and the attacks_right
    guard each hit once already. A guard that protects one file cannot catch
    the class.

    Non-vacuity: the walk must actually FIND build calls, or a rename turns
    this into a silent no-op that reports green.
    """
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    runners = ["auto_run_match.py", "run_match.py"]
    total = 0
    for name in runners:
        path = root / name
        assert path.is_file(), f"runner {name} not found"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        builds = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", None) == "build"
                  and any(getattr(a, "id", None) == "SquadBuilder"
                          for a in getattr(n.func, "value", ast.Name(id="")) and
                          ast.walk(n.func.value))]
        assert builds, f"no SquadBuilder.build call found in {name} — the guard is vacuous"
        for call in builds:
            assert any(kw.arg == "soul_assignments" for kw in call.keywords), (
                f"{name}: a SquadBuilder.build call omits soul_assignments, so "
                "every player in that squad gets no soul ceilings")
        total += len(builds)
    assert total >= 4, f"only {total} build call(s) checked across {runners}"


# ── THE 99 CEILING, AND THE DOUBLE-CHARGE THAT CAUSED IT ────────────────

def test_no_striker_pile_up_at_the_99_ceiling():
    """Was 86% of 57 strikers reading EXACTLY 99.0."""
    vals = [get(dna_of(f"StrikerProbe {i:03d}", "ST", specs=["clinical_finisher"]), "technical.finishing")
            for i in range(300)]
    at99 = sum(1 for v in vals if v == 99.0)
    assert at99 <= 5, f"{at99}/300 strikers pinned at exactly 99.0 — pile-up is back"
    assert len(set(vals)) > 40, f"only {len(set(vals))} distinct finishing values — the band collapsed"


def test_a_specialty_is_not_charged_twice():
    """`clinical_finisher` picks complete_striker AND multiplied by 1.22.

    91 x 1.22 = 111, clamped to 99, so every draw above 81.1 collapsed onto
    the ceiling. The archetype band already embodies the specialty.
    """
    assert ArchetypeLibrary.selecting_specialty("ST", ["clinical_finisher"]) == ("complete_striker", "clinical_finisher")
    with_spec = DNAFactory.create("DoubleCharge", "ST", ["clinical_finisher"], age=27)
    without = DNAFactory.create("DoubleCharge", "ST", [], age=27)
    assert with_spec.archetype == without.archetype == "complete_striker"
    assert get(with_spec, "technical.finishing") == get(without, "technical.finishing"), (
        "the selecting specialty's multiplier is still being applied on top of "
        "the archetype band that already embodies it")


def test_a_rejected_specialty_still_applies_its_multiplier():
    """`creator` -> classic_ten is CAM-only, so at RW it selects nothing.

    Its multiplier must therefore still fire, otherwise a right-footed
    creative winger silently loses their through-ball bonus.
    """
    assert ArchetypeLibrary.selecting_specialty("RW", ["creator"]) == ("traditional_winger", None)
    a = DNAFactory.create("RejectedSpec", "RW", ["creator"], age=27)
    b = DNAFactory.create("RejectedSpec", "RW", [], age=27)
    assert get(a, "passing.through_balls") > get(b, "passing.through_balls")


# ── THE TUNING CONSTANTS THEMSELVES ──────────────────────────────────────

def test_the_soul_band_never_drops_below_the_archetype_floor():
    """A soul is recognisably its position, not a floating elite."""
    for i in range(20):
        nm = f"FloorProbe {i:02d}"
        arch = ArchetypeLibrary.ARCHETYPES["traditional_winger"]
        for key, ceil_val in SoulLibrary.PROFILES[
                SoulArchetype.ATTACKING_PROPHET].attribute_ceilings.items():
            band = arch.get(key)
            if band:
                plain = dna_of(nm, "RW", soul=None)
                souled = dna_of(nm, "RW", soul=make_soul(nm, "ATTACKING_PROPHET"))
                assert get(souled, key) >= band[0] - 12.0, (
                    f"{nm}: soul {key}={get(souled, key):.1f} fell well below the "
                    f"archetype floor {band[0]}")


def test_the_lift_floor_constant_is_in_a_sane_range():
    assert 0.0 <= pd.SOUL_FLOOR_LIFT <= 1.0
    assert SOUL_LIFT_G_FLOOR < SOUL_LIFT_G_CEIL
    assert 0.0 < SOUL_JITTER < 0.5, "SOUL_JITTER above 0.5 lets the seed pick the level"
    assert 0.0 < SOUL_TALENT_SPAN <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
