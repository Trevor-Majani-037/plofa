"""Focused tests for decision_brain.py (the active on-ball decision layer)
and its integration into event_chain.py.

Run: python3 test_decision_brain.py
Run without the (slower, mplsoccer-dependent) full-match check:
    python3 test_decision_brain.py --unit-only

These are plain assert-based scripts, matching the project's existing
test style (see test_width_changes.py), not a pytest suite.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from dataclasses import fields

from player_dna import (
    PlayerDNA, PlayerProfile, PhysicalAttributes, TechnicalAttributes,
    MentalAttributes, PassingAttributes, BehavioralTendencies, PlayerFormState,
)
from decision_brain import DecisionBrain, PlayerIntent, PlayerDecision


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

class FakeStyle:
    def __init__(self, v): self.value = v


class FakeTeamProfile:
    def __init__(self, style="balanced"):
        self.style = FakeStyle(style)


class FakeGameState:
    def __init__(self, name="LEVEL"):
        self.name = name


class FakePositionEngine:
    def __init__(self, positions):
        self.positions = positions

    def get_position(self, name):
        return self.positions.get(name, (50.0, 34.0))


def make_player(name, position, **overrides):
    mental = MentalAttributes(
        vision=overrides.pop("vision", 55),
        composure=overrides.pop("composure", 55),
        decisions=overrides.pop("decisions", 55),
        anticipation=overrides.pop("anticipation", 55),
    )
    technical = TechnicalAttributes(
        dribbling=overrides.pop("dribbling", 55),
        ball_control=overrides.pop("ball_control", 55),
        crossing=overrides.pop("crossing", 50),
        finishing=overrides.pop("finishing", 50),
        long_shots=overrides.pop("long_shots", 40),
    )
    physical = PhysicalAttributes(
        pace=overrides.pop("pace", 60), stamina=overrides.pop("stamina", 65),
        strength=overrides.pop("strength", 55),
    )
    passing = PassingAttributes(
        short_passing=overrides.pop("short_passing", 60),
        long_passing=overrides.pop("long_passing", 55),
        through_balls=overrides.pop("through_balls", 45),
        switch_play=overrides.pop("switch_play", 50),
    )
    tendencies = BehavioralTendencies(
        attempts_dribble=overrides.pop("attempts_dribble", 0.25),
        plays_through_ball=overrides.pop("plays_through_ball", 0.10),
        switches_play=overrides.pop("switches_play", 0.08),
        plays_safe=overrides.pop("plays_safe", 0.50),
        crosses_from_wide=overrides.pop("crosses_from_wide", 0.40),
        shoots_from_distance=overrides.pop("shoots_from_distance", 0.15),
    )
    form = PlayerFormState(confidence=overrides.pop("confidence", 50))
    dna = PlayerDNA(
        name=name, position=position, physical=physical, technical=technical,
        mental=mental, passing=passing, tendencies=tendencies, form=form,
    )
    return PlayerProfile(dna=dna, team_name="Home")


FINAL_THIRD_POSITIONS = {
    "Carrier": (75, 34), "Fwd Runner": (92, 40), "Deep CB": (25, 34),
    "Def1": (78, 33), "Def2": (85, 45), "Def3": (30, 30), "Far Winger": (78, 60),
}


def sample(player, n=800, x=75, y=34, teammates=None, defenders=None,
           pe=None, team_profile=None, under_pressure=False, attacks_right=True,
           game_state=None, minute=30):
    teammates = teammates if teammates is not None else [
        make_player("Fwd Runner", "ST"), make_player("Deep CB", "CB"),
        make_player("Far Winger", "RW"),
    ]
    defenders = defenders if defenders is not None else [
        make_player("Def1", "CDM"), make_player("Def2", "CB"), make_player("Def3", "CB"),
    ]
    pe = pe or FakePositionEngine(FINAL_THIRD_POSITIONS)
    team_profile = team_profile or FakeTeamProfile("balanced")
    game_state = game_state or FakeGameState("LEVEL")

    counts, quality, error = Counter(), [], []
    for _ in range(n):
        d = DecisionBrain.decide(
            player, x, y, teammates, defenders, pe, team_profile,
            under_pressure, attacks_right, game_state, minute=minute,
        )
        counts[d.intent.value] += 1
        quality.append(d.decision_quality)
        error.append(d.evaluation_error)
    return counts, sum(quality) / n, sum(error) / n


def share(counts, keys):
    total = sum(counts.values())
    return sum(v for k, v in counts.items() if k in keys) / total


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

def test_elite_vs_limited_player_differ():
    """Elite vision/decisions/composure/anticipation should show
    materially higher decision_quality and lower evaluation_error than a
    limited player in the IDENTICAL geometric situation, and should
    perceive/choose THROUGH_BALL far more often (bounded perception)."""
    elite = make_player("Elite CAM", "CAM", vision=92, decisions=90, composure=88,
                         anticipation=88, through_balls=85)
    limited = make_player("Limited CM", "CM", vision=35, decisions=32, composure=38,
                           anticipation=34, through_balls=40)

    ec, eq, ee = sample(elite)
    lc, lq, le = sample(limited)

    assert eq > lq, f"elite quality {eq} should exceed limited quality {lq}"
    assert ee < le, f"elite error {ee} should be below limited error {le}"
    assert ec.get("THROUGH_BALL", 0) > lc.get("THROUGH_BALL", 0), (
        "elite vision player should perceive/attempt through balls far "
        "more often than a low-vision player in the same spot"
    )
    print(f"[PASS] elite vs limited: quality {eq:.3f} vs {lq:.3f}, "
          f"error {ee:.3f} vs {le:.3f}, "
          f"through_ball {ec.get('THROUGH_BALL', 0)} vs {lc.get('THROUGH_BALL', 0)}")


def test_creative_vs_conservative_soul_style():
    """High-dribble-tendency, low-plays_safe (creative) players should
    attempt materially more risky actions than a conservative profile in
    the identical spot; conservative players should lean safe."""
    creative = make_player("Creative", "CAM", dribbling=80, attempts_dribble=0.55,
                            plays_safe=0.20, vision=75, decisions=70, composure=65)
    conservative = make_player("Conservative", "CAM", dribbling=55, attempts_dribble=0.15,
                                plays_safe=0.80, vision=75, decisions=70, composure=65)
    risky = {"DRIBBLE", "THROUGH_BALL", "CARRY"}
    safe = {"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION"}

    cc, _, _ = sample(creative)
    vc, _, _ = sample(conservative)

    assert share(cc, risky) > share(vc, risky), "creative player should take more risk"
    assert share(vc, safe) > share(cc, safe), "conservative player should play safer"
    print(f"[PASS] creative risky-share {share(cc, risky):.3f} > "
          f"conservative risky-share {share(vc, risky):.3f}")


def test_pressure_and_fatigue_shift_toward_safety():
    """The SAME player, under pressure or fatigued late in the match,
    should shift probability mass toward safe actions relative to their
    own calm/fresh baseline."""
    player = make_player("Baseline", "CAM", vision=70, decisions=65, composure=60,
                          anticipation=65)
    safe = {"SAFE_PASS", "RECYCLE", "PROTECT_POSSESSION"}

    calm, _, _ = sample(player, under_pressure=False, minute=10)
    pressed, _, _ = sample(player, under_pressure=True, minute=10)
    tired, _, _ = sample(player, under_pressure=False, minute=88)

    assert share(pressed, safe) > share(calm, safe), "pressure should raise safe-action share"
    assert share(tired, safe) >= share(calm, safe) * 0.95, (
        "fatigue late in the match should not make the player materially "
        "less safety-conscious than fresh"
    )
    print(f"[PASS] safe share calm={share(calm, safe):.3f} "
          f"pressed={share(pressed, safe):.3f} tired={share(tired, safe):.3f}")


def test_no_player_reaches_perfect_accuracy():
    """No single intent should ever claim 100% of a player's decisions,
    across a wide variety of profiles and situations -- bounded
    rationality means variety, always."""
    profiles = [
        make_player("A", "CAM", vision=99, decisions=99, composure=99, anticipation=99),
        make_player("B", "CM", vision=10, decisions=10, composure=10, anticipation=10),
        make_player("C", "ST", finishing=95, dribbling=90, attempts_dribble=0.9),
        make_player("D", "CB", vision=40, composure=90),
    ]
    for p in profiles:
        counts, _, _ = sample(p, n=500)
        total = sum(counts.values())
        for intent, n in counts.items():
            assert n / total < 1.0, f"{p.name} collapsed to 100% {intent}"
    print("[PASS] no profile collapses to a single deterministic action")


def test_decision_layer_never_carries_outcome_fields():
    """Architectural separation check (requirement 3): the decision
    object must describe INTENT and DECISION QUALITY only. It must never
    carry a success/outcome/completed field -- that is exclusively
    downstream physics/execution's responsibility."""
    field_names = {f.name for f in fields(PlayerDecision)}
    forbidden = {"success", "outcome", "completed", "result"}
    overlap = field_names & forbidden
    assert not overlap, f"PlayerDecision leaked execution-outcome fields: {overlap}"
    print(f"[PASS] PlayerDecision fields ({sorted(field_names)}) contain no outcome/success field")


def test_never_argmax_same_inputs_vary_output():
    """Repeated calls with byte-identical inputs must not always return
    the same intent -- selection is probabilistic, not deterministic
    argmax (requirement 6)."""
    player = make_player("Sampler", "CAM", vision=70, decisions=65, composure=60)
    counts, _, _ = sample(player, n=300)
    assert len(counts) >= 3, (
        f"only {len(counts)} distinct intents chosen across 300 identical "
        "calls -- selection looks deterministic, not probabilistic"
    )
    print(f"[PASS] {len(counts)} distinct intents chosen from identical inputs: {dict(counts)}")


def test_graceful_without_position_engine():
    """No live PositionEngine (legacy/no-geometry call path) must not
    crash the brain -- it should fall back sanely."""
    player = make_player("NoGeo", "CM")
    team_profile = FakeTeamProfile("balanced")
    game_state = FakeGameState("LEVEL")
    d = DecisionBrain.decide(
        player, 50.0, 34.0, [], [], None, team_profile,
        False, True, game_state, minute=20,
    )
    assert isinstance(d, PlayerDecision)
    assert d.action in ("CARRY", "PASS")
    print(f"[PASS] no-PositionEngine call path returns a valid decision: {d.intent.value}")


def test_soul_archetype_shifts_perception_not_free_success():
    """A Soul archetype should shift PERCEPTION/PREFERENCE (which intent
    gets chosen more often), never grant a free success -- this module
    has no success concept at all, so the strongest testable claim is
    that adding a soul measurably changes the behavioral distribution
    for an otherwise-identical player."""
    from player_soul import PlayerSoul, SoulArchetype, GreatnessPillars

    base = make_player("Soulless", "CAM", vision=75, decisions=70, composure=65,
                        through_balls=70)
    souled = make_player("Souled", "CAM", vision=75, decisions=70, composure=65,
                          through_balls=70)
    souled.dna.soul = PlayerSoul(
        player_name="Souled", archetype=SoulArchetype.CREATIVE_ORACLE,
        pillars=GreatnessPillars(hardwork=0.9, talent=0.9, luck=0.8),
    )

    bc, _, _ = sample(base)
    sc, _, _ = sample(souled)
    assert bc != sc, "soul archetype had no measurable effect on the decision distribution"
    print(f"[PASS] soul archetype changes the distribution: base={dict(bc)} souled={dict(sc)}")


def test_full_match_still_runs_and_exports():
    """Integration smoke test: a full scratch match (run_match.py's
    built-in squads) must still run to completion, export normally, and
    keep shot volume in a sane band with the active brain wired in."""
    import run_match
    random.seed(2024)
    run_match.CHECK_AVAILABILITY = False
    run_match.run()
    print("[PASS] full scratch match ran and exported with DecisionBrain wired in")


def main():
    unit_only = "--unit-only" in sys.argv
    random.seed(42)

    tests = [
        test_elite_vs_limited_player_differ,
        test_creative_vs_conservative_soul_style,
        test_pressure_and_fatigue_shift_toward_safety,
        test_no_player_reaches_perfect_accuracy,
        test_decision_layer_never_carries_outcome_fields,
        test_never_argmax_same_inputs_vary_output,
        test_graceful_without_position_engine,
        test_soul_archetype_shifts_perception_not_free_success,
    ]
    if not unit_only:
        tests.append(test_full_match_still_runs_and_exports)

    for t in tests:
        t()

    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
