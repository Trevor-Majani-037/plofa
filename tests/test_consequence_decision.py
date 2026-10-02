"""Focused tests for the consequence reasoning seam (consequence_decision.py
+ the wiring in brain_integration._decide_core).

Run:  python -m tests.test_consequence_decision

Plain assert-based scripts, matching the project's existing test style.

Covers:
  1. IDENTITY   -- blend=0 returns the policy unchanged (byte-identical).
  2. NORMALISE  -- corrected distribution sums to 1, non-negative.
  3. BATON      -- a critic that rates an intent higher lifts its mass;
                   a critic that rates it lower damps it (advantage weighting).
  4. TRACE      -- payoff_traces exposes q / advantage / corrected per intent.
  5. GATE OFF   -- apply_reasoning returns (probs, None) when the seam is off.
  6. DECIDE OFF -- _decide_core is byte-identical when reasoning is not enabled.
  7. DECIDE ON  -- with a critic installed, _decide_core's sampled distribution
                   shifts toward the higher-payoff intent.
"""
from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from football_brain import FootballBrain, INTENT_LABELS
from consequence_decision import (
    ConsequenceEvaluator, apply_reasoning, consequence_enabled,
    set_evaluator, set_consequence, clear_consequence as cdr_clear,
    init_from_env, default_enabled,
)
from brain_integration import (
    NeuralDecisionBrain, _decide_core, build_sensors,
    register_brain, clear_registry, set_perception,
)
from player_dna import (
    PlayerDNA, PlayerProfile, PhysicalAttributes, TechnicalAttributes,
    MentalAttributes, PassingAttributes, BehavioralTendencies, PlayerFormState,
)

N_INTENTS = len(INTENT_LABELS)
SHOOT_IDX = INTENT_LABELS.index("SHOOT")


class DummyCritic:
    """Minimal critic: predict(sensors, position) -> (10,) payoff vector."""

    def __init__(self, payoffs):
        self.payoffs = np.asarray(payoffs, dtype=np.float64)  # length 10

    def predict(self, sensors, position):
        return self.payoffs.copy()


class FakeStyle:
    def __init__(self, v):
        self.value = v


class FakeTeamProfile:
    def __init__(self, style="balanced"):
        self.style = FakeStyle(style)


class FakeGameState:
    def __init__(self, name="LEVEL", score=None):
        self.name = name


class FakePositionEngine:
    def __init__(self, positions):
        self.positions = positions
    def get_position(self, name):
        return self.positions.get(name, (50.0, 34.0))


def make_player(name, position="ST", vision=60, composure=60, decisions=60):
    mental = MentalAttributes(
        vision=vision, composure=composure, decisions=decisions,
        anticipation=vision,
    )
    dna = PlayerDNA(name=name, position=position, mental=mental)
    return SimpleNamespace(name=name, position=position, dna=dna)


def policy_uniform():
    return np.full(N_INTENTS, 1.0 / N_INTENTS)


def test_identity_when_blend_zero():
    ev = ConsequenceEvaluator(DummyCritic(np.full(N_INTENTS, 0.5)), blend=0.0)
    p = policy_uniform()
    corrected = ev.correct_policy(p, np.zeros(24), "ST")
    assert np.allclose(corrected, p), "blend=0 must return the policy unchanged"
    print("1. identity: OK")


def test_normalise_and_bounds():
    ev = ConsequenceEvaluator(DummyCritic(np.linspace(0.4, 1.3, N_INTENTS)), blend=0.5)
    p = policy_uniform()
    c = ev.correct_policy(p, np.zeros(24), "ST")
    assert np.isclose(c.sum(), 1.0)
    assert np.all(c >= 0.0)
    print("2. normalise/bounds: OK")


def test_baton_advantage_weighting():
    payoffs = np.full(N_INTENTS, 0.45)
    # SHOOT is the runaway best consequence in this state; CARRY is worst.
    payoffs[SHOOT_IDX] = 1.30
    carry_idx = INTENT_LABELS.index("CARRY")
    payoffs[carry_idx] = 0.40
    ev = ConsequenceEvaluator(DummyCritic(payoffs), blend=1.5)
    p = policy_uniform()
    c = ev.correct_policy(p, np.zeros(24), "ST")
    assert c[SHOOT_IDX] > p[SHOOT_IDX], "high-payoff intent must gain mass"
    assert c[carry_idx] < p[carry_idx], "low-payoff intent must lose mass"
    print("3. baton: OK")


def test_trace_dump():
    payoffs = np.linspace(0.4, 1.3, N_INTENTS)
    ev = ConsequenceEvaluator(DummyCritic(payoffs), blend=0.5)
    t = ev.payoff_traces(policy_uniform(), np.zeros(24), "ST")
    assert set(t.keys()) == {"q", "advantage", "corrected"}
    assert len(t["q"]) == N_INTENTS
    print("4. trace: OK")


def test_gate_off():
    cdr_clear()
    assert not consequence_enabled()
    p = policy_uniform()
    out, trace = apply_reasoning(p, np.zeros(24), "ST")
    assert trace is None
    assert out is p, "off seam must pass the very same object"
    print("5. gate-off: OK")


def _register_fake_brain(player_name, probs):
    class FakeFootballBrain(FootballBrain):
        def __init__(self, fixed_probs):
            base = FootballBrain.random(seed=7, input_size=24)
            self.w1, self.b1 = base.w1, base.b1
            self.w2, self.b2 = base.w2, base.b2
            self.w3, self.b3 = base.w3, base.b3
            self.meta = base.meta
            self.input_size = 24
            self._fixed_probs = fixed_probs
        def forward(self, sensors):
            return self._fixed_probs.copy()
    register_brain(player_name, FakeFootballBrain(probs))


def make_decide_player():
    p = make_player("REASON", "ST")
    return p


def decide_args(player, probs):
    teammates = [SimpleNamespace(name="TM"), SimpleNamespace(name="TM2")]
    defenders = [SimpleNamespace(name="DF")]
    pos = FakePositionEngine({"TM": (60.0, 30.0), "TM2": (70.0, 40.0),
                              "DF": (65.0, 34.0)})
    # Deterministic sampling: monkeypatch random stream seed inside the test.
    import random
    random.seed(7)
    return dict(
        player=player, x=45.0, y=34.0, teammates=teammates,
        defenders=defenders, position_engine=pos,
        team_profile=FakeTeamProfile(), under_pressure=False,
        attacks_right=True, game_state=FakeGameState(), minute=30.0,
    )


def test_decide_byte_identical_when_off():
    # Reasoning is on-by-default since graduation (2026-09-20); pin the
    # OFF posture for this test so init_from_env cannot auto-load v5.
    default_enabled(False)
    cdr_clear()
    brain_integration_clear()
    player = make_decide_player()
    probs = np.array([0.01, 0.05, 0.05, 0.10, 0.05, 0.20, 0.04, 0.05, 0.30, 0.15])
    _register_fake_brain("REASON", probs)
    import random
    random.seed(11)
    d1 = _decide_core(record_trace=True, **decide_args(player, probs))
    assert d1.trace is not None and "consequence" not in d1.trace
    random.seed(11)
    d2 = _decide_core(record_trace=True, **decide_args(player, probs))
    assert d1 == d2, "reasoning off must be byte-identical run to run"
    print("6. decide-off byte-identical: OK")


def brain_integration_clear():
    from brain_integration import clear_consequence
    clear_consequence()


def test_decide_shifts_with_critic():
    # Pin the unset-env posture off so init_from_env leaves the explicit
    # DummyCritic evaluator installed instead of auto-loading critic_v5.
    default_enabled(False)
    brain_integration_clear()
    player = make_decide_player()
    probs = np.array([0.01, 0.05, 0.05, 0.10, 0.05, 0.20, 0.04, 0.05, 0.30, 0.15])
    _register_fake_brain("REASON", probs)
    payoffs = np.full(N_INTENTS, 0.42)
    payoffs[SHOOT_IDX] = 1.4  # only SHOOT is worth it here
    set_evaluator(ConsequenceEvaluator(DummyCritic(payoffs), blend=2.0))
    assert consequence_enabled()
    d = _decide_core(record_trace=True, **decide_args(player, probs))
    assert d.trace is not None and "consequence" in d.trace
    assert d.trace["consequence"]["q"]["SHOOT"] == 1.4
    # The corrected probability should boost SHOOT well above its raw
    # policy value (0.05) and make it the top pick of the corrected dist.
    corr_shoot = d.trace["consequence"]["corrected"]["SHOOT"]
    corr_vals = d.trace["consequence"]["corrected"].values()
    assert corr_shoot > 0.05 * 3, f"SHOOT corrected {corr_shoot} must be >> raw 0.05"
    assert corr_shoot == max(corr_vals), "top-payoff intent must top the corrected dist"
    print("7. decide-on critic shift: OK")


if __name__ == "__main__":
    test_identity_when_blend_zero()
    test_normalise_and_bounds()
    test_baton_advantage_weighting()
    test_trace_dump()
    test_gate_off()
    test_decide_byte_identical_when_off()
    test_decide_shifts_with_critic()
    print("ALL CONSEQUENCE-REASONING TESTS PASSED")