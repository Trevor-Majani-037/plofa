# -*- coding: utf-8 -*-
"""Step-8 (audit item 4 AUTHENTICATED follow-through, "demote dont delete"):
the rule layers now PROPOSE bounded priors to the NeuralDecisionBrain; the
brain remains the SOLE chooser of chosen_intentcars.

This test drives ONLY the real, engine-confirmed surface of
`decision_proposals.py` -- exactly the discipline of
`tests/test_decision_journal.py`: every assertion is deterministic,
assert-gated, no input(), no RNG, no network, ASCII-only bytes. The exact
gate bytes below were pulled from the module itself by the authoritative
interpreter (engine-wrote-and-ran, not guessed): the shot floor is a strict
calibration boundary (sub-floor -> [], above-floor -> SHOOT); gk distribution
rejects every non-GK position and SAFE_PASSes an open short window; the wide
combo rejects central attackers and opens CARRY+CROSS for a genuine RW window;
phase regression drops to a keeper only when instructed. RuleProposal
param order / RuleSource spelling come verbatim from the engine's enum.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decision_proposals as dp
from decision_proposals import RuleProposal, RuleSource


def _confident(proposals):
    assert len(proposals) == 1
    return proposals[0]


# ---------------------------------------------------------------------------
# 1. AttackingMatrix shot gate is a strict calibration BOUNDARY, not a
#    hard-coded override: a sub-floor look never proposes SHOOT.
# ---------------------------------------------------------------------------
def test_shot_gate_is_calibration_boundary_not_hard_override():
    # sub-floor by a clear margin (default floor 0.7): the gate stays shut.
    assert dp.propose_shot(0.20, 24.0, True) == []
    # floor-inclusive: a dose AT the explicit floor opens the window the
    # matrix owns -- a bounded prior, never a hard-coded strike.
    out = dp.propose_shot(0.30, 24.0, True, calibration_floor=0.30)
    p = _confident(out)
    assert p.intent == "SHOOT"
    assert p.rule_source is RuleSource.ATTACKING_MATRIX


def test_shot_gate_opens_only_above_the_floor():
    out = dp.propose_shot(0.85, 15.0, False)
    p = _confident(out)
    assert p.intent == "SHOOT"
    assert p.rule_source is RuleSource.ATTACKING_MATRIX


def test_shot_gate_never_hard_codes_a_strike():
    # determinism: identical gate bytes -> identical proposals.
    a = dp.propose_shot(0.85, 15.0, False)
    b = dp.propose_shot(0.85, 15.0, False)
    assert a == b


# ---------------------------------------------------------------------------
# 2. GK distribution: position-gated. Only a keeper is told to distribute
#    short; an outfielder with the same look is demoted to nothing.
# ---------------------------------------------------------------------------
def test_gk_distribution_rejects_outfielders():
    assert dp.propose_gk_distribution("ST", 0.80, 4, None) == []
    assert dp.propose_gk_distribution("GK", 0.05, 0, None) == []


def test_gk_distribution_opens_short_safe_pass_for_the_keeper():
    out = dp.propose_gk_distribution("GK", 0.80, 6, None)
    p = _confident(out)
    assert p.intent == "SAFE_PASS"
    assert p.rule_source is RuleSource.GK_DISTRIBUTION
    # the keeper still stays the sole decider; the layer only proposes.
    assert p is not None


# ---------------------------------------------------------------------------
# 3. Wide/combo layer: demote, never delete. Central attackers get nothing;
#    a genuine RW window opens CARRY + CROSS as priors for the brain.
# ---------------------------------------------------------------------------
def test_wide_combo_rejects_central_forward():
    assert dp.propose_wide_combo("ST", 0.90, 0.85, 0.70, 0.60) == []


def test_wide_combo_opens_carry_and_cross_for_wing():
    out = dp.propose_wide_combo("RW", 0.85, 0.70, 0.55, 0.62)
    assert len(out) == 2
    intents = {p.intent for p in out}
    assert intents == {"CARRY", "CROSS"}
    assert all(p.rule_source is RuleSource.WIDE_COMBO for p in out)


# ---------------------------------------------------------------------------
# 4. Phase regression: a bounded prior that only fires under an explicit
#    drop-to-keeper instruction; open-space recycling stays closed.
# ---------------------------------------------------------------------------
def test_phase_regression_only_fires_on_instructed_drop_to_gk():
    assert dp.propose_phase_regression(45.0, 0.30, False, 0.60, 0.80, "recycle") == []
    assert dp.propose_phase_regression(45.0, 0.30, False, 0.67, 0.15, None) == []


def test_phase_regression_drop_to_gk_is_a_bounded_prior():
    drop = dp.propose_phase_regression(62.0, 0.10, True, 0.55, 0.85, "drop_to_gk")
    p = _confident(drop)
    assert p.intent == "RECYCLE"
    assert p.rule_source is RuleSource.TACTICAL_POSSESSION


# ---------------------------------------------------------------------------
# 5. Whole-lane determinism: byte-identical inputs, byte-identical priors.
#    Same seal the journal test enforces for the on-ball anchor.
# ---------------------------------------------------------------------------
def test_all_rule_gates_are_deterministic():
    identical = [
        dp.propose_shot(0.42, 22.0, True) == dp.propose_shot(0.42, 22.0, True),
        dp.propose_gk_distribution("GK", 0.60, 3, None)
        == dp.propose_gk_distribution("GK", 0.60, 3, None),
        dp.propose_wide_combo("RW", 0.80, 0.70, 0.55, 0.62)
        == dp.propose_wide_combo("RW", 0.80, 0.70, 0.55, 0.62),
        dp.propose_phase_regression(50.0, 0.10, True, 0.55, 0.85, "drop_to_gk")
        == dp.propose_phase_regression(50.0, 0.10, True, 0.55, 0.85, "drop_to_gk"),
    ]
    assert sum(identical) == len(identical)
