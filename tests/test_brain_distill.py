"""DISTILLATION — the net learns the surrogate's own policy.

WHY
---
The genetic algorithm kept collapsing onto the best CONSTANT intent, even
after the reward's balance was fixed (the per-position scale) and the
corpus's coverage was fixed (the situation-aware exploration).  The
diagnosis is not "the objective is the ceiling": **the surrogate's own
optimum IS the broad, football-shaped policy the census wants** — the
striker's per-bucket argmax is nine intents with a 24% maximum and the
striker shooting 11% of the time, worth +0.44 (scaled) over the constant.

So the objective is right and the SEARCH is the blocker: the constant is a
local optimum a bias tweak reaches in one step, and the useful
state-dependence is a specific function that a blind search from a random
initialisation does not reach.

This file pins the distillation: that its TARGETS are the table's argmax
over the ROLE's vocabulary (and never a forbidden action), that the
training reduces the loss and raises the agreement, that the distilled
brain's intent MIX is the table's mix — which is the census-passing
property — and that the distillation is deterministic.

⚠️  THE FIXTURE GOES THROUGH THE REAL `fit()`.  A suite that builds the
thing under test by hand is evidence about nothing.
"""
from __future__ import annotations

import os
import random
import sys
import unittest
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_distill import _targets, _vocabulary, distill_brain  # noqa: E402
from decision_brain import PlayerIntent  # noqa: E402
from surrogate_collect import (  # noqa: E402
    FORBIDDEN_VALUE, FitnessSurrogate,
)


def sensors_for(bucket: str) -> np.ndarray:
    import re
    m = re.fullmatch(r"P(\d)FT(\d)OH(\d)S(\d)C(\d)G(\d)", bucket)
    assert m, f"bad bucket {bucket!r}"
    pressure, ft, oh, space, central, goal = (int(g) for g in m.groups())
    v = np.zeros(24, dtype=np.float64)
    v[16] = 1.0 if pressure else 0.0
    v[12] = 1.0 if ft else 0.0
    v[13] = 1.0 if oh else 0.0
    v[9] = 0.8 if space else 0.0
    v[15] = 1.0 if central else 0.0
    v[14] = 0.2 if goal else 0.9
    return v


#: two situations with CLEARLY different best actions, so the distilled
#: brain has to be state-dependent to fit them
CLEAR = "P0FT1OH0S1C1G1"    # final third, goal-close, central, with space
DARK = "P0FT0OH1S0C0G0"     # own half, no space


def build_surrogate() -> FitnessSurrogate:
    """A miniature corpus, fitted through the REAL fit."""
    data = []
    # the striker: SHOOT wins in the clear chance, PROTECT wins at home
    for _ in range(3):
        data.append((sensors_for(CLEAR), PlayerIntent.SHOOT, 0.40, "PASS", "ST"))
        data.append((sensors_for(CLEAR), PlayerIntent.SAFE_PASS, -0.10,
                     "PASS", "ST"))
        data.append((sensors_for(DARK), PlayerIntent.PROTECT_POSSESSION,
                     0.30, "PASS", "ST"))
        data.append((sensors_for(DARK), PlayerIntent.CARRY, -0.05, "PASS", "ST"))
        # the keeper: never allowed to shoot or cross
        data.append((sensors_for(CLEAR), PlayerIntent.SAFE_PASS, 0.20,
                     "PASS", "GK"))
        data.append((sensors_for(DARK), PlayerIntent.RECYCLE, 0.25, "PASS", "GK"))
    return FitnessSurrogate().fit(data)


class TheTargetsTests(unittest.TestCase):

    def setUp(self):
        self.sur = build_surrogate()

    def test_the_target_is_the_tables_preferred_action(self):
        """The distillation's target is the table's argmax, not a label."""
        rows = np.array([sensors_for(CLEAR), sensors_for(DARK)])
        y = _targets(self.sur, rows, "ST")
        labels = [p.value for p in PlayerIntent]
        self.assertEqual(labels[int(y[0])], "SHOOT",
                         "the clear chance's target is not the shot")
        self.assertEqual(labels[int(y[1])], "PROTECT_POSSESSION",
                         "the own half's target is not the protection")

    def test_a_forbidden_action_is_never_a_target(self):
        """The keeper must never be trained towards a shot."""
        rows = np.array([sensors_for(CLEAR), sensors_for(DARK)])
        y = _targets(self.sur, rows, "GK")
        labels = [p.value for p in PlayerIntent]
        for idx in y:
            self.assertNotIn(labels[int(idx)], ("SHOOT", "CROSS"))

    def test_the_vocabulary_is_the_roles_own(self):
        from role_features import ROLE_INTENTS
        vocab = {i.value for i in _vocabulary("ST")}
        self.assertTrue(vocab.issubset(ROLE_INTENTS["ST"]))
        self.assertNotIn("CROSS", vocab, "the striker may not cross")

    def test_the_forbidden_value_is_never_read_as_a_target(self):
        """The veto is a RAW-scale number; the scaled one is far below every
        real reward, so a target must never be the veto."""
        rows = np.array([sensors_for(CLEAR)])
        y = _targets(self.sur, rows, "GK")
        labels = [p.value for p in PlayerIntent]
        self.assertNotEqual(labels[int(y[0])], "SHOOT")
        # and the veto itself is far below any legitimate value
        veto = self.sur.expected_success(sensors_for(CLEAR),
                                        PlayerIntent.SHOOT, "GK")
        best = self.sur.expected_success(sensors_for(CLEAR),
                                         PlayerIntent.SAFE_PASS, "GK")
        self.assertLess(veto, best)


class TheTrainingTests(unittest.TestCase):

    def setUp(self):
        self.sur = build_surrogate()

    def _distil(self, epochs=60):
        return distill_brain("ST", self.sur, n_states=400, seed=5,
                             epochs=epochs, input_size=24)

    def test_the_training_reduces_the_loss_and_raises_the_agreement(self):
        """Non-vacuity: the training has to actually do something."""
        _brain, rep = self._distil()
        self.assertLess(rep["loss_last"], rep["loss_first"],
                        "the loss did not fall")
        self.assertGreater(rep["agreement"], 0.8,
                           "the net did not learn the table's policy")
        self.assertGreater(rep["agreement"], rep["agreement_first"] + 0.2,
                           "the agreement did not rise")

    def test_the_distilled_brain_is_state_dependent(self):
        """THE POINT: the net must choose DIFFERENT actions in different
        situations, not one constant."""
        brain, rep = self._distil()
        clear = brain.predict(sensors_for(CLEAR))
        dark = brain.predict(sensors_for(DARK))
        self.assertNotEqual(clear[0], dark[0],
                            "the distilled brain plays the same action in "
                            "both situations")

    def test_the_distilled_mix_is_the_tables_mix(self):
        """THE invariant: the brain's mix must be the TABLE's mix.

        **The census criteria are a property of the REAL table's mix, not of
        this fixture's** — the miniature corpus has two situations, so its
        target mix is narrow and the brain that fits it is narrow too. What
        has to hold is that the brain reproduces the target mix exactly, and
        the real-table case (below) is where "broad enough for the census"
        is actually tested.
        """
        brain, rep = self._distil()
        mix = rep["mix"]
        self.assertEqual(set(mix), set(rep["target_mix"]),
                         "the brain and the table disagree on which actions "
                         "exist")
        for action, share in rep["target_mix"].items():
            self.assertAlmostEqual(mix[action], share, places=2,
                                   msg=f"{action}: the brain's mix does not "
                                       f"match the table's")

    def test_the_real_table_distils_into_a_census_passing_brain(self):
        """THE CENSUS GATE, on the real surrogate: the striker's distilled
        brain must use many actions, none over 40%, and it must SHOOT."""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "brains", "surrogate_pos_v7merged.json")
        if not os.path.exists(path):
            self.skipTest("the v7merged surrogate is not present")
        from critic_surrogate import load_any_surrogate
        sur = load_any_surrogate(path)
        from perception import PerceptionConfig
        cfg = PerceptionConfig(enabled=True, role_blocks=True, seed=123)
        brain, rep = distill_brain("ST", sur, n_states=800, seed=123,
                                    epochs=80, perception_config=cfg)
        mix = rep["mix"]
        self.assertGreaterEqual(len(mix), 4,
                                f"the distilled striker collapsed: {mix}")
        self.assertLess(max(mix.values()), 0.40,
                        f"the distilled striker collapsed: {mix}")
        self.assertGreater(mix.get("SHOOT", 0.0), 0.0,
                           f"the distilled striker never shoots: {mix}")
        self.assertGreater(rep["agreement"], 0.9)

    def test_the_distillation_is_deterministic(self):
        b1, r1 = self._distil(epochs=20)
        b2, r2 = self._distil(epochs=20)
        self.assertTrue(np.array_equal(b1.w1, b2.w1))
        self.assertEqual(r1["mix"], r2["mix"])

    def test_the_distillation_does_not_consume_the_global_random_stream(self):
        """The same discipline as the perception reorder: a training path
        that moves the global stream moves every later draw."""
        random.seed(4242)
        before = random.getstate()
        self._distil(epochs=5)
        self.assertEqual(before, random.getstate(),
                         "the distillation moved the global RNG stream")


if __name__ == "__main__":
    unittest.main()
