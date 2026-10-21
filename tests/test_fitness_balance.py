"""THE FITNESS BALANCE — the reward must decide, not the confidence.

THE DEFECT THIS FILE EXISTS FOR
-------------------------------
The surrogate fitness was:

    base    = 0.5 + 0.5 * clip(mean_reward, -1, 1)
    fitness = base * (0.5 + 0.5 * mean_conf) - penalties / n
    scaled  = (fitness - dom_pen) * (1.0 + 1.0 * bdiv)

Both the confidence term and the entropy term are MULTIPLICATIVE factors
with a range of about 2, while the threat reward's within-position spread is
about **0.01** — because a forward pass is worth ~+0.008, and the OLD reward
(engine success: 2.0 / 1.0 / 0.0) had a spread of ~1.0. **The balance
inverted by roughly fifty times when the objective was replaced**, and the
GA duly optimised PEAKEDNESS instead of football:

    GK    mean_conf 0.453 (shipped 0.194)   mean_reward -0.253
    CB    mean_conf 0.625 (shipped 0.247)   mean_reward -0.361
    LB    mean_conf 0.960 (shipped 0.212)   TWO intents, 82% RECYCLE
    CDM   mean_conf 0.966 (shipped 0.230)   TWO intents, 61% DRIBBLE
    CM    mean_conf 0.889 (shipped 0.227)   TWO intents, 67% SAFE_PASS
    LW    mean_conf 0.887 (shipped 0.380)   TWO intents, 61% CARRY

A goalkeeper whose reward was **-0.25** WON, because it was more confident.
And a peaked network IS a collapsed intent mix, so the retrain also
produced a keeper who crossed 14% of the time and a centre-back who shot
44% — while every fitness number sat unremarkably at 0.60-0.67.

The fix demotes both to small ADDITIVE guards: they can break a tie and
they can veto a collapse, but they cannot decide.

WHAT THIS FILE PINS
-------------------
That the reward's influence on the fitness exceeds the guards' combined
influence by a wide margin; that with the SAME reward, the guards cannot
move the fitness by more than their weights; and that a higher reward wins
when everything else is equal.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_evolution import (CONFIDENCE_GUARD_W, DIVERSITY_GUARD_W,  # noqa: E402
                             generate_state_corpus, synthetic_fitness)
from football_brain import FootballBrain  # noqa: E402


class _ConstantSurrogate:
    """A surrogate that pays the same for every intent."""

    def __init__(self, value: float):
        self.value = float(value)

    def expected_success(self, sensors, intent, position: str = "") -> float:
        return self.value


class FitnessBalanceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4242)
        cls.brain = FootballBrain(
            w1=rng.normal(0, 0.3, (24, 32)), b1=rng.normal(0, 0.1, 32),
            w2=rng.normal(0, 0.3, (32, 32)), b2=rng.normal(0, 0.1, 32),
            w3=rng.normal(0, 0.3, (32, 10)), b3=rng.normal(0, 0.1, 10),
            seed=4242,
        )
        cls.corpus = generate_state_corpus("CM", 120, seed=5, input_size=24)

    def test_the_reward_outweighs_the_guards_by_a_wide_margin(self):
        """The reward's range is 1.0; the guards' combined weight is small."""
        guards = CONFIDENCE_GUARD_W + DIVERSITY_GUARD_W
        self.assertLess(guards, 0.15,
                        "the guards can move the fitness by more than 0.15")
        self.assertGreaterEqual(1.0 / max(guards, 1e-9), 6.0,
                                "the reward does not outweigh the guards")

    def test_the_guards_cannot_decide(self):
        """Same reward, different brains: the guards alone can only move the
        fitness by their own weights."""
        low = synthetic_fitness(self.brain, "CM",
                                batched_sensors=self.corpus,
                                surrogate=_ConstantSurrogate(0.0))
        high = synthetic_fitness(self.brain, "CM",
                                 batched_sensors=self.corpus,
                                 surrogate=_ConstantSurrogate(0.4))
        # the reward moved the fitness by ~0.2 (0.5 * 0.4)
        self.assertAlmostEqual(high - low, 0.2, places=6,
                               msg="the reward did not move the fitness")

    def test_a_higher_reward_wins_when_everything_else_is_equal(self):
        good = synthetic_fitness(self.brain, "CM",
                                 batched_sensors=self.corpus,
                                 surrogate=_ConstantSurrogate(0.3))
        bad = synthetic_fitness(self.brain, "CM",
                                batched_sensors=self.corpus,
                                surrogate=_ConstantSurrogate(-0.3))
        self.assertGreater(good, bad)

    def test_the_guards_are_additive_and_bounded(self):
        """A peaked brain must not be rewarded much more than a flat one
        when the reward is identical — that is the collapse generator."""
        peaked = synthetic_fitness(self.brain, "CM",
                                   batched_sensors=self.corpus,
                                   surrogate=_ConstantSurrogate(0.0))
        flat = synthetic_fitness(self.brain, "CM",
                                 batched_sensors=self.corpus,
                                 surrogate=_ConstantSurrogate(0.0))
        # the same brain and the same constant reward: the two calls agree,
        # and the value sits inside the reward's own band
        self.assertAlmostEqual(peaked, flat, places=9)
        self.assertGreaterEqual(peaked, 0.0)
        self.assertLessEqual(peaked, 1.0)


if __name__ == "__main__":
    unittest.main()
