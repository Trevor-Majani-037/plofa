"""THE SEARCH'S TWO FIXES — the shared corpus and the mutation strength.

WHY THEY EXIST
--------------
The genetic algorithm kept collapsing onto the best CONSTANT intent, even
after the reward's balance was fixed and even after the corpus was fixed so
that the surrogate's table could say "shoot".  The objective's own optimum
is a broad nine-intent mix worth +0.44 (scaled) over the constant, so the
objective was NOT the ceiling — the search was.

Two mechanisms in `brain_evolution.evolve` were making the search worse, and
both are measurable:

1. **THE CORPUS CONFOUND.**  Every brain in the population was evaluated on
   its OWN corpus (`seed + gen * 1000 + i`), so the selection compared
   brain i's fitness measured on brain i's states against brain j's measured
   on brain j's.  The corpus-to-corpus noise is ~+-0.02 of fitness, and the
   differences the search must resolve between two near-tied candidates are
   of the same order — so a share of every comparison was noise.  The corpus
   is now drawn ONCE per generation and shared by the whole population, and
   it is still re-drawn every generation, so no brain can overfit one
   sample.

2. **THE MUTATION STRENGTH.**  0.3 perturbs 15% of the weights by 30% of
   their OWN magnitude, which randomises the net's behaviour: that is a
   random walk, not a hill climb.  It is how the algorithm reaches the
   constant (a bias tweak) and then never refines past it.  The strength is
   now a parameter.

WHAT THIS FILE PINS
-------------------
That the shared corpus really is shared (and the per-brain corpus really is
per-brain, for the comparison), that the corpus is re-drawn every
generation, that the mutation strength reaches the mutator, and that the
defaults are the measured ones.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brain_evolution  # noqa: E402
from football_brain import FootballBrain  # noqa: E402


class _Capture:
    """Records the corpus each brain was evaluated on."""

    def __init__(self):
        self.seen = []
        self.original = brain_evolution.synthetic_fitness

    def __call__(self, brain, position, batched_sensors=None, **kwargs):
        self.seen.append(batched_sensors)
        return 0.5

    def install(self):
        brain_evolution.synthetic_fitness = self

    def restore(self):
        brain_evolution.synthetic_fitness = self.original


class TheSharedCorpusTests(unittest.TestCase):

    def setUp(self):
        self.cap = _Capture()
        self.cap.install()

    def tearDown(self):
        self.cap.restore()

    def _run(self, **kwargs):
        brain_evolution.evolve(position="ST", population_size=4,
                               generations=2, n_states=20, seed=7,
                               verbose=False, **kwargs)
        return self.cap.seen

    def test_every_brain_is_evaluated_on_the_same_corpus(self):
        """The confound: fitnesses compared across different samples."""
        seen = self._run(shared_corpus=True)
        self.assertEqual(len(seen), 8, "expected 4 brains x 2 generations")
        first_gen = seen[:4]
        self.assertEqual(len({id(s) for s in first_gen}), 1,
                         "the population was NOT evaluated on one corpus")
        for s in first_gen:
            self.assertEqual(s.shape, (20, 24))

    def test_the_per_brain_corpus_is_still_available(self):
        """The old behaviour, for the comparison and for reproducing it."""
        seen = self._run(shared_corpus=False)
        first_gen = seen[:4]
        self.assertEqual(len({id(s) for s in first_gen}), 4,
                         "the per-brain corpus is not per-brain any more")

    def test_the_corpus_is_redrawn_every_generation(self):
        """Otherwise a brain could overfit one sample of states."""
        seen = self._run(shared_corpus=True)
        self.assertNotEqual(id(seen[0]), id(seen[4]),
                            "the same corpus was reused across generations")
        self.assertFalse(np.array_equal(seen[0], seen[4]))

    def test_the_shared_corpus_is_the_default(self):
        """The measured behaviour is the default; the old one is opt-in."""
        import inspect
        sig = inspect.signature(brain_evolution.evolve)
        self.assertTrue(sig.parameters["shared_corpus"].default,
                        "the per-brain corpus confound is the default again")


class TheMutationStrengthTests(unittest.TestCase):

    def test_the_strength_reaches_the_mutator(self):
        """A parameter that is not passed through is a parameter that does
        nothing — the silent no-op this project keeps finding."""
        seen = []
        original = FootballBrain.mutate

        def spy(self, rate=0.15, strength=0.3):
            seen.append(strength)
            return original(self, rate=rate, strength=strength)

        FootballBrain.mutate = spy
        try:
            brain_evolution.evolve(position="ST", population_size=6,
                                   generations=2, n_states=20, seed=7,
                                   verbose=False, mutation_strength=0.07,
                                   shared_corpus=True)
        finally:
            FootballBrain.mutate = original
        self.assertTrue(seen, "no mutation happened")
        self.assertEqual(set(seen), {0.07},
                         "the mutation strength did not reach the mutator")

    def test_the_default_strength_is_the_measured_one(self):
        import inspect
        sig = inspect.signature(brain_evolution.evolve)
        self.assertAlmostEqual(sig.parameters["mutation_strength"].default,
                               0.3, places=6)


if __name__ == "__main__":
    unittest.main()
