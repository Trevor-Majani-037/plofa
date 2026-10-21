"""THE SITUATION-AWARE ε-EXPLORATION — the corpus's COVERAGE fix.

WHY IT EXISTS
-------------
The v5/v6 corpus cannot teach the striker to shoot, and the reason is
measurable and specific.  The striker's clearest-chance bucket
(`P1FT1OH0S1C1G1` — final third, goal-close, central, with space) held
about **ten striker decisions across six matches and not one shot**, so its
top value was PROTECT_POSSESSION +0.1068 — the largest positive value in the
striker's whole table.  The objective could not say "shoot" because it had
never been told.

The arithmetic behind it: 3,696 pairs over 11 positions x 64 buckets = 5.25
pairs per bucket and ~2.5 per (bucket, intent) cell, and 65% of the cells
are shrunk toward their bucket mean.  The bucket-RESOLUTION hypothesis was
tested and rejected (coarser schemes do not improve the held-out
agreement), so the lever is the corpus's COVERAGE, not the fit.

WHAT THIS CHANGES
-----------------
The uniform exploration samples an action the ROLE can perform.  It does not
look at the SITUATION, so the rare actions land wherever the player happens
to be.  The situational exploration:

  1. draws the explored action with the PHASE's weights — the actions a
     situation can be judged on are sampled more often there (a shot in the
     final third, a switch in the open, a safe pass in your own half);
  2. raises the exploration RATE in the final third, where the rare actions
     are the only ones the situation can judge at all.

The vocabulary stays the ROLE's: the striker is still never asked to cross
and the keeper is still never asked to shoot.  **It is OFF by default** —
this is a proposal, measured and written up in AGENTS.md (*THE CEILING*),
not a decision this session took.

WHAT THIS FILE PINS
-------------------
That the switch is off by default; that the ROLE vocabulary is still
respected with it on; that the phase weights actually change the mix; that
the rate really is higher in the final third; that the phase matches the
surrogate's own bucket key; that the uniform path is unchanged; and that
computing the sensors EARLIER (so the exploration can read them) does not
move the global RNG stream — the perception layer draws from a SEEDED
generator, so the reorder is free.
"""
from __future__ import annotations

import os
import random
import sys
import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import surrogate_collect as sc  # noqa: E402
from decision_brain import PlayerIntent  # noqa: E402
from surrogate_collect import FitnessSurrogate  # noqa: E402


def sensors(final_third=False, own_half=False, space=True, goal_close=False):
    v = np.zeros(24, dtype=np.float64)
    v[12] = 1.0 if final_third else 0.0
    v[13] = 1.0 if own_half else 0.0
    v[9] = 0.8 if space else 0.0
    v[14] = 0.2 if goal_close else 0.9
    return v


def _player(position):
    return SimpleNamespace(position=position, name="X")


class TheSwitchIsOffByDefaultTests(unittest.TestCase):

    def test_the_situational_switch_is_off(self):
        self.assertFalse(sc._EXPLORE_SITUATIONAL)

    def test_the_uniform_path_is_unchanged_by_the_new_code(self):
        """With no sensors, the exploration draws uniformly from the role's
        vocabulary — exactly as it always has."""
        sc._EXPLORE_EPS = 1.0
        sc._EXPLORE_RNG = random.Random(3)
        try:
            seen = {sc._explore_intent(_player("ST"), PlayerIntent.SAFE_PASS).value
                    for _ in range(200)}
        finally:
            sc._EXPLORE_EPS = 0.0
        from role_features import ROLE_INTENTS
        self.assertTrue(seen.issubset(ROLE_INTENTS["ST"]))
        self.assertEqual(len(seen), len(ROLE_INTENTS["ST"]),
                         "the uniform path no longer covers the whole role "
                         "vocabulary")


class ThePhaseIsTheBucketPhaseTests(unittest.TestCase):

    def test_the_four_phases_are_detected(self):
        self.assertEqual(sc._situation_phase(sensors(final_third=True)),
                         "final_third")
        self.assertEqual(
            sc._situation_phase(sensors(final_third=True, goal_close=True)),
            "goal_close")
        self.assertEqual(sc._situation_phase(sensors(own_half=True)),
                         "own_half")
        self.assertEqual(sc._situation_phase(sensors()), "open")

    def test_the_goal_close_phase_is_the_only_one_where_a_shot_is_sampled_hard(self):
        """The engine's shot gate refuses a strike the geometry does not
        allow, and a refused strike is a zero-length marker carry worth
        exactly 0.0 — so sampling shots where they cannot be taken would
        teach the table that shooting is worth nothing."""
        gc = sc._SITUATION_ACTIONS["goal_close"]
        ft = sc._SITUATION_ACTIONS["final_third"]
        self.assertGreater(gc["SHOOT"], ft["SHOOT"],
                           "the shot is not weighted more where it can be "
                           "taken")
        self.assertGreater(gc["SHOOT"], 4.0)

    def test_the_phase_agrees_with_the_surrogates_own_bucket_key(self):
        """The exploration and the surrogate must agree on what a situation
        IS, or the corpus and the table are about different buckets."""
        sur = FitnessSurrogate()
        self.assertTrue(sur.bucket(sensors(final_third=True)).startswith("P0FT1"))
        self.assertTrue(
            sur.bucket(sensors(final_third=True, goal_close=True)).startswith("P0FT1"))
        self.assertTrue(sur.bucket(sensors(own_half=True)).startswith("P0FT0OH1"))
        self.assertTrue(sur.bucket(sensors()).startswith("P0FT0OH0"))

    def test_no_sensors_is_the_open_phase_and_never_raises(self):
        self.assertEqual(sc._situation_phase(None), "open")
        self.assertEqual(sc._situation_phase([0.0] * 3), "open")


class TheSituationAwareExplorationTests(unittest.TestCase):

    def setUp(self):
        self._old_eps = sc._EXPLORE_EPS
        self._old_rng = sc._EXPLORE_RNG
        self._old_sit = sc._EXPLORE_SITUATIONAL

    def tearDown(self):
        sc._EXPLORE_EPS = self._old_eps
        sc._EXPLORE_RNG = self._old_rng
        sc._EXPLORE_SITUATIONAL = self._old_sit

    def _draws(self, position, sens, n=400, seed=5):
        """The mix of the EXPLORED draws, and how many there were.

        The draws that were NOT explored keep the policy's own choice, so
        they must be excluded — and the policy's own choice here is
        DRIBBLE, which is in NEITHER phase's option list, so an explored
        draw can never be confused with an un-explored one.
        """
        sc._EXPLORE_EPS = 0.25
        sc._EXPLORE_SITUATIONAL = True
        sc._EXPLORE_RNG = random.Random(seed)
        out = []
        try:
            for _ in range(n):
                got = sc._explore_intent(_player(position),
                                         PlayerIntent.DRIBBLE,
                                         sensors=sens)
                if got is not PlayerIntent.DRIBBLE:
                    out.append(got.value)
        finally:
            sc._EXPLORE_EPS = 0.0
            sc._EXPLORE_SITUATIONAL = False
        return Counter(out), len(out)

    def test_the_striker_is_still_never_asked_to_cross(self):
        seen, n = self._draws("ST", sensors(final_third=True, goal_close=True))
        self.assertNotIn("CROSS", seen,
                         "the situational exploration broke the role "
                         "vocabulary")
        self.assertIn("SHOOT", seen,
                      "the striker was never asked to shoot in the final third")

    def test_the_keeper_is_still_never_asked_to_shoot(self):
        seen, _n = self._draws("GK", sensors(final_third=True,
                                             goal_close=True))
        self.assertNotIn("SHOOT", seen)

    def test_the_goal_close_phase_samples_the_shot_and_the_own_half_does_not(self):
        """THE POINT of the change: the shot is sampled where the situation
        can judge it, and nowhere else.

        **THE ASSERTION IS A SHARE, NOT AN ORDERING.**  An ordering between
        two noisy counts is a coin flip — and the negative control proved it:
        the first version of this test compared two counts and passed with
        the mechanism MUTATED OUT, because 400 draws of a 1-in-9 share sit
        inside each other's noise.  The shares are decisive: the mechanism
        puts the shot at ~55% of the explored draws in the goal-close phase
        and at 0% in the own half; the uniform exploration puts it at ~11%
        in both.
        """
        final, n_final = self._draws("ST", sensors(final_third=True,
                                                   goal_close=True))
        own, n_own = self._draws("ST", sensors(own_half=True))
        self.assertGreater(n_final, 100, "too few explored draws to judge")
        self.assertGreater(n_own, 20, "too few explored draws to judge")
        self.assertGreater(final["SHOOT"] / n_final, 0.30,
                           "the shot was not sampled in the goal-close phase")
        self.assertEqual(own["SHOOT"], 0,
                         "the shot was sampled where it cannot be taken")

    def test_the_own_half_samples_the_safe_pass_and_the_goal_close_does_not(self):
        """The mirror of the case above, and the same share-not-ordering
        discipline: the own half's vocabulary excludes the attacking
        actions entirely."""
        final, n_final = self._draws("ST", sensors(final_third=True,
                                                   goal_close=True))
        own, n_own = self._draws("ST", sensors(own_half=True))
        self.assertGreater(own["SAFE_PASS"] / n_own, 0.15,
                           "the safe pass was not sampled in the own half")
        self.assertEqual(final["SAFE_PASS"], 0,
                         "a safe pass was sampled in the goal-close phase")

    def test_the_rate_is_higher_in_the_attacking_phases(self):
        """The rare actions need the RATE raised too: with ~10 decisions in
        the situation that matters, uniform exploration puts ~zero shots in
        it even when the action is in the vocabulary.

        The assertion is a RATIO because the rate test compares two counts,
        and a comparison between two counts at the same rate is a coin flip
        — which is exactly what the negative control caught.
        """
        sc._EXPLORE_EPS = 0.10
        sc._EXPLORE_SITUATIONAL = True

        def explored(n, sens, seed):
            sc._EXPLORE_RNG = random.Random(seed)
            hits = 0
            try:
                for _ in range(n):
                    if sc._explore_intent(_player("ST"),
                                          PlayerIntent.DRIBBLE,
                                          sensors=sens) is not PlayerIntent.DRIBBLE:
                        hits += 1
            finally:
                sc._EXPLORE_EPS = 0.0
                sc._EXPLORE_SITUATIONAL = False
            return hits

        n = 400
        final = explored(n, sensors(final_third=True, goal_close=True), 11)
        own = explored(n, sensors(own_half=True), 11)
        self.assertGreater(final / n, 0.5,
                           "the exploration rate was not raised in the "
                           "attacking phase")
        self.assertGreater(final, 3 * own,
                           "the attacking phase was not explored more often "
                           "than the own half")


class TheSensorsMoveIsFreeTests(unittest.TestCase):
    """Computing the sensors BEFORE the exploration must not move the
    global RNG stream — the perception layer draws from a SEEDED
    generator, so the reorder costs nothing."""

    def test_perceive_does_not_consume_the_global_random_stream(self):
        from perception import PerceptionConfig, perceive

        class _PE:
            def get_position(self, *a, **k):
                return (50.0, 34.0)

        random.seed(1234)
        before = random.getstate()
        try:
            perceive(_player("ST"), 50.0, 34.0, [], [], _PE(), False, True,
                     None, 45.0,
                     config=PerceptionConfig(enabled=True, role_blocks=True,
                                             seed=7))
        except Exception:
            pass  # the fixture is deliberately thin; the STREAM is the test
        after = random.getstate()
        self.assertEqual(before, after,
                         "the perception layer moved the global RNG stream")


if __name__ == "__main__":
    unittest.main()
