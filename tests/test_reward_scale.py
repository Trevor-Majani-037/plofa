"""THE PER-POSITION REWARD SCALE — the fix for the v5 census collapse.

THE DEFECT
----------
`synthetic_fitness` scores a brain with

    base    = 0.5 + 0.5 * clip(mean_reward, -1, 1)      # the football
    fitness = base
              + CONFIDENCE_GUARD_W * (mean_conf - 0.5)   # 0.05
              + DIVERSITY_GUARD_W * bdiv                 # 0.05
              - indecision_penalty / n_states            # up to 0.10

Every guard is a FIXED-WEIGHT term.  The reward is not: its magnitude comes
out of the position's own table, and the tables do not agree on a scale.
Measured on the v5 corpus, the reward's total swing — the range `base` can
take between the best available intent and the worst, over the production
300-state corpus — was:

    CAM 0.0119   CB 0.0090   CDM 0.0048   CF 0.0144   CM 0.0068
    GK  0.0067   LB 0.0184   LW  0.0120   RB 0.0131   RW 0.0114   ST 0.0147

against a largest single guard of **0.10**.  So the guards were 5x to 20x
THE OBJECTIVE, and the GA optimised peakedness instead of football.  That is
the measured cause of the v5 census: 8 of 11 positions down to <=3 intents,
mean_conf 0.78-0.97 against the shipped 0.19-0.38, every fitness number
unremarkable (0.60-0.67).

THE FIX
-------
Each position's fitted cells are divided by that position's own count-weighted
standard deviation at READ time, inside `FitnessSurrogate.expected_success`.
Every position then has an O(1) reward spread, the guards' fixed 0.05 becomes
a genuine tie-breaker everywhere, and the reward's swing is 5x-16x the
largest guard.

It is a MONOTONE per-position transform, so the ranking of intents WITHIN a
position is untouched — only the cross-position comparability changes, which
is the whole point.  The stored table stays in RAW units: the scale is applied
at read time, so a fitted value remains interpretable and the raw corpus can
be refitted freely.

WHAT THIS FILE PINS
-------------------
That the scale is computed from the FITTED cells (not the raw samples: the
shrinkage is what makes the spread vary, and the raw per-position sd varies
only ~4.8x), that it is actually applied to the value the GA reads, that it
never touches the FORBIDDEN value, that a position with too few cells is
left alone, that it survives a save/load round trip, that a surrogate saved
WITHOUT a scale still loads and does not divide, and — on the real corpus —
that the reward now outranks the guards for EVERY position.

⚠️  THE FIXTURE GOES THROUGH THE REAL `fit()`.  A suite that builds the
thing under test by hand is evidence about nothing (the lesson from
`test_surrogate_role_pooling.py`, where two negative-control mutations went
unseen because the fixture hand-built the family table).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_evolution import (  # noqa: E402
    CONFIDENCE_GUARD_W, DIVERSITY_GUARD_W, generate_state_corpus,
)
from decision_brain import PlayerIntent  # noqa: E402
from surrogate_collect import (  # noqa: E402
    FORBIDDEN_VALUE, FitnessSurrogate, load_raw_corpus,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CORPUS = os.path.join(ROOT, "brains", "surrogate_pos_v5.raw.json")

#: the indecision penalty's full range, from brain_evolution.synthetic_fitness
INDECISION_MAX = 0.10


def sensors_for(bucket: str) -> np.ndarray:
    """Build a sensor vector that lands in the given bucket.

    bucket = P{pressure}FT{final_third}OH{own_half}S{space}C{central}G{goal_close}
    """
    m = re.fullmatch(r"P(\d)FT(\d)OH(\d)S(\d)C(\d)G(\d)", bucket)
    assert m, f"bad bucket {bucket!r}"
    pressure, ft, oh, space, central, goal = (int(g) for g in m.groups())
    v = np.zeros(24, dtype=np.float64)
    v[16] = 1.0 if pressure else 0.0        # under_pressure
    v[12] = 1.0 if ft else 0.0               # is_final_third
    v[13] = 1.0 if oh else 0.0               # is_own_half
    v[9] = 0.8 if space else 0.0             # space_ahead
    v[15] = 1.0 if central else 0.0          # central_lane
    v[14] = 0.2 if goal else 0.9            # goal_distance_norm
    return v


#: six distinct buckets, so each (position, intent) below is its own cell
BUCKETS = [
    "P0FT0OH1S0C0G0",
    "P0FT0OH1S0C0G1",
    "P0FT0OH1S0C1G0",
    "P0FT0OH1S1C0G0",
    "P1FT0OH1S0C0G0",
    "P0FT1OH0S1C1G1",
]

#: the two positions' fitted values.  A centre-back's cells swing ~0.09 and a
#: midfielder's ~0.003 — the shape of the real v5 corpus.
CB_VALUES = [0.09, -0.09, 0.05, -0.05, 0.02, -0.02]
CM_VALUES = [0.003, -0.003, 0.0015, -0.0015, 0.0045, -0.0045]

#: generic intents every role may perform, so no cell is FORBIDDEN
INTENTS = [
    PlayerIntent.SAFE_PASS, PlayerIntent.RECYCLE, PlayerIntent.CARRY,
    PlayerIntent.DRIBBLE, PlayerIntent.SWITCH, PlayerIntent.PROGRESSIVE_PASS,
]


def build() -> FitnessSurrogate:
    """Two positions with very different spreads, fitted through the REAL fit.

    Three identical samples per cell, so the shrinkage (min_samples=3) does
    not apply and the cell value IS the sample mean.
    """
    data = []
    for pos, values in (("CB", CB_VALUES), ("CM", CM_VALUES)):
        for bucket, value, intent in zip(BUCKETS, values, INTENTS):
            for _ in range(3):
                data.append((sensors_for(bucket), intent, value, "PASS", pos))
    return FitnessSurrogate().fit(data)


def cell_values(sur: FitnessSurrogate, position: str):
    """The values the GA READS for a position, through the public API."""
    out = []
    for bucket, intents in sur.table.get(position, {}).items():
        for ik in intents:
            out.append(sur.expected_success(sensors_for(bucket),
                                            PlayerIntent(ik), position))
    return out


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


class ScaleComputationTests(unittest.TestCase):
    """The scale is computed from the FITTED cells of the position."""

    def setUp(self):
        self.sur = build()

    def test_the_scale_is_computed_for_each_position(self):
        self.assertIn("CB", self.sur.scales)
        self.assertIn("CM", self.sur.scales)
        self.assertNotIn("", self.sur.scales,
                         "the global row pools every position, so no single "
                         "scale is meaningful for it")

    def test_the_scale_is_the_count_weighted_std_of_the_fitted_cells(self):
        expected = _std(CB_VALUES)
        self.assertAlmostEqual(self.sur.scales["CB"], expected, places=9)

    def test_the_scale_is_not_one_for_a_position_with_a_real_spread(self):
        """Non-vacuity: if the scale were always 1.0 this whole file would
        pass without testing anything."""
        self.assertNotAlmostEqual(self.sur.scales["CB"], 1.0, places=6)
        self.assertNotAlmostEqual(self.sur.scales["CM"], 1.0, places=6)

    def test_the_two_positions_raw_spreads_really_do_differ(self):
        """Non-vacuity of the whole premise: the fixture must reproduce the
        shape of the real corpus, or it is testing nothing."""
        raw_cb = _std(CB_VALUES)
        raw_cm = _std(CM_VALUES)
        self.assertGreater(raw_cb / raw_cm, 10.0,
                           "the fixture's two positions are not far apart")


class ScaleApplicationTests(unittest.TestCase):
    """The scale is applied to the value the GA actually reads."""

    def setUp(self):
        self.sur = build()

    def test_the_value_the_ga_reads_is_the_raw_value_divided_by_the_scale(self):
        got = self.sur.expected_success(sensors_for(BUCKETS[0]),
                                        PlayerIntent.SAFE_PASS, "CB")
        self.assertAlmostEqual(got, CB_VALUES[0] / self.sur.scales["CB"],
                               places=9)

    def test_a_wide_position_and_a_narrow_position_end_up_with_the_same_spread(self):
        """THE CLAIM.  After the scale both positions' values the GA reads
        have an O(1) spread, so the guards' fixed weight is a tie-breaker
        everywhere instead of the objective where the reward was flat."""
        cb = _std(cell_values(self.sur, "CB"))
        cm = _std(cell_values(self.sur, "CM"))
        self.assertLess(abs(cb - cm), 0.05,
                        f"the scaled spreads still differ: CB {cb:.5f} "
                        f"CM {cm:.5f}")
        for name, v in (("CB", cb), ("CM", cm)):
            self.assertGreater(v, 0.5, f"{name}'s scaled spread is not O(1)")
            self.assertLess(v, 1.5, f"{name}'s scaled spread is not O(1)")

    def test_the_ranking_of_intents_inside_a_position_is_unchanged(self):
        """The scale is a positive constant per position, so it cannot
        reorder anything.  If this ever fails, the transform is not monotone
        and the brain ranking is not preserved."""
        for pos, values in (("CB", CB_VALUES), ("CM", CM_VALUES)):
            raw_order = [i for _, i in
                         sorted(zip(values, INTENTS), reverse=True)]
            read = [(self.sur.expected_success(sensors_for(bucket), intent, pos),
                     intent)
                    for bucket, intent in zip(BUCKETS, INTENTS)]
            read_order = [i for _, i in sorted(read, reverse=True)]
            self.assertEqual([i.value for i in raw_order],
                             [i.value for i in read_order],
                             f"{pos}: the scale reordered the intents")

    def test_the_veto_is_expressed_in_the_positions_own_units(self):
        """THE CORRECTION TO THE PLAN.

        `FORBIDDEN_VALUE = -1.0` is a RAW-scale number, chosen because the
        raw threat return's range is about [-0.6, +1.0].  Leaving it
        undivided after the scale is NOT safe: on the real corpus the scaled
        cells run as low as -7.59 and 59 cells fall below a fixed -1.0, so a
        fixed veto would sit ABOVE a legitimate action — and the GA would be
        handed a reason to perform the forbidden one.

        Dividing it with the same scale as everything else restores the
        invariant STRUCTURALLY.
        """
        got = self.sur.expected_success(sensors_for(BUCKETS[0]),
                                        PlayerIntent.CROSS, "CB")
        self.assertAlmostEqual(got, FORBIDDEN_VALUE / self.sur.scales["CB"],
                               places=9)

    def test_the_veto_is_below_every_legitimate_action(self):
        """'Ignorance must not outrank evidence' — and neither must a
        normalised veto float above a real reward.

        The comparison is between the values the GA READS, so the veto is
        the scaled one.
        """
        for pos in ("CB", "CM"):
            veto = self.sur.expected_success(sensors_for(BUCKETS[0]),
                                             PlayerIntent.CROSS, pos)
            for v in cell_values(self.sur, pos):
                self.assertLess(veto, v,
                                f"{pos}: a real reward fell below the veto")

    def test_ignorance_is_still_the_neutral_prior(self):
        """An ALLOWED action with no sample in this situation is ignorance:
        the neutral prior, not a divided number."""
        self.assertEqual(
            self.sur.expected_success(sensors_for(BUCKETS[0]),
                                      PlayerIntent.THROUGH_BALL, "CB"),
            0.0)


class TooFewCellsTests(unittest.TestCase):
    """A position with too few cells is left alone.

    Dividing by a standard deviation computed from one or two numbers would
    manufacture a gradient out of noise — and would be the same defect as
    the fixed guards, aimed the other way.
    """

    def test_a_position_with_fewer_than_three_cells_has_no_scale(self):
        data = []
        for i, (bucket, value) in enumerate(zip(BUCKETS[:2], CB_VALUES[:2])):
            for _ in range(3):
                data.append((sensors_for(bucket), INTENTS[i], value, "PASS",
                             "RW"))
        sur = FitnessSurrogate().fit(data)
        self.assertEqual(sur.scales.get("RW"), 1.0)
        got = sur.expected_success(sensors_for(BUCKETS[0]),
                                   INTENTS[0], "RW")
        self.assertAlmostEqual(got, CB_VALUES[0], places=9,
                               msg="a two-cell position was divided anyway")


class PersistenceTests(unittest.TestCase):
    """The scale travels with the table."""

    def setUp(self):
        self.sur = build()

    def test_the_scale_survives_a_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sur.json")
            self.sur.save(path)
            back = FitnessSurrogate.load(path)
        self.assertEqual(back.scales, self.sur.scales)
        self.assertAlmostEqual(
            back.expected_success(sensors_for(BUCKETS[0]),
                                  PlayerIntent.SAFE_PASS, "CB"),
            self.sur.expected_success(sensors_for(BUCKETS[0]),
                                      PlayerIntent.SAFE_PASS, "CB"),
            places=12)

    def test_a_surrogate_saved_without_a_scale_loads_and_does_not_divide(self):
        """Backwards compatibility: the shipped pre-2026-10-21 tables have no
        `scales` key.  They must load (no crash, no silent zero) and behave
        exactly as before."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sur.json")
            self.sur.save(path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data.pop("scales", None)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            back = FitnessSurrogate.load(path)
        self.assertEqual(back.scales, {})
        self.assertAlmostEqual(
            back.expected_success(sensors_for(BUCKETS[0]),
                                  PlayerIntent.SAFE_PASS, "CB"),
            CB_VALUES[0], places=9,
            msg="a scale-less surrogate still divided")


class RealCorpusBalanceTests(unittest.TestCase):
    """THE GATE, on the real corpus: the reward must outrank the guards.

    This is the measurement that decides whether the retrain is worth 30
    minutes of CPU.  It runs the production state corpus (300 states, seed
    123) for every position and asks how much the reward's own term can
    move the fitness — against the largest single guard.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(RAW_CORPUS):
            raise unittest.SkipTest(f"raw corpus not present: {RAW_CORPUS}")
        cls.sur = FitnessSurrogate().fit(load_raw_corpus(RAW_CORPUS))

    def _swing(self, sur, sensors, position):
        """The reward's swing over the corpus, in `base` units.

        Only the ROLE's own vocabulary is considered: the GA chooses among
        the actions the role performs.  The veto is NOT excluded by VALUE
        (it used to be, when it was the fixed -1.0) because a legitimate
        action can now scale below -1.0 too — excluding by value would drop
        the very rewards whose spread matters.
        """
        from role_features import ROLE_INTENTS, _role_family
        fam = _role_family(position) or ""
        allowed = [i for i in PlayerIntent if i.value in ROLE_INTENTS.get(fam, ())]
        best = worst = 0.0
        n = 0
        for row in sensors:
            vals = []
            for intent in allowed:
                v = sur.expected_success(row[:24], intent, position)
                if v is None:
                    continue
                vals.append(float(v))
            if not vals:
                continue
            best += max(vals)
            worst += min(vals)
            n += 1
        if n == 0:
            return 0.0
        return 0.5 * ((best - worst) / n)

    def test_every_position_has_a_scale(self):
        for pos in (p for p in self.sur.table if p):
            self.assertIn(pos, self.sur.scales)
            self.assertGreater(self.sur.scales[pos], 0.0)

    def test_the_reward_outranks_the_guards_for_every_position(self):
        """Without the scale the guards were 5x-20x the objective; with it
        the reward must be the dominant term everywhere."""
        scales = dict(self.sur.scales)
        try:
            for pos in sorted(p for p in self.sur.table if p):
                corpus = generate_state_corpus(pos, 300, 123, 0.0)
                self.sur.scales = {}          # the BEFORE arm
                raw = self._swing(self.sur, corpus, pos)
                self.sur.scales = dict(scales)   # the AFTER arm
                scaled = self._swing(self.sur, corpus, pos)
                self.assertGreater(raw, 0.0, f"{pos}: no reward swing at all")
                self.assertGreater(
                    scaled, INDECISION_MAX,
                    f"{pos}: the reward's swing ({scaled:.5f}) does not beat "
                    f"the largest single guard ({INDECISION_MAX}) — the "
                    f"per-position scale does not fix the balance")
                self.assertGreater(
                    scaled / INDECISION_MAX, 2.0,
                    f"{pos}: the reward only just beats the guard "
                    f"({scaled / INDECISION_MAX:.2f}x)")
        finally:
            self.sur.scales = dict(scales)

    def test_the_guards_stay_small_absolute_terms(self):
        """The guard weights themselves: fixed, and small enough to be a
        tie-breaker rather than the objective."""
        self.assertLessEqual(CONFIDENCE_GUARD_W, 0.05)
        self.assertLessEqual(DIVERSITY_GUARD_W, 0.05)

    def test_the_scaled_mean_is_far_from_the_clip(self):
        """`base = 0.5 + 0.5 * clip(mean_reward, -1, 1)` clips the MEAN
        reward.  A position whose cells all sit far from zero would
        saturate the clip for every brain and lose its ranking entirely."""
        for pos in (p for p in self.sur.table if p):
            vals, wts = [], []
            for bucket, intents in self.sur.table[pos].items():
                for ik, v in intents.items():
                    vals.append(float(v))
                    wts.append(float(self.sur.counts.get(pos, {})
                                      .get(bucket, {}).get(ik, 1)))
            tw = sum(wts)
            m = sum(v * w for v, w in zip(vals, wts)) / tw
            scaled_mean = m / (self.sur.scales.get(pos, 1.0) or 1.0)
            self.assertLess(
                abs(scaled_mean), 1.0,
                f"{pos}: the scaled mean sits at {scaled_mean:.3f}, so the "
                f"clip saturates for every brain")

    def _forbidden_intent_for(self, pos):
        """An intent this role may NOT perform AND has no cell of its own.

        `_lookup` returns the position's own cell BEFORE the forbidden
        check, so a position carrying a stray cell for a role-specific
        action (the corpus has one CB shot sample — the heuristic's noise)
        reads that cell instead of the veto.  That is pre-existing behaviour
        and is not what this test is about.
        """
        from role_features import ROLE_ONLY_INTENTS, _role_family
        fam = _role_family(pos) or ""
        for ik, allowed in ROLE_ONLY_INTENTS.items():
            if fam in allowed:
                continue
            if any(ik in intents for intents in self.sur.table[pos].values()):
                continue
            return ik
        return None

    def test_the_veto_is_below_every_legitimate_action_on_the_real_corpus(self):
        """The veto must be expressed in the position's own units.

        Measured: with a FIXED -1.0 the scaled cells run as low as -7.59 and
        59 cells across the corpus fall below it, so the veto would sit
        ABOVE a legitimate action and hand the GA a reason to cross.
        """
        from role_features import ROLE_ONLY_INTENTS
        checked = 0
        for pos in (p for p in self.sur.table if p):
            scale = self.sur.scales.get(pos, 1.0) or 1.0
            veto = FORBIDDEN_VALUE / scale
            for bucket, intents in self.sur.table[pos].items():
                for ik, v in intents.items():
                    got = self.sur.expected_success(sensors_for(bucket),
                                                    PlayerIntent(ik), pos)
                    self.assertLess(
                        veto, got,
                        f"{pos}: a legitimate action ({ik}) scored below the "
                        f"veto")
            ik = self._forbidden_intent_for(pos)
            if ik is None:
                continue
            checked += 1
            got = self.sur.expected_success(sensors_for(BUCKETS[0]),
                                            PlayerIntent(ik), pos)
            self.assertAlmostEqual(got, veto, places=9,
                                   msg=f"{pos}: the public API did not return "
                                       f"the scaled veto")
        self.assertGreater(checked, 0,
                           "no position had a forbidden intent to check — "
                           "the test proved nothing")


if __name__ == "__main__":
    unittest.main()
