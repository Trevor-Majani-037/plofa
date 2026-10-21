"""The ROLE-SCOPED FALLBACK — and the cross-role leak it closes.

THE DEFECT
----------
`FitnessSurrogate._lookup` used to fall back from the position's cell to the
GLOBAL row's cell for the same bucket and intent.  The global row pools
EVERY position, so a ROLE-SPECIFIC action's value leaked across roles.

Measured, with the v4 surrogate: the ST's own table has no CROSS cell in any
bucket, so the striker was reading the WINGERS' crossing value:

    ST, bucket P1FT1OH0S0C1G1 (final third, central, goal-close, no space)
        CROSS   own=--   global=+0.0420   lookup=+0.0420  <- BORROWED
        SHOOT   own=-0.0017  global=-0.0270  lookup=-0.0017

and in the space bucket the global CROSS was still above the ST's own SHOOT.
The GA duly evolved a striker who CROSSED in 43.3% of states and chose SHOOT
in 0% — on a 2,000-state corpus, against a broad 8-intent mix for the
shipped T1 striker (max share 35%).

THE FIX
-------
The fallback chain is now: position cell -> ROLE FAMILY cell -> the global
row, and the global row may only lend an intent the ROLE FAMILY actually
performed.  The family map is the project's own (`role_features.ROLE_FAMILY`):
ST/CF, LW/RW, LB/RB, CM, CAM, CDM, CB, GK.  The family performed SHOOT but
never CROSS, so a striker can never borrow a crossing value.

WHAT THIS FILE PINS
-------------------
That the chain is position -> family -> role-appropriate global -> ignorance;
that a role-inappropriate intent is NEVER lent across roles; that the
role-appropriate vocabulary is DERIVED FROM THE DATA rather than invented;
and that the family table and the vocabulary survive a save/load round trip
(the shipped surrogate file is read by the trainer and the harnesses).

⚠️  THE FIXTURE GOES THROUGH `fit()`.  The first version of this file
hand-built `family_table` and `family_intents`, which meant the two
mutations that change `fit` itself were MISSES in the negative control — a
suite that builds the thing under test by hand is evidence about nothing.
"""
from __future__ import annotations

import os
import random
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_brain import PlayerIntent  # noqa: E402
from role_features import ROLE_FAMILY  # noqa: E402
from surrogate_collect import FitnessSurrogate  # noqa: E402


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


CLEAR_CHANCE = "P1FT1OH0S1C1G1"   # final third, central, goal-close, space
#: a bucket the ST has no cell in, so the FAMILY row is the fallback
MIDEFIELD = "P0FT0OH0S1C1G0"

#: three identical samples per cell, so the shrinkage (min_samples=3) does
#: not apply and the cell value IS the sample mean
N = 3


def build() -> FitnessSurrogate:
    """A miniature corpus, fitted through the REAL `fit`.

    The wingers cross and pay.  The striker never crosses, and shoots at a
    small loss.  A centre-back dribbles, which no other role does.
    """
    data = []

    def add(pos, intent, reward, bucket, n=N):
        for _ in range(n):
            data.append((sensors_for(bucket), intent, reward, "PASS", pos))

    add("LW", PlayerIntent.CROSS, 0.0420, CLEAR_CHANCE)
    add("RW", PlayerIntent.CROSS, 0.0420, CLEAR_CHANCE)
    add("LB", PlayerIntent.CROSS, 0.0400, CLEAR_CHANCE)
    add("CB", PlayerIntent.DRIBBLE, 0.0500, CLEAR_CHANCE)
    add("ST", PlayerIntent.SHOOT, -0.0017, CLEAR_CHANCE)
    add("ST", PlayerIntent.SAFE_PASS, -0.0278, CLEAR_CHANCE)
    add("CM", PlayerIntent.SAFE_PASS, 0.0100, CLEAR_CHANCE)
    add("CM", PlayerIntent.PROGRESSIVE_PASS, 0.0200, CLEAR_CHANCE)
    # the CF plays in the ST's family: the ST has no cell in MIDEFIELD, so
    # the family row is what answers there.  The LW shoots badly in the same
    # bucket, so the GLOBAL row for that bucket DIFFERS from the ST family's
    # row — without that difference the test could not tell the two apart.
    add("CF", PlayerIntent.SHOOT, 0.0300, MIDEFIELD)
    add("LW", PlayerIntent.SHOOT, -0.0500, MIDEFIELD)
    return FitnessSurrogate().fit(data)


class RoleScopedLookupTests(unittest.TestCase):

    def setUp(self):
        self.sur = build()

    def test_a_striker_cannot_borrow_the_wingers_crossing_value(self):
        """CROSS is a WIDE-role action.  The striker may not read the
        wingers' value, and the answer is FORBIDDEN, not ignorance: the
        neutral prior (0.0) would outrank a slightly-negative real cell and
        the striker would take a share of crossing for free."""
        from surrogate_collect import FORBIDDEN_VALUE
        got = self.sur._lookup(CLEAR_CHANCE, "CROSS", "ST")
        self.assertEqual(got, FORBIDDEN_VALUE,
                         "the ST was handed the wingers' crossing value")

    def test_forbidden_is_worse_than_any_real_value(self):
        """'Ignorance must not outrank evidence'."""
        from surrogate_collect import FORBIDDEN_VALUE
        self.assertLess(FORBIDDEN_VALUE, -0.6,
                        "the forbidden value is inside the real reward range")
        self.assertLess(FORBIDDEN_VALUE, self.sur.prior,
                        "the forbidden value is not below the neutral prior")

    def test_a_striker_still_reads_his_own_shooting_value(self):
        got = self.sur._lookup(CLEAR_CHANCE, "SHOOT", "ST")
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, -0.0017, places=6)

    def test_a_winger_can_still_borrow_the_other_wingers_crossing_value(self):
        """The fix must not starve the roles that DO perform the intent."""
        got = self.sur._lookup(CLEAR_CHANCE, "CROSS", "RW")
        self.assertIsNotNone(got, "the RW lost the crossing value")
        self.assertAlmostEqual(got, 0.0420, places=6)

    def test_a_fullback_can_still_borrow_the_fullback_family(self):
        got = self.sur._lookup(CLEAR_CHANCE, "CROSS", "RB")
        self.assertIsNotNone(got, "the RB lost the crossing value")

    def test_an_intent_nobody_in_the_role_performed_is_not_lent(self):
        """The global row is only available for an intent the FAMILY did.
        The CB dribbled; the ST family never did, so the ST cannot borrow
        the dribbling value from the global row."""
        self.assertIn("DRIBBLE", self.sur.table[""][CLEAR_CHANCE])
        got = self.sur._lookup(CLEAR_CHANCE, "DRIBBLE", "ST")
        self.assertIsNone(got, "the ST was lent another role's dribbling value")

    def test_ignorance_returns_the_neutral_prior_through_the_public_api(self):
        """An ALLOWED action with no sample in this situation is ignorance:
        the neutral prior, NOT a negative value and not a borrowed one."""
        self.assertEqual(
            self.sur.expected_success(sensors_for(CLEAR_CHANCE),
                                      PlayerIntent.THROUGH_BALL, "ST"),
            0.0)

    def test_a_role_specific_action_the_role_cannot_do_is_forbidden(self):
        """Through the PUBLIC API: the striker asking to cross is told no.

        THE PIN IS THE INVARIANT, NOT THE LITERAL CONSTANT.  The veto is
        expressed in the position's own units (the per-position reward
        scale, 2026-10-21), so once a position has a real scale the
        returned number is `FORBIDDEN_VALUE / scale` and not the constant
        itself.  What must hold is that the veto is below EVERY legitimate
        action for that position — a veto that sits above a legitimate
        action is not a veto, it is a reason to perform the forbidden one.
        """
        from surrogate_collect import FORBIDDEN_VALUE
        got = self.sur.expected_success(sensors_for(CLEAR_CHANCE),
                                        PlayerIntent.CROSS, "ST")
        self.assertLessEqual(got, FORBIDDEN_VALUE,
                             "the forbidden action was not returned as a veto")
        for legit in self.sur.table["ST"][CLEAR_CHANCE].values():
            self.assertLess(got, legit,
                            "a legitimate action scored below the veto")

    def test_the_family_row_is_used_before_the_global_row(self):
        """The family row is the fallback when the POSITION has no cell —
        the position's own cell always wins over it."""
        # the ST has a cell in CLEAR_CHANCE, so the family cannot override it
        self.assertAlmostEqual(
            self.sur._lookup(CLEAR_CHANCE, "SHOOT", "ST"), -0.0017, places=6)
        # in a bucket the ST has NO cell of its own, the family row answers
        got = self.sur._lookup(MIDEFIELD, "SHOOT", "ST")
        self.assertIsNotNone(got, "the ST family row was not consulted")
        self.assertAlmostEqual(got, 0.0300, places=6)
        # and the global row for that bucket is NOT the same number, so this
        # test can tell the family row from the global one
        self.assertNotAlmostEqual(
            self.sur.table[""][MIDEFIELD]["SHOOT"], 0.0300, places=6)

    def test_ignorance_returns_the_neutral_prior_through_the_public_api_OLD(self):
        self.skipTest("superseded: a role-specific action the role cannot "
                      "perform is FORBIDDEN, not ignorance")

    def test_the_winger_is_not_forbidden_the_crossing(self):
        """The rule must not starve the role that DOES perform the action."""
        from surrogate_collect import FORBIDDEN_VALUE
        got = self.sur.expected_success(sensors_for(CLEAR_CHANCE),
                                        PlayerIntent.CROSS, "LW")
        self.assertNotEqual(got, FORBIDDEN_VALUE,
                            "the winger was forbidden the crossing")

    def test_the_vocabulary_is_derived_from_the_data_not_invented(self):
        self.assertIn("SHOOT", self.sur.family_intents["ST"])
        self.assertNotIn("CROSS", self.sur.family_intents["ST"])
        self.assertIn("CROSS", self.sur.family_intents["WING"])
        self.assertIn("CROSS", self.sur.family_intents["FB"])
        self.assertIn("DRIBBLE", self.sur.family_intents["CB"])

    def test_a_numbered_position_still_finds_its_family(self):
        """POSITIONS ARE NUMBERED.  The live roster path emits CM1 / CM2 /
        CB1 / CB2, and a family lookup that does not strip the digits is
        INERT on those names — the same trap AGENTS.md records from
        `_cb_same_side_corridor`, where a guard that looked right matched
        nothing at all."""
        from surrogate_collect import FORBIDDEN_VALUE
        sur = build()
        # a numbered striker must read exactly what "ST" reads
        self.assertEqual(sur._lookup(CLEAR_CHANCE, "SHOOT", "ST1"),
                         sur._lookup(CLEAR_CHANCE, "SHOOT", "ST"))
        # and must still be refused the crossing value
        self.assertEqual(sur._lookup(CLEAR_CHANCE, "CROSS", "ST2"),
                         FORBIDDEN_VALUE)

    def test_the_family_map_strips_the_digits(self):
        from role_features import _role_family
        self.assertEqual(_role_family("CM1"), "CM")
        self.assertEqual(_role_family("CB2"), "CB")
        self.assertEqual(_role_family("ST"), "ST")

    def test_the_family_tables_survive_a_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sur.json")
            self.sur.save(path)
            back = FitnessSurrogate.load(path)
        self.assertEqual(back.family_intents, self.sur.family_intents)
        self.assertEqual(back.family_table, self.sur.family_table)
        self.assertAlmostEqual(
            back._lookup(CLEAR_CHANCE, "SHOOT", "ST"), -0.0017, places=6)
        from surrogate_collect import FORBIDDEN_VALUE
        self.assertEqual(back._lookup(CLEAR_CHANCE, "CROSS", "ST"),
                         FORBIDDEN_VALUE)


class ExplorationCoverageTests(unittest.TestCase):
    """ε-EXPLORATION: the corpus must cover the actions the policy never
    chooses.

    The corpus is generated by a policy, and the policy never chooses the
    rare actions: the CM family performed NO through-ball in 4,011 real
    decisions, and the striker performed exactly one shot.  A surrogate
    fitted on that corpus cannot express the value of the actions that
    matter, and the genetic algorithm then learns the wrong thing.

    So the collector occasionally replaces the decision with a random action
    the player's ROLE can perform, and the engine really executes it.
    """

    def _player(self, position):
        return SimpleNamespace(position=position, name="X")

    def test_exploration_is_off_by_default(self):
        import surrogate_collect as sc
        old = sc._EXPLORE_EPS
        sc._EXPLORE_EPS = 0.0
        try:
            for intent in PlayerIntent:
                self.assertIs(sc._explore_intent(self._player("ST"), intent),
                              intent)
        finally:
            sc._EXPLORE_EPS = old

    def test_the_striker_is_never_asked_to_cross(self):
        import surrogate_collect as sc
        from role_features import ROLE_INTENTS
        sc._EXPLORE_EPS = 1.0
        sc._EXPLORE_RNG = random.Random(7)
        try:
            seen = {sc._explore_intent(self._player("ST"), PlayerIntent.SAFE_PASS)
                    for _ in range(200)}
        finally:
            sc._EXPLORE_EPS = 0.0
        self.assertNotIn(PlayerIntent.CROSS, seen,
                         "the striker was asked to cross")
        names = {i.value for i in seen}
        self.assertTrue(names.issubset(ROLE_INTENTS["ST"]),
                        f"the exploration used an action the role cannot do: "
                        f"{sorted(names - set(ROLE_INTENTS['ST']))}")

    def test_the_midfielder_can_be_asked_for_a_through_ball(self):
        """The gap this exists to close."""
        import surrogate_collect as sc
        sc._EXPLORE_EPS = 1.0
        sc._EXPLORE_RNG = random.Random(11)
        try:
            seen = {sc._explore_intent(self._player("CM"), PlayerIntent.SAFE_PASS)
                    for _ in range(300)}
        finally:
            sc._EXPLORE_EPS = 0.0
        self.assertIn(PlayerIntent.THROUGH_BALL, seen,
                      "the midfielder was never asked for a through ball")

    def test_an_unknown_role_is_left_alone(self):
        import surrogate_collect as sc
        sc._EXPLORE_EPS = 1.0
        sc._EXPLORE_RNG = random.Random(3)
        try:
            self.assertIs(sc._explore_intent(self._player("???"),
                                              PlayerIntent.SAFE_PASS),
                          PlayerIntent.SAFE_PASS)
        finally:
            sc._EXPLORE_EPS = 0.0


class TheRealSurrogateDoesNotLeakTests(unittest.TestCase):
    """The shipped v4 surrogate, read with the new lookup chain."""

    PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "brains", "surrogate_pos_v4.json")

    def setUp(self):
        if not os.path.exists(self.PATH):
            self.skipTest("v4 surrogate not present")
        import json
        with open(self.PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.sur = FitnessSurrogate()
        self.sur.table = data.get("table", {})
        self.sur.counts = data.get("counts", {})
        self.sur.family_table = data.get("family_table", {})
        self.sur.family_intents = {k: set(v) for k, v in
                                   data.get("family_intents", {}).items()}

    def test_the_striker_cannot_read_a_crossing_value(self):
        from surrogate_collect import FORBIDDEN_VALUE
        for bucket in self.sur.table.get("", {}):
            if "CROSS" in self.sur.table[""][bucket]:
                self.assertEqual(
                    self.sur._lookup(bucket, "CROSS", "ST"),
                    FORBIDDEN_VALUE,
                    f"the ST was lent the crossing value in {bucket}")

    def test_the_winger_still_reads_a_crossing_value(self):
        found = False
        for bucket, intents in self.sur.table.get("LW", {}).items():
            if "CROSS" in intents:
                found = True
                self.assertIsNotNone(self.sur._lookup(bucket, "CROSS", "LW"))
        self.assertTrue(found, "the LW has no crossing cell at all")


if __name__ == "__main__":
    unittest.main()
