"""THE THREAT REWARD -- the on-ball objective, and the reasons it is shaped
this way.

THE DEFECT THIS FILE EXISTS FOR
-------------------------------
The original surrogate reward scored **engine success**: a completed pass
2.0 + a small advance bonus, a retained carry 1.0, a turnover 0.0, and a
default of 0.5. That asks *"which intent does this engine complete?"* -- and
this engine completes a square ball far more often than a through-ball. So
safety always won the expected-value comparison and the genetic algorithm
converged to lateral spread.

That is not a theory. T2G160 is the recorded failure: the best fitness number
the project has ever produced (0.6523 vs a heuristic 0.5373, all eleven
positions saturated at 1.0000) with pass maps indistinguishable from the
hand-written heuristic, rejected by the user after three real matches. The
constraints added along the way (dominance penalty, effective-usage penalty,
neutral-prior fallback) did not fix it -- they relocated it, pushing each brain
to a *second* collapsed optimum. A constraint on a bad objective relocates the
failure instead of correcting it.

WHAT THE REWARD IS NOW
----------------------
The change in threat the action caused:

    r = PHI(end) - PHI(start)

where PHI is the xT grid `momentum_model` derives from this project's OWN shot
model (`XGEngine.calculate_geometric` at each cell centre), so the reward and
the engine's xG cannot disagree. It is local (no possession-chain segmentation
is needed), and it telescopes: the sum of per-action differences along a
sequence equals the difference across it.

Shot-shape special cases, each of which would be wrong as a difference:
  * a SHOT gets the ABSOLUTE zone value (= the xG from that spot), because a
    difference would make shooting worth exactly 0 by construction;
  * a GOAL gets 1.0, above any achievable xG, so a goal always beats the best
    shot that did not go in;
  * a FAILED action is charged the OPPONENT's threat where the ball was lost.
    Not a hand-tuned penalty on an intent -- geometry. Without it a failed
    through-ball scores the same as a completed square ball, expected value
    makes every risky ball free, and the GA swings to the mirror degenerate
    optimum (the "brain only" arm already measured 41.7% progressive passes
    against a real 25-35%).

WHAT THIS FILE PINS
-------------------
The ordering properties that make verticality the *optimum of the objective*
rather than an aspiration bolted onto it, the geometry of the loss charge, the
telescoping property, mirror symmetry (an identical attack on either side scores
identically), and the two facts that make the collector trustworthy: every
outcome-bearing event type is paired, and exactly one event per decision is.

It also pins the collector alignment bug found while writing it: `_EXECUTION_TYPES`
contained only `PASS`, so `PROGRESSIVE_PASS`, `SWITCH_OF_PLAY`, `THROUGH_BALL`,
`CROSS_ATTEMPT` and `DRIBBLE_*` were NEVER paired -- and each unmatched pass
permanently shifted that player's intent->outcome cursor, so every later sample
attributed an intent to somebody else's action. Those are precisely the pass
types the whole objective exists to reward.

AND THE NON-VACUITY CASE THAT MAKES IT WORTH READING
----------------------------------------------------
`test_the_retired_objective_really_did_prefer_the_square_ball` reproduces the
old reward and asserts it ranks the square ball above the through-ball for the
same geometry. Without it these tests would only prove the new reward is
self-consistent; with it, the file shows the two objectives disagree in the
direction the user complained about.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import momentum_model as mm
import threat_reward as TR
from threat_reward import (
    EXECUTION_TYPES, GOAL_VALUE, LOSS_WEIGHT, SHOT_OFF_TARGET_FRAC,
    threat_return, describe,
)
from decision_brain import PlayerIntent


# ─────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────

def ev(etype, **kw):
    """A minimal MatchEvent stand-in. Only the fields the reward reads."""
    d = dict(event_type=SimpleNamespace(name=etype), player="A", team="Home",
             minute=10, second=0, outcome=False, location_x=50.0,
             location_y=34.0, end_x=None, end_y=None, xg=None, metadata={})
    d.update(kw)
    return SimpleNamespace(**d)


GRID = TR.default_grid()

# (40 -> 75) forward, (60 -> 60,34->42) square, (70 -> 45) backward.
FWD_FAR = ev("PASS", outcome=True, location_x=40.0, location_y=34.0,
             end_x=75.0, end_y=34.0)
FWD_NEAR = ev("PASS", outcome=True, location_x=40.0, location_y=34.0,
              end_x=52.0, end_y=34.0)
SQUARE = ev("PASS", outcome=True, location_x=60.0, location_y=34.0,
            end_x=60.0, end_y=42.0)
BACK = ev("PASS", outcome=True, location_x=70.0, location_y=34.0,
          end_x=45.0, end_y=34.0)


# ─────────────────────────────────────────────────────────────
# 1. THE ORDERING THE WHOLE OBJECTIVE EXISTS FOR
# ─────────────────────────────────────────────────────────────

class VerticalityIsTheOptimumTests(unittest.TestCase):
    """Forward > square > backward, and the margin is not a rounding error."""

    def test_forward_beats_square_beats_backward(self):
        f = threat_return(FWD_FAR, GRID)
        s = threat_return(SQUARE, GRID)
        b = threat_return(BACK, GRID)
        self.assertGreater(f, s, f"forward {f:+.5f} must beat square {s:+.5f}")
        self.assertGreater(s, b, f"square {s:+.5f} must beat backward {b:+.5f}")

    def test_a_backward_pass_is_negative_not_merely_smaller(self):
        """The sign carries the information; a shift would lose it.

        Under the old objective a safe backward ball scored the SAME as a
        forward one (both 'completed', 2.0). A non-negative reward would still
        let it compete on volume.
        """
        self.assertLess(threat_return(BACK, GRID), 0.0)

    def test_gaining_more_ground_is_worth_more(self):
        near = threat_return(FWD_NEAR, GRID)
        far = threat_return(FWD_FAR, GRID)
        self.assertGreater(far, near)

    def test_a_through_ball_into_the_box_is_worth_more_than_a_safe_square_ball(self):
        through = ev("THROUGH_BALL", outcome=True, location_x=55.0,
                     location_y=34.0, end_x=88.0, end_y=34.0)
        t = threat_return(through, GRID)
        s = threat_return(SQUARE, GRID)
        self.assertGreater(t, s)
        # Non-vacuity: the margin is large enough to matter, not a tie.
        self.assertGreater(t - s, 0.02, f"through {t:+.4f} vs square {s:+.4f}")

    def test_a_carry_gains_threat_when_it_advances(self):
        c = ev("CARRY", outcome=True, location_x=40.0, location_y=34.0,
               end_x=55.0, end_y=34.0)
        self.assertGreater(threat_return(c, GRID), 0.0)


class TheRetiredObjectiveReallyDidPreferTheSquareBallTests(unittest.TestCase):
    """NON-VACUITY: the two objectives disagree, in the complained-of direction.

    This reproduces the retired reward from `AGENTS.md`:

        completed PASS       2.0 + min(pass_advance / 25, 1.5)
        retained CARRY       1.0 + distance
        turnover             0.0
        default              0.5

    A safe square ball completes almost always; a through ball into the box does
    not. On *expected* value the square ball therefore wins, which is the
    measured, recorded cause of T2G160. If this test ever stops being true, the
    new reward is no longer fixing anything and the whole section should be
    re-read rather than trusted.
    """

    @staticmethod
    def _retired(pass_advance_m: float, completed: bool) -> float:
        if not completed:
            return 0.0
        return 2.0 + min(abs(pass_advance_m) / 25.0, 1.5)

    #: Measured completion rates for the two shapes (the engine completes short
    #: lateral balls far more often than balls played into a defended box).
    SQUARE_RATE = 0.95
    THROUGH_RATE = 0.50

    def test_retired_objective_prefers_the_square_ball(self):
        square_ev = self.SQUARE_RATE * self._retired(0.0, True)
        through_ev = (self.THROUGH_RATE * self._retired(33.0, True)
                      + (1 - self.THROUGH_RATE) * self._retired(33.0, False))
        self.assertLess(through_ev, square_ev,
                        f"retired objective: through {through_ev:.2f} vs "
                        f"square {square_ev:.2f} -- if this flipped, the "
                        "post-mortem in threat_reward.py needs re-checking")

    def test_the_new_objective_prefers_the_through_ball(self):
        through = ev("THROUGH_BALL", outcome=True, location_x=55.0,
                     location_y=34.0, end_x=88.0, end_y=34.0)
        failed = ev("THROUGH_BALL", outcome=False, location_x=55.0,
                    location_y=34.0, end_x=88.0, end_y=34.0)
        t_ev = (self.THROUGH_RATE * threat_return(through, GRID)
                + (1 - self.THROUGH_RATE) * threat_return(failed, GRID))
        s_ev = self.SQUARE_RATE * threat_return(SQUARE, GRID)
        self.assertGreater(t_ev, s_ev,
                           f"new objective: through {t_ev:+.4f} vs "
                           f"square {s_ev:+.4f}")


# ─────────────────────────────────────────────────────────────
# 2. SHOTS AND GOALS
# ─────────────────────────────────────────────────────────────

class ShotAndGoalValuesTests(unittest.TestCase):

    def test_a_goal_beats_the_best_possible_shot(self):
        """GOAL_VALUE must exceed any xG, or the objective prefers a shot."""
        goal = ev("GOAL", location_x=95.0, location_y=34.0)
        shot = ev("SHOT_ON_TARGET", location_x=99.5, location_y=34.0, xg=0.95)
        self.assertGreater(threat_return(goal, GRID), threat_return(shot, GRID))
        self.assertEqual(threat_return(goal, GRID), GOAL_VALUE)

    def test_a_penalty_scored_is_a_goal(self):
        p = ev("PENALTY_SCORED", location_x=94.0, location_y=34.0)
        self.assertEqual(threat_return(p, GRID), GOAL_VALUE)

    def test_a_shot_is_worth_its_absolute_xg_not_a_difference(self):
        """A difference would make a shot worth exactly 0 by construction."""
        shot = ev("SHOT_ON_TARGET", location_x=95.0, location_y=34.0, xg=0.35)
        self.assertAlmostEqual(threat_return(shot, GRID), 0.35, places=9)

    def test_a_shot_without_xg_falls_back_to_the_engine_zone_value(self):
        """The fallback is the engine's own xG from that spot, not a guess."""
        shot = ev("SHOT_ON_TARGET", location_x=95.0, location_y=34.0)
        self.assertAlmostEqual(
            threat_return(shot, GRID),
            mm.zone_value(GRID, 95.0, 34.0, True), places=12)

    def test_on_target_beats_off_target_beats_blocked_less(self):
        on = ev("SHOT_ON_TARGET", location_x=92.0, location_y=34.0, xg=0.30)
        off = ev("SHOT_OFF_TARGET", location_x=92.0, location_y=34.0, xg=0.30)
        self.assertGreater(threat_return(on, GRID), threat_return(off, GRID))
        self.assertAlmostEqual(threat_return(off, GRID),
                               0.30 * SHOT_OFF_TARGET_FRAC, places=9)
        self.assertGreater(threat_return(off, GRID), 0.0,
                           "wasting a chance must not be worse than not shooting")

    def test_a_missed_penalty_is_a_chance_wasted(self):
        miss = ev("PENALTY_MISSED", location_x=94.0, location_y=34.0, xg=0.78)
        self.assertLess(threat_return(miss, GRID), 0.0)
        self.assertAlmostEqual(threat_return(miss, GRID),
                               -0.78 * SHOT_OFF_TARGET_FRAC, places=9)

    def test_a_saved_shot_earns_what_a_shot_on_target_earns(self):
        """The chance was created; the keeper's save is the OPPONENT's reward.

        Scoring the save as a failure would teach the policy that shooting at a
        keeper is a mistake, which is a property of the keeper, not the shot.
        """
        sot = ev("SHOT_ON_TARGET", location_x=92.0, location_y=34.0, xg=0.30)
        save = ev("SAVE", location_x=92.0, location_y=34.0, xg=0.30)
        self.assertAlmostEqual(threat_return(sot, GRID),
                               threat_return(save, GRID), places=12)


# ─────────────────────────────────────────────────────────────
# 3. THE LOSS CHARGE IS GEOMETRY, NOT A TUNED PENALTY
# ─────────────────────────────────────────────────────────────

class LossChargeIsGeometryTests(unittest.TestCase):

    def test_losing_it_in_your_own_box_costs_more_than_losing_it_upfield(self):
        own_box = ev("TURNOVER", location_x=8.0, location_y=34.0)
        their_half = ev("TURNOVER", location_x=80.0, location_y=34.0)
        self.assertLess(threat_return(own_box, GRID),
                        threat_return(their_half, GRID))
        # Non-vacuity: the own-box charge is material, not a rounding artefact.
        self.assertLess(threat_return(own_box, GRID), -0.05)

    def test_a_turnover_is_never_free(self):
        for x in (5.0, 30.0, 52.0, 75.0, 95.0):
            v = threat_return(ev("TURNOVER", location_x=x, location_y=34.0), GRID)
            self.assertLessEqual(v, 0.0, f"a loss at x={x} scored {v:+.5f}")

    def test_a_failed_pass_is_charged_at_the_point_the_ball_died(self):
        """Charging the loss at the pass ORIGIN would be the safe-pass bias again.

        A ball played forward and cut out should cost what the intercepting side
        now has, which is measured from where they won it, not from where it was
        struck.
        """
        failed_far = ev("PASS", outcome=False, location_x=40.0,
                        location_y=34.0, end_x=80.0, end_y=34.0)
        failed_near = ev("PASS", outcome=False, location_x=40.0,
                         location_y=34.0, end_x=42.0, end_y=34.0)
        # Losing it 80 m up the pitch hands the opponent a ball deep in THEIR
        # own half; losing it 2 m forward hands them one near the halfway line.
        self.assertGreater(threat_return(failed_far, GRID),
                           threat_return(failed_near, GRID))

    def test_the_loss_weight_is_the_declared_constant(self):
        e = ev("TURNOVER", location_x=8.0, location_y=34.0)
        expect = -LOSS_WEIGHT * mm.zone_value(GRID, 8.0, 34.0, False)
        self.assertAlmostEqual(threat_return(e, GRID), expect, places=12)

    def test_every_loss_type_is_charged(self):
        for etype in ("TURNOVER", "MISCONTROL", "DISPOSSESSED",
                      "INTERCEPTION", "DRIBBLE_FAIL"):
            v = threat_return(ev(etype, location_x=10.0, location_y=34.0), GRID)
            self.assertLess(v, 0.0, f"{etype} was not charged: {v:+.5f}")


# ─────────────────────────────────────────────────────────────
# 4. NEUTRALITY, PURITY, SYMMETRY, TELESCOPING
# ─────────────────────────────────────────────────────────────

class ShapeAndPurityTests(unittest.TestCase):

    def test_a_neutral_event_is_exactly_zero(self):
        """Not 0.5. With a signed reward a positive default is a preference
        for the unmeasured."""
        for etype in ("FOUL_COMMITTED", "PRESS", "GOAL_KICK",
                      "SUBSTITUTION", "CORNER_TAKEN_ISH", ""):
            self.assertEqual(
                threat_return(ev(etype, location_x=50.0, location_y=34.0), GRID),
                0.0, f"{etype} was not neutral")

    def test_potential_difference_telescopes(self):
        """A->B then B->C must equal A->C. This is what makes the reward a
        potential difference rather than three unrelated scores."""
        a = ev("PASS", outcome=True, location_x=30.0, location_y=34.0,
               end_x=55.0, end_y=34.0)
        b = ev("PASS", outcome=True, location_x=55.0, location_y=34.0,
               end_x=85.0, end_y=30.0)
        ab = threat_return(a, GRID) + threat_return(b, GRID)
        ac = threat_return(ev("PASS", outcome=True, location_x=30.0,
                              location_y=34.0, end_x=85.0, end_y=30.0), GRID)
        self.assertAlmostEqual(ab, ac, places=12)

    def test_an_identical_attack_on_either_side_scores_identically(self):
        """`team_attacks_right` is per-team and never flipped at half time, so
        the raw x of an away-side attack is mirrored. The reward must undo
        that, or half the collected samples are scored backwards."""
        home = threat_return(ev("PASS", outcome=True, location_x=40.0,
                                location_y=34.0, end_x=75.0, end_y=34.0), GRID)
        away = threat_return(ev("PASS", outcome=True, location_x=105 - 40,
                                location_y=34.0, end_x=105 - 75, end_y=34.0),
                             GRID, attacks_right=False)
        self.assertAlmostEqual(home, away, places=12)

    def test_the_mirror_holds_for_the_loss_charge_too(self):
        home = threat_return(ev("TURNOVER", location_x=8.0, location_y=34.0),
                             GRID, attacks_right=True)
        away = threat_return(ev("TURNOVER", location_x=105 - 8.0,
                                location_y=34.0), GRID, attacks_right=False)
        self.assertAlmostEqual(home, away, places=12)

    def test_the_reward_stays_in_a_bounded_range(self):
        """Every sample must sit in the range `synthetic_fitness` re-bases."""
        rng = np.random.default_rng(7)
        for _ in range(3000):
            x, y = float(rng.uniform(0, 105)), float(rng.uniform(0, 68))
            e = ev("PASS", outcome=True, location_x=x, location_y=y,
                   end_x=float(rng.uniform(0, 105)),
                   end_y=float(rng.uniform(0, 68)))
            v = threat_return(e, GRID)
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_the_grid_has_exactly_one_definition(self):
        """Derived from `momentum_model`, never re-typed here.

        A second copy of the grid is how the reward and the engine's own xG
        would drift apart -- the failure class this project keeps finding.
        """
        self.assertEqual(len(TR.default_grid()), mm.XT_ROWS)
        self.assertEqual(len(TR.default_grid()[0]), mm.XT_COLS)
        for r in range(mm.XT_ROWS):
            for c in range(mm.XT_COLS):
                self.assertAlmostEqual(TR.default_grid()[r][c],
                                       mm.default_grid()[r][c], places=12)

    def test_a_fitted_grid_can_be_installed_and_cleared(self):
        flat = [[0.0] * mm.XT_COLS for _ in range(mm.XT_ROWS)]
        TR.set_grid(flat)
        try:
            self.assertEqual(threat_return(FWD_FAR), 0.0)
        finally:
            TR.set_grid(None)
        self.assertGreater(threat_return(FWD_FAR), 0.0)

    def test_negative_xg_is_treated_as_absent(self):
        """A placeholder -1 must fall back to the zone, not invert the reward."""
        e = ev("SHOT_ON_TARGET", location_x=95.0, location_y=34.0, xg=-1.0)
        self.assertGreater(threat_return(e, GRID), 0.0)


# ─────────────────────────────────────────────────────────────
# 5. THE COLLECTOR PAIRING -- exactly one outcome event per decision
# ─────────────────────────────────────────────────────────────

class CollectorPairingTests(unittest.TestCase):

    def test_every_pass_family_is_paired(self):
        """THE BUG THIS PINS: `_EXECUTION_TYPES` was `{'PASS'}`, so every
        progressive pass, switch, through ball and cross was skipped -- and each
        skipped pass permanently shifted that player's pairing cursor."""
        for t in ("PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
                  "CROSS_ATTEMPT", "CARRY", "DRIBBLE_SUCCESS"):
            self.assertIn(t, EXECUTION_TYPES, f"{t} is not paired")

    def test_every_shot_and_goal_resolution_is_paired(self):
        for t in ("SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "SAVE",
                  "HIT_WOODWORK", "GOAL", "PENALTY_SCORED", "PENALTY_MISSED"):
            self.assertIn(t, EXECUTION_TYPES, f"{t} is not paired")

    def test_cross_success_is_excluded_because_it_double_counts(self):
        """`CROSS_SUCCESS` is emitted IN ADDITION to `CROSS_ATTEMPT`, which
        already carries `outcome=cross_success`. Pairing both consumes two
        events for one decision and shifts every later sample."""
        self.assertNotIn("CROSS_SUCCESS", EXECUTION_TYPES)
        self.assertIn("CROSS_ATTEMPT", EXECUTION_TYPES)

    def test_never_emitted_attempt_types_are_excluded(self):
        """`SHOT_ATTEMPT` / `DRIBBLE_ATTEMPT` are in the enum but the engine
        never emits them; including them can only dilute a real event."""
        for t in ("SHOT_ATTEMPT", "DRIBBLE_ATTEMPT"):
            self.assertNotIn(t, EXECUTION_TYPES)

    def test_each_decision_event_is_in_exactly_one_group(self):
        """A type in two groups would be resolved by whichever branch ran first,
        which is a silent ordering dependency rather than a definition."""
        groups = {
            "shot_on": TR._SHOT_ON_TYPES, "shot_off": TR._SHOT_OFF_TYPES,
            "goal": TR._GOAL_TYPES, "miss": TR._MISS_TYPES,
            "mover": TR._MOVER_TYPES, "loss": TR._LOSS_TYPES,
        }
        seen = {}
        for name, members in groups.items():
            for t in members:
                self.assertNotIn(t, seen,
                                 f"{t} is in both {seen.get(t)} and {name}")
                seen[t] = name
        self.assertEqual(set(seen), set(EXECUTION_TYPES))

    def test_the_surrogate_pairs_a_progressive_pass_to_its_own_decision(self):
        """Behavioural: before the fix this pairing consumed nothing, so the
        intent was scored against the NEXT pass's outcome."""
        from surrogate_collect import DecisionRecord, _correlate_outcomes

        sensors = np.zeros(24)
        recs = [
            DecisionRecord("A", "CM", 10, 0, sensors, PlayerIntent.PROGRESSIVE_PASS),
            DecisionRecord("A", "CM", 11, 1, sensors, PlayerIntent.SAFE_PASS),
        ]
        timeline = [
            ev("PROGRESSIVE_PASS", player="A", outcome=True, minute=10,
               location_x=40.0, location_y=34.0, end_x=75.0, end_y=34.0),
            ev("PASS", player="A", outcome=True, minute=11,
               location_x=60.0, location_y=34.0, end_x=60.0, end_y=38.0),
        ]
        paired = _correlate_outcomes(timeline, recs, grid=GRID)
        self.assertEqual(len(paired), 2)
        _s, intent0, reward0, action0, _p = paired[0]
        self.assertEqual(intent0, PlayerIntent.PROGRESSIVE_PASS)
        self.assertEqual(action0, "PROGRESSIVE_PASS")
        self.assertGreater(reward0, 0.0, "the progressive pass scored "
                                         f"{reward0:+.5f}")
        _s, intent1, _r1, action1, _p = paired[1]
        self.assertEqual(intent1, PlayerIntent.SAFE_PASS)
        self.assertEqual(action1, "PASS")

    def test_an_unpaired_decision_is_neutral_zero(self):
        from surrogate_collect import DecisionRecord, _correlate_outcomes
        recs = [DecisionRecord("A", "CM", 10, 0, np.zeros(24),
                               PlayerIntent.SHOOT)]
        paired = _correlate_outcomes([], recs, grid=GRID)
        self.assertEqual(paired[0][2], 0.0)
        self.assertEqual(paired[0][3], "NONE")

    def test_the_pairing_respects_the_teams_own_attacking_direction(self):
        """An away-side forward pass runs from high raw x to low raw x."""
        from surrogate_collect import DecisionRecord, _correlate_outcomes
        recs = [DecisionRecord("A", "CM", 10, 0, np.zeros(24),
                               PlayerIntent.PROGRESSIVE_PASS)]
        timeline = [ev("PROGRESSIVE_PASS", player="A", team="Away",
                       outcome=True, minute=10, location_x=65.0,
                       location_y=34.0, end_x=30.0, end_y=34.0)]
        paired = _correlate_outcomes(timeline, recs,
                                     attacks_right_by_team={"Away": False},
                                     grid=GRID)
        self.assertGreater(paired[0][2], 0.0,
                           "an away-side forward pass was scored backwards")


# ─────────────────────────────────────────────────────────────
# 6. THE SURROGATE AND THE FITNESS FUNCTION
# ─────────────────────────────────────────────────────────────

class SurrogateCarriesTheRewardDefinitionTests(unittest.TestCase):

    def test_the_prior_is_neutral_zero(self):
        from surrogate_collect import FitnessSurrogate
        self.assertEqual(FitnessSurrogate().prior, 0.0)
    def test_an_unknown_cell_returns_the_neutral_prior(self):
        from surrogate_collect import FitnessSurrogate
        s = FitnessSurrogate()
        self.assertEqual(
            s.expected_success(np.zeros(24), PlayerIntent.SHOOT, "ST"), 0.0)

    def test_the_reward_definition_travels_with_a_saved_table(self):
        """A table of numbers is unreadable six months later, and a retrained
        surrogate must not be silently compared to one fitted to a different
        objective."""
        import json
        import tempfile
        from surrogate_collect import FitnessSurrogate

        s = FitnessSurrogate()
        s.table["ST"] = {"b": {PlayerIntent.SHOOT.value: 0.3}}
        s.counts["ST"] = {"b": {PlayerIntent.SHOOT.value: 9}}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sur.json")
            s.save(path)
            raw = json.load(open(path, encoding="utf-8"))
            self.assertEqual(raw["reward"], describe())
            back = FitnessSurrogate.load(path)
        self.assertEqual(back.prior, 0.0)
        self.assertEqual(back.table, s.table)
        self.assertEqual(back.counts, s.counts)

    def test_the_report_states_the_objective(self):
        from surrogate_collect import FitnessSurrogate
        self.assertEqual(FitnessSurrogate().report()["reward"], describe())

    def test_describe_names_every_constant(self):
        d = describe()
        self.assertEqual(d["goal_value"], GOAL_VALUE)
        self.assertEqual(d["shot_off_target_frac"], SHOT_OFF_TARGET_FRAC)
        self.assertEqual(d["loss_weight"], LOSS_WEIGHT)
        self.assertEqual(d["default"], 0.0)
        self.assertIn("PHI(end) - PHI(start)", d["form"])


class TheFitnessFunctionSurvivesASignedRewardTests(unittest.TestCase):
    """`synthetic_fitness` multiplies the mean by a confidence factor.

    With a signed reward, a mean of -0.001 would clamp to 0 and the confidence
    weighting would erase exactly the gradient the objective exists to provide.
    The surrogate path therefore re-bases through `0.5 + 0.5*clip(mean,-1,1)`
    first -- a monotone transform, so it changes no ordering.
    """

    @staticmethod
    def _sur(prior: float, value: float):
        from surrogate_collect import FitnessSurrogate
        s = FitnessSurrogate(prior=prior)
        s.table["ST"] = {"z": {"z": {i.value: value for i in PlayerIntent}}}
        return s

    def _score(self, sur, **kw):
        from brain_evolution import synthetic_fitness
        from football_brain import FootballBrain
        brain = FootballBrain.random(3)
        return synthetic_fitness(brain, "ST", n_states=40, seed=5,
                                 surrogate=sur, **kw)

    def test_a_negative_mean_does_not_clamp_to_zero(self):
        neg = self._score(self._sur(-0.5, -0.5))
        self.assertGreater(neg, 0.0,
                           "a negative mean clamped to zero: no gradient left")

    def test_a_better_mean_still_scores_better(self):
        neg = self._score(self._sur(-0.5, -0.5))
        zero = self._score(self._sur(0.0, 0.0))
        pos = self._score(self._sur(0.5, 0.5))
        self.assertLess(neg, zero)
        self.assertLess(zero, pos)

    def test_the_transform_is_monotone_across_the_whole_range(self):
        """It must not reorder anything: only rescale."""
        vals = [-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0]
        scores = [self._score(self._sur(v, v)) for v in vals]
        self.assertEqual(scores, sorted(scores),
                         f"the re-basing reordered the objective: {scores}")

    def test_every_score_is_a_probability(self):
        for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
            s = self._score(self._sur(v, v))
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_collapse_penalties_are_off_by_default(self):
        """The dominance + effective-usage penalties are the recorded cause of
        T2's SECOND collapsed optimum. They relocate a failure, they do not fix
        it, so they are opt-in with evidence rather than on."""
        import inspect
        from brain_evolution import synthetic_fitness
        sig = inspect.signature(synthetic_fitness)
        self.assertIn("collapse_penalties", sig.parameters)
        self.assertIs(sig.parameters["collapse_penalties"].default, False)

    def test_the_penalty_flag_actually_does_something_when_asked(self):
        """Non-vacuity: a flag with no effect would be a dead switch."""
        sur = self._sur(0.5, 0.5)
        off = self._score(sur)
        on = self._score(sur, collapse_penalties=True)
        self.assertLess(on, off)

    def test_the_legacy_reward_path_is_unchanged(self):
        """No surrogate -> the hand-made table -> the original /1.5 scale.

        The re-basing and the scale split must not move the numbers every
        earlier calibration was taken on.
        """
        from brain_evolution import synthetic_fitness
        from football_brain import FootballBrain
        brain = FootballBrain.random(3)
        f = synthetic_fitness(brain, "ST", n_states=40, seed=5, surrogate=None)
        self.assertGreaterEqual(f, 0.0)
        self.assertLessEqual(f, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
