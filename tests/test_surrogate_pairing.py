"""THE PAIRING — how a decision is joined to the outcome it caused.

THE DEFECT THIS FILE EXISTS FOR
-------------------------------
The surrogate is fitted from (sensor vector, intent, reward) triples, so the
join between a decision and the event it caused decides what the surrogate
learns.  The original join was "the first execution event by that player at
or after this minute", consuming one event per record.

That is wrong, because a decision can emit MORE THAN ONE execution event: a
failed pass emits PASS *and* TURNOVER / DISPOSSESSED / MISCONTROL.  The
cursor consumes one event per record, so the extra event is charged against
the NEXT decision of that player, and the alignment slides for the rest of
the match.

Measured over 3 real matches, the self-consistency of the resulting pairs
(intent vs the event actually paired with it) was:

    SHOOT         1/8    (12.5%)
    CROSS         0/18   (0%)
    THROUGH_BALL  1/10   (10%)
    DRIBBLE       3/212  (1.4%)
    SAFE_PASS     408/630 (64.8%)

So a shot decision was charged for somebody else's carry.  The surrogate
fitted from that join ranked SHOOT at **-0.0133** — the WORST of all ten
intents — which would have taught the genetic algorithm never to shoot.
That is T2's documented second collapsed optimum (ST never shoots, CM
abandons through-balls), reproduced by a defect in the DATA rather than in
the objective.

THE JOIN NOW
-------------
`ChainDispatcher.possession` / `.attack` / `.set_piece` are the units of
work: the chain that makes the on-ball decision also produces the event, and
it emits the ACTION before its consequences.  So a decision joins to the
first ACTION event of its own chain, by the same player, skipping

  * events that are not the player's action at all — a BALL_RECEIPT is the
    pass ARRIVING at the new carrier (emitted by the carrier, before the
    action he then takes), and a PRESS is not an on-ball action; and
  * the CONSEQUENCES of a failed action — the failed action's OWN event
    already carries the loss charge in `threat_reward` (a PASS with
    outcome=False is charged the opponent's threat where the ball died), so
    claiming the consequence as well would DOUBLE-CHARGE the same loss
    against whichever decision came next.

Measured on the same 3 matches, same decisions, same rewards:

    SAFE_PASS       BALL_RECEIPT:219 -> PASS:480
    SWITCH          -> SWITCH_OF_PLAY:204
    SHOOT           mean -0.0133 -> +0.0372   (highest of all intents)
    drop rate       0.0%  (every decision finds its own chain's action)

WHAT THIS FILE PINS
-------------------
The join itself (not the reward, which `test_threat_reward.py` owns): that
the action is taken from the decision's OWN chain, that a receipt and a
press are not actions, that a consequence is not claimed so the loss is not
double-charged, that no event is claimed twice, that an unexecuted decision
is NEUTRAL rather than charged with another player's event, and that the
production path really installs the capture (the wiring lives in a different
file from the join, so no behavioural test of the join can see it).
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_brain import PlayerIntent  # noqa: E402
import threat_reward as TR  # noqa: E402
import surrogate_collect as SC  # noqa: E402


GRID = TR.default_grid()


def ev(etype, **kw):
    """A minimal MatchEvent stand-in. Only the fields the join and the
    reward read."""
    d = dict(event_type=SimpleNamespace(name=etype), player="A", team="Home",
             minute=10, second=0, outcome=False, location_x=50.0,
             location_y=34.0, end_x=None, end_y=None, xg=None, metadata={})
    d.update(kw)
    return SimpleNamespace(**d)


def rec(intent, player="A", minute=10, order=0, position="CM"):
    return SC.DecisionRecord(player, position, minute, order,
                              np.zeros(24), intent)


class CausalJoinTests(unittest.TestCase):
    """The join itself."""

    def test_the_decision_joins_to_the_action_of_its_own_chain(self):
        calls = [([rec(PlayerIntent.SAFE_PASS)],
                  [ev("PASS", outcome=True, location_x=40.0, end_x=75.0,
                      end_y=34.0)])]
        paired = SC._correlate_outcomes_causal(calls, grid=GRID)
        self.assertEqual(len(paired), 1)
        _s, intent, reward, action, _p = paired[0]
        self.assertEqual(intent, PlayerIntent.SAFE_PASS)
        self.assertEqual(action, "PASS")
        self.assertGreater(reward, 0.0,
                           "a forward pass scored " + f"{reward:+.5f}")

    def test_a_ball_receipt_is_not_the_action(self):
        """The pass ARRIVING at the new carrier is emitted by the carrier,
        before the action he then takes.  Pairing the decision with it
        measures the receipt, not the action."""
        calls = [([rec(PlayerIntent.PROGRESSIVE_PASS)],
                  [ev("BALL_RECEIPT", outcome=True, location_x=40.0,
                      end_x=60.0, end_y=34.0),
                   ev("PROGRESSIVE_PASS", outcome=True, location_x=40.0,
                      end_x=75.0, end_y=34.0)])]
        paired = SC._correlate_outcomes_causal(calls, grid=GRID)
        self.assertEqual(paired[0][3], "PROGRESSIVE_PASS")

    def test_a_press_is_not_the_action(self):
        calls = [([rec(PlayerIntent.SAFE_PASS)],
                  [ev("PRESS", outcome=True, location_x=40.0),
                   ev("PASS", outcome=True, location_x=40.0, end_x=75.0,
                      end_y=34.0)])]
        paired = SC._correlate_outcomes_causal(calls, grid=GRID)
        self.assertEqual(paired[0][3], "PASS")

    def test_a_consequence_is_not_claimed_so_the_loss_is_not_double_charged(self):
        """A failed pass emits PASS (outcome=False) AND a TURNOVER.  The
        PASS already carries the loss charge; claiming the TURNOVER as well
        charges the same loss twice, against the next decision."""
        calls = [([rec(PlayerIntent.SAFE_PASS, order=0),
                   rec(PlayerIntent.PROGRESSIVE_PASS, order=1)],
                  [ev("PASS", outcome=False, location_x=40.0, end_x=45.0,
                      end_y=34.0),
                   ev("TURNOVER", outcome=False, location_x=45.0,
                      end_x=50.0, end_y=34.0)])]
        paired = SC._correlate_outcomes_causal(calls, grid=GRID)
        self.assertEqual(len(paired), 2)
        self.assertEqual(paired[0][3], "PASS",
                         "the first decision did not get the failed pass")
        self.assertLess(paired[0][2], 0.0,
                        "a failed pass was not charged as a loss")
        self.assertEqual(paired[1][3], "NONE",
                         "the second decision was charged with the first "
                         "decision's turnover")
        self.assertEqual(paired[1][2], 0.0)

    def test_no_event_is_claimed_twice(self):
        calls = [([rec(PlayerIntent.SAFE_PASS, order=0),
                   rec(PlayerIntent.CARRY, order=1)],
                  [ev("PASS", outcome=True, location_x=40.0, end_x=75.0,
                      end_y=34.0),
                   ev("CARRY", outcome=True, location_x=40.0, end_x=52.0,
                      end_y=34.0)])]
        paired = SC._correlate_outcomes_causal(calls, grid=GRID)
        self.assertEqual(paired[0][3], "PASS")
        self.assertEqual(paired[1][3], "CARRY")

    def test_a_decision_with_no_action_in_its_chain_is_neutral(self):
        """The chain overrode it.  It must be NEUTRAL, not charged with
        somebody else's event."""
        calls = [([rec(PlayerIntent.SHOOT)],
                  [ev("PASS", player="B", outcome=True, location_x=40.0,
                      end_x=75.0, end_y=34.0)])]
        paired = SC._correlate_outcomes_causal(calls, grid=GRID)
        self.assertEqual(paired[0][3], "NONE")
        self.assertEqual(paired[0][2], 0.0)

    def test_the_join_respects_the_teams_own_attacking_direction(self):
        """An away-side forward pass runs from high raw x to low raw x."""
        calls = [([rec(PlayerIntent.PROGRESSIVE_PASS)],
                  [ev("PROGRESSIVE_PASS", team="Away", outcome=True,
                      minute=10, location_x=65.0, location_y=34.0,
                      end_x=30.0, end_y=34.0)])]
        paired = SC._correlate_outcomes_causal(
            calls, attacks_right_by_team={"Away": False}, grid=GRID)
        self.assertGreater(paired[0][2], 0.0,
                           "an away-side forward pass was scored backwards")


class TheOldJoinIsProvenWrongTests(unittest.TestCase):
    """The load-bearing case: the proximity join charges the NEXT decision
    for the PREVIOUS decision's turnover.  This is the defect the causal
    join exists to remove, and it is pinned so the fix cannot be undone
    silently."""

    def _fixture(self):
        records = [rec(PlayerIntent.SAFE_PASS, order=0),
                   rec(PlayerIntent.PROGRESSIVE_PASS, order=1)]
        timeline = [
            ev("PASS", outcome=False, minute=10, location_x=40.0, end_x=45.0,
               end_y=34.0),
            ev("TURNOVER", outcome=False, minute=10, location_x=45.0,
               end_x=50.0, end_y=34.0),
        ]
        return records, timeline

    def test_the_proximity_join_mis_pairs_the_failed_pass(self):
        records, timeline = self._fixture()
        paired = SC._correlate_outcomes(timeline, records, grid=GRID)
        self.assertEqual(paired[0][3], "PASS")
        # …and the SECOND decision gets the first decision's turnover:
        self.assertEqual(paired[1][3], "TURNOVER",
                         "the proximity join no longer mis-pairs — the "
                         "defect it is kept here to document has changed "
                         "shape and needs re-measuring")
        self.assertLess(paired[1][2], 0.0)

    def test_the_two_joins_disagree_so_the_comparison_is_not_vacuous(self):
        records, timeline = self._fixture()
        prox = SC._correlate_outcomes(timeline, records, grid=GRID)
        causal = SC._correlate_outcomes_causal(
            [([records[0], records[1]], timeline)], grid=GRID)
        self.assertNotEqual([p[3] for p in prox], [c[3] for c in causal],
                            "the two joins agree on the fixture — so it no "
                            "longer distinguishes them and proves nothing")
        self.assertNotEqual([p[2] for p in prox], [c[2] for c in causal])


class ShootDecisionIsAnsweredByItsShotTests(unittest.TestCase):
    """THE THIRD ROUTE TO T2's COLLAPSE, AND THE FIX.

    A policy SHOOT intent makes the possession chain create a ZERO-LENGTH
    marker carry (`location_x=x, end_x=x`).  Its threat change is therefore
    EXACTLY 0.0 — every shoot decision in the corpus was worth precisely
    nothing — while the shot it caused, worth its xG, was fired by the
    ATTACK chain and credited to nobody.

    So the surrogate could not express "shooting is worth it", and the
    genetic algorithm had no reason to learn it: T2's "ST never shoots"
    collapse, reached through the DATA rather than through the objective.

    The fix: a SHOOT decision is answered by the shot it caused — the first
    shot-ish event by the same player, at or after that chain's first event,
    in the SAME MINUTE.  The minute bound is what keeps this from becoming
    the proximity join again.
    """

    def _fixture(self, shot_minute=10):
        marker = ev("CARRY", outcome=True, minute=10,
                    location_x=50.0, location_y=34.0, end_x=50.0, end_y=34.0)
        shot = ev("SHOT_ON_TARGET", player="A", outcome=True,
                  minute=shot_minute, location_x=50.0, location_y=34.0,
                  end_x=95.0, end_y=34.0, xg=0.18)
        return [([rec(PlayerIntent.SHOOT)], [marker])], [marker, shot]

    def test_the_shoot_decision_takes_the_shots_value(self):
        calls, timeline = self._fixture()
        paired = SC._correlate_outcomes_causal(calls, timeline=timeline, grid=GRID)
        self.assertEqual(paired[0][3], "SHOT_ON_TARGET",
                         "the shoot decision was not answered by its shot")
        self.assertGreater(paired[0][2], 0.0,
                            f"the shot scored {paired[0][2]:+.5f}")

    def test_the_marker_carry_would_have_scored_exactly_zero(self):
        """The defect, stated as a measurement: the chain-local join sees a
        zero-length carry, whose threat change is 0.0."""
        calls, timeline = self._fixture()
        marker = timeline[0]
        self.assertEqual(marker.end_x, marker.location_x)
        self.assertEqual(marker.end_y, marker.location_y)
        paired = SC._correlate_outcomes_causal(calls, timeline=None, grid=GRID)
        self.assertEqual(paired[0][2], 0.0,
                         "a zero-length carry did not score zero")

    def test_a_shot_in_another_minute_is_not_claimed(self):
        """The minute bound: without it this would be the proximity join."""
        calls, timeline = self._fixture(shot_minute=14)
        paired = SC._correlate_outcomes_causal(calls, timeline=timeline, grid=GRID)
        self.assertNotEqual(paired[0][3], "SHOT_ON_TARGET",
                            "a shot from a later minute was claimed")
        self.assertEqual(paired[0][2], 0.0)

    def test_another_players_shot_is_not_claimed(self):
        calls, timeline = self._fixture()
        timeline[1].player = "B"
        paired = SC._correlate_outcomes_causal(calls, timeline=timeline, grid=GRID)
        self.assertNotEqual(paired[0][3], "SHOT_ON_TARGET")

    def test_a_non_shot_decision_is_never_answered_by_a_later_shot(self):
        """Only a SHOOT decision looks forward.  A SAFE_PASS still joins to
        its own chain's action."""
        calls, timeline = self._fixture()
        calls[0][0][0].intent = PlayerIntent.SAFE_PASS
        paired = SC._correlate_outcomes_causal(calls, timeline=timeline, grid=GRID)
        self.assertEqual(paired[0][3], "CARRY")


class ProductionWiringTests(unittest.TestCase):
    """The wiring lives in a different file from the join, so no
    behavioural test of the join can see it."""

    def test_the_chain_capture_is_installed_and_restored(self):
        from event_chain import ChainDispatcher
        saved_decide = SC._SAVED_DECIDE
        before = ChainDispatcher.__dict__["possession"]
        try:
            SC._install_collector()
            self.assertIsNot(ChainDispatcher.__dict__["possession"], before,
                             "the chain entry point was not wrapped")
            SC._restore_collector()
            self.assertIs(ChainDispatcher.__dict__["possession"], before,
                          "the chain entry point was not restored")
        finally:
            SC._SAVED_DECIDE = saved_decide
            SC._restore_dispatcher_capture()

    def test_the_wrapper_is_an_observer_and_returns_the_result_unchanged(self):
        ran = []

        def fake_chain(*a, **kw):
            ran.append(1)
            return SimpleNamespace(events=["e1", "e2"])

        SC._COLLECTOR.clear()
        SC._CHAIN_CALLS.clear()
        wrapper = SC._make_chain_wrapper(fake_chain)
        res = wrapper()
        self.assertEqual(len(ran), 1)
        self.assertEqual(res.events, ["e1", "e2"],
                         "the wrapper changed the chain result")
        self.assertEqual(len(SC._CHAIN_CALLS), 1)
        self.assertEqual(SC._CHAIN_CALLS[0][1], ["e1", "e2"])
        SC._CHAIN_CALLS.clear()

    def test_the_chain_buffer_is_cleared_between_matches(self):
        """Otherwise match 2 re-pairs match 1's calls."""
        src = open(SC.__file__, encoding="utf-8").read()
        loop_body = src.split("for m in range(n_matches):", 1)[1]
        self.assertIn("_CHAIN_CALLS.clear()", loop_body,
                      "the chain-call buffer is not cleared per match")


class UnclaimableVocabularyTests(unittest.TestCase):
    """The excluded sets must actually name the events they claim to."""

    def test_the_consequence_set_names_the_real_loss_events(self):
        for name in ("TURNOVER", "MISCONTROL", "DISPOSSESSED", "INTERCEPTION"):
            self.assertIn(name, SC._CONSEQUENCE_EVENTS)
            self.assertIn(name, TR._LOSS_TYPES)

    def test_the_non_action_set_names_the_receipt_and_the_press(self):
        self.assertIn("BALL_RECEIPT", SC._NON_ACTION_EVENTS)
        self.assertIn("PRESS", SC._NON_ACTION_EVENTS)

    def test_no_action_event_is_excluded_by_accident(self):
        """The events the reward treats as the ball actually moving must
        stay claimable, or the join would throw away every pass."""
        for name in TR._MOVER_TYPES:
            self.assertNotIn(name, SC._UNCLAIMABLE_EVENTS,
                             f"{name} is a real action and must be claimable")
        for name in TR._SHOT_ON_TYPES | TR._SHOT_OFF_TYPES | TR._GOAL_TYPES:
            self.assertNotIn(name, SC._UNCLAIMABLE_EVENTS,
                             f"{name} is a real action and must be claimable")


if __name__ == "__main__":
    unittest.main()
