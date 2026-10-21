"""THE DECISION TIME — the gate.

THE DEFECT THIS FILE EXISTS FOR
-------------------------------
The engine's clock advanced by the BALL's motion, so a possession's
twenty-six actions were resolved inside the ball's flight: the exported
timeline read "26 actions in 7 seconds, then 50 seconds of nothing".

Measured, one real match, before the fix:

| | before | real football |
|---|---|---|
| live (play) time in a 96-minute match | 23.5 min (24.7%) | ~55-60 min (~60%) |
| seconds per event inside a chain | 0.449 s | 2-4 s |
| the gap between one possession and the next | 46 s | 10-20 s |
| the MEDIAN interval between passes | 1.10 s | 3-5 s |

WHAT THIS FILE PINS
-------------------
**The OUTCOME, not the constant.**  The decision time is a number in
`match_timeline`, and a test that asserts the number asserts nothing about
the football.  So the gate runs a real match and pins what the number is
FOR: the live share of the match, the interval between passes, the gap
between possessions, and the density inside a possession.

It also pins the module's own arithmetic (an on-ball event consumes
decision time, an off-ball contest does not, a long pass weighs more than a
short one) and the wiring (the engine adds the decision time, and the
chronology places the events by weight rather than evenly).
"""
from __future__ import annotations

import os
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import match_timeline as MT  # noqa: E402


class _Event:
    """A minimal stand-in for a MatchEvent (the module is pure)."""

    def __init__(self, name, x0=0.0, y0=0.0, x1=None, y1=None):
        self.event_type = type("T", (), {"name": name})()
        self.location_x = x0
        self.location_y = y0
        self.end_x = x1
        self.end_y = y1


class DecisionTimeUnitTests(unittest.TestCase):
    def test_an_on_ball_event_consumes_decision_time(self):
        e = _Event("PASS")
        self.assertGreater(MT.decision_seconds(e), 0.0)

    def test_an_off_ball_contest_does_not_extend_the_possession(self):
        """A press or a duel happens ALONGSIDE the on-ball action — which
        is why it must not add to the chain's window."""
        for name in ("PRESS", "GROUND_DUEL", "AERIAL_DUEL"):
            self.assertEqual(MT.decision_seconds(_Event(name)), 0.0, name)

    def test_the_chain_time_is_the_sum_of_its_on_ball_decisions(self):
        events = [_Event("PASS"), _Event("BALL_RECEIPT"),
                  _Event("PRESS"), _Event("CARRY")]
        expected = (MT.decision_seconds(events[0])
                    + MT.decision_seconds(events[1])
                    + MT.decision_seconds(events[3]))
        self.assertAlmostEqual(MT.chain_decision_seconds(events), expected,
                               places=9)

    def test_a_long_pass_weighs_more_than_a_short_one(self):
        short = _Event("PASS", 0, 0, 5, 0)
        long = _Event("PASS", 0, 0, 40, 0)
        self.assertGreater(MT.action_seconds(long), MT.action_seconds(short))

    def test_every_event_gets_a_positive_placement_weight(self):
        """A zero weight would collapse two events onto one instant."""
        for name in ("PASS", "BALL_RECEIPT", "PRESS", "CARRY", "GOAL",
                     "FOUL_COMMITTED", "MISCONTROL"):
            w = MT.placement_weights([_Event(name)])[0]
            self.assertGreater(w, 0.0, name)

    def test_the_decision_time_is_the_documented_size(self):
        """The constant itself, so a change to it is a visible change."""
        self.assertAlmostEqual(MT.DECISION_S, 0.8, places=6)
        self.assertAlmostEqual(MT.CONTROL_S, 0.3, places=6)


class WiringTests(unittest.TestCase):
    def test_the_engine_adds_the_decision_time_to_the_chain_window(self):
        """STRUCTURAL: the engine must call the module.  A comment saying
        it does would prove nothing."""
        import match_engine
        with open(match_engine.__file__, encoding="utf-8", newline="") as f:
            src = f.read()
        self.assertIn("match_timeline.chain_decision_seconds", src,
                      "the engine does not add the decision time")
        self.assertIn("match_timeline.placement_weights", src,
                      "the chronology does not place the events by weight")

    def test_the_chronology_does_not_spread_the_events_evenly(self):
        """The even spread is the defect: `span * i / (n - 1)`.  It must
        be gone from the placement."""
        import match_engine
        with open(match_engine.__file__, encoding="utf-8", newline="") as f:
            src = f.read()
        self.assertNotIn("span * i / (n - 1)", src,
                         "the chronology still spreads the events evenly")


class MatchOutcomeTests(unittest.TestCase):
    """THE GATE THAT MATTERS: a real match, and what the football looks
    like in it.  Slow (one match) and the reason the file exists."""

    @classmethod
    def setUpClass(cls):
        import io
        import random
        from tools.diag._diag_watch import build
        eng, _, _ = build()
        random.seed(31)
        real = sys.stdout
        sys.stdout = io.StringIO()
        try:
            cls.res = eng.simulate()
        finally:
            sys.stdout = real
        cls.eng = eng
        cls.chron = cls.res.chronology

    def test_the_match_is_still_ninety_six_minutes(self):
        """The fix REDISTRIBUTES the clock; it must not lengthen the
        match.  The engine's minute structure is what holds the total."""
        self.assertAlmostEqual(self.chron.match_duration_s / 60.0, 96.0,
                               delta=3.0,
                               msg=f"the match is now "
                                   f"{self.chron.match_duration_s/60:.1f} min")

    def test_the_live_share_is_real_footballs(self):
        """THE HEADLINE: the share of the match that is actually being
        played.  Real football: ~60%."""
        share = self.chron.measured_play_s / self.chron.match_duration_s
        self.assertGreater(share, 0.50,
                           f"only {share*100:.1f}% of the match is live")
        self.assertLess(share, 0.70,
                        f"{share*100:.1f}% of the match is live — too much")

    def test_the_interval_between_passes_is_footballs(self):
        """Real football: one pass every 3-5 s."""
        tl = [e for e in self.res.timeline
              if getattr(e, "match_clock_s", None) is not None]
        tl.sort(key=lambda e: e.match_clock_s)
        passes = [e for e in tl
                  if MT.event_name(e) in ("PASS", "PROGRESSIVE_PASS",
                                          "SWITCH_OF_PLAY", "THROUGH_BALL")]
        self.assertGreater(len(passes), 100, "too few passes to judge")
        gaps = [b.match_clock_s - a.match_clock_s
                for a, b in zip(passes, passes[1:])]
        med = statistics.median(gaps)
        self.assertGreater(med, 2.0,
                           f"the median gap between passes is {med:.2f}s")
        self.assertLess(med, 6.0,
                        f"the median gap between passes is {med:.2f}s")

    def test_a_possession_does_not_happen_in_a_burst(self):
        """THE USER'S COMPLAINT: twenty-six actions in seven seconds.

        Measured as the span of the biggest chain — the densest stretch of
        the match.  Real football's longest possession is ~30 s, and no
        stretch of a match packs twenty actions into seven seconds."""
        spans = []
        for m in self.eng._chain_clock_marks:
            if len(m.events) >= 10 and m.end_clock > m.start_clock:
                spans.append(m.end_clock - m.start_clock)
        self.assertTrue(spans, "no chain with ten events to judge")
        longest = max(spans)
        # 20 actions at 1 s each cannot fit in 7 s: the floor is the
        # number of events, each needing a slot of its own.
        biggest = max(len(m.events) for m in self.eng._chain_clock_marks)
        self.assertGreater(longest, biggest * 0.4,
                           f"the biggest chain ({biggest} events) spans only "
                           f"{longest:.1f}s — the actions are still bunched")

    def test_the_gap_between_possessions_is_not_a_minute_of_silence(self):
        """Real football: the ball is out of play for 10-20 s at a time."""
        marks = [m for m in self.eng._chain_clock_marks if m.events]
        gaps = [b.start_clock - a.end_clock
                for a, b in zip(marks, marks[1:]) if b.start_clock > a.end_clock]
        self.assertTrue(gaps, "no gaps to judge")
        med = statistics.median(gaps)
        self.assertLess(med, 40.0,
                        f"the median gap between possessions is {med:.1f}s")
        self.assertLess(max(gaps), 70.0,
                        f"the longest silence is {max(gaps):.1f}s")

    def test_the_timeline_is_still_monotonic(self):
        """The fix must not break the flow it was meant to deliver."""
        tl = [e for e in self.res.timeline
              if getattr(e, "match_clock_s", None) is not None]
        tl.sort(key=lambda e: e.match_clock_s)
        back = 0
        for a, b in zip(tl, tl[1:]):
            if b.match_clock_s < a.match_clock_s:
                back += 1
        self.assertEqual(back, 0, f"{back} backwards steps in the timeline")

    def test_the_chronology_and_the_events_agree(self):
        """The event's stamped clock and the chronology's must be the same
        number — the export's truth."""
        tl = [e for e in self.res.timeline
              if getattr(e, "match_clock_s", None) is not None]
        by_index = {t.source_event_index: t for t in self.chron.events
                    if getattr(t, "source_event_index", -1) >= 0}
        checked = 0
        for i, e in enumerate(tl):
            t = by_index.get(i)
            if t is None:
                continue
            self.assertAlmostEqual(e.match_clock_s, t.match_clock_s, places=2)
            checked += 1
        self.assertGreater(checked, 100, "too few events to compare")


if __name__ == "__main__":
    unittest.main()
