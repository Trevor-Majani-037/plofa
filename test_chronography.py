"""Tests for the additive chronography layer (global-clock event timing).

chronograph() is a pure post-sim pass: it must never mutate the timeline and
must never change simulation outcomes — it only re-derives real match-seconds
for chain events using the chain clock marks recorded during _absorb_chain.
"""
import unittest
from dataclasses import fields, is_dataclass

from match_engine import (
    MatchState, MatchEvent, EventType, SituationType, MatchPhase, GameState,
    ChainClockMark, TimedEvent, MatchChronology, MatchEngine,
)


def _event(minute, etype, player="P", team="Home", second=7):
    return MatchEvent(
        minute=minute, second=second, event_type=etype, team=team, player=player,
        situation=SituationType.OPEN_PLAY, location_x=40.0, location_y=20.0,
        end_x=60.0, end_y=30.0, phase=MatchPhase.OPENING, game_state=GameState.LEVEL,
    )


def _real_engine(state_clock=0.0):
    """Real MatchEngine instance whose chronograph() runs on synthetic data."""
    eng = object.__new__(MatchEngine)
    eng.state = MatchState(match_clock_s=state_clock)
    eng.timeline = []
    eng._chain_clock_marks = []
    return eng


def _two_chain_engine():
    ev1 = _event(5, EventType.PASS, player="A")
    ev2 = _event(5, EventType.PASS, player="B")
    ev3 = _event(6, EventType.GOAL, player="Z")          # goal early-return chain
    ev4 = _event(9, EventType.KICKOFF, player="K")     # standalone (unmarked)
    eng = _real_engine(state_clock=543.0)
    eng.timeline = [ev1, ev2, ev3, ev4]
    eng._chain_clock_marks = [
        ChainClockMark(minute=5, start_clock=300.0, end_clock=305.0,
                       motion_folded=True, events=[ev1, ev2]),
        ChainClockMark(minute=6, start_clock=520.0, end_clock=520.0,
                       motion_folded=False, events=[ev3]),
    ]
    return eng, ev1, ev2, ev3, ev4


class ChronographerLogicTests(unittest.TestCase):
    def test_timestamps_inside_chain_spans(self):
        eng, *_ = _two_chain_engine()
        orig_seconds = [e.second for e in eng.timeline]
        chrono = eng.chronograph()
        e1 = chrono.events[0]                       # sorted → smallest clock first
        self.assertAlmostEqual(e1.match_clock_s, 300.0)
        self.assertLessEqual(e1.match_clock_s, 305.0)
        goal = [te for te in chrono.events if te.source_event_index == 2][0]
        self.assertAlmostEqual(goal.match_clock_s, 520.0)  # collapsed to start
        self.assertTrue(goal.is_goal)
        # Placeholder random seconds must NOT be touched (read-only pass).
        self.assertEqual([e.second for e in eng.timeline], orig_seconds)

    def test_duration_and_sorting(self):
        eng, *_ = _two_chain_engine()
        chrono = eng.chronograph()
        clocks = [te.match_clock_s for te in chrono.events]
        self.assertEqual(clocks, sorted(clocks))
        first = chrono.events[0]
        self.assertAlmostEqual(first.duration, 5.0)  # spacing to the chain's next
        self.assertEqual(chrono.events[-1].duration, 0.0)

    def test_summaries(self):
        chrono = _two_chain_engine()[0].chronograph()
        self.assertEqual(chrono.n_chains, 2)
        self.assertEqual(chrono.n_unmarked_events, 1)        # the kickoff
        self.assertAlmostEqual(chrono.measured_play_s, 5.0)  # only the folded span
        self.assertAlmostEqual(chrono.dead_time_s, 543.0 - 5.0)
        self.assertAlmostEqual(chrono.play_share,
                               100.0 * 5.0 / 543.0, places=1)

    def test_eventless_chain_skipped_without_crash(self):
        ev = _event(5, EventType.PASS)
        eng = _real_engine()
        eng.timeline = [ev]
        eng._chain_clock_marks = [
            ChainClockMark(minute=5, start_clock=10.0, end_clock=12.0,
                           motion_folded=True, events=[ev]),
            ChainClockMark(minute=5, start_clock=12.0, end_clock=15.0,
                           motion_folded=True, events=[]),
        ]
        chrono = eng.chronograph()
        self.assertEqual(len(chrono.events), 1)
        self.assertEqual(chrono.n_chains, 2)
        self.assertEqual(chrono.n_unmarked_events, 0)

    def test_second_clamped_within_minute(self):
        ev = _event(5, EventType.PASS)
        eng = _real_engine()
        eng.timeline = [ev]
        eng._chain_clock_marks = [
            ChainClockMark(minute=5, start_clock=3540.0, end_clock=3601.0,
                           motion_folded=True, events=[ev]),
        ]
        te = eng.chronograph().events[-1]
        self.assertLessEqual(te.second, 59.9)


class ChronographyShapeTests(unittest.TestCase):
    def test_additive_dataclasses(self):
        self.assertTrue(is_dataclass(MatchChronology))
        self.assertTrue(is_dataclass(TimedEvent))
        self.assertTrue(is_dataclass(ChainClockMark))
        td = TimedEvent(minute=1, second=10.0, match_clock_s=70.0, duration=0.0,
                        event_type="PASS", team="A", player="P")
        self.assertEqual(td.stamp, "1:10.0")
        self.assertIn("match_clock_s", [f.name for f in fields(TimedEvent)])

    def test_match_result_carries_chronology_field(self):
        import match_engine
        self.assertIn("chronology", [f.name for f in fields(match_engine.MatchResult)])


if __name__ == "__main__":
    unittest.main()