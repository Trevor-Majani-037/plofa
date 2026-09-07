"""Tests for Opta-style shot speed tracking.

Every shot-producing chain stamps ``metadata["shot_speed_kmh"]`` (+ m/s and
flight time) at the event top level; ``MatchResult.shot_speed_stats()`` then
aggregates average/max shot velocity per team. The stamps are strictly
additive — no RNG, no ordering effects — so simulation outcomes are
bit-identical with or without them (verified under a fixed PYTHONHASHSEED).
"""
import math
import types
import unittest

from match_engine import (
    MatchConfig, MatchState, MatchResult, MatchEvent,
    EventType, SituationType, MatchPhase, GameState,
)
from threat_engine import ThreatEngine
from possession_physics import shot_speed, header_shot_speed
from event_chain import _shot_speed_for_player


def _drag_player(strength, finishing):
    return types.SimpleNamespace(dna=types.SimpleNamespace(
        physical=types.SimpleNamespace(strength=float(strength)),
        technical=types.SimpleNamespace(finishing=float(finishing)),
    ))


def _ev(etype, team, kmh=None, nested=False):
    md = {}
    if kmh is not None:
        md = ({"physics": {"shot_speed_kmh": kmh}} if nested
              else {"shot_speed_kmh": kmh})
    return MatchEvent(
        minute=1, second=0, event_type=etype, team=team, player="P",
        situation=SituationType.OPEN_PLAY, phase=MatchPhase.OPENING,
        game_state=GameState.LEVEL, metadata=md,
    )


class VelocityHelperTests(unittest.TestCase):
    def test_shot_speed_range(self):
        lo, hi = shot_speed(0.0), shot_speed(100.0)
        self.assertEqual(lo, 20.0)                 # 72 km/h
        self.assertAlmostEqual(hi, 34.0)            # 122.4 km/h
        self.assertGreater(shot_speed(70), lo)

    def test_header_speed_slower_than_foot(self):
        self.assertEqual(header_shot_speed(0.0), 5.0)
        self.assertAlmostEqual(header_shot_speed(100.0), 12.0)
        self.assertLess(header_shot_speed(100.0), shot_speed(0.0))

    def test_player_speed_for_body_part(self):
        tank = _drag_player(100, 100)
        weak = _drag_player(0, 0)
        self.assertGreater(_shot_speed_for_player(tank, "foot"),
                           _shot_speed_for_player(weak, "foot"))
        # A header from the same player is far slower than a struck ball.
        self.assertLess(_shot_speed_for_player(tank, "head"),
                        _shot_speed_for_player(tank, "foot"))
        # Namespace-less fallback stays in the safe range.
        spd = _shot_speed_for_player(types.SimpleNamespace(dna=None), "foot")
        self.assertGreaterEqual(spd, 20.0)


def _result(timeline):
    cfg = MatchConfig(home_team="Home", away_team="Away", matchday=1,
                      referee="R", referee_strictness=0.0)
    return MatchResult(
        config=cfg, state=MatchState(), timeline=list(timeline),
        goals=[], cards=[], subs=[], squads={},
        threat=ThreatEngine("Home", "Away"),
    )


class ShotSpeedStatsTests(unittest.TestCase):
    def test_synthetic_aggregation(self):
        tl = [
            _ev(EventType.GOAL, "Home", kmh=130),            # excluded (paired w/ SOT)
            _ev(EventType.SHOT_ON_TARGET, "Home", kmh=110),
            _ev(EventType.SHOT_ON_TARGET, "Home", kmh=120),
            _ev(EventType.SHOT_ON_TARGET, "Home", kmh=130),
            _ev(EventType.SHOT_OFF_TARGET, "Home"),          # unstamped shot
            _ev(EventType.SHOT_BLOCKED, "Home", kmh=140, nested=True),
            _ev(EventType.SHOT_ON_TARGET, "Away", kmh=95),
            _ev(EventType.SHOT_ON_TARGET, "Away", kmh=105),
            _ev(EventType.PENALTY_SCORED, "Away", kmh=90),
        ]
        s = _result(tl).shot_speed_stats()

        home = s["Home"]
        self.assertEqual(home["shots"], 4)
        self.assertEqual(home["avg_kmh"], 125.0)
        self.assertEqual(home["max_kmh"], 140.0)

        away = s["Away"]
        self.assertEqual(away["shots"], 3)
        self.assertAlmostEqual(away["avg_kmh"], (95 + 105 + 90) / 3, places=1)
        self.assertEqual(away["max_kmh"], 105.0)

        # GOAL excluded → match average from 7 attempt speeds only.
        self.assertAlmostEqual(
            s["match_avg_kmh"], (110 + 120 + 130 + 140 + 95 + 105 + 90) / 7, delta=0.1)
        self.assertEqual(s["shots_total"], 8)
        self.assertEqual(s["shots_with_speed"], 7)
        self.assertAlmostEqual(s["coverage_pct"], 87.5)

    def test_empty_no_shots(self):
        s = _result([]).shot_speed_stats()
        self.assertIsNone(s["match_avg_kmh"])
        self.assertEqual(s["shots_total"], 0)
        self.assertEqual(s["coverage_pct"], 0.0)

    def test_nested_physics_fallback(self):
        # Speed stored under the legacy nested physics dict still counts.
        tl = [_ev(EventType.SHOT_ON_TARGET, "Home", kmh=99, nested=True)]
        self.assertEqual(_result(tl).shot_speed_stats()["Home"]["avg_kmh"], 99.0)


if __name__ == "__main__":
    unittest.main()