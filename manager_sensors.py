"""
PLOFA 26/27 — MANAGER SENSORS
====================================================
manager_sensors.py

The 12-float POV sensor vector a manager "sees" mid-match.  Every value comes
from either an EXACT fact (scoreboard, clock, sub counter) or a PERCEIVED
signal from manager_perception.  Nothing reads the engine's internal floats
(sub_controller stamina, momentum totals, threat levels, ...).

senor layout (D1):
    idx  0  score_diff          (-1..1, clamped to ±5 then / 5)
    idx  1  minute_norm         (0..1)
    idx  2  subs_remaining      (0..1)
    idx  3  game_importance     (0..1)
    idx  4  recent_goal_impact  (-1..1)
    idx  5  team_fatigue        (0..1)
    idx  6  team_confidence     (0..1)
    idx  7  momentum            (0..1)
    idx  8  opp_posture         (0..1)
    idx  9  worst_player_state  (0..1)
    idx 10  best_player_state   (0..1)
    idx 11  key_player_avail    (0..1)

Not wired yet — the brain consumes this in Phase 2.
"""

from __future__ import annotations
from typing import Any, List

import numpy as np

from manager_perception import (
    perceive_best_player_state,
    perceive_game_importance,
    perceive_key_player_available,
    perceive_momentum,
    perceive_opp_posture,
    perceive_player_state,
    perceive_recent_goal_impact,
    perceive_team_confidence,
    perceive_team_fatigue,
)

MANAGER_SENSOR_LAYOUT = {
    0: "score_diff",
    1: "minute_norm",
    2: "subs_remaining",
    3: "game_importance",
    4: "recent_goal_impact",
    5: "team_fatigue",
    6: "team_confidence",
    7: "momentum",
    8: "opp_posture",
    9: "worst_player_state",
    10: "best_player_state",
    11: "key_player_avail",
}
assert len(MANAGER_SENSOR_LAYOUT) == 12


def _worst_player_state(engine: Any, team: str) -> float:
    """max over active players of perceive_player_state (worst on the pitch)."""
    active = engine.active_players.get(team, [])
    if not active:
        return 0.0
    return max(perceive_player_state(engine, p) for p in active)


def extract_manager_sensors(engine: Any, team: str) -> np.ndarray:
    """
    Build the 12-float POV sensor vector for a manager.
    Every value comes from either an EXACT fact or a PERCEIVED signal.
    """
    opponent = engine.config.away_team if team == engine.config.home_team \
        else engine.config.home_team
    is_home = (team == engine.config.home_team)

    score_diff = (engine.state.home_goals - engine.state.away_goals) \
                 if is_home else \
                 (engine.state.away_goals - engine.state.home_goals)
    score_diff = max(-5, min(5, score_diff)) / 5.0

    minute_norm = min(1.0, engine.state.minute / 90.0)

    subs_made = engine.state.home_subs_made if is_home \
        else engine.state.away_subs_made
    subs_remaining = max(0, 3 - subs_made) / 3.0

    game_importance = perceive_game_importance(engine)

    recent_goal_impact = perceive_recent_goal_impact(engine, team)

    team_fatigue = perceive_team_fatigue(engine, team)
    team_conf = perceive_team_confidence(engine, team)
    momentum = perceive_momentum(engine, team)
    opp_posture = perceive_opp_posture(engine, opponent)

    worst = _worst_player_state(engine, team)
    best = perceive_best_player_state(engine, team)
    key_avail = perceive_key_player_available(engine, team)

    sensors = np.array([
        score_diff, minute_norm, subs_remaining, game_importance,
        recent_goal_impact,
        team_fatigue, team_conf, momentum, opp_posture,
        worst, best, key_avail,
    ], dtype=np.float64)

    assert sensors.shape == (12,)
    return sensors