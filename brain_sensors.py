"""Vision sensor extraction — converts match state into a 24-float vector.

This module replaces the implicit "perception" step of DecisionBrain with
an explicit sensor layer.  The geometry helpers mirror those in
decision_brain.py but are self-contained (no circular imports with
event_chain.py).

Sensor vector layout (24 floats, all normalised to ~0..1):
    [ 0] ball_x                (0..1, normalised by pitch length 105)
    [ 1] ball_y                (0..1, normalised by pitch width 68)
    [ 2] player_x              (0..1)
    [ 3] player_y              (0..1)
    [ 4] nearest_defender_dist  (0..1, normalised by 15 m)
    [ 5] defenders_within_5m   (0..1, count / 5)
    [ 6] defenders_within_10m  (0..1, count / 5)
    [ 7] best_forward_dist     (0..1, normalised by 40 m)
    [ 8] best_forward_openness (0..1)
    [ 9] space_ahead           (0..1)
    [10] is_in_crossing_zone   (0 or 1)
    [11] is_near_goal          (0 or 1)
    [12] is_final_third        (0 or 1)
    [13] is_own_half           (0 or 1)
    [14] goal_distance_norm    (0..1, normalised by max diagonal ~70 m)
    [15] central_lane          (0 or 1)
    [16] under_pressure        (0 or 1)
    [17] fatigue               (0..1)
    [18] score_diff            (-1..1, clamped, goals / 5)
    [19] minute_norm           (0..1)
    [20] team_possession       (0 or 1)
    [21] player_vision         (0..1, DNA attribute / 100)
    [22] player_composure      (0..1, DNA attribute / 100)
    [23] player_decisions      (0..1, DNA attribute / 100)
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from football_brain import INPUT_SIZE


# ─────────────────────────────────────────────────────────────
# GEOMETRY HELPERS (self-contained, no circular imports)
# ─────────────────────────────────────────────────────────────

def _pos(position_engine: Any, name: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if position_engine is None:
        return fallback
    try:
        return position_engine.get_position(name)
    except Exception:
        return fallback


def _nearest_defender_dist(
    x: float, y: float, defenders: List[Any], position_engine: Any,
) -> Optional[float]:
    best = None
    for d in defenders or []:
        if getattr(d, "position", "") == "GK":
            continue
        dx, dy = _pos(position_engine, d.name, (x + 10, y))
        dist = math.hypot(dx - x, dy - y)
        if best is None or dist < best:
            best = dist
    return best


def _defenders_within(
    x: float, y: float, defenders: List[Any], position_engine: Any, radius: float,
) -> int:
    count = 0
    for d in defenders or []:
        if getattr(d, "position", "") == "GK":
            continue
        dx, dy = _pos(position_engine, d.name, (x + 999, y))
        if math.hypot(dx - x, dy - y) <= radius:
            count += 1
    return count


def _teammate_openness(
    tx: float, ty: float, defenders: List[Any], position_engine: Any,
) -> float:
    best = None
    for d in defenders or []:
        if getattr(d, "position", "") == "GK":
            continue
        dx, dy = _pos(position_engine, d.name, (tx + 10, ty))
        dist = math.hypot(dx - tx, dy - ty)
        if best is None or dist < best:
            best = dist
    if best is None:
        return 1.0
    return max(0.0, min(1.0, (best - 1.5) / 8.5))


def _best_forward_teammate(
    x: float, y: float, teammates: List[Any], defenders: List[Any],
    position_engine: Any, attacks_right: bool,
) -> Optional[Tuple[float, float]]:
    """Returns (distance_ahead, openness) of the best forward teammate."""
    best_val = -1.0
    best_result = None
    for t in teammates or []:
        if getattr(t, "position", "") == "GK":
            continue
        tx, ty = _pos(position_engine, t.name, (x, y))
        progress = (tx - x) if attacks_right else (x - tx)
        if progress < 4.0:
            continue
        openness = _teammate_openness(tx, ty, defenders, position_engine)
        value = min(progress / 35.0, 1.0) * 0.55 + openness * 0.45
        if value > best_val:
            best_val = value
            best_result = (progress, openness)
    return best_result


def _goal_distance(x: float, y: float, attacks_right: bool) -> float:
    gx = 105.0 if attacks_right else 0.0
    return math.hypot(gx - x, 34.0 - y)


def _fatigue_estimate(player: Any, minute: float) -> float:
    dna = getattr(player, "dna", None)
    stamina_attr = _get_attr(dna, "physical.stamina", 65.0) / 100.0
    match_frac = max(0.0, min(1.15, minute / 90.0))
    live = match_frac * (1.35 - stamina_attr)
    carry_over = _get_attr(dna, "form.fatigue_level", 0.0) / 100.0
    return max(0.0, min(1.0, live * 0.8 + carry_over * 0.25))


def _get_attr(obj: Any, path: str, default: float) -> float:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return float(cur) if cur is not None else default


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────
# SENSOR EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_sensors(
    player: Any,
    x: float,
    y: float,
    teammates: List[Any],
    defenders: List[Any],
    position_engine: Any,
    under_pressure: bool,
    attacks_right: bool,
    game_state: Any,
    minute: float = 45.0,
    team_possession: bool = True,
    score_diff: int = 0,
) -> np.ndarray:
    """Build the 24-float vision vector for the neural brain.

    Parameters match DecisionBrain.decide() so this can be called from
    the same call site in event_chain.py with zero extra plumbing.
    """
    dna = getattr(player, "dna", None)
    mental = getattr(dna, "mental", None)

    # ── Spatial sensors ────────────────────────────────────────
    ball_x = _clamp(x / 105.0)
    ball_y = _clamp(y / 68.0)
    player_x = ball_x
    player_y = ball_y

    near_def = _nearest_defender_dist(x, y, defenders, position_engine)
    near_def_norm = _clamp(near_def / 15.0) if near_def is not None else 1.0

    def_5 = _clamp(_defenders_within(x, y, defenders, position_engine, 5.0) / 5.0)
    def_10 = _clamp(_defenders_within(x, y, defenders, position_engine, 10.0) / 5.0)

    fwd = _best_forward_teammate(x, y, teammates, defenders, position_engine, attacks_right)
    if fwd is not None:
        fwd_dist = _clamp(fwd[0] / 40.0)
        fwd_open = fwd[1]
    else:
        fwd_dist = 0.0
        fwd_open = 0.0

    space = 1.0 if near_def is None else _clamp((near_def - 1.5) / 8.5)

    position = getattr(player, "position", "")
    is_crossing = 1.0 if (position in ("LW", "RW", "LB", "RB") and
                          ((x > 80.0) if attacks_right else (x < 25.0))) else 0.0
    is_near_goal = 1.0 if _goal_distance(x, y, attacks_right) < 25.0 else 0.0
    is_final_third = 1.0 if ((x > 70.0) if attacks_right else (x < 35.0)) else 0.0
    is_own_half = 1.0 if ((x < 52.5) if attacks_right else (x > 52.5)) else 0.0
    goal_dist_norm = _clamp(_goal_distance(x, y, attacks_right) / 70.0)
    central = 1.0 if abs(y - 34.0) < 18.0 else 0.0

    # ── Match-state sensors ────────────────────────────────────
    fatigue = _fatigue_estimate(player, minute)
    sd = _clamp(score_diff / 5.0, -1.0, 1.0)
    minute_n = _clamp(minute / 90.0)
    poss = 1.0 if team_possession else 0.0

    # ── DNA sensors ────────────────────────────────────────────
    vision = _clamp(_get_attr(mental, "vision", 55.0) / 100.0)
    composure = _clamp(_get_attr(mental, "composure", 55.0) / 100.0)
    decisions = _clamp(_get_attr(mental, "decisions", 55.0) / 100.0)

    sensors = np.array([
        ball_x, ball_y, player_x, player_y,
        near_def_norm, def_5, def_10,
        fwd_dist, fwd_open, space,
        is_crossing, is_near_goal, is_final_third, is_own_half,
        goal_dist_norm, central,
        1.0 if under_pressure else 0.0,
        fatigue, sd, minute_n, poss,
        vision, composure, decisions,
    ], dtype=np.float64)

    assert sensors.shape == (INPUT_SIZE,), (
        f"Sensor vector shape mismatch: {sensors.shape} != ({INPUT_SIZE},)"
    )
    return sensors


# ─────────────────────────────────────────────────────────────
# OFF-BALL SENSOR EXTRACTION (off-ball "conscience" input)
# ─────────────────────────────────────────────────────────────

def extract_offball_sensors(
    player: Any,
    runner_x: float,
    runner_y: float,
    ball_x: float,
    ball_y: float,
    teammates: List[Any],
    defenders: List[Any],
    position_engine: Any,
    attacks_right: bool,
    game_state: Any,
    minute: float = 45.0,
    score_diff: int = 0,
) -> np.ndarray:
    """Build the 24-float vision vector for a player NOT on the ball.

    Layout is IDENTICAL to ``extract_sensors`` (slots 0..23) so the same
    24->32->32->10 network architecture can be reused for the off-ball
    "conscience" — only the semantics differ:

        [ 0] ball_x             (real ball position, 0..1)
        [ 1] ball_y             (real ball position, 0..1)
        [ 2] runner_x           (this player, 0..1)
        [ 3] runner_y           (this player, 0..1)
        [ 4] nearest_defender_dist  (defenders measured around the RUNNER)
        [ 5] defenders_within_5m
        [ 6] defenders_within_10m
        [ 7] best_forward_dist  (best teammate AHEAD of the runner)
        [ 8] best_forward_openness
        [ 9] space_ahead
        [10] in_crossing_zone   (runner's corridor flag, wide roles only)
        [11] runner_near_goal
        [12] runner_final_third
        [13] runner_own_half
        [14] runner_goal_dist_norm
        [15] runner_central
        [16] runner_under_pressure
        [17] fatigue            (runner)
        [18] score_diff
        [19] minute_norm
        [20] team_possession    (=1: off-ball conscience is in-possession runs)
        [21] player_vision
        [22] player_composure
        [23] player_decisions

    Because ball and runner are decoupled, the vector implicitly carries
    the runner-to-ball offset (slots 0-1 vs 2-3) that the on-ball net never
    needed — that's the core new signal a conscience must learn.
    """
    dna = getattr(player, "dna", None)
    mental = getattr(dna, "mental", None)

    ball_xn = _clamp(ball_x / 105.0)
    ball_yn = _clamp(ball_y / 68.0)
    runner_xn = _clamp(runner_x / 105.0)
    runner_yn = _clamp(runner_y / 68.0)

    near_def = _nearest_defender_dist(runner_x, runner_y, defenders, position_engine)
    near_def_norm = _clamp(near_def / 15.0) if near_def is not None else 1.0

    def_5 = _clamp(_defenders_within(runner_x, runner_y, defenders, position_engine, 5.0) / 5.0)
    def_10 = _clamp(_defenders_within(runner_x, runner_y, defenders, position_engine, 10.0) / 5.0)

    fwd = _best_forward_teammate(
        runner_x, runner_y, teammates, defenders, position_engine, attacks_right)
    if fwd is not None:
        fwd_dist = _clamp(fwd[0] / 40.0)
        fwd_open = fwd[1]
    else:
        fwd_dist = 0.0
        fwd_open = 0.0

    space = 1.0 if near_def is None else _clamp((near_def - 1.5) / 8.5)

    position = getattr(player, "position", "")
    is_crossing = 1.0 if (position in ("LW", "RW", "LB", "RB") and
                          ((ball_x > 70.0) if attacks_right else (ball_x < 35.0))) else 0.0
    is_near_goal = 1.0 if _goal_distance(runner_x, runner_y, attacks_right) < 25.0 else 0.0
    is_final_third = 1.0 if ((runner_x > 70.0) if attacks_right else (runner_x < 35.0)) else 0.0
    is_own_half = 1.0 if ((runner_x < 52.5) if attacks_right else (runner_x > 52.5)) else 0.0
    goal_dist_norm = _clamp(_goal_distance(runner_x, runner_y, attacks_right) / 70.0)
    central = 1.0 if abs(runner_y - 34.0) < 18.0 else 0.0

    fatigue = _fatigue_estimate(player, minute)
    sd = _clamp(score_diff / 5.0, -1.0, 1.0)
    minute_n = _clamp(minute / 90.0)

    vision = _clamp(_get_attr(mental, "vision", 55.0) / 100.0)
    composure = _clamp(_get_attr(mental, "composure", 55.0) / 100.0)
    decisions = _clamp(_get_attr(mental, "decisions", 55.0) / 100.0)

    sensors = np.array([
        ball_xn, ball_yn, runner_xn, runner_yn,
        near_def_norm, def_5, def_10,
        fwd_dist, fwd_open, space,
        is_crossing, is_near_goal, is_final_third, is_own_half,
        goal_dist_norm, central,
        1.0 if near_def is not None and near_def < 4.0 else 0.0,
        fatigue, sd, minute_n, 1.0,
        vision, composure, decisions,
    ], dtype=np.float64)

    assert sensors.shape == (INPUT_SIZE,), (
        f"Off-ball sensor vector shape mismatch: {sensors.shape} != ({INPUT_SIZE},)"
    )
    return sensors
