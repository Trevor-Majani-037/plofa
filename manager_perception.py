"""
PLOFA 26/27 — MANAGER PERCEPTION
====================================================
manager_perception.py

The manager's "eyes and ears."  Every function here converts INTERNAL engine
state into what a real touchline manager would PERCEIVE — coarse 3-level
categorical signals (0.0 / 0.5 / 1.0), NOT exact internals.

Design contract:
  * Every signal is derived from data the manager could actually observe:
    body positions (position_log / position_engine.states), the ball path
    (match_ball_path), the event timeline (timeline), the scoreboard and the
    match config.  NOT from internal knobs like MomentumEngine.momentum or
    team_stances.
  * All windows have a PERCEPTION DELAY: a rolling window ends at
    (current_minute - 2), so an event today is not seen by the manager until
    two minutes later — real touchline perception lags.
  * Thresholds are relative to the team's OWN start-of-match baseline
    (the "fresh legs" reference), so categorical levels adapt to the team.
  * Nothing is wired into the engine yet — these are pure functions.

Nothing here mutates state and nothing consumes RNG.
"""

from __future__ import annotations
import math
from typing import Any, List

from match_engine import MatchEngine

# ── Perception window helpers ───────────────────────────────────────

PERCEPTION_DELAY_MIN = 2  # rolling window ends at current_minute - 2


def _window_end_minute(engine: MatchEngine) -> int:
    """The NEWEST minute a manager can currently perceive (= now - delay)."""
    return max(0, int(getattr(engine.state, "minute", 0)) - PERCEPTION_DELAY_MIN)


def _current_minute(engine: MatchEngine) -> int:
    return int(getattr(engine.state, "minute", 0))


def _window_frames(engine: MatchEngine, team: str,
                   window_s: float) -> List[dict]:
    """position_log frames for *team* within [end - window, end] where end is
    the perception-delayed minute.  Each frame row -> {player, position, x, y,
    distance_total, physics_distance_m, physics_sprint_count, ...}."""
    end = _window_end_minute(engine)
    win_min = max(1, int(window_s / 60.0))
    start = max(1, end - win_min + 1)
    side = "home" if team == engine.config.home_team else "away"
    out: List[dict] = []
    for frame in engine.position_log:
        m = int(frame.get("minute", 0))
        if start <= m <= end:
            out.append(frame)
    return out


def _team_rows(engine: MatchEngine, frames: List[dict], team: str) -> List[dict]:
    side = "home" if team == engine.config.home_team else "away"
    rows: List[dict] = []
    for frame in frames:
        rows.extend(frame.get(side, []) or [])
    return rows


def _baseline_rows(engine: MatchEngine, team: str) -> List[dict]:
    """The first ~3 minutes of the match = the team's fresh-legs baseline."""
    side = "home" if team == engine.config.home_team else "away"
    rows: List[dict] = []
    for frame in engine.position_log:
        if int(frame.get("minute", 0)) <= 3:
            rows.extend(frame.get(side, []) or [])
    return rows


def _quantize3(value: float, base: float) -> float:
    """Map a signal relative to a neutral midpoint onto 0.0 / 0.5 / 1.0."""
    if value >= base + 0.15:
        return 1.0
    if value <= base - 0.15:
        return 0.0
    return 0.5


# ── Event taxonomy (what a manager on the touchline actually counts) ──

_SUCCESS_EVENTS = frozenset({
    "PASS", "PROGRESSIVE_PASS", "THROUGH_BALL", "SWITCH_OF_PLAY",
    "CROSS_SUCCESS", "DRIBBLE_SUCCESS", "TACKLE_WON", "INTERCEPTION",
    "CLEARANCE", "RECOVERY", "BALL_RECOVERY", "PRESS_SUCCESS",
    "CHANCE_CREATED", "BIG_CHANCE_CREATED", "SHOT_ON_TARGET", "GOAL",
    "PENALTY_SCORED", "CORNER_WON", "FREEKICK_WON", "AERIAL_DUEL",
})

_FAILURE_EVENTS = frozenset({
    "TURNOVER", "TACKLE_LOST", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
    "DRIBBLE_FAIL", "MISCONTROL", "DISPOSSESSED", "OFFSIDE",
    "FOUL_COMMITTED", "PENALTY_MISSED", "HIT_WOODWORK", "SAVE_AGAINST",
})

_ATTACK_ROLES = frozenset({"ST", "CF", "LW", "RW", "CAM"})


def _team_event_counts(engine: MatchEngine, team: str,
                       end: int, minutes: int) -> tuple:
    """(successes, failures) in the [end - minutes + 1, end] minute window."""
    succ = 0
    fail = 0
    start = max(0, end - minutes + 1)
    for ev in engine.timeline:
        m = int(getattr(ev, "minute", 0))
        if m < start or m > end:
            continue
        if getattr(ev, "team", None) != team:
            continue
        name = getattr(ev.event_type, "name", "")
        if name in _SUCCESS_EVENTS:
            succ += 1
        elif name in _FAILURE_EVENTS:
            fail += 1
    return succ, fail


# ═════════════════════════════════════════════════════════════════════
# PERCEPTION PRIMITIVES
# ═════════════════════════════════════════════════════════════════════

def perceive_team_fatigue(engine: MatchEngine, team: str,
                          window_s: float = 300.0) -> float:
    """
    Return 0.0 (FRESH) / 0.5 (TIRING) / 1.0 (SPENT).

    Derived from:
      - the team's recent movement distance per minute (from position_log)
      - recent sprint events per minute
    Delayed by 2 minutes (rolling window ends at current_minute - 2).
    Quantized against the team's own start-of-match (fresh-legs) baseline.
    """
    frames = _window_frames(engine, team, window_s)
    rows = _team_rows(engine, frames, team)
    base_rows = _baseline_rows(engine, team)

    def _per_min(row_gen: List[dict]) -> tuple:
        dist = 0.0
        sprint = 0.0
        n = 0
        for r in row_gen:
            dist += float(r.get("distance_total", 0.0))
            sprint += float(r.get("physics_sprint_count", 0.0))
            n += 1
        return (dist / n if n else 0.0, sprint / n if n else 0.0)

    cur_dist, cur_sprint = _per_min(rows)
    base_dist, base_sprint = _per_min(base_rows)

    # Fresh legs = the "full tank" reference.  Distance per minute below that
    # baseline, for the SAME body effort, is the fatigue signal.
    dist_ratio = cur_dist / base_dist if base_dist > 0 else 1.0
    sprint_ratio = cur_sprint / base_sprint if base_sprint > 0 else 1.0

    effort = 0.75 * dist_ratio + 0.25 * sprint_ratio
    if effort >= 0.80:
        return 0.0
    if effort >= 0.55:
        return 0.5
    return 1.0


def perceive_team_confidence(engine: MatchEngine, team: str,
                             window_s: float = 600.0) -> float:
    """
    Return 0.0 (DOWN) / 0.5 (NEUTRAL) / 1.0 (UP).

    Derived from: successful actions (passes completed, tackles won, shots on
    target) minus failures (turnovers, lost duels, missed shots) in the last
    window_s seconds, delayed by 2 minutes.  Categorical: 3 levels.
    """
    end = _window_end_minute(engine)
    win_min = max(1, int(window_s / 60.0))
    succ, fail = _team_event_counts(engine, team, end, win_min)
    net = succ - fail
    if net >= 6:
        return 1.0
    if net <= -6:
        return 0.0
    return 0.5


def perceive_momentum(engine: MatchEngine, team: str,
                      window_s: float = 300.0) -> float:
    """
    Return 0.0 (AGAINST_US) / 0.5 (NEUTRAL) / 1.0 (WITH_US).

    Derived from: share of the last window_s of the ball path the ball spent
    in the team's ATTACKING third (from match_ball_path).  NOT from
    state.momentum.
    """
    end_s = float(getattr(engine.state, "match_clock_s", 0.0))
    start_s = max(0.0, end_s - window_s)
    att_right = engine.position_engine.team_attacks_right.get(team, True)

    own = zone_total = 0
    for pt in engine.state.match_ball_path:
        t = float(pt.get("t", 0.0))
        if t < start_s or t > end_s:
            continue
        if pt.get("team", "") != team:
            continue
        x = float(pt.get("x", 52.5))
        att_third = x > 70.0 if att_right else x < 35.0
        own += 1
        if att_third:
            zone_total += 1

    share = zone_total / own if own else 0.0
    if share >= 0.5:
        return 1.0
    if share <= 0.2:
        return 0.0
    return 0.5


def perceive_opp_posture(engine: MatchEngine, opponent_team: str,
                         window_s: float = 120.0) -> float:
    """
    Return 0.0 (DEEP) / 0.5 (MID) / 1.0 (HIGH).

    Derived from: average x-position of the opponent's back four (CB / LB /
    RB) over the last window_s minutes.  NOT from state.team_stances.
    """
    frames = _window_frames(engine, opponent_team, window_s)
    rows = _team_rows(engine, frames, opponent_team)

    def_line = engine.position_engine.DEFENSIVE_LINE_POSITIONS
    back4_x: List[float] = []
    for r in rows:
        if r.get("position") in def_line:
            back4_x.append(float(r.get("x", 50.0)))

    if not back4_x:
        return 0.5
    avg_x = sum(back4_x) / len(back4_x)

    # Opponent's own goal line: home defends x=0, away defends x=105.
    attacks_right = engine.position_engine.team_attacks_right.get(
        opponent_team, True)
    own_goal_x = 0.0 if attacks_right else 105.0
    depth = abs(avg_x - own_goal_x) / 55.0   # 0 = on own goal line, ~1 = halfway+

    if depth >= 0.65:
        return 1.0
    if depth <= 0.35:
        return 0.0
    return 0.5


def _player_signal(engine: MatchEngine, player: Any,
                   window_s: float) -> dict:
    """Raw perception data for one player: movement/sprint against baseline,
    and error-proneness from the timeline.  Returns a dict of floats."""
    name = getattr(player, "name", player)
    frames = _window_frames(engine, "", window_s)   # team-agnostic frame grab
    rows = [r for f in frames
            for r in (f.get("home", []) + f.get("away", []))
            if r.get("player") == name]
    base_rows = [r for f in engine.position_log
                 if int(f.get("minute", 0)) <= 3
                 for r in (f.get("home", []) + f.get("away", []))
                 if r.get("player") == name]

    def _per_min(row_gen: List[dict]) -> tuple:
        dist = 0.0
        sprint = 0.0
        n = 0
        for r in row_gen:
            dist += float(r.get("distance_total", 0.0))
            sprint += float(r.get("physics_sprint_count", 0.0))
            n += 1
        return (dist / n if n else 0.0, sprint / n if n else 0.0)

    cur_dist, cur_sprint = _per_min(rows)
    base_dist, base_sprint = _per_min(base_rows)
    dist_ratio = cur_dist / base_dist if base_dist > 0 else 1.0

    # error-proneness from the timeline (lost duels + turnovers) *last window*
    end = _window_end_minute(engine)
    win_min = max(1, int(window_s / 60.0))
    start = max(0, end - win_min + 1)
    errors = 0
    for ev in engine.timeline:
        m = int(getattr(ev, "minute", 0))
        if m < start or m > end:
            continue
        if getattr(ev, "player", None) != name:
            continue
        ename = getattr(ev.event_type, "name", "")
        if ename in _FAILURE_EVENTS:
            errors += 1

    return dict(dist_ratio=dist_ratio, cur_sprint=cur_sprint,
                base_sprint=base_sprint, errors=errors)


def perceive_player_state(engine: MatchEngine, player: Any,
                          window_s: float = 300.0) -> float:
    """
    Return 0.0 (OK) / 0.5 (STRUGGLING) / 1.0 (BAD).

    Derived from: recent movement relative to baseline, recent lost duels +
    turnovers, and recent sprint output.  Aggregated: 3 levels.
    """
    s = _player_signal(engine, player, window_s)
    # Sprint output collapsing vs baseline + errors mounting = BAD.
    sprint_out = s["cur_sprint"] / s["base_sprint"] if s["base_sprint"] > 0 else 1.0
    bad = 0.0
    if s["dist_ratio"] <= 0.55 or sprint_out <= 0.45:
        bad += 1.0
    if s["errors"] >= 3:
        bad += 0.75
    elif s["errors"] >= 1:
        bad += 0.35

    if bad >= 1.0:
        return 1.0
    if bad >= 0.4:
        return 0.5
    return 0.0


def perceive_best_player_state(engine: MatchEngine, team: str,
                               window_s: float = 300.0) -> float:
    """
    Return 0.0 (NORMAL) / 1.0 (ON_FIRE).

    Derived from: recent successful actions by the team's most involved
    attacker (the active attacker with the most touches in the window).
    """
    frames = _window_frames(engine, team, window_s)
    rows = _team_rows(engine, frames, team)
    touches_by: dict = {}
    for r in rows:
        if r.get("position") not in _ATTACK_ROLES:
            continue
        touches_by[r["player"]] = touches_by.get(r["player"], 0) + int(
            r.get("touches", 0) or 0)
    if not touches_by:
        return 0.0
    lead = max(touches_by, key=touches_by.get)

    end = _window_end_minute(engine)
    win_min = max(1, int(window_s / 60.0))
    start = max(0, end - win_min + 1)
    succ = 0
    fail = 0
    for ev in engine.timeline:
        m = int(getattr(ev, "minute", 0))
        if m < start or m > end:
            continue
        if getattr(ev, "player", None) != lead:
            continue
        name = getattr(ev.event_type, "name", "")
        if name in _SUCCESS_EVENTS:
            succ += 1
        elif name in _FAILURE_EVENTS:
            fail += 1
    net = succ - fail
    if net >= 3:
        return 1.0
    if net >= 1 and succ >= 4:
        return 1.0
    return 0.0


def perceive_key_player_available(engine: MatchEngine, team: str) -> float:
    """
    Return 1.0 if the team's designated key player (a team_superstars /
    d.n.is_superstar player, or the highest-rated active player) is on the
    pitch, else 0.0.
    """
    active = engine.active_players.get(team, [])
    if not active:
        return 0.0
    best_rating = -1.0
    for p in active:
        dna = getattr(p, "dna", None)
        if dna is not None and getattr(dna, "is_superstar", False):
            return 1.0
        if dna is not None:
            try:
                best_rating = max(best_rating, float(dna.overall_rating))
            except Exception:
                pass

    # If a superstar exists in the full squad but is NOT currently on the
    # pitch (subbed/injured/sent off), treat the key player as unavailable.
    squad = engine.squads.get(team, {})
    for p in list(squad.get("starters", [])) + list(squad.get("substitutes", [])):
        dna = getattr(p, "dna", None)
        if dna is not None and getattr(dna, "is_superstar", False):
            return 0.0
    return 1.0 if best_rating >= 0.0 else 0.0


def perceive_game_importance(engine: MatchEngine) -> float:
    """
    Return 0..1.
      - Derby: base 0.7
      - Late-season top-of-table or relegation fight: +0.2
      - Cup final: 1.0
      - Otherwise: 0.2
    Note: table positions live in the (read-only) season state; the engine
    alone cannot see them, so the "late-season fight" branch keys off a
    late matchday instead (e.g. ≥ matchday 29 of a 34-round season).
    """
    cfg = engine.config
    if cfg.is_derby:
        base = 0.7
    else:
        for token in ("cup", "final", "FA Cup", "PLOFA Cup"):
            if token.lower() in str(getattr(cfg, "competition", "")).lower():
                return 1.0
        base = 0.2

    # Late-season carries extra weight regardless of the table (the exact
    # top/relegation fight position is not visible to the engine).
    if not cfg.is_derby and int(getattr(cfg, "matchday", 1)) >= 29:
        base += 0.2
    return round(min(1.0, base), 2)


def perceive_recent_goal_impact(engine: MatchEngine, team: str,
                                decay_s: float = 300.0) -> float:
    """
    Return -1..+1.
      - conceded recently: -exp(-Δmin / 5)
      - scored recently:   +exp(-Δmin / 5)
      - sum contributions, clamp to [-1, 1]
    """
    now = _current_minute(engine)
    win_min = max(1, int(decay_s / 60.0))
    score_names = frozenset({"GOAL", "PENALTY_SCORED", "OWN_GOAL"})
    total = 0.0
    for ev in engine.timeline:
        name = getattr(ev.event_type, "name", "")
        if name not in score_names:
            continue
        gmin = int(getattr(ev, "minute", 0))
        if now - gmin > win_min:
            continue
        dmin = max(0, now - gmin)
        contribution = math.exp(-dmin / 5.0)
        if getattr(ev, "team", None) == team:
            total += contribution
        else:
            total -= contribution
    return round(max(-1.0, min(1.0, total)), 4)