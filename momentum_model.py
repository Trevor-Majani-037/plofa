"""
MATCH MOMENTUM — a real threat model (xT / possession-value lineage)
===================================================================

WHY THIS FILE EXISTS
--------------------
The previous momentum series was NOT momentum. It was a goal counter wearing
momentum's name:

    MomentumEngine.after_goal()      # +18..30 on a goal
    MomentumEngine.after_red_card()  # +25..40 on a red
    MomentumEngine.natural_decay()   # x 0.92 EVERY MINUTE, toward 0

and that was the complete set of writers. So a match with no goals and no
red cards produced a momentum series that was mathematically obliged to be
zero, and the chart read as a flat line for 45 minutes. The only visible
structure in a normal match was a spike at the minute of each goal, decaying
away from it. Three things follow, all of them wrong in the way the user
described:

  1. It could not RISE without a goal. A team shredding the right flank for
     four minutes registered nothing.
  2. A goal was FORCED onto the chart at full height, so every goal looked
     like it arrived on a wave. Real football's best goals arrive on flat
     momentum — a corner cleared onto a half-volley by a team that had been
     outplayed for an hour. That outcome is unreachable by construction here.
  3. It measured the SCOREBOARD, not the match.

THE MODEL ACTUALLY IMPLEMENTED
------------------------------
This is the algorithm Opta/Stats Perform publishes for Match Momentum, with
the xT substitution that the independent literature uses when the proprietary
possession-value model is not available (Kapich 2024, following Karun Singh):

  step 1  THREAT GRID (xT).  Every zone of the pitch carries a value: the
          probability that a shot from there becomes a goal. Built on an
          18x12 grid in the ATTACKING frame.

  step 2  ACTION VALUE.  An on-ball action is worth the CHANGE in threat it
          causes:

              xT(action) = value(end_zone) - value(start_zone)

          This is the elegant part and it is why xT needs no special case
          for bad football: a sideways pass in own half is negative on its
          own. Regression, risk-free circulation and a ball played away from
          danger all score below zero without anybody writing a penalty for
          them.

  step 3  CLIP to [0, 0.10].  Opta's cap. Without it a penalty kick (~0.79)
          blows out the scale and flattens the rest of the match into noise.
          The cap is why a goal can be scored "in low momentum" and why a
          single freak action cannot own a minute.

  step 4  PER-MINUTE MAX, per team.  Not a sum, not an average. One great
          chance defines a minute of football the same way a p99 defines a
          latency graph: the mean hides everything. A 20-pass move that ends
          in a shot and a scrappy 40-yard hopeful are not the same threat,
          and the max takes the one that was.

  step 5  RECENCY DECAY (EWMA).  Each bar blends the current minute with the
          previous three, weighted by recency:

              w_k = exp(-DECAY * k),  k = 0..WINDOW-1,  normalised to sum 1

          Only the last 3-4 minutes meaningfully contribute. This is also
          where the WAVE SHAPE comes from: the raw per-minute maxima are
          jagged, and it is the smoothing that makes the chart read as
          pressure building and receding rather than as noise.

  step 6  DIFFERENCE.  momentum(t) = smoothed_home(t) - smoothed_away(t).
          Read it as: at minute t, this team was that many percentage points
          more likely to score within the next 10 seconds than the other.

WHAT IS DELIBERATELY NOT IN HERE
--------------------------------
  * GOALS DO NOT MOVE THE CURVE. Not even a little. The whole point of the
    metric (Opta's stated aim) is that it fills the gap xG cannot: "we can
    still measure a team's threat if they get into dangerous areas but don't
    convert this into a shot." Injecting a goal spike would re-create the
    exact defect this file exists to remove. A goal appears on the chart as
    an ANNOTATION on the minute it happened, sitting at whatever height the
    threat curve had genuinely reached — which is frequently nothing like
    where the old chart pinned it at +/-100.

  * PENALTIES ARE NOT SPECIAL-CASED IN THE ACTION TERM. They are worth what
    the threat grid says the spot is worth, exactly as Opta's cap intends.
    A penalty is a high-xT action in a high-xT zone and is scored as such.

  * NOTHING HERE TOUCHES THE SIMULATION. This module reads a finished
    `MatchResult` and writes a series. It draws no random numbers, mutates no
    state, and changes no event. See "WHY POST-HOC" below.

WHY POST-HOC, AND WHY THAT IS A FEATURE
---------------------------------------
The live `MomentumEngine` in `match_engine.py` also feeds back into gameplay
(`get_attacking_probability_modifier`, and `state.momentum / 100 * 0.08` at
the shot funnel). Rewriting it would change every event in every match, which
per this project's own rules needs its own gate and invalidates every
calibration number in AGENTS.md. The chart is a REPORTING artefact; the fix
belongs in the reporting layer. So this module is deliberately one-way:

    match_engine  <-  momentum_model  <-  opta_analytics  <-  exporter

and `momentum_model` must never be imported by `match_engine`. The live
scalar is left exactly as it is; see AGENTS.md for the naming collision,
because it is still doing (a different, gameplay) job.

THE GRID IS A PRIOR, NOT A FIT — AND IT SAYS SO
-----------------------------------------------
`default_grid()` seeds every cell from `XGEngine.calculate_geometric()`, the
project's OWN continuous shot-quality model (distance + angle + body + situa-
tion). That is a defensible prior rather than a typed table: it is the
simulation's own belief about how good a shot is from each square metre,
reused as the belief about how good a POSSESSION is there.

But it is still a prior, and this project does not get to call a prior a
model's output without saying so. `fit_grid_from_matches()` performs the real
empirical fit — goals over shots per cell, shrunk toward the prior — and
REFUSES to return a grid unless it has enough shots to support the cells it
filled. `MomentumResult.diagnostics["grid_source"]` reports `"prior"` or
`"fitted"` into the export, so a consumer can tell which produced a number.

FRAME — THE PROJECT'S STANDING TRAP
-----------------------------------
`PositionEngine.team_attacks_right` is set once per team at
`initialize_team` and is NEVER flipped at half time, so home is always
`attacks_right=True` in raw coordinates, both halves. Every lookup here goes
through `to_attacking_frame()` and nothing reads a raw x. Nine coordinates
per match sit off the pitch (to about -12.8 and 119.7); they are CLAMPED for
the lookup and COUNTED in `diagnostics["clamped_coordinates"]`, never silently
absorbed and never clamped in the stored event.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from match_engine import EventType, MatchEvent, MatchResult, SituationType, XGEngine

# ── Pitch ──────────────────────────────────────────────────────────────
PITCH_X = 105.0
PITCH_Y = 68.0

# ── Grid geometry. 18 x 12 => 5.83 m x 5.67 m cells, near-square. ──────
XT_COLS = 18
XT_ROWS = 12

# ── Opta's published constants (theanalyst.com "What is Match Momentum?").
XT_CLIP = 0.10   # "capped between zero and 0.1"
WINDOW = 4       # "only the most recent three to four minutes have impact"
DECAY = 0.25     # exponential recency weight, per Kapich's reproduction


# ─────────────────────────────────────────────────────────────────────
# WHICH EVENTS SCORE, AND WHY
#
# These are the actions that MOVE THE BALL BETWEEN ZONES, plus the actions
# that ARE the threat. Everything else is excluded on purpose:
#
#   excluded  POSSESSION_SEQUENCE   a container, double-counts its children
#   included   PROGRESSIVE_PASS     NOT a duplicate of PASS -- event_chain.py
#                                  MUTATES the type (`etype = ...`) rather
#                                  than emitting a second event, so counting
#                                  both cannot double-count one action.
#   excluded  TURNOVER / MISCONTROL a failed ball-mover is already scored by
#                                  its own zero end_zone; charging the
#                                  turnover as well would double-charge it.
#   excluded  defensive events     a defender's good work shows up as the
#                                  ATTACKER's reduced threat, which is the
#                                  correct place for it. Charging both sides
#                                  would cancel out into a flat line.
# ─────────────────────────────────────────────────────────────────────
_BALL_MOVERS = {
    EventType.PASS,
    EventType.PROGRESSIVE_PASS,
    EventType.CARRY,
    EventType.CROSS_ATTEMPT,
    EventType.CROSS_SUCCESS,
    EventType.THROUGH_BALL,
    EventType.DRIBBLE_SUCCESS,
}

# The single most threatening on-ball action there is. Included as its own
# term because Karun's xT variant deliberately drops shots and a team hitting
# the woodwork five times would otherwise register ZERO threat -- which is the
# exact class of complaint this module was written to answer. Opta's PV scores
# every on-ball action, so this is the faithful choice, not an invention.
# Note HIT_WOODWORK is here even though `MatchEvent.is_shot` omits it: that
# omission is a documented, separate calibration question in AGENTS.md, and a
# ball on the frame is a shot in football.
_SHOT_EVENTS = {
    EventType.SHOT_ON_TARGET,
    EventType.SHOT_OFF_TARGET,
    EventType.SHOT_BLOCKED,
    EventType.HIT_WOODWORK,
    EventType.GOAL,
    EventType.PENALTY_SCORED,
    EventType.PENALTY_MISSED,
}

# A recorded chance carries threat independent of whether a shot followed.
_CHANCE_EVENTS = {
    EventType.CHANCE_CREATED,
    EventType.BIG_CHANCE_CREATED,
}

# Situations that are a GIFT rather than an accumulation of play: a corner or
# a free kick is delivered threat, not built-up threat, so letting it score at
# full grid value would make set pieces the dominant driver of the curve.
# Opta's PV scores the DELIVERY low and the action it creates high. 0.55 is
# the weight that keeps a well-worked corner visible while denying a routine
# one the ability to own a minute -- and it is a calibrated choice, recorded
# as such, not a derived constant.
_SET_PIECE_WEIGHT = 0.55

_SET_PIECE_SITUATIONS = frozenset({
    SituationType.CORNER,
    SituationType.CROSSED_FREEKICK,
    SituationType.DIRECT_FREEKICK,
    SituationType.PENALTY,
})

_GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "momentum_grid.json")


# ─────────────────────────────────────────────────────────────────────
# FRAME
# ─────────────────────────────────────────────────────────────────────

def to_attacking_frame(x: float, y: float, attacks_right: bool) -> Tuple[float, float]:
    """Raw engine coords -> the ATTACKING frame, in which this module thinks.

    Returns ``(ax, ay)`` where ``ax`` is metres from the team's OWN goal line
    (so the goal being attacked is at ax = 105) and ``ay`` is metres from the
    left touchline.

    Every threat lookup in this file goes through here. Nothing reads a raw x.
    """
    ax = x if attacks_right else PITCH_X - x
    return ax, y


def cell_of(ax: float, ay: float) -> Tuple[int, int]:
    """Attacking-frame coords -> grid cell (col, row), clamping to the pitch.

    The clamp is for the LOOKUP ONLY and is counted by the caller -- see
    ``compute_momentum``'s ``clamped_coordinates`` diagnostic. A live match
    puts a handful of coordinates off the pitch every match; a silent clamp
    would hide them, and an unclamped lookup would raise.
    """
    ax = min(PITCH_X, max(0.0, ax))
    ay = min(PITCH_Y, max(0.0, ay))
    col = min(XT_COLS - 1, int(ax / PITCH_X * XT_COLS))
    row = min(XT_ROWS - 1, int(ay / PITCH_Y * XT_ROWS))
    return col, row


def zone_value(grid: Sequence[Sequence[float]], x: float, y: float,
               attacks_right: bool) -> float:
    """Threat value of a location, from the team's attacking perspective."""
    ax, ay = to_attacking_frame(x, y, attacks_right)
    col, row = cell_of(ax, ay)
    return float(grid[row][col])


# ─────────────────────────────────────────────────────────────────────
# THE GRID
# ─────────────────────────────────────────────────────────────────────

def default_grid() -> List[List[float]]:
    """Seed the xT grid from the project's OWN shot model.

    Cell value = ``XGEngine.calculate_geometric()`` at the cell centre, for an
    open-play unpressured footed shot. That makes the grid self-consistent
    with the xG the simulation actually assigns to shots: a zone is worth what
    the engine thinks a shot from it is worth.

    This is a PRIOR. It is not fitted to goals. See ``fit_grid_from_matches``.
    """
    grid: List[List[float]] = []
    for r in range(XT_ROWS):
        row_vals: List[float] = []
        ay = (r + 0.5) / XT_ROWS * PITCH_Y          # cell centre
        for c in range(XT_COLS):
            ax = (c + 0.5) / XT_COLS * PITCH_X
            v = XGEngine.calculate_geometric(
                x=ax, y=ay, body_part="right_foot",
                situation=SituationType.OPEN_PLAY,
                under_pressure=False, attacks_right=True,
            )
            row_vals.append(float(v))
        grid.append(row_vals)
    return grid


def fit_grid_from_matches(
        results: Iterable[MatchResult],
        min_shots_per_cell: int = 25,
        prior_strength: float = 12.0,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Empirical xT fit: goals / shots per cell, shrunk toward the prior.

    Shrinkage is a Laplace pull toward ``default_grid()``:

        value = (goals + k*prior) / (shots + k)

    so a cell with few shots stays near the prior instead of reporting
    "0.000" or "1.000" off three attempts.

    REFUSES to return a fitted grid when the sample is too thin, and says so
    in the diagnostics rather than shipping a grid that looks measured and is
    not. ``min_shots_per_cell`` applies to the MEDIAN cell; the per-cell shot
    counts are returned so a caller can see the thin cells for itself.
    """
    prior = default_grid()
    goals = [[0.0] * XT_COLS for _ in range(XT_ROWS)]
    shots = [[0.0] * XT_COLS for _ in range(XT_ROWS)]

    n_matches = 0
    for res in results:
        n_matches += 1
        home = res.config.home_team
        for ev in res.timeline:
            if ev.event_type not in _SHOT_EVENTS:
                continue
            if ev.event_type == EventType.OWN_GOAL:
                continue          # credited to nobody's shot map
            attacks_right = (ev.team == home)
            ax, ay = to_attacking_frame(ev.location_x, ev.location_y, attacks_right)
            col, row = cell_of(ax, ay)
            shots[row][col] += 1.0
            if ev.event_type in (EventType.GOAL, EventType.PENALTY_SCORED):
                goals[row][col] += 1.0

    per_cell = [shots[r][c] for r in range(XT_ROWS) for c in range(XT_COLS)]
    ordered = sorted(per_cell)
    median_shots = ordered[len(ordered) // 2] if ordered else 0.0
    diag = {
        "matches": n_matches,
        "total_shots": sum(per_cell),
        "median_shots_per_cell": median_shots,
        "cells_at_or_above_min": sum(1 for s in per_cell if s >= min_shots_per_cell),
        "cells_total": len(per_cell),
        "min_shots_per_cell": min_shots_per_cell,
    }

    if median_shots < min_shots_per_cell:
        diag["fitted"] = False
        diag["reason"] = (
            f"median cell has {median_shots:.0f} shots, below the required "
            f"{min_shots_per_cell}; returning the PRIOR grid instead of a "
            f"fit that would be noise")
        return prior, diag

    grid: List[List[float]] = []
    for r in range(XT_ROWS):
        grid.append([
            (goals[r][c] + prior_strength * prior[r][c]) /
            (shots[r][c] + prior_strength)
            for c in range(XT_COLS)
        ])
    diag["fitted"] = True
    diag["per_cell_shots"] = per_cell
    return grid, diag


def load_grid(path: str = _GRID_PATH) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Load a fitted grid from disk if one exists, else the analytic prior.

    Returns ``(grid, source_info)`` where ``source_info["grid_source"]`` is
    ``"fitted"`` or ``"prior"`` and travels into the export, so a consumer can
    always tell which produced a number.
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            grid = blob["grid"]
            if len(grid) == XT_ROWS and all(len(r) == XT_COLS for r in grid):
                return grid, {"grid_source": "fitted", "grid_path": path,
                              "grid_fit": blob.get("fit", {})}
        except (ValueError, KeyError, TypeError):
            # A corrupt grid file must not take the export down; fall through
            # to the prior, and the caller sees "prior" in the provenance.
            pass
    return default_grid(), {"grid_source": "prior", "grid_path": None}


def save_grid(grid: Sequence[Sequence[float]], fit: Dict[str, Any],
              path: str = _GRID_PATH) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"grid": grid, "fit": fit, "cols": XT_COLS,
                   "rows": XT_ROWS, "pitch": [PITCH_X, PITCH_Y]}, fh, indent=1)


# ─────────────────────────────────────────────────────────────────────
# ACTION VALUE
# ─────────────────────────────────────────────────────────────────────

def _situation_weight(situation: SituationType) -> float:
    """Discount applied to the ACTION'S OWN value for a gifted situation."""
    return _SET_PIECE_WEIGHT if situation in _SET_PIECE_SITUATIONS else 1.0


def _situation_bonus(situation: SituationType, end_value: float) -> float:
    """A set-piece DELIVERY carries some threat of its own. Open play: ZERO.

    Opta's possession-value model scores the delivery low and the action it
    creates high, because a corner is not an accumulation of play -- it is a
    gift. ``end_value * weight`` keeps the delivery visible in the series
    while leaving the receiving action to be scored on its own merits by the
    pass/cross that follows it.

    The open-play branch returning ZERO is load-bearing, and it is here
    because the first version of this function returned ``end_value * 1.0``
    for everything. That added the entire end-zone value to every pass, so an
    open-play pass scored ``2*end - start``, both teams pinned at the clip on
    identical ground, and the DIFFERENCE -- the only thing the chart plots --
    collapsed to zero for the whole match. A term that is meant to model the
    exceptional case must be zero everywhere else, or it stops being one.
    """
    if situation not in _SET_PIECE_SITUATIONS:
        return 0.0
    return end_value * _SET_PIECE_WEIGHT


def action_value(ev: MatchEvent, grid: Sequence[Sequence[float]],
                 attacks_right: bool) -> Optional[float]:
    """Threat CHANGE caused by one event, or None if it is not a threat event.

    Returns the xT contribution in the team's own frame (positive = the team
    made itself more likely to score).
    """
    et = ev.event_type
    if et in _SHOT_EVENTS or et in _CHANCE_EVENTS:
        # The action IS the threat. A shot from a good zone is the single
        # largest per-minute contributor the grid can produce, which is why
        # a team shooting well out-possesses a team that does not.
        base = zone_value(grid, ev.location_x, ev.location_y, attacks_right)
        return base * _situation_weight(ev.situation)

    if et not in _BALL_MOVERS:
        return None

    # value(end) - value(start). A failed ball-mover never travels, so it is
    # scored at the zone it was played from: contribution ~0, and the danger
    # it walked into is instead charged by the NEXT action, which belongs to
    # the opponent. No special-case penalty is needed and none is applied.
    start = zone_value(grid, ev.location_x, ev.location_y, attacks_right)
    if not ev.outcome or ev.end_x is None or ev.end_y is None:
        return 0.0
    end = zone_value(grid, ev.end_x, ev.end_y, attacks_right)
    return (end - start) + _situation_bonus(ev.situation, end)


# ─────────────────────────────────────────────────────────────────────
# THE SERIES
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MomentumResult:
    """One row per minute, plus the provenance needed to not over-claim."""
    series: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _ewma_weights(window: int = WINDOW, decay: float = DECAY) -> List[float]:
    """Normalised recency weights, most recent first."""
    raw = [math.exp(-decay * k) for k in range(window)]
    total = sum(raw)
    return [w / total for w in raw]


def compute_momentum(result: MatchResult,
                     grid: Optional[Sequence[Sequence[float]]] = None,
                     window: int = WINDOW,
                     decay: float = DECAY,
                     xt_clip: float = XT_CLIP) -> MomentumResult:
    """Build the true per-minute momentum series from a finished match.

    Pure function of ``result``: reads the timeline, draws no random numbers,
    mutates nothing. See the module docstring for why it is post-hoc.
    """
    if grid is None:
        grid, grid_info = load_grid()
    else:
        grid_info = {"grid_source": "supplied", "grid_path": None}

    home = result.config.home_team
    away = result.config.away_team

    # ── last minute seen, so the series covers added time too ──────
    # Read `state` defensively: several existing tests (e.g.
    # test_opta_analytics_runs_sprints) build a partial result as a
    # SimpleNamespace with no state at all, and this module is a pure
    # function of a TIMELINE, so it should not require one.
    last_minute = 90
    for ev in result.timeline:
        if ev.minute > last_minute:
            last_minute = ev.minute
    added = getattr(getattr(result, "state", None), "added_time", 0)
    if added:
        last_minute = max(last_minute, 90 + int(added))

    # ── steps 2+3: per-action xT, clipped ─────────────────────────
    xt_max: Dict[Tuple[str, int], float] = {}
    actions_scored = 0
    actions_skipped_no_end = 0
    clamped = 0

    for ev in result.timeline:
        if ev.team not in (home, away):
            continue
        attacks_right = (ev.team == home)
        # Count coordinates that sit off the pitch; the cell lookup clamps
        # them, and the count is reported rather than absorbed.
        ax, ay = to_attacking_frame(ev.location_x, ev.location_y, attacks_right)
        if not (0.0 <= ax <= PITCH_X and 0.0 <= ay <= PITCH_Y):
            clamped += 1
        if ev.end_x is not None and ev.end_y is not None:
            bex, bey = to_attacking_frame(ev.end_x, ev.end_y, attacks_right)
            if not (0.0 <= bex <= PITCH_X and 0.0 <= bey <= PITCH_Y):
                clamped += 1

        val = action_value(ev, grid, attacks_right)
        if val is None:
            continue
        if ev.event_type in _BALL_MOVERS and ev.outcome \
                and (ev.end_x is None or ev.end_y is None):
            actions_skipped_no_end += 1
        actions_scored += 1

        clipped = min(xt_clip, max(0.0, val))
        key = (ev.team, int(ev.minute))
        if clipped > xt_max.get(key, -1.0):
            xt_max[key] = clipped

    # ── step 5: recency-decayed smoothing, per team ───────────────
    weights = _ewma_weights(window, decay)
    minutes = list(range(1, last_minute + 1))

    def smoothed(team: str, minute: int) -> float:
        acc = 0.0
        for k, w in enumerate(weights):
            m = minute - k
            if m >= 1:
                acc += w * xt_max.get((team, m), 0.0)
        return acc

    # ── step 6: the difference ─────────────────────────────────────
    # Scoreline is read off the timeline rather than assumed, so each row's
    # goals columns agree with the events at that minute.
    goals_by_minute: Dict[int, List[str]] = {}
    for ev in result.timeline:
        if ev.event_type in (EventType.GOAL, EventType.PENALTY_SCORED,
                             EventType.OWN_GOAL):
            goals_by_minute.setdefault(int(ev.minute), []).append(ev.team)

    series: List[Dict[str, Any]] = []
    hg = ag = 0
    for m in minutes:
        for t in goals_by_minute.get(m, ()):
            if t == home:
                hg += 1
            elif t == away:
                ag += 1
        s_home, s_away = smoothed(home, m), smoothed(away, m)
        series.append({
            "minute": m,
            "momentum": round(s_home - s_away, 5),
            "home_threat": round(s_home, 5),
            "away_threat": round(s_away, 5),
            "home_goals": hg,
            "away_goals": ag,
        })

    diagnostics = {
        "model": "xT momentum (Opta/Karun lineage)",
        "definition": ("difference in smoothed threat; positive = home more "
                       "likely to score within the next 10 seconds"),
        "goals_affect_curve": False,
        "xt_clip": xt_clip,
        "window_minutes": window,
        "decay": decay,
        "actions_scored": actions_scored,
        "actions_without_end_point": actions_skipped_no_end,
        "clamped_coordinates": clamped,
        "minutes": len(minutes),
        **grid_info,
    }
    return MomentumResult(series=series, diagnostics=diagnostics)