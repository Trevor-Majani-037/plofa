"""THREAT RETURN — the training reward for the on-ball policy.

WHY THIS REPLACES THE OLD OBJECTIVE
-----------------------------------
The original surrogate (``surrogate_collect._event_success``) scored **engine
success**: a completed pass 2.0 + a small advance bonus, a retained carry 1.0,
a turnover 0.0, default 0.5.  That objective asks *"which intent does this
engine complete?"* — and this engine completes a square ball far more often
than a through-ball.  So a square ball beat a through-ball, safety always won,
and the GA converged to lateral spread.  That is T2G160 exactly: the best
fitness number the project has ever produced (0.6523 vs a heuristic 0.5373,
all eleven positions saturated at 1.0000) with pass maps indistinguishable
from the heuristic, rejected after three real matches.

The lesson recorded in AGENTS.md is that **a constraint on a bad objective
relocates the failure rather than correcting it.**  More generations, more
data and more penalties all made the number better and the football worse.
So the objective is replaced, not tuned.

WHAT IT IS NOW
--------------
The reward is the **change in threat** the action caused, i.e. a potential
difference:

    r = PHI(end) - PHI(start)

where ``PHI`` is the xT grid ``momentum_model`` already derives from this
project's OWN shot model (``XGEngine.calculate_geometric`` at each cell
centre), so the reward and the engine's xG cannot disagree.

This is a *local* reward that needs no possession-chain segmentation — which
matters, because the engine does not record a per-player possession chain to
segment.  It also telescopes: the sum of per-action potential differences
along a sequence equals the potential difference across it, so an intent is
credited with the ground it actually gained rather than with a bucket label.

THE FOUR CASES, AND WHY FAILURES ARE CHARGED
--------------------------------------------
1. **Ball movers** (pass / carry / dribble).  Completed: ``PHI(end) - PHI(start)``,
   signed, so a backward ball is genuinely negative.  Failed: charged the
   OPPONENT'S threat at the point the ball was lost,
   ``-LOSS_WEIGHT * PHI_opp(loss point)``.

   The failure charge is not a hand-tuned penalty on an intent, which is what
   destroyed the v3 surrogate; it is geometry.  Without it, a failed
   through-ball would score 0.0 — the same as a completed square ball — so
   expected value would make every risky ball free and the GA would swing to
   the opposite degenerate optimum.  That is not a hypothetical: the
   "brain only" arm already measured 41.7% progressive against a real 25-35%,
   with a 23.9 m median pass.  A loss in your own third is charged ~-0.02; a
   loss on the edge of your own box ~-0.5.  The geometry carries the gradient.

2. **Shots.**  A shot is the threat, so a shot gets the ABSOLUTE zone value
   (which is the xG from that spot), not a difference.  Using the difference
   would make shooting worth exactly 0 by construction, because
   ``PHI(shot spot)`` is the xG and the ball then leaves the pitch.

3. **Goals.**  ``GOAL_VALUE`` = 1.0, deliberately above any possible xG
   (max ~0.9), so a goal always beats the best shot that did not go in.

4. **Everything else.**  0.0 — neutral.  Not 0.5.  With a signed reward whose
   observed median is near zero, a positive default would make every intent
   never seen during collection (including SHOOT in a new bucket) outrank
   every intent that was, which is the opposite of what the default is for.

NOTHING HERE PENALISES DIVERSITY
--------------------------------
There is no dominance penalty, no effective-usage penalty and no entropy
term.  A brain that always plays the progressive pass because that is what
scores is CORRECT; taxing its share of the argmax is how the v3 objective
pushed every position back to safe lateral spread.  The counterweight against
a degenerate all-through-ball policy is the loss charge above, which is a
property of football rather than of a distribution.

Pure functions.  No RNG.  Reads only ``MatchEvent`` fields and the grid.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import momentum_model as _mm

# ── constants (all published in ``describe()`` so an export can cite them) ──

GOAL_VALUE = 1.0            # a goal beats any achievable xG (max ~0.9)
SHOT_OFF_TARGET_FRAC = 0.5  # the chance was created but wasted
LOSS_WEIGHT = 0.6           # fraction of the OPPONENT's threat a loss costs

# Event-type groups, named from the engine's own `EventType` enum.
#
# THE OUTCOME-BEARING EVENT, exactly one per decision.  This matters more than
# it looks: `CROSS_SUCCESS` is emitted IN ADDITION to `CROSS_ATTEMPT` (which
# already carries `outcome=cross_success`), so counting both would consume two
# events for one decision and shift every later pairing for that player.
# `PASS` / `PROGRESSIVE_PASS` / `SWITCH_OF_PLAY` / `THROUGH_BALL` are mutually
# exclusive: the pass site chooses ONE `etype`.
_SHOT_ON_TYPES = frozenset({
    "SHOT_ON_TARGET", "SAVE", "SHOT_BLOCKED", "HIT_WOODWORK",
})
_SHOT_OFF_TYPES = frozenset({"SHOT_OFF_TARGET"})
_GOAL_TYPES = frozenset({"GOAL", "PENALTY_SCORED"})
_MISS_TYPES = frozenset({"PENALTY_MISSED"})
_MOVER_TYPES = frozenset({
    "PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
    "CARRY", "DRIBBLE_SUCCESS", "CROSS_ATTEMPT",
})
_LOSS_TYPES = frozenset({
    "TURNOVER", "MISCONTROL", "DISPOSSESSED", "INTERCEPTION", "DRIBBLE_FAIL",
})

#: Every event type that IS an execution outcome, for the collector.
EXECUTION_TYPES = frozenset(
    _MOVER_TYPES | _LOSS_TYPES | _SHOT_ON_TYPES | _SHOT_OFF_TYPES
    | _GOAL_TYPES | _MISS_TYPES
)

#: Every event type that IS a shot (a goal counts: a goal IS a shot that
#: went in).  The collector needs this to answer a policy SHOOT decision
#: with the shot it caused: the possession chain only creates a ZERO-LENGTH
#: marker carry for a shoot intent, whose threat change is exactly 0.0, and
#: the real shot is fired by the ATTACK chain, so without this the policy
#: can never be credited with the chance it created.
SHOT_TYPES = frozenset(
    _SHOT_ON_TYPES | _SHOT_OFF_TYPES | _GOAL_TYPES | _MISS_TYPES
)


# ── grid ────────────────────────────────────────────────────────────────

_GRID: Optional[List[List[float]]] = None


def default_grid() -> List[List[float]]:
    """The xT grid, built once from the engine's own shot model.

    Derived, never hand-typed: every cell is ``XGEngine.calculate_geometric``
    at that cell's centre.  Cached because it is pure data and rebuilding it
    is the expensive part of a reward call.
    """
    global _GRID
    if _GRID is None:
        _GRID = _mm.default_grid()
    return _GRID


def set_grid(grid: Optional[Sequence[Sequence[float]]]) -> None:
    """Install a fitted grid (``momentum_model.fit_grid_from_matches``)."""
    global _GRID
    _GRID = None if grid is None else [list(row) for row in grid]


# ── helpers ─────────────────────────────────────────────────────────────

def event_type_name(ev: Any) -> str:
    """``EventType.PASS`` / ``"PASS"`` / ``"EventType.PASS"`` -> ``"PASS"``."""
    et = getattr(ev, "event_type", "")
    name = getattr(et, "name", None)
    if name:
        return str(name)
    s = str(et)
    return s.rsplit(".", 1)[-1]


def _xy(ev: Any, attr_x: str, attr_y: str) -> Optional[tuple]:
    x = getattr(ev, attr_x, None)
    y = getattr(ev, attr_y, None)
    if x is None or y is None:
        return None
    return float(x), float(y)


def _phi(grid, pt: Optional[tuple], attacks_right: bool) -> float:
    if pt is None:
        return 0.0
    return _mm.zone_value(grid, pt[0], pt[1], attacks_right)


def _xg_of(ev: Any, grid, attacks_right: bool) -> float:
    """Shot quality.  ``ev.xg`` when the engine stamped it, else the zone.

    The fallback is not a guess: the grid cell value IS the engine's own xG
    from that spot, so a shot with no ``xg`` attribute is valued exactly as
    the engine would value a shot taken there.
    """
    xg = getattr(ev, "xg", None)
    if xg is None:
        xg = getattr(ev, "expected_goals", None)
    try:
        val = float(xg) if xg is not None else 0.0
    except (TypeError, ValueError):
        val = 0.0
    if val > 0.0:
        return val
    return _phi(grid, _xy(ev, "location_x", "location_y"), attacks_right)


# ── the reward ──────────────────────────────────────────────────────────

def stakes(score_diff: int, minute: float) -> float:
    """How much this moment's swing is WORTH, relative to a neutral moment.

    WHY THIS EXISTS
    ---------------
    Measured 2026-11-16 (`tools/diag/_indifference_probe.py`): the shipped
    striker's intent distribution moves 0.025 in total variation when the
    scoreline sweeps from -3 to +3, against 0.466 for moving him ten metres.
    The clock is the same: 0.020.  Sensor `[18] score_diff` and `[19]
    minute_norm` were verified to carry the values correctly, so the policy is
    discarding information it has.

    The cause is here.  This module had no reference to the scoreline or the
    clock anywhere — a goal was worth ``GOAL_VALUE`` at 0-0 in the first minute
    and at 0-1 down in the 89th.  The brain ignores the score because the
    objective it was trained on has no score.

    Football does not work that way.  A goal that wins a game is worth more than
    one that makes it 4-0; a goal when you are two down late is worth more still,
    because it is the one that keeps the season alive.  That is not pitch
    geometry and no xT grid can express it, so it enters here, as a multiplier.

    THE SHAPE, AND WHY IT IS A FIRST PASS
    -------------------------------------
    ``late`` ramps from 0 at minute 60 to 1 at minute 90, and the scoreline
    decides the direction: chasing is worth more, cruising is worth less, and a
    level game late is the most valuable of all (one goal decides it).

    These constants are football judgement, not measurement.  They are published
    in ``describe()`` so an export can cite them, and they are deliberately
    mild: the range is 0.65 .. 1.60, because a multiplier that swings the reward
    by an order of magnitude would drown the geometry the reward already gets
    right.  Treat them as the first pass to be tuned against goals-per-situation,
    not as truth.
    """
    late = max(0.0, min(1.0, (float(minute) - 60.0) / 30.0))
    sd = int(score_diff)
    if sd >= 2:
        return 1.0 - 0.35 * late      # the game is won; a goal buys little
    if sd == 1:
        return 1.0 - 0.15 * late      # ahead: protect, do not chase
    if sd == 0:
        return 1.0 + 0.50 * late      # level and late: one goal decides it
    if sd == -1:
        return 1.0 + 0.60 * late      # a goal rescues a point
    return 1.0 + 0.25 * late          # two or more down: worth more, but less likely


def stakes_pair(score_diff: int, minute: float) -> Tuple[float, float]:
    """``(gain_mult, loss_mult)`` — the ASYMMETRIC worth of this moment.

    WHY ``stakes()`` ALONE CANNOT WORK
    ----------------------------------
    ``stakes()`` multiplies the WHOLE return by one number.  Multiplying a set
    of values by a positive constant cannot change their order, so if the
    corpus labels of every intent in a situation are scaled by the same ``k``,
    the preferred intent is *identical* before and after.  Splitting the
    surrogate's cells by scoreline would then produce sub-cells that all rank
    their intents the same way, and the policy would stay indifferent no
    matter how long the collection ran.

    That is not a hypothesis, it is arithmetic: ``k·a > k·b`` exactly when
    ``a > b`` for ``k > 0``.  A uniform multiplier can only change how
    *confidently* an action is preferred, never *which* action is preferred.

    THE FOOTBALL FIX: GAINS AND LOSSES ARE NOT WORTH THE SAME THING
    ---------------------------------------------------------------
    Win probability is not linear in the scoreline — it is convex when you are
    behind and concave when you are ahead.  Two goals down with five minutes
    left, the upside of a chance is worth everything and the downside of a
    turnover is worth almost nothing, because you were losing anyway.  One goal
    up with five minutes left, it is the reverse: a goal settles it and buys
    little, while a turnover in your own third is the counterattack that costs
    the lead.  So the reward must scale POSITIVE returns and NEGATIVE returns
    by different factors, and *that* is what changes the ranking: a long shot
    carries a big upside and a real downside, a safe square ball carries
    almost only upside, so damping gains and amplifying losses makes the safe
    ball win — and reversing both makes the shot win.

    THE SHAPE
    ---------
    ``late`` ramps 0 at minute 60 to 1 at minute 90, as in ``stakes()``.  Bands
    are the player's own frame (see ``brain_integration._player_score_diff``).

    ==================  ============  ============  ==========================
    band                gain          loss          football reading
    ==================  ============  ============  ==========================
    2+ down             1 + 0.25·late 1 - 0.55·late desperate: maximum risk
    1 down              1 + 0.60·late 1 - 0.35·late chasing: a goal rescues it
    level               1 + 0.35·late 1 + 0.20·late one goal decides either way
    1 up                1 - 0.25·late 1 + 0.45·late protecting: turnovers kill
    2+ up               1 - 0.35·late 1 + 0.20·late cushion: the game is won
    ==================  ============  ============  ==========================

    At 88' this reads: 2+ down (1.23, 0.49), 1 down (1.56, 0.67), level
    (1.33, 1.19), 1 up (0.77, 1.42), 2+ up (0.67, 1.19).  A team two down is
    told a turnover is worth half of what it normally is and a goal is worth
    more; a team one up is told the opposite.  That is the shape a manager
    would describe, and it is the shape that makes the same player choose
    differently in the same place.

    These are football judgement, like ``stakes()``, not measurement — and
    they are deliberately mild (0.45 .. 1.60) so the geometry the reward
    already gets right is not drowned.  ``describe()`` publishes them.
    """
    late = max(0.0, min(1.0, (float(minute) - 60.0) / 30.0))
    sd = int(score_diff)
    if sd >= 2:
        return 1.0 - 0.35 * late, 1.0 + 0.20 * late
    if sd == 1:
        return 1.0 - 0.25 * late, 1.0 + 0.45 * late
    if sd == 0:
        return 1.0 + 0.35 * late, 1.0 + 0.20 * late
    if sd == -1:
        return 1.0 + 0.60 * late, 1.0 - 0.35 * late
    return 1.0 + 0.25 * late, 1.0 - 0.55 * late


def threat_return(ev: Any, grid: Optional[Sequence[Sequence[float]]] = None,
                  attacks_right: bool = True,
                  stakes_mult: float = 1.0) -> float:
    """Threat return of ONE event for the team that played it.

    Positive = the team made itself more likely to score, negative = it made
    itself less likely (a backward ball, or a loss in a dangerous place).

    ``stakes_mult`` scales the WHOLE return, not just goals, and that is
    deliberate: when you are chasing a game late, winning the ball forty metres
    up the pitch is worth more *and* losing it there costs more.  Scaling only
    the goal would leave a chasing team's turnovers as cheap as a cruising
    team's.  Default 1.0 reproduces the old behaviour exactly, so every existing
    corpus, report and comparison is untouched until a caller passes something
    else — see ``stakes()`` for what to pass and why.
    """
    if grid is None:
        grid = default_grid()
    et = event_type_name(ev)
    s = float(stakes_mult)

    if et in _GOAL_TYPES:
        return GOAL_VALUE * s
    if et in _MISS_TYPES:
        return -SHOT_OFF_TARGET_FRAC * _xg_of(ev, grid, attacks_right) * s
    if et in _SHOT_ON_TYPES:
        return _xg_of(ev, grid, attacks_right) * s
    if et in _SHOT_OFF_TYPES:
        return SHOT_OFF_TARGET_FRAC * _xg_of(ev, grid, attacks_right) * s

    start = _xy(ev, "location_x", "location_y")
    end = _xy(ev, "end_x", "end_y")
    done = bool(getattr(ev, "outcome", False))

    if et in _MOVER_TYPES:
        if done:
            return (_phi(grid, end, attacks_right) -
                    _phi(grid, start, attacks_right)) * s
        # Lost here. Charge what the OPPONENT now has, from where the ball
        # died (or where it was played from, if the event records no end).
        pt = end or start
        opp = _mm.zone_value(grid, pt[0], pt[1], not attacks_right) if pt else 0.0
        return -LOSS_WEIGHT * opp * s

    if et in _LOSS_TYPES:
        opp = _mm.zone_value(grid, start[0], start[1], not attacks_right) \
            if start else 0.0
        return -LOSS_WEIGHT * opp * s

    return 0.0


def describe() -> Dict[str, Any]:
    """The reward definition, for a manifest or an export."""
    return {
        "reward": "threat_return",
        "form": "PHI(end) - PHI(start) for ball movers; absolute PHI for shots",
        "potential": "momentum_model xT grid derived from XGEngine",
        "goal_value": GOAL_VALUE,
        "shot_off_target_frac": SHOT_OFF_TARGET_FRAC,
        "loss_weight": LOSS_WEIGHT,
        "default": 0.0,
        "stakes": {
            "form": "1 +- k(late) by score_diff, multiplies the whole return",
            "range": [0.65, 1.60],
            "measured": ("policy moves 0.025 TV for score_diff -3..+3 vs 0.466 "
                         "for +-10 m of geometry (2026-11-16) — the reward had "
                         "no scoreline at all"),
            "note": ("A UNIFORM multiplier cannot change which intent is "
                     "preferred: k*a > k*b exactly when a > b. It is kept for "
                     "reporting; the corpus labels use `stakes_pair`."),
        },
        "stakes_pair": {
            "form": "separate gain and loss multipliers by score_diff band",
            "at_88": {str(sd): [round(stakes_pair(sd, 88.0)[0], 4),
                                round(stakes_pair(sd, 88.0)[1], 4)]
                      for sd in (-2, -1, 0, 1, 2)},
            "why": ("chasing is convex (upside up, downside down), protecting "
                    "is concave; that asymmetry is what lets a scoreline "
                    "change a preference"),
        },
        "notes": (
            "Failures are charged the opponent's threat at the loss point, "
            "not a penalty on an intent. No diversity/dominance penalty."
        ),
    }
