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

from typing import Any, Dict, List, Optional, Sequence

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

def threat_return(ev: Any, grid: Optional[Sequence[Sequence[float]]] = None,
                  attacks_right: bool = True) -> float:
    """Threat return of ONE event for the team that played it.

    Positive = the team made itself more likely to score, negative = it made
    itself less likely (a backward ball, or a loss in a dangerous place).
    """
    if grid is None:
        grid = default_grid()
    et = event_type_name(ev)

    if et in _GOAL_TYPES:
        return GOAL_VALUE
    if et in _MISS_TYPES:
        return -SHOT_OFF_TARGET_FRAC * _xg_of(ev, grid, attacks_right)
    if et in _SHOT_ON_TYPES:
        return _xg_of(ev, grid, attacks_right)
    if et in _SHOT_OFF_TYPES:
        return SHOT_OFF_TARGET_FRAC * _xg_of(ev, grid, attacks_right)

    start = _xy(ev, "location_x", "location_y")
    end = _xy(ev, "end_x", "end_y")
    done = bool(getattr(ev, "outcome", False))

    if et in _MOVER_TYPES:
        if done:
            return _phi(grid, end, attacks_right) - _phi(grid, start, attacks_right)
        # Lost here. Charge what the OPPONENT now has, from where the ball
        # died (or where it was played from, if the event records no end).
        pt = end or start
        opp = _mm.zone_value(grid, pt[0], pt[1], not attacks_right) if pt else 0.0
        return -LOSS_WEIGHT * opp

    if et in _LOSS_TYPES:
        opp = _mm.zone_value(grid, start[0], start[1], not attacks_right) \
            if start else 0.0
        return -LOSS_WEIGHT * opp

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
        "notes": (
            "Failures are charged the opponent's threat at the loss point, "
            "not a penalty on an intent. No diversity/dominance penalty."
        ),
    }
