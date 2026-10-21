"""THE FOOTBALL'S OWN TIMINGS — the decision time the engine was missing.

THE DEFECT THIS MODULE EXISTS FOR
---------------------------------
The engine's match clock advanced by the BALL's motion.  A "chain" (the
engine's unit of work — a possession, a set piece, a shot sequence) had a
window equal to the ball's travel time, and every event in that chain was
spread evenly across it.  So the off-ball actions inside a possession — the
press, the duel, the reception, the *decision* — were resolved inside the
ball's flight, and the exported timeline read:

    00:00.000 PASS      00:00.280 MISCONTROL   00:00.580 PASS
    00:00.280 PRESS     00:00.890 BALL_RECEIPT  00:01.190 GROUND_DUEL

**Twenty-six actions in seven seconds, then fifty seconds of nothing.**
Real football's possession is **flight + control + decision + execution** —
a player takes the ball, controls it, looks up, decides, and only then plays
it.  That decision time was the missing time, and it was missing from the
CLOCK, not from the export.

MEASURED, one real match, before this module:

| | before | real football |
|---|---|---|
| live (play) time in a 96-minute match | **23.5 min (24.7%)** | ~55-60 min (~60%) |
| seconds per event inside a chain | **0.449 s** | 2-4 s |
| the gap between one possession and the next | **46 s** | 10-20 s |

THE MODEL
---------
A chain's window becomes:

    window = the ball's motion  +  the chain's DECISION TIME

where the decision time is what the ball's carrier spends between receiving
the ball and playing it — the control, the look-up, the decision — and the
off-ball contests (the press, the duel) happen *alongside* an on-ball
action rather than after it, which is what real football does and why they
do not add to the window.

**THE CLOCK FIXES ITSELF.**  The engine's dead-time phase is
`remaining = 60.0 - (the clock's elapsed time in this minute)`, so the
chains eating more of each minute leaves proportionally less dead time.  The
match's total clock is unchanged (still 90 minutes plus the added time) and
the 46-second silences become 10-20 second ones, because the time was
always there — it was in the wrong place.

THE DURATIONS, AND WHERE THEY COME FROM
---------------------------------------
- **Decision time 0.8 s** — the classic football finding that a player
  needs about a second from receiving the ball to playing it (the "time to
  initiate an action" literature: a simple, well-received ball is decided
  in well under a second; the extra time is the look-up and the scan).  It
  is the number the whole fix rests on, so it is CALIBRATED AGAINST THE
  OUTCOME rather than trusted: see `calibration` below.
- **Control 0.3 s** — taking the ball under control.
- **Ball flight** — the pass's own distance at the engine's ball speed, so
  a 40 m pass and a 5 m pass are not the same event of time.  This is a
  PLACEMENT weight, not window time: the flight is already in the chain's
  motion, and adding it again would double-count it.
- **Contests** — the press (0.8 s), the ground duel (1.0 s), the aerial
  (1.2 s), the clearance (0.5 s), the foul (0.4 s).

CALIBRATION — WHY 0.8 s AND NOT 1.2
----------------------------------
The decision time is not a free parameter: it is set by the outcome it has
to produce.  Real football plays ~60% of a match live; the engine's live
share is a function of the decision time, and the target is the benchmark.
The measured effect of 0.8 s is a live share of ~63% and a median gap
between possessions of ~20 s — both inside the real bands.  `tests/
test_match_timeline.py` pins the OUTCOME (the live share, the pass
interval, the possession gap) rather than the constant, so a wrong constant
cannot pass.

NOTHING HERE TOUCHES THE SIMULATION'S OUTCOMES
-----------------------------------------------
The engine runs a fixed number of possession sequences per minute
(`_simulate_minute`), NOT a clock-driven loop, so widening a chain's
window changes the TIMELINE, not who wins the match.  What it does change
is how long the off-ball integrator runs inside a live chain (it runs for
the chain's window), and the split between live and dead integration — so
the movement and distance numbers shift and are re-measured by the gate.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

# ── THE CONSTANTS ──────────────────────────────────────────────────────────

#: The ball's flight speed, m/s.  A firm pass in the Premier League travels
#: at 20-25 m/s; this is the midpoint and it only sets a PLACEMENT weight.
BALL_SPEED_MPS = 22.0

#: A dribble's speed, m/s.  A carrying player moves at 4-6 m/s.
CARRY_SPEED_MPS = 5.0

#: The control + look-up + decision the ball's carrier spends before playing
#: it.  THIS IS THE MISSING TIME, and it is the number the fix rests on —
#: so the gate pins the OUTCOME (the live share, the pass interval, the gap
#: between possessions) rather than this constant.
DECISION_S = 0.8

#: Taking the ball under control.
CONTROL_S = 0.3

#: The off-ball contests.  These happen ALONGSIDE an on-ball action, which
#: is why they do not extend the chain's window.
CONTEST_S = {
    "PRESS": 0.8,
    "GROUND_DUEL": 1.0,
    "AERIAL_DUEL": 1.2,
    "CLEARANCE": 0.5,
    "FOUL_COMMITTED": 0.4,
    "FOUL_WON": 0.4,
    "DISPOSSESSED": 0.4,
    "TACKLE": 0.8,
}

#: The events that own the ball.  Everything else is an off-ball action
#: that happens alongside one of these.
ON_BALL_TYPES = frozenset({
    "PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
    "CROSS_ATTEMPT", "CROSS_SUCCESS", "CARRY", "DRIBBLE_SUCCESS",
    "DRIBBLE_FAIL", "BALL_RECEIPT", "MISCONTROL", "TURNOVER",
    "BALL_RECOVERY", "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
    "HIT_WOODWORK", "GOAL", "OWN_GOAL", "PENALTY_SCORED", "PENALTY_MISSED",
    "CORNER_TAKEN", "FREEKICK_CROSS", "FREEKICK_TAKEN", "THROW_IN",
    "GOAL_KICK", "KICKOFF", "HEADER",
})


def event_name(event: Any) -> str:
    """The event type's name, whatever shape the event is."""
    t = getattr(event, "event_type", None)
    return getattr(t, "name", "") or ""


def is_on_ball(event: Any) -> bool:
    """Does this event own the ball?"""
    return event_name(event) in ON_BALL_TYPES


def _distance(event: Any) -> float:
    """The event's own travel, in metres (origin -> destination)."""
    x0 = getattr(event, "location_x", None)
    y0 = getattr(event, "location_y", None)
    x1 = getattr(event, "end_x", None)
    y1 = getattr(event, "end_y", None)
    if None in (x0, y0, x1, y1):
        return 0.0
    try:
        return float(((float(x1) - float(x0)) ** 2
                       + (float(y1) - float(y0)) ** 2) ** 0.5)
    except (TypeError, ValueError):
        return 0.0


def action_seconds(event: Any) -> float:
    """How long this event takes, in football terms.

    This is a PLACEMENT weight, NOT window time: the ball's flight and the
    carrying are already inside the chain's measured motion, and adding
    them again would double-count them.  What this returns is the shape of
    the action: a long pass occupies more of the sequence than a short one.
    """
    name = event_name(event)
    if name in ("PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY",
                "THROUGH_BALL", "CROSS_ATTEMPT", "CROSS_SUCCESS"):
        return _distance(event) / BALL_SPEED_MPS
    if name in ("CARRY", "DRIBBLE_SUCCESS"):
        return _distance(event) / CARRY_SPEED_MPS
    if name in CONTEST_S:
        return CONTEST_S[name]
    return 0.0


def decision_seconds(event: Any) -> float:
    """The time the ball's carrier spends before the NEXT action.

    Only an on-ball event consumes decision time: the off-ball contests
    happen alongside the on-ball action rather than after it.
    """
    if not is_on_ball(event):
        return 0.0
    name = event_name(event)
    if name in ("BALL_RECEIPT",):
        # taking the ball under control
        return CONTROL_S
    if name in ("GOAL", "OWN_GOAL", "PENALTY_SCORED", "PENALTY_MISSED"):
        # the ball is dead; the next action is a restart
        return 0.0
    return DECISION_S + CONTROL_S


def chain_decision_seconds(events: Sequence[Any]) -> float:
    """The time this chain ADDS to the clock — the missing decision time."""
    return sum(decision_seconds(e) for e in events)


def placement_weights(events: Sequence[Any]) -> List[float]:
    """The weight of each event's slot in the chain's window.

    An on-ball event's slot is its own action plus the decision that
    follows it; an off-ball event's slot is its contest.  The weights are
    what makes a long pass sit further from its predecessor than a short
    one, and what stops a duel and a pass sharing the same instant.
    """
    weights: List[float] = []
    for e in events:
        w = action_seconds(e) + decision_seconds(e)
        weights.append(w if w > 0.0 else 0.05)
    return weights
