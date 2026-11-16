"""The stakes multiplier: a goal is not worth the same in every minute.

Measured 2026-11-16: the shipped striker's intent distribution moves 0.025 in
total variation when the scoreline sweeps from -3 to +3, against 0.466 for
moving him ten metres.  The clock moves it 0.020.  The sensors carry both
values correctly — the policy discards them, because `threat_reward` had no
scoreline anywhere in it.

These tests pin three things:

  * the shape is what it claims to be (chasing > level > cruising, all of it
    increasing with the clock);
  * the default is EXACTLY the old behaviour, so no existing corpus, report or
    comparison silently changes;
  * the multiplier reaches every branch of the reward, not just goals — a
    chasing team's turnovers have to cost more too, or the model is incoherent.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threat_reward as tr  # noqa: E402


# ─────────────────────────────────────────────────────────────
# THE SHAPE
# ─────────────────────────────────────────────────────────────

def test_a_neutral_moment_is_exactly_one():
    """Before the 60th minute nothing is at stake beyond the game itself."""
    for sd in (-3, -1, 0, 1, 3):
        assert tr.stakes(sd, 10.0) == 1.0
        assert tr.stakes(sd, 60.0) == 1.0


def test_chasing_beats_level_beats_cruising():
    late = 89.0
    assert tr.stakes(-1, late) > tr.stakes(0, late) > tr.stakes(1, late) \
        > tr.stakes(2, late)


def test_urgency_rises_with_the_clock_while_chasing_and_falls_while_cruising():
    """Monotone in minute for every scoreline, but not always the same way.

    The first version of this test asserted "rises with the clock" for every
    scoreline and failed on +2 — correctly, because a team two goals up late
    should value another goal LESS, not more.  The property worth pinning is
    that the direction is decided by the scoreline and never reverses within it.
    """
    for sd in (-3, -1, 0):
        vals = [tr.stakes(sd, m) for m in (0, 60, 70, 80, 90)]
        assert vals == sorted(vals), f"score_diff {sd} should rise: {vals}"
    for sd in (1, 3):
        vals = [tr.stakes(sd, m) for m in (0, 60, 70, 80, 90)]
        assert vals == sorted(vals, reverse=True), \
            f"score_diff {sd} should fall: {vals}"


def test_the_range_is_bounded():
    """A multiplier that swings by an order of magnitude would drown geometry."""
    lo = min(tr.stakes(sd, 90.0) for sd in range(-5, 6))
    hi = max(tr.stakes(sd, 90.0) for sd in range(-5, 6))
    assert lo >= 0.60 and hi <= 1.70, (lo, hi)


# ─────────────────────────────────────────────────────────────
# THE DEFAULT MUST NOT CHANGE ANYTHING
# ─────────────────────────────────────────────────────────────

class _Ev:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _goal():
    return _Ev(event_type="GOAL", location_x=100.0, location_y=34.0,
               end_x=105.0, end_y=34.0, outcome=True)


def _pass(x0, y0, x1, y1, done=True):
    return _Ev(event_type="PROGRESSIVE_PASS", location_x=x0, location_y=y0,
               end_x=x1, end_y=y1, outcome=done)


def test_the_default_reproduces_the_old_reward_exactly():
    grid = tr.default_grid()
    evs = [
        _goal(),
        _pass(40, 34, 70, 34),
        _pass(70, 34, 50, 30),                 # backward
        _pass(60, 34, 62, 34, done=False),     # lost
        _Ev(event_type="SHOT", location_x=95, location_y=34,
            end_x=105, end_y=34, outcome=True, xg=0.3),
    ]
    for ev in evs:
        assert tr.threat_return(ev, grid, True) == \
            tr.threat_return(ev, grid, True, stakes_mult=1.0)


def test_a_goal_scales_exactly_with_the_multiplier():
    grid = tr.default_grid()
    base = tr.threat_return(_goal(), grid, True)
    assert base == pytest.approx(tr.GOAL_VALUE)
    assert tr.threat_return(_goal(), grid, True, stakes_mult=1.6) == \
        pytest.approx(tr.GOAL_VALUE * 1.6)


def test_the_multiplier_reaches_the_loss_branch_too():
    """A chasing team's turnover must cost more than a cruising team's.

    Scaling only the goal would leave the chasing team's mistakes as cheap as
    anyone else's, which is the opposite of how a team chasing a game behaves.
    """
    grid = tr.default_grid()
    lost = _pass(60, 34, 62, 34, done=False)
    plain = tr.threat_return(lost, grid, True)
    chased = tr.threat_return(lost, grid, True, stakes_mult=1.6)
    assert plain < 0
    assert chased < plain, "a turnover did not get more expensive under stakes"


def test_every_branch_scales_monotonically():
    """Each branch must move in the direction its own sign implies.

    A gain grows with stakes; a loss grows MORE NEGATIVE with stakes.  The first
    version of this test asserted plain ascending order and failed on the
    turnover branch, whose values are negative — the code was right and the
    test was checking the wrong direction.
    """
    grid = tr.default_grid()
    evs = [
        _goal(),
        _pass(40, 34, 70, 34),
        _pass(60, 34, 62, 34, done=False),
        _Ev(event_type="SHOT", location_x=95, location_y=34,
            end_x=105, end_y=34, outcome=True, xg=0.3),
    ]
    for ev in evs:
        vals = [tr.threat_return(ev, grid, True, stakes_mult=s)
                for s in (0.65, 1.0, 1.6)]
        if vals[1] > 0:
            assert vals == sorted(vals), f"{ev.event_type} does not grow: {vals}"
        else:
            assert vals == sorted(vals, reverse=True), \
                f"{ev.event_type} does not get worse: {vals}"


# ─────────────────────────────────────────────────────────────
# IT IS PUBLISHED
# ─────────────────────────────────────────────────────────────

def test_describe_publishes_the_stakes_block():
    """An export that cites the reward must be able to cite this too."""
    d = tr.describe()
    assert "stakes" in d
    assert d["stakes"]["range"] == [0.65, 1.60]
    assert "2026-11-16" in d["stakes"]["measured"]
