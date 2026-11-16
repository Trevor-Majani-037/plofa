"""A shoot decision must be answered by the shot it caused — including a goal.

THE BUG (found 2026-11-12, and it is the reason the brains never learned to shoot)
---------------------------------------------------------------------------------
``_correlate_outcomes_causal`` contains a carefully documented branch: a SHOOT
policy intent makes the possession chain emit a ZERO-LENGTH marker carry whose
threat change is exactly 0.0, and the real shot is fired by the ATTACK chain.
So the correlator looks FORWARD in the match timeline for the shot the decision
caused.

That branch is guarded by ``and timeline is not None`` — and the production call
site, ``surrogate_rebuild._run_matches``, passed **no timeline**. The branch was
dead code. Every shoot decision was therefore joined to the marker carry and
credited exactly 0.0, while the shot's xG and any goal were credited to nobody.

Measured on the 9,109-sample corpus collected through that call site:
no sample anywhere exceeded reward 0.3841, ``GOAL_VALUE`` is 1.0, 36.5% of shoot
samples were exactly 0.0 and the mean was +0.0009. The surrogate could not
express "shooting is worth it", so the GA faithfully learned not to shoot.

These tests pin the mechanism itself, so the wiring cannot silently rot again.
"""
from types import SimpleNamespace

import pytest

from surrogate_collect import _correlate_outcomes_causal, DecisionRecord
from decision_brain import PlayerIntent
from threat_reward import GOAL_VALUE


def _ev(kind, player="Striker", minute=10.0, team="Home", **kw):
    """A minimal event: ``event_type_name`` reads ``.event_type.name``."""
    et = SimpleNamespace(name=kind)
    base = dict(event_type=et, player=player, minute=minute, team=team,
                location_x=80.0, location_y=34.0, end_x=80.0, end_y=34.0,
                outcome=True)
    base.update(kw)
    return SimpleNamespace(**base)


def _rec(intent, player="Striker", minute=10.0, position="ST", order=0):
    import numpy as np
    return DecisionRecord(
        player=player, position=position, minute=minute, order=order,
        sensors=np.zeros(24), intent=intent,
    )


def test_a_goal_reaches_the_shot_decision():
    """The whole point: scoring must be worth GOAL_VALUE to the shooter."""
    marker = _ev("CARRY", end_x=80.0)          # zero-length marker carry
    goal = _ev("GOAL")
    pairs = _correlate_outcomes_causal(
        [([_rec(PlayerIntent.SHOOT)], [marker])],
        timeline=[marker, goal],
    )
    assert len(pairs) == 1
    assert pairs[0][2] == pytest.approx(GOAL_VALUE), (
        "the goal was not credited to the shoot decision — this is the bug")


def test_without_a_timeline_the_shot_is_credited_nothing():
    """The historical behaviour, isolated: 0.0, always.

    Kept as a test on purpose.  It documents what the broken call site did, so
    the difference between the two call shapes is a fact in the suite rather
    than a claim in a comment.
    """
    marker = _ev("CARRY", end_x=80.0)
    pairs = _correlate_outcomes_causal([([_rec(PlayerIntent.SHOOT)], [marker])])
    assert pairs[0][2] == pytest.approx(0.0)


def test_a_shot_on_target_is_worth_its_xg():
    marker = _ev("CARRY", end_x=80.0)
    shot = _ev("SHOT_ON_TARGET", location_x=88.0, location_y=34.0)
    pairs = _correlate_outcomes_causal(
        [([_rec(PlayerIntent.SHOOT)], [marker])], timeline=[marker, shot])
    assert pairs[0][2] > 0.0, "a shot on target must be worth more than nothing"


def test_a_shot_by_another_player_is_not_claimed():
    """The minute+player bound is what stops this becoming a proximity join."""
    marker = _ev("CARRY", end_x=80.0)
    other = _ev("GOAL", player="Someone Else")
    pairs = _correlate_outcomes_causal(
        [([_rec(PlayerIntent.SHOOT)], [marker])], timeline=[marker, other])
    assert pairs[0][2] != pytest.approx(GOAL_VALUE), (
        "another player's goal was credited to this shooter")


def test_a_goal_in_a_later_minute_is_not_claimed():
    marker = _ev("CARRY", end_x=80.0, minute=10.0)
    late = _ev("GOAL", minute=40.0)
    pairs = _correlate_outcomes_causal(
        [([_rec(PlayerIntent.SHOOT, minute=10.0)], [marker])],
        timeline=[marker, late])
    assert pairs[0][2] != pytest.approx(GOAL_VALUE)


def test_one_goal_cannot_be_claimed_twice():
    marker_a = _ev("CARRY", end_x=80.0)
    marker_b = _ev("CARRY", end_x=81.0)
    goal = _ev("GOAL")
    pairs = _correlate_outcomes_causal(
        [([_rec(PlayerIntent.SHOOT)], [marker_a]),
         ([_rec(PlayerIntent.SHOOT)], [marker_b])],
        timeline=[marker_a, goal, marker_b],
    )
    claimed = sum(1 for p in pairs if abs(p[2] - GOAL_VALUE) < 1e-9)
    assert claimed == 1, f"the same goal paid out {claimed} times"


def test_the_production_call_site_passes_a_timeline():
    """A source assertion, because the failure was a missing ARGUMENT.

    Every test above passes whether or not ``_run_matches`` wires it up — the
    correlator was always correct, and it was never given a timeline.
    """
    import inspect
    import surrogate_rebuild as sr
    src = inspect.getsource(sr._run_matches)
    assert "timeline=" in src, (
        "_run_matches no longer passes a timeline — shoot decisions are being "
        "credited 0.0 again")
