"""The scoreline reached the brain as zero. These tests are why it cannot again.

WHY THIS FILE EXISTS
--------------------
Two silent defects were found on 2026-11-16, and neither could be seen from a
unit test of the thing that looked wrong:

1. ``brain_integration._score_diff_of`` mapped the engine's two CRUISE states to
   "level" (a 2-0 lead read as 0-0) and gave the two CHASE states the INVERTED
   sign (a team chasing a 2-goal deficit read as winning).
2. ``surrogate_collect._snapshot_decide`` decoded ``decide``'s positional
   arguments one slot to the right, so the collection corpus recorded
   ``score_diff = 0`` in **100 % of 15,426 rows** and treated every decision —
   the away team's included — as attacking to the right.

Together they explain why the players are indifferent to the score: the data the
brains were distilled from never contained a scoreline at all.  These tests pin
both, using the ENGINE'S OWN call shape rather than a memory of it.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from brain_integration import _player_score_diff, _score_diff_of
from match_engine import GameState
from surrogate_collect import _decode_situational_args

#: What the home perspective must read, for every state the engine can emit.
#: A side that is CHASING is BEHIND, which is why its sign is negative.
TRUTH = {
    "LEVEL": 0,
    "HOME_AHEAD": 1, "AWAY_AHEAD": -1,
    "HOME_CRUISE": 2, "AWAY_CRUISE": -2,
    "HOME_CHASE": -2, "AWAY_CHASE": 2,
}


@pytest.mark.parametrize("member", list(GameState))
def test_every_engine_state_maps_to_the_right_score_difference(member):
    assert _score_diff_of(member) == TRUTH[member.name], (
        f"{member.name} means the home side is {TRUTH[member.name]:+d}, "
        f"but the brain was told {_score_diff_of(member):+d}")


def test_a_two_goal_lead_is_not_level():
    """The live half of the bug: CRUISE was reported as 0-0."""
    assert _score_diff_of(GameState.HOME_CRUISE) != 0
    assert _score_diff_of(GameState.AWAY_CRUISE) != 0


def test_a_chasing_side_is_behind_not_ahead():
    """The inverted half: HOME_CHASE is a deficit, not a lead."""
    assert _score_diff_of(GameState.HOME_CHASE) < 0
    assert _score_diff_of(GameState.AWAY_CHASE) > 0


def test_no_game_state_is_treated_as_level():
    """The collector passed None for every row; that must stay harmless."""
    assert _score_diff_of(None) == 0
    assert _score_diff_of("LEVEL") == 0


def test_a_decorated_state_name_still_resolves():
    """Substring matching is kept so a suffixed name does not fall to 0."""

    class Named:
        name = "HOME_CRUISE_2"

    assert _score_diff_of(Named()) == 2


# ─────────────────────────────────────────────────────────────
# THE COLLECTOR'S ARGUMENT DECODING
# ─────────────────────────────────────────────────────────────

def test_the_engine_call_shape_decodes_correctly():
    """Exactly how ``event_chain.py`` calls ``decide``: nine positionals.

    ``minute`` is a keyword there, which is precisely why the old off-by-one was
    invisible: the shift happened after ``team_profile`` and the fallback for
    ``minute`` landed on the right keyword by luck.
    """
    args = (50.0, 34.0, ["a"], ["b"], "ENGINE", "PROFILE",
            True, False, GameState.HOME_AHEAD)
    under_pressure, attacks_right, game_state, minute = \
        _decode_situational_args(args, {"minute": 72.0})

    assert under_pressure is True
    assert attacks_right is False, (
        "attacks_right decoded from the wrong slot — this is how the corpus "
        "ended up treating every decision as attacking right")
    assert game_state is GameState.HOME_AHEAD, (
        "game_state decoded as None — this is how the scoreline became 0-0 in "
        "every row of the shipped corpus")
    assert minute == 72.0


def test_the_scoreline_survives_the_decoding():
    """The end that matters: what the collector hands to ``perceive``."""
    args = (50.0, 34.0, [], [], None, None, False, True, GameState.AWAY_CRUISE)
    _, _, game_state, _ = _decode_situational_args(args, {"minute": 10.0})
    assert _score_diff_of(game_state) == -2, (
        "a decoded game state must still yield the real score difference")


def test_a_keyword_only_caller_still_works():
    """Some harnesses pass everything by keyword; that path must not break."""
    args = (50.0, 34.0, [], [], None)
    kw = {"under_pressure": True, "attacks_right": False,
          "game_state": GameState.HOME_AHEAD, "minute": 30.0}
    assert _decode_situational_args(args, kw) == (
        True, False, GameState.HOME_AHEAD, 30.0)


# ─────────────────────────────────────────────────────────────
# THE REWARD MUST KNOW THE SCORELINE
# ─────────────────────────────────────────────────────────────

def test_stakes_mult_is_one_when_disabled():
    """The old objective, byte for byte — the escape hatch must be exact."""
    import surrogate_collect as SC

    class Rec:
        score_diff, minute = -2, 88.0

    SC.set_stakes(False)
    try:
        assert SC._stakes_mult(Rec()) == 1.0
    finally:
        SC.set_stakes(True)


def test_a_chasing_side_late_in_the_game_is_worth_more():
    import surrogate_collect as SC

    class Rec:
        def __init__(self, sd, m):
            self.score_diff, self.minute = sd, m

    SC.set_stakes(True)
    early = SC._stakes_mult(Rec(0, 5.0))
    late_level = SC._stakes_mult(Rec(0, 88.0))
    late_chasing = SC._stakes_mult(Rec(-1, 88.0))
    late_cruising = SC._stakes_mult(Rec(2, 88.0))

    assert early == pytest.approx(1.0), "an early level game is the baseline"
    assert late_level > 1.0, "a late level game must matter more than an early one"
    assert late_chasing > late_level, "chasing late is the most valuable moment"
    assert late_cruising < 1.0, "cruising late must matter less, not more"


def test_the_reward_scales_with_the_multiplier():
    """`_event_success` must actually pass it through, not swallow it.

    THE FIXTURE USED TO BE VACUOUS.  It set ``type = "PASS"``, but the reward
    reads ``ev.event_type``, so both sides of the assertion were ``0.0`` and
    ``0.0 * 2 == 0.0`` passed happily.  A reward of zero is exactly what a
    broken event-name lookup returns, so this test was blind to the failure it
    was written to catch.  It now asserts the base is non-zero FIRST.
    """
    import surrogate_collect as SC

    class Ev:
        event_type = "PASS"
        x, y = 60.0, 34.0
        location_x, location_y = 30.0, 34.0
        end_x, end_y = 90.0, 20.0
        success = True
        outcome = True
        team = "H"

    base = SC._event_success(Ev(), attacks_right=True)
    assert base > 0.0, (
        "the fixture must produce a real return; a zero here would make the "
        "scaling assertion vacuous")
    doubled = SC._event_success(Ev(), attacks_right=True, stakes_mult=2.0)
    assert doubled == pytest.approx(base * 2.0, rel=1e-9, abs=1e-12)


# ─────────────────────────────────────────────────────────────
# GATE 8 — THE SCORELINE MUST BE IN THE PLAYER'S OWN FRAME
# ─────────────────────────────────────────────────────────────
#
# ``_score_diff_of`` is a scoreboard: it answers "how far ahead is HOME".
# A decision needs "how far ahead am I".  In this engine ``attacks_right`` IS
# team identity — ``set_squad`` sets it to ``team_name == home_team`` and
# nothing ever swaps ends — so home is always True and away always False.
# Without the flip, an away player 2-0 up read -2, i.e. "I am two goals down",
# and half of every corpus row pointed the wrong way.

def test_a_home_carrier_reads_the_scoreboard_unchanged():
    assert _player_score_diff(GameState.HOME_CRUISE, True) == 2
    assert _player_score_diff(GameState.HOME_CHASE, True) == -2


def test_an_away_carrier_reads_his_own_scoreline():
    """The gate. Away 2-0 up must read +2, not -2."""
    assert _player_score_diff(GameState.AWAY_CRUISE, False) == 2
    assert _player_score_diff(GameState.AWAY_CHASE, False) == -2
    assert _player_score_diff(GameState.AWAY_AHEAD, False) == 1


@pytest.mark.parametrize("member", list(GameState))
def test_the_two_frames_are_exact_mirrors(member):
    """Whatever the scoreboard says, the two shirts read opposite signs."""
    home = _player_score_diff(member, True)
    away = _player_score_diff(member, False)
    assert home == -away, (
        f"{member.name}: home reads {home:+d} and away {away:+d} — a single "
        "shared policy cannot interpret that consistently")


def test_a_late_chase_is_worth_more_to_the_away_side_too():
    """End to end: the away frame must reach the stakes multiplier.

    The property is FRAME EQUIVALENCE, not monotonicity.  ``stakes()`` is
    deliberately not monotonic in the deficit — it publishes ``1 + 0.25·late``
    for "two or more down", below the level-game ``1 + 0.50·late``, because a
    two-goal deficit is the less recoverable one.  So the assertion is that the
    same football situation pays the same from either shirt: a home side 2-0
    down and an away side 2-0 down must land on the same number.
    """
    import surrogate_collect as SC

    class Rec:
        def __init__(self, sd, m):
            self.score_diff, self.minute = sd, m

    SC.set_stakes(True)
    home_two_down = _player_score_diff(GameState.HOME_CHASE, True)    # -2
    away_two_down = _player_score_diff(GameState.AWAY_CHASE, False)   # -2
    away_two_up = _player_score_diff(GameState.AWAY_CRUISE, False)    # +2

    assert home_two_down == away_two_down == -2
    assert SC._stakes_mult(Rec(away_two_down, 88.0)) == pytest.approx(
        SC._stakes_mult(Rec(home_two_down, 88.0))), (
        "the same situation must be worth the same from either end")
    assert SC._stakes_mult(Rec(away_two_down, 88.0)) > 1.0, (
        "chasing from two down in the 88th must matter more than neutral")
    assert SC._stakes_mult(Rec(away_two_up, 88.0)) < 1.0, (
        "cruising to a win in the 88th must matter less than neutral")
    # And the inversion this test exists to prevent: home-frame would have
    # paid the away side the CRUISING rate for chasing.
    assert SC._stakes_mult(Rec(away_two_down, 88.0)) != pytest.approx(
        SC._stakes_mult(Rec(2, 88.0)))


def test_a_truthy_non_bool_attacks_right_is_refused():
    """The exact shape of the old defect: an enum where a bool belongs.

    ``GameState.HOME_AHEAD`` is truthy, so a lenient check would silently skip
    the flip and restore the inversion.  The collector must fail loudly.
    """
    with pytest.raises(TypeError):
        _player_score_diff(GameState.HOME_AHEAD, GameState.HOME_AHEAD)
    with pytest.raises(TypeError):
        _player_score_diff(GameState.LEVEL, None)

