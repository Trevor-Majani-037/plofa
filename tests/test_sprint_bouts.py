"""ONE sprint definition, three feeders (2026-11-01).

The activity report credits sprints from three code paths -- off-ball
movement, on-ball possession-episode traces, and set-piece travel. Before this
change each had its own rule and they disagreed by up to 4x: the off-ball
counter assigned a cooldown it never read, the episode trace restarted its
count on every chain, and the set-piece path credited a flat ``1`` to anybody
who moved.

These tests pin the shared rule (`PositionEngine.advance_sprint_bout`):
a sprint is a SUSTAINED effort (>= 1.0 s above 7.0 m/s on a smoothed speed)
followed by a real RECOVERY (>= 5.0 s below it), not a crossing of a line.
"""
from __future__ import annotations

import pytest

from position_engine import PositionEngine

TICK = 0.1


def _feed(pe: PositionEngine, speeds, player="P", dt=TICK):
    """Feed a speed profile. Returns (sprint_credits, hi_credits)."""
    s = h = 0.0
    for v in speeds:
        a, b = pe.advance_sprint_bout(player, v, dt)
        s += a
        h += b
    return s, h


def _hold(v: float, seconds: float):
    return [v] * int(round(seconds / TICK))


# ── the bout itself ───────────────────────────────────────────────────

def test_sustained_run_above_threshold_counts_exactly_once():
    pe = PositionEngine()
    credits, _ = _feed(pe, _hold(8.0, 3.0))
    assert credits == 1.0


def test_short_burst_does_not_count():
    """A 1.0 s burst at 8 m/s is below the sustained window once the speed is
    smoothed, so it is a high-intensity effort but not a sprint bout."""
    pe = PositionEngine()
    credits, _ = _feed(pe, _hold(8.0, 1.0))
    assert credits == 0.0


def test_speed_below_threshold_never_counts():
    pe = PositionEngine()
    credits, _ = _feed(pe, _hold(6.5, 30.0))
    assert credits == 0.0


def test_chatter_across_the_threshold_does_not_stack_sprints():
    """THE DEFECT THIS FILE EXISTS FOR.

    A player hovering around 7.0 m/s used to collect a "sprint" every 0.3 s,
    which is how a winger logged 280 of them with the physics layer on. The
    smoothed speed plus the sustained window must make wobble worth nothing.
    """
    pe = PositionEngine()
    wobble = ([7.2, 6.9] * 30)          # ~6 s of hovering
    credits, _ = _feed(pe, wobble)
    assert credits <= 1.0


def test_fragmented_effort_counts_once():
    """THE OTHER DEFECT (2026-11-07): the window used to RESET on the first
    tick below threshold, so an effort had to be one unbroken second.

    The engine does not produce those. A chase is flushed from ~8 m/s to ~5 by
    a target flip and back inside the same effort, so the sustained window was
    never reachable and 16 of 28 players in a real match recorded sprint TIME
    with zero bouts (Tavish Crane 46.7 s, Finn Larsson 39.7 s, Kwame Asante
    35.3 s).

    This profile is two 0.9 s high-speed pieces separated by a 0.2 s dip that
    genuinely pulls the smoothed speed under 7.0. Old behaviour: 0 bouts.
    Football behaviour: one sprint.
    """
    pe = PositionEngine()
    profile = _hold(8.0, 0.9) + _hold(5.0, 0.2) + _hold(8.0, 0.9)
    credits, _ = _feed(pe, profile)
    assert credits == 1.0, (
        "a fragmented 1.8 s effort above sprint pace was discarded")


def test_fragments_that_never_reach_the_window_still_do_not_count():
    """Accumulating must not turn any twitch into a sprint: two 0.4 s pieces
    are still short of the 1.0 s window (Di Salvo's definition)."""
    pe = PositionEngine()
    profile = _hold(8.0, 0.4) + _hold(5.0, 0.2) + _hold(8.0, 0.4)
    credits, _ = _feed(pe, profile)
    assert credits == 0.0


def test_fragments_recover_into_a_second_bout():
    """A real recovery still separates two efforts, so a busy player does not
    get one bout for the whole match."""
    pe = PositionEngine()
    first, _ = _feed(pe, _hold(8.0, 0.9) + _hold(5.0, 0.2) + _hold(8.0, 0.9))
    assert first == 1.0
    _feed(pe, _hold(0.0, 6.0))          # full recovery
    second, _ = _feed(pe, _hold(8.0, 0.9) + _hold(5.0, 0.2) + _hold(8.0, 0.9))
    assert second == 1.0


def test_recovery_window_must_elapse_before_the_next_bout_counts():
    pe = PositionEngine()
    first, _ = _feed(pe, _hold(8.0, 3.0))
    assert first == 1.0
    # 4 s below threshold is not enough recovery (window is 5 s).
    gap, _ = _feed(pe, _hold(0.0, 4.0))
    assert gap == 0.0
    second, _ = _feed(pe, _hold(8.0, 3.0))
    assert second == 0.0


def test_full_recovery_allows_the_next_bout():
    pe = PositionEngine()
    _feed(pe, _hold(8.0, 3.0))
    _feed(pe, _hold(0.0, 6.0))
    second, _ = _feed(pe, _hold(8.0, 3.0))
    assert second == 1.0


def test_state_persists_across_calls_not_episodes():
    """Bouts live on the engine, keyed by player, so a run split across two
    possession chains is still one sprint."""
    pe = PositionEngine()
    a, _ = _feed(pe, _hold(8.0, 1.5))
    b, _ = _feed(pe, _hold(8.0, 1.5))
    assert a + b == 1.0


def test_players_are_tracked_independently():
    pe = PositionEngine()
    _feed(pe, _hold(8.0, 3.0), player="A")
    # B starts from his own zeroed state, so he must earn his own bout.
    credits, _ = _feed(pe, _hold(8.0, 3.0), player="B")
    assert credits == 1.0


# ── the high-speed band ───────────────────────────────────────────────

def test_high_speed_band_needs_half_a_second():
    pe = PositionEngine()
    _, hi = _feed(pe, _hold(9.5, 3.0))
    assert hi == 1.0


def test_high_speed_band_ignores_a_nine_tick_dip():
    pe = PositionEngine()
    _, hi = _feed(pe, _hold(9.5, 0.5))
    assert hi == 0.0


def test_sprinting_credits_high_speed_too():
    """A long sprint crosses both bands; they are not exclusive counters."""
    pe = PositionEngine()
    credits, hi = _feed(pe, _hold(9.5, 3.0))
    assert credits == 1.0
    assert hi == 1.0


# ── degeneracies ──────────────────────────────────────────────────────

@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_non_advancing_tick_is_a_no_op(dt):
    pe = PositionEngine()
    assert pe.advance_sprint_bout("P", 9.0, dt) == (0.0, 0.0)


def test_negative_speed_is_clamped():
    pe = PositionEngine()
    credits, hi = _feed(pe, _hold(-4.0, 3.0))
    assert credits == 0.0
    assert hi == 0.0
