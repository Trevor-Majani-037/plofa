"""Off-ball movement-honesty tests (2026-09-20).

The off-ball movement integrator used the TRUE ball for every individual
consumer too.  ``ball_vision.movement_ball`` + the engine seam now run the
CHASE trigger/resustain and the INVOLVEMENT gate off the player's personal
perceived ball (honest effort — wrong belief = wasted/forfeited sprint)
while team SHAPE compaction keeps the true ball.

  1. movement_ball disabled -> true ball AND perceive_ball untouched.
  2. movement_ball enabled  -> the perceived ball.
  3. movements: a player chases a PHANTOM (perceived near / true far) and
     misses the REAL ball (perceived far / true near) — both deterministic.
  4. e2e: the engine's movement integrator calls the seam in a real match.

Run:  python -m tests.test_ball_vision_movement
"""
from __future__ import annotations

import random
from types import SimpleNamespace

from perception import (get_perception_config, set_perception,
                        PerceptionConfig)
import ball_vision as bv
import match_engine as me


class _St:
    def __init__(self, x, y, pos="CM"):
        self.current_x, self.current_y = x, y
        self.position = pos
        self.home_x, self.home_y = x, y
        self.minute_drift_distance = 0.0


class _PE:
    """Minimal PositionEngine-shaped stub for _offball_move_player.

    NOTE: this stub must track the PositionEngine methods that
    ``_offball_move_player`` calls. It had drifted behind Checkpoints 37/38 and
    the live-spacing nudge, so the engine raised AttributeError on entry and
    these ball-vision assertions never actually ran. Every method here returns
    the real engine's *neutral* result, so the target steer is skipped and the
    test isolates the behaviour it is named for.
    """

    def __init__(self, states, teams=None):
        self.states = states
        self.team_rosters = teams or {}
        self.team_attacks_right = {}

    def wide_stretch_blend(self, pname, ball_y):
        return 0.0

    def midfielder_triangle_support(self, team, bx, by, has_ball, ar):
        return {}

    def backline_build_up_support(self, team, bx, by, has_ball, ar=True):
        # Checkpoint 37 (BACK-LINE BUILD-UP DROP). Real contract:
        # {name: (alpha, tx, ty)}, or {} when the situation is not on.
        return {}

    def backline_spread_pressure(self, team, bx, by, has_ball, ar=True):
        # Checkpoint 38 (SPREAD THE PITCH UNDER PRESSURE). Same contract as 37.
        return {}

    def live_spacing_redirect(self, team, cx, cy, tx, ty):
        # Real contract: (newx, newy). Neutral is "leave the target alone".
        return tx, ty


class _Eng:
    """Minimal MatchEngine-shaped stub for _offball_move_player."""

    # Copy the chase/jog tuning constants straight from the real engine.
    for _n in ("_CHASE_TRIGGER", "_CHASE_RAMP", "_CHASE_LETOFF", "_CHASE_BURST_T",
               "_CHASE_RESUSTAIN", "_CHASE_EFFORT", "_JOG_SPEED", "_PRESS_PROB"):
        vars()[_n] = getattr(me.MatchEngine, _n)

    def __init__(self, runner_x=50.0, runner_y=34.0):
        st = _St(runner_x, runner_y)
        self.position_engine = _PE({"T1": st})
        self._minute_start_snapshot = {"T1": (runner_x, runner_y)}
        self._on_ball_this_minute = set()
        self._shape_apply_clock = {}
        self._top_speed_cache = {}
        self._chase_state = {}
        self._patrol = {}
        self._team_press_g_cache = {}
        self.config = SimpleNamespace(home_team="H", away_team="A")
        self.state = SimpleNamespace(match_clock_s=600.0, minute=10,
                                     cross_active=False, cross_team="",
                                     cross_x=0.0, cross_y=0.0)
        self.threat = SimpleNamespace(danger_at=lambda team: 0.0)
        self.moved = []

    def _offball_press_prob(self, *a, **k):
        return None

    def _record_offball_distance(self, name, moved, speed, top, duration_s):
        self.moved.append((name, moved))


def _set(offball_movement):
    old = get_perception_config()
    set_perception(PerceptionConfig(ball_vision=True,
                                    offball_actor_perception=True,
                                    ball_vision_movement=offball_movement))
    return old


def _patch_perceive(px, py):
    bv.perceive_ball = lambda *a, **k: (px, py, True, 0.1)


# ─────────────────────────────────────────────────────────────
# 1. Disabled -> identity, and perceive_ball is NOT touched
# ─────────────────────────────────────────────────────────────

def test_disabled_movement_returns_true_and_skips_perceive():
    old = _set(False)
    hit = []
    try:
        def spy(*a, **k):
            hit.append(a)
            return (999.0, 3.0, True, 1.0)
        bv.perceive_ball = spy
        px, py = bv.movement_ball(object(), "T1", "CM", True, 55.0, 34.0, 10)
        assert px == 55.0 and py == 34.0, "disabled -> identity true ball"
        assert not hit, "disabled must not touch the ball tracker at all"
    finally:
        bv.perceive_ball = _patch_perceive
        set_perception(old)


# ─────────────────────────────────────────────────────────────
# 2. Enabled -> the perceived ball
# ─────────────────────────────────────────────────────────────

def test_enabled_returns_perceived_ball():
    old = _set(True)
    try:
        _patch_perceive(72.5, 19.0)
        px, py = bv.movement_ball(object(), "T1", "CM", True, 55.0, 34.0, 10)
        assert px == 72.5 and py == 19.0, "enabled -> perceived ball"
    finally:
        set_perception(old)


def test_exception_falls_back_to_true_ball():
    old = _set(True)
    try:
        def boom(*a, **k):
            raise RuntimeError("geometry unavailable")
        bv.perceive_ball = boom
        px, py = bv.movement_ball(object(), "T1", "CM", True, 55.0, 34.0, 10)
        assert px == 55.0 and py == 34.0, "error -> true ball, never dies"
    finally:
        bv.perceive_ball = _patch_perceive
        set_perception(old)


# ─────────────────────────────────────────────────────────────
# 3. The movement integrator acts on the PERCEIVED ball
# ─────────────────────────────────────────────────────────────

# Force every chase Bernoulli to `True` so bursts are deterministic.
def _run_tick(eng, true_x, true_y, perceived):
    _patch_perceive(*perceived)
    me._team_press_g = lambda *a, **k: 1.0
    _orig_random = me.random.random
    me.random.random = lambda: 0.0
    try:
        me.MatchEngine._offball_move_player(
            eng, "T1", "H", true_x, true_y, has_ball=False,
            danger_t=0.0, cross_team="", DT=0.1)
    finally:
        me.random.random = _orig_random
    moved = [m for _n, m in eng.moved]
    return moved[-1] if moved else 0.0


def test_movement_misses_true_near_ball_when_belief_is_far():
    old = _set(True)
    try:
        eng = _Eng(50.0, 34.0)
        step = _run_tick(eng, true_x=55.0, true_y=34.0, perceived=(90.0, 34.0))
        # True ball is 5m away (would trigger a 12.5m chase) but the player
        # believes it is 40m away -> not chasing, not involved: a 0.3x trot.
        assert 0.04 < step < 0.08, f"expected loafing trot, got {step}"
    finally:
        set_perception(old)


def test_movement_chases_phantom_when_belief_is_near():
    old = _set(True)
    try:
        eng = _Eng(50.0, 34.0)
        step = _run_tick(eng, true_x=100.0, true_y=34.0, perceived=(52.0, 34.0))
        # True ball is 50m away but the player believes it is 2m away:
        # he sprints at a phantom — the ramped chase burst, not a jog.
        assert step > 0.15, f"expected a chase burst, got {step}"
    finally:
        set_perception(old)


# ─────────────────────────────────────────────────────────────
# 4. e2e: the engine calls the movement seam in a real match
# ─────────────────────────────────────────────────────────────

def test_movement_seam_fires_during_a_real_match():
    from pitch_replay import run_scratch_match
    old = _set(True)
    real = me.movement_ball
    calls = []
    try:
        def counting(engine, pname, position, ar, tx, ty, minute):
            px, py = real(engine, pname, position, ar, tx, ty, minute)
            calls.append((pname, px, py))
            return px, py
        me.movement_ball = counting
        try:
            run_scratch_match(seed=7, verbose=False)
        finally:
            me.movement_ball = real
        n_names = len({c[0] for c in calls})
        assert len(calls) > 0, "movement seam never fired"
        assert n_names >= 10, f"only {n_names} distinct players moved off-ball"
    finally:
        set_perception(old)


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(7)
    tests = [
        test_disabled_movement_returns_true_and_skips_perceive,
        test_enabled_returns_perceived_ball,
        test_exception_falls_back_to_true_ball,
        test_movement_misses_true_near_ball_when_belief_is_far,
        test_movement_chases_phantom_when_belief_is_near,
        test_movement_seam_fires_during_a_real_match,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise SystemExit(1)
    print(f"\n{passed}/{len(tests)} ball-vision movement tests passed.")