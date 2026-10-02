"""Ball-vision tests (2026-09-20): the perceived-ball seam.

  1. Disabled -> TRUE ball (byte-identical to the pre-ball-vision engine).
  2. Visible ball (inside FOV cone + range) -> perceived ~= true.
  3. Ball behind / out of range -> stale "last known" with growing sigma.
  4. Deterministic per (player, snapshot, staleness).
  5. The production off-ball path actually routes through it (e2e counter).

Run:  python -m tests.test_ball_vision

Plain assert-based, matching the project's test style.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

from perception import get_perception_config, set_perception, PerceptionConfig
from ball_vision import perceive_ball, clear_caches


class _State:
    def __init__(self, x, y):
        self.current_x = x
        self.current_y = y


class _Pos:
    def __init__(self):
        self.states = {}
        self.team_rosters = {}
        self.team_attacks_right = {}


class _Engine:
    def __init__(self):
        self.position_engine = _Pos()

    def register(self, name, x, y):
        self.position_engine.states[name] = _State(x, y)


def _player(vision=70.0):
    return SimpleNamespace(
        name="P",
        dna=SimpleNamespace(mental=SimpleNamespace(
            vision=vision, anticipation=70.0, composure=70.0)),
    )


def _run_ball(engine, name, attacks_right, true_x, true_y, minute):
    return perceive_ball(engine, name, _player(), attacks_right,
                         true_x, true_y, minute)


# ─────────────────────────────────────────────────────────────
# 1. Disabled -> TRUE ball
# ─────────────────────────────────────────────────────────────

def test_disabled_returns_true_ball():
    clear_caches()
    old = get_perception_config()
    set_perception(PerceptionConfig(ball_vision=False))
    try:
        e = _Engine()
        e.register("P", 50.0, 34.0)
        px, py, seen, sigma = _run_ball(e, "P", True, 80.0, 34.0, 45.0)
        assert (px, py) == (80.0, 34.0) and seen is True and sigma == 0.0
        # Behind the player too - still the true ball (disabled = identity).
        px, py, _, _ = _run_ball(e, "P", True, 20.0, 30.0, 60.0)
        assert (px, py) == (20.0, 30.0)
    finally:
        set_perception(old)
        clear_caches()


# ─────────────────────────────────────────────────────────────
# 2. Visible ball -> perceived ~= true
# ─────────────────────────────────────────────────────────────

def test_visible_ball_is_true_within_noise():
    clear_caches()
    old = get_perception_config()
    set_perception(PerceptionConfig(ball_vision=True, ball_noise=1.0))
    try:
        e = _Engine()
        e.register("P", 50.0, 34.0)          # facing +x (attacks_right)
        px, py, seen, sigma = _run_ball(e, "P", True, 80.0, 34.0, 45.0)
        assert seen is True
        assert sigma <= 1.0 + 1e-9
        assert abs(px - 80.0) <= 3.0 and abs(py - 34.0) <= 3.0, (px, py)
    finally:
        set_perception(old)
        clear_caches()


# ─────────────────────────────────────────────────────────────
# 3. Ball behind -> stale last-known, sigma grows
# ─────────────────────────────────────────────────────────────

def test_lost_sight_uses_last_known_with_growing_sigma():
    clear_caches()
    old = get_perception_config()
    set_perception(PerceptionConfig(ball_vision=True, ball_noise=1.0,
                                    ball_stale_growth=8.0))
    try:
        e = _Engine()
        e.register("P", 50.0, 34.0)
        # First he SEES it at (80,34) in minute 45 -> tracker seeded.
        _run_ball(e, "P", True, 80.0, 34.0, 45.0)
        # 15s later it's behind him (180deg, out of the cone).
        px, py, seen, sigma1 = _run_ball(e, "P", True, 20.0, 34.0, 60.0)
        assert seen is False, "ball behind must be invisible"
        assert sigma1 >= 120.0, f"1 + 8*15 = 121 expected, got {sigma1}"
        # Staleness grows: two more seconds -> bigger sigma.
        _px, _py, _seen2, sigma2 = _run_ball(e, "P", True, 20.0, 34.0, 62.0)
        assert sigma2 > sigma1, (sigma1, sigma2)
        # The STALE READ'S DISTRIBUTION is centred on the last known (80,34):
        # sample many staleness-bucketed draws -> mean ~= 80, spread ~ sigma.
        xs, ys = [], []
        for i in range(60):
            m = 60.0 + i * 0.05          # distinct staleness bucket each draw
            x, y, _v, _s = _run_ball(e, "P", True, 20.0, 34.0, m)
            xs.append(x)
            ys.append(y)
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        assert abs(mean_x - 80.0) <= 60.0, (mean_x, sigma1)
        assert abs(mean_y - 34.0) <= 60.0, (mean_y, sigma1)
        # And it is NOT centred on the true ball (20,34) - the estimate is
        # genuinely stale, not omniscient.
        assert abs(mean_x - 80.0) < abs(mean_x - 20.0), mean_x
    finally:
        set_perception(old)
        clear_caches()


# ─────────────────────────────────────────────────────────────
# 4. Determinism
# ─────────────────────────────────────────────────────────────

def test_deterministic_per_snapshot():
    clear_caches()
    old = get_perception_config()
    set_perception(PerceptionConfig(ball_vision=True, ball_noise=1.0))
    try:
        e = _Engine()
        e.register("P", 50.0, 34.0)
        _run_ball(e, "P", True, 80.0, 34.0, 45.0)
        a = _run_ball(e, "P", True, 20.0, 34.0, 60.0)
        b = _run_ball(e, "P", True, 20.0, 34.0, 60.0)
        assert a[0] == b[0] and a[1] == b[1], (a, b)
    finally:
        set_perception(old)
        clear_caches()


# ─────────────────────────────────────────────────────────────
# 5. Missing geometry never breaks the match
# ─────────────────────────────────────────────────────────────

def test_missing_geometry_falls_back_to_true_ball():
    clear_caches()
    old = get_perception_config()
    set_perception(PerceptionConfig(ball_vision=True))
    try:
        e = _Engine()                     # no player state at all
        px, py, seen, _ = perceive_ball(e, "Ghost", _player(), True,
                                        80.0, 34.0, 45.0)
        assert (px, py) == (80.0, 34.0) and seen is True
    finally:
        set_perception(old)
        clear_caches()


# ─────────────────────────────────────────────────────────────
# 6. e2e: the production off-ball path routes through it
# ─────────────────────────────────────────────────────────────

def test_offball_production_path_uses_ball_vision():
    from pitch_replay import run_scratch_match
    old = get_perception_config()
    set_perception(PerceptionConfig(ball_vision=True))
    try:
        import ball_vision as bv
        calls = []
        real = bv.perceive_ball

        def counting(*args, **kwargs):
            out = real(*args, **kwargs)
            calls.append(out)
            return out

        bv.perceive_ball = counting
        try:
            run_scratch_match(seed=7, verbose=False)
        finally:
            bv.perceive_ball = real
        assert len(calls) > 0, "production off-ball path never called ball-vision"
    finally:
        set_perception(old)
        clear_caches()


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(123)
    tests = [
        test_disabled_returns_true_ball,
        test_visible_ball_is_true_within_noise,
        test_lost_sight_uses_last_known_with_growing_sigma,
        test_deterministic_per_snapshot,
        test_missing_geometry_falls_back_to_true_ball,
        test_offball_production_path_uses_ball_vision,
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
    print(f"\n{passed}/{len(tests)} ball-vision tests passed.")