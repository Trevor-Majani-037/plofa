"""The global ball path never moves backwards in time (2026-11-23).

Calibration seeds 1-6 showed timestamp inversions with BOTH 2026-11-23
pass fixes disabled, so this is a pre-existing chronography defect. Three
writers stamped from raw / re-derived clocks: dead-time `rest`, the
loose-ball scramble, and the post-goal walk back to centre (which subtracts
celebration time on purpose). They now share `_append_ball_point`.
"""
from types import SimpleNamespace

from match_engine import MatchEngine


def _engine(last_t=10.0):
    eng = object.__new__(MatchEngine)
    eng.state = SimpleNamespace(match_ball_path=[{"t": last_t}])
    eng._last_path_t = last_t
    return eng


def test_late_writer_is_clamped_past_the_running_cursor():
    eng = _engine()
    eng._append_ball_point(8.3, 52.5, 34.0, "carry", "Home")
    sample = eng.state.match_ball_path[-1]
    assert sample["t"] > 10.0, "a post-goal carry must not rewind the clock"
    assert sample["t"] <= 10.01
    assert (sample["x"], sample["y"], sample["kind"], sample["team"]) == (
        52.5, 34.0, "carry", "Home")


def test_on_time_writer_keeps_its_exact_timestamp():
    eng = _engine()
    eng._append_ball_point(10.5, 20.0, 30.0, "rest", "Away")
    assert eng.state.match_ball_path[-1]["t"] == 10.5


def test_mixed_writers_stay_monotonic():
    eng = _engine()
    for t, kind in ((9.0, "rest"), (10.4, "loose"), (9.7, "carry"),
                    (11.2, "rest"), (11.1, "carry")):
        eng._append_ball_point(t, 0.0, 0.0, kind, "Home")
    times = [p["t"] for p in eng.state.match_ball_path]
    assert times == sorted(times), times
    assert len(times) == len(set(times)), "clamped samples must not collide"


def test_first_writer_without_a_cursor_is_left_alone():
    eng = object.__new__(MatchEngine)
    eng.state = SimpleNamespace(match_ball_path=[])
    eng._last_path_t = None
    eng._append_ball_point(3.0, 1.0, 2.0, "rest", "Home")
    assert eng.state.match_ball_path[-1]["t"] == 3.0
