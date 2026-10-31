"""THE GOAL IS A MOUTH, NOT A DISTANCE (2026-10-31).

A real footballer does not read "26 m to goal" and a central-lane flag.  He
sees the goal mouth: how far off his facing line it sits, and how much of his
view it fills.  Two players 26 m out can see the goal under wildly different
angles, and the old 24-d sensor vector could not tell them apart — its only
goal features were a 25 m proximity flag, a distance, and a `|y-34|<18` bit.

These tests pin the replacement, which lives in the two slots that were
previously exact duplicates:

    [2] goal_angle_norm  signed bearing to the goal CENTRE / 90 deg
    [3] goal_mouth_norm  angle the two POSTS subtend / penalty-spot angle

They also pin the invariants that make the swap safe: the vector width is
unchanged, the carrier's position is still in slots 0-1, central_lane is
untouched in slot 15, and the geometry is deterministic.

Run: python -m pytest tests/test_goal_vision.py -q
"""
from __future__ import annotations

import math

import numpy as np

from brain_sensors import (INPUT_SIZE, _goal_geometry, _REFERENCE_MOUTH_ANGLE,
                           extract_sensors)
from football_brain import FootballBrain


GOAL_X = 105.0
POST = 3.66


def _post_angles(x: float, y: float) -> tuple[float, float]:
    """Bearings to the two posts, in radians, in the world frame."""
    a_top = math.atan2(34.0 + POST - y, GOAL_X - x)
    a_bottom = math.atan2(34.0 - POST - y, GOAL_X - x)
    return a_top, a_bottom


def _vec(x: float, y: float, attacks_right: bool = True) -> np.ndarray:
    return extract_sensors(
        None, x, y, [], [], None, False, attacks_right, None, 30,
    )


# ─────────────────────────────────────────────────────────────
# 1. THE ANGLES ARE THE REAL ANGLES
# ─────────────────────────────────────────────────────────────

def test_goal_centre_bearing_matches_the_real_angle():
    """Slot 2 is the true signed bearing to the goal centre / 90 deg."""
    for x, y in [(80.0, 20.0), (80.0, 50.0), (60.0, 10.0), (94.0, 34.0)]:
        s = _vec(x, y)
        true_centre = math.degrees(math.atan2(34.0 - y, GOAL_X - x))
        expected = min(1.0, max(-1.0, true_centre / 90.0))
        assert abs(s[2] - expected) < 1e-12, (
            f"({x},{y}): slot2={s[2]:.4f} but the goal centre is at "
            f"{true_centre:.2f} deg"
        )


def test_goal_mouth_matches_the_post_angle():
    """Slot 3 is the post-to-post visual angle, in penalty-spot units."""
    for x, y in [(80.0, 20.0), (80.0, 50.0), (85.0, 34.0), (100.0, 34.0)]:
        s = _vec(x, y)
        a_top, a_bottom = _post_angles(x, y)
        true_mouth = abs(a_top - a_bottom) / _REFERENCE_MOUTH_ANGLE
        assert abs(s[3] - min(1.0, true_mouth)) < 1e-12, (
            f"({x},{y}): slot3={s[3]:.4f} but the posts subtend "
            f"{math.degrees(abs(a_top - a_bottom)):.2f} deg"
        )


def test_peer_example_positions():
    """The two positions that exposed the gap now read differently.

    (80,20) and (80,50) are the same distance from goal and both used to be
    `central_lane=1`, `near_goal=0`.  They are opposite sides of the goal:
    the bearing sign flips, and the mouth is slightly narrower from the
    wider angle.
    """
    a = _vec(80.0, 20.0)
    b = _vec(80.0, 50.0)

    assert a[2] > 0.0 > b[2], "the goal is on opposite sides of these two"
    assert abs(a[2] - b[2]) > 0.6, "the two bearings must be clearly distinct"
    # Nearly the same mouth (12.77 deg vs 11.91 deg), but NOT identical:
    # the wide view is the narrower one.  ~0.023 in penalty-spot units.
    assert abs(a[3] - b[3]) < 0.05, (a[3], b[3])
    assert a[3] > b[3], "the wider angle sees the narrower mouth"

    # Both were central_lane=1 before, and still are: slot 15 is untouched.
    assert a[15] == 1.0 and b[15] == 1.0

    # The mouth at 26 m is real: ~12-13 degrees from the peer's example.
    a_top, a_bottom = _post_angles(80.0, 20.0)
    width_deg = math.degrees(abs(a_top - a_bottom))
    assert 12.0 < width_deg < 13.0, width_deg


def test_wide_view_collapses_on_the_byline():
    """The further wide you are, the less mouth you can see — the angle's gone."""
    central_spot = _vec(94.0, 34.0)     # penalty spot, dead central
    wide_18 = _vec(94.0, 18.0)          # same depth, 16 m wide
    byline = _vec(104.0, 20.0)          # on the byline

    assert abs(central_spot[3] - 1.0) < 1e-9, "penalty spot is the reference"
    assert central_spot[3] > wide_18[3] > byline[3]
    assert byline[3] < 0.10, f"byline mouth should be nearly shut: {byline[3]}"


def test_same_distance_central_beats_wide():
    """Distance alone cannot decide how good the sight of goal is."""
    central = _vec(79.0, 34.0)
    wide = _vec(95.0, 10.0)
    d_central = math.hypot(105.0 - 79.0, 34.0 - 34.0)
    d_wide = math.hypot(105.0 - 95.0, 34.0 - 10.0)
    assert abs(d_central - d_wide) < 1.0, "fixture must be equal-distance"
    assert central[3] > 2.0 * wide[3], (
        f"central mouth {central[3]:.3f} must dwarf wide {wide[3]:.3f}"
    )


# ─────────────────────────────────────────────────────────────
# 2. THE SWAP IS SAFE
# ─────────────────────────────────────────────────────────────

def test_slots_two_and_three_are_the_old_duplicates():
    """The replaced slots really were redundant, and still carry position."""
    s = _vec(80.0, 20.0)
    # The carrier stands at the ball: position is slots 0-1.
    assert s[0] == 80.0 / 105.0
    assert s[1] == 20.0 / 68.0
    # Slots 2-3 are no longer copies of 0-1.
    assert s[2] != s[0] and s[3] != s[1]
    # Width and central-lane feature are unchanged.
    assert s.shape == (INPUT_SIZE,)
    assert s[15] == 1.0


def test_mirror_symmetry():
    """Mirroring the attack must mirror the angles exactly."""
    right = _vec(80.0, 20.0, attacks_right=True)
    left = _vec(25.0, 48.0, attacks_right=False)   # 105-x, 68-y
    assert abs(right[2] - left[2]) < 1e-12
    assert abs(right[3] - left[3]) < 1e-12


def test_geometry_is_deterministic_and_signed():
    a = _goal_geometry(80.0, 20.0, True)
    b = _goal_geometry(80.0, 20.0, True)
    assert a == b
    assert -1.0 <= a[0] <= 1.0
    assert 0.0 <= a[1] <= 1.0


def test_brain_still_consumes_the_vector():
    """No shape change, no crash: a 24-wide brain still reads 24 inputs."""
    brain = FootballBrain.random(seed=7)
    s = _vec(80.0, 20.0)
    probs = brain.forward(s)
    assert probs.shape == (10,)
    assert np.all(np.isfinite(probs))
    assert abs(float(probs.sum()) - 1.0) < 1e-9


if __name__ == "__main__":
    test_goal_centre_bearing_matches_the_real_angle()
    test_goal_mouth_matches_the_post_angle()
    test_peer_example_positions()
    test_wide_view_collapses_on_the_byline()
    test_same_distance_central_beats_wide()
    test_slots_two_and_three_are_the_old_duplicates()
    test_mirror_symmetry()
    test_geometry_is_deterministic_and_signed()
    test_brain_still_consumes_the_vector()
    print("all goal-vision tests passed")
