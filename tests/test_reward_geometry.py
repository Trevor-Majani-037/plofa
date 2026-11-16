"""The reward's geometry — the striker can finally tell the goal is there.

THE BUG (measured 2026-11-08)
-----------------------------
``FitnessSurrogate.bucket()`` had six bits — pressure, final third, own half,
space, central, goal-close — and **not one described the goal**.  ``G`` was a
single "close enough" flag, so every strike in the attacking third, from 18 m
dead centre with the mouth open and from the byline with nothing on, landed in
the SAME cell and was averaged into one number.

The consequence was measured, not inferred: in the striker's clearest chance
(``P0FT1OH0S0C1G1``) the table returned DRIBBLE 1.163, SWITCH 1.027,
**SHOOT 0.957**.  The objective was not wrong about football.  It had never
been told where the goal was.

``sensors[2]`` (goal_angle_norm) and ``sensors[3]`` (goal_mouth_norm) were
added on 2026-10-31 for exactly this and were unread by the reward.

THE DESIGN: A LADDER, NOT A WIDER KEY
-------------------------------------
A flat 9-bit key multiplies the cell space by 45, and at ~800 decisions a match
the per-position table is already sparse.  A starved table falls back to
``prior = 0.0``, and a table of zeros teaches the GA *nothing* — a worse and
much quieter failure than the one being fixed.

So the key is hierarchical: rich cell first, 6-bit parent as fallback.  Where
the data can resolve the geometry it does; where it cannot, the player gets
exactly the coarse value he had before.
"""
import numpy as np
import pytest

from surrogate_collect import FitnessSurrogate
from decision_brain import PlayerIntent


def _s(angle=0.0, mouth=0.9, dist=0.25, pressure=0.0, ft=1.0, oh=0.0,
       space=1.0, central=1.0):
    """A 24-d sensor vector with the fields the bucket actually reads."""
    s = np.zeros(24, dtype=np.float64)
    s[2] = angle            # goal_angle_norm, -1..+1
    s[3] = mouth            # goal_mouth_norm, 0..1
    s[9] = space
    s[12] = ft
    s[13] = oh
    s[14] = dist            # goal_distance_norm (0 near, 1 far)
    s[15] = central
    s[16] = pressure
    return s


# ── the geometry bins ────────────────────────────────────────────────────

def test_dead_ahead_and_byline_are_different_bins():
    """``sensors[2]`` is a SIGNED bearing, so the middle bin is dead ahead."""
    ahead = FitnessSurrogate.goal_geometry(_s(angle=0.0))
    byline = FitnessSurrogate.goal_geometry(_s(angle=0.95))
    hard_left = FitnessSurrogate.goal_geometry(_s(angle=-0.95))
    assert ahead.startswith("A2"), "0.0 is dead ahead -> the middle of five bins"
    assert byline.startswith("A4")
    assert hard_left.startswith("A0")
    assert ahead != byline != hard_left


def test_an_open_mouth_differs_from_a_shut_one():
    open_ = FitnessSurrogate.goal_geometry(_s(mouth=0.95))
    shut = FitnessSurrogate.goal_geometry(_s(mouth=0.05))
    assert open_[2:] != shut[2:]
    assert "M2" in open_ and "M0" in shut


def test_distance_is_binned():
    near = FitnessSurrogate.goal_geometry(_s(dist=0.10))
    far = FitnessSurrogate.goal_geometry(_s(dist=0.95))
    assert near.endswith("D2")
    assert far.endswith("D0")


# ── backward compatibility: the old key still exists, unchanged ──────────

def test_the_coarse_key_is_byte_identical_to_the_old_one():
    s = _s()
    assert FitnessSurrogate.bucket(s) == "P0FT1OH0S1C1G1"
    assert "|" not in FitnessSurrogate.bucket(s)


def test_bucket_keys_is_a_ladder_with_the_coarse_parent_as_fallback():
    s = _s()
    rich, coarse = FitnessSurrogate.bucket_keys(s)
    assert coarse == FitnessSurrogate.bucket(s)
    assert rich.startswith(coarse + FitnessSurrogate.RICH_SEP)
    assert FitnessSurrogate.is_rich_key(rich)
    assert not FitnessSurrogate.is_rich_key(coarse)


# ── the ladder actually resolves the bug ─────────────────────────────────

def _fit_two_shot_zones(n_per=40):
    """Two populations that the COARSE key cannot tell apart.

    Identical in all six coarse bits (same third, same lane, same
    goal-close flag, same space, same pressure) and different ONLY in the
    geometry: an 18 m central look with the mouth open, and a byline look
    with nothing on.  Under the old key these are ONE cell.
    """
    data = []
    for _ in range(n_per):
        data.append((_s(angle=0.0, mouth=0.95, dist=0.25),
                     PlayerIntent.SHOOT, 0.55, "SHOOT", "ST"))
    for _ in range(n_per):
        data.append((_s(angle=0.95, mouth=0.05, dist=0.25),
                     PlayerIntent.SHOOT, -0.25, "SHOOT", "ST"))
    return FitnessSurrogate().fit(data)


def test_the_two_zones_share_one_coarse_cell():
    """Proof that the old key could not separate them — the bug, isolated."""
    a = _s(angle=0.0, mouth=0.95, dist=0.25)
    b = _s(angle=0.95, mouth=0.05, dist=0.25)
    assert FitnessSurrogate.bucket(a) == FitnessSurrogate.bucket(b)


def test_the_ladder_separates_them():
    sur = _fit_two_shot_zones()
    good = sur.expected_success(_s(angle=0.0, mouth=0.95, dist=0.25),
                                PlayerIntent.SHOOT, "ST")
    bad = sur.expected_success(_s(angle=0.95, mouth=0.05, dist=0.25),
                               PlayerIntent.SHOOT, "ST")
    assert good > bad, "an 18 m central look must outscore a byline look"
    assert good == pytest.approx(0.55 / sur.scales.get("ST", 1.0), rel=0.05)
    assert bad < 0.0


def test_a_starved_rich_cell_falls_back_to_the_coarse_parent():
    """One sample is below min_samples, but it must NOT be read as nothing."""
    data = [( _s(angle=0.0, mouth=0.95, dist=0.25),
              PlayerIntent.SHOOT, 0.5, "SHOOT", "ST")] * 1
    # 20 samples of other geometry in the same coarse cell
    for _ in range(20):
        data.append((_s(angle=0.9, mouth=0.1, dist=0.25),
                     PlayerIntent.SHOOT, 0.2, "SHOOT", "ST"))
    sur = FitnessSurrogate().fit(data)

    rich, coarse = FitnessSurrogate.bucket_keys(
        _s(angle=0.0, mouth=0.95, dist=0.25))
    # the rich cell exists but is thin; the coarse parent holds everything
    assert rich in sur.table["ST"]
    assert sur.counts["ST"][rich]["SHOOT"] == 1
    assert sur.counts["ST"][coarse]["SHOOT"] == 21

    v = sur.expected_success(_s(angle=0.0, mouth=0.95, dist=0.25),
                             PlayerIntent.SHOOT, "ST")
    assert v != 0.0, "a thin cell must borrow, not vanish into the prior"


def test_both_levels_are_populated_by_fit():
    sur = _fit_two_shot_zones()
    keys = set(sur.table["ST"])
    assert any(FitnessSurrogate.is_rich_key(k) for k in keys)
    assert any(not FitnessSurrogate.is_rich_key(k) for k in keys)


# ── the historical call shape must keep working ──────────────────────────

def test_lookup_still_accepts_a_bare_string_key():
    sur = _fit_two_shot_zones()
    coarse = FitnessSurrogate.bucket(_s())
    v = sur._lookup(coarse, "SHOOT", "ST")
    assert v is not None


def test_unknown_cell_still_returns_the_neutral_prior():
    sur = _fit_two_shot_zones()
    v = sur.expected_success(_s(angle=0.0, mouth=0.95, dist=0.25, oh=1.0,
                                ft=0.0, central=0.0),
                             PlayerIntent.SHOOT, "ST")
    assert v == sur.prior
