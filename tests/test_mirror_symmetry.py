"""The player's view must be the same in both directions.

WHY THIS FILE EXISTS
--------------------
The collection corpus was built with ``attacks_right`` hard-wired to True by an
off-by-one, so the away-team path had **never been exercised on a single corpus
row**.  Before recollecting ~27 matches, the cheapest possible question is:
"would the away half of that corpus even be the same kind of object as the home
half?"

Answering it took minutes and found three defects that an 80-minute run would
have baked in permanently:

1. ``brain_sensors._goal_geometry`` measured the mouth as ``abs(a_top -
   a_bottom)`` without wrapping, so for the away team both post angles sit near
   +-pi and the value clamped to 1.0.  An away striker 20 m out DEAD CENTRAL
   read the goal as penalty-spot wide (1.000) instead of 0.563.
2. ``situation_tokens`` flipped ``dx`` for the away team but not ``dy``, so the
   signed lateral feature used absolute y and the two teams saw opposite
   conventions for the same geometry.
3. ``situation_tokens`` passed the entity list straight into
   ``_nearest_opponent_gap``, but that list holds TUPLES, not players — so
   ``_pos_of`` returned its ``(0, 0)`` fallback and the SPACE feature was
   really "distance to the corner flag".  It looked plausible on the home half
   (the origin correlates with attacking progress) and was never tested on the
   away half.

THE INVARIANT
-------------
Attacking left is attacking right rotated 180 degrees on a 105 x 68 pitch::

    x' = 105 - x,  y' = 68 - y,  attacks_right -> not attacks_right

Every perspective-relative feature must be IDENTICAL.  Only ``ball_x`` and
``ball_y`` may move, because where the ball is on the pitch genuinely changes.
"""
import math
import os
import random
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from brain_evolution import (  # noqa: E402
    _DummyPositionEngine, _carrier, _role_coords_to_players,
    _OUTFIELD_TM_POOL, _OPP_POOL, random_game_state,
)
from brain_sensors import _goal_geometry, extract_sensors  # noqa: E402
from situation_tokens import entity_tokens  # noqa: E402
from situational_context import situation_blocks  # noqa: E402

PITCH_X, PITCH_Y = 105.0, 68.0
ABSOLUTE_SLOTS = {0, 1}


# ─────────────────────────────────────────────────────────────
# THE GOAL MOUTH — the branch-cut bug
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("x,y", [
    (85.0, 34.0),   # 20 m out, dead central  <- the case that broke
    (70.0, 34.0),   # edge of the box, central
    (88.0, 34.0),   # 17 m, central
    (50.0, 34.0),   # halfway, central
    (85.0, 20.0),   # off-centre
    (60.0, 10.0),   # wide and deep
])
def test_goal_geometry_is_the_same_in_both_directions(x, y):
    right = _goal_geometry(x, y, True)
    left = _goal_geometry(PITCH_X - x, PITCH_Y - y, False)
    assert right == pytest.approx(left, abs=1e-12), (
        f"a player at ({x}, {y}) attacking right sees {right}, "
        f"the mirrored player attacking left sees {left}")


def test_a_central_striker_does_not_see_a_penalty_spot_sized_goal():
    """The specific regression: 20 m out dead central used to read 1.000."""
    _, mouth = _goal_geometry(85.0, 34.0, True)
    assert mouth < 0.7, (
        f"20 m out dead central must not read as a penalty-spot view, got {mouth}")
    _, mirrored = _goal_geometry(20.0, 34.0, False)
    assert mirrored == pytest.approx(mouth, abs=1e-12)


def test_the_mouth_shrinks_as_the_angle_goes():
    """Monotonic in distance along the centre line — sanity, not symmetry."""
    near = _goal_geometry(95.0, 34.0, True)[1]
    far = _goal_geometry(50.0, 34.0, True)[1]
    assert near > far


# ─────────────────────────────────────────────────────────────
# THE SPACE FEATURE — measuring the corner flag
# ─────────────────────────────────────────────────────────────

class _P:
    def __init__(self, name, position="CM"):
        self.name = name
        self.position = position


def test_space_is_the_gap_to_the_nearest_marker_not_the_origin():
    """The bug: a 2 m marker used to read as '22 m of space'."""
    carrier = _P("carrier", "CM")
    mate = _P("mate", "CM")
    marker = _P("marker", "CB")
    engine = _DummyPositionEngine({
        "carrier": (60.0, 34.0),
        "mate": (70.0, 34.0),      # 10 m ahead of the carrier
        "marker": (72.0, 34.0),    # 2 m from that teammate
    })

    tokens, mask = entity_tokens(
        carrier, 60.0, 34.0, [mate], [marker], engine, attacks_right=True)

    # The teammate is the first real row; feature 5 is the space feature.
    space = float(np.asarray(tokens)[0][5])
    expected = 2.0 / 15.0
    assert space == pytest.approx(expected, abs=1e-9), (
        f"the gap to a marker 2 m away must read {expected:.3f}, got {space:.3f}. "
        f"Distance to the origin would be ~{(70**2 + 34**2) ** 0.5 / 15:.3f}.")


def test_space_does_not_depend_on_where_the_pitch_corner_is():
    """Two identical relative scenes, far from the origin, must agree."""
    def space_at(cx, cy):
        carrier = _P("carrier", "CM")
        mate = _P("mate", "CM")
        marker = _P("marker", "CB")
        engine = _DummyPositionEngine({
            "carrier": (cx, cy),
            "mate": (cx + 10.0, cy),
            "marker": (cx + 12.0, cy),
        })
        tokens, _ = entity_tokens(
            carrier, cx, cy, [mate], [marker], engine, attacks_right=True)
        return float(np.asarray(tokens)[0][5])

    assert space_at(20.0, 20.0) == pytest.approx(space_at(80.0, 50.0), abs=1e-12)


# ─────────────────────────────────────────────────────────────
# THE WHOLE VIEW
# ─────────────────────────────────────────────────────────────

def _mirrored_pair(seed, position="CM"):
    rng = random.Random(seed)
    st = random_game_state(rng, position)
    x, y = st["x"], st["y"]
    tm, df = st["teammates"], st["defenders"]
    ar = st["attacks_right"]

    ca = {f"t{i}": p for i, p in enumerate(tm)}
    ca.update({f"d{i}": p for i, p in enumerate(df)})
    cb = {f"t{i}": (PITCH_X - p[0], PITCH_Y - p[1]) for i, p in enumerate(tm)}
    cb.update({f"d{i}": (PITCH_X - p[0], PITCH_Y - p[1]) for i, p in enumerate(df)})

    players_t = _role_coords_to_players(tm, "t", _OUTFIELD_TM_POOL, rng)
    players_d = _role_coords_to_players(df, "d", _OPP_POOL, rng)
    carrier = _carrier(position)
    pe_a, pe_b = _DummyPositionEngine(ca), _DummyPositionEngine(cb)

    sa = extract_sensors(None, x, y, players_t, players_d, pe_a,
                         st["under_pressure"], ar, st["game_state"], st["minute"])
    sb = extract_sensors(None, PITCH_X - x, PITCH_Y - y, players_t, players_d, pe_b,
                         st["under_pressure"], not ar, st["game_state"], st["minute"])
    ra, _, _ = situation_blocks(carrier, x, y, players_t, players_d, pe_a, ar, sa)
    rb, _, _ = situation_blocks(carrier, PITCH_X - x, PITCH_Y - y,
                                players_t, players_d, pe_b, not ar, sb)
    ta, _ = entity_tokens(carrier, x, y, players_t, players_d, pe_a, ar)
    tb, _ = entity_tokens(carrier, PITCH_X - x, PITCH_Y - y,
                          players_t, players_d, pe_b, not ar)
    return np.asarray(sa, float), np.asarray(sb, float), \
        np.asarray(ra, float), np.asarray(rb, float), \
        np.asarray(ta, float), np.asarray(tb, float)


@pytest.mark.parametrize("seed", [5, 11, 23])
def test_every_relative_sensor_slot_is_symmetric(seed):
    sa, sb, _, _, _, _ = _mirrored_pair(seed)
    for i in range(len(sa)):
        if i in ABSOLUTE_SLOTS:
            continue
        assert sa[i] == pytest.approx(sb[i], abs=1e-9), (
            f"slot [{i}] differs under mirror: {sa[i]:.4f} vs {sb[i]:.4f}")


@pytest.mark.parametrize("seed", [5, 11, 23])
def test_the_absolute_slots_are_the_only_ones_that_move(seed):
    """A guard against 'fixing' the test by freezing everything."""
    sa, sb, _, _, _, _ = _mirrored_pair(seed)
    assert any(abs(sa[i] - sb[i]) > 1e-6 for i in ABSOLUTE_SLOTS), (
        "ball_x/ball_y did not move — the mirror is not being applied")


@pytest.mark.parametrize("seed", [5, 11, 23])
def test_the_role_block_is_symmetric(seed):
    _, _, ra, rb, _, _ = _mirrored_pair(seed)
    assert np.abs(ra - rb).max() < 1e-9


@pytest.mark.parametrize("seed", [5, 11, 23])
def test_the_entity_token_set_is_symmetric(seed):
    """Compared per column, sorted: attention is a sum over a SET, so the row
    order is not part of the contract and pairing rows would be circular."""
    _, _, _, _, ta, tb = _mirrored_pair(seed)
    for c in range(ta.shape[1]):
        assert np.abs(np.sort(ta[:, c]) - np.sort(tb[:, c])).max() < 1e-9, (
            f"token feature [{c}] differs as a set under mirror")
