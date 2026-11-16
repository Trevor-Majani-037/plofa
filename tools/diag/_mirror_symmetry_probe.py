"""IS THE PLAYER'S VIEW MIRROR-SYMMETRIC?

WHY THIS IS A PRE-FLIGHT AND NOT A NICETY
-----------------------------------------
The collection corpus was built with ``attacks_right`` hard-wired to True by an
off-by-one (see ``surrogate_collect._decode_situational_args``).  That means the
``attacks_right=False`` code path has **never been exercised on a single
corpus row in the project's history**.

A recollected corpus will be half away-team decisions.  If the sensor
extraction, the role block or the goal geometry is not symmetric under a
rotation, then the away half of that corpus is silently wrong — and the only
symptom would be a brain that is subtly worse for half of every match.  That is
not a failure you would notice after 80 minutes; it is a failure you would
notice months later, or never.

WHAT "SYMMETRIC" MEANS HERE, EXACTLY
------------------------------------
Attacking left is the same situation as attacking right, ROTATED 180 degrees:

    x' = 105 - x        (pitch is 105 x 68 -- NOT 100; see below)
    y' = 68 - y
    every teammate and defender likewise
    attacks_right -> not attacks_right

Every RELATIVE quantity is unchanged: distance to the goal, the angle the mouth
subtends, distance to the nearest defender, whether the carrier is central.  So
every perspective-relative feature must come out **identical**.  Only the two
ABSOLUTE position slots (ball_x, ball_y) may move, because where the ball is on
the pitch genuinely does change.

**THE PITCH LENGTH IS 105, AND GETTING THIS WRONG IS ITSELF A TEST FAILURE.**
The first version of this probe mirrored at 100 and reported ``is_final_third``
and ``is_own_half`` as asymmetric on 12-15 % of states.  They are not:
``brain_sensors`` uses ``x > 70`` for attacking right and ``x < 35`` for
attacking left, and ``105 - 70 = 35`` exactly.  A mirror at the wrong length
manufactures the very bug it is looking for.

**THE PLAYERS ARE BUILT ONCE AND SHARED BY BOTH SCENES.**  An earlier version
built each scene separately with the same ``rng``, so the two halves got
different ROLE LABELS for the same coordinates (``_role_coords_to_players``
draws them from ``rng``) and the probe reported 13 of 24 slots as asymmetric —
most of it its own noise.  Only the COORDINATES are mirrored.

WHAT IT FOUND (2026-11-16)
--------------------------
Two genuine defects, both fixed:

* ``brain_sensors._goal_geometry`` measured the mouth as ``abs(a_top -
  a_bottom)`` without wrapping the angle difference, so for the away team —
  where ``dx`` is negative and both post angles sit near +-pi — the value
  clamped to 1.0.  An away striker 20 m out DEAD CENTRAL read the mouth as
  1.000 (penalty-spot wide) instead of 0.563.
* ``situation_tokens`` flipped ``dx`` for the away team but not ``dy``, so the
  signed lateral feature was measured in absolute y and the two teams saw
  opposite conventions for the same relative geometry.

Usage:
    .venv/Scripts/python.exe tools/diag/_mirror_symmetry_probe.py --states 400
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from brain_evolution import (  # noqa: E402
    _DummyPositionEngine, _carrier, _role_coords_to_players,
    _OUTFIELD_TM_POOL, _OPP_POOL, random_game_state,
)
from brain_sensors import extract_sensors  # noqa: E402
from situation_tokens import entity_tokens  # noqa: E402
from situational_context import situation_blocks  # noqa: E402

#: Slots that are ABSOLUTE and therefore MUST change under a rotation.
#: Every other slot is perspective-relative and must be identical; anything
#: outside this set that disagrees is a bug.
ABSOLUTE_SLOTS = {0: "ball_x", 1: "ball_y"}

#: ``brain_sensors`` normalises by 105 and thresholds the thirds at 70 / 35,
#: i.e. ``105 - 70 == 35``.  Mirrored here at the same length or the probe
#: invents an asymmetry that is not in the code.
PITCH_X, PITCH_Y = 105.0, 68.0


def _mirror_coords(coords):
    return [(PITCH_X - x, PITCH_Y - y) for x, y in coords]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=400)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--position", default="CM")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    diffs, role_diffs = [], []
    token_row, token_set = [], []

    for _ in range(args.states):
        st = random_game_state(rng, args.position)
        x, y = st["x"], st["y"]
        tm, df = st["teammates"], st["defenders"]
        ar = st["attacks_right"]

        coords_a = {f"t{i}": p for i, p in enumerate(tm)}
        coords_a.update({f"d{i}": p for i, p in enumerate(df)})
        pe_a = _DummyPositionEngine(coords_a)

        mtm, mdf = _mirror_coords(tm), _mirror_coords(df)
        coords_b = {f"t{i}": p for i, p in enumerate(mtm)}
        coords_b.update({f"d{i}": p for i, p in enumerate(mdf)})
        pe_b = _DummyPositionEngine(coords_b)

        players_t = _role_coords_to_players(tm, "t", _OUTFIELD_TM_POOL, rng)
        players_d = _role_coords_to_players(df, "d", _OPP_POOL, rng)
        carrier = _carrier(args.position)

        s_a = extract_sensors(None, x, y, players_t, players_d, pe_a,
                              st["under_pressure"], ar, st["game_state"],
                              st["minute"])
        role_a, _, _ = situation_blocks(
            carrier, x, y, players_t, players_d, pe_a, ar, s_a)
        tok_a, _ = entity_tokens(
            carrier, x, y, players_t, players_d, pe_a, ar)

        mx, my = PITCH_X - x, PITCH_Y - y
        s_b = extract_sensors(None, mx, my, players_t, players_d, pe_b,
                              st["under_pressure"], not ar, st["game_state"],
                              st["minute"])
        role_b, _, _ = situation_blocks(
            carrier, mx, my, players_t, players_d, pe_b, not ar, s_b)
        tok_b, _ = entity_tokens(
            carrier, mx, my, players_t, players_d, pe_b, not ar)

        diffs.append(np.abs(np.asarray(s_a, float) - np.asarray(s_b, float)))
        role_diffs.append(np.abs(np.asarray(role_a, float) -
                                 np.asarray(role_b, float)))
        if tok_a.shape == tok_b.shape:
            ma = np.asarray(tok_a, float)
            mb = np.asarray(tok_b, float)
            # Attention is a SUM over the set, so the meaningful equivalence is
            # the MULTISET of tokens, not the row order.  Equidistant entities
            # can land in different rows between the two scenes without
            # changing what the attention can see.
            #
            # ROWS CANNOT SIMPLY BE SORTED AND PAIRED.  Lexsort orders by
            # column 0, then 1, then ..., so once the columns being compared
            # are the ones that differ, the pairing becomes circular and
            # reports a difference it created itself.  Measured: sorting
            # reported a 1.088 gap difference while a direct call to
            # `_nearest_opponent_gap` on the same pair returned 9.5679 for
            # BOTH scenes.  So each COLUMN is compared independently, sorted —
            # a necessary condition for multiset equality that needs no pairing
            # at all.
            token_row.append(float(np.abs(ma - mb).mean()))
            per_col = [float(np.abs(np.sort(ma[:, c]) -
                                    np.sort(mb[:, c])).max())
                       for c in range(ma.shape[1])]
            token_set.append(max(per_col))

    D = np.asarray(diffs)
    R = np.asarray(role_diffs)
    worst = D.max(axis=0)
    bad = [i for i, v in enumerate(worst)
           if v > args.tol and i not in ABSOLUTE_SLOTS]

    print(f"\n=== MIRROR SYMMETRY — {args.states} states, position {args.position} ===")
    print(f"sensor vector: {D.shape[1]}-d   max disagreement {D.max():.3e}")
    print(f"role block:    {R.shape[1]}-d   max disagreement {R.max():.3e}")
    if token_row:
        print(f"entity tokens: row-wise mean {max(token_row):.3e}   "
              f"as a SET (sorted) max {max(token_set):.3e}")
    moving = [i for i, v in enumerate(worst)
              if v > args.tol and i in ABSOLUTE_SLOTS]
    if moving:
        print("absolute slots that moved (expected): "
              + ", ".join(f"[{i}] {ABSOLUTE_SLOTS[i]}" for i in moving))
    print()
    if not bad:
        print("SYMMETRIC — every perspective-relative slot is identical.")
        print("An away-team corpus row is therefore the same kind of object as")
        print("a home-team one, and the two halves can be pooled.")
    else:
        print(f"*** {len(bad)} of {D.shape[1]} perspective-relative slots "
              f"DISAGREE under mirror ***")
        for i in bad:
            print(f"    slot [{i:2d}]  max diff {worst[i]:.4f}  "
                  f"mean diff {D[:, i].mean():.4f}")
        print("\nA perspective-relative slot that differs is a BUG: half of any")
        print("recollected corpus would be wrong, and the long run must not start.")

    if token_set and max(token_set) > 1e-6:
        print(f"\n*** the token SET still differs by {max(token_set):.3e} — "
              f"a feature is not direction-relative ***")


if __name__ == "__main__":
    main()
