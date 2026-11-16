"""DOES THE PLAYER CARE? — the indifference test.

THE QUESTION
------------
The user's phrase for what is wrong with these players is "indifference from the
real world": they do not behave like footballers who care about the score or the
clock.  That is a testable claim and it is the premise the whole foresight plan
rests on — a player who cannot see a future has no reason to hurry, and a player
who cannot see the scoreline has no reason to take a risk.

The sensor vector has carried `[18] score_diff` and `[19] minute_norm` since the
v5 schema.  So the information is PRESENT, exactly as the goal geometry was
present before the shot probe found the output layer reading it backwards.  The
question is whether the POLICY moves.

THE TEST
--------
Hold the geometry completely fixed.  Vary ONE context axis at a time.  Measure
the total-variation distance between the resulting intent distribution and the
baseline, and the share of states whose argmax changes.

    TV = 0.5 * sum |p - p_base|        in [0, 1]
    0     = the axis is ignored completely (indifference)
    ~1    = the axis dominates the decision

The CONTROL is a geometry perturbation (the carrier moved 10 m towards goal).
Without it a small number on `score_diff` would be indistinguishable from a
broken harness — the same reason the shot probe needed its distance control.

Usage:
    python tools/diag/_indifference_probe.py --position ST --states 200
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
    _DummyPositionEngine, _OPP_POOL, _OUTFIELD_TM_POOL, _carrier,
    _role_coords_to_players, random_game_state_scoring,
)
from brain_sensors import extract_sensors  # noqa: E402
from football_brain import INTENT_LABELS  # noqa: E402
from situation_tokens import entity_tokens  # noqa: E402
from situational_brain import HIDDEN_D, SituationalBrain, _softmax  # noqa: E402
from situational_context import situation_blocks  # noqa: E402


def _scene(position: str, rng: random.Random):
    """One fixed geometry, plus everything needed to rebuild sensors around it."""
    st = random_game_state_scoring(rng, position)
    t_coords, d_coords = st["teammates"], st["defenders"]
    names = {}
    for i, (tx, ty) in enumerate(t_coords):
        names[f"t{i}"] = (tx, ty)
    for i, (dx, dy) in enumerate(d_coords):
        names[f"d{i}"] = (dx, dy)
    return {
        "st": st,
        "pe": _DummyPositionEngine(names),
        "teammates": _role_coords_to_players(t_coords, "t", _OUTFIELD_TM_POOL, rng),
        "defenders": _role_coords_to_players(d_coords, "d", _OPP_POOL, rng),
        "carrier": _carrier(position),
    }


def _policy(brain, scene, position, x, y, under_pressure, minute, score_diff):
    """The brain's intent distribution for one fully-specified state."""
    st = scene["st"]
    base = extract_sensors(
        None, x, y, scene["teammates"], scene["defenders"], scene["pe"],
        under_pressure, st["attacks_right"], st["game_state"], minute,
        score_diff=score_diff)
    role, mem, plan = situation_blocks(
        scene["carrier"], x, y, scene["teammates"], scene["defenders"],
        scene["pe"], st["attacks_right"], base)
    tok, msk = entity_tokens(
        scene["carrier"], x, y, scene["teammates"], scene["defenders"],
        scene["pe"], st["attacks_right"])
    lg, _plg, _h, _c = brain.forward(
        tok, msk, base, role, mem, plan, np.zeros(HIDDEN_D))
    return _softmax(lg), base


def _compare(brain, scene, position, axis, mutate):
    """Mean TV and argmax-change rate between a mutated state and its baseline."""
    st = scene["st"]
    p0, _ = _policy(brain, scene, position, st["x"], st["y"],
                    st["under_pressure"], 45.0, 0)
    tv, flip = [], []
    for kwargs in mutate:
        p1, _ = _policy(brain, scene, position, **kwargs)
        tv.append(0.5 * float(np.abs(p1 - p0).sum()))
        flip.append(int(p1.argmax() != p0.argmax()))
    return float(np.mean(tv)), float(np.mean(flip))


def main() -> None:
    ap = argparse.ArgumentParser(description="Does the policy respond to context?")
    ap.add_argument("--position", default="ST")
    ap.add_argument("--states", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    brain = SituationalBrain.load(os.path.join("brains", f"{args.position}.json"))
    rng = random.Random(args.seed)
    scenes = [_scene(args.position, rng) for _ in range(args.states)]

    # Each axis is a list of "the same state, one thing changed" overrides.
    axes = {
        "score_diff -3..+3": lambda st: [
            dict(x=st["x"], y=st["y"], under_pressure=st["under_pressure"],
                 minute=45.0, score_diff=sd) for sd in (-3, -1, 1, 3)],
        "minute 5..89": lambda st: [
            dict(x=st["x"], y=st["y"], under_pressure=st["under_pressure"],
                 minute=m, score_diff=0) for m in (5.0, 80.0, 89.0)],
        "pressure on/off": lambda st: [
            dict(x=st["x"], y=st["y"], under_pressure=not st["under_pressure"],
                 minute=45.0, score_diff=0)],
        # CONTROL: geometry, which MUST move the decision.
        "CONTROL: x +-10m": lambda st: [
            dict(x=min(104.0, st["x"] + dx), y=st["y"],
                 under_pressure=st["under_pressure"], minute=45.0, score_diff=0)
            for dx in (-10.0, 10.0)],
    }

    print(f"brain: brains/{args.position}.json   states: {args.states}")
    print(f"{'axis':<22} {'mean TV':>8} {'argmax flip':>12}")
    print("-" * 46)
    results = {}
    for name, mutate in axes.items():
        tvs, flips = [], []
        for scene in scenes:
            tv, flip = _compare(brain, scene, args.position, name, mutate(scene["st"]))
            tvs.append(tv)
            flips.append(flip)
        results[name] = {"tv": float(np.mean(tvs)), "flip": float(np.mean(flips))}
        print(f"{name:<22} {np.mean(tvs):>8.4f} {np.mean(flips):>11.1%}")

    control = results["CONTROL: x +-10m"]["tv"]
    print("-" * 46)
    for name, r in results.items():
        if name.startswith("CONTROL"):
            continue
        share = r["tv"] / control if control > 0 else 0.0
        verdict = ("IGNORED" if share < 0.10 else
                   "weak" if share < 0.30 else
                   "RESPONDS" if share < 0.70 else "DOMINANT")
        print(f"  {name:<20} {share:5.0%} of the control's effect  -> {verdict}")
    print("\n  A share near 0 means the player does not care about that axis.")
    print("  The control proves the harness can see a real change at all.")


if __name__ == "__main__":
    main()
