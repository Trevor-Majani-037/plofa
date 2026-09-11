"""Per-position brain verification per the analyst protocol.

For each of the 11 positions:
  1. 500 random states via random_game_state -> print FULL intent distribution.
  2. Thresholds:  no single intent > 55%,  no top-2 combined > 75%,
                  >= 5 intents each > 3%.
  3. 5 hand-picked situations: signature intent #1/#2 must be in the top-3
     for appropriate situations, and OUT of the top-3 for inappropriate ones.

Usage:
    python verify_brains.py [--states 500] [--brains brains]
"""
from __future__ import annotations

import argparse
import json
import os
import random as _random

import numpy as np

from football_brain import FootballBrain, INTENT_LABELS
from brain_evolution import random_game_state, _DummyPositionEngine, _coords_to_players
from brain_sensors import extract_sensors

ALL_POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]

CONTEXT_SIG = {
    "ST": ["SHOOT", "THROUGH_BALL"],
    "CF": ["SHOOT", "THROUGH_BALL"],
    "CAM": ["THROUGH_BALL", "PROGRESSIVE_PASS"],
    "CM": ["THROUGH_BALL", "PROGRESSIVE_PASS"],
    "CB": ["RECYCLE", "SAFE_PASS"],
    "CDM": ["RECYCLE", "SAFE_PASS"],
    "GK": ["SAFE_PASS", "RECYCLE"],
    "LB": ["CROSS", "SAFE_PASS"],
    "RB": ["CROSS", "SAFE_PASS"],
    "LW": ["CROSS", "PROGRESSIVE_PASS"],
    "RW": ["CROSS", "PROGRESSIVE_PASS"],
}

# Hand-picked situations: (label, {sensor_idx: value}, expect-signature-in-top3)
# Sensor indices (brain_sensors): 4 nearest_defender_dist, 9 space_ahead,
# 12 final_third, 13 own_half, 14 goal_distance_norm, 15 central_lane,
# 16 under_pressure.  Lower = closer / no pressure; higher = more pressure.
SITUATIONS = {
    "GK": [
        ("own box, pressured, safe recycle", {12: 0.1, 13: 0.9, 16: 0.8, 14: 0.1}, True),
        ("own box, open, distribute", {12: 0.1, 13: 0.9, 16: 0.2, 14: 0.1}, True),
        ("opp box (rare), shoot/clear", {12: 0.9, 13: 0.1, 16: 0.2, 14: 0.9}, False),
    ],
    "CB": [
        ("own third, pressured, recycle", {12: 0.1, 13: 0.9, 16: 0.8, 4: 0.2, 15: 0.5, 14: 0.1}, True),
        ("own third, open, spread", {12: 0.1, 13: 0.9, 16: 0.2, 4: 0.9, 15: 0.5, 14: 0.1}, True),
        ("opp box, shoot", {12: 0.9, 13: 0.1, 16: 0.2, 4: 0.9, 15: 0.8, 14: 0.2}, False),
    ],
    "CDM": [
        ("midfield base, pressured, recycle", {12: 0.2, 13: 0.6, 16: 0.8, 4: 0.2, 15: 0.5, 14: 0.4}, True),
        ("midfield base, open, spread", {12: 0.2, 13: 0.6, 16: 0.2, 4: 0.9, 15: 0.5, 14: 0.4}, True),
        ("opp box, shoot", {12: 0.9, 13: 0.1, 16: 0.2, 4: 0.9, 15: 0.8, 14: 0.2}, False),
    ],
    "CM": [
        ("midfield, open, progressive", {12: 0.3, 13: 0.5, 16: 0.2, 4: 0.9, 9: 0.7, 15: 0.5, 14: 0.4}, True),
        ("final third, through-ball lane", {12: 0.8, 13: 0.1, 16: 0.3, 4: 0.6, 9: 0.7, 15: 0.6, 14: 0.5}, True),
        ("own box, clear/panic", {12: 0.1, 13: 0.9, 16: 0.9, 4: 0.2, 15: 0.5, 14: 0.1}, False),
    ],
    "CAM": [
        ("final third, central, through-ball lane", {12: 0.8, 13: 0.1, 16: 0.3, 4: 0.6, 9: 0.8, 15: 0.8, 14: 0.5}, True),
        ("final third, central, progressive", {12: 0.8, 13: 0.1, 16: 0.2, 4: 0.9, 9: 0.8, 15: 0.8, 14: 0.5}, True),
        ("own box, clear/panic", {12: 0.1, 13: 0.9, 16: 0.9, 4: 0.2, 15: 0.5, 14: 0.1}, False),
    ],
    "LW": [
        ("left flank, final third, cross", {12: 0.8, 13: 0.1, 16: 0.2, 4: 0.7, 9: 0.8, 15: 0.2, 14: 0.5}, True),
        ("left flank, final third, cut inside shoot", {12: 0.8, 13: 0.1, 16: 0.2, 4: 0.7, 9: 0.8, 15: 0.2, 14: 0.5}, True),
        ("own box, clear/panic", {12: 0.1, 13: 0.9, 16: 0.9, 4: 0.2, 15: 0.5, 14: 0.1}, False),
    ],
    "RW": [
        ("right flank, final third, cross", {12: 0.8, 13: 0.1, 16: 0.2, 4: 0.7, 9: 0.8, 15: 0.2, 14: 0.5}, True),
        ("right flank, final third, cut inside shoot", {12: 0.8, 13: 0.1, 16: 0.2, 4: 0.7, 9: 0.8, 15: 0.2, 14: 0.5}, True),
        ("own box, clear/panic", {12: 0.1, 13: 0.9, 16: 0.9, 4: 0.2, 15: 0.5, 14: 0.1}, False),
    ],
    "ST": [
        ("in box, goal close, shoot", {12: 0.9, 13: 0.1, 16: 0.2, 4: 0.5, 9: 0.6, 15: 0.8, 14: 0.2}, True),
        ("final third, through-ball run", {12: 0.8, 13: 0.1, 16: 0.2, 4: 0.8, 9: 0.9, 15: 0.8, 14: 0.4}, True),
        ("own box, defend", {12: 0.1, 13: 0.9, 16: 0.9, 4: 0.2, 15: 0.5, 14: 0.1}, False),
    ],
    "CF": [
        ("in box, goal close, shoot", {12: 0.9, 13: 0.1, 16: 0.2, 4: 0.5, 9: 0.6, 15: 0.8, 14: 0.2}, True),
        ("final third, through-ball run", {12: 0.8, 13: 0.1, 16: 0.2, 4: 0.8, 9: 0.9, 15: 0.8, 14: 0.4}, True),
        ("own box, defend", {12: 0.1, 13: 0.9, 16: 0.9, 4: 0.2, 15: 0.5, 14: 0.1}, False),
    ],
    "LB": [
        ("left flank, final third, cross", {12: 0.8, 13: 0.1, 16: 0.2, 4: 0.7, 9: 0.8, 15: 0.2, 14: 0.5}, True),
        ("own third, open, progressive", {12: 0.2, 13: 0.7, 16: 0.2, 4: 0.9, 9: 0.6, 15: 0.2, 14: 0.5}, True),
        ("opp box, shoot (rare)", {12: 0.9, 13: 0.1, 16: 0.2, 4: 0.6, 9: 0.8, 15: 0.2, 14: 0.2}, False),
    ],
    "RB": [
        ("right flank, final third, cross", {12: 0.8, 13: 0.1, 16: 0.2, 4: 0.7, 9: 0.8, 15: 0.2, 14: 0.5}, True),
        ("own third, open, progressive", {12: 0.2, 13: 0.7, 16: 0.2, 4: 0.9, 9: 0.6, 15: 0.2, 14: 0.5}, True),
        ("opp box, shoot (rare)", {12: 0.9, 13: 0.1, 16: 0.2, 4: 0.6, 9: 0.8, 15: 0.2, 14: 0.2}, False),
    ],
}


def _build_sensors(st: dict, position: str) -> np.ndarray:
    """Mirror brain_evolution.synthetic_fitness's sensor construction."""
    t_coords = st["teammates"]
    d_coords = st["defenders"]
    all_names = {}
    for i, (tx, ty) in enumerate(t_coords):
        all_names[f"t{i}"] = (tx, ty)
    for i, (dx, dy) in enumerate(d_coords):
        all_names[f"d{i}"] = (dx, dy)
    pe = _DummyPositionEngine(all_names)
    teammates = _coords_to_players(t_coords, "t")
    defenders = _coords_to_players(d_coords, "d")
    return extract_sensors(
        None, st["x"], st["y"], teammates, defenders, pe,
        st["under_pressure"], st["attacks_right"], st["game_state"],
        st["minute"],
    )


def _decide(brain: FootballBrain, sensors: np.ndarray) -> str:
    out = brain.forward(np.asarray(sensors, dtype=np.float64).reshape(1, -1)).ravel()
    i = int(np.argmax(out))
    return INTENT_LABELS[i]


def _situation_sensors(position: str, flags: dict) -> np.ndarray:
    st = random_game_state(_random.Random(0), position)
    s = _build_sensors(st, position)
    for idx, val in flags.items():
        s[idx] = val
    return s


def verify_position(pos: str, brain: FootballBrain, n_states: int) -> dict:
    rng = _random.Random(123)
    counts = {label: 0 for label in INTENT_LABELS}
    for _ in range(n_states):
        st = random_game_state(rng, pos)
        s = _build_sensors(st, pos)
        counts[_decide(brain, s)] += 1

    total = sum(counts.values())
    dist = {k: v / total for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
    top = list(dist.items())
    single = float(top[0][1])
    top2 = float(top[0][1] + top[1][1])
    n_req = int(np.sum([v > 0.03 for v in dist.values()]))

    checks = {
        "single<=55%": single <= 0.55,
        "top2<=75%": top2 <= 0.75,
        ">=5 intents >3%": n_req >= 5,
    }

    sit_rows = []
    for label, flags, expect_sig in SITUATIONS.get(pos, []):
        s = _situation_sensors(pos, flags)
        out = brain.forward(np.asarray(s).reshape(1, -1)).ravel()
        order = np.argsort(out)[::-1]
        top3 = [INTENT_LABELS[int(i)] for i in order[:3]]
        sig = CONTEXT_SIG.get(pos, [])
        got_sig = any(sc in top3 for sc in sig)
        sit_rows.append({
            "label": label, "expect": expect_sig,
            "got": got_sig, "top3": top3, "pass": got_sig == expect_sig,
        })

    hand_pass = all(r["pass"] for r in sit_rows) if sit_rows else True
    return {
        "position": pos,
        "distribution": dist,
        "single": single, "top2": top2, "n_gt_3pct": n_req,
        "checks": checks, "situations": sit_rows, "hand_pass": hand_pass,
        "pass": all(checks.values()) and hand_pass,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=int, default=500)
    p.add_argument("--brains", default="brains")
    p.add_argument("--json", default="brain_verify.json")
    args = p.parse_args()

    results = []
    for pos in ALL_POSITIONS:
        path = os.path.join(args.brains, f"{pos}.json")
        if not os.path.exists(path):
            print(f"{pos:>3}: SKIP (no {path})")
            continue
        brain = FootballBrain.load(path)
        r = verify_position(pos, brain, args.states)
        results.append(r)
        dist = ", ".join(f"{k}={v:.1%}" for k, v in r["distribution"].items())
        print(f"\n=== {pos} ===  pass={r['pass']}")
        print(f"  distribution: {dist}")
        print(f"  single={r['single']:.1%}  top2={r['top2']:.1%}  "
              f"n>3%={r['n_gt_3pct']}  "
              f"single<=55%:{r['checks']['single<=55%']} "
              f"top2<=75%:{r['checks']['top2<=75%']} "
              f"intents>3%:{r['checks']['>=5 intents >3%']}")
        for s in r["situations"]:
            mark = "OK " if s["pass"] else "FAIL"
            print(f"  [{mark}] {s['label']}: top3={s['top3']} "
                  f"(sig={s['got']}, expect={s['expect']})")

    n_pass = sum(1 for r in results if r["pass"])
    print(f"\n{len(results)} positions verified: {n_pass} pass, "
          f"{len(results) - n_pass} fail")
    with open(args.json, "w") as f:
        json.dump([{**r, "distribution": {k: round(v, 4) for k, v in r["distribution"].items()}}
                   for r in results], f, indent=2)
    print(f"Verification JSON -> {args.json}")


if __name__ == "__main__":
    main()