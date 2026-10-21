"""THE REAL-MATCH GATE for a brain set — the acceptance criteria, measured.

WHY THIS PROBE
---------------
The plan's acceptance gate is **pass maps and the eye, never fitness**, and
it names three things:

  1. forward / square / backward from 27.9 / 41.0 / 31.1 toward
     35-40 / 45-50 / 10-15;
  2. median pass length from 9.7 m into 15-18 m;
  3. three pass maps that read vertical, with a reason for each arrow.

`compare_striker.py` measures the striker's goals and shots but not the pass
geometry, and `validate_neural_xl` reports a FITNESS — which the plan says
is not part of the gate.  So this probe measures the acceptance numbers
directly, on the SAME seeding as the harness of record
(`random.seed(seed + m * 100)` before every match, the heuristic arm pinned
to the identity perception regime), so the numbers are comparable with the
historical ones.

**THE MEASUREMENT OF THE DIRECTION IS THE ENGINE'S OWN**, applied to the
ARRIVAL point and with the sign flipped for the away side — the
`pass_direction` metadata is 98.4% correct (measured 2026-10-02) and the
probe-local band was the thing that was wrong then, not the code.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_v7_gate.py brains_v7 brains \\
        --matches 7 --seed 21
"""
from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "tools", "gates"))

from match_engine import EventType  # noqa: E402
from validate_neural_xl import _run_neural  # noqa: E402

SHOT_TYPES = {
    EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET, EventType.SHOT_BLOCKED,
    EventType.HIT_WOODWORK, EventType.FREEKICK_DIRECT, EventType.PENALTY_SCORED,
    EventType.PENALTY_MISSED,
}
GOAL_TYPES = {EventType.GOAL, EventType.PENALTY_SCORED}

_LABEL = {"forward": "forward", "square": "square", "backward": "backward"}

#: The Opta-style pass-progression rule: a pass is FORWARD if it closes at
#: least 10% of the distance to the opponent's goal, BACKWARD if it loses at
#: least 10%, and SQUARE otherwise.  **This is the band the real-football
#: benchmark (35-40 / 45-50 / 10-15) is measured in**, and the engine's own
#: `classify_direction` is a different one (a +/-1 m dead band), so the two
#: must never be compared to each other.
_PROG_THRESHOLD = 0.10


def _classify_engine(dx: float) -> str:
    """The engine's own band, applied to the ARRIVAL point."""
    from pass_classifier import classify_direction
    return _LABEL.get(classify_direction(dx), "?")


def _classify_progression(x0: float, y0: float, x1: float, y1: float,
                          attacking_right: bool) -> str:
    """The Opta-style progression band."""
    gx, gy = (105.0, 34.0) if attacking_right else (0.0, 34.0)
    d0 = math.hypot(gx - x0, gy - y0)
    d1 = math.hypot(gx - x1, gy - y1)
    if d0 <= 0:
        return "?"
    prog = (d0 - d1) / d0
    if prog >= _PROG_THRESHOLD:
        return "forward"
    if prog <= -_PROG_THRESHOLD:
        return "backward"
    return "square"


def _shape(result, home: str) -> dict:
    """The team's SHAPE, from the events.

    The raw frame is already "home attacks right" (the engine sets
    `team_attacks_right` once per team and never flips it at half time), so
    the home team's x needs no mirroring and y is never mirrored.

    Two families of measure, because they answer different questions:

    * WHERE THE PLAY HAPPENED — the events' depth (the median x) and width
      (the interquartile range of y), and the share of the play in each
      third;
    * WHERE THE PLAYERS ACTUALLY OPERATED — each player's average touch
      position, and the spread of those averages (the team's width and
      depth).  This is the measure the pass-network panel draws, so the
      numbers and the picture are the same thing.
    """
    xs, ys = [], []
    per = {}
    for e in result.timeline:
        if e.team != home:
            continue
        if e.location_x is None or e.location_y is None:
            continue
        x = float(e.location_x)
        y = float(e.location_y)
        xs.append(x)
        ys.append(y)
        if e.player:
            # KEYED BY (team, player), NOT BY player ALONE.  The scratch
            # fixture gives BOTH teams the same generic names (GK, CB1, LB,
            # ...), so keying by name alone merges the two sides' players —
            # and because their frames run in opposite directions the
            # averages CANCEL, which is how the first version of this
            # measurement reported a team width of 15 m.  (The same merged
            # names are what emptied the pass-network panel: see
            # AGENTS.md, *A BLANK PASS-NETWORK PANEL IS A FIXTURE ARTEFACT*.)
            per.setdefault((e.team, e.player), []).append((x, y))
    if not xs:
        return {}
    xs.sort()
    ys.sort()

    def q(vals, p):
        if not vals:
            return 0.0
        i = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
        return vals[i]

    px, py = [], []
    for _name, pts in per.items():
        px.append(sum(p[0] for p in pts) / len(pts))
        py.append(sum(p[1] for p in pts) / len(pts))
    px.sort()
    py.sort()
    return {
        "n_events": len(xs),
        "median_x": q(xs, 0.5),
        "p25_x": q(xs, 0.25),
        "p75_x": q(xs, 0.75),
        "width_y": q(ys, 0.75) - q(ys, 0.25),
        "final_third": 100.0 * sum(1 for x in xs if x > 70.0) / len(xs),
        "own_third": 100.0 * sum(1 for x in xs if x < 35.0) / len(xs),
        "team_depth": (px[-1] - px[0]) if len(px) > 1 else 0.0,
        "team_width": (py[-1] - py[0]) if len(py) > 1 else 0.0,
        "n_players": len(px),
    }


def run_arm(brains_dir: str, n_matches: int, seed: int,
            home_style: str, away_style: str, perception: bool = False) -> dict:
    goals = 0
    poss = []
    scores = []
    shots_by = Counter()
    goals_by = Counter()
    geo = Counter()
    geo_engine = Counter()
    lengths = []
    advances = []
    events = Counter()
    shape_acc = []
    for m in range(n_matches):
        s = seed + m * 100
        random.seed(s)          # identical stream to the harness of record
        result, _fitness = _run_neural(brains_dir, s, home_style, away_style,
                                       perception=perception)
        home = result.config.home_team
        goals += result.home_goals
        poss.append(result.home_possession_pct)
        scores.append(result.score_str)
        sh = _shape(result, home)
        if sh:
            shape_acc.append(sh)
        attacks_right = getattr(result, "attacks_right_by_team", None) or {}
        for e in result.timeline:
            if e.team != home:
                continue
            events[str(e.event_type)] += 1
            if e.event_type in GOAL_TYPES:
                goals_by[e.player or "?"] += 1
            elif e.event_type in SHOT_TYPES:
                shots_by[e.player or "?"] += 1
            if e.end_x is None or e.end_y is None:
                continue
            right = attacks_right.get(e.team, True)
            dx = float(e.end_x) - float(e.location_x)
            if not right:
                dx = -dx
            lengths.append(math.hypot(float(e.end_x) - float(e.location_x),
                                      float(e.end_y) - float(e.location_y)))
            advances.append(dx)
            geo[_classify_progression(float(e.location_x), float(e.location_y),
                                     float(e.end_x), float(e.end_y),
                                     right)] += 1
            geo_engine[_classify_engine(dx)] += 1
    n = sum(geo.values()) or 1
    shape = {}
    if shape_acc:
        for k in shape_acc[0]:
            vals = [s[k] for s in shape_acc]
            shape[k] = sum(vals) / len(vals)
        shape["width_y_median"] = statistics.median(
            [s["width_y"] for s in shape_acc])
        shape["team_width_median"] = statistics.median(
            [s["team_width"] for s in shape_acc])
        shape["team_depth_median"] = statistics.median(
            [s["team_depth"] for s in shape_acc])
    return {
        "goals": goals, "scores": scores,
        "poss": statistics.mean(poss) if poss else 0.0,
        "shots_by": shots_by, "goals_by": goals_by,
        "geo": {k: 100.0 * v / n for k, v in geo.items()},
        "geo_engine": {k: 100.0 * v / n for k, v in geo_engine.items()},
        "n_passes": n,
        "median_len": statistics.median(lengths) if lengths else 0.0,
        "median_adv": statistics.median(advances) if advances else 0.0,
        "events": events,
        "shape": shape,
    }


def show(label: str, r: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  home goals {r['goals']}   possession {r['poss']:.1f}%   "
          f"scores {r['scores']}")
    print(f"  passes measured: {r['n_passes']}")
    print(f"  forward/square/backward (Opta progression band): "
          f"{r['geo'].get('forward', 0):.1f} / "
          f"{r['geo'].get('square', 0):.1f} / {r['geo'].get('backward', 0):.1f}"
          f"   (real PL ~35-40 / 45-50 / 10-15)")
    print(f"  the engine's own band (+/-1 m), for reference: "
          f"{r['geo_engine'].get('forward', 0):.1f} / "
          f"{r['geo_engine'].get('square', 0):.1f} / "
          f"{r['geo_engine'].get('backward', 0):.1f}")
    print(f"  median pass length {r['median_len']:.1f} m  (real ~15-18)   "
          f"median signed advance {r['median_adv']:+.1f} m")
    print(f"  shots: {dict(r['shots_by'])}")
    print(f"  goals: {dict(r['goals_by'])}")
    ev = r["events"]
    print(f"  events: PASS {ev.get('PASS', 0)}  CARRY {ev.get('CARRY', 0)}  "
          f"SHOTS {sum(v for k, v in ev.items() if 'SHOT' in k)}  "
          f"GOALS {ev.get('GOAL', 0)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("brains_a")
    p.add_argument("brains_b")
    p.add_argument("--matches", type=int, default=7)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    p.add_argument("--perception", choices=("off", "on"), default="off",
                   help="'off' is the gate of record (the identity regime: "
                        "the heuristic arm has no perception layer, so "
                        "degrading the neural arm's vision would be unfair). "
                        "'on' is the PRODUCTION condition: the brain is "
                        "scored in the world it actually plays in - a 150-"
                        "degree cone, a range limit, positional noise - which "
                        "is exactly the condition under which the v7 striker "
                        "has to read the situation and shoot.")
    args = p.parse_args()

    for d in (args.brains_a, args.brains_b):
        if not os.path.isdir(d):
            print(f"! missing brain dir: {d}")
            return 1
    os.environ.setdefault("PYTHONPATH", ".")
    perception = args.perception == "on"
    tag = "  [PERCEPTION ON - production condition]" if perception \
        else "  [identity regime - the gate of record]"

    a = run_arm(args.brains_a, args.matches, args.seed, args.home_style,
                args.away_style, perception=perception)
    show(f"A = {os.path.basename(args.brains_a)}{tag}", a)
    b = run_arm(args.brains_b, args.matches, args.seed, args.home_style,
                args.away_style, perception=perception)
    show(f"B = {os.path.basename(args.brains_b)}{tag}", b)

    print("\n" + "=" * 72)
    print(f"{'metric':<34}{'A':>18}{'B':>18}")
    print("-" * 72)
    for key, label in (("goals", "home goals"),
                        ("poss", "possession %")):
        print(f"{label:<34}{a[key]:>18.1f}{b[key]:>18.1f}")
    for d in ("forward", "square", "backward"):
        print(f"{d + ' %':<34}{a['geo'].get(d, 0):>18.1f}"
              f"{b['geo'].get(d, 0):>18.1f}")
    print(f"{'median pass length':<34}{a['median_len']:>18.1f}"
          f"{b['median_len']:>18.1f}")
    print(f"{'median signed advance':<34}{a['median_adv']:>+18.1f}"
          f"{b['median_adv']:>+18.1f}")
    st_a = a["shots_by"].get("ST", 0)
    st_b = b["shots_by"].get("ST", 0)
    print(f"{'ST shots':<34}{st_a:>18}{st_b:>18}")
    print(f"{'ST goals':<34}{a['goals_by'].get('ST', 0):>18}"
          f"{b['goals_by'].get('ST', 0):>18}")
    print("-" * 72)
    print("THE SHAPE (the mean over the matches; the medians in brackets)")
    for key, label, fmt in (
            ("median_x", "the play's depth (median x)", "{:.1f}"),
            ("p25_x", "the play's depth (p25 x)", "{:.1f}"),
            ("p75_x", "the play's depth (p75 x)", "{:.1f}"),
            ("width_y", "the play's width (y IQR)", "{:.1f}"),
            ("width_y_median", "the play's width (y IQR, median)", "{:.1f}"),
            ("final_third", "the play in the final third %", "{:.1f}"),
            ("own_third", "the play in the own third %", "{:.1f}"),
            ("team_width", "the TEAM's width (players' y spread)", "{:.1f}"),
            ("team_width_median", "the TEAM's width (median match)", "{:.1f}"),
            ("team_depth", "the TEAM's depth (players' x spread)", "{:.1f}"),
            ("team_depth_median", "the TEAM's depth (median match)", "{:.1f}"),
    ):
        va = a["shape"].get(key)
        vb = b["shape"].get(key)
        if va is None or vb is None:
            continue
        print(f"{label:<34}" + fmt.format(va) + " " * 10 + fmt.format(vb))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
