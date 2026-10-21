"""POLICY TARGET DEFERENCE -- the measurement the margin was supposed to come from.

The rule (`PossessionChain._policy_target_deference`) is landed, gated, default
OFF, and unit-tested (18 tests, negative control 5/5). What does NOT exist is the
`gap_frac` distribution, so `POLICY_TARGET_DEFERENCE_MARGIN = 0.05` is a guess
and is labelled as one in the source.

This probe exists to turn that guess into a measurement. It runs the SAME match
under both arms (the project's in-process A/B is valid; the "two `simulate()`
calls inherit module caches" rule was retracted 2026-10-04, see AGENTS.md
*REPRODUCIBILITY*), on the real 26/27 roster, and reports:

  1. WHO NAMES THE RECEIVER -- the `target_authority` histogram, per arm.
  2. THE GAP DISTRIBUTION -- `gap_frac` for every pass the rule inspected,
     split fired vs not, with the percentiles the margin should be set from.
  3. THE PASS MAP -- label AND geometry, per team: forward/square/backward,
     median length, median signed advance, lateral spread. Both are printed
     because the label can be right while the assignment is wrong (the
     `pass_direction` probe error, retracted 2026-10-02) -- a disagreement
     between them is visible here rather than hidden.
  4. EVENT COUNT AND SCORE -- the arm-level sanity check, so "the pass map
     changed" is never read off an arm that also changed the match size.

WHAT THIS PROBE DOES NOT DO: decide whether the ON arm is better. Fitness is not
the gate; the pass map and the eye are. It prints evidence.

    $env:PYTHONHASHSEED=0
    .venv\\Scripts\\python.exe tools\\diag\\_diag_deference_ab.py --arm both --seed 777

Two matches (~3 min here). `--arm off|on` runs one arm.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=("off", "on", "both"), default="both")
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--margin", type=float, default=None,
                help="override POLICY_TARGET_DEFERENCE_MARGIN for this run")
ap.add_argument("--dump", default=None,
                help="write per-pass deference records to this JSON path "
                     "(an `_on`/`_off` suffix is inserted for --arm both)")
A = ap.parse_args()

import run_match as RM                              # noqa: E402

random.seed(A.seed)
try:
    import numpy as np
    np.random.seed(A.seed)
except Exception:  # pragma: no cover
    pass

from player_dna import SquadBuilder                # noqa: E402
from match_engine import (MatchConfig, MatchEngine, TeamProfile, TeamStyle,  # noqa: E402
                          PlayingStyle, Intensity)
from match_engine import EventType as ET           # noqa: E402
from event_chain import PossessionChain            # noqa: E402
from pass_classifier import classify_direction     # noqa: E402

HOME, AWAY = RM.HOME_TEAM, RM.AWAY_TEAM

# ⛔ THE BAND. This probe's FIRST version used its own `SQUARE_M = 5.0` dead band
# and then reported "30-35% of passes disagree with `pass_direction`" -- a clean
# number under a heading that asserted more than its expression computed. The
# engine's own band is `pass_classifier.classify_direction`: `|signed_dx| <= 1.0`
# is sideways. So the disagreement was MY threshold, not the code, and the
# correct instrument applies the ENGINE'S classifier to the arrival coordinate.
# What remains after that fix is the real, already-documented open question --
# the label is computed from the AIMED `end_px` while the event stores the
# arrival `end_x` -- and its size is now measurable instead of being inflated
# by a constant the code never used.
#
# Eighth recorded instance of the same shape: when a check disagrees with the
# code, verify the check's frame (or its thresholds) before the code.
_LABEL = {"forward": "fwd", "backward": "back", "sideways": "square",
          None: "?", "": "?"}


def _pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def _build():
    hs = SquadBuilder.build(team_name=HOME, starters=RM.HOME_STARTERS,
                            substitutes=RM.HOME_SUBS,
                            team_superstars=RM.HOME_SUPERSTARS,
                            set_piece_takers=RM.HOME_SP_TAKERS,
                            soul_assignments=RM.SOUL_PLAYERS)
    aw = SquadBuilder.build(team_name=AWAY, starters=RM.AWAY_STARTERS,
                            substitutes=RM.AWAY_SUBS,
                            team_superstars=RM.AWAY_SUPERSTARS,
                            set_piece_takers=RM.AWAY_SP_TAKERS,
                            soul_assignments=RM.SOUL_PLAYERS)
    eng = MatchEngine(
        MatchConfig(home_team=HOME, away_team=AWAY, match_date=RM.MATCH_DATE,
                    matchday=RM.MATCHDAY, venue="X", stadium_capacity=45000),
        TeamProfile(name=HOME, style=RM.HOME_STYLE.style,
                    playing_style=RM.HOME_STYLE.playing_style,
                    intensity=RM.HOME_STYLE.intensity),
        TeamProfile(name=AWAY, style=RM.AWAY_STYLE.style,
                    playing_style=RM.AWAY_STYLE.playing_style,
                    intensity=RM.AWAY_STYLE.intensity))
    eng.set_squad(HOME, hs["starters"], hs["substitutes"])
    eng.set_squad(AWAY, aw["starters"], aw["substitutes"])
    return eng


def _run_arm(on: bool):
    """One arm, on a freshly-built world.

    A fresh engine per arm matters for the same reason it matters in
    `test_policy_deference`: a shared PositionEngine would let arm A's touches
    change arm B's geometry, which is how a "no difference" result is
    manufactured.
    """
    prev_flag = PossessionChain.POLICY_TARGET_DEFERENCE
    prev_margin = PossessionChain.POLICY_TARGET_DEFERENCE_MARGIN
    PossessionChain.POLICY_TARGET_DEFERENCE = bool(on)
    if A.margin is not None:
        PossessionChain.POLICY_TARGET_DEFERENCE_MARGIN = float(A.margin)
    random.seed(A.seed)
    try:
        import numpy as np
        np.random.seed(A.seed)
    except Exception:  # pragma: no cover
        pass
    try:
        eng = _build()
        res = eng.simulate()
        return res, dict(eng.position_engine.team_attacks_right)
    finally:
        PossessionChain.POLICY_TARGET_DEFERENCE = prev_flag
        PossessionChain.POLICY_TARGET_DEFERENCE_MARGIN = prev_margin


def _analyse(res, attacks_right):
    out = {
        "events": len(res.timeline),
        "score": getattr(res, "score_str", "?"),
        "authority": Counter(),
        "fired_gaps": [],
        "unfired_gaps": [],
        "silent_gaps": [],
        "reasons": Counter(),
        "silent_reasons": Counter(),
        "inspected": 0,
        "fired": 0,
        "applied": 0,
        "by_team": defaultdict(lambda: {
            "n": 0, "len": [], "adv": [], "y": [],
            "geo": Counter(), "label": Counter(), "disagree": 0,
        }),
        "samples": [],
        "dump": [],
    }
    for e in res.timeline:
        if e.event_type != ET.PASS:
            continue
        md = e.metadata or {}
        out["authority"][md.get("target_authority", "?")] += 1

        td = md.get("target_deference")
        if isinstance(td, dict):
            g = td.get("gap_frac")
            silent = bool(td.get("matrix_silent"))
            out["inspected"] += 1
            out["reasons"][td.get("matrix_reason", "")] += 1
            if silent:
                out["silent_reasons"][td.get("matrix_reason", "")] += 1
            if td.get("fired"):
                out["fired"] += 1
                if isinstance(g, (int, float)):
                    (out["silent_gaps"] if silent
                     else out["fired_gaps"]).append(float(g))
            elif isinstance(g, (int, float)):
                out["unfired_gaps"].append(float(g))
            if td.get("applied"):
                out["applied"] += 1
                if len(out["samples"]) < 8:
                    out["samples"].append({
                        "min": getattr(e, "minute", None),
                        "team": e.team, "player": e.player,
                        "mtx": td.get("matrix_value"),
                        "pol": td.get("policy_value"),
                        "gap": g, "silent": silent,
                        "reason": td.get("matrix_reason"),
                        "from": td.get("matrix_target"),
                        "to": td.get("policy_target"),
                    })
            out["dump"].append({
                "minute": getattr(e, "minute", None), "team": e.team,
                "player": e.player, **{k: td.get(k) for k in (
                    "fired", "applied", "matrix_value", "policy_value",
                    "gap_frac", "matrix_silent", "matrix_reason",
                    "matrix_target", "policy_target")},
                "authority": md.get("target_authority"),
            })

        # Pass map, team frame.
        if e.end_x is None or e.end_y is None:
            continue
        right = attacks_right.get(e.team, True)
        dx = (float(e.end_x) - float(e.location_x))
        if not right:
            dx = -dx
        n = math.hypot(float(e.end_x) - float(e.location_x),
                       float(e.end_y) - float(e.location_y))
        lbl = _LABEL.get(md.get("pass_direction"), "?")
        # The ENGINE'S OWN band, applied to the ARRIVAL point. Not a probe
        # constant -- see the note at `_LABEL`.
        geo = _LABEL[classify_direction(dx)]
        b = out["by_team"][e.team]
        b["n"] += 1
        b["len"].append(n)
        b["adv"].append(dx)
        b["y"].append(float(e.end_y))
        b["geo"][geo] += 1
        b["label"][lbl] += 1
        if lbl != "?" and lbl != geo:
            b["disagree"] += 1
    return out


def _fmt_hist(c: Counter, total: int) -> str:
    if not total:
        return "  (none)"
    order = [k for k, _ in c.most_common()]
    return "  ".join(f"{k}={c[k]} ({100.0 * c[k] / total:.1f}%)" for k in order)


def report(label: str, r: dict) -> None:
    print(f"\n{'=' * 78}\nARM = {label}\n{'=' * 78}")
    print(f"events={r['events']}  score={r['score']}")
    print(f"\n-- who named the receiver (PASS events) --")
    print("  " + _fmt_hist(r["authority"], sum(r["authority"].values())))

    print(f"\n-- the deference gap distribution --")
    if not r["inspected"]:
        print("  the rule inspected NOTHING (flag off, or no matrix-provenance "
              "pass was reached)")
    else:
        sg, fg, ug = r["silent_gaps"], r["fired_gaps"], r["unfired_gaps"]
        print(f"  inspected={r['inspected']}  fired={r['fired']}  "
              f"applied={r['applied']}")
        # ⛔ THE SPLIT THAT MATTERS, added after the first live run. `gap_frac`
        # alone CONFLATES a comparison with its absence: on `mtx_val <= 0` the
        # rule fires MON conditionally with `gap_frac == 0.0`, and on the first
        # run that mode was the majority of the `fired` set -- so the margin
        # appeared to be doing work it was not doing. Report the two modes
        # separately or the fired distribution is a number about two things.
        print(f"  MATRIX-SILENT fires (matrix_value <= 0, margin NOT applied): "
              f"n={len(sg)}")
        if sg:
            print(f"    gap_frac: p50={_pct(sg, .5):+.3f}  "
                  f"min={min(sg):+.3f}  max={max(sg):+.3f}")
        for nm, v in (("COMPARABLE fired", fg), ("COMPARABLE unfired", ug)):
            if v:
                print(f"  {nm}: n={len(v)}  p10={_pct(v, .1):+.3f}  "
                      f"p50={_pct(v, .5):+.3f}  p90={_pct(v, .9):+.3f}  "
                      f"min={min(v):+.3f}  max={max(v):+.3f}")
        if r["reasons"]:
            print("  matrix `reason` over inspected passes: "
                  + _fmt_hist(r["reasons"], r["inspected"]))
        if r["silent_reasons"]:
            print("  ... of which SILENT: "
                  + _fmt_hist(r["silent_reasons"], len(sg) or 1))
        print("  a NEGATIVE gap means the policy's option outscored the one the "
              "matrix chose;")
        print("  the margin should sit where the two distributions separate.")
        for s in r["samples"]:
            print(f"    applied: {s['team']} {s['player']} @{s['min']}'  "
                  f"matrix={s['mtx']}  policy={s['pol']}  gap={s['gap']}"
                  f"  silent={s['silent']}  reason={s['reason']}\n"
                  f"             {s['from']} -> {s['to']}")

    print(f"\n-- pass map --")
    for team in (HOME, AWAY):
        b = r["by_team"].get(team)
        if not b or not b["n"]:
            continue
        n = b["n"]
        print(f"  {team}  n={n}  median len={statistics.median(b['len']):.1f} m"
              f"  median adv={statistics.median(b['adv']):+.1f} m  "
              f"(band-free: the median pass goes this way)")
        print(f"    engine classifier on the ARRIVAL point:  " + "  ".join(
            f"{k}={b['geo'][k]} ({100.0 * b['geo'][k] / n:.1f}%)"
            for k in ("fwd", "square", "back")))
        print(f"    the event's `pass_direction` LABEL (from the AIM point):  "
              + "  ".join(
                  f"{k}={b['label'][k]} ({100.0 * b['label'][k] / n:.1f}%)"
                  for k in ("fwd", "square", "back", "?")))
        print(f"    label-vs-arrival disagreements: {b['disagree']} "
              f"({100.0 * b['disagree'] / n:.1f}%)  <- aim-vs-arrival, "
              f"a documented open question, NOT a code bug")


def main() -> int:
    if not os.environ.get("PYTHONHASHSEED"):
        print("NOTE: PYTHONHASHSEED is unset. Reproducibility is now crc32-based "
              "so this is hygiene, not a requirement.\n")
    print(f"POLICY_TARGET_DEFERENCE_MARGIN = "
          f"{PossessionChain.POLICY_TARGET_DEFERENCE_MARGIN}"
          f"{'  (overridden by --margin)' if A.margin is not None else '  (the unfitted default)'}")
    print(f"seed={A.seed}  frame: {HOME} {RM.HOME_STYLE.style.value}/"
          f"{RM.HOME_STYLE.playing_style.value} v {AWAY} "
          f"{RM.AWAY_STYLE.style.value}/{RM.AWAY_STYLE.playing_style.value}")

    arms = [False, True] if A.arm == "both" else [A.arm == "on"]
    for on in arms:
        res, ar = _run_arm(on)
        report("ON  (experiment enabled)" if on else "OFF (shipped default)", ar)
        if A.dump:
            path = A.dump if not (A.arm == "both") else (
                A.dump.replace(".json", f"_{'on' if on else 'off'}.json"))
            import json
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(ar["dump"], fh, indent=1)
            print(f"\n  per-pass deference records written to {path} "
                  f"({len(ar['dump'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
