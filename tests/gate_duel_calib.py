#!/usr/bin/env python
"""REJECTED A/B GATE — SIM-SIDE DUEL VOLUME CALIBRATION (2026-11-23).

DO NOT USE THIS AS THE GATE FOR THE SHIPPED CHANGE. The first two paired
matches regressed outcomes (legacy 0-2/0-2, calib 0-7/2-4, possession -4 to
-7 pts), so the sim-side levers were abandoned. The shipped fix is the
passive ledger in exporter.py; it was verified by tests/gate_duel_calib.py
--arm legacy (3 matches: goals 1-3/1-4/0-1, identical to the pre-fix
baseline; zero-sum 0; 86-107 ground duels per team). Kept as the record of
the rejected experiment.

ORIGINAL GATE DESCRIPTION:

THE CHANGE UNDER TEST
---------------------
tests/duel_audit.py measured, on the shipped ledger:
  * ground duels 146-193 att per team per match (real Opta ~90-140);
  * the branch-4 50-50 duel (event_chain PossessionChain) fired on 35% of
    under-pressure touches and the carrier won 96% of them — consequence-free
    duel wins for the possessing side and mirrored losses for the pressers
    (a CM at 3/40);
  * a touch could roll a 50-50 duel AND a take-on against the same man;
  * carry races passed EVERY defender into resolve_dribble, so men closing
    from 12 m who never reached the touch still stamped a dispossession
    (75/match: 32 micro-carry + 43 main-carry, vs real ~20-30);
  * _defensive_recovery could tackle a carrier who had already lost the ball.

ARM `legacy` : PLOFA_DUEL_CALIB=0 — the audit incumbent (post ledger fix).
ARM `calib`  : PLOFA_DUEL_CALIB=1 — branch-4 only when the presser is inside
               3.5 m (a real challenge, not a press from 8 m), one contest
               per touch, recovery is a get-it-away act only.

Both arms carry the ledger fix (named dispossession winner, contested-header
pairing, no-opponent-no-duel), so zero-sum must hold in BOTH arms.

GATE
----
  1. zero-sum: won(A) == lost(B) for ground AND aerial, both arms, every match
  2. volume:   calib mean ground duels att per team in [80, 150]
  3. contest:  calib branch-4 carrier win share in [0.45, 0.75]
  4. outcomes: |GF| and |GA| within 1 goal of incumbent over 7 matches,
               possession within 3 pts (report shots/xG/passes/turnovers)

Run:  python tests/gate_duel_calib.py            (both arms, 7 matches each)
      python tests/gate_duel_calib.py --arm calib --matches 3   (rerun one arm)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from match_engine import EventType
from exporter import StatAccumulator
from duel_audit import build_match, HOME, AWAY


def _ledger(result, squads) -> Dict[str, Any]:
    acc = StatAccumulator(result, squads)
    out: Dict[str, Any] = {}
    tot = {}
    for team in (HOME, AWAY):
        g_att = g_won = a_att = a_won = 0
        for s in acc.stats.values():
            if s.get("team") != team:
                continue
            g_att += s["ground_duels_att"]
            g_won += s["ground_duels_won"]
            a_att += s["aerial_duels_att"]
            a_won += s["aerial_duels_won"]
        tot[team] = (g_att, g_won, a_att, a_won)
    hg, hw, ha, haw = tot[HOME]
    ag, aw, aa, aaw = tot[AWAY]
    out["ground_att_home"], out["ground_att_away"] = hg, ag
    out["ground_won_home"], out["ground_won_away"] = hw, aw
    out["zs_ground_home"] = hw - (ag - aw)
    out["zs_ground_away"] = aw - (hg - hw)
    out["zs_aerial_home"] = haw - (aa - aaw)
    out["zs_aerial_away"] = aaw - (ha - haw)

    c = Counter()
    g50_wins = 0
    for e in result.timeline:
        et = e.event_type
        if et == EventType.GROUND_DUEL:
            c["ground50"] += 1
            g50_wins += 1 if e.outcome else 0
        elif et == EventType.DISPOSSESSED:
            c["dispossessed"] += 1
            if (e.metadata or {}).get("micro_carry"):
                c["dispossessed_micro"] += 1
            if not e.secondary_player:
                c["dispossessed_unnamed"] += 1
        elif et == EventType.TACKLE_WON:
            c["tackle_won"] += 1
        elif et == EventType.TACKLE_LOST:
            c["tackle_lost"] += 1
        elif et in (EventType.DRIBBLE_SUCCESS, EventType.DRIBBLE_FAIL):
            c["dribbles"] += 1
        elif et == EventType.AERIAL_DUEL:
            c["aerial50"] += 1
        elif (e.metadata or {}).get("contested_header"):
            c["contested_header"] += 1
        elif et == EventType.PASS:
            c["passes"] += 1
            c["passes_ok"] += 1 if e.outcome else 0
        elif et == EventType.TURNOVER:
            c["turnovers"] += 1
        elif et in (EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
                    EventType.SHOT_BLOCKED, EventType.GOAL):
            c["shots"] += 1
            out_xg = float(getattr(e, "xg", 0.0) or 0.0)
            c["xg"] += out_xg
    out.update({k: float(v) for k, v in c.items()})
    out["g50_carrier_win_pct"] = round(100.0 * g50_wins / c["ground50"], 1) \
        if c["ground50"] else 0.0
    return out


def run_arm(arm: str, matches: int, seed: int, out_path: str) -> None:
    os.environ["PLOFA_DUEL_CALIB"] = "1" if arm == "calib" else "0"
    rows = []
    for m in range(matches):
        sd = seed + m * 100
        engine, squads = build_match(sd)
        result = engine.simulate()
        row = {"match": m + 1, "seed": sd,
               "home_goals": int(result.home_goals),
               "away_goals": int(result.away_goals),
               "possession": float(result.home_possession_pct)}
        row.update(_ledger(result, squads))
        rows.append(row)
        print(f"  [{arm}] match {m+1} seed {sd}: "
              f"{row['home_goals']}-{row['away_goals']} "
              f"poss {row['possession']:.1f}% "
              f"ground att {row['ground_att_home']}/{row['ground_att_away']} "
              f"zs {row['zs_ground_home']:+d}/{row['zs_ground_away']:+d} "
              f"g50 win {row['g50_carrier_win_pct']}%", flush=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"arm": arm, "matches": rows}, fh, indent=1)
    print(f"  [{arm}] wrote {out_path}", flush=True)


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    return float(np.mean([r.get(key, 0.0) for r in rows]))


def _report(arm: Dict[str, Any]) -> Dict[str, Any]:
    rows = arm["matches"]
    n = len(rows)
    keys = ("ground_att_home", "ground_att_away", "ground_won_home",
            "ground_won_away", "zs_ground_home", "zs_ground_away",
            "zs_aerial_home", "zs_aerial_away", "ground50",
            "g50_carrier_win_pct", "dispossessed", "dispossessed_micro",
            "dispossessed_unnamed", "tackle_won", "tackle_lost", "dribbles",
            "aerial50", "contested_header", "passes", "passes_ok",
            "turnovers", "shots", "xg")
    rep = {"arm": arm["arm"], "matches": n,
           "gf": sum(r["home_goals"] for r in rows),
           "ga": sum(r["away_goals"] for r in rows),
           "possession": round(_mean(rows, "possession"), 1)}
    for k in keys:
        rep[k] = round(_mean(rows, k), 2)
    rep["ground_att_per_team"] = round(
        (rep["ground_att_home"] + rep["ground_att_away"]) / 2.0, 1)
    rep["dispossessed_per_team"] = round(rep["dispossessed"] / 2.0, 1)
    rep["zs_max_abs"] = max(abs(rep["zs_ground_home"]), abs(rep["zs_ground_away"]),
                            abs(rep["zs_aerial_home"]), abs(rep["zs_aerial_away"]))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("legacy", "calib"), default=None)
    ap.add_argument("--matches", type=int, default=7)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.arm:
        out = args.out or os.path.join(here, f"_gate_duel_{args.arm}.json")
        run_arm(args.arm, args.matches, args.seed, out)
        return 0

    # Arms are independent processes (seeded per match, no shared state), so
    # run them CONCURRENTLY: 7 matches of wall time instead of 14. Each arm
    # prints tagged with [arm], so interleaved output stays readable.
    results = {}
    procs = {}
    for arm in ("legacy", "calib"):
        out = os.path.join(here, f"_gate_duel_{arm}.json")
        print(f"\n=== ARM {arm.upper()} — {args.matches} matches ===", flush=True)
        procs[arm] = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--arm", arm,
             "--matches", str(args.matches), "--seed", str(args.seed),
             "--out", out],
        )
    for arm, proc in procs.items():
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, f"arm {arm}")
        with open(os.path.join(here, f"_gate_duel_{arm}.json"),
                  encoding="utf-8") as fh:
            results[arm] = _report(json.load(fh))

    leg, cal = results["legacy"], results["calib"]
    rows = [
        ("record GF-GA", f"{leg['gf']}-{leg['ga']}", f"{cal['gf']}-{cal['ga']}"),
        ("possession %", leg["possession"], cal["possession"]),
        ("ground duels att / team", leg["ground_att_per_team"], cal["ground_att_per_team"]),
        ("ground won home/away", f"{leg['ground_won_home']}/{leg['ground_won_away']}",
         f"{cal['ground_won_home']}/{cal['ground_won_away']}"),
        ("zero-sum max |delta|", leg["zs_max_abs"], cal["zs_max_abs"]),
        ("branch-4 50-50 / match", leg["ground50"], cal["ground50"]),
        ("branch-4 carrier win %", leg["g50_carrier_win_pct"], cal["g50_carrier_win_pct"]),
        ("dispossessed / team", leg["dispossessed_per_team"], cal["dispossessed_per_team"]),
        ("  of which micro-carry", round(leg["dispossessed_micro"] / 2, 1),
         round(cal["dispossessed_micro"] / 2, 1)),
        ("  of which unnamed", round(leg["dispossessed_unnamed"] / 2, 1),
         round(cal["dispossessed_unnamed"] / 2, 1)),
        ("tackles won / match", leg["tackle_won"], cal["tackle_won"]),
        ("tackles lost / match", leg["tackle_lost"], cal["tackle_lost"]),
        ("dribbles / match", leg["dribbles"], cal["dribbles"]),
        ("aerial duels / match", leg["aerial50"], cal["aerial50"]),
        ("contested headers / match", leg["contested_header"], cal["contested_header"]),
        ("passes / match", leg["passes"], cal["passes"]),
        ("pass completion %", round(100 * leg["passes_ok"] / leg["passes"], 1)
         if leg["passes"] else 0.0,
         round(100 * cal["passes_ok"] / cal["passes"], 1) if cal["passes"] else 0.0),
        ("turnovers / match", leg["turnovers"], cal["turnovers"]),
        ("shots / match", leg["shots"], cal["shots"]),
        ("xG / match", leg["xg"], cal["xg"]),
    ]
    width = max(len(r[0]) for r in rows)
    print()
    print(f"{'':<{width}}   {'LEGACY':>14}   {'CALIB':>14}")
    print("-" * (width + 34))
    for name, a, b in rows:
        print(f"{name:<{width}}   {str(a):>14}   {str(b):>14}")

    checks = [
        ("zero-sum both arms", leg["zs_max_abs"] == 0 and cal["zs_max_abs"] == 0),
        ("calib ground att/team in [80,150]", 80 <= cal["ground_att_per_team"] <= 150),
        ("calib branch-4 carrier win in [45,75]%",
         45 <= cal["g50_carrier_win_pct"] <= 75),
        ("GF within 1", abs(cal["gf"] - leg["gf"]) <= 1),
        ("GA within 1", abs(cal["ga"] - leg["ga"]) <= 1),
        ("possession within 3 pts", abs(cal["possession"] - leg["possession"]) <= 3),
    ]
    print("\n--- GATE CHECK ---")
    ok = True
    for name, passed in checks:
        print(f"  {name:42s} {'PASS' if passed else 'FAIL'}")
        ok = ok and passed
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())