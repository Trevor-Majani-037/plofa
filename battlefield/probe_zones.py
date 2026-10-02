"""
Micro-batch zone/outcome probe: logs per-zone xG, shot counts, SOT, goals,
and conversion so calibration decisions (take-gate, XGEngine table, gate
re-affirmation) can be made from measured numbers instead of guesses.

    python -m battlefield.probe_zones --matches 12 --seed 2026
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from match_engine import (                                   # noqa: E402
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from player_dna import SquadBuilder                          # noqa: E402
from squad_manager import SubstitutionController             # noqa: E402
from manager_profile import ManagerPool                      # noqa: E402
from roster_loader import get_loader                         # noqa: E402

# Reuse the battlefield's palette + pairing helpers so the sample matches
# the BF corpus style mix.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_plofa import (_STYLE_PALETTE, _team_profile, _pick_clubs,  # noqa: E402
                       _register_sub_schedule, _shot_in_box)

_SOT = frozenset({"SHOT_ON_TARGET", "HIT_WOODWORK"})
_SHOT_OFF = frozenset({"SHOT_OFF_TARGET", "SHOT_BLOCKED"})
_PEN = frozenset({"PENALTY_SCORED", "PENALTY_MISSED"})
_ATTEMPTS = frozenset({
    "PASS", "PROGRESSIVE_PASS", "THROUGH_BALL", "SWITCH_OF_PLAY",
    "CROSS_ATTEMPT", "CORNER_TAKEN",
})


def describe_zone(e: Any, home: str) -> str:
    """Bucket a shot event by the engine's own xG-zone keys (PitchZone.xg_zone)
    so per-zone xG averages map 1:1 onto XGEngine.ZONE_XG."""
    x = getattr(e, "location_x", None)
    if x is None:
        return "unknown"
    attacks_right = getattr(e, "team", "") == home
    sway = getattr(e, "situation", None)
    sname = getattr(sway, "name", "") if sway else ""
    if sname == "PENALTY":
        return "penalty"
    if not attacks_right and x >= 50.0:
        # Away-team shots are occasionally recorded in an un-mirrored
        # rightward frame (x~85-96 = 16-22m in the attacking-left frame).
        # Normalize so zone metrics are honest; the shot itself resolved
        # with the correct zone-xG already.
        x = 105.0 - x
    if attacks_right:
        if x >= 99:
            return "six_yard_box"
        if x >= 83:
            return "inside_box"
        if x >= 70:
            return "edge_of_box"
        return "outside_box"
    if x <= 6:
        return "six_yard_box"
    if x <= 22:
        return "inside_box"
    if x <= 35:
        return "edge_of_box"
    return "outside_box"


def probe(clubs: List[str], n: int, seed: int) -> Dict[str, Any]:
    loader = get_loader()
    pairs = _pick_clubs(clubs, n)
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: dict(
        shots=0, sot=0, goals=0, xg=0.0,
    ))
    long_examples: List[Dict[str, Any]] = []
    totals = dict(shots=0, sot=0, goals=0, xg=0.0, passes=0, corners=0)
    agg["corner_src"] = {}   # str source -> count (awarded corners)
    t0 = time.time()

    for i, (h, a) in enumerate(pairs, 1):
        random.seed(seed + i * 1000)
        h_raw = loader.build_matchday_squad(h)
        a_raw = loader.build_matchday_squad(a)
        hs = _team_profile(h, i, True)
        as_ = _team_profile(a, i, False)
        h_sq = SquadBuilder.build(team_name=h, starters=h_raw["starters"],
                                  substitutes=h_raw["substitutes"],
                                  team_superstars=h_raw["superstars"],
                                  set_piece_takers=h_raw["sp_takers"])
        a_sq = SquadBuilder.build(team_name=a, starters=a_raw["starters"],
                                  substitutes=a_raw["substitutes"],
                                  team_superstars=a_raw["superstars"],
                                  set_piece_takers=a_raw["sp_takers"])
        cfg = MatchConfig(home_team=h, away_team=a, season="26/27",
                          competition="PLOFA-BATTLEFIELD", venue="BF Arena",
                          stadium_capacity=40000, referee="Battle Ref",
                          referee_strictness=0.5)
        mgr = ManagerPool(clubs=[h, a],
                          style_lookup={h: hs.style.value, a: as_.style.value})
        sub = SubstitutionController(home_team=h, away_team=a,
                                     home_subs_bench=h_sq["substitutes"],
                                     away_subs_bench=a_sq["substitutes"],
                                     home_style=hs.style.value,
                                     away_style=as_.style.value,
                                     manager_stubbornness=0.35)
        sub.MAX_SUBS = 3
        _register_sub_schedule(sub, {h: h_sq, a: a_sq})

        eng = MatchEngine(cfg, hs, as_)
        eng.set_squad(h, h_sq["starters"], h_sq["substitutes"])
        eng.set_squad(a, a_sq["starters"], a_sq["substitutes"])
        eng.set_stamina_controller(sub)
        eng.set_managers(home_manager=mgr.manager_for(h),
                         away_manager=mgr.manager_for(a))
        eng.quiet = True
        with contextlib.redirect_stdout(io.StringIO()):
            res = eng.simulate()

        for e in res.timeline:
            et = getattr(e, "event_type", None)
            name = getattr(et, "name", "") if et else ""
            team = getattr(e, "team", "")
            if name == "CORNER_WON":
                m = getattr(e, "metadata", {}) or {}
                src = (m.get("from_save_deflection") or m.get("from_woodwork")
                       or m.get("from_shot_block") or m.get("deflection")
                       or m.get("from_byline") or m.get("from_clearance")
                       or "other")
                agg["corner_src"][str(src)] = agg["corner_src"].get(str(src), 0) + 1
                continue
            if name == "CORNER_TAKEN":
                totals["corners"] += 1
                continue
            if name in _ATTEMPTS:
                totals["passes"] += 1
                continue
            if name == "GOAL":
                z = describe_zone(e, h)
                agg[z]["goals"] += 1
                totals["goals"] += 1
                continue
            is_shot = name in _SOT or name in _SHOT_OFF or name in _PEN
            if not is_shot:
                continue
            z = "penalty" if name in _PEN else describe_zone(e, h)
            if z == "outside_box" and len(long_examples) < 12:
                long_examples.append({
                    "x": round(getattr(e, "location_x", -1), 1),
                    "y": round(getattr(e, "location_y", -1), 1),
                    "situation": name,
                    "meta": getattr(e, "metadata", None),
                })
            xg = 0.79 if name in _PEN else (getattr(e, "xg", 0.0) or 0.0)
            k = agg[z]
            k["shots"] += 1
            k["xg"] += xg
            totals["shots"] += 1
            totals["xg"] += xg
            if name in _SOT or name == "PENALTY_SCORED":
                k["sot"] += 1
                totals["sot"] += 1
        if i % 4 == 0 or i == n:
            per = (time.time() - t0) / i
            print(f"  [{i}/{n}] {res.state.home_goals}-{res.state.away_goals} "
                  f"({per:.1f}s/match, ETA {(n-i)*per:.0f}s)", flush=True)

    totals["matches"] = n
    return dict(agg=dict(agg), totals=totals, long_examples=long_examples)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=12)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    loader = get_loader()
    clubs = loader.get_all_clubs()
    out = probe(clubs, args.matches, args.seed)

    print("\n=== LONG-RANGE SHOT EXAMPLES ===")
    for ex in out.get("long_examples", []):
        print(ex)

    print("\n=== PER-ZONE (per 100 shots) ===")
    print(f"{'zone':<12}{'n%':>6}{'xg/shot':>9}{'sot%':>7}{'goals/shot':>11}")
    for z, k in sorted(out["agg"].items(), key=lambda kv: -kv[1].get("shots", -1)):
        if "shots" not in k:
            continue
        n = k["shots"]
        pct = 100.0 * n / out["totals"]["shots"]
        print(f"{z:<12}{pct:6.1f}{k['xg']/n:9.3f}{100*k['sot']/n:7.1f}"
              f"{100*k['goals']/n:11.1f}")
    print("\n=== CORNER SOURCES (per team-match) ===")
    tm = out["totals"]["matches"] * 2
    for src, cnt in sorted(out["agg"].get("corner_src", {}).items(),
                           key=lambda kv: -kv[1]):
        print(f"  {src:<28}{cnt/tm:8.2f}")
    t = out["totals"]
    print("\n=== TOTALS (per team-match) ===")
    tm = t["matches"] * 2
    print(f" shots           {t['shots']/tm:6.2f}")
    print(f" xg              {t['xg']/tm:6.3f}")
    print(f" sot-share       {100*t['sot']/max(1,t['shots']):6.1f}%")
    print(f" goals           {t['goals']/tm:6.3f}")
    print(f" goals/xg ratio  {t['goals']/max(0.001,t['xg']):6.3f}")
    print(f" passes          {t['passes']/tm:6.1f}")
    print(f" corners         {t['corners']/tm:6.2f}")


if __name__ == "__main__":
    main()