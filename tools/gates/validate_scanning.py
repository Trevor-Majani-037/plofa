"""SCANNING CENSUS — is the head-scan model realistic yet?

WHY THIS EXISTS
---------------
`ball_vision.py` grew a "human perception stack" (2026-10-27): head-turn
toward the ball, a periodic scan sweep, peripheral glimpses, fatigue-aware
FOV and anticipation lead-time.  It reads plausibly.  It has never been
measured against the thing it claims to model, and the units are not
obviously right — the scan cadence is computed from `minute`, which is the
match MINUTE, not seconds.

WHAT IT MEASURES
----------------
1. BALL SCANS, per player per minute of match time:
     - deliberate sweeps fired (`track.last_scan_time` changed)
     - mean gap between sweeps (seconds)
     - fraction of ticks where the head is turned more than 2 degrees off
       the attacking direction
     - fraction of calls where the ball is visible
2. ACTOR VISIBILITY, sampled once per minute for every outfield player:
     - actors (of the other 21) inside the constant-forward FOV cone
     - actors inside the SAME cone rotated by the head-turn facing
   If those two numbers are equal, the head-turn cannot help a player see
   pressure, teammates or runners — only the ball.  That is the difference
   between "scanning" and "a neck that tracks the ball".

The real benchmark used throughout: elite players scan about 0.6 times per
second when the ball is behind them and 0.2-0.4 /s otherwise (Jordet et al.),
i.e. roughly 12-40 deliberate glances per minute, each lasting ~0.2-0.5 s.

Usage
-----
    .venv\\Scripts\\python.exe tools/gates/validate_scanning.py --matches 1
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from datetime import date

import numpy as np

import ball_vision
import match_probe
import perception as perception_mod
from copy import deepcopy


def _unique_squad(squad, prefix: str):
    """Copy with globally unique player names (shared position engine keys
    every player by bare name — both probe teams reuse ST/CB1/...)."""
    squad = deepcopy(squad)
    for group in ("starters", "substitutes"):
        for player in squad.get(group, []):
            player.dna.name = f"{prefix}_{player.dna.name}"
    return squad


def _actor_visibility(engine) -> dict:
    """Once-per-minute: how many of the other 21 can each player see?

    Compared under two facings:
      FORWARD — the constant attacking-direction cone the actor path uses
      HEAD    — the same cone rotated by the head-turn facing
    """
    pe = engine.position_engine
    seen_fwd = []
    seen_head = []
    for team in (engine.config.home_team, engine.config.away_team):
        att_right = pe.team_attacks_right.get(team, True)
        others = (engine.config.away_team if team == engine.config.home_team
                  else engine.config.home_team)
        for p in engine.active_players.get(team, []):
            if getattr(p, "position", "") == "GK":
                continue
            st = pe.states.get(p.name)
            if st is None or st.current_x is None:
                continue
            try:
                from perception import _resolve_profile, _mental_scale, _get_attr
                cfg = perception_mod.get_perception_config()
                prof = _resolve_profile(getattr(p, "position", ""), cfg)
                power, _acc = _mental_scale(p)
                radius = (cfg.range_min + (cfg.range_max - cfg.range_min)
                          * power) * prof["range_scale"]
                fov = math.radians(prof["fov_deg"])
            except Exception:
                continue
            ball_x = engine.state.last_ball_x
            ball_y = engine.state.last_ball_y
            head = perception_mod._facing_angle(
                p, st.current_x, st.current_y, ball_x, ball_y, att_right)
            fwd = 0.0 if att_right else math.pi
            n_fwd = n_head = 0
            for n in pe.team_rosters.get(others, []) + \
                    [x for x in pe.team_rosters.get(team, []) if x != p.name]:
                st2 = pe.states.get(n)
                if st2 is None or st2.current_x is None:
                    continue
                if getattr(st2, "position", "") == "GK":
                    continue
                d = math.hypot(st2.current_x - st.current_x,
                               st2.current_y - st.current_y)
                if d > radius:
                    continue
                b_fwd = abs(perception_mod._bearing_diff(
                    st.current_x, st.current_y,
                    st2.current_x, st2.current_y, fwd))
                b_head = abs(perception_mod._bearing_diff(
                    st.current_x, st.current_y,
                    st2.current_x, st2.current_y, head))
                if b_fwd <= fov / 2.0:
                    n_fwd += 1
                if b_head <= fov / 2.0:
                    n_head += 1
            seen_fwd.append(n_fwd)
            seen_head.append(n_head)
    return {
        "samples": len(seen_fwd),
        "mean_seen_forward": statistics.mean(seen_fwd) if seen_fwd else 0.0,
        "mean_seen_headturn": statistics.mean(seen_head) if seen_head else 0.0,
    }


def run_match(seed: int, home_style="balanced", away_style="fluid_counter"):
    random.seed(seed)
    home_team, away_team = "Probe FC", "Rival FC"
    built_home, built_away = match_probe._build_squads(home_team, away_team)
    home_squad = _unique_squad(built_home, "Probe")
    away_squad = _unique_squad(built_away, "Rival")
    config = match_probe.MatchConfig(
        home_team=home_team, away_team=away_team,
        match_date=date(2026, 9, 6), matchday=3, season="26/27")
    hp = match_probe._team_profile(home_team, home_style)
    ap = match_probe._team_profile(away_team, away_style)
    eng = match_probe.MatchEngine(config, hp, ap)
    eng.set_squad(home_team, home_squad["starters"], home_squad["substitutes"])
    eng.set_squad(away_team, away_squad["starters"], away_squad["substitutes"])

    stats = {
        "calls": 0, "visible": 0, "head_turned": 0,
        "scans": 0, "scan_gaps": [], "players": set(),
        "sigma": [], "minutes": set(),
    }
    _last_scan = {}
    _orig_ball = ball_vision.perceive_ball

    def _patched_ball(engine_, pname, player, attacks_right, tx, ty, minute):
        out = _orig_ball(engine_, pname, player, attacks_right, tx, ty, minute)
        stats["calls"] += 1
        stats["players"].add(pname)
        stats["minutes"].add(int(minute))
        px, py, vis, sigma = out
        if vis:
            stats["visible"] += 1
        stats["sigma"].append(float(sigma))
        st = engine_.position_engine.states.get(pname)
        if st is not None and st.current_x is not None:
            head = perception_mod._facing_angle(
                player, st.current_x, st.current_y, tx, ty, attacks_right)
            fwd = 0.0 if attacks_right else math.pi
            if abs(perception_mod._bearing_diff(
                    st.current_x, st.current_y, tx, ty, head)) > math.radians(1.0):
                stats["head_turned"] += 1
        try:
            track = ball_vision._tracks_for(engine_).get(pname)
        except Exception:
            track = None
        if track is not None:
            prev = _last_scan.get(pname)
            if prev is not None and track.last_scan_time != prev:
                stats["scans"] += 1
                gap = (minute - prev) * 60.0
                if 0.0 < gap < 3600.0:
                    stats["scan_gaps"].append(gap)
            _last_scan[pname] = track.last_scan_time
        return out

    # Actor-visibility census, once per match minute.
    _orig_pc = eng.position_engine.update_pitch_control
    _actor = {"rows": []}

    def _pc(*a, **k):
        out = _orig_pc(*a, **k)
        m = int(getattr(eng.state, "minute", 0) or 0)
        if m not in _actor.setdefault("seen_minutes", set()):
            _actor["seen_minutes"].add(m)
            _actor["rows"].append(_actor_visibility(eng))
        return out

    ball_vision.perceive_ball = _patched_ball
    eng.position_engine.update_pitch_control = _pc
    try:
        result = eng.simulate()
    finally:
        ball_vision.perceive_ball = _orig_ball

    rows = [r for r in _actor["rows"] if r["samples"]]
    actor = {
        "samples": sum(r["samples"] for r in rows),
        "mean_seen_forward": statistics.mean(r["mean_seen_forward"] for r in rows) if rows else 0.0,
        "mean_seen_headturn": statistics.mean(r["mean_seen_headturn"] for r in rows) if rows else 0.0,
    }
    return result, stats, actor


def main():
    p = argparse.ArgumentParser(description="Scanning census gate.")
    p.add_argument("--matches", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--json", default="")
    args = p.parse_args()

    all_stats, all_actor = [], []
    for m in range(args.matches):
        seed = args.seed + m * 100
        result, stats, actor = run_match(seed)
        all_stats.append(stats)
        all_actor.append(actor)
        print(f"  m{m+1} seed={seed}: {result.home_goals}-{result.away_goals}", flush=True)

    calls = sum(s["calls"] for s in all_stats)
    scans = sum(s["scans"] for s in all_stats)
    players = set().union(*[s["players"] for s in all_stats])
    minutes = set().union(*[s["minutes"] for s in all_stats])
    gaps = [g for s in all_stats for g in s["scan_gaps"]]
    sigma = [v for s in all_stats for v in s["sigma"]]
    head = sum(s["head_turned"] for s in all_stats)
    vis = sum(s["visible"] for s in all_stats)
    matches = args.matches
    outfield_ticks = len(players) * max(1, len(minutes))

    print("\n" + "=" * 72)
    print("SCANNING CENSUS")
    print("=" * 72)
    print(f"  ball-vision calls            : {calls}")
    print(f"  distinct players             : {len(players)}")
    print(f"  match minutes sampled        : {len(minutes)}")
    print(f"  ball visible                 : {vis/calls*100:5.1f}%")
    print(f"  head turned off forward axis : {head/calls*100:5.1f}%")
    print(f"  deliberate scan sweeps       : {scans}")
    denom = max(1, len(players)) * matches * max(1, len(minutes))
    per_pm = scans / denom
    print(f"  scans per player per minute  : {per_pm:.4f}"
          f"   ({scans/max(1,len(players)*matches):.2f} per player per match)")
    if gaps:
        print(f"  mean gap between scans       : {statistics.mean(gaps):8.1f} s"
              f"   median {statistics.median(gaps):8.1f} s")
    if sigma:
        print(f"  last-known sigma             : mean {statistics.mean(sigma):5.2f} m"
              f"  max {max(sigma):5.2f} m")
    a = all_actor[0]
    print(f"\n  actors visible (of 20 outfield):")
    print(f"      forward cone   {a['mean_seen_forward']:5.2f}")
    print(f"      head-turn cone {a['mean_seen_headturn']:5.2f}")
    delta = a["mean_seen_headturn"] - a["mean_seen_forward"]
    print(f"      head-turn gain {delta:+5.2f}  "
          f"{'-> scanning cannot reveal a single extra player'
             if abs(delta) < 0.05 else '-> scanning widens the actor world'}")

    print("\n" + "=" * 72)
    print("BENCHMARK (Jordet et al.): 12-40 deliberate glances per player per")
    print("minute (0.2-0.6 /s), each ~0.2-0.5 s, and a glance is what lets a")
    print("player see PRESSURE and RUNNERS, not the ball.")
    print("=" * 72)
    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"calls": calls, "visible_frac": vis/calls if calls else 0,
                       "head_turned_frac": head/calls if calls else 0,
                       "scans": scans, "scan_gap_mean_s":
                       statistics.mean(gaps) if gaps else None,
                       "sigma_mean": statistics.mean(sigma) if sigma else None,
                       "actor": a}, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
