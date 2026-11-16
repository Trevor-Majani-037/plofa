"""STRIKER-PRESS CENSUS — is the opposing centre-forward the most-pressed man?

WHY THIS EXISTS
---------------
The league asks for an EPL-style centre-back duel: the two CBs should be
locked onto the opposing striker nearly all game, so that the striker is the
MOST PRESSED opponent on the pitch.  Nothing in the engine asserted that, and
nothing measured it.  A change that "adds striker pressing" without a census
is unfalsifiable — the same trap AGENTS.md records for the defensive-action
brain (a patch that called the original path and reported success).

WHAT IT MEASURES, PER SNAPSHOT
------------------------------
Sampled once per live tick (deduped on `match_clock_s`), for each defending
team:

  * `cb_min_m`   — distance from the striker to the NEAREST of the two CBs
  * `cb_pair_m`  — distance from the striker to the SECOND-nearest CB
  * `def_min_m`  — distance from the striker to the nearest defender of ANY
                   role (so a CDM screening him does not look like a free man)
  * per-role mean nearest-defender distance for every opponent position —
    the "is the striker actually the most pressed?" table

and reports the fraction of defending-tick snapshots inside 2 / 3 / 5 / 8 m,
plus the CB-pair's mean separation.

BENCHMARK
---------
A real CB pair against a lone striker in the Premier League spends the large
majority of out-of-possession time within ~5 m of him (marker touch-tight,
cover 4-8 m).  The gate thresholds are stated in `GATE` below; they are
deliberately weaker than "99%" because a striker who has run the channel and
a team in a deep block both legitimately spend time away from the ball-side
CB's reach.

Usage
-----
    .venv\\Scripts\\python.exe tools/gates/validate_striker_press.py --matches 3
    .venv\\Scripts\\python.exe tools/gates/validate_striker_press.py --matches 1 --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import date

import match_probe
from position_engine import PositionEngine
from squad_manager import SubstitutionController
from copy import deepcopy

#: Roles that count as "the striker" when the opponent has one on the pitch.
STRIKER_POSITIONS = ("ST", "CF", "SS")

#: Gate thresholds on the defending-tick census.  See module docstring.
GATE = {
    "cb_within_5m_frac": 0.60,    # at least one CB inside 5 m of the striker
    "cb_within_8m_frac": 0.85,    # at least one CB inside 8 m
    "striker_most_pressed": True,  # striker has the lowest mean nearest-defender
}


def _snapshot_hook(engine, census):
    """Build a per-tick census recorder for one engine."""
    pe = engine.position_engine
    state = engine.state
    last_tick = {"k": None}

    def record(has_ball_team):
        if not has_ball_team:
            return                    # possession unknown this tick
        key = round(float(state.match_clock_s), 1)
        if last_tick["k"] == (key, has_ball_team):
            return
        last_tick["k"] = (key, has_ball_team)
        for def_team in (engine.config.home_team, engine.config.away_team):
            if def_team == has_ball_team:
                continue          # this team is attacking; not a defending tick
            att_team = has_ball_team
            defenders = []
            strikers = []
            for n in pe.team_rosters.get(def_team, []):
                st = pe.states.get(n)
                if st is None or st.position == "GK":
                    continue
                if st.current_x is None:
                    continue
                defenders.append((n, st.position, st.current_x, st.current_y))
            for n in pe.team_rosters.get(att_team, []):
                st = pe.states.get(n)
                if st is None or st.position == "GK":
                    continue
                if st.current_x is None:
                    continue
                strikers.append((n, st.position, st.current_x, st.current_y))
            if not defenders or not strikers:
                continue
            cbs = [d for d in defenders if d[1] == "CB"]
            # The striker being defended: the opponent's most advanced ST/CF/SS
            # (closest to the defended goal).
            own_goal_x = 0.0 if pe.team_attacks_right.get(def_team, True) else 105.0
            cands = [s for s in strikers if s[1] in STRIKER_POSITIONS] or strikers
            sname, spos, sx, sy = min(
                cands, key=lambda s: abs(s[2] - own_goal_x))
            cb_d = sorted(math.hypot(c[2] - sx, c[3] - sy) for c in cbs)
            all_d = sorted(math.hypot(d[2] - sx, d[3] - sy) for d in defenders)
            rec = census[def_team]
            rec["ticks"] += 1
            rec["striker_pos"] = spos
            if len(cb_d) >= 1:
                rec["cb_min"].append(cb_d[0])
            if len(cb_d) >= 2:
                rec["cb_pair"].append(cb_d[1])
            rec["def_min"].append(all_d[0])
            # per-role nearest-defender distance, measured over EVERY opponent
            for (aname, apos, ax, ay) in strikers:
                nearest = min(math.hypot(d[2] - ax, d[3] - ay) for d in defenders)
                rec["role_min"][apos].append(nearest)
            # CB pair separation (are the two CBs a pair, or strangers?)
            if len(cbs) >= 2:
                rec["cb_sep"].append(math.hypot(cbs[0][2] - cbs[1][2],
                                                cbs[0][3] - cbs[1][3]))

    return record


def _unique_squad(squad, prefix: str):
    """Return a copy whose player names are globally unique.

    The shared position engine keys every spatial state by bare player name.
    The probe template uses `ST`, `CB1`, etc. for both teams, so one team's
    registry silently overwrites the other's and any per-team distance census
    becomes meaningless. Real squads have unique names; this fixture should
    too.
    """
    squad = deepcopy(squad)
    for group in ("starters", "substitutes"):
        for player in squad.get(group, []):
            player.dna.name = f"{prefix}_{player.dna.name}"
    return squad


def run_match(seed: int, home_style: str = "balanced",
              away_style: str = "fluid_counter", verbose: bool = False,
              press: bool = True):
    random.seed(seed)
    PositionEngine.STRIKER_PRESS_ENABLED = bool(press)
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
    eng.set_stamina_controller(SubstitutionController(
        home_team=home_team, away_team=away_team,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"]))

    census = {home_team: _new_rec(), away_team: _new_rec()}
    record = _snapshot_hook(eng, census)

    _orig_move = eng._offball_move_player

    def _patched(pname, team, bx, by, has_ball, danger_t, cross_team, DT,
                 include_on_ball=False):
        # `has_ball` here is PER PLAYER (does THIS man carry the ball), so
        # using it as "which team is defending" recorded the attacking side's
        # off-ball runners as defenders — which is why an earlier version of
        # this gate reported every opponent role being marked at 0.00 m. The
        # possession team is the only correct source, and the hook is deduped
        # on (clock, possession) so 11 calls per tick record one snapshot.
        record(getattr(eng.state, "possession_team", "") or "")
        return _orig_move(pname, team, bx, by, has_ball, danger_t, cross_team,
                          DT, include_on_ball=include_on_ball)

    eng._offball_move_player = _patched
    result = eng.simulate()
    return result, census


def _new_rec() -> dict:
    return {"ticks": 0, "striker_pos": "", "cb_min": [], "cb_pair": [],
            "def_min": [], "role_min": defaultdict(list), "cb_sep": []}


def _frac(vals, limit: float) -> float:
    return (sum(1 for v in vals if v <= limit) / len(vals)) if vals else 0.0


def summarise(census: dict) -> dict:
    out = {}
    for team, rec in census.items():
        if not rec["ticks"]:
            continue
        out[team] = {
            "ticks": rec["ticks"],
            "cb_min_mean": statistics.mean(rec["cb_min"]) if rec["cb_min"] else None,
            "cb_min_median": statistics.median(rec["cb_min"]) if rec["cb_min"] else None,
            "within_2m": _frac(rec["cb_min"], 2.0),
            "within_3m": _frac(rec["cb_min"], 3.0),
            "within_5m": _frac(rec["cb_min"], 5.0),
            "within_8m": _frac(rec["cb_min"], 8.0),
            "def_min_mean": statistics.mean(rec["def_min"]) if rec["def_min"] else None,
            "cb_sep_mean": statistics.mean(rec["cb_sep"]) if rec["cb_sep"] else None,
            "role_def_min_mean": {
                pos: statistics.mean(v) for pos, v in sorted(rec["role_min"].items())
                if v
            },
        }
    return out


def main():
    p = argparse.ArgumentParser(description="Striker-press census gate.")
    p.add_argument("--matches", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    p.add_argument("--json", default="")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-press", dest="press", action="store_false",
                   help="switch the CB striker-press constraint OFF "
                        "(the BEFORE arm of the A/B)")
    args = p.parse_args()

    merged = {}
    outcomes = []
    for m in range(args.matches):
        seed = args.seed + m * 100
        result, census = run_match(seed, args.home_style, args.away_style,
                                   press=args.press)
        outcomes.append((result.home_goals, result.away_goals,
                         result.home_possession_pct))
        print(f"  m{m+1} seed={seed}: {result.home_goals}-{result.away_goals} "
              f"poss {result.home_possession_pct:.1f}%", flush=True)
        for team, rec in census.items():
            tgt = merged.setdefault(team, _new_rec())
            tgt["ticks"] += rec["ticks"]
            for k in ("cb_min", "cb_pair", "def_min", "cb_sep"):
                tgt[k].extend(rec[k])
            for pos, v in rec["role_min"].items():
                tgt["role_min"][pos].extend(v)

    summary = summarise(merged)
    print("\n" + "=" * 72)
    print("STRIKER PRESS CENSUS")
    print("=" * 72)
    for team, s in summary.items():
        print(f"\n{team} (defending ticks {s['ticks']})")
        print(f"  nearest CB to opposing striker : mean {s['cb_min_mean']:.2f} m "
              f"median {s['cb_min_median']:.2f} m")
        print(f"  within 2 m {s['within_2m']*100:5.1f}%   within 3 m "
              f"{s['within_3m']*100:5.1f}%   within 5 m {s['within_5m']*100:5.1f}%"
              f"   within 8 m {s['within_8m']*100:5.1f}%")
        print(f"  nearest defender of any role  : mean {s['def_min_mean']:.2f} m")
        print(f"  CB pair separation            : mean {s['cb_sep_mean']:.2f} m")
        print("  mean nearest-defender distance by opponent role "
              "(lower = pressed harder):")
        for pos, v in sorted(s["role_def_min_mean"].items(), key=lambda kv: kv[1]):
            print(f"      {pos:4s} {v:6.2f} m")
        ranks = [pos for pos, _ in sorted(s["role_def_min_mean"].items(),
                                          key=lambda kv: kv[1])]
        sp = [p for p in ranks if p in STRIKER_POSITIONS]
        if sp:
            rank = ranks.index(sp[0]) + 1
            print(f"  striker rank among opponent roles: {rank} of {len(ranks)}")

    # ── GATE ──
    print("\n" + "=" * 72)
    print("GATE")
    print("=" * 72)
    ok = True
    for team, s in summary.items():
        w5, w8 = s["within_5m"], s["within_8m"]
        ranks = [pos for pos, _ in sorted(s["role_def_min_mean"].items(),
                                          key=lambda kv: kv[1])]
        sp = [p for p in ranks if p in STRIKER_POSITIONS]
        rank = ranks.index(sp[0]) + 1 if sp else 99
        p5 = w5 >= GATE["cb_within_5m_frac"]
        p8 = w8 >= GATE["cb_within_8m_frac"]
        pm = (rank == 1) if GATE["striker_most_pressed"] else True
        ok = ok and p5 and p8 and pm
        print(f"  {team:9s} CB<=5m {w5*100:5.1f}% (>={GATE['cb_within_5m_frac']*100:.0f}%)"
              f" {'PASS' if p5 else 'FAIL'} | CB<=8m {w8*100:5.1f}% "
              f"(>={GATE['cb_within_8m_frac']*100:.0f}%) {'PASS' if p8 else 'FAIL'}"
              f" | striker rank {rank} ({'PASS' if pm else 'FAIL'})")
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary,
                       "outcomes": [{"home": h, "away": a, "home_poss": p}
                                    for h, a, p in outcomes]}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
