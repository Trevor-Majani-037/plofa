"""Real-Excel-roster brain audition — WATCHABLE, NON-PERSISTENT.

Runs one or more brain XIs on the REAL production roster path (real Excel
roster -> build_matchday_squad -> SquadBuilder -> MatchEngine, real
availability/stamina/soul glue) so the user can WATCH the football and
judge with their eyes, not metrics.  Mirrors auto_run_match.run() UP TO
simulate() but deliberately OMITS every persistence step — no
SeasonState.save(), exporter, referee record, manager save.  A before/after
snapshot proves zero season writes.

Usage:
    .venv\\Scripts\\python.exe audit_roster_match.py
    .venv\\Scripts\\python.exe audit_roster_match.py --brains "brains,brains_v2"
    .venv\\Scripts\\python.exe audit_roster_match.py --brains brains_v2 --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from datetime import date

from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder
from squad_manager import SubstitutionController
from roster_loader import get_loader

from auto_run_match import (
    TEAM_CATALOG,
    SOUL_PLAYERS,
    _resolve_team_profile,
    _build_availability,
    _apply_starting_stamina,
    _attach_souls,
)

from brain_integration import set_brain_dir, clear_registry

SEASON_STATE_FILE = "season_state.json"
OUTPUTS_DIR = "plofa_output"
WATCH_FILES = [
    SEASON_STATE_FILE,
    "manager_state.json",
    "referee_state.json",
    "fixtures.json",
    OUTPUTS_DIR,
]


def _snapshot() -> dict:
    snap = {}
    for path in WATCH_FILES:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                snap[path] = ("file", hashlib.md5(f.read()).hexdigest())
        elif os.path.isdir(path):
            entries = {}
            for root, _dirs, files in os.walk(path):
                for name in sorted(files):
                    fp = os.path.join(root, name)
                    rel = os.path.relpath(fp, path)
                    st = os.stat(fp)
                    entries[rel] = (st.st_size, st.st_mtime_ns)
            snap[path] = ("dir", entries)
        else:
            snap[path] = ("missing", None)
    return snap


def run_match(brains_dir: str, home: str, away: str, matchday: int,
              season: str, match_date: date, seed: int) -> dict:
    set_brain_dir(brains_dir)
    clear_registry()

    loader = get_loader()
    from season_manager import SeasonState
    season_state = SeasonState(season, SEASON_STATE_FILE)
    home_avail = _build_availability(home, matchday, season_state,
                                     OUTPUTS_DIR, match_date)
    away_avail = _build_availability(away, matchday, season_state,
                                     OUTPUTS_DIR, match_date)
    home_raw = loader.build_matchday_squad(home, availability=home_avail)
    away_raw = loader.build_matchday_squad(away, availability=away_avail)
    home_form, away_form = home_raw["formation"], away_raw["formation"]
    home_style = _resolve_team_profile(home, home_form, is_home=True)
    away_style = _resolve_team_profile(away, away_form, is_home=False)

    home_squad = SquadBuilder.build(
        team_name=home, starters=home_raw["starters"],
        substitutes=home_raw["substitutes"],
        team_superstars=home_raw["superstars"],
        set_piece_takers=home_raw["sp_takers"])
    away_squad = SquadBuilder.build(
        team_name=away, starters=away_raw["starters"],
        substitutes=away_raw["substitutes"],
        team_superstars=away_raw["superstars"],
        set_piece_takers=away_raw["sp_takers"])

    all_players = (home_squad["starters"] + home_squad["substitutes"]
                   + away_squad["starters"] + away_squad["substitutes"])
    _apply_starting_stamina(all_players, {**home_avail, **away_avail},
                            season_state, match_date)
    souls = _attach_souls(all_players)

    print(f"  X: {home}  ({home_form})")
    for p in home_squad["starters"]:
        print(f"     {p.position:<4} {p.name}")
    print(f"  X: {away}  ({away_form})")
    for p in away_squad["starters"]:
        print(f"     {p.position:<4} {p.name}")
    if souls:
        print(f"  🔮 souls active: {', '.join(getattr(p, 'name', '?') for p in souls)}")

    config = MatchConfig(
        home_team=home, away_team=away,
        match_date=match_date, matchday=matchday, season=season,
        venue=f"{home} Stadium", stadium_capacity=45000,
        referee="Audit Referee", referee_strictness=0.5,)
    sub_ctrl = SubstitutionController(
        home_team=home, away_team=away,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"])
    random.seed(seed)
    engine = MatchEngine(config, home_style, away_style)
    engine.set_squad(home, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(away, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(sub_ctrl)

    print(f"\n  ⚽ Simulating ({brains_dir} brains)...")
    result = engine.simulate()
    score_str = result.score_str
    print(f"  ⚽ FINAL: {score_str}")

    extras = {}
    for key in ("home_xg", "away_xg", "home_possession", "possession"):
        if hasattr(result, key):
            extras[key] = getattr(result, key)
    return {"score_str": score_str, "extras": extras}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", default="brains,brains_v2",
                    help="Comma-separated brain dirs to audition (default T1 then V3).")
    ap.add_argument("--home", default="Natrican")
    ap.add_argument("--away", default="Tryox City")
    ap.add_argument("--matchday", type=int, default=3)
    ap.add_argument("--season", default="26/27")
    ap.add_argument("--date", default="2026-09-06")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    match_date = date.fromisoformat(args.date)
    before = _snapshot()

    print(f"REAL-ROSTER AUDITION (non-persistent, watchable) — seed {args.seed}")
    print(f"  {args.home} vs {args.away} | MD{args.matchday} | {match_date}\n")

    loader = get_loader()
    clubs = loader.get_all_clubs()
    for team in (args.home, args.away):
        if team not in clubs:
            print(f"  ❌ Team '{team}' not found. Available:")
            print(f"     {', '.join(sorted(clubs))}")
            sys.exit(1)

    results = {}
    for i, d in enumerate(args.brains.split(",")):
        d = d.strip()
        if not os.path.isdir(d):
            print(f"  ❌ brains dir not found: {d}")
            sys.exit(1)
        print("\n" + "═" * 72)
        print(f"  ▶ ARM {i+1}: brains = {d}")
        print("═" * 72)
        results[d] = run_match(d, args.home, args.away, args.matchday,
                               args.season, match_date, args.seed)

    after = _snapshot()
    writes = [p for p in WATCH_FILES if before.get(p) != after.get(p)]

    print("\n" + "═" * 72)
    print("  AUDITION SUMMARY")
    print("═" * 72)
    for d, r in results.items():
        extra = ""
        if r["extras"]:
            extra = "  " + "  ".join(f"{k}={v}" for k, v in r["extras"].items())
        print(f"  {d:12s}: {r['score_str']}{extra}")
    if writes:
        print(f"  ⚠ zero-write FAIL: {', '.join(writes)}")
        sys.exit(1)
    print("  ✅ zero season-data writes — ran in-memory only.")


if __name__ == "__main__":
    main()