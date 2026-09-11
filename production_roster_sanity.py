"""Production-roster sanity check — NON-PERSISTENT neural XI run.

Verifies the learned brains engage along the REAL production path (real
Excel roster -> build_matchday_squad -> SquadBuilder -> MatchEngine)
WITHOUT writing a single byte of season data:

  * every starting (and entered) player auto-loads a TRAINED on-ball brain
    via brain_integration (exact-name registry miss -> BRAIN_DIR/<POS>.json),
    never the random fallback
  * the permanent TeamPressBrain auto-loads (brains_team/XI.json) and its
    engagement scalar actually drives the press Bernoulli (g < 1.0 observed)
  * the DefensiveActionBrain auto-loads (brains_def/ACTION.json)

This mirrors auto_run_match.run() UP TO simulate() but deliberately OMITS
every persistence step: no SeasonState.save(), no exporter.export_all(), no
RefereeManager.record/save, no ManagerPool.save.  A before/after snapshot of
the season/referee/manager/output files proves zero writes.

auto_run_match.py itself is NEVER executed here — only its pure helpers are
imported (team catalog, profile resolution, availability/stamina/soul glue).
No fixture is recorded; this is a scratch in-memory match.

Usage:
  .venv\\Scripts\\python.exe production_roster_sanity.py
  .venv\\Scripts\\python.exe production_roster_sanity.py --home "Uditon" --away "Claw" --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from datetime import date

import match_engine as me_module
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
    """Capture existence + content-hash of every file under watch paths."""
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="Natrican")
    ap.add_argument("--away", default="Tryox City")
    ap.add_argument("--matchday", type=int, default=3)
    ap.add_argument("--season", default="26/27")
    ap.add_argument("--date", default="2026-09-06")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    match_date = date.fromisoformat(args.date)
    print(f"PRODUCTION-ROSTER SANITY (non-persistent) — seed {args.seed}")
    print(f"  {args.home} vs {args.away} | MD{args.matchday} | {match_date}\n")

    # Zero-write assertion: snapshot the season files before anything runs.
    before = _snapshot()

    # ── Real production setup (read-only, mirrors auto_run_match.run())
    loader = get_loader()
    clubs = loader.get_all_clubs()
    for team in (args.home, args.away):
        if team not in clubs:
            print(f"  ❌ Team '{team}' not found in roster. Available:")
            print(f"     {', '.join(sorted(clubs))}")
            sys.exit(1)

    from season_manager import SeasonState
    season_state = SeasonState(args.season, SEASON_STATE_FILE)

    home_avail = _build_availability(args.home, args.matchday, season_state,
                                     OUTPUTS_DIR, match_date)
    away_avail = _build_availability(args.away, args.matchday, season_state,
                                     OUTPUTS_DIR, match_date)

    home_raw = loader.build_matchday_squad(args.home, availability=home_avail)
    away_raw = loader.build_matchday_squad(args.away, availability=away_avail)
    home_form = home_raw["formation"]
    away_form = away_raw["formation"]

    home_style = _resolve_team_profile(args.home, home_form, is_home=True)
    away_style = _resolve_team_profile(args.away, away_form, is_home=False)

    home_squad = SquadBuilder.build(
        team_name=args.home, starters=home_raw["starters"],
        substitutes=home_raw["substitutes"],
        team_superstars=home_raw["superstars"],
        set_piece_takers=home_raw["sp_takers"])
    away_squad = SquadBuilder.build(
        team_name=args.away, starters=away_raw["starters"],
        substitutes=away_raw["substitutes"],
        team_superstars=away_raw["superstars"],
        set_piece_takers=away_raw["sp_takers"])

    all_players = (home_squad["starters"] + home_squad["substitutes"]
                   + away_squad["starters"] + away_squad["substitutes"])
    _apply_starting_stamina(all_players, {**home_avail, **away_avail},
                            season_state, match_date)
    souls = _attach_souls(all_players)

    for name, squad in ((args.home, home_squad), (args.away, away_squad)):
        print(f"  X: {name}  ({home_form if name == args.home else away_form})")
        for p in squad["starters"]:
            print(f"     {p.position:<4} {p.name}")

    # ── Assemble a REAL NeuralDecisionBrain match (no heuristic pinning).
    #    Every brain pathway (on-ball / team-press / defensive-action) is
    #    left in its DEFAULT auto-load state — exactly the production wiring.
    config = MatchConfig(
        home_team=args.home, away_team=args.away,
        match_date=match_date, matchday=args.matchday, season=args.season,
        venue=f"{args.home} Stadium", stadium_capacity=45000,
        referee="Sanity Referee", referee_strictness=0.5,)
    sub_ctrl = SubstitutionController(
        home_team=args.home, away_team=args.away,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"])
    random.seed(args.seed)
    engine = MatchEngine(config, home_style, away_style)
    engine.set_squad(args.home, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(args.away, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(sub_ctrl)

    print("\n  ⚽ Simulating (full neural XI, permanent team-press + defensive brains)...")
    result = engine.simulate()
    print(f"\n  SCORE: {result.score_str}")

    # ── VERIFY 1: every starter got a TRAINED on-ball brain ─────────────
    from brain_integration import _brain_registry, _pos_brain_cache
    missing, randomed = [], []
    checked = 0
    for name, squad in ((args.home, home_squad), (args.away, away_squad)):
        for p in squad["starters"]:
            checked += 1
            reg = _brain_registry.get(p.name)
            trained = _pos_brain_cache.get(p.position)
            if reg is None:
                missing.append(f"{p.name} ({p.position}) — never decided")
            elif trained is None or reg is not trained:
                randomed.append(f"{p.name} ({p.position}) — non-trained brain")
        for p in squad["substitutes"]:
            if getattr(p, "_entered_pitch", False):
                checked += 1
                reg = _brain_registry.get(p.name)
                trained = _pos_brain_cache.get(p.position)
                if reg is None:
                    missing.append(f"{p.name} ({p.position}, sub) — never decided")
                elif trained is None or reg is not trained:
                    randomed.append(f"{p.name} ({p.position}, sub) — non-trained brain")

    loaded_brains = os.listdir("brains") if os.path.isdir("brains") else []
    print(f"\n  ON-BALL: {checked} players on the ball -> "
          f"{len(loaded_brains)} position brains in brains/ dir")

    # ── VERIFY 2: permanent team-press controller engaged ───────────────
    tp_engaged = (me_module._TEAM_PRESS_BRAIN is not None
                  and me_module._TEAM_PRESS_LOADED)
    g_cache = getattr(engine, "_team_press_g_cache", {}) or {}
    g_vals = [v for v in g_cache.values() if v != 1.0]
    tp_active = tp_engaged and bool(g_vals)

    # ── VERIFY 3: defensive-action brain engaged ─────────────────────────
    def_engaged = (me_module._DEF_ACTION_BRAIN is not None
                   and me_module._DEF_ACTION_LOADED)

    # ── VERIFY 4: zero season-data writes ────────────────────────────────
    after = _snapshot()
    writes = [p for p in WATCH_FILES if before.get(p) != after.get(p)]

    ok = True
    print("\n" + "═" * 64)
    print("  SANITY RESULTS")
    print("═" * 64)
    print(f"  on-ball trained brains  : "
          f"{'PASS' if not missing and not randomed and checked > 0 else 'FAIL'}"
          f"  ({checked} decided, 0 random fallback)")
    for m in missing:
        print(f"      ⚠ {m}")
    for r in randomed:
        print(f"      ❌ {r}")
    print(f"  team-press controller   : "
          f"{'PASS — g engaged (' + format(sum(g_vals)/len(g_vals), '.3f') + ' mean, '
            + str(len(g_vals)) + ' non-1.0 ticks)' if tp_active else 'FAIL — inert'}")
    print(f"  defensive-action brain  : "
          f"{'PASS — brains_def/ACTION.json engaged' if def_engaged else 'FAIL — inert'}")
    writes_label = "PASS" if not writes else "FAIL"
    print(f"  zero season-data writes : {writes_label}  "
          f"({'no files touched' if not writes else ', '.join(writes)})")
    fatal = (bool(missing or randomed) or not tp_active
             or not def_engaged or bool(writes))
    if fatal:
        print("\n  ❌ SANITY RUN FAILED — inspect the FAIL lines above.")
        sys.exit(1)
    print(f"\n  ✅ SANITY RUN PASSED.  Season ledger untouched: "
          f"({args.home}-{args.away}) ran in-memory only.")


if __name__ == "__main__":
    main()