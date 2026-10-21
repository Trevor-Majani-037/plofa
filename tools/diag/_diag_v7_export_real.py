"""THE EYE GATE, on a REAL roster — the export the plan wants to be looked at.

WHY THE REAL ROSTER
-------------------
The scratch fixture's export is a TRAP: `match_probe._build_squads("Probe
FC", "Rival FC")` gives BOTH teams the same generic names (GK, CB1, CB2,
LB, RB, CDM, CM1, CM2, LW, ST, RW, SUB1-3), and the exporter's accumulator
keys its per-player stats BY NAME (`exporter.py:413`,
`self.stats[p.name] = ...`).  So the home team's rows are overwritten by
the away team's, the players sheet contains only Rival FC, and **the
pass-network plot draws an EMPTY PITCH for the home team** — with the
pitch, the frame and the attacking arrow all perfectly drawn, so it reads
as "no passes happened" when in fact the home team had 234 of them.

**That is a fixture artefact, not an export defect** (the real roster has
distinct names, and the scratch runner `run_match.py` uses the real
Hartwell/Thornfield squads ✓), and it is recorded in AGENTS.md because a
blank panel is exactly the kind of thing a probe invites you to
misdiagnose.

This probe therefore runs a REAL-ROSTER match (the production squad
builder, read-only) with the given brain set bound by POSITION, and writes
the export to a scratch folder.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_v7_export_real.py brains_v7 \\
        --home Natrican --away Tryox City --out plofa_output/_v7_gate_real
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "gates"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("brains", nargs="?", default="brains_v7")
    ap.add_argument("--home", default="Natrican")
    ap.add_argument("--away", default="Tryox City")
    ap.add_argument("--matchday", type=int, default=3)
    ap.add_argument("--season", default="26/27")
    ap.add_argument("--date", default="2026-09-06")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="plofa_output/_v7_gate_real")
    args = ap.parse_args()

    from brain_integration import set_brain_dir
    # The brain set is bound BY NAME for the HOME team only, so the away
    # team auto-loads the SHIPPED set (the default brain dir) and the
    # export compares the two.  Binding the whole set to BOTH teams would
    # have made the eye gate a v7-vs-v7 match — which is a different
    # question, and one that looks like a comparison because the two
    # networks are drawn in the same figure.
    set_brain_dir("brains")
    from football_brain import FootballBrain
    from brain_integration import register_brain, clear_registry
    from validate_neural_xl import POSITION_BRAIN_MAP

    from match_engine import MatchConfig, MatchEngine
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    from squad_manager import SubstitutionController
    import production_roster_sanity as prs

    match_date = date.fromisoformat(args.date)
    loader = get_loader()
    clubs = loader.get_all_clubs()
    for team in (args.home, args.away):
        if team not in clubs:
            print(f"  team {team!r} not in the roster: {sorted(clubs)}")
            return 1

    from season_manager import SeasonState
    season_state = SeasonState(args.season, prs.SEASON_STATE_FILE)
    home_avail = prs._build_availability(args.home, args.matchday, season_state,
                                        prs.OUTPUTS_DIR, match_date)
    away_avail = prs._build_availability(args.away, args.matchday, season_state,
                                        prs.OUTPUTS_DIR, match_date)
    home_raw = loader.build_matchday_squad(args.home, availability=home_avail)
    away_raw = loader.build_matchday_squad(args.away, availability=away_avail)
    home_style = prs._resolve_team_profile(args.home, home_raw["formation"],
                                           is_home=True)
    away_style = prs._resolve_team_profile(args.away, away_raw["formation"],
                                           is_home=False)
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
    prs._apply_starting_stamina(all_players, {**home_avail, **away_avail},
                                season_state, match_date)
    prs._attach_souls(all_players)

    # Bind the CHALLENGER set to the home team only, by name.
    clear_registry()
    bound = 0
    for p in home_squad["starters"]:
        stem = POSITION_BRAIN_MAP.get(p.position)
        if not stem:
            continue
        path = os.path.join(args.brains, f"{stem}.json")
        if not os.path.exists(path):
            continue
        register_brain(p.name, FootballBrain.load(path))
        bound += 1
    print(f"challenger brains bound to the home XI: {bound}")

    config = MatchConfig(
        home_team=args.home, away_team=args.away,
        match_date=match_date, matchday=args.matchday, season=args.season,
        venue=f"{args.home} Stadium", stadium_capacity=45000,
        referee="Sanity Referee", referee_strictness=0.5)
    sub = SubstitutionController(
        home_team=args.home, away_team=args.away,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"])
    random.seed(args.seed)
    engine = MatchEngine(config, home_style, away_style)
    engine.set_squad(args.home, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(args.away, away_squad["starters"], away_squad["substitutes"])
    engine.set_stamina_controller(sub)
    result = engine.simulate()
    print(result.summary())

    from exporter import PLOFAExporter
    all_players_by_team = {args.home: home_squad, args.away: away_squad}
    out = args.out
    os.makedirs(out, exist_ok=True)
    exporter = PLOFAExporter(result=result, all_players=all_players_by_team,
                             home_color="#C8102E", away_color="#FFFFFF",
                             sub_controller=sub)
    exporter.export_all(out)
    print(f"\nexport written to {out}")
    for root, _dirs, files in os.walk(out):
        for f in sorted(files):
            print("   ", os.path.join(root, f).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
