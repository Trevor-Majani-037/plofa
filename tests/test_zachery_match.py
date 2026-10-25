"""Test match: Triumpher (with Zachery Worth) vs an opponent — non-persistent."""
from __future__ import annotations

import argparse
import sys
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")

from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder
from roster_loader import get_loader

from auto_run_match import (
    _resolve_team_profile,
    _build_availability,
    _apply_starting_stamina,
    _attach_souls,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="Triumpher")
    ap.add_argument("--away", default="Tryox City")
    ap.add_argument("--matchday", type=int, default=3)
    ap.add_argument("--season", default="26/27")
    ap.add_argument("--date", default="2026-09-06")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    match_date = date.fromisoformat(args.date)
    import random
    random.seed(args.seed)

    print(f"TEST MATCH (non-persistent) — seed {args.seed}")
    print(f"  {args.home} vs {args.away} | MD{args.matchday} | {match_date}\n")

    loader = get_loader()
    clubs = loader.get_all_clubs()
    for team in (args.home, args.away):
        if team not in clubs:
            print(f"  Team '{team}' not found. Available: {', '.join(sorted(clubs))}")
            sys.exit(1)

    from season_manager import SeasonState
    season_state = SeasonState(args.season, "season_state.json")

    home_avail = _build_availability(args.home, args.matchday, season_state, "plofa_output", match_date)
    away_avail = _build_availability(args.away, args.matchday, season_state, "plofa_output", match_date)

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

    _apply_starting_stamina(home_squad["starters"] + home_squad["substitutes"], home_avail, season_state, match_date)
    _apply_starting_stamina(away_squad["starters"] + away_squad["substitutes"], away_avail, season_state, match_date)
    _attach_souls(home_squad["starters"] + home_squad["substitutes"])
    _attach_souls(away_squad["starters"] + away_squad["substitutes"])

    all_players_flat = home_squad["starters"] + home_squad["substitutes"] + away_squad["starters"] + away_squad["substitutes"]

    # ── Wire Zachery (The Neymar) ──
    from zachery_the_chosen import wire_zachery
    zachery_player = None
    for p in all_players_flat:
        if p.name in ("Zachery Worth", "Zachery"):
            zachery_player = p
            break
    if zachery_player is not None:
        wire_zachery(player=zachery_player, train_if_missing=False)
        print(f"  Zachery Worth wired (The Neymar)")
    else:
        print("  Zachery not in squad")

    config = MatchConfig(
        home_team=args.home,
        away_team=args.away,
        match_date=match_date,
        matchday=args.matchday,
        season=args.season,
    )
    engine = MatchEngine(config, home_style, away_style)
    engine.set_squad(args.home, home_squad["starters"], home_squad["substitutes"])
    engine.set_squad(args.away, away_squad["starters"], away_squad["substitutes"])

    from squad_manager import SubstitutionController
    sub_ctrl = SubstitutionController(
        home_team=args.home, away_team=args.away,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"])
    engine.set_stamina_controller(sub_ctrl)

    random.seed(args.seed)
    result = engine.simulate()

    print(f"\n{'='*60}")
    print(f"  RESULT: {args.home} {result.home_goals} - {result.away_goals} {args.away}")
    print(f"{'='*60}")

    # Print lineups
    print(f"\n  Home starters: {[p.name + ' (' + p.position + ')' for p in home_squad['starters']]}")
    print(f"  Away starters: {[p.name + ' (' + p.position + ')' for p in away_squad['starters']]}")

    # Check if Zachery is in the result
    if hasattr(result, 'player_stats') and result.player_stats:
        for name, stats in result.player_stats.items():
            if 'Zachery' in name or 'Worth' in name:
                print(f"\n  Zachery Worth stats:")
                for k, v in stats.items():
                    print(f"    {k}: {v}")

if __name__ == "__main__":
    main()