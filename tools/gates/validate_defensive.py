"""Live validation of the defensive-action brain vs the heuristic table.

Both arms run the SAME matches (same seeds, same neural on-ball XI) against
the same opponent.  The ONLY difference is the defensive-action selector:

  * neural arm:  DefensiveActionBrain (brains_def/ACTION.json, auto-loads)
  * heuristic arm: legacy _danger_scaled_action_weights table (the exact
                   behaviour the baseline had before the controller existed)

Reported: score, goals conceded, shots faced, defensive event counts, and
% of clearances/blocks/tackles/interceptions used (proves the brain is
actually driving decisions, not silently falling back).
"""
from __future__ import annotations

import argparse
import collections
import os
import random
from datetime import date

import match_engine as me_module
import match_probe
from squad_manager import SubstitutionController


def _event_counts(timeline) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for ev in timeline:
        name = getattr(ev, "event_type", None)
        name = name.name if hasattr(name, "name") else str(name)
        c[name] += 1
    return c


def _shots_against(result, team_home: str) -> int:
    n = 0
    for ev in result.timeline:
        nm = getattr(ev, "event_type", None)
        nm = nm.name if hasattr(nm, "name") else str(nm)
        if nm in ("SHOT_ON_TARGET", "GOAL", "SHOT_OFF_TARGET", "SHOT_BLOCKED"):
            if getattr(ev, "team", team_home) != team_home:
                n += 1
    return n


def _run(n_matches: int, seed: int, use_brain: bool,
         home_team="Probe FC", away_team="Rival FC",
         home_style="balanced", away_style="fluid_counter") -> dict:
    me_module.set_defensive_action_auto(True if use_brain else False)
    me_module._DEF_ACTION_BRAIN = None
    me_module._DEF_ACTION_LOADED = False

    total_gf = total_ga = total_poss = 0
    outcomes = []
    ev_counters: collections.Counter = collections.Counter()
    shots_against = 0

    for m in range(n_matches):
        seed_i = seed + m * 1000
        random.seed(seed_i)
        home_squad, away_squad = match_probe._build_squads(home_team, away_team)
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
        match_probe._pin_heuristic()  # hold the ON-BALL layer constant
        try:
            result = eng.simulate()
        finally:
            match_probe._restore_neural()

        g = result.home_goals if home_team == eng.config.home_team \
            else result.away_goals
        ga = result.away_goals if home_team == eng.config.home_team \
            else result.home_goals
        total_gf += g
        total_ga += ga
        hp_s = getattr(eng.state, "home_possession_s", 0.0)
        ap_s = getattr(eng.state, "away_possession_s", 0.0)
        poss_pct = (hp_s / (hp_s + ap_s) * 100.0) if (hp_s + ap_s) else 50.0
        if home_team != eng.config.home_team:
            poss_pct = 100.0 - poss_pct
        total_poss += poss_pct
        outcomes.append((g, ga))
        ev_counters += _event_counts(result.timeline)
        shots_against += _shots_against(result, home_team)

    return {
        "outcomes": outcomes,
        "total_gf": total_gf,
        "total_ga": total_ga,
        "possession": total_poss / n_matches,
        "shots_against": shots_against,
        "events": ev_counters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    tag = "neural-defbrain" if os.path.exists("brains_def/ACTION.json") else "no-brain"
    print(f"neural arm ({tag}) vs heuristic arm — {args.matches} matches "
          f"each, seed {args.seed}, neural on-ball held constant")
    n = _run(args.matches, args.seed, use_brain=True)
    h = _run(args.matches, args.seed, use_brain=False)

    def _brief(r, name):
        print(f"\n{name}:")
        print(f"  record {r['outcomes']}  GF={r['total_gf']}  GA={r['total_ga']}  "
              f"possession={r['possession']:.1f}  shots_against={r['shots_against']}")
        ev = r["events"]
        print(f"  defensive: tackles={ev.get('TACKLE_WON',0)+ev.get('TACKLE_LOST',0)} "
              f"interceptions={ev.get('INTERCEPTION',0)} "
              f"clearances={ev.get('CLEARANCE',0)} blocks={ev.get('SHOT_BLOCKED',0)+ev.get('BLOCK',0)}")

    _brief(n, "NEURAL defensive brain")
    _brief(h, "HEURISTIC defensive table")

    print("\ndelta (neural - heuristic):")
    print(f"  GA: {n['total_ga'] - h['total_ga']}   shots_against: "
          f"{n['shots_against'] - h['shots_against']}   possession: "
          f"{n['possession'] - h['possession']:+.1f}")
    n_wins = sum(1 for i, o in enumerate(n["outcomes"]) if o[0] > o[1])
    w_h = sum(1 for i, o in enumerate(h["outcomes"]) if o[0] > o[1])
    print(f"  wins: neural {n_wins}/{args.matches}  heuristic {w_h}/{args.matches}")


if __name__ == "__main__":
    main()