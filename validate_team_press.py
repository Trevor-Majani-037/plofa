"""Team-press controller vs heuristic-baseline comparison (PERMANENT wiring).

Both arms run the SAME matches (same seeds, same neural on-ball XI + same
defensive-action brain) against the same opponent.  The ONLY difference:

  * controller arm: _TEAM_PRESS_AUTO=True -> first tick auto-loads
                    brains_team/XI.json and scales the press Bernoulli
                    prob = _PRESS_PROB[pos] * g  (g = brain(team-state))
  * baseline arm:   _TEAM_PRESS_AUTO=False -> g = 1.0 (pure role-rate press,
                    exactly the pre-controller behaviour)

Reported: score, goals for/against, possession, shots faced, and — for the
controller arm — the actual engagement scalar the brain chose each tick
(mean g, proving it is live and how coordinated its pressing really is).
"""
from __future__ import annotations

import argparse
import collections
import random
import statistics
from datetime import date

import match_engine as me_module
import match_probe
from squad_manager import SubstitutionController


def _arm(n_matches: int, seed: int, use_controller: bool,
         home_team="Probe FC", away_team="Rival FC",
         home_style="balanced", away_style="fluid_counter") -> dict:
    me_module.set_team_press_auto(use_controller)
    me_module._TEAM_PRESS_BRAIN = None
    me_module._TEAM_PRESS_LOADED = False

    outcomes = []
    g_vals: list = []
    caches = 0
    ev: collections.Counter = collections.Counter()

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
        match_probe._pin_heuristic()          # hold the ON-BALL layer constant
        # _pin_heuristic turns the press controller OFF by design (heuristic
        # baseline).  Re-assert this arm's choice so the controller arm runs
        # with the brain engaged through the PERMANENT auto-load path.
        me_module.set_team_press_auto(use_controller)
        me_module._TEAM_PRESS_BRAIN = None
        me_module._TEAM_PRESS_LOADED = False
        try:
            result = eng.simulate()
        finally:
            match_probe._restore_neural()

        g_home = eng.config.home_team == home_team
        gf = result.home_goals if g_home else result.away_goals
        ga = result.away_goals if g_home else result.home_goals

        cache = getattr(eng, "_team_press_g_cache", {}) or {}
        caches += len(cache)
        if use_controller:
            vals = [v for v in cache.values() if v != 1.0]
            g_vals.extend(vals if vals else list(cache.values()))

        hp_s = getattr(eng.state, "home_possession_s", 0.0)
        ap_s = getattr(eng.state, "away_possession_s", 0.0)
        poss = (hp_s / (hp_s + ap_s) * 100.0) if (hp_s + ap_s) else 50.0
        if not g_home:
            poss = 100.0 - poss
        outcomes.append({"gf": gf, "ga": ga, "poss": poss})

        for e in result.timeline:
            nm = getattr(e, "event_type", None)
            nm = nm.name if hasattr(nm, "name") else str(nm)
            ev[nm] += 1

    shots_against = ev.get("SHOT_ON_TARGET", 0) + ev.get("GOAL", 0) \
        + ev.get("SHOT_OFF_TARGET", 0) + ev.get("SHOT_BLOCKED", 0)

    return {
        "outcomes": outcomes,
        "gf": sum(o["gf"] for o in outcomes),
        "ga": sum(o["ga"] for o in outcomes),
        "poss": statistics.mean(o["poss"] for o in outcomes),
        "shots_against": shots_against,
        "g_cache_ticks": caches,
        "mean_g": statistics.mean(g_vals) if g_vals else 1.0,
        "good_g": int(sum(1 for v in g_vals if v >= 0.6)) if g_vals else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    print(f"controller-vs-baseline — {args.matches} matches each, seed "
          f"{args.seed}, neural on-ball + defensive brain held constant\n")
    c = _arm(args.matches, args.seed, use_controller=True)
    b = _arm(args.matches, args.seed, use_controller=False)

    def _brief(r, name):
        print(f"{name}:")
        print(f"  record = {[(o['gf'], o['ga']) for o in r['outcomes']]}")
        print(f"  GF={r['gf']}  GA={r['ga']}  possession={r['poss']:.1f}  "
              f"shots_against={r['shots_against']}")
        print(f"  press-g ticks={r['g_cache_ticks']}  "
              f"mean_g={r['mean_g']:.3f}  "
              f"g>=0.6 (press band)={r['good_g']}")

    _brief(b, "\nBASELINE  (g=1.0, role-rate press only)")
    _brief(c, "CONTROLLER (TeamPressBrain, permanent auto-load)")

    print("\ndelta (controller - baseline):")
    print(f"  GF {c['gf'] - b['gf']:+d}   GA {c['ga'] - b['ga']:+d}   "
          f"possession {c['poss'] - b['poss']:+.1f}   "
          f"shots_against {c['shots_against'] - b['shots_against']:+d}")
    wins_c = sum(1 for o in c["outcomes"] if o["gf"] > o["ga"])
    wins_b = sum(1 for o in b["outcomes"] if o["gf"] > o["ga"])
    draws_c = sum(1 for o in c["outcomes"] if o["gf"] == o["ga"])
    draws_b = sum(1 for o in b["outcomes"] if o["gf"] == o["ga"])
    print(f"  wins {wins_c}/{args.matches} vs {wins_b}/{args.matches}  "
          f"draws {draws_c} vs {draws_b}")


if __name__ == "__main__":
    main()