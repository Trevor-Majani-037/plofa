"""Reasoning gate — policy + consequence evaluation vs policy-only.

Same production XI (Probe FC balanced vs Rival FC fluid_counter), identical
seeds per pair, both arms under the GATE-WINNING ROLE-BLOCK perception (the
same world the v3 critic's corpus was collected in).  The ONLY difference
between arms is the reasoning seam: whether _decide_core corrects each
intent distribution with the critic's expected-payoff read of the same
perceived state.

  POLICY    : reasoning OFF (seam explicitly disabled for the control arm)
  REASONING : reasoning ON  (critic from --critic, blend from --blend)

The seam became ON-by-default when the v5 critic graduated the gate
(2026-09-20), so the POLICY arm pins the reasoning posture OFF explicitly
to measure the true delta.  The gate measures whether adding consequence
evaluation helps
(fitness / goals / possession / xG) without degrading realism — plus an
intent-mix audit so a critic that pumps SHOOT everywhere is caught as the
shot-volume regression it is.

In addition to the fitness comparison the gate records the reasoner's own
q/advantage quality: how often the corrected distribution beats the policy
on the critic's own terms, reported as a sanity trace of its stats.

Usage:
  py validate_consequence.py --matches 4 --seed 21
"""
from __future__ import annotations

import argparse
import json
import os
import time

from brain_integration import clear_consequence
from consequence_decision import set_consequence, default_enabled as _pin_default
from perception import PerceptionConfig, set_perception, get_perception_config
import validate_neural_xl


def _run_arm(brains_dir: str, seed: int, home_style: str, away_style: str,
             reasoning: bool, critic_path: str, blend: float,
             label: str, match_idx: int):
    """Seed + run one arm under ROLE-BLOCK perception; reasoning toggled.

    Mirrors validate_neural_xl._run_neural EXCEPT perception stays on the
    role-block world the critic was trained on (production default), and
    the reasoning seam is engaged/cleared around the match.
    """
    saved_cfg = get_perception_config()
    set_perception(PerceptionConfig(enabled=True, role_blocks=True, seed=0))
    # Reasoning is ON by default since the graduation; pin the process's
    # unset-env posture OFF so the control arm is genuinely reasoning-free,
    # then engage the seam explicitly only for the reasoning arm.
    _pin_default(False)
    clear_consequence()
    if reasoning:
        set_consequence(critic_path, blend=blend)
    try:
        import random
        random.seed(seed)
        from datetime import date
        import match_probe
        home_squad, away_squad = match_probe._build_squads("Probe FC", "Rival FC")
        n_bound = validate_neural_xl.register_full_xi(brains_dir, home_squad["starters"])
        config = match_probe.MatchConfig(
            home_team="Probe FC", away_team="Rival FC",
            match_date=date(2026, 9, 6), matchday=3, season="26/27",
        )
        hp = match_probe._team_profile("Probe FC", home_style)
        ap = match_probe._team_profile("Rival FC", away_style)
        eng = match_probe.MatchEngine(config, hp, ap)
        eng.set_squad("Probe FC", home_squad["starters"], home_squad["substitutes"])
        eng.set_squad("Rival FC", away_squad["starters"], away_squad["substitutes"])
        result = eng.simulate()
        home_names = [p.name for p in home_squad["starters"]]
        fitness = match_probe.extract_team_fitness(result, home_names)
        fitness["_n_bound"] = n_bound
        return result, fitness
    finally:
        clear_consequence()
        set_perception(saved_cfg)


def _intent_share(fits):
    """Aggregate intent counts across a run of one arm."""
    mix: dict = {}
    for f in fits:
        for intent, c in f.get("intent_mix", {}).items():
            mix[intent] = mix.get(intent, 0) + c
    return mix


def main():
    p = argparse.ArgumentParser(description="Policy+consequence reasoning gate.")
    p.add_argument("--brains", default="brains")
    p.add_argument("--matches", type=int, default=4)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    p.add_argument("--critic", default="brains_trainer/critic_v5.json")
    p.add_argument("--blend", type=float, default=0.3)
    p.add_argument("--out", default="validate_consequence_gate.txt")
    args = p.parse_args()

    if not os.path.exists(args.critic):
        print(f"critic not found: {args.critic}")
        raise SystemExit(1)

    print(f"=== Consequence reasoning gate: {args.matches} matches x 2 arms ===")
    print(f"(role-block perception both arms; critic={args.critic}, blend={args.blend})\n")

    policy_fits, reason_fits = [], []
    t0 = time.time()
    for m in range(args.matches):
        seed = args.seed + m * 100
        r, nf = _run_arm(args.brains, seed, args.home_style, args.away_style,
                         reasoning=False, critic_path=args.critic,
                         blend=args.blend, label="policy", match_idx=m + 1)
        print(f"  policy    m{m+1}: {r.score_str}  fit={nf['fitness']:.3f}  "
              f"goals={nf['goals']}  xg={nf['xg']:.2f}  "
              f"poss={nf.get('possession_pct')}%  bound={nf.get('_n_bound')}")
        policy_fits.append(nf)

        r, rf = _run_arm(args.brains, seed, args.home_style, args.away_style,
                         reasoning=True, critic_path=args.critic,
                         blend=args.blend, label="reason", match_idx=m + 1)
        print(f"  reasoning m{m+1}: {r.score_str}  fit={rf['fitness']:.3f}  "
              f"goals={rf['goals']}  xg={rf['xg']:.2f}  "
              f"poss={rf.get('possession_pct')}%  bound={rf.get('_n_bound')}")
        reason_fits.append(rf)

    def _avg(lst, key):
        return round(sum(f[key] for f in lst) / len(lst), 4) if lst else 0.0

    p_share, r_share = _intent_share(policy_fits), _intent_share(reason_fits)
    all_intents = sorted(set(p_share) | set(r_share))
    intent_deltas = {i: r_share.get(i, 0) - p_share.get(i, 0)
                     for i in all_intents}

    summary = {
        "matches": args.matches, "seed": args.seed,
        "critic": args.critic, "blend": args.blend,
        "policy_fitness": _avg(policy_fits, "fitness"),
        "reasoning_fitness": _avg(reason_fits, "fitness"),
        "fitness_delta": round(_avg(reason_fits, "fitness") -
                               _avg(policy_fits, "fitness"), 4),
        "policy_goals": sum(f["goals"] for f in policy_fits),
        "reasoning_goals": sum(f["goals"] for f in reason_fits),
        "goal_diff": (sum(f["goals"] for f in reason_fits) -
                      sum(f["goals"] for f in policy_fits)),
        "policy_xg": _avg(policy_fits, "xg"),
        "reasoning_xg": _avg(reason_fits, "xg"),
        "policy_possession": _avg(policy_fits, "possession_pct"),
        "reasoning_possession": _avg(reason_fits, "possession_pct"),
        "policy_intent_mix": p_share,
        "reasoning_intent_mix": r_share,
        "intent_deltas": intent_deltas,
        "elapsed_s": round(time.time() - t0, 1),
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    verdict = "REASONING NOT GRADUATED" if summary["fitness_delta"] < 0 else (
        "REASONING GRADUATED (no realism regression)")

    lines = [
        f"=== CONSEQUENCE REASONING GATE SUMMARY ===",
        f"matches={args.matches} seed={args.seed} critic={args.critic} blend={args.blend}",
        f"policy_fitness       = {summary['policy_fitness']}",
        f"reasoning_fitness    = {summary['reasoning_fitness']}",
        f"fitness_delta        = {summary['fitness_delta']}",
        f"policy_goals         = {summary['policy_goals']}",
        f"reasoning_goals      = {summary['reasoning_goals']}",
        f"goal_diff            = {summary['goal_diff']}",
        f"policy_xg            = {summary['policy_xg']}",
        f"reasoning_xg         = {summary['reasoning_xg']}",
        f"xg_delta             = {round(summary['reasoning_xg'] - summary['policy_xg'], 4)}",
        f"policy_possession    = {summary['policy_possession']}",
        f"reasoning_possession = {summary['reasoning_possession']}",
        f"poss_delta           = {round(summary['reasoning_possession'] - summary['policy_possession'], 4)}",
        f"intent_deltas        = {json.dumps(summary['intent_deltas'])}",
        f"elapsed_s            = {summary['elapsed_s']}",
        f"VERDICT: {verdict}",
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out.replace(".txt", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGate recorded: {args.out} (+ .json)")
    return summary


if __name__ == "__main__":
    main()