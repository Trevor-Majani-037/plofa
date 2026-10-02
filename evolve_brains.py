"""Evolve per-position football brains.

CLI usage:
    python evolve_brains.py                       # all 11 positions, defaults
    python evolve_brains.py --position ST         # just striker
    python evolve_brains.py --positions ST,CAM,CB # specific set
    python evolve_brains.py --generations 60 --population 48 --fast

Output: writes per-position best brains to:
    brains/<POSITION>.json

Each JSON holds the serialized FootballBrain weights (a single float
array per layer) ready to be loaded via FootballBrain.load() and
registered into the match via brain_integration.register_brain().

Quick sanity check after evolving:
    python -c "from football_brain import FootballBrain;
    b = FootballBrain.load('brains/ST.json'); print(b)"
"""

from __future__ import annotations

import argparse
import os
import json
import time

from brain_evolution import evolve
from football_brain import INPUT_SIZE


ALL_POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evolve per-position FootballBrain nets via a genetic algorithm."
    )
    p.add_argument("--position", type=str, default=None,
                   help="Single position to evolve (default: all).")
    p.add_argument("--positions", type=str, default=None,
                   help="Comma-separated positions to evolve (e.g. ST,CAM,CB).")
    p.add_argument("--generations", type=int, default=40,
                   help="Number of evolution generations per position.")
    p.add_argument("--population", type=int, default=32,
                   help="Population size per position (higher = more diverse).")
    p.add_argument("--states", type=int, default=400,
                   help="Number of random match states to score each brain against.")
    p.add_argument("--seed", type=int, default=0,
                   help="Global RNG seed for reproducibility.")
    p.add_argument("--out", type=str, default="brains",
                   help="Output directory for per-position brain JSON files.")
    p.add_argument("--fast", action="store_true",
                   help="Use reduced defaults for a quick smoke test.")
    p.add_argument("--validate", action="store_true",
                   help="Skip evolution; run the evolved brains in real matches "
                        "via match_probe and print validation fitness. "
                        "Also enables --ab A/B comparison.")
    p.add_argument("--ab", action="store_true",
                   help="A/B compare evolved vs heuristic DecisionBrain in real matches "
                        "(requires a --position and an existing brain file).")
    p.add_argument("--matches", type=int, default=1,
                   help="Number of real matches per position when validating. "
                        "CAUTION: each real match takes ~15-20s.")
    p.add_argument("--outcome", action="store_true",
                   help="Score evolved brains by the engine's own outcomes "
                        "(xG diff, turnover rate, chance creation) instead of "
                        "the intention-quality baseline.  Implies --validate "
                        "for the requested positions.")
    p.add_argument("--surrogate", type=str, default=None,
                   help="Path to a trained FitnessSurrogate JSON. When provided, "
                        "evolution scores brains against the learned expected-success "
                        "table instead of the hand-made position reward weights.")
    p.add_argument("--goal-bias", type=float, default=0.0,
                   help="Fraction of sampled states drawn from SCORING situations "
                        "(final third, goal-close, central). 0.0 = pure random states "
                        "like the v3 retrain; >0.25 keeps rare scoring intents "
                        "(SHOOT/cross/through-ball) in the argmax.")
    p.add_argument("--role-features", action="store_true",
                   help="Evolve a schema-v2 role-features brain (24 + role block "
                        "inputs instead of 24). Gives each family its structured "
                        "position menu (GK/CB/FB/DM/CM 31-d, AM/WING/ST 32-d).")
    p.add_argument("--tactics-context", action="store_true",
                   help="Evolve a schema-v3 tactics-context brain (24 + role "
                        "block + 12 manager-instruction dials). Implies "
                        "--role-features: the context block ONLY makes sense "
                        "stacked on top of the role tail (ST 44-d, DM 43-d).")
    p.add_argument("--perception", action="store_true",
                   help="Train THROUGH imperfect perception: synthetic states are "
                        "generated via the perception layer (range/FOV/noise), so "
                        "the challenger learns on the world it will actually see. "
                        "Requires --role-features to be meaningful (perception "
                        "degrades the shared block only).")
    p.add_argument("--perception-roles", action="store_true",
                   help="With --perception, also enable per-role perception "
                        "profiles (relevance top-k etc.) during training.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.fast:
        args.generations = min(args.generations, 10)
        args.population = min(args.population, 16)
        args.states = min(args.states, 100)

    if args.position:
        positions = [args.position.strip().upper()]
    elif args.positions:
        positions = [s.strip().upper() for s in args.positions.split(",") if s.strip()]
    else:
        positions = ALL_POSITIONS

    # ── Validation / A-B mode ─────────────────────────────────
    # ── Outcome-driven scoring (engine reality) ─────────────────
    if args.outcome:
        try:
            from match_probe import run_outcome_validation, _slot_for_position
        except ImportError as e:
            print(f"  ! cannot import match_probe: {e}")
            return
        targets = None
        if args.position:
            pos = args.position.strip().upper()
            targets = [(_slot_for_position(pos), pos)]
        n_matches = max(3, args.matches)
        print(f"\n=== Outcome-scoring evolved brains (engine reality, {n_matches} matches each) ===")
        print("Signals: xG diff, turnover rate, chance creation, possession, goal diff.")
        res = run_outcome_validation(
            targets=targets, brains_dir=args.out,
            n_matches=n_matches, seed=args.seed,
        )
        print(json.dumps(res, indent=2))
        return

    if args.validate or args.ab:
        try:
            from match_probe import run_neural_validation, ab_compare
        except ImportError as e:
            print(f"  ! cannot import match_probe: {e}")
            return

        if args.ab:
            if not args.position:
                print("  ! --ab requires --position <POS> with an existing brain file.")
                return
            pos = args.position.strip().upper()
            print(f"\n=== A/B: {pos} neural vs heuristic ({args.matches} matches each) ===")
            res = ab_compare(pos, brains_dir=args.out, n_matches=args.matches, seed=args.seed)
            print(json.dumps(res, indent=2))
            return

        print(f"\n=== Validating evolved brains in real matches ===")
        print(f"(Each match takes ~15-20s; {args.matches} match(es) per position)")
        targets = None
        if args.position:
            pos = args.position.strip().upper()
            targets = [(f"{pos}{pos}", pos)]
        res = run_neural_validation(
            targets=targets, brains_dir=args.out,
            n_matches=args.matches, seed=args.seed,
        )
        print(json.dumps(res, indent=2))
        return

    # ── Evolution mode ────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)

    # optional surrogate drives fitness instead of the reward table
    surrogate = None
    if args.surrogate:
        from critic_surrogate import load_any_surrogate
        surrogate = load_any_surrogate(args.surrogate)
        nbins = len(getattr(surrogate, "table", {})) if surrogate is not None else 0
        print(f"Using fitness surrogate: {args.surrogate}"
              f" ({nbins or 'value_critic'} situation bins)")

    # role-features width per family; None when --role-features is OFF
    role_width = None
    if args.role_features:
        from role_features import V2_INPUT_D
        role_width = V2_INPUT_D
        print(f"Role features ON: {role_width}")

    # schema-v3 tactics-context width per family (stacked on role tail)
    tactics_width = None
    if args.tactics_context:
        from tactics_context import V3_INPUT_D
        tactics_width = V3_INPUT_D
        print(f"Tactics-context ON: {V3_INPUT_D}")

    # perception config (train-through-imperfect-perception)
    perception_cfg = None
    if args.perception:
        from perception import PerceptionConfig
        perception_cfg = PerceptionConfig(
            enabled=True, role_blocks=args.perception_roles, seed=args.seed)
        print(f"Perception mode ON (role_blocks={args.perception_roles}, "
              f"seed={args.seed})")

    # maintain a running index so differing positions with the same seed
    # still diverge
    results = {}

    for pos in positions:
        if pos not in ALL_POSITIONS:
            print(f"  ! unknown position '{pos}', skipping. Valid: {ALL_POSITIONS}")
            continue

        # resolve per-position role width
        if tactics_width is not None:
            from role_features import _role_family
            fam = _role_family(pos)
            input_size = tactics_width.get(fam, INPUT_SIZE) if fam else INPUT_SIZE
            schema_label = "v3_tactics_context" if input_size != INPUT_SIZE else "v1_24d"
        elif role_width is not None:
            from role_features import _role_family
            fam = _role_family(pos)
            input_size = role_width.get(fam, INPUT_SIZE) if fam else INPUT_SIZE
            schema_label = "v2_role_features" if input_size != INPUT_SIZE else "v1_24d"
        else:
            input_size = INPUT_SIZE
            schema_label = "v1_24d"

        print(f"\n=== Evolving {pos} brain {'(role D=' + str(input_size-INPUT_SIZE) + ')' if input_size != INPUT_SIZE else ''} ===")
        start = time.time()
        res = evolve(
            position=pos,
            population_size=args.population,
            generations=args.generations,
            n_states=args.states,
            seed=args.seed,
            verbose=True,
            surrogate=surrogate,
            goal_bias=args.goal_bias,
            input_size=input_size,
            perception_config=perception_cfg,
        )
        elapsed = time.time() - start

        # save this position's best brain
        path = os.path.join(args.out, f"{pos}.json")
        res.best_brain.save(path)
        results[pos] = {
            "best_fitness": round(res.best_fitness, 4),
            "generations": args.generations,
            "seed": args.seed,
            "elapsed_s": round(elapsed, 1),
            "saved_to": path,
            "input_size": input_size,
            "sensor_schema": schema_label,
        }
        print(f"  {pos}: fitness={res.best_fitness:.4f} ({elapsed:.1f}s) -> {path}")

    # write a summary manifest
    manifest = os.path.join(args.out, "_manifest.json")
    with open(manifest, "w") as f:
        json.dump({
            "positions": positions,
            "generations": args.generations,
            "population": args.population,
            "states": args.states,
            "role_features": args.role_features,
            "tactics_context": args.tactics_context,
            "perception": args.perception,
            "perception_roles": args.perception_roles,
            "results": results,
        }, f, indent=2)
    print(f"\nSummary written to {manifest}")
    print("Done. Load brains into a match with:")
    print("  from brain_integration import load_brains_from_file, register_brain")


if __name__ == "__main__":
    main()
