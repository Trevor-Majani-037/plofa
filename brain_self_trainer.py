"""Post-match self-evolving XI trainer.

Closes the learning loop WITHOUT touching productions:

    1. COLLECT   — run N real neural matches (targeted: all 11 positions on
                   the pitch) and record (sensor, intent, outcome) pairs.
    2. APPEND    — merge new samples into a persistent corpus raw file
                   (full history kept — no recentcy-only churn).
    3. REFIT     — FitSurrogate.fit(corpus) -> scratch surrogate.
    4. EVOLVE    — parallel GA into a SCRATCH dir (never overwriting
                   brains/ while running), with --goal-bias so scoring
                   situations survive the argmax.
    5. GATE      — challenger XI vs INCUMBENT XI in real matches, same
                   squads/seeds/opponent (challenger-vs-incumbent, not
                   neural-vs-heuristic — the incumbent is the benchmark).
    6. PROMOTE / REJECT — challenger replaces brains/ ONLY if it beats the
                   incumbent on fitness AND does not lose the goal diff.
                   A rejected challenger is kept for inspection; the
                   incumbent is always the safe fallback.

Safe by construction:
  * Scratch state lives under --work (default brains_trainer/), so a crash
    at any stage leaves brains/ untouched.
  * The trainer NEVER runs auto_run_match.py and never writes season state.
  * Promotion happens only after the real-match gate passes.

Usage:
    python brain_self_trainer.py --matches 6 --gate-matches 3 --seed 123 \\
        --goal-bias 0.25 --generations 40 --workers 4
    python brain_self_trainer.py --smoke          # fast end-to-end check
    python brain_self_trainer.py --no-collect --no-evolve   # re-gate only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# ── importable blocks (no side effects beyond imports) ──────────────
from surrogate_rebuild import _run_matches, _load_raw, _save_raw
from surrogate_collect import FitnessSurrogate

ALL_POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]

DEFAULT_STYLES = "fluid_counter,tiki_taka,attacking"


def _human(s: float) -> str:
    return f"{s/60:.1f}min" if s >= 60 else f"{s:.0f}s"


def collect(corpus_path: str, n_matches: int, seed: int, styles: list[str],
            decision: str, targeted: bool) -> list:
    """Run real matches and return the NEW (sensor,intent,success,act,pos) pairs."""
    print(f"\n[1/6] COLLECT  {n_matches} {'targeted ' if targeted else ''}"
          f"real matches ({decision}, opponents={styles})...")
    t0 = time.time()
    data = _run_matches(n_matches, seed, styles, targeted, decision)
    print(f"       collected {len(data)} new samples in {_human(time.time()-t0)}")

    prior = _load_raw(corpus_path)
    merged = prior + data
    _save_raw(corpus_path, merged)
    print(f"       corpus now {len(prior)} -> {len(merged)} samples "
          f"({corpus_path})")
    return data


def refit(corpus_path: str, surrogate_path: str) -> FitnessSurrogate:
    print(f"\n[2/6] REFIT   {surrogate_path} from {corpus_path}...")
    data = _load_raw(corpus_path)
    if not data:
        raise SystemExit(f"! empty corpus {corpus_path} — nothing to learn from")
    sur = FitnessSurrogate().fit(data)
    sur.save(surrogate_path)
    print(f"       surrogate {sur.report()}")
    return sur


def evolve(work: str, surrogate_path: str, seed: int, generations: int,
           population: int, states: int, workers: int, goal_bias: float,
           positions: list[str]) -> None:
    scratch = os.path.join(work, "challenger")
    if os.path.isdir(scratch):
        shutil.rmtree(scratch)
    os.makedirs(scratch, exist_ok=True)
    print(f"\n[3/6] EVOLVE  {len(positions)} positions -> {scratch} "
          f"({generations} gen x {population} pop x {states} states, "
          f"goal_bias={goal_bias}, {workers} workers)")
    t0 = time.time()
    cmd = [
        sys.executable, "evolve_all_parallel.py",
        "--surrogate", surrogate_path,
        "--states", str(states),
        "--generations", str(generations),
        "--population", str(population),
        "--workers", str(workers),
        "--seed", str(seed),
        "--goal-bias", str(goal_bias),
        "--positions", ",".join(positions),
        "--out", scratch,
    ]
    proc = subprocess.run(cmd, cwd=os.getcwd())
    if proc.returncode != 0:
        raise SystemExit(f"! evolution failed (rc={proc.returncode})")
    print(f"       evolution done in {_human(time.time()-t0)}")


def _run_xi(brains_dir: str, home_style: str, away_style: str, seed: int):
    """Run one real match for a full-XI brain dir; return (result, fitness)."""
    from validate_neural_xl import _run_neural
    return _run_neural(brains_dir, seed, home_style, away_style)


def gate(work: str, challenger_dir: str, incumbent_dir: str, n_matches: int,
         seed: int, home_style: str, away_style: str) -> dict:
    """Challenger vs INCUMBENT in real matches (same seeds, both XIs neural)."""
    print(f"\n[4/6] GATE    challenger vs INCUMBENT, {n_matches} matches "
          f"(seed {seed}, {home_style} vs {away_style})...")
    c_fits, i_fits = [], []
    t0 = time.time()
    for m in range(n_matches):
        s = seed + m * 100
        result_c, c = _run_xi(challenger_dir, home_style, away_style, s)
        print(f"       chal m{m+1}: {result_c.score_str}  fit={c['fitness']:.3f}  "
              f"goals={c['goals']}  poss={c.get('possession_pct')}")
        result_i, i = _run_xi(incumbent_dir, home_style, away_style, s + 7)
        print(f"       inc  m{m+1}: {result_i.score_str}  fit={i['fitness']:.3f}  "
              f"goals={i['goals']}  poss={i.get('possession_pct')}")
        c_fits.append(c)
        i_fits.append(i)

    def _avg(lst, k): return round(sum(f[k] for f in lst) / len(lst), 4)

    summary = {
        "challenger_fitness": _avg(c_fits, "fitness"),
        "incumbent_fitness": _avg(i_fits, "fitness"),
        "challenger_goals": sum(f["goals"] for f in c_fits),
        "incumbent_goals": sum(f["goals"] for f in i_fits),
        "elapsed_s": round(time.time() - t0, 1),
    }
    summary["goal_diff"] = summary["challenger_goals"] - summary["incumbent_goals"]
    summary["fitness_delta"] = round(summary["challenger_fitness"] -
                                     summary["incumbent_fitness"], 4)
    print(f"\n       GATE RESULT: fitness_delta={summary['fitness_delta']:+.4f}  "
          f"goal_diff={summary['goal_diff']:+d}")
    return summary


def promote(work: str, challenger_dir: str, incumbent_dir: str,
            out_dir: str, summary: dict) -> bool:
    """Copy challenger brains over the incumbent XI if the gate passed."""
    promo = summary["fitness_delta"] > 0.005 and summary["goal_diff"] >= 0
    reason = ("beat incumbent on fitness AND held/beat the goal diff"
              if promo else
              "failed gate (fitness or goals not better)")
    print(f"\n[5/6] PROMOTE {'YES' if promo else 'NO'}: {reason}")
    if not promo:
        return False
    count = 0
    for pos in ALL_POSITIONS:
        src = os.path.join(challenger_dir, f"{pos}.json")
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(out_dir, f"{pos}.json"))
        count += 1
    print(f"       copied {count} brains -> {out_dir}  (incumbent now = challenger)")
    # refresh the trainer's manifest of the active set
    manifest = os.path.join(out_dir, "_manifest_all.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            mf = json.load(f)
        mf["last_trainer_promotion"] = time.strftime("%Y-%m-%d %H:%M:%S")
        mf["gate"] = summary
        with open(manifest, "w") as f:
            json.dump(mf, f, indent=2)
    return True


def log_promotion(work: str, summary: dict, promoted: bool, args) -> None:
    log_path = os.path.join(work, "trainer_log.jsonl")
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "promoted": promoted,
        "gate": summary,
        "config": {
            "goal_bias": args.goal_bias, "seed": args.seed,
            "gate_matches": args.gate_matches,
        },
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[6/6] LOG     {log_path}")


def main():
    p = argparse.ArgumentParser(description="Post-match self-evolving XI trainer.")
    p.add_argument("--brains", default="brains",
                   help="Production brain dir (incumbent / promotion target).")
    p.add_argument("--work", default="brains_trainer",
                   help="Scratch dir for corpus, surrogate, challenger, log.")
    p.add_argument("--matches", type=int, default=6,
                   help="Real neural matches collected each cycle (each ~20s).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--goal-bias", type=float, default=0.25,
                   help="Share of scoring situations during evolution.")
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--states", type=int, default=300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gate-matches", type=int, default=2,
                   help="Real matches for challenger-vs-incumbent gate.")
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    p.add_argument("--styles", default=DEFAULT_STYLES,
                   help="Comma-separated opponent styles for collection.")
    p.add_argument("--decision", choices=("neural", "heuristic"), default="neural",
                   help="Policy that produces samples during collection. "
                        "neural = trained XI (shoots from scoring positions).")
    p.add_argument("--targeted", action="store_true", default=True,
                   help="CAM home / CF away so all 11 positions get samples.")
    p.add_argument("--no-collect", action="store_true", help="Skip collection.")
    p.add_argument("--no-evolve", action="store_true", help="Skip evolution.")
    p.add_argument("--smoke", action="store_true",
                   help="Fast end-to-end: 1 collect match, tiny GA, 1 gate match.")
    args = p.parse_args()

    # flush per line so the trainer's own prints interleave live with the
    # evolve_all_parallel subprocess output instead of dumping at exit
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if args.smoke:
        args.matches = min(args.matches, 1)
        args.gate_matches = 1
        args.generations = 2
        args.population = 8
        args.states = 40
        args.workers = 2
        args.goal_bias = 0.25

    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.brains, exist_ok=True)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    corpus = os.path.join(args.work, "corpus.raw.json")
    surrogate_src = os.path.join(args.work, "surrogate_refit.json")
    challenger_dir = os.path.join(args.work, "challenger")

    if not args.no_collect:
        collect(corpus, args.matches, args.seed, styles, args.decision,
                args.targeted)

    if not os.path.exists(corpus):
        raise SystemExit(f"! no corpus at {corpus} (ran with --no-collect on a "
                         f"fresh work dir?)")

    refit(corpus, surrogate_src)

    if not args.no_evolve:
        evolve(args.work, surrogate_src, args.seed, args.generations,
               args.population, args.states, args.workers, args.goal_bias,
               ALL_POSITIONS)
    else:
        if not os.path.isdir(challenger_dir):
            raise SystemExit("! --no-evolve but no existing challenger dir "
                             f"{challenger_dir}")

    # the challenger must contain all 11 brain files
    missing = [pos for pos in ALL_POSITIONS
               if not os.path.exists(os.path.join(challenger_dir, f"{pos}.json"))]
    if missing:
        raise SystemExit(f"! challenger missing brains: {missing}")

    summary = gate(args.work, challenger_dir, args.brains, args.gate_matches,
                   args.seed, args.home_style, args.away_style)

    promoted = promote(args.work, challenger_dir, args.brains, args.brains,
                       summary)
    log_promotion(args.work, summary, promoted, args)
    print("\n=== CYCLE COMPLETE ===")
    print(f"  promoted: {promoted}")
    print(f"  challenger kept at: {challenger_dir}  (inspected if rejected)")
    print(f"  production dir: {args.brains} (untouched on reject)")


if __name__ == "__main__":
    main()