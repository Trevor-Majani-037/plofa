"""Collect more corpus, without the trainer's gate.

WHY THIS EXISTS
---------------
``brain_self_trainer`` runs COLLECT -> REFIT -> EVOLVE -> GATE -> PROMOTE as one
unit.  When the only goal is to grow the corpus, the GATE is pure cost: it plays
``--gate-matches`` more matches (2 by default, ~6 min each) comparing a
challenger that was never evolved against the incumbent.  And
``--gate-matches 0`` is not an escape hatch — ``_avg`` divides by ``len(lst)``,
so zero matches raises ZeroDivisionError.

So this calls the trainer's OWN ``collect`` and ``refit`` and nothing else.  No
logic is duplicated: the collection path, the exploration flags and the fitting
are the production ones.

THE SEED IS NOT OPTIONAL.  ``collect`` is deterministic in ``seed``, so running
it twice with the same seed replays the SAME matches and appends a duplicate of
the corpus.  That inflates the sample count without adding information — the
worst kind of progress.  Every call here must pass a fresh seed, and the script
prints the one it used so the next call can avoid it.

Usage:
    python tools/ops/collect_corpus.py --matches 4 --seed 2001
    python tools/ops/collect_corpus.py --matches 4 --seed 2001 --no-refit
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import brain_self_trainer as trainer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Grow the corpus (collect + refit).")
    p.add_argument("--matches", type=int, default=4)
    p.add_argument("--seed", type=int, required=True,
                   help="A FRESH seed each call — a repeat replays the same "
                        "matches and appends a duplicate.")
    p.add_argument("--styles", default=trainer.DEFAULT_STYLES)
    p.add_argument("--explore", type=float, default=0.15)
    p.add_argument("--explore-situational", action="store_true", default=True)
    p.add_argument("--work", default="brains_trainer")
    p.add_argument("--no-refit", action="store_true")
    args = p.parse_args()

    corpus = os.path.join(args.work, "corpus.raw.json")
    surrogate = os.path.join(args.work, "surrogate_refit.json")
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    print(f"seed {args.seed} | {args.matches} matches | styles {styles}")
    trainer.collect(corpus, args.matches, args.seed, styles, "neural", True,
                    explore=args.explore,
                    explore_situational=args.explore_situational)
    if not args.no_refit:
        trainer.refit(corpus, surrogate)
    print(f"\nNEXT: use a seed other than {args.seed}")


if __name__ == "__main__":
    main()
