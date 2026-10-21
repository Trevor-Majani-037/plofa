"""Refit the surrogate from the saved raw corpus.

A fitted table cannot be refitted (the shrinkage hides the raw sums), so
before the raw corpus existed, changing the fit meant re-running every
match.  Now it is free.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_refit.py \\
        brains/surrogate_pos_v4.raw.json brains/surrogate_pos_v4.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from surrogate_collect import FitnessSurrogate, load_raw_corpus  # noqa: E402


def main():
    raw_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else raw_path.replace(
        ".raw.json", ".json")

    pairs = load_raw_corpus(raw_path)
    print(f"loaded {len(pairs)} pairs from {raw_path}")
    sur = FitnessSurrogate().fit(pairs)
    sur.save(out_path)
    print(f"refitted and saved to {out_path}")
    print(f"  positions:    {sorted(p for p in sur.table if p)}")
    print(f"  families:     {sorted(sur.family_table)}")
    for fam in sorted(sur.family_table):
        print(f"    {fam:<6} intents performed: "
              f"{sorted(sur.family_intents.get(fam, ()))}")


if __name__ == "__main__":
    sys.exit(main())
