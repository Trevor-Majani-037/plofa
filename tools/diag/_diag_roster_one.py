"""ONE real-roster match with a fixed seed, in its OWN process.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_roster_one.py <brains_dir> <seed>

Prints one JSON line with the match's metrics, read from the export (so
nothing is re-implemented).

⚠️  TWO TRAPS THIS FILE EXISTS TO AVOID, both found the hard way:

1. **ONE PROCESS PER MATCH.**  The brain registry, the per-position brain
   cache and the mind registry are MODULE state.  Running the two arms in
   one process gives the second arm the first arm's brains: the first
   version of the roster gate produced byte-identical matches for the two
   arms (124 shots each, 11 goals each) and reported "the brains make no
   difference".  They make a very large difference — the same seed in two
   separate processes gives xG 2.55-1.77 against 0.77-1.06.

2. **THE CHALLENGER BINDS TO THE HOME XI ONLY.**  The brain DIR is global,
   so `set_brain_dir(challenger)` gives the AWAY side the challenger too
   and the match becomes "challenger against challenger" — a match with no
   side bias, which is not the shipping question.  The auto-load stays on
   the shipped set (the control) and the challenger is registered BY NAME
   for the home XI, which takes precedence over the position auto-load.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "diag"))

OUT_DIR = os.path.join(ROOT, "plofa_output", "_roster_gate")


def main() -> int:
    brains_dir = sys.argv[1]
    seed = int(sys.argv[2])
    # THE REAL ROSTER'S clubs.  "Hartwell City" and "Thornfield United" are
    # the SCRATCH teams (hand-entered in run_match.py, with the souls and
    # Percy) and are NOT in the workbook — the real roster is these eighteen.
    home = sys.argv[3] if len(sys.argv) > 3 else "Natrican"
    away = sys.argv[4] if len(sys.argv) > 4 else "Tryox City"
    # The KIND of brain the challenger directory holds: "mlp" (the shipped
    # FootballBrain) or "situational" (the new attention + recurrence +
    # memory + plan brain).  The router in brain_integration chooses by
    # the brain's TYPE, so this is all that differs between the arms.
    brain_kind = sys.argv[5] if len(sys.argv) > 5 else "mlp"

    import numpy as np
    from _diag_v7_export_real import run_real_match

    random.seed(seed)
    np.random.seed(seed)
    out = os.path.join(OUT_DIR, f"seed{seed}_{os.path.basename(brains_dir)}")
    result, bound, _out = run_real_match(brains_dir, home, away, seed, out,
                                         brain_kind=brain_kind)

    import glob
    jsons = glob.glob(os.path.join(out, "*.json"))
    data = {}
    for j in jsons:
        with open(j, encoding="utf-8") as f:
            data = json.load(f)
        if "match" in data:
            break
    match = data.get("match", {})
    events = data.get("timeline", [])
    shots = [e for e in events
             if str(e.get("type", "")).startswith("SHOT")
             or str(e.get("type", "")) in ("HIT_WOODWORK", "FREEKICK_DIRECT")]
    goals = [e for e in events
             if str(e.get("type", "")) in ("GOAL", "PENALTY_SCORED")]
    offsides = [e for e in events if str(e.get("type", "")) == "OFFSIDE"]
    parts = [p for p in str(match.get("score", "")).replace("-", "–").split("–")
             if p.strip().isdigit()]
    score = [int(parts[0]), int(parts[1])] if len(parts) == 2 else [0, 0]
    print(json.dumps({
        "brains": brains_dir, "seed": seed, "bound": bound, "score": score,
        "home_goals": sum(1 for g in goals if g.get("team") == home),
        "away_goals": sum(1 for g in goals if g.get("team") != home),
        "shots": len(shots), "offsides": len(offsides),
        "poss": match.get("home_possession_pct", 0.0),
        "xg": match.get("home_xg", 0.0),
        "away_xg": match.get("away_xg", 0.0),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
