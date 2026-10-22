"""The real-roster gate, per seed and per arm, from the exports.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_roster_report.py \\
        plofa_output/_roster_gate

⚠️  ONLY the per-arm folders ("seedNN_<brains>") are read.  The older
"seedNN" folders (before the per-arm fix) were overwritten by whichever arm
ran last, so they MIX the two arms and must never be read.
"""
import glob
import json
import os
import sys


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "plofa_output/_roster_gate"
    folders = sorted(p for p in glob.glob(os.path.join(root, "seed*_*"))
                     if os.path.isdir(p))
    for folder in folders:
        data = None
        for j in glob.glob(os.path.join(folder, "*.json")):
            with open(j, encoding="utf-8") as f:
                d = json.load(f)
            if "match" in d:
                data = d
                break
        if not data:
            print(f"{os.path.basename(folder):<22} (no export)")
            continue
        m = data["match"]
        ev = data.get("timeline", [])
        shots = sum(1 for e in ev if str(e.get("type", "")).startswith("SHOT")
                    or str(e.get("type", "")) in ("HIT_WOODWORK", "FREEKICK_DIRECT"))
        off = sum(1 for e in ev if str(e.get("type", "")) == "OFFSIDE")
        print(f"{os.path.basename(folder):<22} {m.get('score', '?'):<6} "
              f"xg {m.get('home_xg', 0):.2f}-{m.get('away_xg', 0):.2f}  "
              f"poss {m.get('home_possession_pct', 0):>4.1f}  "
              f"shots {shots:>3}  offsides {off}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
