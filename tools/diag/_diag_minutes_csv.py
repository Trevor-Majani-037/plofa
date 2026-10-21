"""The export's per-player minutes — the pass network's filter.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_minutes_csv.py \\
        plofa_output/_v7_gate/Probe_FC_vs_Rival_FC_MD3_players.csv
"""
import csv
import sys


def main() -> int:
    path = sys.argv[1]
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"{path}: {len(rows)} players")
    for r in rows:
        print("  {:<22} {:<12} minutes={:>5} starter={}".format(
            r.get("player", ""), r.get("team", ""),
            r.get("minutes_played", ""), r.get("is_starter", "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
