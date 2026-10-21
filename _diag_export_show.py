"""WHAT WILL THE USER ACTUALLY SEE IN THE EXPORT?

The user asked: "now i will see flow by flow events with correct times?"

This prints the REAL exported timeline (the file they read), checks that it
flows, and measures the one thing that could still be a wart: the row's
`minute` is the engine's MATCH minute, while `second` comes from the true
CLOCK — and the clock drifts from the match minute because dead time and
the 10-30 s goal celebrations are inside it.

Run:  .venv\\Scripts\\python.exe -u _diag_export_show.py
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HOME, AWAY = "Oxton", "Natrican"


def main():
    from tools.diag._diag_watch import build
    from player_dna import SquadBuilder
    import exporter as EX

    eng, home_raw, away_raw = build()
    home = SquadBuilder.build(HOME, starters=home_raw["starters"],
                              substitutes=home_raw["substitutes"])
    away = SquadBuilder.build(AWAY, starters=away_raw["starters"],
                              substitutes=away_raw["substitutes"])
    all_players = {HOME: home, AWAY: away}

    random.seed(31)
    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real

    ex = EX.PLOFAExporter(res, all_players)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m.json")
    ex.export_json(path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    tl = doc["timeline"]
    print(f"timeline rows: {len(tl)}")
    print("\n=== THE FIRST 14 ROWS — this is what the user reads ===")
    print(f"{'#':>5} {'minute':>6} {'second':>7} {'clock':>10} {'clock_s':>9} "
          f"{'timestamp(ms)':>13}  {'type':<20} player")
    for i, r in enumerate(tl[:14]):
        print(f"{i:>5} {r['minute']:>6} {r['second']:>7} {r['clock']:>10} "
              f"{r['clock_s']:>9} {r['timestamp']:>13}  {r['type']:<20} "
              f"{r['player']}")

    print("\n=== THE LAST 4 ROWS ===")
    for i, r in enumerate(tl[-4:], start=len(tl) - 4):
        print(f"{i:>5} {r['minute']:>6} {r['second']:>7} {r['clock']:>10} "
              f"{r['clock_s']:>9} {r['timestamp']:>13}  {r['type']:<20} "
              f"{r['player']}")

    # ── the flow ────────────────────────────────────────────────────────
    ts = [r["timestamp"] for r in tl]
    back = [(i, a, b) for i, (a, b) in enumerate(zip(ts, ts[1:])) if b < a]
    print(f"\n=== FLOW ===")
    print(f"backwards steps: {len(back)}")
    if back:
        for i, a, b in back[:5]:
            print(f"   at {i}: {a} -> {b}")
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    gaps_sorted = sorted(gaps)
    print(f"gap between consecutive events (ms): "
          f"min {gaps_sorted[0]} / median {gaps_sorted[len(gaps_sorted)//2]} / "
          f"max {gaps_sorted[-1]}")
    print(f"span: {ts[0]} -> {ts[-1]} ms = {ts[-1]/60000:.2f} min")

    # ── THE WART: minute vs clock ───────────────────────────────────────
    print("\n=== MINUTE (the engine's MATCH minute) vs THE CLOCK ===")
    drift = []
    for r in tl:
        clock_min = int(r["clock_s"] // 60) + 1
        drift.append(r["minute"] - clock_min)
    n_back = sum(1 for d in drift if d < 0)
    n_fwd = sum(1 for d in drift if d > 0)
    worst = max(drift, key=abs)
    print(f"rows where the clock's minute is AHEAD of the match minute: {n_fwd}")
    print(f"rows where it is BEHIND: {n_back}")
    print(f"largest single-row difference: {worst} min")
    print(f"the last 5 rows' minute vs clock: "
          + ", ".join(f"{r['minute']}/{int(r['clock_s']//60)+1}" for r in tl[-5:]))
    print("(the clock carries dead time and the 10-30 s goal celebrations, "
          "so it drifts a minute or two over a match — `clock`/`clock_s`/"
          "timestamp are the truth, `minute` is the football minute)")

    # ── what the EXCEL sheets carry ─────────────────────────────────────
    print("\n=== THE EXCEL SHEETS: which carry a time? ===")
    try:
        import pandas as pd
        xlsx = os.path.join(tmp, "m.xlsx")
        ex.export_excel(xlsx) if hasattr(ex, "export_excel") else None
        if os.path.exists(xlsx):
            names = pd.ExcelFile(xlsx).sheet_names
            for n in names:
                df = pd.read_excel(xlsx, sheet_name=n, nrows=2)
                cols = [c for c in df.columns
                        if str(c).strip().lower() in
                        ("minute", "second", "clock", "time", "timestamp")]
                if cols:
                    print(f"   {n:<28} {cols}")
    except Exception as exc:
        print(f"   (could not read the workbook back: {exc})")


if __name__ == "__main__":
    main()
