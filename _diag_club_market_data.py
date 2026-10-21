"""WHAT PER-CLUB REAL DATA EXISTS FOR THE 18 LIVE CLUBS?

The user wants the fans scaled from something real — city populations "or
something realistic".  Before designing anything, read what the project
already holds per club, because the answer decides the design:

  * the 1997 club register (an EXTERNAL workbook the world layer already
    reads): Club_ID, Club_Name, City, District, Foundation_Year, Manager,
    Home_Ground, Capacity, Surface, Pitch_Size, Kit_Colors, Style, Sponsor,
    Budget_Class;
  * the live roster workbook: market value, salary, squad rating;
  * the world layer's own model: City.population, Stadium.capacity_by_season.

This prints the live clubs' register rows so the model can be designed
against what actually exists rather than against what would be convenient.

READ-ONLY: it opens the register workbook and never writes.

Run:  .venv\\Scripts\\python.exe -u _diag_club_market_data.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from world.archive import build_club_register, REGISTER_XLSX  # noqa: E402


def main():
    print(f"register: {REGISTER_XLSX}")
    if not os.path.exists(REGISTER_XLSX):
        print("  MISSING — the external register is not on this machine")
        return
    reg = build_club_register(REGISTER_XLSX)
    print(f"clubs in the register: {len(reg)}")

    live = reg.live_clubs()
    print(f"live clubs (per the history archive): {len(live)}")

    # The 18 clubs that actually play in the 26/27 season, looked up BY NAME
    # in the register (the history archive may be absent, so `live_clubs()`
    # can be empty on this machine).
    from roster_loader import get_loader
    loader = get_loader("PLOFA-2026-2027.xlsx")
    import openpyxl
    wb = openpyxl.load_workbook("PLOFA-2026-2027.xlsx", read_only=True,
                                data_only=True)
    ws = wb["PLAYERS"]
    rows = ws.iter_rows(min_row=1, values_only=True)
    header = next(rows)
    ix_club = header.index("CLUB")
    season_clubs = sorted({str(r[ix_club]) for r in rows
                            if r and r[ix_club]})
    wb.close()
    print(f"clubs in the 26/27 roster workbook: {len(season_clubs)}")

    print(f"\n{'club':<20}{'in reg?':<9}{'city':<16}{'district':<14}"
          f"{'founded':>8}{'capacity':>10}{'budget':>10}")
    for name in season_clubs:
        c = reg.get(name)
        if c is None:
            print(f"{name:<20}{'NO':<9}{'-':<16}{'-':<14}{'-':>8}{'-':>10}{'-':>10}")
            continue
        print(f"{name:<20}{'yes':<9}{c.city:<16}{c.district:<14}"
              f"{str(c.founded or ''):>8}{str(c.capacity or ''):>10}"
              f"{c.budget_class:<10}")

    # what the register does NOT have
    print("\n=== WHAT THE REGISTER DOES NOT HAVE ===")
    fields = set()
    for c in list(reg)[:1]:
        fields = set(c.to_dict().keys())
    for f in ("population", "city_population", "catchment", "supporters",
              "season_tickets", "members"):
        print(f"  {f:<18} {'PRESENT' if f in fields else 'ABSENT'}")

    caps = [c.capacity for c in reg if c.capacity]
    if caps:
        print(f"\ncapacities across ALL {len(caps)} register clubs: "
              f"min {min(caps):,}  max {max(caps):,}  ({len(set(caps))} distinct)")
    bc = {}
    for c in reg:
        bc[c.budget_class] = bc.get(c.budget_class, 0) + 1
    print(f"budget classes: {bc}")


if __name__ == "__main__":
    main()
