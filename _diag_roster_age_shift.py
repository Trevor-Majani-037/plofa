"""Why did the probe's match change between two runs of the SAME fixture?

The first probe run (2026-10-20) reported 3,056 events; the second
(2026-10-21) reported 2,885 — with the same seed and the same code path
apart from the export fix, which consumes no RNG.

`roster_loader.PlayerRecord.age` uses `date.today()`, and `player_dna`
multiplies every drawn attribute by the player's AGE CURVE. So a player whose
birthday falls between the two runs is a day older, their attributes move,
and the match diverges completely (chaos — one changed probability is enough).

This reads the workbook's own DOB column and counts the birthdays on the
boundary, so the explanation is CHECKED rather than asserted.

Run:  .venv\\Scripts\\python.exe -u _diag_roster_age_shift.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime

sys.path.insert(0, ".")

XLSX = "PLOFA-2026-2027.xlsx"
BOUNDARY_DAYS = {(10, 20), (10, 21)}


def main():
    # The Windows console is cp1252 and real player names carry 'ć' / 'ń'
    # (the same crash `alltime_db` had).  Force UTF-8 rather than dropping
    # the names.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["PLAYERS"]
    rows = ws.iter_rows(min_row=1, values_only=True)
    header = next(rows)
    ix_club = header.index("CLUB")
    ix_name = header.index("PLAYER NAME")
    ix_dob = header.index("DOB")
    ix_age = header.index("AGE")
    today = date.today()
    total = 0
    hits = []
    for r in rows:
        if not r or not r[ix_name]:
            continue
        total += 1
        dob = r[ix_dob]
        if dob is None:
            continue
        if isinstance(dob, datetime):
            dob = dob.date()
        elif isinstance(dob, str):
            try:
                dob = datetime.strptime(dob.strip(), "%Y-%m-%d").date()
            except ValueError:
                continue
        if not isinstance(dob, date):
            continue
        if (dob.month, dob.day) in BOUNDARY_DAYS:
            hits.append((str(r[ix_club]), str(r[ix_name]), dob, r[ix_age]))
    wb.close()

    print(f"today: {today}")
    print(f"players in the workbook: {total}")
    print(f"birthdays on the boundary ({sorted(BOUNDARY_DAYS)}): {len(hits)}")
    for c, n, d, a in hits:
        print(f"  {c:<18} {n:<22} dob {d}   workbook AGE cell {a}")
    if not hits:
        print("  (none on the boundary — the age shift is then NOT the cause "
              "and the difference is something else; do not assume)")


if __name__ == "__main__":
    main()
