"""Is TODAY'S fixture date-sensitive?  (the calendar trap)

`roster_loader` computes each player's age against `date.today()`, and
`player_dna` multiplies the drawn attributes by the age curve — so a player
whose BIRTHDAY IS TODAY is a day older than they were yesterday, their
attributes move, and the match diverges completely (chaos: one changed
probability is enough).

So a real-roster match is reproducible for a FIXED roster and NOT across
calendar days.  This lists every player whose birthday falls in the window
that matters for the transition, and says whether the FIXTURE's two clubs are
affected.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_roster_birthday_window.py \\
        "Natrican" "Tryox City"
"""
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def main() -> int:
    from roster_loader import get_loader
    today = date.today()
    window = {(today.month, today.day - 1), (today.month, today.day)}
    if today.day == 1:
        window = {(today.month, 30), (today.month, 1)}
    fixture = set(sys.argv[1:3]) if len(sys.argv) > 2 else set()
    loader = get_loader()
    rows = []
    for club in loader.get_all_clubs():
        for p in loader.get_players(club) if hasattr(loader, "get_players") else []:
            rows.append((club, p))
    if not rows:
        # fall back: the loader's own structure
        for club in loader.get_all_clubs():
            data = loader.get_club(club) if hasattr(loader, "get_club") else None
            if not data:
                continue
            for p in data.get("players", []):
                rows.append((club, p))
    print(f"today: {today.isoformat()}   the window: {sorted(window)}")
    hits = []
    for club, p in rows:
        dob = getattr(p, "dob", None) or getattr(p, "date_of_birth", None) \
            or getattr(p, "birth_date", None)
        if dob is None:
            continue
        if isinstance(dob, str):
            dob = date.fromisoformat(dob[:10])
        if (dob.month, dob.day) in window:
            hits.append((club, getattr(p, "name", "?"), dob))
    print(f"players in the workbook: {len(rows)}")
    print(f"birthdays in the window: {len(hits)}")
    affected = False
    for club, name, dob in hits:
        flag = ""
        if fixture and club in fixture:
            flag = "   <-- IN THE FIXTURE"
            affected = True
        print(f"  {club:<12} {name:<22} dob {dob.isoformat()}{flag}")
    if not hits:
        print("  (none)")
    print()
    if fixture and affected:
        print("VERDICT: THE FIXTURE IS DATE-SENSITIVE — matches run on")
        print("different days are not comparable.  Re-run BOTH arms today.")
    else:
        print("VERDICT: the fixture's clubs are not on the boundary, so the")
        print("matches are comparable across the date change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
