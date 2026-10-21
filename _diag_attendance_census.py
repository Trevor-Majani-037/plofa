"""THE ATTENDANCE CENSUS — the new model against the real benchmark.

The old model was a hand-tuned tier table.  The new one is built on the
real Premier League 24/25 benchmark (96.8% mean occupancy, 40,474 per
match).  This census asks the only question that matters: **does PLOFA's
league now look like a real top flight?**

It answers it per club, from the season's own inputs:
  * the ground's capacity (the recorded fixture capacity),
  * the club's final league position (the warehouse's standings),
  * the club's average squad rating (the real squad, built from the
    workbook).

It then prints the OLD numbers beside the NEW ones, because the old numbers
are the recorded attendance — the thing the user was looking at.

READ-ONLY: the warehouse is opened read-only and nothing is written.

Run:  .venv\\Scripts\\python.exe -u _diag_attendance_census.py
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import attendance_model as AM  # noqa: E402

DB = "alltime.db"


def _standings(con):
    """{club: rank} from the warehouse's standings for the current season.

    The table's column is `rank`, not `position`, and it carries a
    `scope` — the all-time cumulative table is a different population
    from the season's, so the scope has to be picked deliberately (the
    first version of this probe read the wrong scope and silently
    returned nothing, which is why every club showed position 0).
    """
    cur = con.cursor()
    scopes = list(cur.execute(
        "SELECT season, scope, COUNT(*) FROM season_standings "
        "GROUP BY season, scope"))
    print("standings populations:", scopes)
    out = {}
    for season, scope, _n in scopes:
        if str(scope).startswith("alltime"):
            continue
        rows = list(cur.execute(
            "SELECT t.name, s.rank FROM season_standings s "
            "JOIN teams t ON t.team_id = s.team_id "
            "WHERE s.season = ? AND s.scope = ?", (season, scope)))
        if rows:
            out = {name: rank for name, rank in rows}
            print(f"  using season={season} scope={scope} ({len(out)} teams)")
    return out


def main():
    import attendance_model
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = con.cursor()
    rows = list(cur.execute(
        "SELECT ht.name, m.attendance, m.capacity FROM matches m "
        "LEFT JOIN teams ht ON ht.team_id = m.home_team_id "
        "WHERE m.attendance IS NOT NULL"))
    con.close()
    if not rows:
        print("no recorded matches")
        return

    pos = _standings(sqlite3.connect(f"file:{DB}?mode=ro", uri=True))

    # the squad ratings, measured per club
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    loader = get_loader("PLOFA-2026-2027.xlsx")
    import openpyxl
    wb = openpyxl.load_workbook("PLOFA-2026-2027.xlsx", read_only=True,
                                data_only=True)
    ws = wb["PLAYERS"]
    hrows = ws.iter_rows(min_row=1, values_only=True)
    header = next(hrows)
    ix_club = header.index("CLUB")
    clubs = sorted({str(r[ix_club]) for r in hrows if r and r[ix_club]})
    wb.close()
    clubs = [c for c in clubs if c.lower() != "total"]

    rating = {}
    for c in clubs:
        try:
            raw = loader.build_matchday_squad(c)
            sq = SquadBuilder.build(c, starters=raw["starters"],
                                    substitutes=raw["substitutes"])
            rs = [p.dna.overall_rating for p in sq["starters"]]
            rating[c] = statistics.mean(rs) if rs else 70.0
        except Exception:
            rating[c] = 70.0

    # the league's ground sizes, and the old recorded numbers
    caps = defaultdict(list)
    old_att = defaultdict(list)
    for club, att, cap in rows:
        caps[club].append(cap)
        old_att[club].append(att)

    n_teams = max(len(caps), 2)
    table = standings = {}
    # the position score, on the same scale the exporter uses
    def pos_score(club):
        p = pos.get(club)
        if not p:
            return 0.0
        return 1.0 - 2.0 * ((p - 1) / (n_teams - 1))

    print("=== THE ATTENDANCE CENSUS: PLOFA 26/27 vs the real PL 24/25 ===\n")
    print(f"{'club':<18}{'ground':>9}{'pos':>5}{'OLD att':>10}{'OLD occ':>9}"
          f"{'NEW att':>10}{'NEW occ':>9}")
    new_occ, old_occ = [], []
    for club in sorted(caps):
        cap = statistics.mean(caps[club])
        oa = statistics.mean(old_att[club])
        hs = AM.market_score(rating.get(club, 70.0), pos_score(club))
        # a league-average opponent (0.5): the opponent's pull is an
        # ABSOLUTE term worth ±1 point of occupancy, so a per-club season
        # average is barely affected by it.  The positions used are the
        # FINAL table applied to every matchday, which is an approximation
        # and is stated as one.
        out = AM.attendance_for(capacity=int(round(cap)), home_score=hs,
                                away_score=0.5, match_multiplier=1.0,
                                residual=0.0)
        new_occ.append(out["fill_rate"] / 100.0)
        old_occ.append(oa / cap if cap else 0.0)
        print(f"{club:<18}{int(round(cap)):>9,}{pos.get(club, 0):>5}"
              f"{oa:>10,.0f}{100*oa/cap:>8.1f}%"
              f"{out['attendance']:>10,}{out['fill_rate']:>8.1f}%")

    print(f"\n=== THE TWO DISTRIBUTIONS ===")
    print(f"{'':22}{'OLD (recorded)':>16}{'NEW (model)':>15}"
          f"{'REAL PL 24/25':>16}")
    print(f"{'mean occupancy':22}{100*statistics.mean(old_occ):>15.1f}%"
          f"{100*statistics.mean(new_occ):>14.1f}%"
          f"{100*AM.PL_MEAN_OCCUPANCY_24_25:>15.1f}%")
    print(f"{'median occupancy':22}"
          f"{100*statistics.median(old_occ):>15.1f}%"
          f"{100*statistics.median(new_occ):>14.1f}%"
          f"{100*statistics.median([o/100 for _, _, _, _, o in AM.PL_24_25]):>15.1f}%")
    print(f"{'min occupancy':22}{100*min(old_occ):>15.1f}%"
          f"{100*min(new_occ):>14.1f}%"
          f"{100*min(o/100 for _, _, _, _, o in AM.PL_24_25):>15.1f}%")
    print(f"{'max occupancy':22}{100*max(old_occ):>15.1f}%"
          f"{100*max(new_occ):>14.1f}%"
          f"{100*max(o/100 for _, _, _, _, o in AM.PL_24_25):>15.1f}%")
    pinned_old = sum(1 for o in old_occ if o >= 0.9849)
    pinned_new = sum(1 for o in new_occ if o >= 0.9949)
    print(f"{'clubs pinned at the clamp':22}{pinned_old:>16}{pinned_new:>15}"
          f"{'0':>16}")

    mean_cap = statistics.mean([statistics.mean(caps[c]) for c in caps])
    mean_att = statistics.mean([statistics.mean(old_att[c]) for c in caps])
    new_att = statistics.mean(
        [AM.attendance_for(capacity=int(round(statistics.mean(caps[c]))),
                            home_score=AM.market_score(rating.get(c, 70.0),
                                                       pos_score(c)),
                            away_score=0.5, residual=0.0)["attendance"]
         for c in caps])
    print(f"\n{'mean attendance per club':22}{mean_att:>16,.0f}"
          f"{new_att:>15,.0f}"
          f"{AM.PL_MEAN_ATTENDANCE_24_25:>16,}")
    print(f"{'mean ground':22}{mean_cap:>16,.0f}{mean_cap:>15,.0f}"
          f"{'':>16}")


if __name__ == "__main__":
    main()
