"""WHAT THE ATTENDANCE ACTUALLY LOOKS LIKE ACROSS THE SEASON.

The user says the fan numbers are hand-tuned and should be scaled from
something real.  Before proposing anything, read what the model currently
produces: every match in the warehouse, with the club, the ground's capacity
and the crowd.

This is READ-ONLY: it opens the warehouse with a read-only connection and
never writes.

Run:  .venv\\Scripts\\python.exe -u _diag_attendance_now.py
"""
from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DB = "alltime.db"


def main():
    if not os.path.exists(DB):
        print(f"no {DB} in {os.getcwd()}")
        return
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = con.cursor()

    tabs = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    print("tables:", ", ".join(tabs))

    cols = [r[1] for r in cur.execute("PRAGMA table_info(matches)")]
    print("\nmatches columns:", ", ".join(cols))

    if not {"attendance", "capacity"} <= set(cols):
        print("\nno attendance/capacity on `matches` — looking elsewhere")
        for t in tabs:
            c = [r[1] for r in cur.execute(f"PRAGMA table_info({t})")]
            if "attendance" in c:
                print(f"  {t}: {c}")
        con.close()
        return

    rows = list(cur.execute(
        "SELECT ht.name, at.name, m.attendance, m.capacity, m.match_date "
        "FROM matches m "
        "LEFT JOIN teams ht ON ht.team_id = m.home_team_id "
        "LEFT JOIN teams at ON at.team_id = m.away_team_id "
        "WHERE m.attendance IS NOT NULL ORDER BY m.match_date"))

    print(f"\nmatches with attendance: {len(rows)}")
    if not rows:
        con.close()
        return

    print(f"\n{'date':<12}{'home':<22}{'away':<22}"
          f"{'attendance':>11}{'capacity':>10}{'fill%':>8}")
    for h, a, att, cap, d in rows:
        fill = (100.0 * att / cap) if cap else 0.0
        print(f"{str(d)[:10]:<12}{h:<22}{a:<22}{att:>11,}{cap:>10,}{fill:>7.1f}%")

    # ── the spread: is every club the same size? ──────────────────────
    from collections import defaultdict
    by_team = defaultdict(list)
    for h, a, att, cap, d in rows:
        by_team[h].append((att, cap))
    print(f"\n=== PER-HOME-CLUB SUMMARY ===")
    print(f"{'club':<22}{'n':>3}{'mean att':>10}{'mean cap':>10}{'mean fill%':>12}"
          f"{'min fill%':>11}{'max fill%':>11}")
    fills = []
    for team, v in sorted(by_team.items()):
        m_att = sum(x[0] for x in v) / len(v)
        m_cap = sum(x[1] for x in v) / len(v)
        f = [100.0 * x[0] / x[1] for x in v if x[1]]
        fills.extend(f)
        print(f"{team:<22}{len(v):>3}{m_att:>10,.0f}{m_cap:>10,.0f}"
              f"{sum(f)/len(f):>11.1f}%{min(f):>10.1f}%{max(f):>10.1f}%")

    allatt = [x[0] for h, a, x0, c, d in rows for x in [(x0, c)]]
    print(f"\n=== THE WHOLE SEASON ===")
    print(f"attendance: min {min(allatt):,}  mean {sum(allatt)/len(allatt):,.0f}  "
          f"max {max(allatt):,}")
    print(f"fill rate:  min {min(fills):.1f}%  mean {sum(fills)/len(fills):.1f}%  "
          f"max {max(fills):.1f}%")
    print(f"capacity:   {len(set(c for h,a,x,c,d in rows))} distinct values "
          f"over {len(set(h for h,a,x,c,d in rows))} home clubs")

    con.close()


if __name__ == "__main__":
    main()
