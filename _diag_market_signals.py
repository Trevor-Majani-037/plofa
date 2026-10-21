"""THE MARKET SIGNAL: PLOFA's own squad ratings, and the season's standings.

The new attendance model needs a per-club MARKET SCORE.  The candidate
signals the exporter already has are the squad's average DNA rating and the
league position.  Before mapping either to a 0..1 score, MEASURE their
distributions — a mapping invented against an unmeasured distribution is
exactly the hand-tuned number this work is replacing.

Also prints the season's standings shape (the position scores the exporter
already computes) and the league's ground capacities, so the over-build
term has a real reference scale.

Run:  .venv\\Scripts\\python.exe -u _diag_market_signals.py
"""
from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from player_dna import SquadBuilder  # noqa: E402
from roster_loader import get_loader  # noqa: E402


def main():
    loader = get_loader("PLOFA-2026-2027.xlsx")
    import openpyxl
    wb = openpyxl.load_workbook("PLOFA-2026-2027.xlsx", read_only=True,
                                data_only=True)
    ws = wb["PLAYERS"]
    rows = ws.iter_rows(min_row=1, values_only=True)
    header = next(rows)
    ix_club = header.index("CLUB")
    # the workbook's header carries a newline in this column name
    ix_mv = next((i for i, h in enumerate(header)
                  if h and "Market" in str(h)), None)
    clubs = sorted({str(r[ix_club]) for r in rows if r and r[ix_club]})
    wb.close()
    # the workbook carries a trailing "Total" row — a spreadsheet artefact,
    # not a club (the register's own loader rejects the same shape)
    clubs = [c for c in clubs if c.lower() != "total"]

    print(f"clubs: {len(clubs)}")
    ratings = {}
    values = {}
    for c in clubs:
        raw = loader.build_matchday_squad(c)
        squad = SquadBuilder.build(c, starters=raw["starters"],
                                   substitutes=raw["substitutes"])
        rs = [p.dna.overall_rating for p in squad["starters"]]
        ratings[c] = statistics.mean(rs) if rs else 0.0
        mv = 0.0
        for p in list(raw["starters"]) + list(raw["substitutes"]):
            v = getattr(p, "market_value", None)
            if isinstance(v, (int, float)):
                mv += float(v)
        values[c] = mv

    print(f"\n{'club':<20}{'avg DNA rating':>16}{'squad market value':>22}")
    for c in clubs:
        print(f"{c:<20}{ratings.get(c, 0):>16.1f}{values.get(c, 0):>22,.0f}")

    vals = [v for v in ratings.values() if v]
    print(f"\nsquad rating: min {min(vals):.1f}  median "
          f"{statistics.median(vals):.1f}  max {max(vals):.1f}")
    mvs = [v for v in values.values() if v]
    if mvs:
        print(f"squad market value: min {min(mvs):,.0f}  median "
              f"{statistics.median(mvs):,.0f}  max {max(mvs):,.0f}")

    # the correlation between the two market signals
    try:
        pairs = [(ratings[c], values[c]) for c in clubs
                 if ratings.get(c) and values.get(c)]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        print(f"correlation(squad rating, squad market value) = "
              f"{cov / (sx * sy):.3f}  over {len(pairs)} clubs")
    except Exception as exc:
        print(f"correlation failed: {exc}")


if __name__ == "__main__":
    main()
