"""IS THE CAM's PASS MAP A DEFECT, OR IS EVERYONE'S? (2026-10-14)

The user's second complaint: "Kwame Asante's pass third distribution really
haunts me." Reading the dashboard gives a tight starburst of origins around the
centre circle with 27 of 50 passes lateral. Two questions hide in that:

  A. Is Asante an OUTLIER, or is this the engine's normal pass map for
     everybody? An outlier is a player problem; a norm is a model problem, and
     they get completely different fixes. Reporting one player's percentage
     without the distribution behind it is the error AGENTS.md keeps recording
     as "a column that reads cleanly is not a column that is right."
  B. Where does his ball actually arrive? BALL_RECEIPT events carry the real
     tracked position, so pass ORIGIN (which is the chain ANCHOR, per
     AGENTS.md) can be compared against where he demonstrably received it.

BENCHMARKS USED, and stated so they can be argued with:
  real PL #10 (e.g. Bruno Fernandes, Odegaard): pass origins centred ~62-70
  nx, roughly 35-40% forward / 35-40% lateral / 20-25% backward, and 25-35%
  of his passes ORIGINATE in the final third.

Run:  python _diag_cam_passmap.py [path.json]
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

DEFAULT = os.path.join(
    "plofa_output", "Hartwell_City_vs_Thornfield_United_MD01",
    "Hartwell_City_vs_Thornfield_United_MD1.json",
)
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
doc = json.load(open(path, encoding="utf-8"))
tl = doc["timeline"]
home = doc["match"]["home_team"]
away = doc["match"]["away_team"]


def nx(e):
    x = e.get("x")
    return None if x is None else (float(x) if e["team"] == home else 105.0 - float(x))


def third(v):
    return "?" if v is None else ("DEF" if v < 35 else ("MID" if v < 70 else "ATT"))


POS = {}
# `players` is a DICT here (team -> rows) in this exporter version, not a list.
# Probed rather than assumed: a `p.get` on a str raises, which is the cheapest
# possible way to find out the shape.
_players = doc["players"]
if isinstance(_players, dict):
    for _team, _plist in _players.items():
        for p in _plist:
            if isinstance(p, dict):
                nm = p.get("name") or p.get("player")
                if nm:
                    POS[nm] = p.get("position") or p.get("pos") or "?"

print(f"file {path}   events {len(tl)}")
_teams = list(doc["players"].keys()) if isinstance(doc["players"], dict) else []
print(f"players block: {'dict, teams=' + str(_teams) if _teams else type(doc['players'])}")
print()

# ── A. DIRECTION MIX, EVERY PLAYER ────────────────────────────────────
print("=" * 100)
print("A. PASS DIRECTION MIX, EVERY PLAYER WITH >= 10 PASSES")
print("   REAL PL: roughly 35-40 fwd / 35-40 lateral / 20-25 back, by role.")
print("-" * 100)
print(f"{'player':<18}{'pos':<6}{'n':>5}{'fwd':>7}{'lat':>7}{'back':>7}"
      f"{'p50 nx':>9}{'orig ATT%':>11}{'len med':>9}")
rows = []
for e in tl:
    if e.get("kind") != "pass":
        continue
    ex = e.get("end_x")
    if ex is None or e.get("x") is None:
        continue
    d = float(ex) - float(e["x"])
    if e["team"] != home:
        d = -d
    rows.append((e["player"], d, nx(e),
                 math.hypot(float(ex) - float(e["x"]),
                            float(e.get("end_y") or 0) - float(e.get("y") or 0))))
by = defaultdict(list)
for r in rows:
    by[r[0]].append(r)
for nm, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    if len(v) < 10:
        continue
    f = sum(1 for r in v if r[1] > 6.0)
    b = sum(1 for r in v if r[1] < -6.0)
    s = len(v) - f - b
    xs = [r[2] for r in v if r[2] is not None]
    at = sum(1 for x in xs if third(x) == "ATT")
    ln = statistics.median([r[3] for r in v])
    print(f"{nm:<18}{POS.get(nm,'?'):<6}{len(v):>5}"
          f"{f/len(v)*100:>6.0f}%{s/len(v)*100:>6.0f}%{b/len(v)*100:>6.0f}%"
          f"{statistics.median(xs):>9.1f}{at/max(1,len(xs))*100:>10.0f}%{ln:>9.1f}")
print()

# ── B. RECEIPT vs ORIGIN, for the two CAMs and their fullbacks ─────────
print("=" * 100)
print("B. WHERE THE BALL ARRIVED vs WHERE THE PASS LEFT FROM")
print("   BALL_RECEIPT is a real tracked touch; a PASS origin is the chain")
print("   anchor. If they agree, the pass map is honest about the position.")
print("-" * 100)
print(f"{'player':<18}{'pos':<6}{'n_rec':>7}{'rec p50':>9}{'rec ATT%':>10}"
      f"{'n_pass':>8}{'org p50':>9}{'org ATT%':>10}{'gap':>8}")
for nm in ("Kwame Asante", "Kofi Mensah", "Darius Frost", "Rico Alves",
           "Finn Larsson", "Luca Ferrini", "Nico Strauss", "Adri Vela"):
    rec = [nx(e) for e in tl if e.get("type") == "BALL_RECEIPT"
           and e.get("player") == nm]
    rec = [v for v in rec if v is not None]
    pas = [r[2] for r in by.get(nm, []) if r[2] is not None]
    if len(rec) < 5 or len(pas) < 5:
        print(f"{nm:<18}{POS.get(nm,'?'):<6}{len(rec):>7}"
              f"   -- NO VERDICT (need >=5 of each) --")
        continue
    ra = sum(1 for v in rec if third(v) == "ATT") / len(rec) * 100
    pa = sum(1 for v in pas if third(v) == "ATT") / len(pas) * 100
    print(f"{nm:<18}{POS.get(nm,'?'):<6}{len(rec):>7}{statistics.median(rec):>9.1f}"
          f"{ra:>9.0f}%{len(pas):>8}{statistics.median(pas):>9.1f}{pa:>9.0f}%"
          f"{statistics.median(rec)-statistics.median(pas):>8.1f}")
print()

# ── C. THE ONE NUMBER THAT SEPARATES A #10 FROM A BLOB ────────────────
print("=" * 100)
print("C. IS THE PASS MAP A MAP OR A STARBURST?")
print("   Two passes from the same origin bucket are a 'map'. One tight")
print("   cluster is a 'blob'. Measured as the 90% ellipse area of the")
print("   origins, and as how many passes leave the SAME 8 m cell.")
print("-" * 100)
print(f"{'player':<18}{'n':>5}{'origin bbox m2':>17}{'top cell share':>16}"
      f"{'distinct 8m cells':>19}")
for nm in ("Kwame Asante", "Kofi Mensah", "Finn Larsson", "Luca Ferrini",
           "Darius Frost", "Rico Alves", "Adri Vela", "Bruno Reis"):
    v = [r for r in by.get(nm, []) if r[2] is not None]
    if len(v) < 10:
        print(f"{nm:<18}{len(v):>5}   -- NO VERDICT --")
        continue
    xs = [r[2] for r in v]
    ys = []
    for e in tl:
        pass
    # rebuild y in the raw frame alongside nx
    pts = []
    for e in tl:
        if e.get("kind") != "pass" or e.get("player") != nm or e.get("x") is None:
            continue
        yy = e.get("y")
        if yy is None:
            continue
        pts.append((nx(e), float(yy)))
    if len(pts) < 10:
        print(f"{nm:<18}{len(pts):>5}   -- NO VERDICT --")
        continue
    axs = [p[0] for p in pts]
    ays = [p[1] for p in pts]
    bbox = (max(axs) - min(axs)) * (max(ays) - min(ays))
    cells = Counter((int(p[0] // 8), int(p[1] // 8)) for p in pts)
    top, topn = cells.most_common(1)[0]
    print(f"{nm:<18}{len(pts):>5}{bbox:>17.0f}{topn/len(pts)*100:>15.0f}%"
          f"{len(cells):>19}")
print()
print("   A real distributor's origins occupy 15-30 distinct 8 m cells.")
print("   top-cell share above ~35% means he receives in one place.")
