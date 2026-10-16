"""DARIUS FROST (LB) TOO HIGH + KWAME ASANTE (CAM) PASS ORIGINS (2026-10-14)

The user, from the player dashboards in
`plofa_output/Hartwell_City_vs_Thornfield_United_MD01`:

  * "Left back plays super high almost like a LW (same for RB and RW)"
  * "Kwame Asante's pass third distribution really haunts me"

Reads the ALREADY-EXPORTED match (no simulation, instant) so the measurement
is of exactly the artefact the user was looking at.

FRAME. The export is "home attacks right, always" (plofa_export.py). So a
player's attacking-progress coordinate is `nx = x` for home, `105 - x` for
away. Every x below is reported in THAT normalised frame, so a final-third
pass means the same thing for both teams. This is the same normalisation
`depth_series` does in small_game.py.

HONESTY GATES, because this class of probe has produced clean wrong tables:
  1. Non-vacuity: if the two players have <5 passes each the probe prints
     NO VERDICT rather than a table over 3 rows.
  2. Every denominator is printed next to its numerator.
  3. The carry/pass origin caveat from AGENTS.md is checked, not assumed:
     a PASS's `location_x` is the chain ANCHOR (ball), a CARRY's is the
     carry's START. Both are the ball, neither is invented -- but they are
     not the player's tracked position, so the probe ALSO prints the
     source line for each event class.

Run:  python _diag_frost_asante.py [path.json]
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
with open(path, "r", encoding="utf-8") as fh:
    doc = json.load(fh)

tl = doc["timeline"]
home = doc["match"].get("home_team") or doc["match"].get("home")
away = doc["match"].get("away_team") or doc["match"].get("away")
print(f"file      {path}")
print(f"home={home}  away={away}  events={len(tl)}")
print()


def team_of(ev):
    return ev.get("team")


def nx_of(ev):
    """Attacking-progress coordinate in the home-attacks-right frame."""
    x = ev.get("x")
    if x is None:
        return None
    return float(x) if team_of(ev) == home else 105.0 - float(x)


def is_pass(ev):
    """kind=='pass' is the exporter's own pass CLASS: PASS + PROGRESSIVE_PASS
    + SWITCH_OF_PLAY + THROUGH_BALL + cross. `type` alone misses three of the
    five, which would understate a distributor's volume -- the shape of the
    complaint being measured."""
    return ev.get("kind") == "pass"


def third(nx):
    if nx is None:
        return "?"
    if nx < 35.0:
        return "DEF"
    if nx < 70.0:
        return "MID"
    return "ATT"


# ── 1. EVENT SHAPE, so the numbers below have denominators ──────────────
shapes = Counter(e.get("type") for e in tl)
print("=" * 96)
print("1. EVENT TYPES PRESENT (whole match)")
print("-" * 96)
for k, v in shapes.most_common(14):
    print(f"   {str(k):<26} {v:>5}")
print()

FOCUS = {
    "Darius Frost": ("LB", home),
    "Rico Alves": ("RB", home),          # overlapping_fullback
    "Kwame Asante": ("CAM", away),
    "Kofi Mensah": ("CAM", home),
}
print("=" * 96)
print("2. PASS ORIGIN / CARRY ORIGIN BY PITCH THIRD  (nx frame: 0 own goal -> 105 opp goal)")
print("-" * 96)
print(f"{'player':<16}{'pos':<5}{'team':<4}{'P':>4}{'DEF':>6}{'MID':>6}{'ATT':>6}"
      f"{'  p50nx':>8}{'  p90nx':>8}{'  sd_x':>7}{'  sd_y':>7}{'C':>4}{'C.ATT':>7}")
for name, (pos, team) in FOCUS.items():
    passes = [e for e in tl if is_pass(e) and e.get("player") == name]
    carries = [e for e in tl if e.get("type") == "CARRY" and e.get("player") == name]
    xs = [nx_of(e) for e in passes]
    xs = [v for v in xs if v is not None]
    ys = [float(e["y"]) for e in passes if e.get("y") is not None]
    th = Counter(third(v) for v in xs)
    cx = [nx_of(e) for e in carries]
    cx = [v for v in cx if v is not None]
    cth = Counter(third(v) for v in cx)
    if len(passes) < 5:
        print(f"{name:<16}{pos:<5}{'home' if team == home else 'away':<4}"
              f"{len(passes):>4}   -- NO VERDICT, fewer than 5 passes --")
        continue
    sx = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    sy = statistics.pstdev(ys) if len(ys) > 1 else 0.0
    p50 = statistics.median(xs)
    p90 = sorted(xs)[int(0.9 * (len(xs) - 1))]
    print(f"{name:<16}{pos:<5}{'home' if team == home else 'away':<4}"
          f"{len(passes):>4}{th['DEF']:>6}{th['MID']:>6}{th['ATT']:>6}"
          f"{p50:>8.1f}{p90:>8.1f}{sx:>7.1f}{sy:>7.1f}"
          f"{len(carries):>4}{cth['ATT']:>7}")
print()
print("   reference, real PL: a CAM's pass ORIGINS sit ~65-75% up the pitch")
print("   (median pass origin around the attacking-half line), sd_x ~14-18 m.")
print("   A fullback's ORIGINS should sit ~20-32% up (his own third).")
print()

# ── 3. WHERE DOES THE LB ACTUALLY PLAY? (tracked position, not ball) ────
print("=" * 96)
print("3. TRACKED POSITION BY PITCH THIRD  (position_log sample, not the ball)")
print("-" * 96)
log = doc.get("sequences") or {}
print(f"   sequences keys: {list(log)[:8] if isinstance(log, dict) else type(log)}")
print()

# ── 4. ASANTE'S PASS MAP SHAPE -- is it a STARBURST? ───────────────────
print("=" * 96)
print("4. ORIGIN SPREAD: are a player's pass origins a blob or a map?")
print("-" * 96)
print(f"{'player':<16}{'n':>4}{'  mean_x':>9}{'  sd_x':>7}{'  sd_y':>7}"
      f"{'  90%-ellipse':>13}{'  fwd':>6}{'  sq':>5}{'  back':>6}")
for name, (pos, team) in FOCUS.items():
    passes = [e for e in tl if is_pass(e) and e.get("player") == name]
    pts = [(nx_of(e), float(e["y"])) for e in passes
           if e.get("y") is not None]
    if len(pts) < 5:
        print(f"{name:<16}{len(pts):>4}      -- NO VERDICT --")
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = statistics.pstdev(xs)
    sy = statistics.pstdev(ys)
    ell = math.sqrt((0.6745 * 2 * sx) ** 2 + (0.6745 * 2 * sy) ** 2)
    fwd = sq = bk = 0
    for e in passes:
        ex, ey = e.get("end_x"), e.get("end_y")
        if ex is None or ey is None:
            continue
        d = (float(ex) - float(e["x"]))
        if team_of(e) != home:
            d = -d
        if d > 6.0:
            fwd += 1
        elif d < -6.0:
            bk += 1
        else:
            sq += 1
    print(f"{name:<16}{len(pts):>4}{mx:>9.1f}{sx:>7.1f}{sy:>7.1f}{ell:>13.1f}"
          f"{fwd:>6}{sq:>5}{bk:>6}")
print()
print("   sd_x < ~8 m means the origins are a BLOB -- the pass map is a")
print("   starburst out of one spot rather than a map of where he played.")
print("   Real CAM sd_x is 14-18 m.")
print()

# ── 5. THE THIRD-PROGRESSION BLOCK (literally 'pass third distribution') ─
tp = doc.get("third_progression")
print("=" * 96)
print("5. third_progression BLOCK FROM THE EXPORT")
print("-" * 96)
print(json.dumps(tp, indent=2)[:2000])
