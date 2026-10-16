"""
_diag_momentum.py — semantics of the xT momentum model.

Every check here corresponds to a claim in `momentum_model`'s docstring that a
unit test alone would take on trust: that the frame is mirrored correctly,
that a goal cannot move the curve, and that an away attack is the mirror of a
home attack. Run:

    .venv\\Scripts\\python.exe tools\\diag\\_diag_momentum.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from match_engine import (EventType, MatchConfig, MatchEvent, MatchResult,
                          MatchState, SituationType, XGEngine)
import momentum_model as mm

HOME, AWAY = "Home FC", "Away FC"
FIXTURE_LAST = 30
FAILS = []


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
          f"{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


def ev(minute, etype, team, x, y, ex=None, ey=None, outcome=True,
       situation=SituationType.OPEN_PLAY, xg=0.0):
    return MatchEvent(minute=minute, second=30, event_type=etype, team=team,
                      player="P1", secondary_player="P2", situation=situation,
                      location_x=x, location_y=y, end_x=ex, end_y=ey,
                      xg=xg, outcome=outcome)


def build(tl):
    st = MatchState()
    return MatchResult(config=MatchConfig(home_team=HOME, away_team=AWAY),
                       state=st, timeline=list(tl), goals=[], cards=[], subs=[],
                       squads={}, threat=None, position_log=[], momentum_log=[])


def at(result, minute):
    """Momentum at ONE minute. Index arithmetic by eye is how the first
    version of this probe asserted minute 24 while calling it minute 25."""
    return next(r["momentum"] for r in result.series if r["minute"] == minute)


def covered(result, last=FIXTURE_LAST, start=6):
    """Only the populated minutes: the series always spans 90 and the EWMA
    window is 4, so minutes past the fixture are legitimately 0.0000.
    Asserting over all 90 rows was the first version's bug, and it produced
    four false failures at once."""
    return [r["momentum"] for r in result.series
            if start <= r["minute"] <= last]


# ── 1. GRID ─────────────────────────────────────────────────────────────
print("\n1. GRID")
g = mm.default_grid()
check("grid shape is 12x18",
      len(g) == 12 and all(len(r) == 18 for r in g))
check("value rises toward goal", g[6][0] < g[6][9] < g[6][17],
      f"own-half={g[6][0]:.4f} midfield={g[6][9]:.4f} edge={g[6][17]:.4f}")
check("grid symmetric about y=34",
      all(abs(g[r][c] - g[11 - r][c]) < 1e-12
          for r in range(12) for c in range(18)))

col, row = mm.cell_of(95.0, 34.0)
cx, cy = (col + 0.5) / 18 * 105.0, (row + 0.5) / 12 * 68.0
centre_xg = XGEngine.calculate_geometric(cx, cy, "right_foot",
                                         SituationType.OPEN_PLAY, False, True)
check("grid is XGEngine at each cell CENTRE (not a typed table)",
      abs(g[row][col] - centre_xg) < 1e-12,
      f"cell[{row}][{col}]={g[row][col]:.6f} vs engine at ({cx:.2f},{cy:.2f})"
      f"={centre_xg:.6f}")
check("lookup is BINNED: two points in one cell agree",
      mm.zone_value(g, 95.0, 33.0, True) == mm.zone_value(g, 95.0, 36.0, True))

# ── 2. IT MOVES WITHOUT A GOAL ──────────────────────────────────────────
print("\n2. IT MOVES WITHOUT A GOAL  (the defect being fixed)")
churn = []
for m in range(1, FIXTURE_LAST + 1):
    deep_x = 70 + (m / FIXTURE_LAST) * 22.0          # 70 -> 92
    churn.append(ev(m, EventType.PASS, HOME, 60, 30, deep_x, 32))
    # away REGRESSES: away's own goal is raw 105, so raw x INCREASES backwards
    churn.append(ev(m, EventType.PASS, AWAY, 60, 34, 78, 20))
r = mm.compute_momentum(build(churn))
vals = covered(r)
check("no goal scored anywhere",
      not any(e.event_type == EventType.GOAL for e in churn))
check("curve RISES as the match intensifies",
      at(r, 28) > 2.0 * at(r, 8) and max(vals) - min(vals) > 0.01,
      f"m8={at(r, 8):.4f} m28={at(r, 28):.4f}")
check("home is positive all match", all(v > 0 for v in vals),
      f"min={min(vals):.4f}")
check("regression is punished",
      mm.action_value(churn[1], g, False) < 0,
      f"away raw 60,34->78,20 = {mm.action_value(churn[1], g, False):+.5f}")
check("decays to zero once the fixture stops (correct, not a bug)",
      all(row["momentum"] == 0.0 for row in r.series
          if row["minute"] > FIXTURE_LAST + mm.WINDOW))

# ── 3. GOALS CARRY NO BONUS ─────────────────────────────────────────────
print("\n3. GOALS CARRY NO BONUS BEYOND THE SHOT THEY WERE")
clean_g = mm.action_value(ev(1, EventType.GOAL, HOME, 94, 34, 105, 34), g, True)
clean_s = mm.action_value(ev(1, EventType.SHOT_ON_TARGET, HOME, 94, 34), g, True)
scrap_g = mm.action_value(ev(1, EventType.GOAL, HOME, 88, 30,
                             situation=SituationType.CORNER), g, True)
scrap_s = mm.action_value(ev(1, EventType.SHOT_ON_TARGET, HOME, 88, 30,
                             situation=SituationType.CORNER), g, True)
check("a goal is valued exactly as the shot it was",
      abs(clean_g - clean_s) < 1e-12, f"goal={clean_g:.5f} shot={clean_s:.5f}")
check("same for a scrappy set-piece goal",
      abs(scrap_g - scrap_s) < 1e-12, f"{scrap_g:.5f}")
check("a scrappy corner goal is worth FAR LESS than a clean one",
      scrap_g < clean_g / 3.0,
      f"scramble={scrap_g:.5f} vs clean={clean_g:.5f} ({clean_g/scrap_g:.1f}x)")

base = mm.compute_momentum(build(churn))
withg = mm.compute_momentum(build(
    churn + [ev(20, EventType.GOAL, HOME, 94, 34, 105, 34, xg=0.5)]))
shift = at(withg, 20) - at(base, 20)
check("the goal minute moves only by the shot's own value, capped",
      0.0 <= shift <= mm.XT_CLIP + 1e-12,
      f"m20 {at(base, 20):.4f} -> {at(withg, 20):.4f} (delta {shift:+.5f})")
check("non-goal minutes outside the window are untouched",
      all(abs(at(withg, m) - at(base, m)) < 1e-12 for m in (10, 15, 40, 60)))
check("the goal is still visible as an annotation (scoreline advances)",
      next(x for x in withg.series if x["minute"] == 20)["home_goals"] == 1)

# ── 4. CORNER GOAL IN LOW MOMENTUM ──────────────────────────────────────
print("\n4. CORNER GOAL IN A LOW-MOMENTUM MINUTE")
quiet = [e for m in range(1, FIXTURE_LAST + 1)
         for e in (ev(m, EventType.PASS, HOME, 50, 34, 51, 33),
                   ev(m, EventType.PASS, AWAY, 50, 34, 51, 33))]
quiet += [ev(25, EventType.CORNER_TAKEN, HOME, 105, 0.5, 88, 30,
             situation=SituationType.CORNER),
          ev(25, EventType.GOAL, HOME, 88, 30, 105, 34, xg=0.05,
             situation=SituationType.CORNER)]
rq = mm.compute_momentum(build(quiet))
# "turgid" is only true BEFORE the corner's EWMA tail reaches back
turgid = max(r["momentum"] for r in rq.series if 6 <= r["minute"] <= 20)
clean_same = mm.compute_momentum(build(
    quiet[:24] + [ev(25, EventType.CORNER_TAKEN, HOME, 105, 0.5, 88, 30,
                     situation=SituationType.CORNER),
                  ev(25, EventType.GOAL, HOME, 94, 34, 105, ey=34, xg=0.4,
                     situation=SituationType.CORNER)]))
check("a turgid midfield has almost NO momentum",
      turgid < 0.005, f"peak 6-20 = {turgid:.5f}")
check("...so the scrappy corner goal sits in LOW momentum",
      at(rq, 25) < 0.02, f"m25 = {at(rq, 25):.5f}")
check("a CLEAN goal in the same minute sits much higher",
      at(clean_same, 25) > 1.5 * at(rq, 25),
      f"scrappy {at(rq, 25):.5f} vs clean {at(clean_same, 25):.5f} "
      f"({at(clean_same, 25)/at(rq, 25):.1f}x)")

# ── 5. FRAME ────────────────────────────────────────────────────────────
print("\n5. FRAME  (away attacks left)")
mirrored = []
for m in range(1, FIXTURE_LAST + 1):
    mirrored.append(ev(m, EventType.PASS, HOME, 60, 30, 88, 32))
    mirrored.append(ev(m, EventType.PASS, AWAY, 45, 30, 17, 32))
vm = covered(mm.compute_momentum(build(mirrored)))
check("identical attack either side => momentum ~0",
      abs(sum(vm) / len(vm)) < 1e-9, f"mean = {sum(vm)/len(vm):.9f}")

vh = covered(mm.compute_momentum(build(
    [ev(m, EventType.PASS, HOME, 60, 30, 88, 32)
     for m in range(1, FIXTURE_LAST + 1)])))
va = covered(mm.compute_momentum(build(
    [ev(m, EventType.PASS, AWAY, 45, 30, 17, 32)
     for m in range(1, FIXTURE_LAST + 1)])))
check("a one-sided match gives the mirrored sign exactly",
      all(h > 0 for h in vh) and all(a < 0 for a in va)
      and abs(sum(vh) + sum(va)) < 1e-9,
      f"home {sum(vh):+.4f} vs away {sum(va):+.4f}")

# ── 6. PURITY ───────────────────────────────────────────────────────────
print("\n6. PURITY")
tl = [ev(m, EventType.PASS, HOME, 60, 30, 88, 32)
      for m in range(1, FIXTURE_LAST + 1)]
random.seed(1234)
before = random.getstate()
mm.compute_momentum(build(tl))
check("global random stream untouched", random.getstate() == before)
check("deterministic across calls",
      mm.compute_momentum(build(tl)).series ==
      mm.compute_momentum(build(tl)).series)

# ── 7. OFF-PITCH COORDINATES ────────────────────────────────────────────
print("\n7. OFF-PITCH COORDINATES")
ro = mm.compute_momentum(build(
    [ev(m, EventType.PASS, HOME, 60, 30, 119.7, -12.8)
     for m in range(1, FIXTURE_LAST + 1)]))
check("did not raise, and spans the full match",
      len(ro.series) == 90, f"rows={len(ro.series)}")
check("clamps are COUNTED, not absorbed",
      ro.diagnostics["clamped_coordinates"] > 0,
      f"clamped_coordinates={ro.diagnostics['clamped_coordinates']}")

print("\n" + "=" * 62)
print("RESULT:", "ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED")
for f in FAILS:
    print("   FAILED:", f)
print("=" * 62)
sys.exit(1 if FAILS else 0)