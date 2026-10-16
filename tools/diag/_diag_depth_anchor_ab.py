"""A/B: DEPTH HOME ANCHOR -- fullbacks stop living at the byline (2026-10-14)

The change under test is ONE call added to the live shape chain
(`match_engine._offball_move_player`):

    depth_w = position_engine.depth_home_anchor_blend(pname)
    if depth_w > 0.0:
        tx += (getattr(st, "home_x", tx) - tx) * depth_w

It is a PREFERENCE, placed beside CK35, so it cannot outvote rest defence and
a pending set-piece slot overrides it for free.

THE FALSIFIABLE PREDICTION, stated before the run so it cannot be edited after:

  IF the diagnosis in `_diag_wide_depth_live.py` is right -- the shape target's
  anchor is `_minute_start_snapshot` (last minute's TRACKED position) rather
  than `state.home_x`, so the depth axis has no restoring force toward the
  formation post -- THEN:

    1. LB/RB median depth collapses from ~66 m to roughly 30-40 m.
    2. The CB clamp STOPS SATURATING. Fed the old recursion, every CB target
       exceeded CB_MAX_ADVANCE_NX = 58 and the three centre-backs sat ON the
       ceiling (p25/p50/p75 = 55.5/56.8/57.9). Anchored, they should fall to
       the high 20s and the clamp's share should go to near zero. **If the
       clamp does not un-saturate, the diagnosis is wrong.**
    3. LW/RW/CAM/ST are UNCHANGED. They have no depth defect and they are not
       in DEPTH_ANCHOR_POSITIONS. If they move, the shape has been flattened
       rather than the depth axis repaired, and the change is wrong.

ARM CONTROL. `--arm off` sets `DEPTH_ANCHOR_ENABLED = False`, which is
byte-identical to the pre-change behaviour (the method returns 0.0 before it
reads anything else). Matches replay byte-identically across processes since
the crc32 fix, so a two-arm comparison at a fixed seed measures this rule and
not hash order. **One seed is one sample** and the direction has to be
consistent across several surfaces before it means anything.

Run:  python _diag_depth_anchor_ab.py --arm off|on --seed N
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=("off", "on", "anchor", "anchor_only"), required=True,
                help="off = pre-change baseline | on = baseline + depth anchor | "
                     "anchor = baseline + formation anchor + depth anchor | "
                     "anchor_only = formation anchor WITHOUT the depth backstop")
# ── THE FRAME. Defaults to run_match.py's OWN HOME_STYLE / AWAY_STYLE, because
# that is the frame `plofa_output/Hartwell_City_vs_Thornfield_United_MD01` was
# produced under -- the match the user was looking at when they raised this.
#
# The first version of this probe hard-coded BALANCED/MIXED for both sides and
# then compared Asante's depth against a "real CAM 58-70" band. That band is a
# POSSESSION benchmark. Thornfield were FLUID_COUNTER/COUNTER, and a
# counter-attacking CAM is supposed to sit deeper -- so the comparison was
# invalid and the "regression" it reported was a category error. A benchmark
# that does not name its frame is not a benchmark.
# Defaults are None here on purpose: `run_match` is imported below, and an
# argparse default is evaluated at add_argument time. Resolved into real
# enums immediately after the import.
ap.add_argument("--home-style", default=None)
ap.add_argument("--away-style", default=None)
ap.add_argument("--home-playing", default=None)
ap.add_argument("--away-playing", default=None)
ap.add_argument("--home-intensity", default=None)
ap.add_argument("--away-intensity", default=None)
ap.add_argument("--seed", type=int, default=777)
A = ap.parse_args()

random.seed(A.seed)
try:
    import numpy as np
    np.random.seed(A.seed)
except Exception:  # pragma: no cover
    pass

import position_engine as PEMOD                     # noqa: E402
import run_match as RM                              # noqa: E402
from player_dna import SquadBuilder                # noqa: E402
from match_engine import (MatchConfig, MatchEngine, TeamProfile, TeamStyle,  # noqa: E402
                          PlayingStyle, Intensity)

# Resolve the frame defaults from run_match.py's OWN HOME_STYLE / AWAY_STYLE,
# because that is the frame `plofa_output/Hartwell_City_vs_Thornfield_United_
# MD01` was produced under -- the match the user was looking at when they
# raised this.
#
# The first version of this probe hard-coded BALANCED/MIXED for both sides and
# then compared Asante's depth against a "real CAM 58-70" band. That band is a
# POSSESSION benchmark. Thornfield were FLUID_COUNTER/COUNTER, and a
# counter-attacking CAM is supposed to sit deeper -- so the comparison was
# invalid and the "regression" it reported was a category error. A benchmark
# that does not name its frame is not a benchmark.
if A.home_style is None:
    A.home_style = RM.HOME_STYLE.style
if A.away_style is None:
    A.away_style = RM.AWAY_STYLE.style
if A.home_playing is None:
    A.home_playing = RM.HOME_STYLE.playing_style
if A.away_playing is None:
    A.away_playing = RM.AWAY_STYLE.playing_style
if A.home_intensity is None:
    A.home_intensity = RM.HOME_STYLE.intensity
if A.away_intensity is None:
    A.away_intensity = RM.AWAY_STYLE.intensity

# ── THE ARM. Set BEFORE the engine exists, never patched afterwards: a lever
# installed by monkey-patching after construction would leave every already-
# built state un-switched, which is the "probe reads exactly like an inert
# lever" failure recorded in AGENTS.md.
PEMOD.PositionEngine.DEPTH_ANCHOR_ENABLED = (A.arm in ("on", "anchor"))
PEMOD.PositionEngine.SHAPE_ANCHOR_FORMATION = (A.arm in ("anchor", "anchor_only"))

HOME, AWAY = RM.HOME_TEAM, RM.AWAY_TEAM
WANT = tuple(p[0] for p in RM.HOME_STARTERS) + tuple(p[0] for p in RM.AWAY_STARTERS)
POSNAME = {p[0]: p[1] for p in RM.HOME_STARTERS}
POSNAME.update({p[0]: p[1] for p in RM.AWAY_STARTERS})

SAMP = defaultdict(list)
CLAMP = {"n": 0, "bound": 0, "by_pos": defaultdict(lambda: [0, 0])}
IN_CLAWNS = {}

_pe = PEMOD.PositionEngine
_raw_lsr = _pe.live_spacing_redirect
_raw_clamp = _pe.cb_advance_clamp


def _patched_lsr(self, team_name, cx, cy, tx, ty):
    f = sys._getframe(1).f_locals
    nm = f.get("pname")
    if nm in WANT:
        st = self.states.get(nm)
        SAMP[nm].append({
            "nx": cx if team_name == HOME else 105.0 - cx,
            "tnx": tx if team_name == HOME else 105.0 - tx,
            "hnx": (st.home_x if team_name == HOME else 105.0 - st.home_x)
                   if st else None,
            "cy": cy,
            "hny": st.home_y if st else None,
            "pos": f.get("pos"), "poss": bool(f.get("has_ball")),
            # The target/position gap. `live_spacing_redirect` is called with
            # the FINISHED target (match_engine.py:2927), so tx IS where the
            # engine wants him and cx is where he is. A large lag means the
            # INTEGRATOR is not arriving, which is a different defect from the
            # target being wrong -- and the two are easy to confuse because a
            # rule that sets the target near where the player already stands
            # produces a small lag while being just as wrong.
            "lag": math.hypot(tx - cx, ty - cy),
        })
    return _raw_lsr(self, team_name, cx, cy, tx, ty)


def _patched_clamp(self, tx, attacks_right):
    out = _raw_clamp(self, tx, attacks_right)
    f = sys._getframe(1).f_locals
    nm = f.get("pname")
    pos = f.get("pos")
    CLAMP["n"] += 1
    if abs(out - tx) > 1e-9:
        CLAMP["bound"] += 1
        if nm in WANT:
            CLAMP["by_pos"][nm][0] += 1
    if nm in WANT:
        CLAMP["by_pos"][nm][1] += 1
    IN_CLAWNS["tx"] = tx
    return out


_pe.live_spacing_redirect = _patched_lsr
_pe.cb_advance_clamp = _patched_clamp

hs = SquadBuilder.build(team_name=HOME, starters=RM.HOME_STARTERS,
                        substitutes=RM.HOME_SUBS,
                        team_superstars=RM.HOME_SUPERSTARS,
                        set_piece_takers=RM.HOME_SP_TAKERS,
                        soul_assignments=RM.SOUL_PLAYERS)
aw = SquadBuilder.build(team_name=AWAY, starters=RM.AWAY_STARTERS,
                        substitutes=RM.AWAY_SUBS,
                        team_superstars=RM.AWAY_SUPERSTARS,
                        set_piece_takers=RM.AWAY_SP_TAKERS,
                        soul_assignments=RM.SOUL_PLAYERS)
eng = MatchEngine(
    MatchConfig(home_team=HOME, away_team=AWAY, match_date=RM.MATCH_DATE,
                matchday=RM.MATCHDAY, venue="X", stadium_capacity=45000),
    TeamProfile(name=HOME, style=A.home_style,
                playing_style=A.home_playing,
                intensity=A.home_intensity),
    TeamProfile(name=AWAY, style=A.away_style,
                playing_style=A.away_playing,
                intensity=A.away_intensity))
eng.set_squad(HOME, hs["starters"], hs["substitutes"])
eng.set_squad(AWAY, aw["starters"], aw["substitutes"])
res = eng.simulate()


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def rows0pos(rows):
    return next((r["pos"] for r in rows if r["pos"]), "?")


print(f"\n### ARM={A.arm.upper()}  DEPTH_ANCHOR="
      f"{PEMOD.PositionEngine.DEPTH_ANCHOR_ENABLED}  SHAPE_ANCHOR_FORMATION="
      f"{PEMOD.PositionEngine.SHAPE_ANCHOR_FORMATION}  seed={A.seed}  "
      f"events={len(res.timeline)}  "
      f"score={res.state.home_goals}-{res.state.away_goals}")
print(f"FRAME  home={A.home_style.value}/{A.home_playing.value}"
      f"  away={A.away_style.value}/{A.away_playing.value}"
      f"  (run_match.py's own frame -- the one the export was made under)")

print()
print("=" * 100)
print("1. PER-ROLE DEPTH  (nx: 0 own goal -> 105 opp goal)")
print("   PREDICTED: LB/RB and CB fall sharply; LW/RW/CAM/ST essentially flat.")
print("-" * 100)
print(f"{'pos':<6}{'n':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}"
      f"{'p50_in':>9}{'p50_out':>9}{'REAL*':>10}")
# BENCHMARK, CORRECTED 2026-10-15. The LB/RB row used to read "22-30", which is
# a deep-block defensive line and NOT a fullback. Opta/The Analyst, PL 2024-25,
# average full-back TOUCH location: league 53.2 m, Chelsea 46.8 (deepest team),
# Man City 60.1 (highest). Marked * because touches are events and these are tick
# medians -- same instrument, different aggregation, so treat as a band not a
# target. The pass-origin distribution in section 4 is the comparable one.
_real = {"CB": "20-26", "LB": "47-60*", "RB": "47-60*", "CDM": "35-45",
         "CM": "45-55", "CAM": "58-70", "LW": "62-78", "RW": "62-78",
         "ST": "70-80"}
_bp, _bi, _bo = defaultdict(list), defaultdict(list), defaultdict(list)
for nm, rows in SAMP.items():
    for r in rows:
        p = r["pos"] or "?"
        _bp[p].append(r["nx"])
        (_bi if r["poss"] else _bo)[p].append(r["nx"])
for p in sorted(_bp):
    v = _bp[p]
    print(f"{p:<6}{len(v):>8}{pct(v,.25):>8.1f}{pct(v,.5):>8.1f}"
          f"{pct(v,.75):>8.1f}{pct(v,.9):>8.1f}"
          f"{pct(_bi[p],.5):>9.1f}{pct(_bo[p],.5):>9.1f}{_real.get(p,'?'):>11}")
print()

print("=" * 100)
print("2. CB CLAMP SATURATION -- the falsifiable one")
print("   Fed the old recursion every CB target exceeded 58 and the CBs sat ON")
print("   the ceiling. Anchored, the clamp's binding share must COLLAPSE.")
print("-" * 100)
tot_b = sum(v[0] for v in CLAMP["by_pos"].values())
tot_n = sum(v[1] for v in CLAMP["by_pos"].values())
print(f"   cb_advance_clamp calls: {CLAMP['n']}   binding: {CLAMP['bound']} "
      f"({CLAMP['bound']/max(1,CLAMP['n'])*100:.1f}%)")
print(f"   on tracked CBs: {tot_b}/{tot_n} ({tot_b/max(1,tot_n)*100:.1f}%)")
cbs = [v for v in _bp.get("CB", [])]
if cbs:
    above = sum(1 for v in cbs if v > 58.0)
    print(f"   CB ticks beyond nx 58: {above}/{len(cbs)} "
          f"({above/len(cbs)*100:.1f}%)   [was: a distribution MODE at 56.8]")
print()

print("=" * 100)
print("3. THE THREE NAMED PLAYERS")
print("-" * 100)
print(f"{'player':<18}{'pos':<5}{'n':>7}{'p50':>8}{'p75':>8}{'p90':>8}"
      f"{'p50_tgt':>9}{'home':>7}")
for nm in ("Darius Frost", "Rico Alves", "Jide Afolabi", "Lee Sung-jin",
           "Emeka Obi", "Adri Vela", "Percy", "Kwame Asante",
           "Kofi Mensah", "Dragan Novak"):
    rows = SAMP.get(nm, [])
    if len(rows) < 300:
        print(f"{nm:<18}  -- NO VERDICT, {len(rows)} samples --")
        continue
    xs = [r["nx"] for r in rows]
    hn = next((r["hnx"] for r in rows if r["hnx"] is not None), None)
    print(f"{nm:<18}{rows[0]['pos']:<5}{len(rows):>7}{pct(xs,.5):>8.1f}"
          f"{pct(xs,.75):>8.1f}{pct(xs,.9):>8.1f}"
          f"{pct([r['tnx'] for r in rows],.5):>9.1f}"
          f"{(f'{hn:.1f}' if hn is not None else '?'):>7}")
print()

print("=" * 100)
print("4. THE METRIC THE COMPLAINT WAS MADE IN -- PASS ORIGINS")
print("   Opta/The Analyst, PL 2024-25, average full-back TOUCH location:")
print("   league 53.2 m, Chelsea 46.8 (deepest), Man City 60.1 (highest).")
print("   Touches are EVENTS and these are TICK medians, so they are not")
print("   directly comparable -- which is why this section reports the pass")
print("   origin distribution, the one instrument that is.")
print("   The complaint: 65% of Frost's passes from the attacking third, 61%")
print("   of them backward.")
print("-" * 100)
print(f"{'player':<18}{'pos':<6}{'n':>5}"
      f"{'DEF':>6}{'MID':>6}{'ATT':>6}{'p50 nx':>9}{'p90':>8}"
      f"{'fwd':>6}{'lat':>6}{'back':>7}")
from match_engine import EventType as _ET
for nm in ("Darius Frost", "Rico Alves", "Jide Afolabi", "Lee Sung-jin",
           "Adri Vela", "Dragan Novak",
           "Kwame Asante", "Kofi Mensah", "Mateo Sanz", "Finn Larsson"):
    # end_x is None on events with no recorded destination; the export counts
    # that as a WARNING rather than inventing a coordinate, and so does this.
    evs = [e for e in res.timeline
           if e.event_type == _ET.PASS and e.player == nm and e.end_x is not None]
    xs, fwd, lat, back = [], 0, 0, 0
    for e in evs:
        nx_ = (e.location_x if res.config.home_team == e.team
               else 105.0 - e.location_x)
        xs.append(nx_)
        dx = (e.end_x - e.location_x)
        if res.config.home_team != e.team:
            dx = -dx
        if dx > 6.0:
            fwd += 1
        elif dx < -6.0:
            back += 1
        else:
            lat += 1
    if len(xs) < 5:
        print(f"{nm:<18}  -- NO VERDICT, {len(xs)} passes --")
        continue
    d = sum(1 for v in xs if v < 35), sum(1 for v in xs if 35 <= v < 70), \
        sum(1 for v in xs if v >= 70)
    print(f"{nm:<18}{rows0pos(SAMP.get(nm, [])):<6}{len(xs):>5}"
          f"{d[0]:>6}{d[1]:>6}{d[2]:>6}"
          f"{pct(xs,.5):>9.1f}{pct(xs,.9):>8.1f}"
          f"{fwd/len(xs)*100:>5.0f}%{lat/len(xs)*100:>5.0f}%"
          f"{back/len(xs)*100:>6.0f}%")
print()
print("   pre-fix Frost: 31 passes, 3 DEF / 8 MID / 20 ATT, median 74.8,")
print("                  fwd 23 / lat 16 / back 61%.")
print()


print("=" * 100)
print("5. TARGET vs POSITION -- is the integrator arriving?")
print("   The `anchor_only` run produced p50 POSITION 55.6 against p50 TARGET")
print("   42.1. Those are DIFFERENT defects: a bad target, and a player who")
print("   cannot walk to the target it was given. Easy to confuse, because a")
print("   rule that sets the target near where the player already stands")
print("   yields a small lag while being just as wrong.")
print("-" * 100)
print(f"{'player':<18}{'pos':<5}{'n':>7}{'p50 lag':>9}{'p90 lag':>9}"
      f"{'lag IN':>9}{'lag OUT':>9}{'p50 tgt':>9}{'p50 pos':>9}")
for nm in ("Darius Frost", "Rico Alves", "Jide Afolabi", "Lee Sung-jin",
           "Emeka Obi", "Tavish Crane", "Adri Vela", "Kwame Asante"):
    rows = SAMP.get(nm, [])
    if len(rows) < 300:
        print(f"{nm:<18}  -- NO VERDICT, {len(rows)} samples --")
        continue
    lags = [r["lag"] for r in rows]
    lin = [r["lag"] for r in rows if r["poss"]]
    lout = [r["lag"] for r in rows if not r["poss"]]
    print(f"{nm:<18}{rows[0]['pos']:<5}{len(rows):>7}{pct(lags,.5):>9.1f}"
          f"{pct(lags,.9):>9.1f}{pct(lin,.5):>9.1f}{pct(lout,.5):>9.1f}"
          f"{pct([r['tnx'] for r in rows],.5):>9.1f}"
          f"{pct([r['nx'] for r in rows],.5):>9.1f}")
print()
print("   The `involved` gate at match_engine.py:2981 multiplies the jog speed")
print("   by 0.30 when the ball is >30 m from the player's MINUTE-START")
print("   position. Out of possession that is usually true, so the recovery jog")
print("   runs near 0.54 m/s and a 40 m walk home takes ~74 s -- longer than the")
print("   spell. lag OUT >> lag IN is the signature.")
print()

print("=" * 100)
print("6. THE CONTROL -- WIDTH MUST NOT HAVE MOVED")
print("   The rule touches tx only. If drift off the touchline changes, it is")
print("   NOT a depth-only rule and something else in the chain moved.")
print("-" * 100)
print(f"{'player':<18}{'pos':<5}{'n':>8}{'sd_y':>8}{'p50 |y-home_y|':>16}"
      f"{'p90 |y-home_y|':>16}")
for nm in ("Darius Frost", "Rico Alves", "Jide Afolabi", "Lee Sung-jin",
           "Adri Vela", "Percy", "Bruno Reis", "Tariq El-Amin"):
    rows = SAMP.get(nm, [])
    ys = [r for r in rows if r.get("hny") is not None and r.get("cy") is not None]
    if len(ys) < 300:
        print(f"{nm:<18}  -- NO VERDICT, {len(ys)} samples --")
        continue
    d = [abs(r["cy"] - r["hny"]) for r in ys]
    sy = statistics.pstdev([r["cy"] for r in ys])
    print(f"{nm:<18}{ys[0]['pos']:<5}{len(ys):>8}{sy:>8.1f}"
          f"{pct(d,.5):>16.1f}{pct(d,.9):>16.1f}")
print()
print("   Non-vacuity: if every cell above is 0.00, the y fields are not being")
print("   stored and this control has measured nothing -- that is a probe bug,")
print("   not a finding.")
print()
print("=" * 100)
print("7. IS ASANTE AN OUTLIER, OR HIS TEAM?  (the question actually asked)")
print("   Thornfield are FLUID_COUNTER/COUNTER. A counter-attacking side wins")
print("   the ball deep and breaks, so a team-wide low final-third share is the")
print("   STYLE, not a player defect. Compare like with like.")
print("-" * 100)
print(f"{'team':<20}{'n':>6}{'DEF':>6}{'MID':>6}{'ATT':>6}"
      f"{'p50 nx':>9}{'p90':>8}{'med len':>9}")
for tm in (HOME, AWAY):
    evs = [e for e in res.timeline
           if e.event_type == _ET.PASS and e.team == tm and e.end_x is not None]
    xs, ln = [], []
    for e in evs:
        xs.append(e.location_x if res.config.home_team == tm
                   else 105.0 - e.location_x)
        ln.append(math.hypot(e.end_x - e.location_x, e.end_y - e.location_y))
    if len(xs) < 5:
        print(f"{tm:<20}  -- NO VERDICT --")
        continue
    d = (sum(1 for v in xs if v < 35), sum(1 for v in xs if 35 <= v < 70),
         sum(1 for v in xs if v >= 70))
    print(f"{tm:<20}{len(xs):>6}{d[0]:>6}{d[1]:>6}{d[2]:>6}"
          f"{pct(xs,.5):>9.1f}{pct(xs,.9):>8.1f}"
          f"{statistics.median(ln):>9.1f}")
print()
print(f"{'player':<20}{'side':<6}{'n':>5}{'ATT':>6}{'p50 nx':>9}"
      f"{'% of team ATT':>16}")
for nm in ("Kwame Asante", "Kofi Mensah", "Bruno Reis", "Nico Strauss",
           "Tariq El-Amin", "Demi Adeola", "Finn Larsson", "Darius Frost"):
    evs = [e for e in res.timeline
           if e.event_type == _ET.PASS and e.player == nm and e.end_x is not None]
    if len(evs) < 5:
        continue
    xs = [e.location_x if res.config.home_team == e.team
          else 105.0 - e.location_x for e in evs]
    att = sum(1 for v in xs if v >= 70)
    tm = evs[0].team
    t_att = t_n = 0
    for e in res.timeline:
        if e.event_type != _ET.PASS or e.team != tm or e.end_x is None:
            continue
        t_n += 1
        if (e.location_x if res.config.home_team == tm
                else 105.0 - e.location_x) >= 70:
            t_att += 1
    print(f"{nm:<20}{'home' if tm == HOME else 'away':<6}{len(xs):>5}"
          f"{att/len(xs)*100:>5.0f}%{pct(xs,.5):>9.1f}"
          f"{att/max(1,t_att)*100:>15.0f}%")
print()
print("   '% of team ATT' = his share of the team's final-third passes. A CAM")
print("   far below it is not being given the ball there; a CAM at or above it")
print("   is on a team that simply does not go there.")
print()
print("=" * 100)
print("8. DOES THE BALL EVER REACH THE CAM?  (separates position from selection)")
print("   If his RECEIPTS are at the same depth as his PASS ORIGINS, his")
print("   positioning is doing what it is told and the gap is WHO GETS THE BALL")
print("   -- i.e. receiver selection, which AGENTS.md records as hand-coded in")
print("   AttackingMatrix (80% of targets) and TacticalPhase (20%).")
print("-" * 100)
print(f"{'player':<20}{'pos':<6}{'n_rec':>7}{'rec p50':>9}{'rec ATT%':>10}"
      f"{'org p50':>9}{'gap':>7}")
for nm in ("Kwame Asante", "Kofi Mensah", "Demi Adeola", "Finn Larsson",
           "Nico Strauss", "Bruno Reis", "Tariq El-Amin", "Darius Frost",
           "Adri Vela"):
    rec = [e.location_x if res.config.home_team == e.team
           else 105.0 - e.location_x
           for e in res.timeline
           if e.event_type == _ET.BALL_RECEIPT and e.player == nm]
    org = [e.location_x if res.config.home_team == e.team
           else 105.0 - e.location_x
           for e in res.timeline
           if e.event_type == _ET.PASS and e.player == nm and e.end_x is not None]
    if len(rec) < 5 or len(org) < 5:
        print(f"{nm:<20}  -- NO VERDICT (need >=5 of each) --")
        continue
    pos = POSNAME.get(nm, "?")
    print(f"{nm:<20}{pos:<6}{len(rec):>7}{pct(rec,.5):>9.1f}"
          f"{sum(1 for v in rec if v >= 70)/len(rec)*100:>9.0f}%"
          f"{pct(org,.5):>9.1f}{pct(rec,.5)-pct(org,.5):>7.1f}")
print()
print("   gap ~ 0  => he receives where he plays from: POSITION is fine and the")
print("                 problem is that the attack never picks him upfield.")
print("   gap >> 0 => he is being found deeper than he stands: SELECTION bug.")
print("   gap <  0 => he plays from deeper than he is found: carry/pass drift.")
