"""WHY DOES THE LEFT-BACK PLAY LIKE A WINGER, AND WHY IS THE CAM A BLOB? (2026-10-14)

Two complaints, both read off the player dashboards in
`plofa_output/Hartwell_City_vs_Thornfield_United_MD01`:

  1. "Left back plays super high almost like a LW (same for RB and RW)"
     -- Darius Frost, LB, Hartwell City.
  2. "Kwame Asante's pass third distribution really haunts me"
     -- CAM, Thornfield United.

MEASURED OFF THE EXPORT ALREADY (no match needed; see
`_diag_frost_asante.py`):
  Frost  31 passes: 3 DEF / 8 MID / 20 ATT, median origin nx 74.8, p90 95.2,
        39 carries of which 24 START in the attacking third, and 19 of his 31
        passes go BACKWARD. A left back whose median pass origin is 75 m up a
        105 m pitch is a winger with a different name.
  Asante 50 passes: 2 DEF / 41 MID / 7 ATT, median origin nx 60.6,
        **sd_y = 9.0 m** against sd_x 14.8 -- the origins are a narrow lateral
        BLOB, and 27 of 50 passes land within 6 m of lateral
        (fwd 15 / square 27 / back 7). That is the starburst on the
        dashboard: one spot, ball-like directions.

INSTRUMENT, and why this one.
`PositionEngine.live_spacing_redirect` is called at match_engine.py:2927
with the FINISHED target of the whole shape chain, so its arguments ARE the
answer for that player on that tick. `pname`, `pos`, `has_ball`, `ball_x`,
`ball_y`, `team` come from the caller's frame locals (`sys._getframe(1)`) --
the `_diag_touch_sites.py` identity join, not a coordinate match.

TRAPS THIS PROBE IS BUILT AGAINST (all from AGENTS.md, all previously paid
for at least once here):
  * Read state AFTER `simulate()` -> final-whistle state attributed to a
    mid-match moment. Everything is sampled INSIDE the loop.
  * `PositionEngine.get_position` returns (50.0, 34.0) for an untracked
    player -- a coordinate that reads as real. Uses `current_x/current_y`.
  * `_offball_tick_seq` is an int ATTRIBUTE, not a method. (Not used here.)
  * A descriptor mismatch: `live_spacing_redirect` is a plain instance
    method, so the wrapper takes `self`.

GATES BEFORE ANY VERDICT IS PRINTED:
  * non-vacuity: >= 300 samples per player, else NO VERDICT.

COST DISCIPLINE. A match is ~90 s here. The match runs ONCE and dumps its
samples to `--cache`; every later run uses `--report-only` and simulates
nothing. Changing a print format must never cost a match.

Run:  python _diag_wide_depth_live.py [--seed N] [--report-only]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--cache", default="_diag_wide_depth_cache.json")
ap.add_argument("--report-only", action="store_true",
                help="re-print from the cache; runs NO match")
A = ap.parse_args()

HOME = "Hartwell City"
AWAY = "Thornfield United"
# The whole XI of both sides, not just the two the user named. Sampling only the
# named pair would leave sections 6/7 -- the FRAME VALIDATION and the per-role
# shape -- with no keeper and no centre-backs to check against, and a table
# that cannot be checked is the exact failure mode AGENTS.md keeps recording.
ALLP = tuple(p[0] for p in __import__("run_match").HOME_STARTERS) + \
        tuple(p[0] for p in __import__("run_match").AWAY_STARTERS)
NAMED = ("Darius Frost", "Rico Alves", "Adri Vela", "Percy",
         "Kwame Asante", "Kofi Mensah")

SAMP: dict = defaultdict(list)
RUNS: dict = defaultdict(list)
RUNCALLS: Counter = Counter()
PROFILES: dict = {}
ROSTER: dict = {}
META: dict = {}


def simulate():
    random.seed(A.seed)
    try:
        import numpy as np
        np.random.seed(A.seed)
    except Exception:  # pragma: no cover
        pass

    import match_engine as ME                       # noqa: F401
    import position_engine as PEMOD
    import run_match as RM
    from player_dna import SquadBuilder
    from match_engine import (MatchConfig, MatchEngine, TeamProfile,
                              TeamStyle, PlayingStyle, Intensity)

    pe_cls = PEMOD.PositionEngine
    raw_lsr = pe_cls.live_spacing_redirect
    raw_runs = pe_cls.offball_run_targets

    def _patched_lsr(self, team_name, cx, cy, tx, ty):
        f = sys._getframe(1).f_locals
        pname = f.get("pname")
        if pname in ALLP:
            st = self.states.get(pname)
            bx = f.get("ball_x")
            by = f.get("ball_y")
            SAMP[pname].append({
                "cx": cx, "cy": cy, "tx": tx, "ty": ty,
                "nx": cx if team_name == HOME else 105.0 - cx,
                "tnx": tx if team_name == HOME else 105.0 - tx,
                "hnx": (st.home_x if team_name == HOME else 105.0 - st.home_x)
                       if st else None,
                "pos": f.get("pos"),
                "possession": bool(f.get("has_ball")),
                "ball_nx": (float(bx) if team_name == HOME
                            else 105.0 - float(bx)) if bx is not None else None,
                "ball_y": float(by) if by is not None else None,
            })
        return raw_lsr(self, team_name, cx, cy, tx, ty)

    def _patched_runs(self, team_name, ball_x, ball_y, has_ball,
                      attacks_right=True, opp_block=None, minute=None):
        out = raw_runs(self, team_name, ball_x, ball_y, has_ball,
                       attacks_right, opp_block, minute)
        RUNCALLS[team_name] += 1
        if has_ball:
            for name, entry in out.items():
                if name in ALLP:
                    blend, rx, ry, mode = entry
                    RUNS[name].append([mode,
                                       rx if team_name == HOME else 105.0 - rx,
                                       ry, blend])
        return out

    pe_cls.live_spacing_redirect = _patched_lsr
    pe_cls.offball_run_targets = _patched_runs

    hs = SquadBuilder.build(
        team_name=HOME, starters=RM.HOME_STARTERS, substitutes=RM.HOME_SUBS,
        team_superstars=RM.HOME_SUPERSTARS, set_piece_takers=RM.HOME_SP_TAKERS,
        soul_assignments=RM.SOUL_PLAYERS)
    aw = SquadBuilder.build(
        team_name=AWAY, starters=RM.AWAY_STARTERS, substitutes=RM.AWAY_SUBS,
        team_superstars=RM.AWAY_SUPERSTARS, set_piece_takers=RM.AWAY_SP_TAKERS,
        soul_assignments=RM.SOUL_PLAYERS)
    eng = MatchEngine(
        MatchConfig(home_team=HOME, away_team=AWAY, match_date=RM.MATCH_DATE,
                    matchday=RM.MATCHDAY, venue="X", stadium_capacity=45000),
        TeamProfile(name=HOME, style=TeamStyle.BALANCED,
                    playing_style=PlayingStyle.POSSESSION,
                    intensity=Intensity.MEDIUM),
        TeamProfile(name=AWAY, style=TeamStyle.BALANCED,
                    playing_style=PlayingStyle.MIXED,
                    intensity=Intensity.MEDIUM))
    eng.set_squad(HOME, hs["starters"], hs["substitutes"])
    eng.set_squad(AWAY, aw["starters"], aw["substitutes"])
    res = eng.simulate()
    pe = eng.position_engine

    for nm, p in pe.fullback_registry.profiles.items():
        PROFILES[nm] = dict(advance=p.advance_instinct, overlap=p.overlap_instinct,
                            underlap=p.underlap_instinct, tuck=p.tuck_instinct,
                            hold=p.hold_discipline, flank=p.flank_commitment,
                            recovery=p.recovery_urgency,
                            delivery=p.delivery_instinct)
    for _p in list(eng.squads[HOME]["starters"]) + list(eng.squads[AWAY]["starters"]):
        _d = getattr(_p, "dna", None)
        ROSTER[_p.name] = dict(position=_p.position,
                               archetype=str(getattr(_d, "archetype", "")),
                               specialties=list(getattr(_d, "specialties", []) or []))
    # getattr fallbacks on purpose: this line sits directly in front of the
    # cache write, so a typo in a field name would throw away a 90-second
    # match. Two of them already have.
    hg = getattr(res.state, "home_goals", "?")
    ag = getattr(res.state, "away_goals", "?")
    META.update(seed=A.seed, events=len(res.timeline), score=f"{hg}-{ag}")

    with open(A.cache, "w", encoding="utf-8") as fh:
        json.dump({"meta": META, "samp": SAMP, "runs": RUNS,
                   "runcalls": dict(RUNCALLS), "profiles": PROFILES,
                   "roster": ROSTER}, fh)


def load_cache():
    with open(A.cache, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    META.update(d["meta"])
    for k, v in d["samp"].items():
        SAMP[k] = v
    for k, v in d["runs"].items():
        RUNS[k] = v
    RUNCALLS.update(d["runcalls"])
    PROFILES.update(d["profiles"])
    ROSTER.update(d["roster"])


if A.report_only:
    load_cache()
else:
    simulate()


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def corr(a, b):
    n = min(len(a), len(b))
    if n < 10:
        return float("nan")
    a, b = list(a[:n]), list(b[:n])
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((v - ma) ** 2 for v in a)
    vb = sum((v - mb) ** 2 for v in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


def sd(vals):
    vals = [v for v in vals if v is not None]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


print(f"seed={META.get('seed')}  events={META.get('events')}  "
      f"final={META.get('score')}")
print()

print("=" * 104)
print("0. GATE -- sample counts. NO VERDICT below without them.")
print("-" * 104)
for n in ALLP:
    print(f"   {n:<16} samples={len(SAMP[n]):>7}  run_entries={len(RUNS[n]):>6}")
print(f"   offball_run_targets calls: {dict(RUNCALLS)}")
print()

print("=" * 104)
print("1. FullbackSpatialProfile -- what the engine decided at kickoff")
print("-" * 104)
print(f"{'player':<16}{'archetype':<20}{'advance':>8}{'overlap':>9}{'underlap':>10}"
      f"{'tuck':>7}{'hold':>7}{'flankC':>8}{'home_nx':>9}")
for nm, p in PROFILES.items():
    if nm not in ALLP:
        continue
    r = ROSTER.get(nm, {})
    hn = next((s["hnx"] for s in SAMP[nm] if s.get("hnx") is not None), None)
    hns = f"{hn:>9.1f}" if hn is not None else "        ?"
    print(f"{nm:<16}{r.get('archetype','?')[:19]:<20}{p['advance']:>8.2f}"
          f"{p['overlap']:>9.2f}{p['underlap']:>10.2f}{p['tuck']:>7.2f}"
          f"{p['hold']:>7.2f}{p['flank']:>8.2f}{hns}")
    if r.get("specialties"):
        print(f"{'':<16}specialties: {', '.join(r['specialties'])}")
print()
print("   should_advance prob = advance*0.70 + (1-hold)*0.20, x0.45 if the ball")
print("   is on the far flank. It is re-ROLLED on every call, so a high")
print("   advance_instinct is not a rare event -- it is the default.")
print()

print("=" * 104)
print("2. TRACKED DEPTH (nx: 0 own goal -> 105 opp goal)")
print("   REAL FOOTBALL:  LB/RB 22-30   LW/RW 62-78   CAM 62-72")
print("-" * 104)
print(f"{'player':<16}{'pos':<5}{'n_in':>6}{'p50_in':>8}{'p90_in':>8}{'>70_in':>8}"
      f"{'n_out':>7}{'p50_out':>9}{'p90_out':>9}{'p50_tgt':>9}{'corr(nx,ball)':>15}")
for nm in ALLP:
    rows = SAMP[nm]
    if len(rows) < 300:
        print(f"{nm:<16}  -- NO VERDICT, {len(rows)} samples --")
        continue
    pos = next((r["pos"] for r in rows if r["pos"]), "?")
    inn = [r for r in rows if r["possession"]]
    out = [r for r in rows if not r["possession"]]
    xi = [r["nx"] for r in inn]
    xo = [r["nx"] for r in out]
    xt = [r["tnx"] for r in rows]
    nb = [r for r in rows if r["ball_nx"] is not None]
    c = corr([r["nx"] for r in nb], [r["ball_nx"] for r in nb])
    hi = sum(1 for v in xi if v > 70) / max(1, len(xi)) * 100
    print(f"{nm:<16}{pos:<5}{len(xi):>6}{pct(xi,.5):>8.1f}{pct(xi,.9):>8.1f}"
          f"{hi:>7.0f}%{len(xo):>7}{pct(xo,.5):>9.1f}{pct(xo,.9):>9.1f}"
          f"{pct(xt,.5):>9.1f}{c:>15.2f}")
print()
print("   corr(nx, ball_nx) near +1.0 = the player is the ball's shadow.")
print("   >70_in = share of in-possession ticks past 70 m up the pitch.")
print()

print("=" * 104)
print("3. LIVE RUN TARGETS -- which run, how often, how high")
print("-" * 104)
for nm in ALLP:
    ent = RUNS[nm]
    if len(ent) < 20:
        print(f"{nm:<16}  -- {len(ent)} run entries, too few to characterise --")
        continue
    modes = Counter(e[0] for e in ent)
    print(f"{nm:<16} entries={len(ent):>6}  modes={dict(modes)}  "
          f"mean blend={statistics.mean([e[3] for e in ent]):.2f}")
    for label in ("overlap", "underlap", "tuck", "byline", "cut", "box"):
        v = [e[1] for e in ent if e[0] == label]
        if v:
            print(f"{'':<16}   {label:<9} n={len(v):>6}  target nx p50={pct(v,.5):>6.1f}"
                  f"  p90={pct(v,.9):>6.1f}  max={max(v):>6.1f}")
print()

print("=" * 104)
print("4. SATELLITE TEST -- how tightly is each player glued to the BALL?")
print("-" * 104)
print(f"{'player':<16}{'n':>7}{'sd_x':>8}{'sd_y':>8}{'corr(y,ball_y)':>16}"
      f"{'med|y-ball_y|':>16}")
for nm in ALLP:
    rows = [r for r in SAMP[nm] if r.get("ball_y") is not None]
    if len(rows) < 300:
        print(f"{nm:<16}  -- NO VERDICT --")
        continue
    ys = [r["cy"] for r in rows]
    by = [r["ball_y"] for r in rows]
    dy = [abs(a - b) for a, b in zip(ys, by)]
    print(f"{nm:<16}{len(rows):>7}{sd([r['cx'] for r in rows]):>8.1f}"
          f"{sd(ys):>8.1f}{corr(ys, by):>16.2f}{pct(dy,.5):>16.1f}")
print()
print("   A #10 is NOT a ball satellite. corr with the ball's y well under 0.8,")
print("   sitting 8-18 m off it on his own terms. A fullback whose corr(nx,")
print("   ball_nx) is ~1.0 is playing where the ball is, not where his job is.")
print()

print("=" * 104)
print("6. WHOLE-TEAM DEPTH -- FRAME VALIDATION. If nx is right, the keeper")
print("   sits at ~5 and the CBs at ~22. If the keeper reads ~100 the frame is")
print("   flipped and EVERY number above is void.")
print("-" * 104)
print(f"{'player':<18}{'pos':<5}{'n':>7}{'p05':>7}{'p50':>7}{'p95':>7}"
      f"{'p50_in':>8}{'p50_out':>9}{'home_nx':>9}")
_people = sorted(SAMP)
for nm in _people:
    rows = SAMP[nm]
    if len(rows) < 200:
        continue
    pos = next((r["pos"] for r in rows if r["pos"]), "?")
    xs = [r["nx"] for r in rows]
    xi = [r["nx"] for r in rows if r["possession"]] or [float("nan")]
    xo = [r["nx"] for r in rows if not r["possession"]] or [float("nan")]
    hn = next((r["hnx"] for r in rows if r.get("hnx") is not None), None)
    hns = f"{hn:>9.1f}" if hn is not None else "        ?"
    print(f"{nm:<18}{pos:<5}{len(rows):>7}{pct(xs,.05):>7.1f}{pct(xs,.5):>7.1f}"
          f"{pct(xs,.95):>7.1f}{pct(xi,.5):>8.1f}{pct(xo,.5):>9.1f}{hns}")
print()
print("   REAL: GK 4-8, CB 20-26, LB/RB 22-30 (in poss 30-45 on an overlap),")
print("   CDM 35-45, CM 45-55, CAM 58-70, LW/RW 62-78, ST 70-80.")
print()

print("=" * 104)
print("7. WHERE EACH ROLE SITS, AGGREGATED (the shape, not the individuals)")
print("-" * 104)
_by_pos = defaultdict(list)
_by_pos_in = defaultdict(list)
_by_pos_out = defaultdict(list)
for nm, rows in SAMP.items():
    for r in rows:
        p = r["pos"] or "?"
        _by_pos[p].append(r["nx"])
        (_by_pos_in if r["possession"] else _by_pos_out)[p].append(r["nx"])
print(f"{'pos':<6}{'n':>8}{'p50':>8}{'p25':>8}{'p75':>8}{'p50_in':>9}"
      f"{'p50_out':>9}{'REAL p50':>11}")
REAL = {"GK": "4-8", "CB": "20-26", "LB": "22-30", "RB": "22-30", "CDM": "35-45",
        "CM": "45-55", "CAM": "58-70", "LW": "62-78", "RW": "62-78", "ST": "70-80"}
for p in sorted(_by_pos):
    v = _by_pos[p]
    print(f"{p:<6}{len(v):>8}{pct(v,.5):>8.1f}{pct(v,.25):>8.1f}{pct(v,.75):>8.1f}"
          f"{pct(_by_pos_in[p],.5):>9.1f}{pct(_by_pos_out[p],.5):>9.1f}"
          f"{REAL.get(p,'?'):>11}")
print()

print("=" * 104)
print("8. THE BLEND WEIGHTS -- how hard each run layer pulls")
print("-" * 104)
import position_engine as _PEM  # noqa: E402
print(f"   OFFBALL_RUN_BLEND_WIDE (LW/RW/LB/RB) = {_PEM.PositionEngine.OFFBALL_RUN_BLEND_WIDE}")
print(f"   OFFBALL_RUN_BLEND_MID  (CM)           = {_PEM.PositionEngine.OFFBALL_RUN_BLEND_MID}")
print("   CAM roam weight        (position_engine 3329) = 0.40 + awareness_bonus")
print("   -> a CAM is steered at 0.51 while a fullback is steered at 0.30, and")
print("      every one of those numbers is a fraction of the way to a target")
print("      computed FROM the live ball position.")
print()
print("=" * 104)
print("9. WHERE THE DEPTH RULE FOR A FULLBACK EXISTS -- LIVE vs DEAD")
print("-" * 104)
print("   LIVE chain (match_engine._offball_move_player, 2809-2965):")
print("     anchor compaction     tx = ax + (ball_x - ax)*0.14     14% only")
print("     OOP danger squeeze    pulls toward OWN goal            x only, OOP")
print("     cross in flight       tx += (cross_x - tx)*0.20        PUSHES UP")
print("     run-target blend      tx += (run_x  - tx)*0.30         PUSHES UP")
print("     CK35 pitch stretch    ty ONLY -- no x term at all")
print("     CK36/37/38            midfielder socket / ball-side CB")
print("     CB ceiling            CB only")
print("     rest defence          ONE named player, only when violated")
print("   DEAD chain (position_engine.drift_minute 1592-1658; MatchEngine")
print("   never calls drift_minute):")
print("     flank hold            y only")
print("     forward x-anchor      tx toward home_x when current_x < 70")
print("     byline retreat        LW/RW only -- no LB/RB equivalent")
print()
print("   => an in-possession fullback advanced to x=90 has NO live rule that")
print("      walks him back. The only x rule that does that is in drift_minute,")
print("      which no match reaches.")
