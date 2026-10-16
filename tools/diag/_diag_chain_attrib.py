"""TERM-BY-TERM ATTRIBUTION OF tx IN THE LIVE SHAPE CHAIN (2026-10-14)

WHERE THIS IS COMING FROM. The depth-anchor rule landed and moved the
fullbacks 67.4 -> 51.6 m. Real but partial: it closed about a third of the gap
to the real 22-30. Three levers could close the rest and they are NOT
interchangeable:

  (a) raise DEPTH_ANCHOR_FAR_W               -- pure gain tuning
  (b) make advance_run_target ball-relative  -- removes a ratchet
  (c) change what `ax` MEANS                  -- the actual root cause

Picking between them from arithmetic alone is a guess. So: measure.

INSTRUMENT. `_offball_move_player` builds `tx` through a fixed sequence:

    anchor compaction -> OOP danger -> cross crash -> RUN BLEND
      -> depth_home_anchor -> CK35(y only) -> CK36 -> CK37 -> CK38
      -> live_spacing -> [setpiece | CB ceiling | rest defence]

Steps 1-6 are PositionEngine methods (wrappable) or two lines of arithmetic
(reconstructible from the caller's frame locals). `_striker_runs` is the call
site that sits BETWEEN the compaction and the run blend, so wrapping it splits
the two apart -- which is the whole question, because a previous version
reported them as one lumped term.

ALIGNMENT, and why this file is built around a per-tick dict. The first version
appended each stage to its own list and aligned by index with
`n = min(len(...))`. Two stages fire only conditionally -- `cb_advance_clamp` is
CB-only, `rest_defence_clamp` only when the invariant is actually violated --
so `min` collapsed every alignment onto the SHORTEST list and the stage table
reported 68 rows against a 29,843-row sample, with plausible-looking numbers.
Index alignment across conditionally-appended lists is the same bug class as
index-aligned pairing, which has already produced one false finding here
("never both flanks occupied at the same tick"). Every figure below is
therefore computed WITHIN a tick, never across lists.

NON-VACUITY GATES, asserted before any verdict prints:
  * every stage must have >= 1000 aligned ticks for a player to be reported;
  * a stage reported as `--` means "did not fire", never "fired and did
    nothing" -- the two are different findings.

One match. Run:  python _diag_chain_attrib.py [--seed N]
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=777)
A = ap.parse_args()

random.seed(A.seed)
try:
    import numpy as np
    np.random.seed(A.seed)
except Exception:  # pragma: no cover
    pass

import position_engine as PEMOD                    # noqa: E402
import match_engine as MEMOD                       # noqa: E402
import run_match as RM                             # noqa: E402
from player_dna import SquadBuilder                # noqa: E402
from match_engine import (MatchConfig, MatchEngine, TeamProfile, TeamStyle,  # noqa: E402
                          PlayingStyle, Intensity)

HOME, AWAY = RM.HOME_TEAM, RM.AWAY_TEAM
WANT = tuple(p[0] for p in RM.HOME_STARTERS) + tuple(p[0] for p in RM.AWAY_STARTERS)
SCOPED = ("CB", "LB", "RB")

# name -> tick_id -> {stage: nx}
TICK = defaultdict(dict)
TICKID = defaultdict(int)
POSOF = {}

_pe = PEMOD.PositionEngine
_me = MEMOD.MatchEngine
RAW = {n: getattr(_pe, n) for n in
       ("depth_home_anchor_blend", "wide_stretch_blend",
        "midfielder_triangle_support", "backline_build_up_support",
        "backline_spread_pressure", "live_spacing_redirect",
        "cb_advance_clamp", "rest_defence_clamp")}
RAW_RUNS = _me._striker_runs

COMPACTION_W = 0.14   # match_engine.py:2809


def _nx_of(team, x):
    return float(x) if team == HOME else 105.0 - float(x)


def _who(f):
    nm = f.get("pname")
    if nm not in WANT:
        return None, None
    team = f.get("team")
    if not team:
        return None, None
    pos = POSOF.setdefault(nm, f.get("pos"))
    if pos not in SCOPED:
        return None, None
    return nm, team


def _open_tick(f):
    """First stage of the chain for this tick: allocate the tick id and
    reconstruct the two values that only exist as arithmetic."""
    nm, team = _who(f)
    if nm is None:
        return None, None, None
    TICKID[nm] += 1
    rec = TICK[nm].setdefault(TICKID[nm], {})
    # `_striker_runs` is called AFTER the anchor compaction / OOP danger /
    # cross crash and BEFORE the run blend, so `tx` here is the pre-blend
    # value and `ax` is the anchor it was compacted FROM.
    if "tx" in f and "ax" in f and f.get("ball_x") is not None:
        rec["s0_pre_run"] = _nx_of(team, f["tx"])
        rec["anchor"] = _nx_of(team, f["ax"])
        rec["tx0"] = _nx_of(team, f["ax"] + (float(f["ball_x"]) - f["ax"])
                            * COMPACTION_W)
    return nm, team, rec


def _snap(step):
    def deco(fn):
        def wrapper(*a, **kw):
            f = sys._getframe(1).f_locals
            nm, team, rec = _open_tick(f)
            if rec is not None and "tx" in f:
                rec[step] = _nx_of(team, f["tx"])
            return fn(*a, **kw)
        return wrapper
    return deco


def _patched_runs(self, team, ball_x, ball_y, has_ball):
    f = sys._getframe(1).f_locals
    nm, team_f, rec = _open_tick(f)
    return RAW_RUNS(self, team, ball_x, ball_y, has_ball)


def _patched_lsr(self, team_name, cx, cy, tx, ty):
    f = sys._getframe(1).f_locals
    nm, team, rec = _open_tick(f)
    out = RAW["live_spacing_redirect"](self, team_name, cx, cy, tx, ty)
    if rec is not None:
        rec["s6_in"] = _nx_of(team, tx)
        rec["s6_out"] = out[0] if team == HOME else 105.0 - out[0]
    return out


def _patched_cb(self, tx, attacks_right):
    f = sys._getframe(1).f_locals
    nm, team, rec = _open_tick(f)
    out = RAW["cb_advance_clamp"](self, tx, attacks_right)
    if rec is not None:
        rec["s7_in"] = _nx_of(team, tx)
        rec["s7_out"] = _nx_of(team, out)
    return out


def _patched_rest(self, tx, ty, ball_x, attacks_right):
    f = sys._getframe(1).f_locals
    nm, team, rec = _open_tick(f)
    out = RAW["rest_defence_clamp"](self, tx, ty, ball_x, attacks_right)
    if rec is not None:
        rec["s8_in"] = _nx_of(team, f.get("tx", tx))
        rec["s8_out"] = _nx_of(team, out[0])
    return out


_me._striker_runs = _patched_runs
_pe.depth_home_anchor_blend = _snap("s1_post_run")(
    RAW["depth_home_anchor_blend"])
_pe.wide_stretch_blend = _snap("s2_post_anchor")(RAW["wide_stretch_blend"])
_pe.midfielder_triangle_support = _snap("s3_post_ck36")(
    RAW["midfielder_triangle_support"])
_pe.backline_build_up_support = _snap("s4_post_ck37")(
    RAW["backline_build_up_support"])
_pe.backline_spread_pressure = _snap("s5_post_ck38")(
    RAW["backline_spread_pressure"])
_pe.live_spacing_redirect = _patched_lsr
_pe.cb_advance_clamp = _patched_cb
_pe.rest_defence_clamp = _patched_rest

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
    TeamProfile(name=HOME, style=TeamStyle.BALANCED,
                playing_style=PlayingStyle.POSSESSION,
                intensity=Intensity.MEDIUM),
    TeamProfile(name=AWAY, style=TeamStyle.BALANCED,
                playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM))
eng.set_squad(HOME, hs["starters"], hs["substitutes"])
eng.set_squad(AWAY, aw["starters"], aw["substitutes"])
res = eng.simulate()

MIN_TICKS = 1000
NAMES = ("Darius Frost", "Rico Alves", "Jide Afolabi", "Emeka Obi")


def mean(v):
    return statistics.fmean(v) if v else float("nan")


def paired(nm, a, b):
    """Deltas WITHIN a tick: mean(rec[a] - rec[b]) over ticks holding both."""
    d = []
    for rec in TICK[nm].values():
        if a in rec and b in rec:
            d.append(rec[a] - rec[b])
    return d


print(f"\nseed={A.seed} events={len(res.timeline)} "
      f"score={res.state.home_goals}-{res.state.away_goals}")
print()

print("=" * 100)
print("1. MEAN SIGNED CONTRIBUTION OF EACH TERM TO tx, IN nx METRES")
print("   + = pushes the player UP the pitch. Scoped to CB / LB / RB.")
print("   `--` = the stage did not fire (a different finding from zero).")
print("-" * 100)
TERMS = [
    ("compaction: tx0 - anchor", "tx0", "anchor",
     "the 0.14 ball-compaction, measured against the anchor it pulls FROM"),
    ("danger squeeze + cross crash", "s0_pre_run", "tx0", None),
    ("RUN BLEND", "s1_post_run", "s0_pre_run",
     "the live run target -- the ratchet"),
    ("DEPTH ANCHOR (new)", "s2_post_anchor", "s1_post_run", None),
    ("CK36 triangle", "s3_post_ck36", "s2_post_anchor", None),
    ("CK37 build-up", "s4_post_ck37", "s3_post_ck36", None),
    ("CK38 spread", "s5_post_ck38", "s4_post_ck37", None),
    ("live spacing", "s6_out", "s6_in", None),
    ("CB ceiling", "s7_out", "s7_in", None),
    ("rest defence", "s8_out", "s8_in", None),
]
print(f"{'term':<32}" + "".join(f"{n.split()[-1][:9]:>11}" for n in NAMES))
for label, a, b, _ in TERMS:
    line = f"{label:<32}"
    for nm in NAMES:
        d = paired(nm, a, b)
        if len(d) < MIN_TICKS:
            line += f"{'--':>11}"
        else:
            line += f"{mean(d):>+11.2f}"
    print(line)
print()
print("   ticks per player:", {n: len(TICK[n]) for n in NAMES})
print()

print("=" * 100)
print("2. WHERE tx IS AT EACH STAGE (mean nx). The gap tells you what is left")
print("   to be won, and WHERE it is lost.")
print("-" * 100)
STAGES = [("anchor", "the anchor itself"),
          ("tx0", "+ compaction"),
          ("s0_pre_run", "+ danger/cross"),
          ("s1_post_run", "+ RUN BLEND"),
          ("s2_post_anchor", "+ DEPTH ANCHOR"),
          ("s3_post_ck36", "+ CK36"),
          ("s4_post_ck37", "+ CK37"),
          ("s5_post_ck38", "+ CK38"),
          ("s6_out", "+ spacing"),
          ("s7_out", "+ CB ceiling"),
          ("s8_out", "+ rest defence")]
for nm in NAMES:
    recs = TICK[nm]
    if len(recs) < MIN_TICKS:
        print(f"   {nm}: NO VERDICT, {len(recs)} ticks")
        continue
    print(f"   {nm}")
    for st, label in STAGES:
        v = [r[st] for r in recs.values() if st in r]
        tag = "" if len(v) >= MIN_TICKS else f"   (only {len(v)} ticks)"
        print(f"      {st:<16} {label:<18} mean nx {mean(v):>7.2f}{tag}")
print()

print("=" * 100)
print("3. THE DECISION THIS IS FOR")
print("   LB/RB post is ~22-26 and the real in-possession band is 30-45.")
print("   `anchor` is what the chain compacts FROM; `s2_post_anchor` is the")
print("   target our rule leaves behind. If `anchor` is already far up the")
print("   pitch, the ratchet is upstream of every rule in the chain and no")
print("   gain tuning here can fix it.")
print("-" * 100)
for nm in NAMES:
    recs = TICK[nm]
    if len(recs) < MIN_TICKS:
        continue
    a = [r["anchor"] for r in recs.values() if "anchor" in r]
    s2 = [r["s2_post_anchor"] for r in recs.values() if "s2_post_anchor" in r]
    s1 = [r["s1_post_run"] for r in recs.values() if "s1_post_run" in r]
    print(f"   {nm:<18} anchor {mean(a):>6.2f}   ->  after run blend "
          f"{mean(s1):>6.2f}   ->  after anchor rule {mean(s2):>6.2f}")
