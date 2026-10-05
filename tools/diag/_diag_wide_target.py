"""Is the wide player's TARGET narrow, or does he lag a wide target? (2026-10-04)

`_diag_wide_channel.py` established the fact: side-to-side width is ~38 m
median against a real-PL band of 45-55 m, in BOTH possession phases, and only
18-27% of samples land inside that band. That is the symptom. This probe asks
the one question that decides the fix.

Two candidate mechanisms, with opposite fixes:

  (A) THE TARGET IS NARROW.  Something in the target chain
      (`_offball_move_player`: block anchor substitution -> 7% ball compaction
      -> live run targets -> CK35 stretch -> CK36 triangle -> CK37/38 ->
      live-spacing guard) is steering the winger's target y INSIDE, so the
      integrator faithfully walks to a bad socket. Fixing this means changing
      which term wins.

  (B) THE TARGET IS WIDE AND THE PLAYER LAGS IT.  The anchor chain produces a
      correct wide target, but the jog integrator never arrives — `_JOG_SPEED`
      is a shape-holding speed and the `involved` gate multiplies it by 0.30
      when the ball is on the far side. Fixing this means changing approach
      speed, NOT the shape.

They look identical in every aggregate the first probe printed, and they are
the difference between editing `position_engine.py`'s anchor logic and editing
`match_engine.py`'s integrator.

INSTRUMENT. `PositionEngine.live_spacing_redirect(team, cx, cy, tx, ty)` is
called at match_engine.py:2917 with the FINISHED target of the whole chain, so
its arguments are the answer. It does not receive the player name, so the
wrapper reads `pname`, `team`, `has_ball` and `_in_block` out of the CALLER's
frame locals (`sys._getframe(1)`) -- the same identity join
`_diag_touch_sites.py` uses. `live_spacing_redirect` is a plain instance
method (not a static/classmethod), so wrapping it on the class needs no
descriptor handling; `_JOG_SPEED` is read straight off the class so the speed
question is answered from the same run rather than from the source.

Run:  python _diag_wide_target.py [--seed N]
"""
import argparse
import sys
from collections import defaultdict

sys.path.insert(0, ".")

WIDE = ("LW", "RW", "LB", "RB")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--arm", default="both", choices=("none", "patched", "both"),
                help="'none' runs NO patch at all -- the negative control. "
                     "An in-process A/B is valid (the 'brain caches' objection "
                     "was retracted 2026-10-04 and falsified by "
                     "_diag_seed_repro.py arm none), so both arms can run here "
                     "for the price of one process.")
A = ap.parse_args()

import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(A.seed)
np.random.seed(A.seed)

import match_engine as ME  # noqa: E402
from position_engine import PositionEngine  # noqa: E402
from _diag_chance_coords import build_pair  # noqa: E402

# name -> list of (role, ty_target, cy_now, has_ball, in_block, line_dist_now)
TGT = defaultdict(list)

_raw = PositionEngine.live_spacing_redirect


def _patched(self, team, cx, cy, tx, ty):
    out = _raw(self, team, cx, cy, tx, ty)
    f = sys._getframe(1).f_locals
    pname = f.get("pname")
    pos = f.get("pos")
    # Gate on the ROLE from the caller frame, not on `pname in TGT`: membership
    # testing a defaultdict does not create the key, so the first run of this
    # probe recorded ZERO samples for every player and then divided by an empty
    # list. A member check against the collection you are about to append to is
    # not a filter, it is a coincidence.
    if pname and pos in WIDE:
        TGT[pname].append((
            pos,
            float(out[1]),                 # the chain's final target y
            float(cy),                     # where he is now
            bool(f.get("has_ball")),
            bool(f.get("_in_block")),
            min(float(cy), 68.0 - float(cy)),
            # `match_clock_s` cannot key this -- both tick loops advance it
            # outside -- so the engine's own tick identity is used. Pairing the
            # two flanks by LIST INDEX would be the documented mispairing trap:
            # one skipped player shifts every later index, and a player IS
            # skipped whenever `moved <= 0` short-circuits above the target
            # chain. Same-instant has to mean same-instant.
            _tickid(f),
        ))
    return out


_TICKFN = [None]


def _tickid(f):
    """Same-instant pairing key for the live shape integrator.

    `_offball_tick_seq` is a plain int ATTRIBUTE (match_engine.py:1990), NOT a
    method. This function used to CALL it, so `int(fn())` raised TypeError, the
    `except Exception` swallowed it, and every wide player in the match was
    filed under tick -1 -- i.e. ONE group that each call overwrote.

    Consequence: the per-player medians in `_diag_wide_target.txt` are FINE
    (they key on player name, not on tick), but every SIDE-TO-SIDE SEPARATION
    number this probe reports is computed from a single overwriting dict, so
    those figures are not measurements of a distribution. That is the likeliest
    source of the "dense and sparse instruments disagree by 2-3 m" note in
    AGENTS.md's WIDTH section; that disagreement should not be read as real
    variance until this is fixed and re-run.

    A swallowed exception that silently degrades an instrument is worse than a
    crash, and the guard below is deliberately NOT a bare `except`.
    """
    eng = f.get("self")
    seq = getattr(eng, "_offball_tick_seq", None)
    if isinstance(seq, int):
        return seq
    if seq is None:
        return -1
    raise TypeError(
        f"_offball_tick_seq is {type(seq).__name__}, expected int. "
        "The tick key must be an ATTRIBUTE read, not a call -- re-check "
        "match_engine.py:1990 before trusting any separation figure."
    )


# ── BISECT THE CHAIN ────────────────────────────────────────────────────
# For wide players the target chain has only three reachable terms:
#   ty = ay + (ball_y - ay) * 0.07        ball compaction, ~2 m when central
#   live run targets (match_engine.py:2835-2843)  a PREFERENCE, placed
#                                                   BEFORE CK35 on purpose
#   CK35 stretch (match_engine.py:2859-2862)       zero inside a block
# CK36/37/38 are midfielder and backline terms keyed by player name, so they
# cannot touch LW/RW/LB/RB -- which leaves these two as the only suspects.
# Both are hooked here so the bisect happens INSIDE the same live loop as the
# target reading above; a separate run could not pair them tick-by-tick.
#
# TWO dicts, not one: both hooks fire once per wide-player tick and are zipped
# afterwards. Appending both shapes to one list would interleave them and every
# pairing would be off by one -- and an off-by-one pairing of two steering
# terms produces a confident, entirely fictional attribution.
RUNS_P = defaultdict(list)    # name -> [run_blend, run_ty] | None
STR_P = defaultdict(list)     # name -> [stretch_w, in_block, has_ball, ball_y]

_stretch_raw = PositionEngine.wide_stretch_blend
_runs_raw = ME.MatchEngine._striker_runs


def _stretch_patched(self, player_name, ball_y):
    w = _stretch_raw(self, player_name, ball_y)
    f = sys._getframe(1).f_locals
    pname = f.get("pname")
    if pname and f.get("pos") in WIDE:
        STR_P[pname].append(
            [float(w), bool(f.get("_in_block", False)),
             bool(f.get("has_ball", False)), float(ball_y)])
    return w


def _runs_patched(self, team, ball_x, ball_y, has_ball):
    runs = _runs_raw(self, team, ball_x, ball_y, has_ball)
    f = sys._getframe(1).f_locals
    pname = f.get("pname")
    if pname and f.get("pos") in WIDE:
        r = runs.get(pname)
        RUNS_P[pname].append(
            None if r is None else [float(r[0]), float(r[2])])
    return runs


def run_match(tag, patch):
    """One match at the current seed. `patch=False` is the NEGATIVE CONTROL.

    Every earlier seed-777 run reported 2656 events (digest 79857b22c6e2). The
    first patched run of THIS probe reported 2644. Either the patch is not
    behaviourally inert -- which would invalidate any A/B built on it -- or the
    control is wrong. One match answers it, and the AGENTS.md rule is that a
    number without its control is not a measurement.
    """
    if patch:
        PositionEngine.live_spacing_redirect = _patched
        PositionEngine.wide_stretch_blend = _stretch_patched
        ME.MatchEngine._striker_runs = _runs_patched
    random.seed(A.seed)
    np.random.seed(A.seed)
    eng = build_pair("Oxton", "Natrican")
    out = eng.simulate()
    print(f"[arm {tag:<7}] events={len(out.timeline)}")
    return eng, out


if A.arm in ("none", "both"):
    run_match("none", patch=False)
eng, res = run_match("patched", patch=True)
pe = eng.position_engine

print()
print(f"seed={A.seed}  analysed arm=patched  events={len(res.timeline)}")
print()


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))]


def med(v):
    return pct(v, 50)


def mean(v):
    return (sum(v) / len(v)) if v else 0.0


print("=" * 112)
print("THE VERDICT: is the target narrow, or does the player lag it?")
print("-" * 112)
print(f"{'player':<20} {'pos':<4} {'n':>5} {'tgtLine':>8} {'nowLine':>8} "
      f"{'LAG':>6} {'tgtSep':>7} {'nowSep':>7}  verdict")
print("-" * 112)

pos_of = {}
for team in (res.config.home_team, res.config.away_team):
    for nm in pe.team_rosters.get(team, []):
        st = pe.states.get(nm)
        if st is not None:
            pos_of[nm] = st.position

# Pair the two flanks on the engine's TICK IDENTITY, never on list index --
# see the `_tickid` note in `_patched`. Index alignment is the documented
# mispairing trap: one skipped player shifts every later index, and a player IS
# skipped whenever `moved <= 0` short-circuits above the target chain.
LAG, TL, NL = [], [], []
by_tick = defaultdict(dict)
for nm, rows in TGT.items():
    for r in rows:
        by_tick[r[6]][nm] = r

tgt_sep, now_sep = [], []
for row in by_tick.values():
    for grp in (("LW", "RW"), ("LB", "RB")):
        m = {pos_of.get(nm): r for nm, r in row.items()
             if pos_of.get(nm) in grp}
        if len(m) == 2:
            tgt_sep.append(abs(m[grp[0]][1] - m[grp[1]][1]))
            now_sep.append(abs(m[grp[0]][2] - m[grp[1]][2]))

for nm in sorted(TGT, key=lambda n: (pos_of.get(n, ""), n)):
    rows = TGT[nm]
    tl = [min(r[1], 68.0 - r[1]) for r in rows]
    nl = [r[5] for r in rows]
    lag = [abs(r[2] - r[1]) for r in rows]
    LAG += lag
    TL += tl
    NL += nl
    print(f"{nm:<20} {pos_of.get(nm,'?'):<4} {len(rows):>5} {med(tl):>8.1f} "
          f"{med(nl):>8.1f} {med(lag):>6.1f} {'':>7} {'':>7}  "
          f"{'target is ' + ('WIDE' if med(tl) < 9 else 'INSIDE')}")

print("-" * 112)
print(f"{'ALL WIDE PLAYERS':<20} {'-':<4} {len(LAG):>5} {med(TL):>8.1f} "
      f"{med(NL):>8.1f} {med(LAG):>6.1f}")
print("-" * 112)
print()
print(f"side-to-side separation from the CHAIN TARGET : median "
      f"{med(tgt_sep):.1f} m  (p10 {pct(tgt_sep,10):.1f}, p90 "
      f"{pct(tgt_sep,90):.1f})")
print(f"side-to-side separation from the PLAYER       : median "
      f"{med(now_sep):.1f} m  (p10 {pct(now_sep,10):.1f}, p90 "
      f"{pct(now_sep,90):.1f})")
band = (100.0 * sum(1 for v in tgt_sep if 45.0 <= v <= 55.0) / len(tgt_sep)
        if tgt_sep else 0.0)
band2 = (100.0 * sum(1 for v in now_sep if 45.0 <= v <= 55.0) / len(now_sep)
         if now_sep else 0.0)
print(f"inside the real 45-55 m band: target {band:.1f}%   actual {band2:.1f}%")
print()
print("READ THIS BEFORE CHANGING ANYTHING:")
print("  If target separation is already ~50 m and actual is ~37 m, the shape")
print("  chain is RIGHT and the jog integrator is not arriving -> fix approach")
print("  speed, not the anchors.")
print("  If target separation is itself ~37 m, a term in the chain is steering")
print("  wide players inside -> fix which term wins, and the integrator is fine.")
print()

print("=" * 112)
print("BY PHASE — a low block legitimately narrows, so the two must be separate")
print("-" * 112)
for lbl, sel in (("in possession", lambda r: r[3]),
                 ("out, in block", lambda r: not r[3] and r[4]),
                 ("out, no block", lambda r: not r[3] and not r[4])):
    lag = [abs(r[2] - r[1]) for nm in TGT for r in TGT[nm] if sel(r)]
    tl = [min(r[1], 68.0 - r[1]) for nm in TGT for r in TGT[nm] if sel(r)]
    nl = [r[5] for nm in TGT for r in TGT[nm] if sel(r)]
    if not lag:
        print(f"{lbl:<18} (no samples)")
        continue
    print(f"{lbl:<18} n={len(lag):>6}  target-to-line {med(tl):>5.1f} m   "
          f"actual-to-line {med(nl):>5.1f} m   median lag {med(lag):>5.1f} m")
print("-" * 112)
print()

print("=" * 112)
print("THE APPROACH SPEED THAT EXPLAINS ANY LAG (read from the class, not a guess)")
print("-" * 112)
for pos in WIDE:
    j = ME.MatchEngine._JOG_SPEED.get(pos, 1.8)
    print(f"  {pos}: _JOG_SPEED {j:.2f} m/s   x0.30 when uninvolved = "
          f"{j * 0.30:.2f} m/s  -> closing a {med(LAG):.0f} m gap takes "
          f"{med(LAG) / max(j * 0.30, 1e-6):.0f} s")
print(f"  arrive deadband: 2.0 m (dist <= arrive -> hold)")
print("-" * 112)
print("A wide role's jog is a SHAPE-HOLDING speed, the same category of")
print("mistake as the low-block recovery branch documented at")
print("match_engine.py:3001-3018: right for holding a shape you are already")
print("in, useless for reaching one you are not. Measure before changing it.")
print("=" * 112)
print("CHAIN BISECT -- which of the two wide-player terms is pulling the")
print("target inside? (the integrator was exonerated above: lag ~1.2 m)")
print("-" * 112)
n_run, n_str = 0, 0
st_vals, str_vals = [], []
run_ty, run_bl = [], []
no_stretch_out, stretch_out = [], []
for nm, rows in TGT.items():
    rs = RUNS_P.get(nm, [])
    ss = STR_P.get(nm, [])
    k = min(len(rows), len(rs), len(ss))
    for i in range(k):
        w = ss[i][0]
        str_vals.append(w)
        if w > 0.0:
            n_str += 1
            if not ss[i][1]:
                stretch_out.append(min(ss[i][3], 68.0 - ss[i][3]))
        else:
            no_stretch_out.append(min(ss[i][3], 68.0 - ss[i][3]))
        if rs[i] is not None:
            n_run += 1
            run_bl.append(rs[i][0])
            run_ty.append(min(rs[i][1], 68.0 - rs[i][1]))

print(f"CK35 stretch active      : {n_str:>7} / {len(str_vals):>7} samples "
      f"({100.0*n_str/max(len(str_vals),1):.1f}%)")
print(f"CK35 stretch IN a block  : suppressed by design (match_engine.py:2859)")
print(f"run target present       : {n_run:>7} / {len(str_vals):>7} samples "
      f"({100.0*n_run/max(len(str_vals),1):.1f}%)")
if run_ty:
    print(f"run target to line       : median {med(run_ty):.1f} m   "
          f"p90 {pct(run_ty,90):.1f} m   blend median {med(run_bl):.2f}")
print("-" * 112)
print(f"ball y when stretch OFF  : median line-dist {med(no_stretch_out):.1f} m "
      f"(n={len(no_stretch_out)})")
if stretch_out:
    print(f"ball y when stretch ON   : median line-dist {med(stretch_out):.1f} m "
          f"(n={len(stretch_out)})")
print("-" * 112)
print("A run target that sits INSIDE the anchor is CORRECT football -- a winger")
print("cutting in is a real move. The question is how often it fires and with")
print("what blend. CK35 is what puts him back on the line afterwards, and it")
print("is dead whenever the team is defending.")
