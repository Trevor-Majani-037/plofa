"""CB DEPTH v2 — does a deep CB WALK there or is he TELEPORTED there?

v1 (`_diag_cb_ceiling.py`) established:
  * `cb_advance_clamp` EXISTS, is live-called, and binds in possession
    (Massey: 14,977 calls, 4,400 real clamps) -- this is NOT an unwired rule.
  * CBs are skipped off-ball 37% of ticks because they are in
    `_on_ball_this_minute` (the early return at match_engine.py:2755 sits
    ABOVE the clamp at :2955).
  * CBs still reach nx up to 105.6 m.
  * The deep positions are WRITTEN by `record_touch` snap sites
    (match_engine.py:5835/5843 = `_absorb_chain`; event_chain.py:2922/1376).

What v1 CANNOT say is the thing that decides the fix:

    When a CB is beyond 58 m, did the SHAPE integrator put him there
    (walk, so the clamp is failing) or is he simply on-ball and snapped to
    the event coordinate (teleport, so the clamp is irrelevant because it
    only ever steers an off-ball target)?

v2 answers it per tick, split by possession state:

  A. off-ball ticks at nx>58, split in-possession / out-of-possession
  B. ON-BALL (skipped) ticks at nx>58  -> teleport, clamp cannot reach
  C. the deep target the clamp actually sees (pre-clamp nx distribution)
  D. the ball's own depth when a CB is deep (is he deep because the BALL is?)
  E. first-exceedance approach: the 25 ticks before a CB first passes 58

Run:
    .venv\\Scripts\\python.exe tools\\diag\\_diag_cb_ceiling2.py
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _diag_chance_coords import build_pair  # noqa: E402
from position_engine import PositionEngine, PITCH_X  # noqa: E402
from match_engine import MatchEngine  # noqa: E402

CEIL = PositionEngine.CB_MAX_ADVANCE_NX

CB_TEAM: dict = {}
CB_AR: dict = {}

SKIP_DEEP: Counter = Counter()          # teleported: on-ball and nx>58
OFF_IN: Counter = Counter()             # walked: off-ball, in possession, nx>58
OFF_OUT: Counter = Counter()            # walked: off-ball, out of possession, nx>58
TICKS: Counter = Counter()
TICKS_IN: Counter = Counter()
DEEP_BALL_NX = {n: [] for n in ()}      # placeholder, filled later
CLAMP_PRE: dict = {}                    # cb -> list of pre-clamp nx (>58 only)
CLAMP_CALLS: Counter = Counter()
CLAMP_AT: Counter = Counter()
WALK_DEEP_IN: Counter = Counter()       # off-ball in-possession tick that MOVED past 58
FIRST_SEEN: dict = {}
APPROACH: dict = {}


def _strip(pos) -> str:
    return str(pos or "").rstrip("0123456789")


def _nx(x, ar):
    return x if ar else (PITCH_X - x)


_orig_clamp = PositionEngine.cb_advance_clamp


def _clamp_wrap(self, tx, attacks_right):
    out = _orig_clamp(self, tx, attacks_right)
    try:
        pname = sys._getframe(1).f_locals.get("pname")
    except Exception:  # noqa: BLE001
        pname = None
    if pname in CB_AR:
        CLAMP_CALLS[pname] += 1
        pre_nx = _nx(tx, attacks_right)
        if pre_nx > CEIL:
            CLAMP_PRE.setdefault(pname, []).append(round(pre_nx, 1))
    return out


PositionEngine.cb_advance_clamp = _clamp_wrap

_orig_ob = MatchEngine._offball_move_player


def _ob_wrap(self, pname, team, ball_x, ball_y, has_ball, danger_t,
             cross_team, DT):
    if pname in CB_AR:
        ar = CB_AR[pname]
        st = self.position_engine.states.get(pname)
        if st is not None:
            nx = _nx(st.current_x, ar)
            ball_nx = _nx(ball_x, ar)
            TICKS[pname] += 1
            if has_ball:
                TICKS_IN[pname] += 1
            if pname in self._on_ball_this_minute:
                if nx > CEIL:
                    SKIP_DEEP[pname] += 1
            else:
                if nx > CEIL:
                    if has_ball:
                        OFF_IN[pname] += 1
                    else:
                        OFF_OUT[pname] += 1
            if nx > CEIL:
                DEEP_BALL_NX.setdefault(pname, []).append(round(ball_nx))
            if nx > CEIL and pname not in FIRST_SEEN:
                FIRST_SEEN[pname] = (int(self.state.minute), round(nx, 1),
                                     has_ball)
    return _orig_ob(self, pname, team, ball_x, ball_y, has_ball, danger_t,
                    cross_team, DT)


MatchEngine._offball_move_player = _ob_wrap


def main():
    lines = []

    def p(m=""):
        print(m)
        lines.append(m)

    eng0 = build_pair("Pearls", "Justice")
    for team in (eng0.config.home_team, eng0.config.away_team):
        ar = eng0.position_engine.team_attacks_right.get(team, True)
        for pl in eng0.squads[team]["starters"]:
            if _strip(getattr(pl, "position", "")) == "CB":
                CB_TEAM[pl.name] = team
                CB_AR[pl.name] = ar

    p(f"CB_MAX_ADVANCE_NX = {CEIL}   (pitch {PITCH_X}, halfway "
      f"{PITCH_X/2})")
    p("CBs: " + ", ".join(f"{n} ({CB_TEAM[n]})" for n in sorted(CB_TEAM)))
    p()

    random.seed(4000)
    res = build_pair("Pearls", "Justice").simulate()
    st = getattr(res, "state", res)

    p(f"match Pearls {st.home_goals}-{st.away_goals} Justice, "
      f"{len(res.timeline)} events")
    p()
    p("WALK vs TELEPORT -- ticks with CB depth nx > 58, by possession state")
    p(f"   {'cb':<20} {'ticks':>7} {'in-poss':>8} "
      f"{'TELEPORT(on-ball)':>18} {'WALK in-poss':>13} {'WALK out-poss':>14}")
    for n in sorted(CB_AR):
        p(f"   {n:<20} {TICKS[n]:>7} {TICKS_IN[n]:>8} "
          f"{SKIP_DEEP[n]:>18} {OFF_IN[n]:>13} {OFF_OUT[n]:>14}")
    p()

    p("WHAT THE CLAMP ACTUALLY SEES (in-possession off-ball targets)")
    p(f"   {'cb':<20} {'clamp calls':>11} {'targets > 58':>13} "
      f"{'max pre-clamp nx':>17}")
    for n in sorted(CB_AR):
        pre = CLAMP_PRE.get(n) or []
        mx = max(pre) if pre else -1.0
        p(f"   {n:<20} {CLAMP_CALLS[n]:>11} {len(pre):>13} {mx:>17.1f}")
    p()

    p("BALL DEPTH WHEN THE CB IS DEEP (nx > 58) -- is he deep because the BALL is?")
    p(f"   {'cb':<20} {'n':>6} {'median ball nx':>15} {'p10':>6} {'p90':>6}")
    for n in sorted(CB_AR):
        vals = sorted(DEEP_BALL_NX.get(n) or [])
        if not vals:
            p(f"   {n:<20} {0:>6}   (never deep)")
            continue
        k = len(vals)
        p(f"   {n:<20} {k:>6} {vals[k//2]:>15} {vals[int(k*0.1)]:>6} "
          f"{vals[int(k*0.9)]:>6}")
    p()

    p("FIRST EXCEEDANCE (first off-ball tick with nx > 58)")
    for n in sorted(CB_AR):
        fs = FIRST_SEEN.get(n)
        p(f"   {n:<20} {('m%d  nx %.1f  has_ball=%s' % fs) if fs else 'never'}")
    p()

    with open("_diag_cb_ceiling2.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_cb_ceiling2.txt")


if __name__ == "__main__":
    main()
