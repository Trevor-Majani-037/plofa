"""CB DEPTH v3 — the CB ceiling, measured on BOTH symptom surfaces.

v1/v2 established the ceiling (`cb_advance_clamp`, CB_MAX_ADVANCE_NX=58) is
live and binds IN POSSESSION, but CBs still reach nx 97-105, because:
  * the on-ball path skips `_offball_move_player` entirely, and
  * the clamp is gated `pos == "CB" and has_ball` -> out of possession there
    is NO ceiling at all, so the off-ball anchor (which is the player's own
    minute-start position) ratchets past it.

The reported symptom is a PASS MAP (Juan Massey, CB: 39/58 passes in the
opponent half, 20 in the final third at x>=70). The position probe alone
cannot show whether the fix moves that. `event.location_x` for a pass is the
chain anchor (`_starting_anchor` = 0.75*last_ball + 0.25*style), and
`state.last_ball_x` is the previous event's end -- so a deep CB touch drags
the ball, and the ball drags the next chain. This probe reports both surfaces
so the fix is judged on the thing the user actually saw.

Run:
    .venv\\Scripts\\python.exe tools\\diag\\_diag_cb_ceiling3.py [label]
Writes `_diag_cb_ceiling3_<label>.txt` (default label `run`).
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
HALFWAY = PITCH_X / 2.0          # 52.5
FINAL_THIRD = 70.0

CB_TEAM: dict = {}
CB_AR: dict = {}

TICKS = Counter()
TICKS_DEEP = Counter()
TICKS_DEEP_IN = Counter()
TICKS_DEEP_OUT = Counter()
MAXNX = {}

PASS_EVENTS = {"PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
               "CROSS_ATTEMPT", "CROSS_SUCCESS", "FREEKICK_CROSS"}
CARRY_EVENTS = {"CARRY"}

PASSES = Counter()
PASS_OPPHALF = Counter()
PASS_FINALTHIRD = Counter()
CARRIES = Counter()
CARRY_OPPHALF = Counter()
CARRY_FINALTHIRD = Counter()
PASS_NX = {}
CARRY_NX = {}


def _strip(pos) -> str:
    return str(pos or "").rstrip("0123456789")


def _nx(x, ar):
    return x if ar else (PITCH_X - x)


_orig_ob = MatchEngine._offball_move_player


def _ob_wrap(self, pname, team, ball_x, ball_y, has_ball, danger_t,
             cross_team, DT):
    if pname in CB_AR:
        st = self.position_engine.states.get(pname)
        if st is not None:
            nx = _nx(st.current_x, CB_AR[pname])
            TICKS[pname] += 1
            if nx > CEIL:
                TICKS_DEEP[pname] += 1
                if has_ball:
                    TICKS_DEEP_IN[pname] += 1
                else:
                    TICKS_DEEP_OUT[pname] += 1
            if nx > MAXNX.get(pname, -1e9):
                MAXNX[pname] = nx
    return _orig_ob(self, pname, team, ball_x, ball_y, has_ball, danger_t,
                    cross_team, DT)


MatchEngine._offball_move_player = _ob_wrap


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
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

    random.seed(4000)
    res = build_pair("Pearls", "Justice").simulate()
    st = getattr(res, "state", res)

    # ── SURFACE 1: the pass / carry map ──────────────────────────────
    for ev in res.timeline:
        nm = getattr(ev, "player", "") or ""
        if nm not in CB_AR:
            continue
        etype = getattr(getattr(ev, "event_type", None), "name", "") or ""
        lx = getattr(ev, "location_x", None)
        if lx is None:
            continue
        nx = _nx(lx, CB_AR[nm])
        if etype in PASS_EVENTS:
            PASSES[nm] += 1
            PASS_NX[nm] = max(PASS_NX.get(nm, 0.0), nx)
            if nx > HALFWAY:
                PASS_OPPHALF[nm] += 1
            if nx >= FINAL_THIRD:
                PASS_FINALTHIRD[nm] += 1
        elif etype in CARRY_EVENTS:
            CARRIES[nm] += 1
            CARRY_NX[nm] = max(CARRY_NX.get(nm, 0.0), nx)
            if nx > HALFWAY:
                CARRY_OPPHALF[nm] += 1
            if nx >= FINAL_THIRD:
                CARRY_FINALTHIRD[nm] += 1

    p(f"CB ceiling = {CEIL} m   halfway = {HALFWAY}   final third = {FINAL_THIRD}")
    p(f"label = {label}")
    p("CBs: " + ", ".join(f"{n} ({CB_TEAM[n]})" for n in sorted(CB_AR)))
    p()
    p(f"match Pearls {st.home_goals}-{st.away_goals} Justice, {len(res.timeline)} events")
    p()
    p("SURFACE 1 -- CB PASS / CARRY MAP (event.location_x)")
    p(f"   {'cb':<18} {'pass':>5} {'opp-half':>9} {'fin-third':>10} "
      f"{'p-max':>6} {'carry':>7} {'carr opp':>9} {'carr fin':>9} {'c-max':>6}")
    for n in sorted(CB_AR):
        p(f"   {n:<18} {PASSES[n]:>5} {PASS_OPPHALF[n]:>9} "
          f"{PASS_FINALTHIRD[n]:>10} {PASS_NX.get(n, 0.0):>6.1f} "
          f"{CARRIES[n]:>7} {CARRY_OPPHALF[n]:>9} "
          f"{CARRY_FINALTHIRD[n]:>9} {CARRY_NX.get(n, 0.0):>6.1f}")
    tp = sum(PASSES.values())
    to = sum(PASS_OPPHALF.values())
    tf = sum(PASS_FINALTHIRD.values())
    tc = sum(CARRIES.values())
    tcf = sum(CARRY_FINALTHIRD.values())
    p(f"   {'TEAM CB TOTAL':<18} {tp:>5} {to:>9} {tf:>10} {'':>6} "
      f"{tc:>7} {sum(CARRY_OPPHALF.values()):>9} {tcf:>9}")
    p(f"   max pass nx over all CBs = {max(PASS_NX.values()) if PASS_NX else 0.0:.1f}"
      f"   max carry nx = {max(CARRY_NX.values()) if CARRY_NX else 0.0:.1f}")
    p()
    p("SURFACE 2 -- CB TRACKED POSITION DEPTH (off-ball ticks)")
    p(f"   {'cb':<18} {'ticks':>7} {'deep>58':>8} {'deep in-poss':>13} "
      f"{'deep out-poss':>14} {'max nx':>7}")
    for n in sorted(CB_AR):
        p(f"   {n:<18} {TICKS[n]:>7} {TICKS_DEEP[n]:>8} "
          f"{TICKS_DEEP_IN[n]:>13} {TICKS_DEEP_OUT[n]:>14} "
          f"{MAXNX.get(n, 0.0):>7.1f}")
    p()

    out = os.path.join(_HERE, f"_diag_cb_ceiling3_{label}.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
