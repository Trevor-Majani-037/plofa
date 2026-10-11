"""WHY DOES `cb_advance_clamp` NOT HOLD A CB AT x~86?  (2026-10-11)

The user's report: Juan Massey (CB, Pearls) spent MD06 at x=75-88 in the
opponent half, 20 passes into the final third, 2 shots, flagged offside twice
at x=86.2.  The proposed fix was "add a CB attacking ceiling".

But the ceiling ALREADY EXISTS and is live-called:

    PositionEngine.cb_advance_clamp   (position_engine.py:2501)
    PositionEngine.CB_MAX_ADVANCE_NX = 58.0
    match_engine.py:2955             tx = self.position_engine.cb_advance_clamp(tx, _attacks_right)
    ...gated `if pos == "CB" and has_ball:`

`pos` is plain "CB" (roster_loader.POS_MAP normalises every Excel CB to "CB"),
so the gate DOES match -- this is not the numbered-position bug class.

So the real question is not "add a ceiling" but "why doesn't the existing one
bind".  This probe answers it with attribution, not inference.  One real match,
real roster (Pearls v Justice), observation-only + wrapper counting:

  A. Wrap `cb_advance_clamp` -> how often it is CALLED, how often it actually
     changes tx, and for WHICH centre-back (read `pname` from the caller frame).
  B. Wrap `_offball_move_player` -> for each CB, how many 10 Hz ticks it runs
     for him, and how many of those it SKIPS because `pname in
     _on_ball_this_minute` (match_engine.py:2755 -- the early return that is
     ABOVE the clamp).
  C. Wrap `PositionEngine.record_touch` -> every CB position write, its source
     (file:lineno of the caller), the event type that produced it (the caller
     frame's `event` local, when present), and the CB's depth normalised to his
     own attacking direction.  Reports every write beyond the ceiling.

Run:
    .venv\\Scripts\\python.exe tools\\diag\\_diag_cb_ceiling.py
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _diag_chance_coords import build_pair  # noqa: E402
from position_engine import PositionEngine  # noqa: E402
from match_engine import MatchEngine  # noqa: E402

CEIL = PositionEngine.CB_MAX_ADVANCE_NX

STATS: Counter = Counter()
CLAMP_CALLS: Counter = Counter()      # cb name -> calls
CLAMP_ACTUAL: Counter = Counter()     # cb name -> calls that changed tx
SKIP_TICKS: Counter = Counter()       # cb name -> ticks where skipped (on-ball)
ALL_TICKS: Counter = Counter()        # cb name -> ticks
DEEP_SITES: Counter = Counter()       # "file:lineno" -> writes with nx > CEIL
DEEP_EVENTS: Counter = Counter()      # event type -> writes with nx > CEIL
MAX_NX: dict = {}                     # cb name -> max normalised x
DEEP_TRACE = defaultdict(list)        # cb name -> [(minute, nx, before_nx, site, ev)]
CB_TEAM: dict = {}                    # cb name -> team
CB_ATTACKS_RIGHT: dict = {}           # cb name -> bool


def _strip(pos) -> str:
    return str(pos or "").rstrip("0123456789")


# ── A. the clamp itself ────────────────────────────────────────────────
_orig_clamp = PositionEngine.cb_advance_clamp


def _clamp_wrap(self, tx, attacks_right):
    out = _orig_clamp(self, tx, attacks_right)
    try:
        pname = sys._getframe(1).f_locals.get("pname")
    except Exception:  # noqa: BLE001
        pname = None
    if pname in CB_TEAM:
        CLAMP_CALLS[pname] += 1
        if abs(out - tx) > 1e-9:
            CLAMP_ACTUAL[pname] += 1
    return out


PositionEngine.cb_advance_clamp = _clamp_wrap


# ── B. off-ball movement / the on-ball skip ────────────────────────────
_orig_ob = MatchEngine._offball_move_player


def _ob_wrap(self, pname, team, ball_x, ball_y, has_ball, danger_t,
             cross_team, DT):
    if pname in CB_TEAM:
        ALL_TICKS[pname] += 1
        if pname in self._on_ball_this_minute:
            SKIP_TICKS[pname] += 1
    return _orig_ob(self, pname, team, ball_x, ball_y, has_ball, danger_t,
                    cross_team, DT)


MatchEngine._offball_move_player = _ob_wrap


# ── C. every CB position write, attributed ─────────────────────────────
_orig_rt = PositionEngine.record_touch


def _rt_wrap(self, player_name, x, y, minute):
    if player_name in CB_TEAM and x is not None:
        st = self.states.get(player_name)
        before_x = getattr(st, "current_x", None)
        ar = CB_ATTACKS_RIGHT.get(player_name, True)
        nx = x if ar else (105.0 - x)
        before_nx = None if before_x is None else (
            before_x if ar else (105.0 - before_x))
        if nx > MAX_NX.get(player_name, -1.0):
            MAX_NX[player_name] = nx
        if nx > CEIL:
            fr = sys._getframe(1)
            site = f"{os.path.basename(fr.f_code.co_filename)}:{fr.f_lineno}"
            ev = "?"
            try:
                e = fr.f_locals.get("event")
                ev = getattr(getattr(e, "event_type", None), "name", "?")
            except Exception:  # noqa: BLE001
                pass
            DEEP_SITES[site] += 1
            DEEP_EVENTS[ev] += 1
            if len(DEEP_TRACE[player_name]) < 80:
                DEEP_TRACE[player_name].append(
                    (minute, round(nx, 1),
                     None if before_nx is None else round(before_nx, 1),
                     site, ev))
    return _orig_rt(self, player_name, x, y, minute)


PositionEngine.record_touch = _rt_wrap


def main():
    lines = []

    def p(m=""):
        print(m)
        lines.append(m)

    # Build first (no simulate) so we can learn the CB names, then patch the
    # name sets and run.  build_pair builds + returns an engine; we rebuild for
    # the real run so the name maps are populated before simulate().
    eng0 = build_pair("Pearls", "Justice")
    for team in (eng0.config.home_team, eng0.config.away_team):
        ar = eng0.position_engine.team_attacks_right.get(team, True)
        for pl in eng0.squads[team]["starters"]:
            if _strip(getattr(pl, "position", "")) == "CB":
                CB_TEAM[pl.name] = team
                CB_ATTACKS_RIGHT[pl.name] = ar

    p(f"CB_MAX_ADVANCE_NX = {CEIL} m   (own-attacking-frame depth ceiling)")
    p(f"CBs tracked: " + ", ".join(
        f"{n} ({CB_TEAM[n]}, {'R' if CB_ATTACKS_RIGHT[n] else 'L'})"
        for n in sorted(CB_TEAM)))
    p()

    random.seed(4000)
    res = build_pair("Pearls", "Justice").simulate()
    st = getattr(res, "state", res)

    p(f"match Pearls {st.home_goals}-{st.away_goals} Justice, "
      f"{len(res.timeline)} events")
    p()
    p("A. IS THE EXISTING CLAMP EVER REACHED FOR A CB?")
    for n in sorted(CB_TEAM):
        p(f"   {n:<22} clamp called {CLAMP_CALLS[n]:>6}  "
          f"actually clamped {CLAMP_ACTUAL[n]:>5}")
    p()
    p("B. WHY NOT -- the ON-BALL SKIP (match_engine.py:2755, ABOVE the clamp)")
    p(f"   {'cb':<22} {'ticks':>7} {'skipped(on-ball)':>17} {'share':>7} "
      f"{'max depth nx':>13}")
    for n in sorted(CB_TEAM):
        t = ALL_TICKS[n] or 1
        p(f"   {n:<22} {ALL_TICKS[n]:>7} {SKIP_TICKS[n]:>17} "
          f"{100.0*SKIP_TICKS[n]/t:>6.1f}% {MAX_NX.get(n, -1.0):>12.1f}")
    p()
    total_sk = sum(SKIP_TICKS.values())
    total_t = sum(ALL_TICKS.values()) or 1
    p(f"   ALL CBs: skipped on-ball {total_sk}/{total_t} "
      f"({100.0*total_sk/total_t:.1f}% of off-ball ticks)")
    p()
    p("C. IF A CB GOES DEEP, WHAT WROTE HIM THERE? (writes beyond the ceiling)")
    deep_total = sum(DEEP_SITES.values())
    p(f"   {deep_total} CB position writes beyond {CEIL} m")
    if DEEP_SITES:
        p("   by source site:")
        for site, c in DEEP_SITES.most_common():
            p(f"      {site:<40} {c:>5}")
        p("   by event type:")
        for ev, c in DEEP_EVENTS.most_common():
            p(f"      {ev:<40} {c:>5}")
    p()
    for n in sorted(CB_TEAM):
        tr = DEEP_TRACE.get(n) or []
        if not tr:
            p(f"   {n}: never wrote beyond the ceiling.")
            continue
        p(f"   {n} ({CB_TEAM[n]}) -- {len(tr)} deep writes, first/last:")
        for row in tr[:8]:
            p(f"      m{row[0]:>3}  nx {row[2]} -> {row[1]}  "
              f"[{row[4]}] {row[3]}")
        if len(tr) > 8:
            p(f"      ... ({len(tr)-8} more)")
            for row in tr[-4:]:
                p(f"      m{row[0]:>3}  nx {row[2]} -> {row[1]}  "
                  f"[{row[4]}] {row[3]}")
    p()

    with open("_diag_cb_ceiling.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_cb_ceiling.txt")


if __name__ == "__main__":
    main()
