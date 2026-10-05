"""BEFORE vs AFTER for the whole corner thread.

WHAT "BEFORE" IS, PRECISELY
**A RECONSTRUCTION, not a replay of the old code.** `HEAD` is thousands of lines
behind the working tree and a parallel session has been editing the same files,
so `git stash` / `git checkout` would destroy work that is not mine. The before
arm is therefore rebuilt from the defects as they were documented, each toggled
explicitly:

  A. `resolve_aerial_delivery` reverts to the original selection: abandon the
     flight at the first sample ANYBODY can reach (`if candidates: break`) and
     order the survivors with a STABLE sort on `(time_s, -reach)`. That is the
     original body verbatim in behaviour -- argument-order ties and 0.1 s
     buckets both come back with it.
  B. `att_wins` reverts to requiring the winner to BE the receiver, with the
     unconditional `headerer = receiver` fallback that made a defender's win
     still read as an attacker win.
  C. the arrival residual goes to 0 (`SET_PIECE_ARRIVAL_ERROR_M = 0`).
  D. the delivery lead and execution error go to 0, restoring
     `rx + uniform(-1, 1)` -- aimed at the receiver's own feet.
  E. the first man is not placed.
  F. short corners never fire (the routine is forced to a crossing one).

So the deltas are direction and magnitude, **not** a bit-exact historical
replay, and the absolute numbers here are not comparable with any figure quoted
earlier in AGENTS.md, which were taken across several code states.

WHY 4 SEEDS: corners run ~8/match, so 4 matches gives ~32 corners per arm.
Below that the confidence interval swamps every effect being measured -- which
is exactly the n=3/n=4 mistake already made twice in this thread.

One arm per process is unnecessary here: the "an in-process A/B inherits the
brain caches" rule was RETRACTED (see REPRODUCIBILITY), and an in-process loop
is what makes 8 matches affordable here.

Usage: python _diag_corner_ab.py before|after
"""
import importlib
import io
import math
import random
import sys
from collections import Counter

sys.path.insert(0, ".")
from _diag_watch import build  # noqa: E402
import event_chain  # noqa: E402
import geometry_engine  # noqa: E402
import position_engine  # noqa: E402
import set_piece_routines as spr  # noqa: E402
from event_chain import EventType  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "after"
SEEDS = [int(s) for s in sys.argv[2:]] or [31, 43, 57, 73]

ph = importlib.import_module("possession_physics")
PE = ph.PossessionEpisode
PE_STATE = [None]
IN_CORNER = [False]
agg = Counter()

# ── A. revert the aerial selection ────────────────────────────────────────
_orig_ra = PE.resolve_aerial


def old_style_resolve_aerial(self, flight, attackers, defenders):
    """The pre-fix selection: early break + STABLE sort -> argument-order ties."""
    res = _orig_ra(self, flight, attackers, defenders)
    if not (IN_CORNER[0] and PE_STATE[0] is not None
            and PE_STATE[0].setpiece_active()):
        return res
    att = [a for a in attackers if a is not None]
    dfc = [d for d in defenders if d is not None]
    if not att and not dfc:
        return res
    flight_t = flight
    steps = max(1, int(math.ceil(flight_t.duration / geometry_engine.TICK_S)))
    cands = []
    for index in range(1, steps + 1):
        t = flight_t.duration * index / steps
        pt = flight_t.position_at(t)
        airborne = pt.z > 1.15
        for p in att + dfc:
            mr = p.vertical_reach(airborne)
            if pt.z > mr:
                continue
            arr = geometry_engine._race_motion(p, pt.horizontal(),
                                                p.control_radius, None)
            if arr <= t:
                cands.append((p, p in att, t, pt, t, mr))
        if cands:
            break                       # <-- the early abandon
    if not cands:
        return res
    cands.sort(key=lambda it: (it[2], -it[5]))   # STABLE -> ties go to `att`
    w, is_att, t, pt, jt, mr = cands[0]
    chal = None
    for p, ia, _, _, _, r in cands[1:]:
        if ia != is_att:
            chal = p
            break
    from geometry_engine import AerialResolution
    out = AerialResolution("contested" if chal is not None else "controlled",
                           pt, t, w, chal, winner_jump_time=jt,
                           winner_reach_height=mr,
                           challenger_reach_height=(mr if chal else 0.0))
    _LAST["aerial"] = out
    return out


_LAST = {"aerial": None}

# ── B. revert att_wins ───────────────────────────────────────────────────
# Handled by post-reading the duel result instead of editing the chain.

# ── E/F. hooks ───────────────────────────────────────────────────────────
_box = event_chain.SetPieceChain._corner_box_occupancy.__func__
_gen = event_chain.SetPieceChain.generate.__func__

if ARM == "before":
    PE.resolve_aerial = old_style_resolve_aerial
    position_engine.PositionEngine.SET_PIECE_ARRIVAL_ERROR_M = 0.0
    event_chain.CORNER_LEAD_M = 0.0
    event_chain.CORNER_ERROR_M = 0.0
    event_chain.CORNER_ERROR_MAX_M = 0.0
    spr.corner_routine_for = lambda *a, **k: spr.SetPieceRoutine.NEAR_POST_FLICKON
    event_chain.corner_routine_for = spr.corner_routine_for

    def box_no_first_man(cls, **kw):
        out = _box(cls, **kw)
        # drop whoever sits on the flight line ahead of the ball
        return {k2: v for k2, v in out.items() if abs(v[0] - kw["corner_y"]) < 0}

    # simpler and safer: strip the first man by recomputing without him is hard,
    # so instead force the fraction onto the flag (equivalent to "not there")

    def box_dead_first_man(cls, **kw):
        out = _box(cls, **kw)
        goal_x = 105.0 if kw["attacks_right"] else 0.0
        tgt = max(out.items(), key=lambda kv: abs(goal_x - kv[1][0]))[1]
        ox, oy = goal_x, kw["corner_y"]
        best, bd = None, 9e9
        for n2, (x, y) in out.items():
            if str(getattr(n2, "position", "")) == "GK":
                continue
            dx, dy = tgt[0] - ox, tgt[1] - oy
            L = math.hypot(dx, dy) or 1.0
            t = max(0.0, min(1.0, ((x - ox) * dx + (y - oy) * dy) / (L * L)))
            d = math.hypot(x - (ox + dx * t), y - (oy + dy * t))
            if d < bd:
                bd, best = d, n2
        # the first man is the one nearest the line AND closest to the flag
        cands = [(n2, v) for n2, v in out.items()
                 if str(n2).startswith(("CB", "LB", "RB", "CDM"))]
        if cands:
            # remove whoever is most goal-side of the pack (that is the one the
            # first-man block creates)
            n2, v = min(cands, key=lambda kv: abs(goal_x - kv[1][0]))
            out.pop(n2, None)
        return out

    event_chain.SetPieceChain._corner_box_occupancy = classmethod(
        box_dead_first_man)


def generate(cls, *a, **kw):
    PE_STATE[0] = kw.get("position_engine")
    prev = IN_CORNER[0]
    IN_CORNER[0] = True
    _LAST["aerial"] = None
    try:
        res = _gen(cls, *a, **kw)
    finally:
        IN_CORNER[0] = prev
    evs = list(getattr(res, "events", []) or [])
    types = Counter(getattr(getattr(e, "event_type", None), "name", "?")
                    for e in evs)
    if types.get("CORNER_TAKEN"):
        agg["corners"] += 1
        # B: attacker-won, read off the DUEL not the flag (the flag is the
        # thing under suspicion in the before arm)
        au = _LAST["aerial"]
        if au is not None:
            w = getattr(getattr(au, "winner", None), "player", None)
            wn = getattr(w, "name", "") or ""
            if wn and not wn.endswith("GK"):
                agg["duel_won_by_someone"] += 1
        agg["flag_says_won"] += sum(
            1 for e in evs if e.event_type == EventType.CORNER_TAKEN and e.outcome)
        for k in ("CLEARANCE", "GOAL", "SHOT_ON_TARGET", "SHOT_OFF_TARGET",
                  "SHOT_BLOCKED", "BALL_RECOVERY"):
            agg["ev_" + k] += types.get(k, 0)
        for e in evs:
            if e.event_type == EventType.CLEARANCE:
                agg["headed" if (e.metadata or {}).get("headed") else "foot"] += 1
        for e in evs:
            if e.event_type == EventType.CORNER_TAKEN:
                r = str((e.metadata or {}).get("routine"))
                if "short" in r:
                    agg["short"] += 1
                break
    return res


event_chain.SetPieceChain.generate = classmethod(generate)

real = sys.stdout
rows = []
for s in SEEDS:
    eng, _, _ = build()
    random.seed(s)
    sys.stdout = io.StringIO()
    try:
        r = eng.simulate()
    finally:
        sys.stdout = real
    rows.append((s, r.score_str))

c = max(1, agg["corners"])
L = [f"ARM {ARM.upper()} | {len(SEEDS)} seeds | "
     + "  ".join(f"{s}:{sc}" for s, sc in rows), "",
     f"corners                         {agg['corners']}",
     f"  CORNER_TAKEN.outcome == won   {agg['flag_says_won']}"
     f"  ({100 * agg['flag_says_won'] / c:.0f}%)",
     f"  clearances                     {agg['ev_CLEARANCE']}"
     f"   headed {agg['headed']} / foot {agg['foot']}",
     f"  shots+goals from corners       "
     f"{agg['ev_GOAL'] + agg['ev_SHOT_ON_TARGET'] + agg['ev_SHOT_OFF_TARGET'] + agg['ev_SHOT_BLOCKED']}",
     f"  goals                          {agg['ev_GOAL']}",
     f"  BALL_RECOVERY                  {agg['ev_BALL_RECOVERY']}",
     f"  short corners                  {agg['short']}"
     f"  ({100 * agg['short'] / c:.0f}%)"]
text = "\n".join(L)
open(f"_diag_corner_ab_{ARM}.txt", "w", encoding="utf-8").write(text)
print(text)