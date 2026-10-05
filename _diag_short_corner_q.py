"""The three open questions about short corners, answered in ONE match pass.

  Q1  is the short pass usually lost?
  Q2  how often is it worked back into the box?
  Q3  how does `resolve_aerial_delivery` resolve a duel on a 0.35 m ground
      ball -- specifically, how many men can even REACH it?

WHY THE ROUTINE IS FORCED. A short corner runs at ~13% of corners and a match
has ~8, so a natural match yields roughly ONE short corner. Ten of them would
need ten matches. The routine is therefore FORCED here so every corner in the
match is short, which samples the mechanic properly. **This is a behavioural
probe: the shipped 13% pool weighting is NOT what is being measured, and the
forced-arm numbers must never be quoted as the frequency.** `_diag_short_corner_match.py`
measures the unforced frequency.

Q3 is the risk, and it is a real one. `resolve_aerial_delivery` requires
`arrival <= time_s` -- the man must be able to BE THERE before the ball gets
there -- and a short ground pass has a sub-second flight. A defender 15 m away
cannot arrive in 0.4 s, so he is excluded outright. If almost nobody qualifies
then a short pass is un-interceptable, which is wrong: real short corners are
lost all the time. That is what the CONTESTANT COUNT measures.
"""
import io
import random
import sys
from collections import Counter

sys.path.insert(0, ".")
from _diag_watch import build  # noqa: E402
import event_chain  # noqa: E402
import set_piece_routines as spr  # noqa: E402
import importlib  # noqa: E402
from event_chain import EventType  # noqa: E402

# ── force the routine ──
spr.corner_routine_for = lambda *a, **k: spr.SetPieceRoutine.SHORT_CORNER
event_chain.corner_routine_for = spr.corner_routine_for

rows = []
stats = Counter()
_pe = [None]
_in = [False]

ph = importlib.import_module("possession_physics")
PE = ph.PossessionEpisode
_ra = PE.resolve_aerial
_duel = []


def resolve_aerial(self, flight, attackers, defenders):
    r = _ra(self, flight, attackers, defenders)
    if _in[0]:
        att = [a for a in attackers if a is not None]
        dfc = [d for d in defenders if d is not None]
        w = getattr(getattr(r, "winner", None), "player", None)
        _duel.append({
            "dur": round(getattr(flight, "duration", 0.0), 3),
            "apex": round(getattr(flight, "apex_z", 0.0), 2),
            "n_att": len(att), "n_def": len(dfc),
            "n_total": len(att) + len(dfc),
            "winner": getattr(w, "name", "") or "",
            "outcome": getattr(r, "outcome", ""),
            # THE decisive number for a SHORT corner: how far is the ball
            # actually going? A short pass is 8-18 m. If this reads ~35 m the
            # target edit is not firing and the "short" corner is a cross.
            "dist": round(
                ((getattr(flight.target, "x", 0.0)
                  - getattr(flight.start, "x", 0.0)) ** 2
                 + (getattr(flight.target, "y", 0.0)
                    - getattr(flight.start, "y", 0.0)) ** 2) ** 0.5, 1),
            "tz": round(getattr(flight.target, "z", 0.0), 2),
        })
    return r


PE.resolve_aerial = resolve_aerial

_gen = event_chain.SetPieceChain.generate.__func__


def generate(cls, *a, **kw):
    prev = _in[0]
    _pe[0] = kw.get("position_engine")
    _in[0] = True
    _duel.clear()
    try:
        res = _gen(cls, *a, **kw)
    finally:
        _in[0] = prev
    evs = list(getattr(res, "events", []) or [])
    types = Counter(getattr(getattr(e, "event_type", None), "name", "?")
                    for e in evs)
    if types.get("CORNER_TAKEN"):
        stats["corners"] += 1
        stats["duels"] += len(_duel)
        for k, v in types.items():
            stats["ev_" + k] += v
        # Q1: did the attacking side keep the ball?
        won = [e for e in evs if e.event_type == EventType.CORNER_TAKEN
               and e.outcome]
        stats["att_won"] += len(won)
        # Q2: anything that got the ball back toward the box?
        stats["into_box"] += types.get("SHOT_ON_TARGET", 0) + \
            types.get("SHOT_OFF_TARGET", 0) + types.get("SHOT_BLOCKED", 0) + \
            types.get("GOAL", 0)
        rows.append((dict(types), [_duel[i] for i in range(len(_duel))]))
    return res


event_chain.SetPieceChain.generate = classmethod(generate)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 31
eng, _, _ = build()
random.seed(seed)
real = sys.stdout
sys.stdout = io.StringIO()
try:
    res = eng.simulate()
finally:
    sys.stdout = real

c = max(1, stats["corners"])
L = [f"SHORT CORNER FORCED | seed {seed} | {res.score_str} | "
     f"events {len(res.timeline)}", "",
     f"corners (all short)  {stats['corners']}", ""]

L.append("Q1  IS THE SHORT PASS USUALLY LOST?")
L.append(f"    attacking side won the delivery   {stats['att_won']}/{c} "
         f"({100 * stats['att_won'] / c:.0f}%)")
L.append(f"    chain produced a CLEARANCE        {stats['ev_CLEARANCE']}")
L.append(f"    chain produced a BALL_RECOVERY    {stats['ev_BALL_RECOVERY']}")
L.append("")
L.append("Q2  HOW OFTEN IS IT WORKED BACK INTO THE BOX?")
L.append(f"    shots+goals from the corner       {stats['into_box']}"
         f"  ({stats['into_box'] / c:.2f} per corner)")
L.append(f"    AERIAL_DUEL events                {stats['ev_AERIAL_DUEL']}")
L.append(f"    BALL_RECOVERY events              {stats['ev_BALL_RECOVERY']}")
L.append("")
L.append("Q3  THE GROUND-BALL DUEL -- who can even reach it?")
alld = [d for _, ds in rows for d in ds]
if alld:
    L.append(f"    duels recorded                   {len(alld)}")
    L.append(f"    flight duration                  "
             f"{min(d['dur'] for d in alld):.2f}-"
             f"{max(d['dur'] for d in alld):.2f} s")
    L.append(f"    PASS LENGTH (start -> target)     "
             f"{min(d['dist'] for d in alld):.1f}-"
             f"{max(d['dist'] for d in alld):.1f} m"
             f"   <- a short corner is 8-18 m")
    L.append(f"    target height                    "
             f"{min(d['tz'] for d in alld):.2f}-"
             f"{max(d['tz'] for d in alld):.2f} m")
    L.append(f"    apex                             "
             f"{min(d['apex'] for d in alld):.2f}-"
             f"{max(d['apex'] for d in alld):.2f} m"
             f"   <- if target z is 0.35 but apex is high, it is MAGNUS")
    ns = [d["n_total"] for d in alld]
    L.append(f"    contestants QUALIFIED (could arrive) "
             f"{min(ns)}-{max(ns)}, median "
             f"{sorted(ns)[len(ns) // 2]}")
    L.append(f"      attackers offered              "
             f"{sum(d['n_att'] for d in alld) / len(alld):.1f} per duel")
    L.append(f"      defenders offered              "
             f"{sum(d['n_def'] for d in alld) / len(alld):.1f} per duel")
    L.append(f"    outcomes                         "
             f"{dict(Counter(d['outcome'] for d in alld))}")
    one = sum(1 for d in alld if d["n_total"] <= 2)
    L.append(f"    duels with <=2 contestants       {one}/{len(alld)}"
             f"  <- if most are here, the pass is uncontestable")
else:
    L.append("    NO DUELS RECORDED")

text = "\n".join(L)
open("_diag_short_corner_q.txt", "w", encoding="utf-8").write(text)
print(text)