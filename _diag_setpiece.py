"""SET-PIECE MEASUREMENT PROBE (2026-10-01).

Answers three questions with numbers instead of reading the code and deciding:

  1. HOW MANY PLAYERS ARE ACTUALLY IN THE BOX when a corner is crossed?
     (`record_touch` is logged per chain call, so "the other eighteen outfielders
     are never repositioned" is measured, not asserted.)

  2. HOW FAR DOES ANYONE GET REPOSITIONED IN ONE CALL?
     The teleport measure. A corner header plants the receiver at
     `88.0 + uniform(0, 6)` with no travel and no time; the displacement from
     wherever he actually stood is what the run jump-filter counts as a
     reposition rather than a run.

  3. DOES A DEFENSIVE WALL EXIST FOR A DIRECT FREE KICK?
     Count defenders inside the ball -> near-post corridor at 6.5-11.5 m
     (the 9.15 m wall distance plus the radius of a body) at the moment the
     kick is taken. A wall is 4-5 men. Zero is not a badly-built wall, it is
     no wall, and a direct free kick then goes at an undefended goal.

Everything is observation-only: the chain wrappers call the originals and
`record_touch` is wrapped, not replaced. No engine behaviour is altered.

Run:  .venv\\Scripts\\python.exe _diag_setpiece.py [n_matches]
"""
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import date

from event_chain import SetPieceChain
from match_engine import (
    EventType, SituationType,
    MatchConfig, MatchEngine, PlayingStyle, TeamProfile, TeamStyle, Intensity,
)
from player_dna import SquadBuilder
from position_engine import PositionEngine
from roster_loader import get_loader

XLSX = "PLOFA-2026-2027.xlsx"
OUT = "_diag_setpiece"

# ── probe state ────────────────────────────────────────────────────────
_LOG = []            # record_touch calls made inside the current chain
_CTX = {"kind": None, "minute": None, "attacks_right": True,
        "att": (), "def": (), "pe": None}
_CORNER = []         # per-corner records
_FK = []             # per-direct-FK records
_MAXJUMP = [0.0]


def _in_box(x, y, attacks_right, margin=0.0):
    """Inside the penalty box, in either team's own frame."""
    near = 88.0 - margin if attacks_right else 17.0 + margin
    far = 105.0 + margin if attacks_right else -margin
    lo, hi = (near, far) if attacks_right else (far, near)
    return lo <= x <= hi and 16.0 <= y <= 52.0


def _patch():
    orig_corner = SetPieceChain._corner_chain.__func__
    orig_fk = SetPieceChain._freekick_chain.__func__
    orig_touch = PositionEngine.record_touch

    def touch(self, name, x, y, minute):
        if _CTX["kind"] and x is not None and y is not None:
            try:
                px, py = self.get_position(name)
                d = math.hypot(x - px, y - py)
            except Exception:
                px, py, d = None, None, 0.0
            _MAXJUMP[0] = max(_MAXJUMP[0], d)
            _LOG.append({"kind": _CTX["kind"], "minute": minute, "name": name,
                         "from": (px, py), "to": (x, y), "dist": d})
        return orig_touch(self, name, x, y, minute)

    def corner(cls, minute, att_team, def_team, att_players, def_players,
               state, attacks_right=True, position_engine=None, routine=None,
               delivery_origin=None):
        prev = dict(_CTX)
        _CTX.update(kind="corner", minute=minute, attacks_right=attacks_right,
                    att=tuple(p.name for p in att_players),
                    dfn=tuple(p.name for p in def_players),
                    pe=position_engine)
        _LOG.clear()
        before = _snapshot(position_engine, att_players, def_players)
        try:
            res = orig_corner(cls, minute, att_team, def_team, att_players,
                              def_players, state, attacks_right,
                              position_engine, routine, delivery_origin)
        finally:
            after = _snapshot(position_engine, att_players, def_players)
            calls = list(_LOG)
            _CTX.clear(); _CTX.update(prev)

        if position_engine is not None:
            _CORNER.append({
                "minute": minute,
                "calls": calls,
                "moved": len({c["name"] for c in calls}),
                "att_in_box_before": _count(before, att_players, attacks_right),
                "def_in_box_before": _count(before, def_players, attacks_right),
                "att_in_box_after": _count(after, att_players, attacks_right),
                "def_in_box_after": _count(after, def_players, attacks_right),
                "routine": None if routine is None else str(routine),
                "events": res.events,
                "goal": res.goal_scored,
                "xg": res.xg_generated,
            })
        return res

    def fk(cls, minute, att_team, def_team, att_players, def_players, state,
           situation, attacks_right=True, context_x=None, context_y=None,
           position_engine=None, routine=None):
        prev = dict(_CTX)
        _CTX.update(kind="fk", minute=minute, attacks_right=attacks_right,
                    att=tuple(p.name for p in att_players),
                    dfn=tuple(p.name for p in def_players),
                    pe=position_engine)
        _LOG.clear()
        before = _snapshot(position_engine, att_players, def_players)
        try:
            res = orig_fk(cls, minute, att_team, def_team, att_players,
                          def_players, state, situation, attacks_right,
                          context_x, context_y, position_engine, routine)
        finally:
            after = _snapshot(position_engine, att_players, def_players)
            calls = list(_LOG)
            _CTX.clear(); _CTX.update(prev)

        if position_engine is not None:
            types = [e.event_type for e in res.events]
            direct = EventType.FREEKICK_DIRECT in types
            _FK.append({
                "minute": minute,
                "direct": direct,
                "crossed": EventType.FREEKICK_CROSS in types,
                "calls": calls,
                "moved": len({c["name"] for c in calls}),
                "fx": context_x, "fy": context_y,
                "wall_before": _wall(before, def_players, context_x, context_y,
                                     attacks_right),
                "wall_after": _wall(after, def_players, context_x, context_y,
                                    attacks_right),
                "before": before, "after": after,
                "events": res.events,
                "goal": res.goal_scored,
            })
        return res

    PositionEngine.record_touch = touch
    SetPieceChain._corner_chain = classmethod(corner)
    SetPieceChain._freekick_chain = classmethod(fk)
    return lambda: (setattr(PositionEngine, "record_touch", orig_touch),
                    setattr(SetPieceChain, "_corner_chain",
                            classmethod(orig_corner)),
                    setattr(SetPieceChain, "_freekick_chain",
                            classmethod(orig_fk)))


def _snapshot(pe, att_players, def_players):
    out = {}
    if pe is None:
        return out
    for p in list(att_players) + list(def_players):
        try:
            out[p.name] = pe.get_position(p.name)
        except Exception:
            pass
    return out


def _count(snap, players, attacks_right):
    n = 0
    for p in players:
        pos = snap.get(p.name)
        if pos and _in_box(pos[0], pos[1], attacks_right):
            n += 1
    return n


def _wall(snap, def_players, fx, fy, attacks_right):
    """Defenders standing in the ball -> near-post corridor at wall distance.

    A wall is a LINE of men between the ball and the goal, 9.15 m out, facing
    the ball. Counted as: on the goal side of the ball, 6.5-11.5 m away, and
    within 12 m laterally of the ball->near-post line.
    """
    if fx is None or fy is None:
        return -1
    goal_x = 105.0 if attacks_right else 0.0
    # near post is the post on the taker's side of the goal
    near_y = 30.34 if fy < 34 else 37.66
    dx, dy = goal_x - fx, near_y - fy
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    n = 0
    for p in def_players:
        if getattr(p, "position", "") == "GK":
            continue
        pos = snap.get(p.name)
        if not pos:
            continue
        vx, vy = pos[0] - fx, pos[1] - fy
        along = vx * ux + vy * uy
        if not (6.5 <= along <= 11.5):
            continue
        perp = abs(vx * uy - vy * ux)
        if perp <= 12.0:
            n += 1
    return n


def _situation_of(e):
    return getattr(e, "situation", None)


def build_pair(seed, home, away):
    loader = get_loader(XLSX)
    hr = loader.build_matchday_squad(home)
    ar = loader.build_matchday_squad(away)
    hs = SquadBuilder.build(home, starters=hr["starters"],
                            substitutes=hr["substitutes"])
    aws = SquadBuilder.build(away, starters=ar["starters"],
                             substitutes=ar["substitutes"])
    cfg = MatchConfig(home_team=home, away_team=away,
                      match_date=date(2026, 8, 16), matchday=1,
                      venue=f"{home} Stadium", stadium_capacity=45000)
    hp = TeamProfile(name=home, style=TeamStyle.BALANCED,
                     playing_style=PlayingStyle.POSSESSION,
                     intensity=Intensity.MEDIUM)
    ap = TeamProfile(name=away, style=TeamStyle.BALANCED,
                     playing_style=PlayingStyle.MIXED,
                     intensity=Intensity.MEDIUM)
    eng = MatchEngine(cfg, hp, ap)
    eng.set_squad(home, hs["starters"], hs["substitutes"])
    eng.set_squad(away, aws["starters"], aws["substitutes"])
    return eng


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    restore = _patch()
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
             ("Justice", "Triumpher")]
    totals = {"corners": 0, "corner_shots": 0, "corner_sot": 0,
              "corner_goals": 0, "corner_xg": 0.0}
    direct = {"n": 0, "shots": 0, "goals": 0}
    wall_counts = Counter()
    moved_counts = Counter()
    jumps = []
    try:
        for i in range(n):
            h, a = pairs[i % len(pairs)]
            random.seed(1000 + i)
            eng = build_pair(1000 + i, h, a)
            res = eng.simulate()
            print(f"[{i+1}/{n}] {h} v {a}: {res.score_str}  "
                  f"({len(_CORNER)} corners, {len(_FK)} fk so far)",
                  flush=True)
    finally:
        restore()

    # ── corner accounting ────────────────────────────────────────────────
    for c in _CORNER:
        totals["corners"] += 1
        shots = [e for e in c["events"]
                 if e.event_type in (EventType.SHOT_ON_TARGET,
                                     EventType.SHOT_OFF_TARGET)
                 and _situation_of(e) == SituationType.CORNER]
        # events emitted without an explicit situation default to OPEN_PLAY;
        # count those separately because they are a known labelling defect.
        stray = [e for e in c["events"]
                 if e.event_type in (EventType.SHOT_ON_TARGET,
                                     EventType.SHOT_OFF_TARGET)
                 and _situation_of(e) == SituationType.OPEN_PLAY]
        totals["corner_shots"] += len(shots) + len(stray)
        totals["corner_sot"] += sum(1 for e in shots
                                    if e.event_type == EventType.SHOT_ON_TARGET)
        totals["corner_goals"] += 1 if c["goal"] else 0
        totals["corner_xg"] += (c["xg"] or 0.0)
        moved_counts[c["moved"]] += 1
        jumps.extend(x["dist"] for x in c["calls"])
        c["_stray"] = len(stray)

    for f in _FK:
        if not f["direct"]:
            continue
        direct["n"] += 1
        direct["goals"] += 1 if f["goal"] else 0
        direct["shots"] += sum(1 for e in f["events"]
                               if e.event_type in (EventType.SHOT_ON_TARGET,
                                                   EventType.SHOT_OFF_TARGET))
        wall_counts[max(f["wall_before"], f["wall_after"])] += 1
        moved_counts[("fk", f["moved"])] += 1

    cn = max(totals["corners"], 1)
    dn = max(direct["n"], 1)
    print("\n" + "=" * 72)
    print("CORNERS")
    print(f"  corners taken              {totals['corners']}")
    print(f"  players repositioned/corner{'':<14}"
          f"{dict(sorted(moved_counts.items(), key=lambda kv: str(kv[0])))}")
    if totals["corner_shots"]:
        print(f"  shots from corners         {totals['corner_shots']}"
              f"  ({totals['corner_shots']/cn:.2f} per corner)")
    if totals["corner_sot"]:
        print(f"  on target                  {totals['corner_sot']}"
              f"  ({totals['corner_sot']/totals['corner_shots']*100:.1f}% of shots)")
    print(f"  xG from corners            {totals['corner_xg']:.2f}")
    print(f"  GOALS from corners         {totals['corner_goals']}"
          f"   ({totals['corner_goals']/cn*100:.2f}% of corners)"
          f"   [real PL ~ 3.0-4.0%]")

    ab = [(c["att_in_box_before"], c["att_in_box_before"] + c["def_in_box_before"])
          for c in _CORNER]
    aa = [(c["att_in_box_after"], c["att_in_box_after"] + c["def_in_box_after"])
          for c in _CORNER]
    if ab:
        import statistics as st
        print(f"  players in box, BEFORE      att {st.mean(a for a,_ in ab):.2f}"
              f"  total {st.mean(t for _,t in ab):.2f}")
        print(f"  players in box, AFTER       att {st.mean(a for a,_ in aa):.2f}"
              f"  total {st.mean(t for _,t in aa):.2f}")
        print(f"  [real corner: ~7-9 attackers + ~9-10 defenders in/around the box]")

    if jumps:
        import statistics as st
        print(f"  record_touch displacement  max {max(jumps):.1f} m"
              f"   mean {st.mean(jumps):.1f} m   n={len(jumps)}")
        print(f"  displ > 10 m (teleport)    {sum(1 for d in jumps if d > 10)}"
              f" / {len(jumps)}")

    stray = sum(c.get("_stray", 0) for c in _CORNER)
    if stray:
        print(f"  NOTE: {stray} corner shots carry situation=OPEN_PLAY "
              f"(event_chain.py:6965 omits it)")

    print("\nDIRECT FREE KICKS")
    print(f"  direct FKs                 {direct['n']}")
    print(f"  shots                      {direct['shots']}"
          f"  ({direct['shots']/dn:.2f} per direct FK)")
    print(f"  GOALS                      {direct['goals']}"
          f"   ({direct['goals']/dn*100:.1f}% of direct FKs)")
    print(f"  defenders in the wall       {dict(sorted(wall_counts.items()))}")
    print(f"  players repositioned       "
          f"{ {k[1]: v for k, v in moved_counts.items() if isinstance(k, tuple)} }")
    print("  [a real wall is 4-5 men; 0 means NO wall is built at all]")
    print("=" * 72)


if __name__ == "__main__":
    main()