"""SET-PIECE BASELINE, FULLER (2026-10-01).

`_diag_setpiece.py` answered the box-crowding and teleport questions. This one
answers the two that decide the SHAPE OF THE FIX:

  A. Do direct free kicks ever reach `_freekick_chain` as DIRECT?
     The only thing standing between "add a wall" and a wall that never
     executes is whether the direct branch is live. AGENTS.md's recurring
     pathology is a mechanism wired to something that never runs - so this is
     measured before anything is built.

  B. How far is each player from the box at the moment of a corner?
     A bounded approach (real speed x jostling time) can only bring players
     who are CLOSE. The distance distribution is what decides whether a
     crowded box is even physically reachable, and it is the difference
     between "crowd the box" and "teleport 20 men in".

  C. Corner xG vs real. If corners already score BELOW the real 3-4%, then
     "at the cost of corner goals" is a cost that does not currently exist,
     and the honest finding is the opposite of the premise.

Observation-only. Run:  .venv\\Scripts\\python.exe _diag_sp_baseline.py [n]
"""
import math
import random
import statistics as st
import sys
from collections import Counter
from datetime import date

from event_chain import SetPieceChain
from match_engine import (
    EventType, SituationType, MatchConfig, MatchEngine, PlayingStyle,
    TeamProfile, TeamStyle, Intensity,
)
from player_dna import SquadBuilder
from roster_loader import get_loader

XLSX = "PLOFA-2026-2027.xlsx"
JOSTLE_S = 5.0          # the real window between the ball being placed and struck

_C = {"corners": [], "fks": []}


def _patch():
    oc = SetPieceChain._corner_chain.__func__
    of = SetPieceChain._freekick_chain.__func__

    def corner(cls, minute, att_team, def_team, att_players, def_players,
               state, attacks_right=True, position_engine=None, routine=None,
               delivery_origin=None):
        dist = []
        if position_engine is not None:
            gx = 105.0 if attacks_right else 0.0
            dist = []
            for p in list(att_players) + list(def_players):
                if getattr(p, "position", "") == "GK":
                    continue
                try:
                    x, y = position_engine.get_position(p.name)
                except Exception:
                    continue
                # distance INTO the box: how far goal-side of the 88/17 line
                inside = (gx - x) if attacks_right else (x - gx)
                dist.append((inside, p.position))
        res = oc(cls, minute, att_team, def_team, att_players, def_players,
                 state, attacks_right, position_engine, routine,
                 delivery_origin)
        types = [e.event_type for e in res.events]
        _C["corners"].append({
            "attacks_right": attacks_right,
            "inside": dist,
            "goal": res.goal_scored, "xg": res.xg_generated or 0.0,
            "aerial": any(t == EventType.AERIAL_DUEL for t in types),
            "aerial_won": any(
                t == EventType.AERIAL_DUEL and e.metadata
                and e.metadata.get("winner_team") == att_team
                for t, e in zip(types, res.events) if t == EventType.AERIAL_DUEL),
            "shot": any(t in (EventType.SHOT_ON_TARGET,
                              EventType.SHOT_OFF_TARGET) for t in types),
            "sot": any(t == EventType.SHOT_ON_TARGET for t in types),
            "n_att": len([p for p in att_players
                          if getattr(p, "position", "") != "GK"]),
            "n_def": len([p for p in def_players
                          if getattr(p, "position", "") != "GK"]),
        })
        return res

    def fk(cls, minute, att_team, def_team, att_players, def_players, state,
           situation, attacks_right=True, context_x=None, context_y=None,
           position_engine=None, routine=None):
        res = of(cls, minute, att_team, def_team, att_players, def_players,
                 state, situation, attacks_right, context_x, context_y,
                 position_engine, routine)
        types = [e.event_type for e in res.events]
        _C["fks"].append({
            "situation": str(getattr(situation, "value", situation)),
            "x": context_x, "y": context_y,
            "attacks_right": attacks_right,
            "direct_range": (context_x > 78) if (context_x is not None
                                                 and attacks_right)
                            else ((context_x < 27)
                                  if context_x is not None else False),
            "routine": None if routine is None else str(routine),
            "emitted_direct": EventType.FREEKICK_DIRECT in types,
            "emitted_cross": EventType.FREEKICK_CROSS in types,
            "goal": res.goal_scored,
        })
        return res

    SetPieceChain._corner_chain = classmethod(corner)
    SetPieceChain._freekick_chain = classmethod(fk)
    return lambda: (setattr(SetPieceChain, "_corner_chain", classmethod(oc)),
                    setattr(SetPieceChain, "_freekick_chain", classmethod(of)))


def build_pair(home, away):
    loader = get_loader(XLSX)
    hr = loader.build_matchday_squad(home)
    ar = loader.build_matchday_squad(away)
    hs = SquadBuilder.build(home, starters=hr["starters"],
                            substitutes=hr["substitutes"])
    aw = SquadBuilder.build(away, starters=ar["starters"],
                             substitutes=ar["substitutes"])
    cfg = MatchConfig(home_team=home, away_team=away,
                      match_date=date(2026, 8, 16), matchday=1,
                      venue=f"{home} Stadium", stadium_capacity=45000)
    eng = MatchEngine(
        cfg,
        TeamProfile(name=home, style=TeamStyle.BALANCED,
                    playing_style=PlayingStyle.POSSESSION,
                    intensity=Intensity.MEDIUM),
        TeamProfile(name=away, style=TeamStyle.BALANCED,
                    playing_style=PlayingStyle.MIXED,
                    intensity=Intensity.MEDIUM))
    eng.set_squad(home, hs["starters"], hs["substitutes"])
    eng.set_squad(away, aw["starters"], aw["substitutes"])
    return eng


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    restore = _patch()
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City"),
             ("Justice", "Triumpher"), ("Lige-8", "Telbey")]
    try:
        for i in range(n):
            h, a = pairs[i % len(pairs)]
            random.seed(1000 + i)
            build_pair(h, a).simulate()
            print(f"  [{i+1}/{n}] {h} v {a} done", flush=True)
    finally:
        restore()

    C, F = _C["corners"], _C["fks"]
    print("\n" + "=" * 74)
    print("A. DO DIRECT FREE KICKS EVER REACH THE DIRECT BRANCH?")
    print(f"  free kick chains entered          {len(F)}")
    print(f"  situation values                  {dict(Counter(f['situation'] for f in F))}")
    print(f"  in direct range (x>78 / x<27)     {sum(1 for f in F if f['direct_range'])}")
    print(f"  emitted FREEKICK_DIRECT           {sum(1 for f in F if f['emitted_direct'])}")
    print(f"  emitted FREEKICK_CROSS            {sum(1 for f in F if f['emitted_cross'])}")
    xs = [f["x"] for f in F if f["x"] is not None]
    if xs:
        print(f"  fk x: min {min(xs):.1f} max {max(xs):.1f} "
              f"mean {st.mean(xs):.1f}  (direct needs >78)")
        print(f"  x > 78 exactly                   {sum(1 for x in xs if x > 78)}")
    print("  -> a wall is only worth building if the count above is non-zero.")

    print("\nB. HOW FAR IS EVERYONE FROM THE BOX AT A CORNER?")
    print("     `inside` = metres goal-side of the box line (88 / 17).")
    allin = [d for c in C for d, _ in (c["inside"] or [])]
    per = [len(c["inside"] or []) for c in C]
    for lo, hi, lab in [(99, 1e9, "in the box already"),
                        (0, 99, "outside the box"),
                        (-15, 0, "behind the box line"),
                        (-40, -15, "15-40 m out"),
                        (-1e9, -40, "40+ m out")]:
        k = len([d for d in allin if lo <= d < hi])
        print(f"     {lab:<24}{k:>5}  ({100*k/max(len(allin),1):.1f}%)")
    reach = [d for d in allin if d < 15]     # can cover 15 m in 5 s? ~3 m/s+
    print(f"     within 15 m of the box line   {len(reach)} "
          f"({100*len(reach)/max(len(allin),1):.1f}% of all outfielders)")
    print(f"     outfielders per corner        mean {st.mean(per) if per else 0:.1f}"
          f"  (att {st.mean([c['n_att'] for c in C]):.1f}"
          f" / def {st.mean([c['n_def'] for c in C]) if C else 0:.1f})")
    print(f"     real corner                   att ~7-9 in box, def ~9-10 around it")

    print("\nC. CORNER OUTCOMES vs REAL")
    nc = max(len(C), 1)
    _ = [c for c in C if not c["inside"]]
    print(f"  corners                          {len(C)}")
    print(f"  aerial duel resolved             {sum(1 for c in C if c['aerial'])}")
    print(f"  ATTACKER won the aerial          {sum(1 for c in C if c['aerial_won'])}"
          f"  ({100*sum(1 for c in C if c['aerial_won'])/nc:.1f}% of corners)")
    print(f"  shot from corner                 {sum(1 for c in C if c['shot'])}"
          f"  ({100*sum(1 for c in C if c['shot'])/nc:.1f}%)")
    print(f"  on target                        {sum(1 for c in C if c['sot'])}"
          f"  ({100*sum(1 for c in C if c['sot'])/nc:.1f}% of corners)")
    print(f"  GOAL                             {sum(1 for c in C if c['goal'])}"
          f"  ({100*sum(1 for c in C if c['goal'])/nc:.2f}%)")
    print(f"  xG per corner                    "
          f"{sum(c['xg'] for c in C)/nc:.4f}   (real ~0.030-0.045)")
    print("  -> if goals are ALREADY below 3%, the 'cost of corner goals' the")
    print("     premise assumes does not exist in this direction yet.")
    print("=" * 74)


if __name__ == "__main__":
    main()