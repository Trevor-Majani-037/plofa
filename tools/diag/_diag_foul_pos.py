"""WHERE DO FOULS HAPPEN, AND HOW MANY WOULD BE DIRECT FREE KICKS?

(2026-10-01) Measured BEFORE building the foul -> free-kick award, because the
volume of direct free kicks is an OUTPUT of this distribution, not something
to pick. If I invent a rate and the positions say otherwise, the invention is
what gets tuned, not the football.

The award gap: `pending_offside_fk_for` (match_engine.py:5693) is the ONLY
queue that feeds `_freekick_chain`, so today every free kick in a match is an
OFFSIDE restart. Ordinary fouls (~21/match) award cards and penalties but no
free kick. Measured on the offside queue: 13 of 13 free kicks came from 4627,
0 from the open-play raise, 0 reached the direct branch.

This measures, per real match:
  * fouls committed, and where
  * how many land in DIRECT RANGE of the fouled team (x > 78 / x < 27)
  * how many are in the fouled team's attacking third generally
  * the same for the offside restarts, as the control group

The control group matters: it is the ONLY existing free-kick source, so if the
foul distribution is broadly similar, the missing award is the whole story
rather than a position problem.

Observation-only. Run:  .venv\\Scripts\\python.exe _diag_foul_pos.py [n]
"""
import random
import statistics as st
import sys
from collections import Counter
from datetime import date

from match_engine import (
    EventType, SituationType, MatchConfig, MatchEngine, PlayingStyle,
    TeamProfile, TeamStyle, Intensity,
)
from event_chain import ChainDispatcher, DisciplineChain
from player_dna import SquadBuilder
from roster_loader import get_loader

XLSX = "PLOFA-2026-2027.xlsx"
FOULS = []
OFFSIDE_FKS = []


def _patch():
    od = ChainDispatcher.discipline
    osp = ChainDispatcher.set_piece

    def discipline(*a, **kw):
        res = od(*a, **kw)
        # fouling_team, fouled_team, attacks_right, x, y
        for e in res.events:
            if e.event_type == EventType.FOUL_COMMITTED:
                fouling = e.team
                fouled = (a[2] if len(a) > 2 else kw.get("fouled_team"))
                att_right = kw.get("attacks_right",
                                   a[8] if len(a) > 8 else True)
                FOULS.append({
                    "x": e.location_x, "y": e.location_y,
                    "att_right": bool(att_right),
                    "fouled": fouled,
                    "penalty": res.penalty_won,
                    # EventType has no SECOND_YELLOW; a second yellow and a
                    # straight red both surface as RED_CARD.
                    "card": any(ev.event_type in
                                (EventType.YELLOW_CARD, EventType.RED_CARD)
                                for ev in res.events),
                })
                break
        return res

    def set_piece(minute, att_team, def_team, att_players, def_players, state,
                  situation, attacks_right=True, context_x=None, context_y=None,
                  position_engine=None, routine=None):
        if situation in (SituationType.DIRECT_FREEKICK,
                         SituationType.CROSSED_FREEKICK):
            OFFSIDE_FKS.append({
                "x": context_x, "y": context_y,
                "att_right": bool(attacks_right),
            })
        return osp(minute, att_team, def_team, att_players, def_players, state,
                   situation, attacks_right=attacks_right, context_x=context_x,
                   context_y=context_y, position_engine=position_engine,
                   routine=routine)

    ChainDispatcher.discipline = staticmethod(discipline)
    ChainDispatcher.set_piece = staticmethod(set_piece)
    return lambda: (setattr(ChainDispatcher, "discipline", od),
                     setattr(ChainDispatcher, "set_piece", osp))


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

    F = FOULS
    # Per-MATCH, not per-foul. The first version of this script divided by
    # len(F) and labelled the result "/match", which reported cards as
    # 0.22/match against a real 3.5-4.5 and looked like a 20x defect. It was
    # a units bug in the probe, not a calibration defect in the engine. Read
    # the units off the denominator before believing any rate printed here.
    nm = max(n, 1)
    print("\n" + "=" * 74)
    print(f"FOULS: {len(F)} over {n} matches = {len(F)/nm:.1f}/match"
          f"   [real ~20-30]")

    def in_direct(f):
        return f["x"] > 78 if f["att_right"] else f["x"] < 27

    def in_third(f):
        return f["x"] > 63 if f["att_right"] else f["x"] < 42

    nd = sum(1 for f in F if in_direct(f))
    nt = sum(1 for f in F if in_third(f))
    npen = sum(1 for f in F if f["penalty"])
    ncard = sum(1 for f in F if f["card"])
    print(f"  in DIRECT RANGE of fouled team  {nd}"
          f"   ({nd/nm:.2f}/match)   <- would become direct FKs"
          f"   [real direct FKs ~2-4/match]")
    print(f"  in fouled attacking third       {nt}   ({nt/nm:.2f}/match)")
    print(f"  became a PENALTY                {npen}"
          f"   ({npen/nm:.2f}/match)   [real ~0.25-0.35/match]")
    print(f"  became a CARD                   {ncard}"
          f"   ({ncard/nm:.2f}/match)   [real ~3.5-4.5/match]")
    xs = [f["x"] for f in F]
    if xs:
        print(f"  x: min {min(xs):.1f} max {max(xs):.1f} mean {st.mean(xs):.1f}"
              f" median {st.median(xs):.1f}")
        print(f"  x deciles  "
              f"{dict(sorted(Counter(int(x // 10) * 10 for x in xs).items()))}")

    print("\n  NOTE the frame asymmetry that makes the count meaningful:")
    fa = [f for f in F if f["att_right"]]
    fb = [f for f in F if not f["att_right"]]
    print(f"    fouled team attacking RIGHT (home)  n={len(fa)}"
          f"  direct-range {sum(1 for f in fa if in_direct(f))}")
    print(f"    fouled team attacking LEFT  (away)  n={len(fb)}"
          f"  direct-range {sum(1 for f in fb if in_direct(f))}")

    print("\nCONTROL — existing free kicks (offside restarts):")
    O = OFFSIDE_FKS
    nom = max(len(O), 1)
    print(f"  free kicks: {len(O)} = {len(O)/max(n,1):.2f}/match")
    od = sum(1 for o in O if (o["x"] > 78 if o["att_right"] else o["x"] < 27))
    print(f"  in direct range                      {od}")
    ox = [o["x"] for o in O if o["x"] is not None]
    if ox:
        print(f"  x: min {min(ox):.1f} max {max(ox):.1f} mean {st.mean(ox):.1f}"
              f" median {st.median(ox):.1f}")
    print("\n  -> if fouls in direct range land near the real 2-4/match, the")
    print("     missing award is the whole story, and no rate needs inventing.")
    print("=" * 74)


if __name__ == "__main__":
    main()