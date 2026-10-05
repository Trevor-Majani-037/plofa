"""DOES THE FREE-KICK WALL ACTUALLY STAND UP? (2026-10-01)

The wall is built by `SetPieceChain._build_freekick_wall` with a BOUNDED
approach: a man is placed at 9.15 m from the ball only if he can cover the
distance at his own top speed inside `WALL_SETUP_S`. A defender 40 m away
cannot, so he is placed at the edge of what he can reach and the wall that
exists is NOT the wall that was ordered.

That is the honest design, but it is also a claim, and claims need measuring.
If every wall ends up 25 m from the ball and 3 men wide, then "the wall
exists" is true only in the sense that a list of names exists.

Measures, per real match:
  * direct free kicks taken
  * how many men the wall could actually place
  * the ACHIEVED distance from the ball to each wall man (vs the 9.15 m the
    Law requires) — this is the number that decides whether it is a wall
  * achieved lateral spread (is it a flat line, or a scattered handful?)
  * the resolution breakdown: goal / saved / blocked / off target
  * conversion vs real (~4-5% of direct free kicks are goals)

Observation-only; patches nothing. Run:
    .venv\\Scripts\\python.exe _diag_fk_wall.py [n]
"""
import math
import random
import statistics as st
import sys
from collections import Counter
from datetime import date

from match_engine import (
    EventType, SituationType, MatchConfig, MatchEngine, PlayingStyle,
    TeamProfile, TeamStyle, Intensity,
)
from event_chain import ChainDispatcher, SetPieceChain
from player_dna import SquadBuilder
from roster_loader import get_loader

XLSX = "PLOFA-2026-2027.xlsx"

WALLS = []      # one dict per wall actually built
SHOTS = []      # one dict per direct free kick resolved

_orig_wall = SetPieceChain._build_freekick_wall.__func__


def _patch():
    def wall(cls, fk_x, fk_y, def_players, attacks_right,
             position_engine, minute, gk):
        out = _orig_wall(cls, fk_x, fk_y, def_players, attacks_right,
                         position_engine, minute, gk)
        mps, gk_mp, reach = out
        dists, laterals = [], []
        goal_x = 105.0 if attacks_right else 0.0
        near_y = 30.34 if fk_y < 34 else 37.66
        dx, dy = goal_x - fk_x, near_y - fk_y
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        for m in mps:
            px_, py_ = m.position.x, m.position.y
            vx, vy = px_ - fk_x, py_ - fk_y
            dists.append(math.hypot(vx, vy))
            # component along the wall face (perpendicular to ball->near post)
            laterals.append(vx * (-uy) + vy * ux)
        WALLS.append({
            "men": len(mps),
            "dists": dists,
            "laterals": laterals,
            "fk_x": fk_x, "fk_y": fk_y,
            "att_right": attacks_right,
            "reach": list(reach),
        })
        return out

    SetPieceChain._build_freekick_wall = classmethod(wall)

    osp = ChainDispatcher.set_piece

    def set_piece(minute, att_team, def_team, att_players, def_players,
                  state, situation, attacks_right=True, context_x=None,
                  context_y=None, position_engine=None, routine=None):
        before = len(WALLS)
        res = osp(minute, att_team, def_team, att_players, def_players,
                  state, situation, attacks_right=attacks_right,
                  context_x=context_x, context_y=context_y,
                  position_engine=position_engine, routine=routine)
        if len(WALLS) > before:
            w = WALLS[-1]
            goal = any(e.event_type == EventType.GOAL for e in res.events)
            save = any(e.event_type == EventType.SAVE for e in res.events)
            ot = any(e.event_type == EventType.SHOT_OFF_TARGET
                     for e in res.events)
            blk = any(e.event_type == EventType.SHOT_BLOCKED
                      for e in res.events)
            sot = any(e.event_type == EventType.SHOT_ON_TARGET
                      for e in res.events)
            meta = {}
            for e in res.events:
                if e.event_type in (EventType.SHOT_ON_TARGET,
                                    EventType.SHOT_OFF_TARGET,
                                    EventType.SHOT_BLOCKED):
                    meta = e.metadata or {}
                    break
            reso = (meta.get("resolution")
                    or ("goal" if goal else "saved" if save
                        else "blocked" if blk else "off" if ot else "?"))
            SHOTS.append({
                "reso": reso, "goal": goal, "save": save, "on_target": sot,
                "xg": res.xg_generated, "wall_men": w["men"],
                "speed": meta.get("shot_speed_mps"),
                "wall": w,
            })
        return res

    ChainDispatcher.set_piece = staticmethod(set_piece)
    return lambda: (setattr(SetPieceChain, "_build_freekick_wall",
                            classmethod(_orig_wall)),
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
             ("Justice", "Triumpher")]
    try:
        for i in range(n):
            h, a = pairs[i % len(pairs)]
            random.seed(2000 + i)
            build_pair(h, a).simulate()
            print(f"  [{i+1}/{n}] {h} v {a}", flush=True)
    finally:
        restore()

    print("\n" + "=" * 74)
    print(f"A. WALL GEOMETRY  ({len(WALLS)} walls over {n} matches"
          f" = {len(WALLS)/max(n,1):.1f}/match)")
    if not WALLS:
        print("  no walls built")
        return
    men = [w["men"] for w in WALLS]
    print(f"  men placed per wall   mean {st.mean(men):.2f}  "
          f"min {min(men)}  max {max(men)}   [Law/real 4-5]")
    alld = [d for w in WALLS for d in w["dists"]]
    on = [d for d in alld if d <= 10.5]
    print(f"  achieved distance from ball:")
    print(f"    mean {st.mean(alld):.2f} m   median {st.median(alld):.2f} m"
          f"   max {max(alld):.2f} m")
    print(f"    within 10.5 m of the ball  {len(on)}/{len(alld)}"
          f"  ({100.0*len(on)/len(alld):.1f}%)   [a real wall: ~100%]")
    sp = [max(w["laterals"])-min(w["laterals"]) for w in WALLS
          if len(w["laterals"]) > 1]
    if sp:
        print(f"  lateral spread (wall width)  mean {st.mean(sp):.2f} m"
              f"  median {st.median(sp):.2f} m   [real 4-5 men ~2.0-2.5 m]")
        # Per-wall detail, because a mean hides WHICH wall is wrong. A wall
        # wider than n*spacing means at least one man never reached his slot
        # and is standing somewhere else entirely, which is the failure that
        # turns a screen into a scatter of bodies.
        for i, w in enumerate(WALLS, 1):
            exp = (len(w["laterals"]) - 1) * 0.55
            got = max(w["laterals"]) - min(w["laterals"])
            flag = "" if got <= exp + 0.6 else "   <-- NOT A WALL"
            print(f"    wall {i}: men {w['men']}  expected width {exp:.2f}"
                  f"  actual {got:.2f}"
                  f"  dists {['%.1f' % d for d in w['dists']]}{flag}")

    print(f"\nB. WHAT THE SHOT DID  ({len(SHOTS)} direct free kicks"
          f" = {len(SHOTS)/max(n,1):.1f}/match)   [real 2-4]")
    c = Counter(s["reso"] for s in SHOTS)
    tot = max(len(SHOTS), 1)
    for k, v in c.most_common():
        print(f"    {k:<10} {v:>4}  ({100.0*v/tot:5.1f}%)")
    goals = sum(1 for s in SHOTS if s["goal"])
    print(f"    GOALS      {goals:>4}  ({100.0*goals/tot:5.1f}%)"
          f"   [real direct-FK conversion ~4-5%]")
    on_t = sum(1 for s in SHOTS if s["on_target"] or s["goal"])
    print(f"    on target  {on_t:>4}  ({100.0*on_t/tot:5.1f}%)"
          f"   [real ~25-30%]")
    sps = [s["speed"] for s in SHOTS if s["speed"]]
    if sps:
        print(f"    shot speed  mean {st.mean(sps):.1f} m/s"
              f"  range {min(sps):.1f}-{max(sps):.1f}"
              f"   [real direct FK ~25-30 m/s]")
    xg = [s["xg"] for s in SHOTS if s["xg"]]
    if xg:
        print(f"    xG awarded  mean {st.mean(xg):.4f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
