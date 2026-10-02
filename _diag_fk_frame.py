"""WHY SO FEW FOULS LAND IN DIRECT RANGE? (2026-10-01)

The chain of measurements currently reads:

    24.5 fouls/match, of which 6.50 in "direct range"  (foul probe)
    33   free kicks/match, of which  5 in direct range  (same probe)
     6   free kicks at x > 78, but only 1 counted in direct range
     0   actually emit FREEKICK_DIRECT                      (chain probe)

Each of those was measured separately and they disagree, which means at
least one probe is applying a DIFFERENT definition of "direct range" to the
same free kicks. Two candidates:

  * `fk_x > 78 if attacks_right else fk_x < 27` — the chain's own test
    (event_chain.py:7275), which is direction-relative;
  * `x > 78` alone — a bare home-frame threshold.

The second is wrong for the away team and would double-count or miss. Rather
than guess which probe is right, this attributes every free kick to ONE
definition and prints the full distribution, so the direct-range count has a
single source of truth.

It also splits the misses by cause, because "6 at x>78 but 1 in direct
range" has three possible explanations and they need different fixes:
  * the away team's free kicks are being tested in the wrong frame;
  * the routine selector chose TRAINED_CROSS (legitimate football);
  * the award never queued them (a real bug).

Run:  .venv\\Scripts\\python.exe _diag_fk_frame.py [n]
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
from event_chain import ChainDispatcher
from player_dna import SquadBuilder
from roster_loader import get_loader

XLSX = "PLOFA-2026-2027.xlsx"
FKS = []


def _patch():
    osp = ChainDispatcher.set_piece

    def set_piece(minute, att_team, def_team, att_players, def_players,
                  state, situation, attacks_right=True, context_x=None,
                  context_y=None, position_engine=None, routine=None):
        res = osp(minute, att_team, def_team, att_players, def_players,
                  state, situation, attacks_right=attacks_right,
                  context_x=context_x, context_y=context_y,
                  position_engine=position_engine, routine=routine)
        if situation in (SituationType.DIRECT_FREEKICK,
                         SituationType.CROSSED_FREEKICK):
            types = [e.event_type.name for e in res.events]
            FKS.append({
                "x": context_x, "y": context_y,
                "att_right": bool(attacks_right),
                "team": att_team,
                "routine": routine.value if routine is not None else None,
                "emitted_direct": "FREEKICK_DIRECT" in types,
                "emitted_cross": "FREEKICK_CROSS" in types,
            })
        return res

    ChainDispatcher.set_piece = staticmethod(set_piece)
    return lambda: setattr(ChainDispatcher, "set_piece", osp)


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
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City")]
    try:
        for i in range(n):
            h, a = pairs[i % len(pairs)]
            random.seed(3000 + i)
            build_pair(h, a).simulate()
            print(f"  [{i+1}/{n}] done", flush=True)
    finally:
        restore()

    print("\n" + "=" * 74)
    print(f"FREE KICKS: {len(FKS)} over {n} matches"
          f" = {len(FKS)/max(n,1):.1f}/match")
    home = [f for f in FKS if f["att_right"]]
    away = [f for f in FKS if not f["att_right"]]
    print(f"  home (attacking right)  n={len(home)}")
    print(f"  away (attacking left)   n={len(away)}")

    # THE chain's own test, verbatim from event_chain.py:7275.
    def chain_direct(f):
        return f["x"] > 78 if f["att_right"] else f["x"] < 27

    # The naive home-frame test the other probe appears to use.
    def naive_direct(f):
        return f["x"] > 78

    nd_chain = sum(1 for f in FKS if chain_direct(f))
    nd_naive = sum(1 for f in FKS if naive_direct(f))
    print(f"\n  in direct range, chain's test (x>78 R / x<27 L)"
          f"  {nd_chain}  ({nd_chain/max(n,1):.2f}/match)")
    print(f"  in direct range, naive home-frame (x>78 only)      "
          f"  {nd_naive}  ({nd_naive/max(n,1):.2f}/match)")
    print("  -> if these differ, one of the two probes is measuring in the")
    print("     wrong frame for the away team.")

    # Distance to the goal the ATTACKING team is aiming at.
    print("\n  distance from the goal being attacked:")
    dists = []
    for f in FKS:
        gx = 105.0 if f["att_right"] else 0.0
        dists.append(abs(f["x"] - gx))
    if dists:
        print(f"    min {min(dists):.1f}  max {max(dists):.1f}"
              f"  mean {st.mean(dists):.1f}  median {st.median(dists):.1f}")
        for lo, hi, lbl in ((0, 18, "IN THE BOX (no direct FK)"),
                            (18, 25, "18-25 m (edge of range)"),
                            (25, 35, "25-35 m"),
                            (35, 60, "35-60 m"),
                            (60, 200, "60 m+ (defensive half)")):
            c = sum(1 for d in dists if lo <= d < hi)
            print(f"    {lbl:<26} {c:>4}  ({100.0*c/len(dists):5.1f}%)")

    print("\n  ROUTINE CHOSEN (this is the real gate on the direct branch):")
    for k, v in Counter(f["routine"] for f in FKS).most_common():
        print(f"    {str(k):<16} {v:>4}")
    dr = [f for f in FKS if chain_direct(f)]
    print(f"\n  the {len(dr)} in direct range, by routine:")
    for k, v in Counter(f["routine"] for f in dr).most_common():
        print(f"    {str(k):<16} {v:>4}")
    ed = sum(1 for f in dr if f["emitted_direct"])
    print(f"  of those, emitted FREEKICK_DIRECT : {ed}")
    print(f"  of those, emitted FREEKICK_CROSS  : "
          f"{sum(1 for f in dr if f['emitted_cross'])}")
    print("\n  READ: routine=trained_cross in direct range is CORRECT")
    print("  football (a rehearsed cross is a legitimate choice). The")
    print("  question is only whether the SHARE of direct_range free kicks")
    print("  is being crossed is plausible, and whether the in-box count is")
    print("  sane — a dead ball 4 m out is not a free kick at all.")
    print("=" * 74)


if __name__ == "__main__":
    main()
