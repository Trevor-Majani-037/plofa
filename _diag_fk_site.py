"""WHICH SET-PIECE CALL SITE PRODUCES THE FREE KICKS? (2026-10-01)

My first diagnosis was wrong and this records why. I assumed the home-frame
clamp at match_engine.py:5088 was turning every free kick into a cross. After
"fixing" it, direct free kicks were still 0 and the observed spot range fell
outside the band the fix produces — so the free kicks were coming from
somewhere else and the clamp was never the cause.

Two call sites in `_simulate_minute` award free kicks:

    4627  the OFFSIDE queue. Passes `pending_offside_fk_x` through RAW and
          UNCAMPED, and stamps `attacks_right=(fk_team == home_team)`.
    5105  the open-play raise. Clamps the spot, writes it back into
          `last_ball_x`, then passes `context_x=state.last_ball_x`.

Stack-frame attribution failed twice (getframe(1) is this probe;
__file__ comparison didn't hold), so this discriminates on a VALUE invariant
that holds at each site:

    4627 -> context_x != state.last_ball_x  (the ball was never moved to the
              offside spot; last_ball_x keeps its open-play value)
    5105 -> context_x == state.last_ball_x  (written back at 5090 before the
              call at 5096)

Observation-only. Run:  .venv\\Scripts\\python.exe _diag_fk_site.py [n]
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
REC = []


def _patch():
    orig = ChainDispatcher.set_piece

    def sp(minute, att_team, def_team, att_players, def_players, state,
           situation, attacks_right=True, context_x=None, context_y=None,
           position_engine=None, routine=None):
        res = orig(minute, att_team, def_team, att_players, def_players, state,
                   situation, attacks_right=attacks_right, context_x=context_x,
                   context_y=context_y, position_engine=position_engine,
                   routine=routine)
        types = [e.event_type for e in res.events]
        if situation in (SituationType.DIRECT_FREEKICK,
                         SituationType.CROSSED_FREEKICK):
            same = (context_x is not None
                    and abs(context_x - getattr(state, "last_ball_x", -1)) < 0.01)
            in_range = ((context_x > 78) if attacks_right
                        else (context_x < 27)) if context_x is not None else False
            REC.append({
                "site": "5105 open-play raise" if same else "4627 offside queue",
                "situation": str(getattr(situation, "value", situation)),
                "attacks_right": attacks_right,
                "x": context_x, "y": context_y,
                "in_range": bool(in_range),
                "emitted": ("direct" if EventType.FREEKICK_DIRECT in types
                            else ("cross" if EventType.FREEKICK_CROSS in types
                                  else "?")),
                "goal": res.goal_scored,
                "routine": None if routine is None else str(routine),
            })
        return res

    ChainDispatcher.set_piece = staticmethod(sp)
    return lambda: setattr(ChainDispatcher, "set_piece", orig)


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

    print("\n" + "=" * 74)
    print(f"free kicks attributed: {len(REC)}")
    print(f"  call site            {dict(Counter(r['site'] for r in REC))}")
    print(f"  attacks_right        {dict(Counter(str(r['attacks_right']) for r in REC))}")
    print(f"  emitted              {dict(Counter(r['emitted'] for r in REC))}")
    print(f"  IN direct range      {sum(1 for r in REC if r['in_range'])}")
    xs = [r["x"] for r in REC if r["x"] is not None]
    if xs:
        print(f"  x min {min(xs):.1f} max {max(xs):.1f} mean {st.mean(xs):.1f}"
              f" median {st.median(xs):.1f}")
        print(f"  x decile histogram   "
              f"{dict(sorted(Counter(int(x // 10) * 10 for x in xs).items()))}")
    print()
    for site in sorted({r["site"] for r in REC}):
        sub = [r for r in REC if r["site"] == site]
        sx = [r["x"] for r in sub if r["x"] is not None]
        print(f"  {site}: n={len(sub)}"
              f"  in_range={sum(1 for r in sub if r['in_range'])}"
              f"  emitted={dict(Counter(r['emitted'] for r in sub))}")
        if sx:
            print(f"      x min {min(sx):.1f} max {max(sx):.1f}"
                  f" mean {st.mean(sx):.1f}")
    print("=" * 74)


if __name__ == "__main__":
    main()