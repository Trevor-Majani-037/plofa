"""EXACTLY WHY IS THE HOME KEEPER AVERAGING x=26.5? (2026-10-01)

Three isolations have now given three different stories, and they cannot all
be right:

  * award OFF, everything else live  -> PASSES
  * award ON, box fouls excluded      -> x = 27.61  (FAIL)
  * award ON, box + half excluded     -> x = 26.46  (FAIL)

Halving the restart count barely moved the number, yet removing the award
entirely fixes it. That is the signature of a NONLINEAR effect: the keeper is
not being dragged upfield a little at a time by each extra restart, he is
being moved by a small number of extreme events whose contribution does not
shrink when you halve the population.

So stop measuring the mean and measure the DISTRIBUTION. If a handful of
placements sit at x ~90-101, the mean is 26 for the same reason a single
loud note ruins a mix: the tail, not the body.

This records every position write to the keeper, whoever makes it, and
reports the tail separately from the body. The writers it distinguishes:
  * set_piece_place          — my wall/keeper code
  * record_touch             — the flank hold + GK box anchor
  * anything else            — the ordinary shape engine

Run:  .venv\\Scripts\\python.exe _diag_gk_hist.py
"""
import random
from collections import Counter
from datetime import date

from match_engine import (
    MatchConfig, MatchEngine, PlayingStyle, TeamProfile, TeamStyle, Intensity,
)
from position_engine import PositionEngine
from player_dna import SquadBuilder
from roster_loader import get_loader

XLSX = "PLOFA-2026-2027.xlsx"
WRITES = []      # (source, minute, x, y, name)

_sp = PositionEngine.set_piece_place
_rt = PositionEngine.record_touch


def _patch():
    def spp(self, player_name, x, y, minute):
        st = self.states.get(player_name)
        if st is not None and getattr(st, "position", "") == "GK":
            WRITES.append(("set_piece_place", minute, x, y, player_name))
        return _sp(self, player_name, x, y, minute)

    def rt(self, player_name, x, y, minute):
        st = self.states.get(player_name)
        if st is not None and getattr(st, "position", "") == "GK":
            WRITES.append(("record_touch", minute, x, y, player_name))
        return _rt(self, player_name, x, y, minute)

    PositionEngine.set_piece_place = spp
    PositionEngine.record_touch = rt
    return lambda: (setattr(PositionEngine, "set_piece_place", _sp),
                    setattr(PositionEngine, "record_touch", _rt))


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
    restore = _patch()
    try:
        random.seed(4242)
        eng = build_pair("Oxton", "Natrican")
        eng.simulate()
    finally:
        restore()

    print("=" * 74)
    print(f"GK position writes recorded: {len(WRITES)}")
    for src, n in Counter(w[0] for w in WRITES).most_common():
        print(f"  {src:<18} {n:>5}")
    if not WRITES:
        print("  none")
        return

    for src in sorted({w[0] for w in WRITES}):
        rows = [w for w in WRITES if w[0] == src]
        xs = [w[2] for w in rows if w[2] is not None]
        if not xs:
            continue
        body = [x for x in xs if x < 40]
        tail = [x for x in xs if x >= 40]
        print(f"\n  --- {src}  n={len(xs)} ---")
        print(f"    min {min(xs):.1f}  max {max(xs):.1f}"
              f"  mean {sum(xs)/len(xs):.1f}")
        print(f"    body (x<40)  n={len(body):>4}"
              f"  mean {(sum(body)/len(body) if body else 0):.1f}")
        print(f"    TAIL (x>=40) n={len(tail):>4}"
              f"  mean {(sum(tail)/len(tail) if tail else 0):.1f}")
        if tail:
            print(f"    tail values: {sorted(int(x) for x in tail)}")
            print(f"    tail writers at minutes: "
                  f"{[w[1] for w in rows if (w[2] or 0) >= 40]}")
    print("\n  READ: if one source has a TAIL at x>=40 while its body is")
    print("  normal, the mean is being set by a few extreme writes, and the")
    print("  fix is to bound THOSE, not to keep shrinking the population.")
    print("=" * 74)


if __name__ == "__main__":
    main()
