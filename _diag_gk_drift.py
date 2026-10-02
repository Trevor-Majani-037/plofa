"""WHY IS THE KEEPER AVERAGING x=26.6? (2026-10-01)

`tests.py::test_pass_network_positions_stay_realistic` asserts the home
keeper averages x < 25. It passes with the foul award disabled and fails
with it enabled, so the award is responsible — but the award only touches
the keeper through `_build_freekick_wall`, so the question is which of two
very different mechanisms actually moved him:

  (A) MY KEEPER PLACEMENT. The bisector line runs from the BALL to the goal
      centre, and I place the keeper 2.5 m along it. If a direct free kick
      is taken from deep, that line sweeps upfield and drags him out.

  (B) MATCH FLOW. The award adds ~22 free kicks a match. More dead balls in
      the attacking third means the home team spends more time defending
      high, and the ordinary shape engine (not my code) walks the keeper up
      to meet it.

These need opposite fixes, and (A) is a 3-line guard while (B) means the
award volume is wrong. So measure which one it is: instrument
`set_piece_place` and record every keeper move, then compare the keeper's
average x with the award on and off.

Run:  .venv\\Scripts\\python.exe _diag_gk_drift.py
"""
import io
import random
import sys
from collections import Counter
from datetime import date

from match_engine import (
    MatchConfig, MatchEngine, PlayingStyle, TeamProfile, TeamStyle, Intensity,
)
from position_engine import PositionEngine
from player_dna import SquadBuilder
from roster_loader import get_loader

XLSX = "PLOFA-2026-2027.xlsx"

GK_MOVES = []          # every set_piece_place call on a GK
_orig = PositionEngine.set_piece_place
AWARD = [True]


def _patch():
    def spp(self, player_name, x, y, minute):
        st = self.states.get(player_name)
        was_gk = bool(st and getattr(st, "position", "") == "GK")
        before = (st.current_x, st.current_y) if st else (None, None)
        _orig(self, player_name, x, y, minute)
        if was_gk:
            GK_MOVES.append({
                "name": player_name, "x": x, "y": y, "minute": minute,
                "from_x": before[0],
            })
    PositionEngine.set_piece_place = spp
    return lambda: setattr(PositionEngine, "set_piece_place", _orig)


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


def toggle_award(on):
    """Flip the foul->free kick award in match_engine, in memory only."""
    import match_engine
    src = io.open("match_engine.py", encoding="utf-8").read()
    needle = "if disc_result.foul_committed and not disc_result.penalty_won:"
    on_s = needle
    off_s = "if False and " + needle.split("if ", 1)[1]
    new = src.replace(on_s, off_s) if not on else src.replace(off_s, on_s)
    tmp = io.open("match_engine.py", "w", encoding="utf-8", newline="")
    tmp.write(new)
    tmp.close()


def run(n=1):
    restore = _patch()
    try:
        for i in range(n):
            random.seed(7000 + i)
            build_pair("Oxton", "Natrican").simulate()
            print(f"  [{i+1}/{n}] done", flush=True)
    finally:
        restore()


def main():
    print("=" * 74)
    print("A. WITH the foul award ON (shipped state)")
    GK_MOVES.clear()
    run(1)
    on_moves = list(GK_MOVES)
    xs = [m["x"] for m in on_moves if m["x"] is not None]
    print(f"  keeper placements by set_piece_place : {len(on_moves)}")
    if xs:
        print(f"  their x: min {min(xs):.1f}  max {max(xs):.1f}"
              f"  mean {sum(xs)/len(xs):.1f}")
        print(f"  x deciles: {dict(sorted(Counter(int(x//10)*10 for x in xs).items()))}")

    print("\nB. WITH the foul award OFF (keeper placement still live)")
    toggle_award(False)
    try:
        GK_MOVES.clear()
        run(1)
        off_moves = list(GK_MOVES)
    finally:
        toggle_award(True)
    xs2 = [m["x"] for m in off_moves if m["x"] is not None]
    print(f"  keeper placements by set_piece_place : {len(off_moves)}")
    if xs2:
        print(f"  their x: min {min(xs2):.1f}  max {max(xs2):.1f}"
              f"  mean {sum(xs2)/len(xs2):.1f}")

    print("\n  READ:")
    print("  If (A) is the cause, the award-ON placements are at HIGH x")
    print("  (upfield) and numerous — my bisector is dragging him out.")
    print("  If (B) is the cause, placements are few and near his own goal,")
    print("  and the drift is the ordinary shape engine responding to a")
    print("  match with ~22 more restarts in it.")
    print("=" * 74)


if __name__ == "__main__":
    main()
