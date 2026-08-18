"""Deep-dive: where does the winger RECEIVE vs PASS from? (single match)"""
import sys
from collections import defaultdict
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tests import HOME_STARTERS, build_match
from match_engine import TeamStyle

def main():
    engine, home, away = build_match(
        home_style=TeamStyle.ATTACKING, away_style=TeamStyle.PARK_THE_BUS, seed=1,
    )
    result = engine.simulate()
    pe = engine.position_engine

    for wname in ("Adri Vela", "Percy"):
        st = pe.states[wname]
        anchor = st.home_y
        regions = defaultdict(lambda: defaultdict(int))
        origins = []
        touches = []
        prev = None
        for e in result.timeline:
            if getattr(e, "player", None) != wname:
                continue
            etn = getattr(getattr(e, "event_type", None), "name", "") or ""
            ox, oy = e.location_x, e.location_y
            touches.append((ox, oy))
            if etn in ("PASS", "THROUGH_BALL", "CROSS_ATTEMPT", "SHOT_ON_TARGET",
                       "SHOT_OFF_TARGET", "SHOT_BLOCKED", "GOAL"):
                dev = abs(oy - anchor)
                region = "final_third" if ox > 70 else ("mid_third" if ox > 35 else "own_third")
                regions[region][0 if dev <= 6 else (1 if dev <= 12 else (2 if dev <= 20 else 3))] += 1
                origins.append((ox, oy, dev, region))
        print(f"\n=== {wname} anchor_y={anchor} ===")
        print("Pass origins: final-third (x>70) vs mid (35-70) vs own (<35):")
        for region in ("final_third", "mid_third", "own_third"):
            buckets = regions[region]
            if buckets:
                print(f"  {region:<12} on-flank<=6m: {buckets.get(0,0):>3}  "
                      f"6-12m: {buckets.get(1,0):>3}  "
                      f"12-20m: {buckets.get(2,0):>3}  "
                      f"20m+ central: {buckets.get(3,0):>3}")
        # final-third pass origins only
        ft = [o for o in origins if o[0] > 70]
        if ft:
            print(f"  final-third pass origins: n={len(ft)} mean|lateral|={mean(o[2] for o in ft):.1f}m")
            for o in sorted(ft, key=lambda r: -r[2])[:8]:
                print(f"    x={o[0]:.0f} y={o[1]:.0f} dev={o[2]:.1f}")
        mt = [o for o in origins if 35 < o[0] <= 70]
        if mt:
            print(f"  mid-third pass origins:  n={len(mt)} mean|lateral|={mean(o[2] for o in mt):.1f}m")
            for o in sorted(mt, key=lambda r: -r[2])[:6]:
                print(f"    x={o[0]:.0f} y={o[1]:.0f} dev={o[2]:.1f}")

if __name__ == "__main__":
    main()
