"""WHY DID test_match_crosses_stamped_geometrically STOP PASSING? (2026-10-02)

The test asserts `any(e.metadata["cross"] for e in CROSS_ATTEMPT events)` —
i.e. at least one cross in a full wing-play match must be CONFIRMED by the
geometric detector. It passes with event_chain.py reverted and fails with the
chance-coordinate / delivery-origin / clamp_attack_x changes in.

Two candidate mechanisms, and they need different fixes:
  (a) STREAM SHIFT. The removed `random.uniform(5, 20)` was a real draw from
      the global football stream, so every later random number in the match
      moves. This project has been bitten by exactly that before.
  (b) A REAL regression — e.g. the away team's cross receiver is now placed at
      the correct end and that somehow suppresses cross selection.

Distinguish them: print every CROSS_ATTEMPT with the geometry the detector
actually saw. If the crosses exist and are wide-and-into-the-box, the verdict
is a stream artefact. If they are central passes being called crosses, or no
crosses at all, it is a real change in behaviour.

Run:
    .venv\\Scripts\\python.exe _diag_cross_regress.py [seed]
"""
import random
import sys
from datetime import date

from match_engine import (
    EventType, MatchConfig, MatchEngine, PlayingStyle,
    TeamProfile, TeamStyle, Intensity,
)
from player_dna import SquadBuilder


def build_engine(seed=11):
    random.seed(seed)
    home = SquadBuilder.build("Hartwell City", [
        ("HGK", "GK", [], 25), ("H1", "CB", [], 25), ("H2", "CB", [], 25),
        ("H3", "LB", [], 25), ("H4", "RB", [], 25), ("H5", "CDM", [], 25),
        ("H6", "CM", [], 25), ("H7", "CAM", [], 25), ("H8", "LW", [], 25),
        ("H9", "RW", [], 25), ("H10", "ST", [], 25),
    ])
    away = SquadBuilder.build("Away", [
        ("AGK", "GK", [], 25)] + [(f"A{i}", "CB", [], 25) for i in range(10)])
    config = MatchConfig(home_team="Hartwell City", away_team="Away",
                         match_date=date(2026, 8, 16))
    hs = TeamProfile("Hartwell City", TeamStyle.WING_PLAY,
                     PlayingStyle.HIGH_PRESS, Intensity.HIGH)
    as_ = TeamProfile("Away", TeamStyle.PARK_THE_BUS,
                      PlayingStyle.COUNTER, Intensity.MEDIUM)
    engine = MatchEngine(config, hs, as_)
    engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
    engine.set_squad("Away", away["starters"], away["substitutes"])
    return engine


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    res = build_engine(seed).simulate()
    tl = res.timeline
    crosses = [e for e in tl if e.event_type == EventType.CROSS_ATTEMPT]
    generic = [e for e in tl
               if e.event_type == EventType.PASS and (e.metadata or {}).get("cross")]

    print(f"\nseed {seed}: {len(tl)} events, "
          f"{len(crosses)} CROSS_ATTEMPT, {len(generic)} generic passes "
          f"reclassified as crosses")
    print(f"  confirmed CROSS_ATTEMPT : "
          f"{sum(1 for e in crosses if (e.metadata or {}).get('cross'))}/{len(crosses)}")
    print(f"  confirmed generic pass  : {len(generic)}")

    print("\n  every CROSS_ATTEMPT, with the geometry the detector saw:")
    for e in crosses[:24]:
        md = e.metadata or {}
        print(f"    min {e.minute:>3} {e.team:<15} {e.player:<8} "
              f"({e.location_x:5.1f},{e.location_y:4.1f})->"
              f"({(e.end_x or 0):5.1f},{(e.end_y or 0):4.1f})  "
              f"cross={md.get('cross')}  origin={md.get('cross_origin','')!r} "
              f"dest={md.get('cross_dest','')!r} "
              f"air={md.get('is_airborne')}")

    # The decisive comparison: how wide were the origins, and did any delivery
    # reach the box from wide?
    if crosses:
        ys = sorted(e.location_y for e in crosses)
        n = len(ys)
        print(f"\n  cross origin y: min {ys[0]:.1f}  median {ys[n//2]:.1f}  "
              f"max {ys[-1]:.1f}   [a real cross is struck from wide: y<20 or y>48]")
        wide = sum(1 for y in ys if y < 20 or y > 48)
        print(f"  struck from wide: {wide}/{n} ({100.0*wide/n:.0f}%)")

    # How many wide deliveries were attempted at all, regardless of label?
    wide_deliveries = [e for e in tl
                       if e.event_type in (EventType.CROSS_ATTEMPT,
                                           EventType.CROSS_SUCCESS,
                                           EventType.FREEKICK_CROSS)
                       and (e.location_y < 20 or e.location_y > 48)]
    print(f"  wide deliveries of any cross type: {len(wide_deliveries)}")


if __name__ == "__main__":
    main()