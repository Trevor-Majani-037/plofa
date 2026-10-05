"""Fixture A is failing. Is the DECISION wrong, or the EXECUTION?

The striker does not get in behind. Two very different bugs produce that
symptom and they need opposite fixes:

  * the run layer DECIDED 'hold' or 'box' when it should have decided
    'behind'  -> a decision-logic problem
  * the run layer DECIDED 'behind' and the player did not get there
    -> an execution problem (off-ball pace, arrival damping, the 0.30 blend)

So print the decision, the target, and the path. Guessing between those two is
exactly the mistake this harness was built to stop.
"""
import json

from small_game import Scenario, depth_series, play
from tests.test_small_game import HOME, _defenders, _by_position


def main():
    sc = Scenario("run_in_behind",
                  "ball 15 m behind the striker, two-man defensive line",
                  home=HOME, away=_defenders(), ball=(45.0, 34.0),
                  possessing="home", seconds=6.0, attacks_right=True)
    r = play(sc)
    eng = r["engine"]
    st = _by_position(eng, "Oxton", "ST")[0]

    print("INTENDED (what the engine told him to do):")
    print(" ", json.dumps(r["intended"].get(st, {})))
    print("  diagnostics:", r["intended_diag"])
    print("\nOBSERVED (geometric, from the RunTracker):")
    print(" ", json.dumps(r["observed"].get(st, {})))

    d = depth_series(r["traces"][st], True)
    print(f"\nST depth over 6 s, start 60.0, offside line 70.0, last man 80.0")
    print("  " + "  ".join(f"{v:.0f}" for v in d[::5]))

    # What did the run layer actually ask for? Re-derive the decision from the
    # same inputs the engine used, so the number is the engine's, not mine.
    ps = eng.position_engine
    tgt = ps.striker_run_targets(
        "Oxton", 45.0, 34.0, True, True, eng._avg_stamina("Oxton"),
        (("Oxton", 0, 3)))
    print(f"\nstriker_run_targets -> {tgt}")
    print(f"avg stamina: {eng._avg_stamina('Oxton'):.3f}")
    print(f"team_attacks_right: {ps.team_attacks_right}")
    st_obj = ps.states.get(st)
    print(f"ST state: current=({st_obj.current_x:.1f},{st_obj.current_y:.1f}) "
          f"home=({st_obj.home_x:.1f},{st_obj.home_y:.1f}) zone={st_obj.zone}")

    # and the offside geometry the layer should be reasoning about
    from striker_behavior import StrikerSpatialProfile
    print(f"\nStrikerSpatialProfile methods: "
          f"{[m for m in dir(StrikerSpatialProfile) if not m.startswith('_')]}")

    # What depth are the away defenders ACTUALLY at? If the scenario placement
    # did not take, every downstream number is meaningless.
    print("\naway defenders as the engine sees them:")
    for p in eng.active_players["Natrican"]:
        s = ps.states.get(p.name)
        if s is not None:
            print(f"  {p.position:<5} {p.name:<18} "
                  f"current=({s.current_x:6.1f},{s.current_y:5.1f})  "
                  f"home=({s.home_x:6.1f},{s.home_y:5.1f})")

    # The shape target the run layer has to fight. If this is behind the run
    # target, the 0.30 blend explains a striker who cannot get in behind.
    from position_engine import _ShapeShim  # noqa: F401
    print(f"\nST blended target check: run target vs shape target vs result")
    print(f"  run target x        : {tgt.get(st, (0, None, None, '?'))[1]}")
    print(f"  ST final x          : {r['final'][st][0]:.1f}")
    print(f"  ball x              : {sc.ball[0]:.1f}")


if __name__ == "__main__":
    main()
