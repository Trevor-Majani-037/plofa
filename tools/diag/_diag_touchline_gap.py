"""Why did test_out_of_possession_touchline_press_gated_by_press_intensity
drop below the 8 m gap? Compare the live press push with the home anchor
rate limit on and off."""
import sys

sys.path.insert(0, ".")

from player_dna import SquadBuilder
from position_engine import PositionEngine


def _profile(press_intensity):
    return type("Prof", (), {
        "defensive_line": 0.5,
        "width": 0.9,
        "tempo": 0.5,
        "directness": 0.5,
        "press_intensity": press_intensity,
    })


def _build(press):
    s = SquadBuilder.build("Test FC", [
        ("GK", "GK", []), ("CB1", "CB", []), ("CB2", "CB", []),
        ("LB", "LB", ["aggressive_fullback"]),
        ("RB", "RB", ["overlapping_fullback"]),
        ("CDM", "CDM", []), ("CM1", "CM", []), ("CM2", "CM", []),
        ("LW", "LW", ["dribbler", "speedster"]), ("ST", "ST", []),
        ("RW", "RW", ["inverted"]),
    ])
    pe = PositionEngine()
    pe.initialize_team("Test FC", s["starters"], _profile(press),
                       attacks_right=True)
    return pe


def report(tag, rate_m):
    PositionEngine.HOME_RATE_M = rate_m
    pe_hi = _build(0.9)
    pe_lo = _build(0.4)
    lb_hi, lb_lo = pe_hi.states["LB"], pe_lo.states["LB"]
    phase = type("P", (), {"value": "second_open"})()
    print(f"{tag} (home rate {rate_m} m/call)")
    for m in range(1, 4):
        pe_hi.drift_minute("Test FC", _profile(0.9), phase, minute=m,
                           in_possession=False, ball_x=70.0, ball_y=12.0)
        pe_lo.drift_minute("Test FC", _profile(0.4), phase, minute=m,
                           in_possession=False, ball_x=70.0, ball_y=12.0)
        print(f"  m{m} hi home {lb_hi.home_x:6.1f} cur {lb_hi.current_x:6.1f}"
              f" | lo home {lb_lo.home_x:6.1f} cur {lb_lo.current_x:6.1f}"
              f" | gap {lb_hi.current_x - lb_lo.current_x:5.1f}")
    print(f"  target hi {lb_hi.home_target_x} lo {lb_lo.home_target_x}")


report("as shipped", 8.0)
report("no rate limit", 1e6)
