"""Feature-extraction cost per brain, measured on the REAL engine.

`offball_press_prob` returned in 0.5 us for a GK because there is no
GK brain -- it bails immediately. That is a fallthrough, not a measurement.
This version walks every position that HAS an evolved brain (LW/RW/CM) and
reports the honest end-to-end cost: sensors + forward.

Run: py -3 _brain_sensor_bench2.py
"""
import time

from small_game import _engine
import football_brain as fb
import offball_brain_wiring as obw

eng = _engine()
eng._initialize_simulation()

HOME = eng.config.home_team
AWAY = eng.config.away_team
print(f"engine: {HOME!r} vs {AWAY!r}")

# find one real player per position
by_pos = {}
for team in (HOME, AWAY):
    for p in eng.active_players[team]:
        st = eng.position_engine.states.get(p.name)
        pos = getattr(st, "position", "") if st else ""
        by_pos.setdefault(pos, (p.name, team))

print("positions on pitch:", sorted(by_pos))

ball_x, ball_y = 52.5, 34.0
RESULTS = []


def bench(label, fn, n=800):
    try:
        for _ in range(30):
            r = fn()
    except Exception as e:
        print(f"{label:<46} FAILED {type(e).__name__}: {e}")
        return None
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    us = (time.perf_counter() - t0) / n * 1e6
    RESULTS.append((label, us))
    print(f"{label:<46} {us:9.2f} us")
    return us


print("\n--- off-ball press brain, END TO END (sensors + forward) ---")
for pos in ("LW", "RW", "CM", "CB", "ST"):
    if pos not in by_pos:
        continue
    pname, team = by_pos[pos]
    bench(f"offball_press_prob [{pos}] {pname}",
          lambda pn=pname, t=team, p=pos: obw.offball_press_prob(
              eng, pn, p, t, ball_x, ball_y, 0.5,
              minute=10, score_diff=0))

print("\n--- shared unit brains (per TEAM-tick, not per player) ---")
tp = fb.TeamPressBrain.load("brains_team/XI.json")


def _team_press():
    return tp.forward(fb.extract_team_sensors(eng, HOME, ball_x, ball_y, 0.5))


bench("extract_team_sensors + TeamPress.forward", _team_press)
bench("extract_defensive_sensors (sensors only)",
      lambda: fb.extract_defensive_sensors(
          eng, AWAY, HOME, ball_x, ball_y,
          danger_level=50.0, ball_aerial=False,
          contest_x=ball_x, contest_y=ball_y,
          opponent_distance=3.0, press_occurred=False))

print("\n--- PROJECTION: ONE 90-MINUTE MATCH ---")
MATCH_S, PLAYERS = 5400.0, 22
print(f"{'unit of work':<46} {'1 Hz':>8} {'3 Hz':>8} {'10 Hz':>8}")
for label, us in RESULTS:
    cells = [PLAYERS * MATCH_S * hz * us / 1e6 for hz in (1, 3, 10)]
    print(f"{label:<46} {cells[0]:7.1f}s {cells[1]:7.1f}s {cells[2]:7.1f}s")
