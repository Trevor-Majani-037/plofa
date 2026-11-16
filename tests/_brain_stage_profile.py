"""Where does the 350 us go? Break `offball_press_prob` into its stages.

The whole call is ~293-391 us but the net is only 18 us. This pins the
blame on a specific stage so the cost is actionable rather than mystical.

Run: py -3 _brain_stage_profile.py
"""
import time
from small_game import _engine
import football_brain as fb
import offball_brain_wiring as obw
from perception import perceive_offball_actors
from ball_vision import perceive_ball
from brain_sensors import extract_offball_sensors

eng = _engine()
eng._initialize_simulation()

HOME = eng.config.home_team
AWAY = eng.config.away_team

# a CM -- one of the positions that actually owns a brain
target = None
for p in eng.active_players[HOME]:
    if getattr(p, "position", "").upper().startswith("CM"):
        target = p
        break
if target is None:
    target = eng.active_players[HOME][5]
pname = target.name
pos = "CM"

ball_x, ball_y = 52.5, 34.0
views = obw._build_views(eng)
team = HOME
rosters = eng.position_engine.team_rosters

rows = []


def bench(label, fn, n=1500):
    for _ in range(50):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    us = (time.perf_counter() - t0) / n * 1e6
    rows.append((label, us))
    print(f"{label:<52} {us:9.2f} us")


st = eng.position_engine.states[pname]
rx, ry = st.current_x, st.current_y
attacks_right = eng.position_engine.team_attacks_right.get(team, True)

teammates = [views[team][n] for n in rosters.get(team, []) if n in views.get(team, {})]
defenders = []
for ot, names in rosters.items():
    if ot == team:
        continue
    defenders.extend(views[ot][n] for n in names if n in views.get(ot, {}))

px, py, _seen, sigma = perceive_ball(
    eng, pname, views.get(team, {}).get(pname) or obw._View(pname, pos),
    attacks_right, ball_x, ball_y, 10.0)

g_tms, g_defs, g_eng = perceive_offball_actors(
    obw._View(pname, pos), rx, ry,
    teammates=teammates, defenders=defenders,
    position_engine=eng.position_engine, attacks_right=attacks_right)

print(f"player={pname} pos={pos}  teammates={len(teammates)} "
      f"defenders={len(defenders)}\n")

bench("1. _build_views (cached)", lambda: obw._build_views(eng))
bench("2. roster list comprehensions",
      lambda: [views[team][n] for n in rosters.get(team, []) if n in views.get(team, {})])
bench("3. perceive_ball", lambda: perceive_ball(
    eng, pname, views.get(team, {}).get(pname) or obw._View(pname, pos),
    attacks_right, ball_x, ball_y, 10.0))
bench("4. perceive_offball_actors  <-- suspects",
      lambda: perceive_offball_actors(
          obw._View(pname, pos), rx, ry,
          teammates=teammates, defenders=defenders,
          position_engine=eng.position_engine,
          attacks_right=attacks_right))
bench("5. extract_offball_sensors", lambda: extract_offball_sensors(
    obw._View(pname, pos), rx, ry, px, py,
    teammates=g_tms, defenders=g_defs, position_engine=g_eng,
    attacks_right=attacks_right, game_state=None,
    minute=10.0, score_diff=0, ball_sigma=sigma))

brain = obw._load_position_brain(pos)
if brain is not None:
    sensors = extract_offball_sensors(
        obw._View(pname, pos), rx, ry, px, py,
        teammates=g_tms, defenders=g_defs, position_engine=g_eng,
        attacks_right=attacks_right, game_state=None,
        minute=10.0, score_diff=0, ball_sigma=sigma)
    bench("6. brain.forward (the actual NN)", lambda: brain.forward(sensors))

bench("*** TOTAL end-to-end", lambda: obw.offball_press_prob(
    eng, pname, pos, team, ball_x, ball_y, 0.5, minute=10, score_diff=0))

print("\n--- SHARE OF THE END-TO-END CALL ---")
total = [u for l, u in rows if l.startswith("***")][0]
sum_stages = sum(u for l, u in rows if l[:1] in "123456")
for label, us in rows:
    if label.startswith("***"):
        continue
    print(f"{label:<52} {us/total*100:6.1f}%")
print(f"\nsum of stages = {sum_stages:.1f} us vs total {total:.1f} us")
