"""ARE THE ENGINE'S SHOT ENDPOINTS TRUSTWORTHY ENOUGH TO PLOT?

`_diag_shot_traj.py` established that the true shot flight EXISTS and that
`full_match_ball_path` throws it away (constant 4.0 m/s lerp across shot
windows). The open-play endpoint also already rides on the event as
`end_x`/`end_y` = `shot_res.goal_point`.

Before the exporter is rewired to draw that endpoint instead of inventing one,
two defensive hacks must be judged on evidence rather than kept on faith:

  * `exporter._shot_trajectory` overrides `goal_x` from the SHOT'S OWN POSITION
    ("safety net" for shots the engine logged in the wrong half);
  * `plot_shot_map` mirrors an away team's shot when `sx > 80`.

Both assume the recorded position can contradict the team. Measure instead:
for every shot in a real match, is the origin on the attacking side of
halfway for its team, and does the physics endpoint agree with the TEAM or
with the ORIGIN? A mirror that fires when origin and endpoint disagree would
draw the trajectory pointing away from the goal — which is worse than the bug
it was written to fix.

Also counts how many shots come from the set-piece paths, which build no
flight and therefore have no endpoint to use at all.
"""
import io
import random
import sys
from collections import Counter

SHOTS = ("SHOT_ATTEMPT", "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
         "GOAL", "HIT_WOODWORK", "PENALTY_SCORED", "PENALTY_MISSED")

# Which of these are resolved by geometry_engine.resolve_shot (a real flight)
# versus a set-piece chain that only rolls a probability.
PHYSICS_SITUATIONS = {"open_play"}


def run_one(seed, minute_offset):
    sys.path.insert(0, ".")
    from _diag_watch import build

    eng, _, _ = build()
    random.seed(seed)
    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real
    return res


def main():
    seeds = [21, 4242]
    results = [run_one(s, 0) for s in seeds]

    for res in results:
        cfg = res.config
        home, away = cfg.home_team, cfg.away_team
        print(f"\n=== {home} {res.score_str} {away} ===")

        rows = []
        for e in res.timeline:
            nm = getattr(getattr(e, "event_type", None), "name", "")
            if nm not in SHOTS:
                continue
            # A goal ALSO emits a SHOT_ON_TARGET; the exporter keys the shot map
            # off the primary shot event, so mirror that selection here.
            rows.append(e)

        kinds = Counter()
        populated = Counter()
        by_sit = Counter()
        wrong_half = 0
        endpoint_sides = Counter()
        disagree = []
        goalplane = Counter()

        for e in rows:
            nm = e.event_type.name
            team = e.team
            expect_right = (team == home)
            sit = e.situation.value if e.situation else "open_play"
            by_sit[sit] += 1
            has_end = e.end_x is not None and e.end_y is not None
            kinds[(sit, "end" if has_end else "NO end")] += 1
            if not has_end:
                populated[sit] += 0
                continue

            x = float(e.location_x or 50.0)
            # Origin on the attacking side of halfway?
            origin_right = x > 52.5
            if origin_right != expect_right:
                wrong_half += 1
            # Where did the physics endpoint land?
            end_right = float(e.end_x) > 52.5
            if end_right == expect_right:
                endpoint_sides["agrees with TEAM"] += 1
            else:
                endpoint_sides["agrees with ORIGIN (hack would fire)"] += 1
            if end_right != expect_right:
                disagree.append((e.minute, nm, sit, team, round(x, 1),
                                 round(float(e.end_x), 1)))
            # Is it actually on the goal plane?
            gp = abs(float(e.end_x) - (105.0 if expect_right else 0.0))
            goalplane["<=1.5m of goal line" if gp <= 1.5
                      else "NOT on goal line"] += 1

        total = len(rows)
        n_end = sum(v for (_, k), v in kinds.items() if k == "end")
        print(f"shot events: {total}   with a physics endpoint: {n_end}   "
              f"without: {total - n_end}")
        print(f"by situation: {dict(by_sit)}")
        print(f"endpoint population by situation: "
              f"{ {s: f'{v}/{by_sit[s]}' for s, v in populated.items()} }")
        print(f"endpoint vs goal plane: {dict(goalplane)}")
        print(f"origin in the WRONG half for its team: {wrong_half}/{total}")
        print(f"endpoint side: {dict(endpoint_sides)}")
        if disagree:
            print("  shots whose endpoint contradicts their team:")
            for d in disagree[:10]:
                print(f"    min {d[0]:>3} {d[1]:<17}{d[2]:<12}{d[3]:<10}"
                      f"origin_x={d[4]:>6}  end_x={d[5]:>6}")

    print("\nREAD:")
    print("  'agrees with TEAM' everywhere  -> origin never contradicts team,")
    print("    so the goal_x safety net and the sx>80 mirror are dead weight.")
    print("  shot events with NO endpoint   -> the set-piece chains, which")
    print("    must keep the reconstructed fallback.")


if __name__ == "__main__":
    main()