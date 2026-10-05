"""DERIVE the StatsBomb coordinate convention from the reference file.

StatsBomb records every event in the frame of the team IN POSSESSION, always
attacking left to right. So a home-team pass and an away-team pass need
different transforms, and there is a further subtlety about whether `y` also
flips. Getting this wrong produces a file that looks perfect and plots
backwards — which is the same failure mode as `plot_ball_motion`'s
VerticalPitch, where a 3-2 with 2.82 away xG rendered as "the ball never leaves
one half".

So do not guess it. The file ships `pass.length` and `pass.angle` for all 996
passes, so the convention can be DERIVED: try each candidate and keep the one
that reproduces the provider's own numbers.

    PITCH 120 x 80. Candidate transforms, applied per event:
      none      x,               y
      flip_x    120 - x,         y
      flip_xy   120 - x,         80 - y
      swap      y,               x

`pass.angle` is signed in degrees, so it discriminates where `length` cannot.
"""
import collections
import json
import math
import pathlib
import sys

SRC = r"D:\Downloads ⬇️\Statsbomb Date\15956.json"


def cands(x, y):
    return {
        "none": (x, y),
        "flip_x": (120.0 - x, y),
        "flip_xy": (120.0 - x, 80.0 - y),
        "swap": (y, x),
    }


def loc2(v):
    """StatsBomb shot end_location can be [x, y, z]. Take the first two."""
    return float(v[0]), float(v[1])


def angle_conventions(dx, dy):
    """`pass.angle` is signed in degrees but is NOT atan2(dy, dx) mod 360 —
    every one of the four spatial transforms missed it by 68-113 degrees, which
    is roughly a quarter turn in every case. So the provider measures the angle
    from a different axis and/or in the opposite rotational sense. Rather than
    guess which, enumerate the plausible ones and let the data choose."""
    a = math.degrees(math.atan2(dy, dx)) % 360.0
    return {
        "ccw_from_+x": a,
        "cw_from_+x": (360.0 - a) % 360.0,
        "ccw_from_+y": (a + 90.0) % 360.0,
        "cw_from_+y": (a - 90.0) % 360.0,
        "cw_from_+y_mod": (90.0 - a) % 360.0,
    }


def circdiff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def main():
    p = pathlib.Path(SRC)
    evs = json.loads(p.read_text(encoding="utf-8"))
    passes = [e for e in evs
              if e["type"]["name"] == "Pass" and e.get("location")
              and e["pass"].get("end_location")
              and e["pass"].get("angle") is not None
              and e["pass"].get("length") is not None]

    # Split by acting team: the whole question is whether the flip is per-team.
    teams = collections.Counter(e["team"]["name"] for e in passes)
    print(f"passes with full geometry: {len(passes)}")
    print(f"acting teams: {dict(teams)}")
    print(f"pitch: 120 x 80\n")

    home = teams.most_common(1)[0][0]
    for team in teams:
        sub = [e for e in passes if e["team"]["name"] == team]
        print(f"--- acting team: {team}  (n={len(sub)}) ---")
        for name in ("none", "flip_x", "flip_xy", "swap"):
            dl, per = [], collections.defaultdict(list)
            for e in sub:
                x0, y0 = cands(*loc2(e["location"]))[name]
                x1, y1 = cands(*loc2(e["pass"]["end_location"]))[name]
                dx, dy = x1 - x0, y1 - y0
                dl.append(abs(math.hypot(dx, dy) - e["pass"]["length"]))
                for k, v in angle_conventions(dx, dy).items():
                    per[k].append(circdiff(v, e["pass"]["angle"]))
            best = min(per.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
            bd = sum(best[1]) / len(best[1])
            print(f"   {name:<8} len_err={sum(dl)/len(dl):.3f} m   "
                  f"best angle convention: {best[0]:<14} "
                  f"mean_err={bd:>7.3f} deg")
        print()

    print("VERDICT: the transform with ~0 error on BOTH columns is the one the")
    print("provider used, and whether it differs per team is the whole")
    print("question. Read the two blocks above together.")

    # Where do shots end up, under each convention? Shots should cluster at the
    # attacking end. An independent check that does not rely on angle at all.
    print("\nINDEPENDENT CHECK — shot end_location, by acting team")
    shots = [e for e in evs if e["type"]["name"] == "Shot"
             and e.get("location") and e["shot"].get("end_location")]
    for team in dict.fromkeys(e["team"]["name"] for e in shots):
        sub = [e for e in shots if e["team"]["name"] == team]
        print(f"  {team} (n={len(sub)}):")
        for name in ("none", "flip_x", "flip_xy"):
            xs = [cands(*loc2(e["shot"]["end_location"]))[name][0] for e in sub]
            print(f"     {name:<8} end x  min={min(xs):6.1f} "
                  f"median={sorted(xs)[len(xs)//2]:6.1f} max={max(xs):6.1f}")
    print("  (a team's shots should end NEAR x=120 under the right convention,")
    print("   because that team is attacking towards x=120 by definition)")

    # And goalkeepers: a keeper's action should sit near his OWN goal, which
    # under 'attacking left to right' is x~0 for whoever is defending.
    print("\nINDEPENDENT CHECK — Goal Keeper event locations")
    gks = [e for e in evs if e["type"]["name"] == "Goal Keeper"
           and e.get("location")]
    for team in dict.fromkeys(e["team"]["name"] for e in gks):
        sub = [e for e in gks if e["team"]["name"] == team]
        print(f"  {team} (n={len(sub)}):")
        for name in ("none", "flip_x", "flip_xy"):
            xs = [cands(*loc2(e["location"]))[name][0] for e in sub]
            print(f"     {name:<8} x  min={min(xs):6.1f} "
                  f"median={sorted(xs)[len(xs)//2]:6.1f} max={max(xs):6.1f}")


if __name__ == "__main__":
    main()
