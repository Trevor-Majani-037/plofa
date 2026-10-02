"""
What does a PLOFA low block actually DO on the pitch?
===================================================

The post the user found describes a low block as a *shape*:

  * three lines inside your own defensive third — four, four and two
  * minimal distance between those lines, so there is nothing to play into
  * the block shifts as a RIGID UNIT to the ball side
  * only the nearest wide player presses; nobody else leaves the structure
  * the front pair stay HIGH and connected, so a regain is an immediate outlet

PLOFA has a `LOW_BLOCK_CONTAIN` press profile (intensity 0.25, engage line
nx=65) and a `BlockShapeType.LOW_BLOCK`. Those are press-*intensity* and
block-*shape* concepts. Neither, on its own, guarantees the three lines exist,
that the gaps between them are small, or that the front two stay high.

This measures it rather than assuming. Everything is expressed as
"metres from own goal", so it reads the same whichever way the team attacks.

Run:  python -m scripts.physics.probe_low_block
"""
from __future__ import annotations

import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, r"D:\PLOFA\plofa")

BACK_FOUR = ("CB", "LB", "RB", "LWB", "RWB")
MID_FOUR = ("CM", "CDM", "LM", "RM", "CAM", "LCM", "RCM", "LW", "RW")
FRONT_TWO = ("ST", "CF")

HASH_SEED = os.environ.get("PYTHONHASHSEED")


def _side_depth(players, attacks_right: bool):
    """Group a frame's players into lines, measured in metres from own goal."""
    lines = {"back": [], "mid": [], "front": [], "gk": []}
    for p in players:
        pos = (p.get("position") or "").upper()
        x = float(p.get("x", 0.0))
        depth = (105.0 - x) if attacks_right else x     # 0 = own goal line
        if pos == "GK":
            lines["gk"].append(depth)
        elif pos in FRONT_TWO:
            lines["front"].append(depth)
        elif pos in BACK_FOUR:
            lines["back"].append(depth)
        else:
            lines["mid"].append(depth)
    return {k: (statistics.mean(v) if v else None) for k, v in lines.items()}


def run(home_style, away_style, label, seed=20260928):
    import random
    from datetime import date

    from _repro import (AWAY_STARTERS, AWAY_SUBS, AWAY_TEAM, HOME_STARTERS,
                        HOME_SUBS, HOME_TEAM)
    from match_engine import (Intensity, MatchConfig, MatchEngine, PlayingStyle,
                              TeamProfile, TeamStyle)
    from player_dna import SquadBuilder
    from squad_manager import SubstitutionController

    def build(team, starters, subs, stars, takers):
        return SquadBuilder.build(team_name=team, starters=starters,
                                  substitutes=subs, team_superstars=stars,
                                  set_piece_takers=takers)

    hs = build(HOME_TEAM, HOME_STARTERS, HOME_SUBS, ["Percy", "Dragan Novak"],
               ["Percy", "Kofi Mensah"])
    as_ = build(AWAY_TEAM, AWAY_STARTERS, AWAY_SUBS, ["Kwame Asante"],
                ["Kwame Asante", "Bruno Reis"])

    random.seed(seed)
    hp = TeamProfile(name=HOME_TEAM, style=home_style,
                     playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
    ap = TeamProfile(name=AWAY_TEAM, style=away_style,
                     playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
    random.seed(seed)
    cfg = MatchConfig(home_team=HOME_TEAM, away_team=AWAY_TEAM,
                      match_date=date(2026, 9, 8))
    eng = MatchEngine(cfg, hp, ap)
    eng.quiet = True
    tracker = _TrackingLog(eng)
    eng.position_log = tracker
    eng._lb_phases = tracker.phases
    for team, sq in ((HOME_TEAM, hs), (AWAY_TEAM, as_)):
        eng.set_squad(team, sq["starters"], sq["substitutes"])
    bench = [x for sq in (hs, as_) for x in sq["substitutes"]]
    eng.set_stamina_controller(SubstitutionController(
        home_team=HOME_TEAM, away_team=AWAY_TEAM,
        home_subs_bench=bench, away_subs_bench=bench))
    res = eng.simulate()
    return res, eng


class _TrackingLog(list):
    """A ``position_log`` that also records who had the ball when each frame
    was taken.

    The frames themselves carry only positions, and averaging over all of them
    is meaningless for this question: a low block's back four push up when they
    have the ball, so a naive mean showed them 34.9 m from their own goal —
    a number that says "they are in the opponent's half" and means nothing
    about how they defend.

    A low block is defined while DEFENDING. So the phase has to be captured
    alongside the shape, or the measurement answers a different question.
    """

    def __init__(self, engine):
        super().__init__()
        self._engine = engine
        self.phases = []          # parallel to the frames

    def append(self, frame):            # noqa: D102
        st = self._engine.state
        self.phases.append((
            getattr(st, "possession_team", None),
            float(getattr(st, "last_ball_x", 52.5) or 52.5),
            float(getattr(st, "last_ball_y", 34.0) or 34.0),
        ))
        super().append(frame)


def _shape_stats(rows, attacks_right: bool):
    """Line depths in metres from own goal, plus the width of each line."""
    depth = defaultdict(list)
    width = defaultdict(list)
    for r in rows:
        pos = (r.get("position") or "").upper()
        x = float(r.get("x", 0.0))
        d = (105.0 - x) if attacks_right else x
        if pos == "GK":
            depth["gk"].append(d)
        elif pos in FRONT_TWO:
            depth["front"].append(d)
        elif pos in BACK_FOUR:
            depth["back"].append(d)
        else:
            depth["mid"].append(d)
        y = float(r.get("y", 0.0))
        bucket = ("back" if pos in BACK_FOUR else
                  "front" if pos in FRONT_TWO else
                  "mid" if pos != "GK" else "gk")
        width[bucket].append(y)
    mean_d = {k: (statistics.mean(v) if v else None) for k, v in depth.items()}
    span = {}
    for k, ys in width.items():
        if len(ys) > 1:
            span[k] = (max(ys) - min(ys), statistics.mean(ys))
    return mean_d, span


def report(res, team_name, attacks_right: bool, label: str, engine=None,
           side: str = "home"):
    frames = getattr(res, "position_log", None) or []
    phases = getattr(engine, "_lb_phases", None) or []
    if not frames:
        print(f"  {label}: no position_log")
        return
    side_key = next((k for k in frames[0] if k in ("home", "away")), None)
    if side_key is None:
        print(f"  {label}: frame keys are {list(frames[0])}")
        return
    # Which side of the pitch this team DEFENDS is decided by the side key,
    # not by matching the club name against it. An earlier version did
    # `team_name in str(side_key).lower()` — comparing "Hartwell City" with
    # "home", which is never true — so is_home came out False, the depth axis
    # was mirrored, and every number in the report was wrong while looking
    # entirely plausible. A measurement bug that produces confident nonsense
    # is worse than one that crashes.
    is_home = (side_key == side)
    ar = attacks_right if is_home else (not attacks_right)

    defending, attacking = [], []
    for i, frame in enumerate(frames):
        rows = frame.get(side_key) or []
        if not rows:
            continue
        phase = phases[i] if i < len(phases) else (None, 52.5, 34.0)
        # Ball position expressed relative to the team DEFENDING, so "wide" and
        # "central" mean the same thing whichever way they attack.
        bx, by = phase[1], phase[2]
        rel_x = (bx if not ar else 105.0 - bx)      # 0 = their own goal
        rel_y = (by if ar else 68.0 - by)
        central = abs(rel_y - 34.0) < 11.0 and rel_x > 40.0
        (attacking if phase[0] == team_name else defending).append(
            (rows, ar, central))

    for phase_name, entries in (("DEFENDING", defending), ("ATTACKING", attacking)):
        if not entries:
            continue
        print(f"    {phase_name}  ({len(entries)} frames)")
        for central_only, name in ((False, "all       "), (True, "ball central")):
            subset = [e for e in entries if e[2] == central_only] if central_only \
                else entries
            if not subset:
                continue
            all_rows = [r for rows, _a, _c in subset for r in rows]
            md, span = _shape_stats(all_rows, ar)
            gaps = defaultdict(list)
            for rows, attacks, _c in subset:
                d, _s = _shape_stats(rows, attacks)
                if d.get("back") is not None and d.get("mid") is not None:
                    gaps["bm"].append(d["mid"] - d["back"])
                if d.get("mid") is not None and d.get("front") is not None:
                    gaps["mf"].append(d["front"] - d["mid"])
            bit = ""
            if md.get("back") is not None:
                bit = (f"  |  block width {span.get('back', (0,))[0]:5.1f} m"
                       f"  back->mid {statistics.mean(gaps['bm']):4.1f} m"
                       f"  mid->front {statistics.mean(gaps['mf']):4.1f} m")
            print(f"      {name:<13} back {md.get('back', 0):5.1f}"
                  f"  mid {md.get('mid', 0):5.1f}"
                  f"  front {md.get('front', 0):5.1f} m from own goal{bit}")
        print()


def main() -> int:
    print("=" * 78)
    print("WHAT DOES A PLOFA LOW BLOCK ACTUALLY LOOK LIKE?")
    print("=" * 78)
    if HASH_SEED is None:
        print("  (PYTHONHASHSEED unset; these are single runs, not averages)")
    print()
    print("  The post's model, in metres from own goal on a 105 m pitch:")
    print("    back four   ~15-20 m    (inside own third)")
    print("    midfield    ~28-32 m    (just outside own third)")
    print("    front pair  ~30-36 m    (HIGH, and the outlet on a regain)")
    print("    gaps        under ~15 m between lines, so there is nothing to")
    print("                play into; and narrow enough that the middle is shut")
    print("")

    from match_engine import TeamStyle
    res, eng = run(TeamStyle.PARK_THE_BUS, TeamStyle.ATTACKING,
                   "park_the_bus", seed=20260928)
    report(res, "Hartwell City", True,
           "park_the_bus — DEFENDING is the shape that matters", engine=eng)

    print("  --- control: ATTACKING at both ends ---")
    res2, eng2 = run(TeamStyle.ATTACKING, TeamStyle.ATTACKING,
                     "attacking", seed=20260928)
    report(res2, "Hartwell City", True, "attacking (control)", engine=eng2)

    print("  The decisive column is 'ball central': a real low block NARROWS when")
    print("  the ball is central, denying the middle. If the two widths match,")
    print("  the team is not compact — it is just spread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
