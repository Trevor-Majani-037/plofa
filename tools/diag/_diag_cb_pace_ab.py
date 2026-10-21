"""CB PACE A/B ACROSS SEEDS — does raising the centre-back jog actually move the
published distance, and does it break any other role?

`_diag_cb_pace.py` answered WHERE a centre-back's metres come from (one match,
branch-attributed). This answers the calibration question the project's own
rules demand a sweep for: one match is a direction, not a calibration.

   --arm off   pins `_JOG_SPEED["CB"]` back to the pre-fix 1.5 at RUNTIME
   --arm on    ships the current table
   --arm both  both, per seed (default)

The arm is a runtime dict write, not an edit, so the two arms differ in exactly
one value and nothing else. That matters: the pre-fix figure this replaces was
quoted from a *different match* in the note, which is not a controlled pair.

Reported per arm: published `distance_covered` per position, plus the
accounting invariants (published seconds vs the clock) so a pace change cannot
silently corrupt the measured-activity blend that `tests/test_measured_activity`
guards.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_cb_pace_ab.py --seeds 777,31,57
"""
import argparse
import contextlib
import io
import random
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, ".")

from match_engine import (  # noqa: E402
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from player_dna import SquadBuilder  # noqa: E402
import opta_analytics  # noqa: E402

HOME_STARTERS = [
    ("Keano Walsh", "GK", ["sweeper_keeper"], 29),
    ("Darius Frost", "LB", ["aggressive_fullback"], 24),
    ("Emeka Obi", "CB", ["ball_playing_cb"], 27),
    ("Tavish Crane", "CB", ["stopper_defender"], 30),
    ("Rico Alves", "RB", ["overlapping_fullback"], 25),
    ("Mateo Sanz", "CDM", ["anchor_man"], 28),
    ("Luca Ferrini", "CM", ["box_box"], 26),
    ("Jonas Keller", "CM", ["engine"], 25),
    ("Adriela Vela", "LW", ["dribbler"], 22),
    ("Dragan Novak", "ST", ["clinical_finisher"], 29),
    ("Percy", "RW", ["grand_dribbler"], 24),
]
AWAY_STARTERS = [
    ("Bram Osei", "GK", ["shot_stopper"], 28),
    ("Yuri Tanaka", "LB", ["aggressive_fullback"], 23),
    ("Marek Dubois", "CB", ["ball_playing_cb"], 26),
    ("Stefan Berg", "CB", ["stopper_defender"], 29),
    ("Owen Castillo", "RB", ["overlapping_fullback"], 24),
    ("Tariq Aziz", "CDM", ["anchor_man"], 27),
    ("Nils Werner", "CM", ["box_box"], 25),
    ("Diego Ramos", "CAM", ["creator"], 24),
    ("Leo Mbeki", "LW", ["speedster"], 21),
    ("Ivan Petrov", "ST", ["target_man"], 28),
    ("Samir Haddad", "RW", ["dribbler"], 23),
]

# CIES Football Observatory Monthly Report n°68 (Oct 2021), 7,855 matches,
# 31 leagues, SkillCorner tracking. NOTE the note in AGENTS.md claiming
# "real CB 10.5-11.0 km" is wrong: centre backs are the LEAST of any
# position at 9.2 km/match, midfielders the most at 10.6.
CIES = {"CB": 9.2, "CM": 10.6, "CDM": 10.6, "CAM": 10.6,
        "LB": 10.3, "RB": 10.3, "LW": 10.3, "RW": 10.3, "ST": 9.9}


def run(seed: int, cb_jog: float) -> dict:
    """One real match; returns published km per position + accounting checks."""
    MatchEngine._JOG_SPEED["CB"] = cb_jog
    MatchEngine._JOG_SPEED["DC"] = cb_jog
    MatchEngine._JOG_SPEED["LCB"] = cb_jog
    MatchEngine._JOG_SPEED["RCB"] = cb_jog

    random.seed(seed)
    home = SquadBuilder.build("Hartwell City", HOME_STARTERS)
    away = SquadBuilder.build("Thornfield United", AWAY_STARTERS)
    config = MatchConfig(home_team="Hartwell City", away_team="Thornfield United",
                         match_date=date(2026, 8, 22), matchday=2)
    hp = TeamProfile("Hartwell City", TeamStyle.TIKI_TAKA,
                     PlayingStyle.POSSESSION, Intensity.MEDIUM)
    ap = TeamProfile("Thornfield United", TeamStyle.BALANCED,
                     PlayingStyle.MIXED, Intensity.MEDIUM)
    engine = MatchEngine(config, hp, ap)
    engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
    engine.set_squad("Thornfield United", away["starters"], away["substitutes"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = engine.simulate()

    all_players = {"Hartwell City": {"starters": home["starters"],
                                     "substitutes": home["substitutes"]},
                   "Thornfield United": {"starters": away["starters"],
                                         "substitutes": away["substitutes"]}}
    analytics = opta_analytics.OptaAnalytics(result, all_players)
    analytics.compute()

    pos, secs, seconds_sum = {}, {}, {}
    for f in result.position_log:
        for side in ("home", "away"):
            for row in f.get(side, []):
                n = row["player"]
                pos[n] = row.get("position", "?")
                b = row.get("physics_standing_time_s", 0.0) \
                    + row.get("physics_walk_time_s", 0.0) \
                    + row.get("physics_jog_time_s", 0.0) \
                    + row.get("physics_running_time_s", 0.0) \
                    + row.get("physics_sprint_time_s", 0.0)
                secs[n] = secs.get(n, 0.0) + b
    for n, d in analytics.player_data.items():
        seconds_sum[n] = sum(d.get(f"{k}_seconds", 0.0) for k in
                             ("standing", "walking", "jogging",
                              "running", "sprinting"))

    per_pos = defaultdict(list)
    for n, d in analytics.player_data.items():
        per_pos[pos.get(n, "?")].append(d["distance_covered"])

    clock = len(result.position_log) * 60.0
    worst_sec = max(seconds_sum.values()) if seconds_sum else 0.0
    return {
        "per_pos": {k: v for k, v in per_pos.items()},
        "clock": clock,
        "worst_published_seconds": worst_sec,
        "events": len(result.timeline),
        "score": (result.state.home_goals, result.state.away_goals),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="777,31,57")
    ap.add_argument("--arm", choices=("off", "on", "both"), default="both")
    ap.add_argument("--off-jog", type=float, default=1.5,
                    help="the pre-fix centre-back jog speed")
    args = ap.parse_args()

    shipped = MatchEngine._JOG_SPEED["CB"]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = {"off": args.off_jog, "on": shipped}
    if args.arm != "both":
        arms = {args.arm: arms[args.arm]}

    print(f"CB jog: off={args.off_jog}  on={shipped}   seeds={seeds}")
    print()

    results = {name: [] for name in arms}
    for name, jog in arms.items():
        for seed in seeds:
            r = run(seed, jog)
            r["seed"] = seed
            results[name].append(r)
            s = r["score"]
            print(f"  {name:<4} seed {seed:<5} events {r['events']:<5} "
                  f"{s[0]}-{s[1]}   clock {r['clock']:.0f}s   "
                  f"max published seconds {r['worst_published_seconds']:.0f}")
    print()

    roles = ("GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST")
    print("=" * 84)
    print("PUBLISHED distance_covered (km), mean over seeds")
    print("=" * 84)
    hdr = f"{'pos':<5}" + "".join(f"{n:>9}" for n in arms)
    if len(arms) == 2:
        hdr += f"{'delta':>9}{'CIES':>8}{'err(on)':>9}"
    print(hdr)
    for p_ in roles:
        cells = ""
        means = {}
        for name in arms:
            vals = [x["per_pos"].get(p_, []) for x in results[name]]
            flat = [v for group in vals for v in group]
            m = sum(flat) / len(flat) if flat else float("nan")
            means[name] = m
            cells += f"{m:>9.2f}"
        if len(arms) == 2:
            d = means["on"] - means["off"]
            ref = CIES.get(p_)
            err = (f"{means['on'] - ref:>+9.2f}" if ref else f"{'--':>9}")
            cells += f"{d:>+9.2f}{(f'{ref:.1f}' if ref else '--'):>8}{err}"
        print(f"{p_:<5}{cells}")
    print()
    print("  'delta' is on minus off at identical seeds. 'err(on)' is against the")
    print("  CIES figure; wingers/full-backs share one 10.3 anchor there, so read")
    print("  the ordering, not the third decimal.")
    print()
    print("PER SEED, so it is visible which deltas are the subject and which are")
    print("trajectory divergence (the arms are DIFFERENT matches after the first")
    print("pace-driven position change; only CB has a mechanism behind its delta):")
    print(f"{'seed':<7}{'arm':<5}" + "".join(f"{p_:>8}" for p_ in roles))
    for name in arms:
        for r in results[name]:
            cells = ""
            for p_ in roles:
                v = r["per_pos"].get(p_)
                cells += f"{(sum(v) / len(v)):>8.2f}" if v else f"{'--':>8}"
            print(f"{r['seed']:<7}{name:<5}{cells}")
    print()
    spread = []
    for p_ in ("CB",):
        vals = [v for x in results.get("on", []) for v in x["per_pos"].get(p_, [])]
        if vals:
            spread.append(f"CB on: min {min(vals):.2f} mean "
                          f"{sum(vals) / len(vals):.2f} max {max(vals):.2f} "
                          f"(n={len(vals)})")
    for line in spread:
        print("  " + line)


if __name__ == "__main__":
    main()
