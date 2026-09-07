"""Verify the VirtualGPS recorder against the match engine's internal
physics accumulation. Runs a full match with GPS enabled, then compares
per-player GPS-derived physical stats (distance, sprints, high-speed,
top speed) against the engine's own per-minute physics fields that already
feed opta_analytics (position_log physics_distance_m / physics_sprint_count
/ physics_high_speed_sprint_count / physics_top_speed_mps).

Also dumps the raw GPS log + summary to CSV in the temp output dir.
"""
import os
import sys
import random

OUT = os.path.join("output", "gps_verify")


def run(seed=4242):
    # Build a match exactly like _repro.run_one but with GPS enabled.
    import _repro as R
    from match_engine import (MatchEngine, MatchConfig, TeamProfile,
                              TeamStyle, PlayingStyle, Intensity)
    from player_dna import SquadBuilder

    random.seed(seed)
    home_squad = SquadBuilder.build(team_name=R.HOME_TEAM, starters=R.HOME_STARTERS,
                                    substitutes=R.HOME_SUBS,
                                    team_superstars=R.HOME_SUPERSTARS,
                                    set_piece_takers=R.HOME_SP_TAKERS)
    away_squad = SquadBuilder.build(team_name=R.AWAY_TEAM, starters=R.AWAY_STARTERS,
                                    substitutes=R.AWAY_SUBS,
                                    team_superstars=R.AWAY_SUPERSTARS,
                                    set_piece_takers=R.AWAY_SP_TAKERS)
    hpro = TeamProfile(name=R.HOME_TEAM, style=TeamStyle.ATTACKING,
                       playing_style=PlayingStyle.HIGH_PRESS, intensity=Intensity.HIGH)
    apro = TeamProfile(name=R.AWAY_TEAM, style=TeamStyle.FLUID_COUNTER,
                       playing_style=PlayingStyle.COUNTER, intensity=Intensity.MEDIUM)
    config = MatchConfig(home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM, matchday=1,
                         referee="Test", referee_strictness=0.0)
    eng = MatchEngine(config, hpro, apro)
    eng.quiet = True
    eng.enable_virtual_gps()
    eng.set_squad(R.HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    eng.set_squad(R.AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    res = eng.simulate()

    gps = res.gps
    os.makedirs(OUT, exist_ok=True)
    raw = gps.to_csv(os.path.join(OUT, f"gps_raw_m{seed}.csv"))
    summ = gps.to_summary_csv(os.path.join(OUT, f"gps_summary_m{seed}.csv"))

    # Build engine internal physics totals from position_log.
    engine = {}
    for frame in res.position_log:
        for side in ("home", "away"):
            for row in frame[side]:
                n = row["player"]
                e = engine.setdefault(n, {
                    "dist": 0.0, "sprint": 0, "hi": 0, "top": 0.0,
                })
                e["dist"] += row.get("physics_distance_m", 0.0)
                e["sprint"] += row.get("physics_sprint_count", 0.0)
                e["hi"] += row.get("physics_high_speed_sprint_count", 0.0)
                e["top"] = max(e["top"], row.get("physics_top_speed_mps", 0.0))

    print(f"Score: {res.home_goals}-{res.away_goals}")
    print(f"Raw GPS samples: {len(gps.samples)} "
          f"(~{len(gps.samples) / 22:.0f} per player, {len(gps.samples)/22/60:.1f} min)")
    print(f"{'Player':<22}{'GPS_dist_m':>12}{'Eng_dist_m':>12}{'GPS_spr':>8}"
          f"{'Eng_spr':>8}{'spd_diff':>9}\n" + "-" * 75)

    diffs = {"dist": [], "sprint": [], "hi": []}
    for name in sorted(engine):
        s = gps.player_summary(name)
        if s is None:
            continue
        e = engine[name]
        dd = s["distance_m"] - e["dist"]
        ds = s["sprint_count"] - e["sprint"]
        dh = s["high_speed_count"] - e["hi"]
        diffs["dist"].append(dd)
        diffs["sprint"].append(ds)
        diffs["hi"].append(dh)
        print(f"{name:<22}{s['distance_m']:>12.1f}{e['dist']:>12.1f}"
              f"{s['sprint_count']:>8}{e['sprint']:>8}{dd:>+9.1f}")

    print("\nMean |GPS - engine| per player:")
    import statistics
    for k, v in diffs.items():
        if v:
            print(f"  {k:<8}: mean {statistics.mean(abs(x) for x in v):.2f}, "
                  f"max {max(abs(x) for x in v):.2f}")
    print(f"\nWrote:\n  {raw}\n  {summ}")
    return res, gps


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4242
    run(seed)
