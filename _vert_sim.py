"""Quick full-match sim that reports verticality metrics from the timeline."""
import random, sys, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _repro as R
from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder

def build(seed=7, balanced=False):
    random.seed(seed)
    home_squad = SquadBuilder.build(team_name=R.HOME_TEAM, starters=R.HOME_STARTERS,
                                    substitutes=R.HOME_SUBS, team_superstars=R.HOME_SUPERSTARS,
                                    set_piece_takers=R.HOME_SP_TAKERS)
    away_squad = SquadBuilder.build(team_name=R.AWAY_TEAM, starters=R.AWAY_STARTERS,
                                    substitutes=R.AWAY_SUBS, team_superstars=R.AWAY_SUPERSTARS,
                                    set_piece_takers=R.AWAY_SP_TAKERS)
    if balanced:
        hpro = TeamProfile(name=R.HOME_TEAM, style=TeamStyle.BALANCED,
                           playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
        apro = TeamProfile(name=R.AWAY_TEAM, style=TeamStyle.BALANCED,
                           playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM)
    else:
        hpro = TeamProfile(name=R.HOME_TEAM, style=TeamStyle.ATTACKING,
                           playing_style=PlayingStyle.HIGH_PRESS, intensity=Intensity.HIGH)
        apro = TeamProfile(name=R.AWAY_TEAM, style=TeamStyle.FLUID_COUNTER,
                           playing_style=PlayingStyle.COUNTER, intensity=Intensity.MEDIUM)
    config = MatchConfig(home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM, matchday=1,
                         referee="Test", referee_strictness=0.0)
    eng = MatchEngine(config, hpro, apro)
    eng.quiet = True
    eng.set_squad(R.HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    eng.set_squad(R.AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    res = eng.simulate()
    return res

def metrics(res):
    tl = [e for e in getattr(res, "timeline", []) if hasattr(e, "event_type")]
    from match_engine import EventType
    per = {}
    for team in (R.HOME_TEAM, R.AWAY_TEAM):
        ar = (team == R.HOME_TEAM)
        completed = 0; ft3 = 0; box = 0; prog = 0; total_len = 0.0
        fwd = 0; lat = 0; back = 0
        for e in tl:
            if getattr(e, "team", None) != team:
                continue
            t = e.event_type
            if t not in (EventType.PASS, EventType.PROGRESSIVE_PASS,
                         EventType.SWITCH_OF_PLAY, EventType.THROUGH_BALL):
                continue
            if not e.outcome:
                continue
            x = e.location_x; y = e.location_y
            ex = e.end_x; ey = e.end_y
            if ex is None or ey is None:
                continue
            completed += 1
            total_len += math.hypot(ex - x, ey - y)
            dx = (ex - x) if ar else (x - ex)
            nx = ex if ar else (105.0 - ex)
            if nx >= 70: ft3 += 1
            if nx >= 88.5: box += 1
            if dx >= 9.0: prog += 1
            if dx > 2.0: fwd += 1
            elif dx < -2.0: back += 1
            else: lat += 1
        per[team] = dict(completed=completed, ft3=ft3, box=box, prog=prog,
                         avg_len=round(total_len/max(1,completed),1),
                         fwd_pct=round(100*fwd/max(1,completed),1),
                         lat_pct=round(100*lat/max(1,completed),1),
                         back_pct=round(100*back/max(1,completed),1))
    return per

if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [7]
    balanced = len(sys.argv) > 1 and sys.argv[-1] == "b"
    for seed in seeds:
        res = build(seed=seed, balanced=balanced)
        for team, m in metrics(res).items():
            print(f"seed={seed} {team:<16} completed={m['completed']:>3} into_final_third={m['ft3']:>3} "
                  f"into_box={m['box']:>2} prog9m={m['prog']:>3} avg_len={m['avg_len']:>4} "
                  f"fwd%={m['fwd_pct']:>3} lat%={m['lat_pct']:>3} back%={m['back_pct']:>3}")
