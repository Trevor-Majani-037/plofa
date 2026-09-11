import random
import run_match as R
from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder

_SHOT_NAMES = {
    "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "SHOT_SAVED",
    "GOAL", "OWN_GOAL", "PENALTY_SCORED", "PENALTY_MISSED", "HIT_WOODWORK",
}

def run_one(seed):
    random.seed(seed)
    home = SquadBuilder.build(team_name=R.HOME_TEAM, starters=R.HOME_STARTERS, substitutes=R.HOME_SUBS,
                              team_superstars=R.HOME_SUPERSTARS, set_piece_takers=R.HOME_SP_TAKERS)
    away = SquadBuilder.build(team_name=R.AWAY_TEAM, starters=R.AWAY_STARTERS, substitutes=R.AWAY_SUBS,
                              team_superstars=R.AWAY_SUPERSTARS, set_piece_takers=R.AWAY_SP_TAKERS)
    for p in (home["starters"]+home["substitutes"]+away["starters"]+away["substitutes"]):
        if p.name in R.SOUL_PLAYERS:
            p.dna.soul = R.SOUL_PLAYERS[p.name]
    cfg = MatchConfig(home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM, match_date=R.MATCH_DATE,
                      matchday=R.MATCHDAY, season=R.SEASON, competition=R.COMPETITION, venue=R.VENUE,
                      stadium_capacity=R.CAPACITY, referee=R.REFEREE, referee_strictness=R.STRICTNESS,
                      is_derby=R.IS_DERBY)
    eng = MatchEngine(cfg, R.HOME_STYLE, R.AWAY_STYLE)
    eng.set_squad(R.HOME_TEAM, home["starters"], home["substitutes"])
    eng.set_squad(R.AWAY_TEAM, away["starters"], away["substitutes"])
    res = eng.simulate()
    h_shots = a_shots = 0
    for e in res.timeline:
        if getattr(e.event_type, "name", "") in _SHOT_NAMES:
            if e.team == R.HOME_TEAM: h_shots += 1
            elif e.team == R.AWAY_TEAM: a_shots += 1
    return {
        "hg": res.home_goals, "ag": res.away_goals,
        "hxg": res.state.home_xg, "axg": res.state.away_xg,
        "hshots": h_shots, "ashots": a_shots,
    }

print(f"{'seed':>4} {'HG':>3} {'AG':>3} {'HxG':>5} {'AxG':>5} {'Hsh':>4} {'Ash':>4} {'xg/shot':>8} {'conv%':>6}")
allg = []
for s in [1,2,3,4]:
    d = run_one(s)
    allg.append(d["hg"]+d["ag"])
    xg_shot = (d["hxg"]+d["axg"]) / max(1, d["hshots"]+d["ashots"])
    conv = (d["hg"]+d["ag"]) / max(1, d["hshots"]+d["ashots"])
    print(f"{s:>4} {d['hg']:>3} {d['ag']:>3} {d['hxg']:>5.2f} {d['axg']:>5.2f} {d['hshots']:>4} {d['ashots']:>4} {xg_shot:>8.3f} {conv*100:>5.1f}%")
print(f"\navg goals/match = {sum(allg)/len(allg):.2f}")
