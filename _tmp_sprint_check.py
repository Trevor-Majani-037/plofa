import random, run_match as R
from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder

random.seed(7)
hs = SquadBuilder.build(team_name=R.HOME_TEAM, starters=R.HOME_STARTERS,
                        substitutes=R.HOME_SUBS, team_superstars=R.HOME_SUPERSTARS,
                        set_piece_takers=R.HOME_SP_TAKERS)
asq = SquadBuilder.build(team_name=R.AWAY_TEAM, starters=R.AWAY_STARTERS,
                         substitutes=R.AWAY_SUBS, team_superstars=R.AWAY_SUPERSTARS,
                         set_piece_takers=R.AWAY_SP_TAKERS)
for p in (hs["starters"]+hs["substitutes"]+asq["starters"]+asq["substitutes"]):
    if p.name in R.SOUL_PLAYERS:
        p.dna.soul = R.SOUL_PLAYERS[p.name]
cfg = MatchConfig(home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM)
eng = MatchEngine(cfg, R.HOME_STYLE, R.AWAY_STYLE)
eng.set_squad(R.HOME_TEAM, hs["starters"], hs["substitutes"])
eng.set_squad(R.AWAY_TEAM, asq["starters"], asq["substitutes"])
res = eng.simulate()

tot = res.physics_totals()
sprints = sorted((v["sprint_count"] for v in tot.values()))
hi = sorted((v["high_speed_sprint_count"] for v in tot.values()))
dists = sorted((v["distance_m"] for v in tot.values()))
print("score:", res.score_str, "| clock:", round(res.match_clock_s,1))
print("sprints  min/med/max:", sprints[0], "/", sprints[len(sprints)//2], "/", sprints[-1])
print("hi_sprint min/med/max:", hi[0], "/", hi[len(hi)//2], "/", hi[-1])
print("dist_km  min/med/max:", round(dists[0]/1000,2), "/", round(dists[len(dists)//2]/1000,2), "/", round(dists[-1]/1000,2))

# print Telbey Jion Masel specifically
for side in ("home","away"):
    for row in res.position_log[-1].get(side, []):
        if "Masel" in row["player"]:
            print(row["player"], "physics:", row.get("physics_distance_m"), row.get("physics_sprint_count"), row.get("physics_high_speed_sprint_count"))
