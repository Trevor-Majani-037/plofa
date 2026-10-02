"""Count PLOFA event-type mix per team-match (pass/carry split, etc.)."""
import sys
import random
import contextlib
import io
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from match_engine import (MatchEngine, MatchConfig, TeamStyle, PlayingStyle, Intensity)
from player_dna import SquadBuilder
from squad_manager import SubstitutionController
from manager_profile import ManagerPool
from battlefield.run_plofa import _STYLE_PALETTE, _team_profile, _pick_clubs, _register_sub_schedule
from roster_loader import get_loader

loader = get_loader()
clubs = loader.get_all_clubs()
pairs = _pick_clubs(clubs, 2)
tot = Counter()
for i, (h, a) in enumerate(pairs, 1):
    random.seed(2026 + i * 1000)
    h_raw = loader.build_matchday_squad(h)
    a_raw = loader.build_matchday_squad(a)
    hs = _team_profile(h, i, True)
    as_ = _team_profile(a, i, False)
    h_sq = SquadBuilder.build(team_name=h, starters=h_raw["starters"],
                              substitutes=h_raw["substitutes"],
                              team_superstars=h_raw["superstars"],
                              set_piece_takers=h_raw["sp_takers"])
    a_sq = SquadBuilder.build(team_name=a, starters=a_raw["starters"],
                              substitutes=a_raw["substitutes"],
                              team_superstars=a_raw["superstars"],
                              set_piece_takers=a_raw["sp_takers"])
    cfg = MatchConfig(home_team=h, away_team=a, season="26/27",
                      competition="PLOFA-BATTLEFIELD", venue="BF Arena",
                      stadium_capacity=40000, referee="Battle Ref",
                      referee_strictness=0.5)
    mgr = ManagerPool(clubs=[h, a], style_lookup={h: hs.style.value, a: as_.style.value})
    sub = SubstitutionController(home_team=h, away_team=a,
                                 home_subs_bench=h_sq["substitutes"],
                                 away_subs_bench=a_sq["substitutes"],
                                 home_style=hs.style.value, away_style=as_.style.value,
                                 manager_stubbornness=0.35)
    sub.MAX_SUBS = 3
    _register_sub_schedule(sub, {h: h_sq, a: a_sq})
    eng = MatchEngine(cfg, hs, as_)
    eng.set_squad(h, h_sq["starters"], h_sq["substitutes"])
    eng.set_squad(a, a_sq["starters"], a_sq["substitutes"])
    eng.set_stamina_controller(sub)
    eng.set_managers(home_manager=mgr.manager_for(h), away_manager=mgr.manager_for(a))
    eng.quiet = True
    with contextlib.redirect_stdout(io.StringIO()):
        res = eng.simulate()
    for e in res.timeline:
        tot[getattr(getattr(e, "event_type", None), "name", "")] += 1
    print(f"[{i}] {res.state.home_goals}-{res.state.away_goals}")

n = 4
print("\n=== per team-match event counts ===")
for k, v in tot.most_common(30):
    print(f"  {k:<28}{v/n:8.1f}")