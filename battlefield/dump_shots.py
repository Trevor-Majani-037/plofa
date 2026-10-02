import sys, io, contextlib, random
sys.path.insert(0, '.')
from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder
from squad_manager import SubstitutionController
from manager_profile import ManagerPool
from battlefield.run_plofa import _team_profile, _pick_clubs, _register_sub_schedule
from roster_loader import get_loader

loader = get_loader(); clubs = loader.get_all_clubs()
random.seed(2026 + 3*1000)
h, a = _pick_clubs(clubs, 1)[0]
hr = loader.build_matchday_squad(h); ar = loader.build_matchday_squad(a)
hs = _team_profile(h, 1, True); as_ = _team_profile(a, 1, False)
hsq = SquadBuilder.build(team_name=h, starters=hr['starters'], substitutes=hr['substitutes'],
                         team_superstars=hr['superstars'], set_piece_takers=hr['sp_takers'])
asq = SquadBuilder.build(team_name=a, starters=ar['starters'], substitutes=ar['substitutes'],
                         team_superstars=ar['superstars'], set_piece_takers=ar['sp_takers'])
cfg = MatchConfig(home_team=h, away_team=a, season='26/27', competition='P', venue='V',
                  stadium_capacity=40000, referee='R', referee_strictness=0.5)
mgr = ManagerPool(clubs=[h, a], style_lookup={h: hs.style.value, a: as_.style.value})
sub = SubstitutionController(home_team=h, away_team=a, home_subs_bench=hsq['substitutes'],
                             away_subs_bench=asq['substitutes'], home_style=hs.style.value,
                             away_style=as_.style.value, manager_stubbornness=0.35)
sub.MAX_SUBS = 3
_register_sub_schedule(sub, {h: hsq, a: asq})
eng = MatchEngine(cfg, hs, as_)
eng.set_squad(h, hsq['starters'], hsq['substitutes'])
eng.set_squad(a, asq['starters'], asq['substitutes'])
eng.set_stamina_controller(sub)
eng.set_managers(home_manager=mgr.manager_for(h), away_manager=mgr.manager_for(a))
eng.quiet = True
with contextlib.redirect_stdout(io.StringIO()):
    res = eng.simulate()
print('home', h, 'away', a)
print('score', res.state.home_goals, '-', res.state.away_goals)
for e in res.timeline:
    n = getattr(getattr(e, 'event_type', None), 'name', '')
    if n.startswith('SHOT') or n.startswith('GOAL') or 'PENALTY' in n:
        print('{} team={} x={:.1f} y={:.1f} xg={:.3f}'.format(
            n, getattr(e, 'team', '')[:3], getattr(e, 'location_x', -1),
            getattr(e, 'location_y', -1), getattr(e, 'xg', 0.0)))