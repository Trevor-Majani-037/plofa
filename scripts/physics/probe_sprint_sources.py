"""Count where sprint credits actually come from."""
from __future__ import annotations

import random
import sys
from datetime import date
from collections import Counter

sys.path.insert(0, r"D:\PLOFA\plofa")

HOME, AWAY = "Uditon", "Claw"

import match_engine as ME

_counts = Counter()
_orig = ME.MatchEngine._record_offball_distance


def _patched(self, name, moved, speed, top, duration_s):
    _counts["offball_calls"] += 1
    if moved <= 0:
        _counts["offball_standing"] += 1
    st = self.position_engine.states.get(name)
    before = st.physics_sprint_count if st is not None else 0.0
    before_hi = st.physics_high_speed_sprint_count if st is not None else 0.0
    _orig(self, name, moved, speed, top, duration_s)
    if st is not None:
        _counts["offball_sprint"] += st.physics_sprint_count - before
        _counts["offball_hi"] += st.physics_high_speed_sprint_count - before_hi


ME.MatchEngine._record_offball_distance = _patched

from auto_run_match import _resolve_team_profile
from player_dna import SquadBuilder
from roster_loader import get_loader
from squad_manager import SubstitutionController

random.seed(123)
loader = get_loader()
raw = {c: loader.build_matchday_squad(c) for c in (HOME, AWAY)}
squads = {c: SquadBuilder.build(
    team_name=c, starters=raw[c]["starters"], substitutes=raw[c]["substitutes"],
    team_superstars=raw[c]["superstars"], set_piece_takers=raw[c]["sp_takers"],
) for c in (HOME, AWAY)}
profs = {c: _resolve_team_profile(c, raw[c]["formation"], is_home=(c == HOME))
         for c in (HOME, AWAY)}

cfg = ME.MatchConfig(home_team=HOME, away_team=AWAY,
                     match_date=date(2026, 9, 8), physics_enabled=False)
eng = ME.MatchEngine(cfg, profs[HOME], profs[AWAY])
for c in (HOME, AWAY):
    eng.set_squad(c, squads[c]["starters"], squads[c]["substitutes"])
subs = [x for c in (HOME, AWAY) for x in squads[c]["substitutes"]]
eng.set_stamina_controller(SubstitutionController(
    home_team=HOME, away_team=AWAY, home_subs_bench=subs, away_subs_bench=subs))

# Instrument the two other writers.
_orig_rec = eng.position_engine.record_physics_distance
_by_caller = Counter()


def _rec(player_name, **kw):
    sc = float(kw.get("sprint_count", 0.0) or 0.0)
    hsc = float(kw.get("high_speed_sprint_count", 0.0) or 0.0)
    if sc or hsc:
        _by_caller["sprint"] += sc
        _by_caller["hi"] += hsc
        _by_caller["calls_with_sprint"] += 1
    _by_caller["total_calls"] += 1
    return _orig_rec(player_name, **kw)


eng.position_engine.record_physics_distance = _rec

res = eng.simulate()
tot = res.physics_totals()

print("offball calls        :", _counts["offball_calls"])
print("offball standing     :", _counts["offball_standing"])
print("offball sprints      :", _counts["offball_sprint"])
print("offball hi-sprints   :", _counts["offball_hi"])
print("record_physics calls :", _by_caller["total_calls"])
print("calls w/ sprint      :", _by_caller["calls_with_sprint"])
print("sprint credits via record_physics (all sources):", _by_caller["sprint"])
print("sprint credits in physics_totals (per-minute):", sum(a["sprint_count"] for a in tot.values()))
print()
top = sorted(tot.items(), key=lambda kv: -kv[1]["sprint_count"])[:8]
for n, a in top:
    print(f"  {n[:24]:<25}{a['distance_m'] / 1000:6.2f} km  {a['sprint_count']:5.0f} sprints")
