"""STYLE-CONDITIONED DEEP CIRCULATION — pass-chart A/B (2026-11-26).

The claim under test: possession TeamStyles (tiki_taka, structured_possession,
vertical_tiki_taka) should let CB/CDM/CM own the pass chart; direct and
transition styles must be untouched. This probe measures exactly that, per
team, with the catalog's REAL style (not a hardcoded BALANCED fixture).

  OFF (default): `python -X utf8 tools\\diag\\_diag_style_circulation.py`
  ON:            set PLOFA_DEEP_CIRC=1 first

For each team it reports completed pass-like events, the role-group split,
the CB pass zones (own/mid/final third), how many passes carry
`target_authority="phase"` (so a matrix-overwritten phase target cannot hide),
and the top passer. Scratch in-memory MatchConfig; no export, no file writes.

NOTE: `PlayingStyle` is inert at runtime in this engine; every live branch
uses `TeamStyle`, which is what this probe reads from TEAM_CATALOG.
"""
import os
import random
import sys
from collections import Counter
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from auto_run_match import TEAM_CATALOG
from roster_loader import get_loader
from player_dna import SquadBuilder
from match_engine import MatchEngine, MatchConfig, TeamProfile

XLSX = "PLOFA-2026-2027.xlsx"
PASS = {"PASS", "PROGRESSIVE_PASS", "SWITCH_OF_PLAY", "THROUGH_BALL",
        "GOAL_KICK", "CORNER_TAKEN", "FREEKICK_CROSS"}
PAIRS = [
    ("Lige-8", "Trendboys"),                     # tiki_taka vs route_one
    ("Play City", "Natrican"),                   # tiki_taka vs attacking
    ("Avada Zenith", "Tryox City"),              # attacking vs fluid_counter
]
# All three clubs exist in the live workbook; Hartwell/Thornfield are test-only
# squads and are NOT in TEAM_CATALOG's roster source.
SEEDS = [123, 223, 323]
GROUP = {"CB": "CB", "CDM": "CM/CDM", "CM": "CM/CDM", "LB": "FB", "RB": "FB",
         "LW": "Wide", "RW": "Wide", "ST": "ST/CAM", "CF": "ST/CAM",
         "CAM": "ST/CAM", "GK": "GK"}


def build(home, away):
    loader = get_loader(XLSX)
    hr, ar = loader.build_matchday_squad(home), loader.build_matchday_squad(away)
    hs = SquadBuilder.build(home, starters=hr["starters"],
                            substitutes=hr["substitutes"])
    aws = SquadBuilder.build(away, starters=ar["starters"],
                             substitutes=ar["substitutes"])
    cfg = MatchConfig(home_team=home, away_team=away,
                      match_date=date(2026, 8, 16), matchday=1,
                      venue="Scratch Stadium", stadium_capacity=45000)

    def prof(team):
        e = TEAM_CATALOG[team]
        return TeamProfile(name=team, style=e["style"],
                           playing_style=e["playing_style"], intensity=e["intensity"])

    eng = MatchEngine(cfg, prof(home), prof(away))
    eng.set_squad(home, hs["starters"], hs["substitutes"])
    eng.set_squad(away, aws["starters"], aws["substitutes"])
    pos = {}
    for p in (hs["starters"] + hs["substitutes"]
              + aws["starters"] + aws["substitutes"]):
        pos[p.name] = p.position
    return eng, pos


def measure(res, pos, team, home):
    total = 0
    groups, authority, cb_zones, per_player = (Counter(), Counter(),
                                               Counter(), Counter())
    for e in res.timeline:
        if e.team != team or getattr(e.event_type, "name", "") not in PASS:
            continue
        total += 1
        role = pos.get(e.player, "?")
        groups[GROUP.get(role, "Other")] += 1
        per_player[(e.player, role)] += 1
        authority[(e.metadata or {}).get("target_authority", "unknown")] += 1
        if role == "CB":
            x = e.location_x or 50
            nx = x if team == home else (105.0 - x)
            cb_zones["own" if nx < 35 else ("mid" if nx < 70 else "final")] += 1
    return {"total": total, "groups": groups, "authority": authority,
            "cb_zones": cb_zones, "per_player": per_player}


def main():
    print("PLOFA_DEEP_CIRC", bool(os.environ.get("PLOFA_DEEP_CIRC")),
          "seeds", SEEDS)
    for home, away in PAIRS:
        agg = {home: Counter(), away: Counter()}
        leaders = {home: Counter(), away: Counter()}
        for seed in SEEDS:
            random.seed(seed)
            eng, pos = build(home, away)
            res = eng.simulate()
            for team in (home, away):
                m = measure(res, pos, team, home)
                agg[team]["total"] += m["total"]
                for k, v in m["groups"].items():
                    agg[team]["g:" + k] += v
                for k, v in m["authority"].items():
                    agg[team]["a:" + k] += v
                for k, v in m["cb_zones"].items():
                    agg[team]["z:" + k] += v
                for key, v in m["per_player"].items():
                    leaders[team][key] += v
        print("\n%s [%s] vs %s [%s] — %d seeds, per-match means" % (
            home, TEAM_CATALOG[home]["style"].value,
            away, TEAM_CATALOG[away]["style"].value, len(SEEDS)))
        n = len(SEEDS)
        for team in (home, away):
            a = agg[team]
            lead = leaders[team].most_common(1)[0] if leaders[team] else (("-", "?"), 0)
            print("  %-16s passes=%5.1f CB=%5.1f CM/CDM=%5.1f FB=%5.1f Wide=%5.1f "
                  "ST/CAM=%5.1f CBzones=%.1f/%.1f/%.1f phaseAuth=%.1f top=%s(%s)" % (
                      team, a["total"] / n, a["g:CB"] / n, a["g:CM/CDM"] / n,
                      a["g:FB"] / n, a["g:Wide"] / n, a["g:ST/CAM"] / n,
                      a["z:own"] / n, a["z:mid"] / n, a["z:final"] / n,
                      a["a:phase"] / n, lead[0][0], lead[0][1]))


if __name__ == "__main__":
    main()
