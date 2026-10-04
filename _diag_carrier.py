"""Why is ball_carrier empty? Debug the possession hand-off."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datetime import date
from event_chain import ChainDispatcher
from match_engine import (MatchConfig, MatchEngine, TeamProfile, TeamStyle,
                          PlayingStyle, Intensity)
from player_dna import SquadBuilder


def build():
    home = SquadBuilder.build("Home", [
        ("GK", "GK", []), ("CB1", "CB", []), ("CB2", "CB", []), ("LB", "LB", []),
        ("RB", "RB", []), ("CDM", "CDM", ["anchor_man"]), ("CM1", "CM", []),
        ("CM2", "CM", []), ("LW", "LW", ["dribbler"]), ("ST", "ST", []),
        ("RW", "RW", ["grand_dribbler"]),
    ])
    away = SquadBuilder.build("Away", [
        ("AGK", "GK", []), ("A0", "CB", []), ("A1", "CB", []), ("A2", "CB", []),
        ("A3", "CB", []), ("A4", "CB", []), ("A5", "CM", []), ("A6", "CM", []),
        ("A7", "CM", []), ("A8", "ST", []), ("A9", "ST", []),
    ])
    cfg = MatchConfig(home_team="Home", away_team="Away",
                      match_date=date(2026, 8, 16), matchday=1)
    hp = TeamProfile("Home", TeamStyle.ROUTE_ONE, PlayingStyle.DIRECT,
                     Intensity.HIGH)
    ap = TeamProfile("Away", TeamStyle.TIKI_TAKA, PlayingStyle.POSSESSION,
                     Intensity.MEDIUM)
    eng = MatchEngine(cfg, hp, ap)
    eng.set_squad("Home", home["starters"])
    eng.set_squad("Away", away["starters"])
    return eng


import random
lost = 0
empty = 0
total = 0
for seed in range(20):
    for n in (2, 4, 6):
        random.seed(seed * 10 + n)
        eng = build()
        res = ChainDispatcher.possession(
            30, "Home", eng.active_players["Home"], eng.home_profile,
            eng.state, n, position_engine=eng.position_engine,
            context_x=62.0, context_y=30.0,
        )
        total += 1
        lost += bool(res.possession_lost)
        if not res.ball_carrier:
            empty += 1
            if empty <= 4:
                kinds = [e.event_type.name for e in res.events]
                last = res.events[-1] if res.events else None
                print(f"seed {seed} n={n}: possession_lost={res.possession_lost} "
                      f"restart={getattr(res,'restart_type','')!r} "
                      f"carrier={res.ball_carrier!r}")
                print(f"   events: {kinds}")
                if last is not None:
                    print(f"   last: {last.event_type.name} {last.player} "
                          f"@({last.location_x},{last.location_y}) "
                          f"2nd={last.secondary_player}")

print(f"\n{total} possessions: possession_lost {lost}, empty carrier {empty}")
