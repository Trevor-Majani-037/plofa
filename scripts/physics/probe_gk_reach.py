"""
How often is the keeper gate actually reached?

``physics/measure_gate.py`` reported ``gk_considered = 2`` in a match containing
21 shots. Two possibilities, with very different consequences:

  1. ``GoalkeeperEngine.evaluate_save`` is only one of several shot-resolution
     paths, so the gate sees a small minority of shots.
  2. ``active_adapter()`` is None for most shots, so the gate is skipped.

Either way "the keeper gate is high value" would be an overstatement, so this
counts the truth directly: how many shots exist, how many reach
``evaluate_save``, and how many of those found an adapter waiting.

Run:  python physics/probe_gk_reach.py
"""
from __future__ import annotations

import collections
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
sys.path.insert(0, r"D:\PLOFA\plofa")

HITS: collections.Counter = collections.Counter()
_REAL = None


def main() -> int:
    from datetime import date

    from auto_run_match import _resolve_team_profile
    from event_chain import GoalkeeperEngine
    from match_engine import EventType, MatchConfig, MatchEngine
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    from squad_manager import SubstitutionController

    global _REAL
    _REAL = GoalkeeperEngine.evaluate_save

    def counting_save(*a, **kw):
        HITS["evaluate_save_calls"] += 1
        from physics.adapter import active_adapter
        if active_adapter() is not None:
            HITS["with_adapter"] += 1
        try:
            return _REAL(*a, **kw)
        finally:
            HITS["returned"] += 1

    GoalkeeperEngine.evaluate_save = staticmethod(counting_save)

    loader = get_loader()
    clubs = sorted(loader.get_all_clubs())
    home, away = clubs[0], clubs[1]
    raw = {c: loader.build_matchday_squad(c) for c in (home, away)}
    squads = {c: SquadBuilder.build(
        team_name=c, starters=raw[c]["starters"],
        substitutes=raw[c]["substitutes"],
        team_superstars=raw[c]["superstars"],
        set_piece_takers=raw[c]["sp_takers"]) for c in (home, away)}
    profs = {c: _resolve_team_profile(c, raw[c]["formation"], is_home=(c == home))
             for c in (home, away)}

    cfg = MatchConfig(home_team=home, away_team=away, match_date=date(2026, 9, 8),
                      physics_enabled=True)
    eng = MatchEngine(cfg, profs[home], profs[away])
    for c in (home, away):
        eng.set_squad(c, squads[c]["starters"], squads[c]["substitutes"])
    subs = [x for c in (home, away) for x in squads[c]["substitutes"]]
    eng.set_stamina_controller(SubstitutionController(
        home_team=home, away_team=away, home_subs_bench=subs, away_subs_bench=subs))
    result = eng.simulate()

    events = list(getattr(result, "timeline", None) or [])
    shot_types = {EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
                  EventType.SHOT_BLOCKED, EventType.GOAL}
    shots = [e for e in events if getattr(e, "event_type", None) in shot_types]

    print("=" * 72)
    print("HOW OFTEN IS THE KEEPER GATE REACHED?")
    print("=" * 72)
    print(f"  {home} v {away}   final {result.home_goals}-{result.away_goals}")
    print()
    counts = collections.Counter(
        getattr(e.event_type, "name", "?") for e in events)
    print("  shot-ish events in the timeline:")
    for name in ("SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED", "GOAL"):
        print(f"    {name:<18} {counts[name]:>4}")
    print(f"    {'TOTAL':<18} {len(shots):>4}")
    print()
    print("  evaluate_save() reach:")
    for key in ("evaluate_save_calls", "with_adapter", "returned"):
        print(f"    {key:<24} {HITS[key]:>4}")
    print()
    calls = HITS["evaluate_save_calls"]
    if shots and calls < len(shots):
        print(f"  !! {len(shots)} shot events but only {calls} calls to "
              f"evaluate_save.")
        print("     The keeper gate therefore sees only "
              f"{calls / len(shots) * 100:.0f}% of shots; the rest are "
              "resolved by other code paths.")
    elif not calls:
        print("  !! evaluate_save was never called. The gate is unreachable.")
    else:
        print(f"  every shot went through evaluate_save ({calls} calls)")
    print()
    from physics import adapter as A
    print("  gate counters actually recorded:")
    for k, v in sorted(A._GATE_COUNTERS.items()):
        print(f"    {k:<20} {v:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
