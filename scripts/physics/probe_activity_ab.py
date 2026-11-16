"""
MEASURE: what does MatchConfig.physics_enabled do to distance and sprints?
===========================================================================

Runs the SAME fixture twice in one process, seeded identically, once with the
continuous-physics layer off and once on, and prints per-player distance /
sprint totals from ``physics_totals()`` -- the 10 Hz integration store, not a
snapshot baseline.

Why this tool exists
--------------------
A report of ~18 km covered and 300+ sprints for a winger with the layer ON is
not a tuning disagreement, it is a defect: no player runs 18 km in 90 minutes.
This prints the store so the inflation can be attributed to a band (standing /
walk / jog / run / sprint) and to a player, rather than eyeballed.

Reference ranges (Di Salvo / FIFA GPS, outfield, 90 min):
    total distance   9-13 km   (wingers / box-to-box mid ~11-12.5)
    sprints          10-40
    high-speed (>8.5) 2-12

Run:  $env:PYTHONHASHSEED="0"; python -m scripts.physics.probe_activity_ab
"""
from __future__ import annotations

import random
import sys
from datetime import date

sys.path.insert(0, r"D:\PLOFA\plofa")

HOME = "Uditon"
AWAY = "Claw"
SEED = 123


def _build(enabled: bool):
    random.seed(SEED)

    from datetime import date as _d
    from auto_run_match import _resolve_team_profile
    from match_engine import MatchConfig, MatchEngine
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    from squad_manager import SubstitutionController

    loader = get_loader()
    raw = {c: loader.build_matchday_squad(c) for c in (HOME, AWAY)}
    squads = {c: SquadBuilder.build(
        team_name=c,
        starters=raw[c]["starters"],
        substitutes=raw[c]["substitutes"],
        team_superstars=raw[c]["superstars"],
        set_piece_takers=raw[c]["sp_takers"],
    ) for c in (HOME, AWAY)}
    profs = {c: _resolve_team_profile(c, raw[c]["formation"], is_home=(c == HOME))
             for c in (HOME, AWAY)}

    cfg = MatchConfig(home_team=HOME, away_team=AWAY,
                      match_date=_d(2026, 9, 8),
                      physics_enabled=enabled)
    eng = MatchEngine(cfg, profs[HOME], profs[AWAY])
    for c in (HOME, AWAY):
        eng.set_squad(c, squads[c]["starters"], squads[c]["substitutes"])
    subs = [x for c in (HOME, AWAY) for x in squads[c]["substitutes"]]
    eng.set_stamina_controller(SubstitutionController(
        home_team=HOME, away_team=AWAY,
        home_subs_bench=subs, away_subs_bench=subs))
    return eng


def _run(enabled: bool):
    eng = _build(enabled)
    res = eng.simulate()
    return eng, res


def _pos_of(eng, name: str) -> str:
    for team in (HOME, AWAY):
        for r in eng.position_engine.snapshot(team):
            if r["player"] == name:
                return r.get("position", "?")
    return "?"


def _team_of(eng, name: str) -> str:
    for team in (HOME, AWAY):
        if name in eng.position_engine.states:
            return team
    return "?"


def _report(label: str, eng, res):
    totals = res.physics_totals()
    rows = sorted(totals.items(), key=lambda kv: -kv[1]["distance_m"])
    print("=" * 96)
    print(f"{label}   score {res.home_goals}-{res.away_goals}")
    print("=" * 96)
    print(f"{'player':<24}{'pos':<5}{'km':>7}{'sprints':>9}{'hi>8.5':>8}"
          f"{'top m/s':>9}{'team':>6}")
    for name, agg in rows:
        print(f"{name[:23]:<24}{_pos_of(eng, name):<5}"
              f"{agg['distance_m'] / 1000.0:>7.2f}"
              f"{agg['sprint_count']:>9.0f}"
              f"{agg['high_speed_sprint_count']:>8.0f}"
              f"{agg['top_speed_mps']:>9.2f}"
              f"{_team_of(eng, name)[:5]:>6}")
    if rows:
        km = [a["distance_m"] / 1000.0 for _, a in rows]
        sp = [a["sprint_count"] for _, a in rows]
        print(f"{'':<24}{'':<5}{sum(km) / len(km):>7.2f}"
              f"{sum(sp):>9.0f}   <- mean km / total sprints")
    print()
    return rows


def main() -> int:
    print("fixture:", HOME, "v", AWAY, " seed:", SEED)
    print()
    off_eng, off_res = _run(False)
    off_rows = _report("PHYSICS OFF (MatchConfig.physics_enabled=False)", off_eng, off_res)

    on_eng, on_res = _run(True)
    on_rows = _report("PHYSICS ON  (MatchConfig.physics_enabled=True)", on_eng, on_res)

    print("=" * 96)
    print("DELTA per player (physics on - off)")
    print("=" * 96)
    print(f"{'player':<24}{'pos':<5}{'d_km':>8}{'d_spr':>8}{'d_top':>8}")
    off_map = dict(off_rows)
    for name, on in on_rows:
        off = off_map.get(name)
        if not off:
            continue
        print(f"{name[:23]:<24}{_pos_of(on_eng, name):<5}"
              f"{(on['distance_m'] - off['distance_m']) / 1000.0:>+8.2f}"
              f"{on['sprint_count'] - off['sprint_count']:>+8.0f}"
              f"{on['top_speed_mps'] - off['top_speed_mps']:>+8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
