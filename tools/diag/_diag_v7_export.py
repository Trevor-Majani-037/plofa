"""EXPORT one match with a given brain set — the "eye" half of the gate.

The plan's third acceptance criterion is *"three pass maps that read
vertical, with a reason for each arrow"*.  This probe runs ONE real match
with the given brain set and writes the full export (the workbook, the
pass-map PNG, the shot map), so the maps can be LOOKED at rather than
summarised.

It uses the harness of record's seeding and the identity perception regime,
so the match is the same one the numbers come from.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_v7_export.py brains_v7 \\
        --seed 21 --out plofa_output/_v7_gate
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "gates"))

import match_probe  # noqa: E402
from brain_integration import clear_registry  # noqa: E402
from perception import PerceptionConfig, get_perception_config, set_perception  # noqa: E402
from validate_neural_xl import register_full_xi  # noqa: E402

HOME = "Probe FC"
AWAY = "Rival FC"


def _run_with_controller(brains_dir: str, seed: int, home_style: str,
                         away_style: str):
    """The harness of record's match, PLUS a substitution controller.

    **The controller is not optional.**  The exporter's accumulator reads
    `minutes_played` from each player's DNA, and only the controller writes
    it — so without one, every player has 0 minutes, the pass-network plot's
    `team_stats` filter (`minutes_played > 0`) is empty, and the plot draws
    an EMPTY PITCH.  That is a defect in the probe, not in the export, and
    it is exactly the kind of thing a blank image invites you to
    misdiagnose.
    """
    from match_engine import MatchEngine
    from squad_manager import SubstitutionController

    saved_cfg = get_perception_config()
    set_perception(PerceptionConfig(enabled=False))
    try:
        home_squad, away_squad = match_probe._build_squads(HOME, AWAY)
        n_bound = register_full_xi(brains_dir, home_squad["starters"])
        config = match_probe.MatchConfig(
            home_team=HOME, away_team=AWAY,
            match_date=date(2026, 9, 6), matchday=3, season="26/27",
        )
        hp = match_probe._team_profile(HOME, home_style)
        ap = match_probe._team_profile(AWAY, away_style)
        eng = MatchEngine(config, hp, ap)
        eng.set_squad(HOME, home_squad["starters"], home_squad["substitutes"])
        eng.set_squad(AWAY, away_squad["starters"], away_squad["substitutes"])
        sub = SubstitutionController(
            home_team=HOME, away_team=AWAY,
            home_subs_bench=home_squad["substitutes"],
            away_subs_bench=away_squad["substitutes"],
            home_style=home_style, away_style=away_style,
        )
        eng.set_stamina_controller(sub)
        result = eng.simulate()
        return result, home_squad, away_squad, sub, n_bound
    finally:
        clear_registry()
        set_perception(saved_cfg)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("brains", nargs="?", default="brains_v7")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    p.add_argument("--out", default="plofa_output/_v7_gate")
    args = p.parse_args()
    os.environ.setdefault("PYTHONPATH", ROOT)

    random.seed(args.seed)
    result, home_squad, away_squad, sub, n_bound = _run_with_controller(
        args.brains, args.seed, args.home_style, args.away_style)
    print(result.summary())
    print(f"brains bound: {n_bound}")

    from exporter import PLOFAExporter

    all_players = {HOME: home_squad, AWAY: away_squad}
    out = args.out
    os.makedirs(out, exist_ok=True)
    exporter = PLOFAExporter(result=result, all_players=all_players,
                              home_color="#C8102E", away_color="#FFFFFF",
                              sub_controller=sub)
    exporter.export_all(out)
    print(f"\nexport written to {out}")
    for root, _dirs, files in os.walk(out):
        for f in sorted(files):
            print("   ", os.path.join(root, f).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
