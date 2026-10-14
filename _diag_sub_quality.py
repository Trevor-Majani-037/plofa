"""
DOES THE NEW SUBSTITUTION LOGIC ACTUALLY DO ANYTHING ON THE PITCH?
===============================================================

The unit tests prove the rule is implemented. They cannot prove it fires in
a match, or that the man it picks is the better man. This probe runs ONE
real match and reports what the controller actually decided.

Deliberately one match. The selection change shifts the RNG stream, so two
arms would be two different matches rather than a controlled comparison
(that has been established in this project's notes repeatedly: `_attr` draws
`random.uniform`, so a naive "pre-fix" arm measures STREAM SHIFT, not the
mechanism). The honest instrument here is the substitution LOG, which is
per-decision evidence rather than a match-level aggregate.

Run:  .venv\\Scripts\\python.exe _diag_sub_quality.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import random
    from datetime import date
    from match_engine import (
        MatchConfig, MatchEngine, TeamProfile, TeamStyle,
        PlayingStyle, Intensity,
    )
    from player_dna import SquadBuilder
    from squad_manager import SubstitutionController

    SEED = 4242

    XI = [("GK", "GK", []), ("CB1", "CB", []), ("CB2", "CB", []),
          ("LB", "LB", []), ("RB", "RB", []), ("CDM", "CDM", []),
          ("CM1", "CM", []), ("CM2", "CM", []), ("CAM", "CAM", []),
          ("LW", "LW", []), ("ST", "ST", []), ("RW", "RW", [])]
    XI = [XI[0], XI[1], XI[2], XI[3], XI[4], XI[5], XI[6], XI[7], XI[8],
          XI[9], XI[10]]          # 11 — CM2 is on the bench, not in the XI
    assert len(XI) == 11, f"the XI has {len(XI)} entries"
    # A real bench: seven men, the positions a modern squad actually carries.
    BENCH = [("B1", "GK", []), ("B2", "CB", []), ("B3", "LB", []),
             ("B4", "RB", []), ("B5", "CDM", []), ("B6", "CM", []),
             ("B7", "CAM", []), ("B8", "ST", [])]

    def _squad(tag, xi, bench):
        # `SquadBuilder.build` takes starters and substitutes as SEPARATE
        # lists — passing one combined list fails the "exactly 11 starters"
        # check at player_dna.py:1768.
        entries = [(f"{tag}{n}", p, s) for n, p, s in xi]
        subs = [(f"{tag}{n}", p, s) for n, p, s in bench]
        return SquadBuilder.build(tag, entries, subs)

    random.seed(SEED)
    home = _squad("H", XI, BENCH)
    away = _squad("A", XI, BENCH)

    config = MatchConfig(home_team="Hartwell City", away_team="Thornfield",
                         match_date=date(2026, 8, 16), matchday=1)
    hs = TeamProfile("Hartwell City", TeamStyle.ATTACKING,
                     PlayingStyle.HIGH_PRESS, Intensity.HIGH)
    as_ = TeamProfile("Thornfield", TeamStyle.FLUID_COUNTER,
                      PlayingStyle.COUNTER, Intensity.MEDIUM)
    engine = MatchEngine(config, hs, as_)
    engine.set_squad("Hartwell City", home["starters"], home["substitutes"])
    engine.set_squad("Thornfield", away["starters"], away["substitutes"])

    ctrl = SubstitutionController(
        home_team="Hartwell City", away_team="Thornfield",
        home_subs_bench=home["substitutes"], away_subs_bench=away["substitutes"],
    )
    ctrl.MAX_SUBS = 5
    engine.set_stamina_controller(ctrl)

    result = engine.simulate()

    subs = list(getattr(ctrl, "subs_log", []))

    print("=" * 78)
    print(f"REAL MATCH — seed {SEED}: Hartwell City v Thornfield")
    print("=" * 78)
    print(f"  events: {len(result.timeline)}   score: "
          f"{result.home_goals}-{result.away_goals}")

    hth = sum(1 for s in subs if s.get("team") == "Hartwell City")
    ttu = len(subs) - hth
    print(f"\n  substitutions made : {len(subs)} total"
          f"   ({hth} Hartwell, {ttu} Thornfield)")

    if ctrl is not None:
        for team, win in (ctrl.windows_used or {}).items():
            used = ctrl.subs_made.get(team, 0)
            print(f"  {team:<20} {used} subs in {win} windows"
                  f"   (limit {ctrl.MAX_SUBS}/{ctrl.MAX_WINDOWS})")

    reasons = {}
    for s in subs:
        reasons[s.get("reason", "?")] = reasons.get(s.get("reason", "?"), 0) + 1
    if reasons:
        print(f"\n  by reason           : {reasons}")

    if not subs:
        print("\n  NO SUBSTITUTIONS FIRED. That is the finding: the logic is")
        print("  implemented and unit-tested but never reached in a match,")
        print("  which is the exact failure mode this project keeps hitting.")
        return 1

    print("\n  --- every decision, with the quality it traded ---")
    hdr = (f"  {'min':>4} {'team':<9} {'off':<16} {'on':<16} "
           f"{'Qoff':>6} {'Qon':>6} {'dAtk':>7} {'dDef':>7} {'why':<12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    downgrades = upgrades = 0
    for s in subs:
        q_off = s.get("quality_off")
        q_on = s.get("quality_on")
        if q_off is not None and q_on is not None:
            delta = q_on - q_off
            if delta < -0.05:
                downgrades += 1
            elif delta > 0.05:
                upgrades += 1
        d_atk = (s.get("attack_on", 0) or 0) - (s.get("attack_off", 0) or 0)
        d_def = (s.get("defence_on", 0) or 0) - (s.get("defence_off", 0) or 0)
        team = str(s.get("team", ""))[:9]
        print(f"  {s.get('minute', 0):>4} {team:<9} "
              f"{str(s.get('player_off', ''))[:15]:<16} "
              f"{str(s.get('player_on', ''))[:15]:<16} "
              f"{q_off if q_off is not None else 0:>6.1f} "
              f"{q_on if q_on is not None else 0:>6.1f} "
              f"{d_atk:>+7.1f} {d_def:>+7.1f} "
              f"{str(s.get('reason', ''))[:12]:<12}")

    print(f"\n  overall quality moved UP on   {upgrades}")
    print(f"  overall quality moved DOWN on {downgrades}")

    batched = sum(1 for s in subs if s.get("is_batched"))
    if batched:
        print(f"  changes made inside an existing window (free): {batched}")

    gate = getattr(ctrl, "quality_gate_stats", None)
    if gate:
        print("\n  --- did the scoreline rule ever BIND? ---")
        print(f"    candidates considered : {gate['considered']}")
        print(f"    accepted              : {gate['accepted']}")
        print(f"    REJECTED on quality   : {gate['rejected']}")
        print(f"    bench exhausted       : {gate['exhausted']}")
        if gate["rejected"] == 0:
            print("\n    ⚠ THE RULE NEVER REJECTED ANYTHING in this match.")
            print("      That means every change was an improvement on both")
            print("      axes, or the budget was never tight enough to bite.")
            print("      Unit tests cover the rejection path; this match does")
            print("      not evidence that it fires at match level.")

    print("\n  dAtk/dDef are the two Wharton columns: what the change cost")
    print("  or gained in ATTACKING and DEFENSIVE quality specifically. A")
    print("  leading team taking a defensive hit here would be a defect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())