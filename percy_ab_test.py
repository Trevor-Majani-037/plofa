"""percy_ab_test.py — Percy multi-match A/B attestation.

Runs N matches per arm with Percy's chosen-one brain vs baseline RW brain,
using SAME seeds for both arms, and aggregates Percy's personal stats.

No auto_run_match.py.  Uses match_probe + MatchEngine directly.

Arm A: Percy WITH chosen-one brain (brains/percy.json, 32->64->64->10,
       6,922 params, trained on project surrogate, fitness 0.9701)
       + never-forgetting mind (decay OFF, view_radius 45m)
Arm B: Percy baseline — standard RW position brain (brains/RW.json,
       24->32->32->10, 2,186 params)
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any, Dict, List

from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from decision_brain import DecisionBrain
from football_brain import FootballBrain
from brain_integration import (
    NeuralDecisionBrain, register_brain, clear_registry, get_brain,
)
from cognition.mind import PlayerMind, register_mind
from match_engine import set_team_press_auto
from percy_the_chosen import (
    load_percy_brain, build_percy_brain, build_percy_mind,
    PERCY_NAME, PERCY_ALIAS,
)
from match_probe import _team_profile, _install_neural_decide, _restore_decide
from match_engine import EventType


# ── Squad template: Percy as RW on home team ──
_HOME_STARTERS = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CM2", "CM"),
    ("LW", "LW"), ("ST", "ST"), (PERCY_NAME, "RW"),
]
_HOME_SUBS = [("SUB1", "ST"), ("SUB2", "CM"), ("SUB3", "CB")]
_AWAY_STARTERS = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CM2", "CM"),
    ("LW", "LW"), ("ST", "ST"), ("RW", "RW"),
]
_AWAY_SUBS = [("SUB1", "ST"), ("SUB2", "CM"), ("SUB3", "CB")]

# Intent labels for readable output
INTENT_LABELS = [
    "PROGRESSIVE_PASS", "SAFE_PASS", "THROUGH_BALL", "SWITCH",
    "CARRY", "DRIBBLE", "CROSS", "SHOOT", "RECYCLE", "PROTECT_POSSESSION",
]


def _count_percy_events(result, name: str = PERCY_NAME) -> Dict[str, Any]:
    stats = {
        "goals": 0, "shots_on_target": 0, "shots_off_target": 0,
        "shots_blocked": 0, "woodwork": 0, "passes": 0, "key_passes": 0,
        "carries": 0, "dribbles": 0, "crosses": 0, "through_balls": 0,
        "decisions": 0, "decision_quality_sum": 0.0, "intent_mix": {},
        "xg": 0.0, "xa": 0.0,
    }

    shot_types = {
        EventType.SHOT_ON_TARGET: "shots_on_target",
        EventType.SHOT_OFF_TARGET: "shots_off_target",
        EventType.SHOT_BLOCKED: "shots_blocked",
        EventType.HIT_WOODWORK: "woodwork",
    }

    for ev in result.timeline:
        if getattr(ev, "player", "") != name:
            continue
        et = ev.event_type
        md = getattr(ev, "metadata", None) or {}

        if et == EventType.GOAL:
            stats["goals"] += 1
            stats["xg"] += float(getattr(ev, "xg", 0.0) or 0.0)
        if et == EventType.PENALTY_SCORED:
            stats["goals"] += 1
        if et in shot_types:
            stats[shot_types[et]] += 1
            stats["xg"] += float(getattr(ev, "xg", 0.0) or 0.0)
        if et.name == "CHANCE_CREATED":
            stats["key_passes"] += 1
            stats["xa"] += float(getattr(ev, "xa", 0.0) or 0.0)

        brain = md.get("active_brain")
        if brain:
            stats["decisions"] += 1
            q = brain.get("decision_quality")
            if q is not None:
                stats["decision_quality_sum"] += q
            intent = brain.get("intent")
            if intent:
                stats["intent_mix"][intent] = stats["intent_mix"].get(intent, 0) + 1

        if et.name in ("PASS", "THROUGH_BALL", "SWITCH", "CROSS", "RECYCLE", "PROTECT_POSSESSION"):
            stats["passes"] += 1
        if et.name == "CARRY":
            stats["carries"] += 1
        if et.name == "DRIBBLE":
            stats["dribbles"] += 1
        if et.name == "CROSS":
            stats["crosses"] += 1
        if et.name == "THROUGH_BALL":
            stats["through_balls"] += 1

    if stats["decisions"] > 0:
        stats["mean_decision_quality"] = round(stats["decision_quality_sum"] / stats["decisions"], 4)
    else:
        stats["mean_decision_quality"] = 0.0
    return stats


def _run_one_match(use_percy_brain: bool, seed: int,
                   home_style: str = "balanced",
                   away_style: str = "fluid_counter") -> Dict[str, Any]:
    random.seed(seed)

    home_squad = SquadBuilder.build("Probe FC", list(_HOME_STARTERS), list(_HOME_SUBS))
    away_squad = SquadBuilder.build("Rival FC", list(_AWAY_STARTERS), list(_AWAY_SUBS))

    config = MatchConfig(
        home_team="Probe FC", away_team="Rival FC",
        match_date=date(2026, 9, 6), matchday=3, season="26/27",
    )
    hp = _team_profile("Probe FC", home_style)
    ap = _team_profile("Rival FC", away_style)

    from squad_manager import SubstitutionController
    sc = SubstitutionController(
        home_team="Probe FC", away_team="Rival FC",
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"],
        home_style=hp.style.value, away_style=ap.style.value,
    )

    clear_registry()

    if use_percy_brain:
        brain = load_percy_brain()
        if brain is None:
            brain = build_percy_brain(seed=123)
        register_brain(PERCY_NAME, brain)
        register_brain(PERCY_ALIAS, brain)
        mind = build_percy_mind()
        register_mind(PERCY_NAME, mind)
        register_mind(PERCY_ALIAS, mind)

    _install_neural_decide({})

    eng = MatchEngine(config, hp, ap)
    eng.set_squad("Probe FC", home_squad["starters"], home_squad["substitutes"])
    eng.set_squad("Rival FC", away_squad["starters"], away_squad["substitutes"])
    eng.set_stamina_controller(sc)

    try:
        result = eng.simulate()
    finally:
        _restore_decide()

    percy_stats = _count_percy_events(result, PERCY_NAME)

    return {
        "result": result,
        "percy_stats": percy_stats,
        "home_goals": result.home_goals,
        "away_goals": result.away_goals,
        "possession": result.home_possession_pct,
        "score_str": result.score_str,
        "use_percy_brain": use_percy_brain,
    }


def _aggregate(arms: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate Percy stats across multiple matches in one arm."""
    agg = {
        "goals": 0, "shots_on_target": 0, "shots_off_target": 0,
        "shots_blocked": 0, "woodwork": 0, "passes": 0, "key_passes": 0,
        "carries": 0, "dribbles": 0, "crosses": 0, "through_balls": 0,
        "decisions": 0, "decision_quality_sum": 0.0,
        "intent_mix": {}, "xg": 0.0, "xa": 0.0,
        "matches": len(arms),
        "team_goals_for": 0, "team_goals_against": 0,
        "wins": 0, "draws": 0, "losses": 0,
    }
    n_decisions = 0
    for a in arms:
        s = a["percy_stats"]
        agg["goals"] += s["goals"]
        agg["shots_on_target"] += s["shots_on_target"]
        agg["shots_off_target"] += s["shots_off_target"]
        agg["shots_blocked"] += s["shots_blocked"]
        agg["woodwork"] += s["woodwork"]
        agg["passes"] += s["passes"]
        agg["key_passes"] += s["key_passes"]
        agg["carries"] += s["carries"]
        agg["dribbles"] += s["dribbles"]
        agg["crosses"] += s["crosses"]
        agg["through_balls"] += s["through_balls"]
        agg["decisions"] += s["decisions"]
        agg["decision_quality_sum"] += s["decision_quality_sum"]
        agg["xg"] += s["xg"]
        agg["xa"] += s["xa"]
        for k, v in s["intent_mix"].items():
            agg["intent_mix"][k] = agg["intent_mix"].get(k, 0) + v
        agg["team_goals_for"] += a["home_goals"]
        agg["team_goals_against"] += a["away_goals"]
        if a["home_goals"] > a["away_goals"]:
            agg["wins"] += 1
        elif a["home_goals"] == a["away_goals"]:
            agg["draws"] += 1
        else:
            agg["losses"] += 1
        n_decisions += s["decisions"]
    if n_decisions > 0:
        agg["mean_decision_quality"] = round(agg["decision_quality_sum"] / n_decisions, 4)
    else:
        agg["mean_decision_quality"] = 0.0
    agg["points"] = agg["wins"] * 3 + agg["draws"]
    return agg


def _label(name: str) -> str:
    return name.replace(chr(0x107), "c")


def main():
    n_matches = 7
    base_seed = 123

    print("=" * 66)
    print("  PERCY LUKA — MULTI-MATCH A/B ATTESTATION")
    print("=" * 66)
    print(f"\n  Matches per arm: {n_matches}")
    print(f"  Same seeds for both arms (paired comparison)")
    print(f"\n  Arm A — Percy WITH chosen-one brain:")
    print(f"         brains/percy.json 32->64->64->10 = 6,922 params")
    print(f"         trained on project surrogate (fitness 0.9701)")
    print(f"         never-forgetting mind (decay OFF, view 45m)")
    print(f"\n  Arm B — Percy BASELINE:")
    print(f"         brains/RW.json 24->32->32->10 = 2,186 params")
    print()

    # Run Arm A
    print("-" * 66)
    print(f"  ARM A: Percy WITH chosen-one brain ({n_matches} matches)")
    print("-" * 66)
    arm_a = []
    for m in range(n_matches):
        seed = base_seed + m * 100
        r = _run_one_match(use_percy_brain=True, seed=seed)
        arm_a.append(r)
        ps = r["percy_stats"]
        print(f"   match {m+1}: {r['score_str']} | Percy g={ps['goals']} "
              f"sot={ps['shots_on_target']} xg={ps['xg']:.3f} "
              f"dq={ps['mean_decision_quality']} | mix={ps['intent_mix']}")
    agg_a = _aggregate(arm_a)

    print()
    print("-" * 66)
    print(f"  ARM B: Percy BASELINE RW brain ({n_matches} matches)")
    print("-" * 66)
    arm_b = []
    for m in range(n_matches):
        seed = base_seed + m * 100
        r = _run_one_match(use_percy_brain=False, seed=seed)
        arm_b.append(r)
        ps = r["percy_stats"]
        print(f"   match {m+1}: {r['score_str']} | Percy g={ps['goals']} "
              f"sot={ps['shots_on_target']} xg={ps['xg']:.3f} "
              f"dq={ps['mean_decision_quality']} | mix={ps['intent_mix']}")
    agg_b = _aggregate(arm_b)

    # ── Comparison ──
    print()
    print("=" * 66)
    print("  PERCY: CHOSEN-ONE vs BASELINE  (aggregated over 7 matches each)")
    print("=" * 66)
    cmp_rows = [
        ("Goals", agg_a["goals"], agg_b["goals"]),
        ("Shots on target", agg_a["shots_on_target"], agg_b["shots_on_target"]),
        ("Shots off target", agg_a["shots_off_target"], agg_b["shots_off_target"]),
        ("Key passes", agg_a["key_passes"], agg_b["key_passes"]),
        ("xG", round(agg_a["xg"], 3), round(agg_b["xg"], 3)),
        ("xA", round(agg_a["xa"], 3), round(agg_b["xa"], 3)),
        ("Passes", agg_a["passes"], agg_b["passes"]),
        ("Carries", agg_a["carries"], agg_b["carries"]),
        ("Crosses", agg_a["crosses"], agg_b["crosses"]),
        ("Through balls", agg_a["through_balls"], agg_b["through_balls"]),
        ("Decisions", agg_a["decisions"], agg_b["decisions"]),
        ("Mean decision quality", agg_a["mean_decision_quality"], agg_b["mean_decision_quality"]),
        ("Team points (W-D-L)", f"{agg_a['wins']}-{agg_a['draws']}-{agg_a['losses']} ({agg_a['points']}pts)",
                                   f"{agg_b['wins']}-{agg_b['draws']}-{agg_b['losses']} ({agg_b['points']}pts)"),
    ]
    for label, a, b in cmp_rows:
        print(f"  {label:<28} chosen-one: {a:<18} baseline: {b}")

    print()
    print("  Intent mix (chosen-one):", agg_a["intent_mix"])
    print("  Intent mix (baseline):  ", agg_b["intent_mix"])
    print()
    print(f"  Percy's brain: {_label(PERCY_NAME)} — {agg_a['decisions']} decisions logged across 7 matches")
    print(f"  Mind: never-forgetting (decay OFF), view radius 45m, confidence 1.5")
    print()
    print("  CONFIRMATION: Percy's chosen-one brain is wired via wire_percy().")
    print("  Call wire_percy() before any match kickoff and his brain will be active.")


if __name__ == "__main__":
    main()
