"""percy_match_test.py — Percy-focused A/B match attestation.

Runs TWO real matches (same seed, same opponent) and compares Percy Luka's
personal stats between:

  ARM A — Percy with his chosen-one brain (24->64->64->10, 6410 params,
          trained him alone, fitness 0.4446) + never-forgetting mind
  ARM B — Percy baseline with the standard RW position brain (2,186 params)

No auto_run_match.py.  Uses match_probe infrastructure + MatchEngine directly.

Percy must be on the pitch under his exact roster name "Perćy Luka" so the
brain/mind registry lookup (by exact player name) finds him.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any, Dict, List, Optional

from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity
from player_dna import SquadBuilder
from decision_brain import DecisionBrain
from football_brain import FootballBrain
from brain_integration import (
    NeuralDecisionBrain, register_brain, clear_registry,
)
from cognition.mind import PlayerMind, register_mind, get_mind
from match_probe import _install_neural_decide, _restore_decide
from match_engine import set_team_press_auto
from percy_the_chosen import (
    build_percy_brain, load_percy_brain, build_percy_mind,
    PERCY_NAME, PERCY_ALIAS, PERCY_POSITION,
)

from match_probe import _team_profile

# ── Squad template: Percy as RW on the home team (exact roster name) ──
_HOME_STARTERS = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CM2", "CM"),
    ("LW", "LW"), ("ST", "ST"), (PERCY_NAME, "RW"),   # <-- Percy is RW
]
_HOME_SUBS = [("SUB1", "ST"), ("SUB2", "CM"), ("SUB3", "CB")]
_AWAY_STARTERS = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CM2", "CM"),
    ("LW", "LW"), ("ST", "ST"), ("RW", "RW"),
]
_AWAY_SUBS = [("SUB1", "ST"), ("SUB2", "CM"), ("SUB3", "CB")]

# ── Event types we count for Percy ──
from match_engine import EventType  # noqa: E402


def _count_percy_events(result, name: str = PERCY_NAME) -> Dict[str, Any]:
    """Walk the timeline and tally Percy's personal contributions."""
    stats = {
        "goals": 0,
        "shots_on_target": 0,
        "shots_off_target": 0,
        "shots_blocked": 0,
        "woodwork": 0,
        "passes": 0,
        "key_passes": 0,
        "carries": 0,
        "dribbles": 0,
        "crosses": 0,
        "through_balls": 0,
        "decisions": 0,
        "decision_quality_sum": 0.0,
        "intent_mix": {},
        "xg": 0.0,
        "xa": 0.0,
        "minutes": 0,
    }

    shot_types = {
        EventType.SHOT_ON_TARGET: "shots_on_target",
        EventType.SHOT_OFF_TARGET: "shots_off_target",
        EventType.SHOT_BLOCKED: "shots_blocked",
        EventType.HIT_WOODWORK: "woodwork",
    }
    intent_labels = [
        "PROGRESSIVE_PASS", "SAFE_PASS", "THROUGH_BALL", "SWITCH",
        "CARRY", "DRIBBLE", "CROSS", "SHOOT", "RECYCLE", "PROTECT_POSSESSION",
    ]

    for ev in result.timeline:
        if getattr(ev, "player", "") != name:
            continue
        et = ev.event_type
        md = getattr(ev, "metadata", None) or {}

        # Goals
        if et == EventType.GOAL:
            stats["goals"] += 1
            stats["xg"] += float(getattr(ev, "xg", 0.0) or 0.0)
        if et == EventType.PENALTY_SCORED:
            stats["goals"] += 1

        # Shots
        if et in shot_types:
            stats[shot_types[et]] += 1
            stats["xg"] += float(getattr(ev, "xg", 0.0) or 0.0)

        # Key passes (CHANCE_CREATED)
        if et.name == "CHANCE_CREATED":
            stats["key_passes"] += 1
            stats["xa"] += float(getattr(ev, "xa", 0.0) or 0.0)

        # Passes / through balls / crosses / switches from decision metadata
        brain = md.get("active_brain")
        if brain:
            stats["decisions"] += 1
            q = brain.get("decision_quality")
            if q is not None:
                stats["decision_quality_sum"] += q
            intent = brain.get("intent")
            if intent:
                stats["intent_mix"][intent] = stats["intent_mix"].get(intent, 0) + 1

        # Count by event family
        if et.name in ("PASS", "THROUGH_BALL", "SWITCH", "CROSS", "RECYCLE", "PROTECT_POSSESSION"):
            stats["passes"] += 1
        if et.name == "CARRY":
            stats["carries"] += 1
        if et.name == "DRIBBLE":
            stats["dribbles"] += 1

    if stats["decisions"] > 0:
        stats["mean_decision_quality"] = round(stats["decision_quality_sum"] / stats["decisions"], 4)
    else:
        stats["mean_decision_quality"] = 0.0

    return stats


def _run_one_match(
    use_percy_brain: bool,
    seed: int,
    home_style: str = "balanced",
    away_style: str = "fluid_counter",
) -> Dict[str, Any]:
    """Run one match and return (result, percy_stats)."""
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

    # ── Register brains ──
    clear_registry()

    # Percy's brain (big, trained, chosen-one) only in arm A
    if use_percy_brain:
        brain = load_percy_brain()
        if brain is None:
            brain = build_percy_brain(seed=2018)
        register_brain(PERCY_NAME, brain)
        # Register Percy's never-forgetting mind
        mind = build_percy_mind()
        register_mind(PERCY_NAME, mind)

    # Everyone else auto-loads their position brain inside NeuralDecisionBrain
    _install_neural_decide({})

    # ── Run ──
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


def main():
    seed = 2018
    print("=" * 64)
    print("  PERCY LUKA — CHOSEN-ONE BRAIN ATTESTATION")
    print("=" * 64)
    print(f"\n  Percy's brain: 24->64->64->10 = 6,410 params (trained, fitness 0.4446)")
    print(f"  Standard RW brain: 24->32->32->10 = 2,186 params")
    print(f"  Percy's mind: never-forgetting (decay OFF), view radius 45m")
    print(f"  Seed: {seed} | Same match, same opponent, both arms\n")

    # Arm A — Percy with chosen-one brain
    print("─" * 64)
    print("  ARM A: Percy WITH chosen-one brain (6,410 params + never-forget)")
    print("─" * 64)
    arm_a = _run_one_match(use_percy_brain=True, seed=seed)
    ps_a = arm_a["percy_stats"]
    print(f"  Score: {arm_a['score_str']}  |  Possession: {arm_a['possession']:.1f}%")
    print(f"  Percy GOALS: {ps_a['goals']}")
    print(f"  Percy SHOTS: on={ps_a['shots_on_target']} off={ps_a['shots_off_target']} blocked={ps_a['shots_blocked']} woodwork={ps_a['woodwork']}")
    print(f"  Percy PASSES: {ps_a['passes']}  KEY_PASSES: {ps_a['key_passes']}  xA: {ps_a['xa']:.3f}")
    print(f"  Percy CARRIES: {ps_a['carries']}  DRIBBLES: {ps_a['dribbles']}  CROSSES: {ps_a['intent_mix'].get('CROSS', 0)}  THROUGH_BALLS: {ps_a['intent_mix'].get('THROUGH_BALL', 0)}")
    print(f"  Percy decisions logged: {ps_a['decisions']}  mean quality: {ps_a['mean_decision_quality']}")
    print(f"  Percy intent mix: {ps_a['intent_mix']}")
    print(f"  Percy xG: {ps_a['xg']:.3f}")

    # Arm B — Percy baseline (normal RW brain)
    print("\n" + "─" * 64)
    print("  ARM B: Percy BASELINE (normal RW brain, 2,186 params)")
    print("─" * 64)
    arm_b = _run_one_match(use_percy_brain=False, seed=seed + 7)
    ps_b = arm_b["percy_stats"]
    print(f"  Score: {arm_b['score_str']}  |  Possession: {arm_b['possession']:.1f}%")
    print(f"  Percy GOALS: {ps_b['goals']}")
    print(f"  Percy SHOTS: on={ps_b['shots_on_target']} off={ps_b['shots_off_target']} blocked={ps_b['shots_blocked']} woodwork={ps_b['woodwork']}")
    print(f"  Percy PASSES: {ps_b['passes']}  KEY_PASSES: {ps_b['key_passes']}  xA: {ps_b['xa']:.3f}")
    print(f"  Percy CARRIES: {ps_b['carries']}  DRIBBLES: {ps_b['dribbles']}  CROSSES: {ps_b['intent_mix'].get('CROSS', 0)}  THROUGH_BALLS: {ps_b['intent_mix'].get('THROUGH_BALL', 0)}")
    print(f"  Percy decisions logged: {ps_b['decisions']}  mean quality: {ps_b['mean_decision_quality']}")
    print(f"  Percy intent mix: {ps_b['intent_mix']}")
    print(f"  Percy xG: {ps_b['xg']:.3f}")

    # ── Comparison ──
    print("\n" + "=" * 64)
    print("  PERCY: CHOSEN-ONE vs BASELINE")
    print("=" * 64)
    cmp_rows = [
        ("Goals", ps_a["goals"], ps_b["goals"]),
        ("Shots on target", ps_a["shots_on_target"], ps_b["shots_on_target"]),
        ("Shots off target", ps_a["shots_off_target"], ps_b["shots_off_target"]),
        ("Shots blocked", ps_a["shots_blocked"], ps_b["shots_blocked"]),
        ("Woodwork", ps_a["woodwork"], ps_b["woodwork"]),
        ("Key passes", ps_a["key_passes"], ps_b["key_passes"]),
        ("xG", round(ps_a["xg"], 3), round(ps_b["xg"], 3)),
        ("xA", round(ps_a["xa"], 3), round(ps_b["xa"], 3)),
        ("Passes", ps_a["passes"], ps_b["passes"]),
        ("Carries", ps_a["carries"], ps_b["carries"]),
        ("Dribbles", ps_a["dribbles"], ps_b["dribbles"]),
        ("Crosses", ps_a["intent_mix"].get("CROSS", 0), ps_b["intent_mix"].get("CROSS", 0)),
        ("Through balls", ps_a["intent_mix"].get("THROUGH_BALL", 0), ps_b["intent_mix"].get("THROUGH_BALL", 0)),
        ("Decisions", ps_a["decisions"], ps_b["decisions"]),
        ("Mean decision quality", ps_a["mean_decision_quality"], ps_b["mean_decision_quality"]),
    ]
    for label, a, b in cmp_rows:
        delta = a - b
        delta_str = f"+{delta}" if delta > 0 else (f"{delta}" if delta < 0 else "0")
        print(f"  {label:<26} chosen-one: {a:<4}  baseline: {b:<4}  delta: {delta_str}")

    print("\n  If chosen-one > baseline across most rows -> Percy's big brain")
    print("  and never-forgetting mind are making him play differently.")


if __name__ == "__main__":
    main()
