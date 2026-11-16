#!/usr/bin/env python
"""A/B GATE: WHICH MEN THE ATTENTION SET IS BUILT FROM (2026-11-16).

THE CHANGE UNDER TEST
---------------------
`situational_decision.SituationalDecisionBrain.decide` used to build the
attention tokens from whatever the FOV gate happened to leave:

  * with a mind registered (production, ``PLOFA_COGNITION=1``) -> the mind's
    25 m / 2 + 2 frame: at most FOUR men, no angle test, and ~35 % of the
    tokens outside the cone;
  * with no mind -> the RAW lists, no cone at all, up to 16 men.

The distillation corpus did neither: ``surrogate_collect._situational_blocks``
built the tokens from the list ``perceive`` kept (cone + range + top-k),
measured at a mean of 4.76 men a look, up to 12 for a GK.

ARM `legacy` : the feed in force until 2026-11-16 (PLOFA_TOKEN_SOURCE=legacy)
ARM `scene`  : the corpus feed                                  (the fix)

WHY TWO PROCESSES
-----------------
Each arm runs in its own interpreter so a cached brain, a warm RNG stream or
a leftover mind cannot tilt one side of the comparison.

WHAT IS MEASURED
----------------
* wiring: men per decision, max, zero-token rate (corpus: mean 4.76, 6.9 %)
* on the ball: passes, completion %, progressive passes, through balls,
  switches, carries, turnovers lost, interceptions suffered
* outcome: goals, shot volume, xG, possession

Run:  python -m tests.gate_token_source          (both arms, 7 matches each)
      python -m tests.gate_token_source --arm scene   (one arm, for a rerun)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter
from datetime import date
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from match_engine import (
    MatchConfig, MatchEngine, MatchEvent, MatchPhase, MatchState,
)
from player_dna import SquadBuilder

import auto_run_match

HOME, AWAY = "Oxton", "Natrican"
HOME_STYLE, AWAY_STYLE = "fluid_counter", "attacking"


# ─────────────────────────────────────────────────────────────
# instrumentation
# ─────────────────────────────────────────────────────────────

class Probe:
    """Token counts and chosen intents, straight off the live path."""

    def __init__(self) -> None:
        self.token_counts: List[int] = []
        self.intents: Counter = Counter()
        self.actions: Counter = Counter()
        self._orig_tokens = None
        self._orig_decide = None

    def install(self) -> None:
        import situation_tokens
        import situational_decision

        orig_tokens = situation_tokens.entity_tokens
        orig_decide = situational_decision.SituationalDecisionBrain.decide
        probe = self

        def counted_entity_tokens(player, x, y, teammates, defenders,
                                  position_engine, attacks_right):
            tokens, mask = orig_tokens(
                player, x, y, teammates, defenders, position_engine,
                attacks_right)
            probe.token_counts.append(int(np.sum(mask)))
            return tokens, mask

        def counted_decide(player, *args, **kwargs):
            decision = orig_decide(player, *args, **kwargs)
            intent = getattr(getattr(decision, "intent", None), "value", None)
            probe.intents[str(intent)] += 1
            probe.actions[str(getattr(decision, "action", "?"))] += 1
            return decision

        situation_tokens.entity_tokens = counted_entity_tokens
        situational_decision.SituationalDecisionBrain.decide = staticmethod(
            counted_decide)
        self._orig_tokens = orig_tokens
        self._orig_decide = orig_decide

    def uninstall(self) -> None:
        import situation_tokens
        import situational_decision

        if self._orig_tokens is not None:
            situation_tokens.entity_tokens = self._orig_tokens
        if self._orig_decide is not None:
            situational_decision.SituationalDecisionBrain.decide = (
                staticmethod(self._orig_decide))


# ─────────────────────────────────────────────────────────────
# one match
# ─────────────────────────────────────────────────────────────

def _build_squad(name: str) -> Dict[str, Any]:
    loader = auto_run_match.get_loader()
    raw = loader.build_matchday_squad(name, availability=None)
    squad = SquadBuilder.build(
        team_name=name,
        starters=raw["starters"],
        substitutes=raw["substitutes"],
        team_superstars=raw["superstars"],
        set_piece_takers=raw["sp_takers"],
        soul_assignments=auto_run_match.SOUL_PLAYERS,
    )
    return {"raw": raw, "squad": squad}


def _event_name(event: MatchEvent) -> str:
    return getattr(getattr(event, "event_type", None), "name", "")


def _tally(events: List[MatchEvent], team: str) -> Dict[str, float]:
    """On-ball volume for one team, read off the timeline."""
    out = Counter()
    for e in events:
        if getattr(e, "team", None) != team:
            continue
        name = _event_name(e)
        ok = bool(getattr(e, "outcome", False))
        if name == "PASS":
            out["passes"] += 1
            out["passes_ok"] += int(ok)
        elif name == "PROGRESSIVE_PASS":
            out["prog_passes"] += 1
            out["prog_ok"] += int(ok)
        elif name == "THROUGH_BALL":
            out["through_balls"] += 1
            out["through_ok"] += int(ok)
        elif name == "SWITCH_OF_PLAY":
            out["switches"] += 1
        elif name == "CROSS_ATTEMPT":
            out["crosses"] += 1
            out["crosses_ok"] += int(ok)
        elif name == "CARRY":
            out["carries"] += 1
        elif name == "DRIBBLE_ATTEMPT":
            out["dribbles"] += 1
            out["dribbles_ok"] += int(ok)
        elif name == "TURNOVER":
            out["turnovers"] += 1
        elif name == "INTERCEPTION":
            out["interceptions_won"] += 1
        elif name in ("SHOT_ATTEMPT", "SHOT_ON_TARGET", "SHOT_OFF_TARGET",
                      "SHOT_BLOCKED", "GOAL"):
            out["shots"] += 1
            out["xg"] += float(getattr(e, "xg", 0.0) or 0.0)
            if name in ("SHOT_ON_TARGET", "GOAL"):
                out["shots_on_target"] += 1
        elif name == "CHANCE_CREATED":
            out["chances"] += 1
    return {k: float(v) for k, v in out.items()}


def run_match(arm: str, index: int, seed: int, probe: Probe) -> Dict[str, Any]:
    from cognition_brain import engage_mind_observer, disengage_mind_observer
    from cognition.mind import clear_minds, register_mind, PlayerMind
    from brain_integration import set_cognition, clear_registry

    os.environ["PLOFA_TOKEN_SOURCE"] = arm
    random.seed(seed)
    np.random.seed(seed)

    home = _build_squad(HOME)
    away = _build_squad(AWAY)

    home_style = auto_run_match._resolve_team_profile(
        HOME, home["raw"]["formation"], True, None, HOME_STYLE)
    away_style = auto_run_match._resolve_team_profile(
        AWAY, away["raw"]["formation"], False, None, AWAY_STYLE)

    config = MatchConfig(
        home_team=HOME, away_team=AWAY,
        match_date=date(2026, 9, 6), matchday=3, season="26/27",
    )
    state = MatchState()
    state.phase = MatchPhase.OPENING
    state.minute = 0
    state.match_clock_s = 0.0

    engine = MatchEngine(config, home_style, away_style)
    engine.state = state
    engine.set_squad(HOME, home["squad"]["starters"], home["squad"]["substitutes"])
    engine.set_squad(AWAY, away["squad"]["starters"], away["squad"]["substitutes"])

    # ── exactly what auto_run_match does before kickoff ──────────
    clear_registry()
    clear_minds()
    for side in (home, away):
        for p in side["squad"]["starters"]:
            register_mind(p.name, PlayerMind())
    set_cognition(True)
    engage_mind_observer()

    before = len(probe.token_counts)
    result = engine.simulate()
    disengage_mind_observer()

    events = list(result.timeline)
    row = {
        "arm": arm,
        "match": index + 1,
        "seed": seed,
        "home_goals": int(result.home_goals),
        "away_goals": int(result.away_goals),
        "home_possession": float(result.home_possession_pct),
        "decisions": len(probe.token_counts) - before,
        "home": _tally(events, HOME),
        "away": _tally(events, AWAY),
    }
    print(f"  [{arm}] match {index+1} seed {seed}: "
          f"{row['home_goals']}-{row['away_goals']}  "
          f"poss {row['home_possession']:.1f}%  "
          f"decisions {row['decisions']}")
    return row


# ─────────────────────────────────────────────────────────────
# arm runner (child process)
# ─────────────────────────────────────────────────────────────

def run_arm(arm: str, matches: int, seed: int, out_path: str) -> None:
    probe = Probe()
    probe.install()
    rows = []
    for m in range(matches):
        rows.append(run_match(arm, m, seed + m * 100, probe))

    counts = np.asarray(probe.token_counts, dtype=np.float64)
    summary = {
        "arm": arm,
        "matches": rows,
        "tokens": {
            "decisions": int(counts.size),
            "mean": round(float(counts.mean()), 3) if counts.size else 0.0,
            "max": int(counts.max()) if counts.size else 0,
            "zero": int((counts == 0).sum()),
            "zero_pct": round(float((counts == 0).mean() * 100.0), 2)
            if counts.size else 0.0,
            "histogram": {str(k): int(v) for k, v in
                          sorted(Counter(counts.astype(int).tolist()).items())},
        },
        "intents": dict(probe.intents.most_common()),
        "actions": dict(probe.actions.most_common()),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"  [{arm}] wrote {out_path}")


# ─────────────────────────────────────────────────────────────
# aggregation (parent process)
# ─────────────────────────────────────────────────────────────

SIDES = ("home", "away")


def _sum(rows: List[Dict[str, Any]], key: str, field: str) -> float:
    return float(sum(r[key].get(field, 0.0) for r in rows))


def _report(arm: Dict[str, Any]) -> Dict[str, Any]:
    rows = arm["matches"]
    n = len(rows)
    home, away = {}, {}
    for field in ("passes", "passes_ok", "prog_passes", "prog_ok",
                  "through_balls", "through_ok", "switches", "crosses",
                  "crosses_ok", "carries", "dribbles", "dribbles_ok",
                  "turnovers", "interceptions_won", "shots",
                  "shots_on_target", "chances", "xg"):
        home[field] = _sum(rows, "home", field) / n
        away[field] = _sum(rows, "away", field) / n

    def pct(ok: float, of: float) -> float:
        return round(100.0 * ok / of, 1) if of else 0.0

    return {
        "arm": arm["arm"],
        "matches": n,
        "record": (f"{sum(1 for r in rows if r['home_goals'] > r['away_goals'])}"
                   f"W-{sum(1 for r in rows if r['home_goals'] == r['away_goals'])}D-"
                   f"{sum(1 for r in rows if r['away_goals'] > r['home_goals'])}L"),
        "gf": sum(r["home_goals"] for r in rows),
        "ga": sum(r["away_goals"] for r in rows),
        "gd": sum(r["home_goals"] - r["away_goals"] for r in rows),
        "possession": round(float(np.mean([r["home_possession"] for r in rows])), 1),
        "tokens": arm["tokens"],
        "intents": arm["intents"],
        "home": home,
        "away": away,
        "pass_pct": pct(home["passes_ok"], home["passes"]),
        "prog_pct": pct(home["prog_ok"], home["prog_passes"]),
        "through_pct": pct(home["through_ok"], home["through_balls"]),
        "cross_pct": pct(home["crosses_ok"], home["crosses"]),
        "dribble_pct": pct(home["dribbles_ok"], home["dribbles"]),
        "conceded_pass_pct": pct(away["passes_ok"], away["passes"]),
        "conceded_xg": round(away["xg"], 2),
    }


def _print_side_by_side(legacy: Dict[str, Any], scene: Dict[str, Any]) -> None:
    rows = [
        ("token source", "mind frame / raw", "perceived scene"),
        ("men per decision (mean)", legacy["tokens"]["mean"], scene["tokens"]["mean"]),
        ("men per decision (max)", legacy["tokens"]["max"], scene["tokens"]["max"]),
        ("zero-token decisions %", legacy["tokens"]["zero_pct"], scene["tokens"]["zero_pct"]),
        ("record", legacy["record"], scene["record"]),
        ("goals for", legacy["gf"], scene["gf"]),
        ("goals against", legacy["ga"], scene["ga"]),
        ("goal difference", legacy["gd"], scene["gd"]),
        ("possession %", legacy["possession"], scene["possession"]),
        ("passes / match", round(legacy["home"]["passes"], 1), round(scene["home"]["passes"], 1)),
        ("pass completion %", legacy["pass_pct"], scene["pass_pct"]),
        ("progressive passes / match", round(legacy["home"]["prog_passes"], 1),
         round(scene["home"]["prog_passes"], 1)),
        ("progressive completion %", legacy["prog_pct"], scene["prog_pct"]),
        ("through balls / match", round(legacy["home"]["through_balls"], 1),
         round(scene["home"]["through_balls"], 1)),
        ("switches / match", round(legacy["home"]["switches"], 1),
         round(scene["home"]["switches"], 1)),
        ("carries / match", round(legacy["home"]["carries"], 1),
         round(scene["home"]["carries"], 1)),
        ("turnovers lost / match", round(legacy["home"]["turnovers"], 1),
         round(scene["home"]["turnovers"], 1)),
        ("shots / match", round(legacy["home"]["shots"], 1), round(scene["home"]["shots"], 1)),
        ("shots on target / match", round(legacy["home"]["shots_on_target"], 1),
         round(scene["home"]["shots_on_target"], 1)),
        ("xG / match", round(legacy["home"]["xg"], 2), round(scene["home"]["xg"], 2)),
        ("xG conceded / match", legacy["conceded_xg"], scene["conceded_xg"]),
        ("chances created / match", round(legacy["home"]["chances"], 1),
         round(scene["home"]["chances"], 1)),
    ]
    width = max(len(r[0]) for r in rows)
    print()
    print(f"{'':<{width}}   {'LEGACY':>14}   {'SCENE':>14}")
    print("-" * (width + 34))
    for name, a, b in rows:
        print(f"{name:<{width}}   {str(a):>14}   {str(b):>14}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("legacy", "scene"), default=None)
    ap.add_argument("--matches", type=int, default=7)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.arm:
        out = args.out or os.path.join(here, f"_gate_token_{args.arm}.json")
        run_arm(args.arm, args.matches, args.seed, out)
        return 0

    results = {}
    for arm in ("legacy", "scene"):
        out = os.path.join(here, f"_gate_token_{arm}.json")
        print(f"\n=== ARM {arm.upper()} — {args.matches} matches ===")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--arm", arm,
             "--matches", str(args.matches), "--seed", str(args.seed),
             "--out", out],
            check=True,
        )
        with open(out, encoding="utf-8") as fh:
            results[arm] = _report(json.load(fh))

    _print_side_by_side(results["legacy"], results["scene"])
    print("\nintent mix (home side, all players):")
    for arm in ("legacy", "scene"):
        top = list(results[arm]["intents"].items())[:8]
        print(f"  {arm:<7} " + "  ".join(f"{k}:{v}" for k, v in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
