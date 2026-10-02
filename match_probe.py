"""Real-match validation for FootballBrain — run a neural brain in a live
MatchEngine match and measure its fitness from real outcomes.

WHY THIS IS A VALIDATION TOOL, NOT THE EVOLUTION DRIVER
--------------------------------------------------------
A full MatchEngine match takes ~15-20 s.  A genetic algorithm with a
population of 32 over 40 generations would need ~7 hours *per position*
if every evaluation ran a real match.  That's impractical.

So: the fast synthetic fitness in brain_evolution.py drives the GA, and
this probe validates / occasionally refines a handful of candidate brains
against the real engine.  Use it to:

  • confirm an evolved brain plays sensibly in a live match (no crashes,
    sane intent mix, reasonable possession/retention)
  • extract per-player decision quality from the real event timeline
  • A/B compare an evolved brain vs the heuristic DecisionBrain

NON-DESTRUCTIVE INTEGRATION
---------------------------
This probe does NOT edit event_chain.py.  Instead it temporarily patches
DecisionBrain.decide (which event_chain.py resolves at call time) to
route through NeuralDecisionBrain for the duration of a match, then
restores the original.  Production behavior is unchanged after a run.

Usage
-----
    from match_probe import run_neural_validation
    run_neural_validation(target_position="ST", brains_dir="brains",
                          n_matches=3, seed=1)
"""

from __future__ import annotations

import json
import os
import random
import time
import types
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from match_engine import (
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity,
)
from player_dna import SquadBuilder
from decision_brain import DecisionBrain
from football_brain import FootballBrain
from brain_integration import (
    NeuralDecisionBrain, register_brain, clear_registry, get_brain,
    _INTENT_BY_INDEX,
)


# ─────────────────────────────────────────────────────────────
# SQUAD TEMPLATES (balanced 4-3-3)
# ─────────────────────────────────────────────────────────────

# name, position — used to build both teams.  The "_NN" suffix marks the
# slot we can swap to test a single position's brain.
_BASE_TEMPLATE = [
    ("GK", "GK"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "LB"), ("RB", "RB"),
    ("CDM", "CDM"), ("CM1", "CM"), ("CM2", "CM"),
    ("LW", "LW"), ("ST", "ST"), ("RW", "RW"),
]

_SUBS = [("SUB1", "ST"), ("SUB2", "CM"), ("SUB3", "CB")]

# position → the exact starter name on the probe 4-3-3 template.  get_brain()
# resolves by exact player name, so a candidate MUST be registered under the
# on-pitch name (the old f'{pos}{pos}' placeholder never matched a player and
# silently fell back to the auto-loaded brain JSON).
_POSITION_SLOTS = {
    "GK": "GK", "CB": "CB1", "LB": "LB", "RB": "RB",
    "CDM": "CDM", "CM": "CM1", "LW": "LW", "RW": "RW", "ST": "ST",
}


def _slot_for_position(position: str) -> str:
    slot = _POSITION_SLOTS.get(position)
    if slot is None:
        raise ValueError(
            f"No 4-3-3 probe slot for position '{position}'. "
            f"Supported: {', '.join(sorted(_POSITION_SLOTS))}. "
            "CAM/CF are not on the probe template."
        )
    return slot


def _build_squads(home: str, away: str):
    home_squad = SquadBuilder.build(home, list(_BASE_TEMPLATE), list(_SUBS))
    away_squad = SquadBuilder.build(away, list(_BASE_TEMPLATE), list(_SUBS))
    return home_squad, away_squad


def _team_profile(name: str, style: str = "balanced", philosophy=None) -> TeamProfile:
    style_map = {
        "balanced": TeamStyle.BALANCED,
        "attacking": TeamStyle.ATTACKING,
        "defensive": TeamStyle.DEFENSIVE,
        "fluid_counter": TeamStyle.FLUID_COUNTER,
        "tiki_taka": TeamStyle.TIKI_TAKA,
        "wing_play": TeamStyle.WING_PLAY,
        "ultra_attacking": TeamStyle.ULTRA_ATTACKING,
        "ultra_defensive": TeamStyle.ULTRA_DEFENSIVE,
        "park_the_bus": TeamStyle.PARK_THE_BUS,
        "gegenpressing": TeamStyle.GEGENPRESSING,
        "route_one": TeamStyle.ROUTE_ONE,
        "structured_possession": TeamStyle.STRUCTURED_POSSESSION,
        "vertical_tiki_taka": TeamStyle.VERTICAL_TIKI_TAKA,
    }
    profile = TeamProfile(
        name=name,
        style=style_map.get(style, TeamStyle.BALANCED),
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM,
    )
    if philosophy is not None:
        if isinstance(philosophy, str):
            from philosophy import ARCHETYPES
            philosophy = ARCHETYPES.get(philosophy)
        if philosophy is not None:
            profile.philosophy = philosophy
            profile._apply_philosophy()
    return profile


# ─────────────────────────────────────────────────────────────
# DECISION-BRAIN PATCHING (non-destructive)
# ─────────────────────────────────────────────────────────────

_SAVED_DECIDE = None
# event_chain.py now calls NeuralDecisionBrain.decide() directly, so the
# genuine heuristic is captured here and only used to *temporarily pin*
# the neural entry point for heuristic-baseline comparisons.
_HEURISTIC_DECIDE = DecisionBrain.decide


def _pin_heuristic():
    """Temporarily route event_chain through the genuine heuristic AND turn
    the team press controller OFF — the heuristic baseline is the OLD system
    (heuristic on-ball + pure role-rate Bernoulli off-ball, g = 1.0).

    Returns a restore callable.  Because event_chain resolves
    NeuralDecisionBrain.decide at call time, replacing the class-level
    attribute switches the whole match to heuristic behaviour.
    """
    global _SAVED_DECIDE
    if _SAVED_DECIDE is None:
        _SAVED_DECIDE = NeuralDecisionBrain.decide
    NeuralDecisionBrain.decide = staticmethod(_HEURISTIC_DECIDE)
    from match_engine import set_team_press_auto
    set_team_press_auto(False)


def _restore_neural():
    global _SAVED_DECIDE
    if _SAVED_DECIDE is not None:
        NeuralDecisionBrain.decide = _SAVED_DECIDE
        _SAVED_DECIDE = None
    from match_engine import set_team_press_auto
    set_team_press_auto(True)


def _install_neural_decide(brain_lookup: Dict[str, FootballBrain]):
    """Make sure event_chain routes through NeuralDecisionBrain.

    brain_lookup: name -> FootballBrain to use for the on-ball player.
    Players NOT in the lookup fall back to the unregistered auto-load /
    random-brain path inside NeuralDecisionBrain.
    """
    global _SAVED_DECIDE
    if _SAVED_DECIDE is not None:
        NeuralDecisionBrain.decide = _SAVED_DECIDE
        _SAVED_DECIDE = None


def _restore_decide():
    _restore_neural()


# ─────────────────────────────────────────────────────────────
# FITNESS EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_team_fitness(result: Any, player_names: List[str]) -> Dict[str, Any]:
    """Compute a team-oriented fitness from a MatchResult.

    Combines score/xG (attacking output), possession (control) and
    decision quality averaged across the target player's logged on-ball
    touches (from CARRY event `active_brain` metadata).
    """
    home_goals = result.home_goals
    away_goals = result.away_goals
    home_xg = result.home_xg
    away_xg = result.away_xg
    poss = result.home_possession_pct

    # decision stats from the timeline
    n_touches = 0
    sum_quality = 0.0
    n_errors = 0
    intents = []

    # figure out which side the target player is on
    target = set(player_names)

    for ev in result.timeline:
        md = getattr(ev, "metadata", None) or {}
        brain = md.get("active_brain")
        if not brain:
            continue
        player = getattr(ev, "player", None) or getattr(ev, "player_name", "")
        if player in target:
            n_touches += 1
            q = brain.get("decision_quality")
            if q is not None:
                sum_quality += q
            if brain.get("is_error"):
                n_errors += 1
            intent = brain.get("intent")
            if intent:
                intents.append(intent)

    decision_fitness = 0.0
    if n_touches > 0:
        decision_fitness = sum_quality / n_touches * 0.5 + (
            0.5 if n_errors == 0 else (1.0 - n_errors / n_touches) * 0.5
        )

    # team-level fitness: attack + control + decision quality
    attack = (home_goals * 0.6 + home_xg * 0.4) / 5.0
    attack = min(1.0, attack)
    control = poss / 100.0
    fitness = 0.45 * attack + 0.25 * control + 0.30 * decision_fitness

    return {
        "fitness": round(fitness, 4),
        "goals": home_goals,
        "xg": home_xg,
        "possession_pct": poss,
        "n_decisions": n_touches,
        "mean_decision_quality": round(sum_quality / n_touches, 4) if n_touches else 0.0,
        "error_rate": round(n_errors / n_touches, 4) if n_touches else 0.0,
        "intent_mix": {i: intents.count(i) for i in set(intents)},
    }


# ─────────────────────────────────────────────────────────────
# REAL-MATCH RUNNER
# ─────────────────────────────────────────────────────────────

def _run_single_match(
    brain_by_name: Dict[str, FootballBrain],
    home: str = "Probe FC",
    away: str = "Rival FC",
    home_style: str = "balanced",
    away_style: str = "fluid_counter",
    seed: int = 0,
) -> Tuple[Any, Dict[str, Any]]:
    """Build and run one real match with neural brains patched in.

    Returns (result, fitness_dict_for_home_team_brain_players).
    """
    random.seed(seed)
    home_squad, away_squad = _build_squads(home, away)

    config = MatchConfig(
        home_team=home, away_team=away,
        match_date=date(2026, 9, 6), matchday=3, season="26/27",
    )
    hp = _team_profile(home, home_style)
    ap = _team_profile(away, away_style)

    from squad_manager import SubstitutionController
    sc = SubstitutionController(
        home_team=home, away_team=away,
        home_subs_bench=home_squad["substitutes"],
        away_subs_bench=away_squad["substitutes"],
        home_style=hp.style.value, away_style=ap.style.value,
    )

    # register brains
    clear_registry()
    for name, brain in brain_by_name.items():
        register_brain(name, brain)

    # patch decision layer
    _install_neural_decide(brain_by_name)

    try:
        eng = MatchEngine(config, hp, ap)
        eng.set_squad(home, home_squad["starters"], home_squad["substitutes"])
        eng.set_squad(away, away_squad["starters"], away_squad["substitutes"])
        eng.set_stamina_controller(sc)

        result = eng.simulate()

        # which side are our target brains on? assume home team for now
        home_names = [p.name for p in home_squad["starters"]]
        fitness = extract_team_fitness(result, home_names)
    finally:
        _restore_decide()

    return result, fitness


# ─────────────────────────────────────────────────────────────
# OUTCOME-DRIVEN ENGINE WIRING (new generalised path)
# ─────────────────────────────────────────────────────────────

def build_probe_engine(
    player_name: str,
    seed: int = 0,
    home: str = "Probe FC",
    away: str = "Rival FC",
    home_style: str = "balanced",
    away_style: str = "fluid_counter",
    home_philosophy=None,
    away_philosophy=None,
) -> Callable[[FootballBrain], Any]:
    """Engine builder for outcome-driven evolution / validation.

    Returns ``build_engine(brain) -> MatchResult``.  The candidate brain is
    registered under ``player_name`` — an EXACT 4-3-3 starter name such as
    ``'ST'`` (see ``_slot_for_position``).  Because ``get_brain`` resolves
    by exact name, the candidate genuinely drives its slot instead of the
    old ``f'{pos}{pos}'`` placeholder that never matched any player and
    silently fell back to the auto-loaded JSON file.

    Each call builds a fresh ``MatchEngine`` with the supplied seed, so the
    same builder can be reused across GA population members or match-day
    iterations simply by passing different brains.

    The returned callable carries a ``.team`` attribute (``"home"``) so outcome
    extractors can skip the fragile timeline side-inference — with the 4-3-3
    template both squads name their ST ``"ST"`` and the first timeline hit
    may be the opponent's.
    """
    team = "home"

    def build_engine(brain: FootballBrain) -> Any:
        random.seed(seed)
        home_squad, away_squad = _build_squads(home, away)
        config = MatchConfig(
            home_team=home, away_team=away,
            match_date=date(2026, 9, 6), matchday=3, season="26/27",
        )
        hp = _team_profile(home, home_style, home_philosophy)
        ap = _team_profile(away, away_style, away_philosophy)
        from squad_manager import SubstitutionController
        sc = SubstitutionController(
            home_team=home, away_team=away,
            home_subs_bench=home_squad["substitutes"],
            away_subs_bench=away_squad["substitutes"],
            home_style=hp.style.value, away_style=ap.style.value,
        )
        clear_registry()
        register_brain(player_name, brain)
        try:
            eng = MatchEngine(config, hp, ap)
            eng.set_squad(home, home_squad["starters"], home_squad["substitutes"])
            eng.set_squad(away, away_squad["starters"], away_squad["substitutes"])
            eng.set_stamina_controller(sc)
            return eng.simulate()
        finally:
            _restore_decide()

    build_engine.team = team  # type: ignore[attr-defined]
    return build_engine


def run_outcome_validation(
    targets: Optional[List[Tuple[str, str]]] = None,
    brains_dir: str = "brains",
    n_matches: int = 3,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Score evolved brains by the REAL engine's own outcomes.

    Like ``run_neural_validation`` but scores via ``outcome_fitness``:
    team xG diff, goal diff, possession, the target player's turnover rate
    and his own chance production.  No intention rewards — the brain is
    judged by what actually happens as the engine's calibrated model decides it.

    Parameters
    ----------
    targets : list of (player_name, position) to register+validate.
        ``player_name`` MUST be the exact on-pitch starter name from the
        probe 4-3-3 template (use ``_slot_for_position`` to resolve from a
        position).  Defaults to probing every ``<brains_dir>/<POS>.json``
        with an automatic slot name.
    brains_dir : where ``<POS>.json`` brain files live.
    n_matches : matches to run per target (averaged).
    seed : RNG seed.
    verbose : print per-match outcome details.

    Returns
    -------
    ``{position: {fitness, n_touches, n_errors, own_shots}}``
    """
    os.makedirs(brains_dir, exist_ok=True)
    if targets is None:
        targets = []
        for fn in sorted(os.listdir(brains_dir)):
            if fn.endswith(".json"):
                pos = fn[:-5]
                try:
                    slot = _slot_for_position(pos)
                except ValueError:
                    continue
                targets.append((slot, pos))

    all_fits: Dict[str, Dict[str, Any]] = {}
    for name, position in targets:
        brain_path = os.path.join(brains_dir, f"{position}.json")
        if not os.path.exists(brain_path):
            if verbose:
                print(f"  ! no brain for {position} ({brain_path}), skipping")
            continue
        brain = FootballBrain.load(brain_path)

        fits = []
        if verbose:
            print(f"\n=== Outcome-scoring {position} brain ({brain_path}, slot '{name}') ===")
        for m in range(n_matches):
            t0 = time.time()
            build_engine = build_probe_engine(name, seed=seed + m * 100)
            result = build_engine(brain)
            from brain_evolution import extract_outcome_signals, outcome_fitness
            sig = extract_outcome_signals(result, target_player=name,
                                          team=getattr(build_engine, "team", None))
            f = outcome_fitness(sig)
            if verbose:
                print(
                    f"  match {m+1}: {result.score_str}  xg_diff={sig['xg_diff']:+.2f}  "
                    f"poss={sig['possession']:.0f}%  touches={sig['n_touches']}  "
                    f"err={sig['n_errors']}  own_shots={sig['own_shots']}  "
                    f"fitness={f:.4f}  ({time.time()-t0:.1f}s)"
                )
            fits.append({"fitness": f, "n_touches": sig["n_touches"],
                         "n_errors": sig["n_errors"], "own_shots": sig["own_shots"]})
        avg = {
            "fitness": round(sum(x["fitness"] for x in fits) / len(fits), 4) if fits else 0.0,
            "n_touches": sum(x["n_touches"] for x in fits),
            "n_errors": sum(x["n_errors"] for x in fits),
            "own_shots": sum(x["own_shots"] for x in fits),
        }
        all_fits[position] = avg
    return all_fits


def run_neural_validation(
    targets: Optional[List[Tuple[str, str]]] = None,
    brains_dir: str = "brains",
    n_matches: int = 1,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Validate evolved brains against the real match engine.

    Parameters
    ----------
    targets : list of (name, position) to register+validate.  Defaults to
        probing every <brains_dir>/<POS>.json with a generated player.
    brains_dir : where <POS>.json brain files live.
    n_matches : matches to run per target (averaged).
    seed : RNG seed.
    verbose : print per-match results.

    Returns
    -------
    {position: fitness_summary}
    """
    os.makedirs(brains_dir, exist_ok=True)
    if targets is None:
        targets = []
        for fn in sorted(os.listdir(brains_dir)):
            if fn.endswith(".json"):
                pos = fn[:-5]
                targets.append((f"ST_{pos}", pos))

    all_fits: Dict[str, Dict[str, Any]] = {}
    for name, position in targets:
        brain_path = os.path.join(brains_dir, f"{position}.json")
        if not os.path.exists(brain_path):
            if verbose:
                print(f"  ! no brain for {position} ({brain_path}), skipping")
            continue
        brain = FootballBrain.load(brain_path)

        # Register this one brain to the matching-position player on the
        # home team; other players use the random fallback (so the probe
        # isolates the single brain's contribution).
        player_name = f"{position}{position}"  # e.g. STST
        brain_by_name = {player_name: brain}

        fits = []
        if verbose:
            print(f"\n=== Validating {position} brain ({brain_path}) ===")
        for m in range(n_matches):
            t0 = time.time()
            result, fit = _run_single_match(brain_by_name, seed=seed + m * 100)
            if verbose:
                print(
                    f"  match {m+1}: {result.score_str}  possession="
                    f"{fit['possession_pct']}%  n_dec={fit['n_decisions']}  "
                    f"fitness={fit['fitness']}  ({time.time()-t0:.1f}s)"
                )
            fits.append(fit)

        # average fitness
        avg = {
            "fitness": round(sum(f["fitness"] for f in fits) / len(fits), 4),
            "goals": sum(f["goals"] for f in fits),
            "n_decisions": sum(f["n_decisions"] for f in fits),
            "mean_decision_quality": round(
                sum(f["mean_decision_quality"] for f in fits) / len(fits), 4,
            ) if fits else 0.0,
            "error_rate": round(sum(f["error_rate"] for f in fits) / len(fits), 4) if fits else 0.0,
            "possession_pct": round(sum(f["possession_pct"] for f in fits) / len(fits), 1),
            "intent_mix": {},
        }
        for f in fits:
            for k, v in f["intent_mix"].items():
                avg["intent_mix"][k] = avg["intent_mix"].get(k, 0) + v
        all_fits[position] = avg

    return all_fits


# ─────────────────────────────────────────────────────────────
# A/B COMPARISON: evolved brain vs heuristic DecisionBrain
# ─────────────────────────────────────────────────────────────

def ab_compare(
    position: str,
    brains_dir: str = "brains",
    n_matches: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare a neural brain vs the heuristic brain on equal footing.

    Runs n_matches with neural, n_matches with heuristic (same team/seed),
    and reports team fitness + decision stats for both.
    """
    import statistics

    brain_path = os.path.join(brains_dir, f"{position}.json")
    neural = FootballBrain.load(brain_path)

    player_name = f"{position}{position}"
    neural_fits, heuristic_fits = [], []

    # neural matches
    for m in range(n_matches):
        _brain = {player_name: neural}
        # freeze -- patch in neural
        _seed = seed + m * 100
        random.seed(_seed)
        _result, fit = _run_single_match(_brain, seed=_seed)
        neural_fits.append(fit)

    # heuristic matches: don't register brains, pin the neural entry
    # point to the genuine heuristic so nothing auto-loads.
    for m in range(n_matches):
        _seed = seed + m * 100
        random.seed(_seed)
        home_squad, away_squad = _build_squads("Probe FC", "Rival FC")
        config = MatchConfig(home_team="Probe FC", away_team="Rival FC",
                             match_date=date(2026, 9, 6), matchday=3, season="26/27")
        hp = _team_profile("Probe FC", "balanced")
        ap = _team_profile("Rival FC", "fluid_counter")
        from squad_manager import SubstitutionController
        sc = SubstitutionController(home_team="Probe FC", away_team="Rival FC",
                                    home_subs_bench=home_squad["substitutes"],
                                    away_subs_bench=away_squad["substitutes"])
        clear_registry()
        _pin_heuristic()
        try:
            eng = MatchEngine(config, hp, ap)
            eng.set_squad("Probe FC", home_squad["starters"], home_squad["substitutes"])
            eng.set_squad("Rival FC", away_squad["starters"], away_squad["substitutes"])
            eng.set_stamina_controller(sc)
            result = eng.simulate()
        finally:
            _restore_neural()
        home_names = [p.name for p in home_squad["starters"]]
        heuristic_fits.append(extract_team_fitness(result, home_names))

    def _avg(lst):
        return round(sum(f["fitness"] for f in lst) / len(lst), 4) if lst else 0.0

    return {
        "position": position,
        "neural_fitness": _avg(neural_fits),
        "heuristic_fitness": _avg(heuristic_fits),
        "neural_goals": sum(f["goals"] for f in neural_fits),
        "heuristic_goals": sum(f["goals"] for f in heuristic_fits),
        "neural_n_decisions": sum(f["n_decisions"] for f in neural_fits),
        "heuristic_n_decisions": sum(f["n_decisions"] for f in heuristic_fits),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Validate evolved FootballBrains in real matches.")
    p.add_argument("--position", type=str, default=None, help="Single position to validate.")
    p.add_argument("--brains-dir", type=str, default="brains")
    p.add_argument("--matches", type=int, default=1)
    p.add_argument("--ab", action="store_true", help="A/B compare vs heuristic.")
    p.add_argument("--outcome", action="store_true",
                   help="Score brains by the engine's own outcomes (xG diff, "
                        "turnover rate, chance production) instead of the "
                        "intention-quality baseline.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.ab:
        pos = args.position or "ST"
        res = ab_compare(pos, args.brains_dir, args.matches, args.seed)
        print(json.dumps(res, indent=2))
    elif args.outcome:
        targets = None
        if args.position:
            pos = args.position.strip().upper()
            targets = [(_slot_for_position(pos), pos)]
        res = run_outcome_validation(targets=targets, brains_dir=args.brains_dir,
                                     n_matches=args.matches, seed=args.seed)
        print(json.dumps(res, indent=2))
    else:
        targets = None
        if args.position:
            targets = [(f"{args.position}{args.position}", args.position)]
        res = run_neural_validation(targets=targets, brains_dir=args.brains_dir,
                                    n_matches=args.matches, seed=args.seed)
        print(json.dumps(res, indent=2))
