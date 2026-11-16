"""Validate a fully-neural XI against the heuristic XI in real matches.

This is the CORRECT full-team comparison.  The probe helpers in
``match_probe`` register brains under ``{POS}{POS}`` (e.g. "STST"), but real
players are named "ST", "CM1", "CM2", "CB1", "CB2", etc.  Because
``NeuralDecisionBrain`` looks up brains by exact ``player.name``, that wiring
silently fell back to *random* brains — so those A/B numbers did not test the
evolved nets.

Here we register each position's brain to every starter whose ``position``
matches, then run real matches:
    • NEURAL:  all 11 roles driven by learned brains.
    • HEURISTIC: the unchanged DecisionBrain (baseline).

Both use identical squads, seed, and opponent.

RESTORED 2026-11-08
-------------------
The module was missing from the repo, which made ``brain_self_trainer.gate()``
raise ModuleNotFoundError at stage [4/6] — no challenger could ever be promoted
or rejected.  Restored from ``.kilo/worktrees/aspiring-gem/`` with two fixes,
both found by checking rather than assuming:

1. **Generation dispatch.**  The original loaded every file with
   ``FootballBrain.load``.  The shipped set is ``arch_version 3``
   (``SituationalBrain``), and ``FootballBrain.load`` on those raises
   ``KeyError: 'w1'`` — the gate would have crashed on the first brain.
   ``load_any_brain`` now dispatches on the file's ``arch_version``.

2. **Binding is verifiable.**  ``register_full_xi`` returns a census
   (how many brains bound, and of which generation) instead of a bare count,
   so a gate that binds nothing can no longer look like a gate that ran.  The
   count is still exposed as ``_n_bound`` for existing callers.

The heuristic arm also gets a kwargs-tolerant pin: ``DecisionBrain.decide``
takes ``rule_exec`` where ``NeuralDecisionBrain.decide`` takes
``coach_instructions``, so pinning the raw staticmethod raises TypeError the
moment the engine passes the newer keyword.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import date
from typing import Any, Dict, List, Optional

from football_brain import FootballBrain
from situational_brain import SituationalBrain
from brain_integration import (
    NeuralDecisionBrain, register_brain, clear_registry,
)
from decision_brain import PlayerIntent, DecisionBrain
import match_probe

#: position (as on the player object) -> brain file stem
POSITION_BRAIN_MAP = {
    "GK": "GK", "CB": "CB", "LB": "LB", "RB": "RB", "CDM": "CDM",
    "CM": "CM", "CAM": "CAM", "LW": "LW", "RW": "RW",
    "ST": "ST", "CF": "CF",
}

#: The hand-calibrated heuristic, captured before any monkeypatching.
#: event_chain.py calls NeuralDecisionBrain.decide() directly, so the
#: heuristic baseline is obtained by temporarily pinning the neural entry
#: point to this function for the duration of a heuristic match.
_HEURISTIC_DECIDE = DecisionBrain.decide


def brain_arch_version(path: str) -> int:
    """The ``arch_version`` a brain file declares, defaulting to 1.

    Reads the top-level key first (``SituationalBrain.save`` writes it there),
    then ``meta``.  A v1 ``FootballBrain`` has no key at all and is version 1.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if "arch_version" in data:
        return int(data["arch_version"])
    meta = data.get("meta") or {}
    return int(meta.get("arch_version", 1))


def load_any_brain(path: str):
    """Load whichever brain generation the file holds.

    ``arch_version >= 3`` is the situational generation (attention + GRU);
    everything below it is a plain MLP ``FootballBrain``.  Loading a v3 file
    as a FootballBrain raises ``KeyError: 'w1'``, which is why this exists.
    """
    if brain_arch_version(path) >= 3:
        return SituationalBrain.load(path)
    return FootballBrain.load(path)


def register_full_xi(brains_dir: str, starters: List[Any]) -> Dict[str, Any]:
    """Register every position's brain to all matching starters.

    Returns a census, not a bare count::

        {"bound": int, "skipped": [names], "arch": {version: n}, "brains": {...}}

    The census matters.  A gate that binds zero brains runs a match between two
    unregistered teams and reports a difference — indistinguishable from a real
    result unless the binding is recorded.
    """
    clear_registry()
    bound = 0
    skipped: List[str] = []
    arch_counts: Dict[int, int] = {}
    per_player: Dict[str, str] = {}
    for p in starters:
        stem = POSITION_BRAIN_MAP.get(getattr(p, "position", ""))
        if not stem:
            skipped.append(getattr(p, "name", "?"))
            continue
        path = os.path.join(brains_dir, f"{stem}.json")
        if not os.path.exists(path):
            skipped.append(getattr(p, "name", "?"))
            continue
        brain = load_any_brain(path)
        register_brain(p.name, brain)
        bound += 1
        arch = brain_arch_version(path)
        arch_counts[arch] = arch_counts.get(arch, 0) + 1
        per_player[p.name] = f"{stem}:v{arch}"
    return {"bound": bound, "skipped": skipped,
            "arch": arch_counts, "brains": per_player}


def _run_neural(brains_dir: str, seed: int, home_style: str, away_style: str):
    """One real match with a full neural XI.

    NOTE ON ``seed``: this function does NOT seed internally.  The caller must
    ``random.seed(s)`` immediately before each arm — that is how
    ``brain_self_trainer.gate`` gives the challenger and the incumbent the same
    draw.  Seeding here as well would be harmless but would hide a caller that
    forgot, so it is deliberately left out.
    """
    home_squad, away_squad = match_probe._build_squads("Probe FC", "Rival FC")
    census = register_full_xi(brains_dir, home_squad["starters"])
    try:
        config = match_probe.MatchConfig(
            home_team="Probe FC", away_team="Rival FC",
            match_date=date(2026, 9, 6), matchday=3, season="26/27",
        )
        hp = match_probe._team_profile("Probe FC", home_style)
        ap = match_probe._team_profile("Rival FC", away_style)
        eng = match_probe.MatchEngine(config, hp, ap)
        eng.set_squad("Probe FC", home_squad["starters"], home_squad["substitutes"])
        eng.set_squad("Rival FC", away_squad["starters"], away_squad["substitutes"])
        result = eng.simulate()
        home_names = [p.name for p in home_squad["starters"]]
        fitness = match_probe.extract_team_fitness(result, home_names)
        fitness["_n_bound"] = census["bound"]
        fitness["_bound_arch"] = census["arch"]
        fitness["_bound_skipped"] = census["skipped"]
        return result, fitness
    finally:
        clear_registry()


def _heuristic_shim(*args, **kwargs):
    """``DecisionBrain.decide`` with the newer keywords dropped.

    ``DecisionBrain.decide`` accepts ``rule_exec``; ``NeuralDecisionBrain.decide``
    accepts ``coach_instructions``.  Pinning the raw staticmethod therefore
    raises TypeError as soon as the engine passes the newer keyword, so drop
    anything the heuristic does not understand.
    """
    allowed = {"record_trace"}
    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return _HEURISTIC_DECIDE(*args, **kwargs)


def _run_heuristic(seed: int, home_style: str, away_style: str):
    home_squad, away_squad = match_probe._build_squads("Probe FC", "Rival FC")
    clear_registry()
    # Temporarily pin the neural entry point to the genuine heuristic so
    # substitutes/unregistered players cannot auto-load neural brains, and
    # turn the team press controller OFF: the heuristic arm is the OLD system
    # (heuristic on-ball + pure role-rate Bernoulli off-ball).
    saved_neural = NeuralDecisionBrain.decide
    NeuralDecisionBrain.decide = staticmethod(_heuristic_shim)
    from match_engine import set_team_press_auto
    set_team_press_auto(False)
    try:
        config = match_probe.MatchConfig(
            home_team="Probe FC", away_team="Rival FC",
            match_date=date(2026, 9, 6), matchday=3, season="26/27",
        )
        hp = match_probe._team_profile("Probe FC", home_style)
        ap = match_probe._team_profile("Rival FC", away_style)
        eng = match_probe.MatchEngine(config, hp, ap)
        eng.set_squad("Probe FC", home_squad["starters"], home_squad["substitutes"])
        eng.set_squad("Rival FC", away_squad["starters"], away_squad["substitutes"])
        result = eng.simulate()
        home_names = [p.name for p in home_squad["starters"]]
        fitness = match_probe.extract_team_fitness(result, home_names)
        return result, fitness
    finally:
        NeuralDecisionBrain.decide = saved_neural
        set_team_press_auto(True)
        clear_registry()


def main():
    p = argparse.ArgumentParser(description="Full-XI neural vs heuristic validation.")
    p.add_argument("--brains", default="brains")
    p.add_argument("--matches", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--home-style", default="balanced")
    p.add_argument("--away-style", default="fluid_counter")
    args = p.parse_args()

    print(f"=== Full-XI Neural vs Heuristic ({args.matches} matches each) ===")

    n_fits, h_fits = [], []
    t0 = time.time()
    for m in range(args.matches):
        seed = args.seed + m * 100
        random.seed(seed)
        r, nf = _run_neural(args.brains, seed, args.home_style, args.away_style)
        print(f"  neural m{m+1}: {r.score_str}  "
              f"fit={nf['fitness']:.3f}  goals={nf['goals']}  "
              f"possession={nf.get('possession_pct')}%  "
              f"bound={nf.get('_n_bound')} arch={nf.get('_bound_arch')}")
        n_fits.append(nf)

        random.seed(seed + 7)
        r, hf = _run_heuristic(seed + 7, args.home_style, args.away_style)
        print(f"  heur   m{m+1}: {r.score_str}  "
              f"fit={hf['fitness']:.3f}  goals={hf['goals']}  "
              f"possession={hf.get('possession_pct')}%")
        h_fits.append(hf)

    def _avg(lst, key):
        return round(sum(f[key] for f in lst) / len(lst), 4)

    summary = {
        "neural_fitness": _avg(n_fits, "fitness"),
        "heuristic_fitness": _avg(h_fits, "fitness"),
        "neural_goals": sum(f["goals"] for f in n_fits),
        "heuristic_goals": sum(f["goals"] for f in h_fits),
        "neural_possession": _avg(n_fits, "possession_pct"),
        "heuristic_possession": _avg(h_fits, "possession_pct"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    summary["goal_diff"] = summary["neural_goals"] - summary["heuristic_goals"]
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
