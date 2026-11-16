"""WHICH PIPELINE ACTUALLY SERVED THE DECISION?

WHY THIS EXISTS
---------------
On 2026-11-16 a paired real-match gate ran the shot read-out OFF and ON and
returned **bit-identical matches** -- same score, same xG to two decimals, same
shot count, same fitness to four decimals.  On a synthetic corpus the same head
flips 134 of 1,200 states, so "identical" could not mean "no effect"; it meant
"not applied".

The cause was a fork.  ``brain_integration._maybe_situational`` did not delegate
to ``situational_decision`` -- it contained a second, complete copy of the
situational pipeline, and that copy is what every match ran.
``SituationalDecisionBrain`` was imported by nothing but its own tests.

A full match takes ~4-5 minutes and tells you *that* nothing happened.  This
probe takes seconds and tells you *where*.  It asks three questions, in the
order that matters:

  1. ROUTING   Does a decision requested through ``NeuralDecisionBrain.decide``
               actually enter ``SituationalDecisionBrain.decide``?  If not, the
               whole module is dead code and every measurement made against it
               describes a program that does not run.
  2. SEAM      Is ``_apply_shot_head`` reached inside that path, and does it
               return different logits when armed?
  3. EFFECT    With sampling pinned to argmax (so the question is the *choice*,
               not the dice), how many states change their chosen intent?

  4. LEAKAGE   The safety argument for a SHOOT-only read-out is that it "cannot
               reorder the other nine intents".  That argument is only true of
               the raw softmax: multiplying one probability by a constant
               rescales the rest by a common factor and preserves their order.
               It is NOT obviously true downstream, because the consequence
               critic blends probabilities with its own vector in PROBABILITY
               space, which is not scale-invariant.  So the probe runs a third
               arm with the critic pinned to identity and reports the gap.  If
               the two arms agree, the head is contained.  If they do not, the
               renormalisation is moving decisions the head was never supposed
               to touch, and that is a number, not an opinion.

WHAT IT DOES NOT DO
-------------------
It does not judge whether the head is any good.  Whether a changed intent is a
*correct* change is the gate's question and the fit report's question.  This
probe only answers "is the wire connected", which is the thing that was wrong.

Usage:
    .venv/Scripts/python.exe tools/diag/_route_probe.py --position ST --states 200
    PLOFA_SHOT_HEAD=1 .venv/Scripts/python.exe tools/diag/_route_probe.py --position ST
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from attacking_matrix import shot_score  # noqa: E402
from brain_evolution import (  # noqa: E402
    _DummyPositionEngine, _carrier, _role_coords_to_players,
    _OUTFIELD_TM_POOL, _OPP_POOL, random_game_state, random_game_state_scoring,
)


def _scene(position: str, rng: random.Random, scoring: bool):
    """One synthetic but structurally real scene, as the engine would build it."""
    st = (random_game_state_scoring if scoring else random_game_state)(rng, position)
    coords = {}
    for i, (tx, ty) in enumerate(st["teammates"]):
        coords[f"t{i}"] = (tx, ty)
    for i, (dx, dy) in enumerate(st["defenders"]):
        coords[f"d{i}"] = (dx, dy)
    pe = _DummyPositionEngine(coords)
    teammates = _role_coords_to_players(st["teammates"], "t", _OUTFIELD_TM_POOL, rng)
    defenders = _role_coords_to_players(st["defenders"], "d", _OPP_POOL, rng)
    carrier = _carrier(position)
    score = shot_score(st["x"], st["y"], defenders, pe,
                       st["attacks_right"], dna=None)
    return st, pe, teammates, defenders, carrier, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="ST")
    ap.add_argument("--states", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scoring-frac", type=float, default=0.5,
                    help="share of scenes drawn from the final third")
    args = ap.parse_args()

    from brain_integration import (
        NeuralDecisionBrain, clear_registry, register_brain,
    )
    import brain_integration
    import situational_decision

    brain = None
    brain_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "brains", f"{args.position}.json")
    from situational_brain import SituationalBrain
    brain = SituationalBrain.load(brain_path)

    # ── COUNTERS ─────────────────────────────────────────────────
    stats = {"decide_calls": 0, "situational_entered": 0,
             "head_calls": 0, "head_changed": 0, "errors": 0,
             "first_error": ""}

    _real_decide = situational_decision.SituationalDecisionBrain.decide
    _real_head = situational_decision._apply_shot_head

    def counting_decide(*a, **kw):
        stats["situational_entered"] += 1
        return _real_decide(*a, **kw)

    def counting_head(player, logits, hidden):
        stats["head_calls"] += 1
        out = _real_head(player, logits, hidden)
        if not np.allclose(np.asarray(out), np.asarray(logits)):
            stats["head_changed"] += 1
        return out

    situational_decision.SituationalDecisionBrain.decide = staticmethod(counting_decide)
    situational_decision._apply_shot_head = counting_head

    # Pin the dice: the question is the CHOICE, not the sampling.
    _real_sample = brain_integration._sample_from_probs
    brain_integration._sample_from_probs = lambda probs, temperature: int(np.argmax(probs))

    # The critic, so the leakage arm can be run with it pinned to identity.
    # NOTE: ``decide`` imports ``_apply_consequence`` from ``brain_integration``
    # INSIDE the function body, so the module attribute is what to patch — a
    # name set on ``situational_decision`` would never be read.
    _real_cons = brain_integration._apply_consequence

    def _identity_cons(probs, base24, position, record_trace=False):
        return probs, None

    def _name(d) -> str:
        it = getattr(d, "intent", d)
        return str(getattr(it, "name", it))

    rng = random.Random(args.seed)
    rows = []
    try:
        for i in range(args.states):
            st, pe, teammates, defenders, carrier, score = _scene(
                args.position, rng, rng.random() < args.scoring_frac)
            register_brain(carrier.name, brain)

            def one(armed: bool, no_critic: bool = False):
                os.environ["PLOFA_SHOT_HEAD"] = "1" if armed else "0"
                brain_integration._apply_consequence = (
                    _identity_cons if no_critic else _real_cons)
                stats["decide_calls"] += 1
                return NeuralDecisionBrain.decide(
                    carrier, st["x"], st["y"], teammates, defenders, pe,
                    None, st["under_pressure"], st["attacks_right"],
                    st["game_state"], st["minute"], None, False, None,
                )

            try:
                off = one(False)
                on = one(True)
                on_bare = one(True, no_critic=True)
                off_bare = one(False, no_critic=True)
            except Exception as e:                      # noqa: BLE001
                stats["errors"] += 1
                if not stats["first_error"]:
                    stats["first_error"] = f"{type(e).__name__}: {e}"
                continue

            rows.append((float(score), _name(off), _name(on),
                         _name(off_bare), _name(on_bare)))
    finally:
        situational_decision.SituationalDecisionBrain.decide = staticmethod(_real_decide)
        situational_decision._apply_shot_head = _real_head
        brain_integration._sample_from_probs = _real_sample
        brain_integration._apply_consequence = _real_cons
        clear_registry()

    print(f"\n=== ROUTE PROBE — {args.position} "
          f"({args.states} states, scoring_frac={args.scoring_frac}) ===")
    print(f"1) ROUTING   decide() entered {stats['situational_entered']} / "
          f"{stats['decide_calls']} decisions")
    if stats["situational_entered"] == 0:
        print("   *** DEAD CODE: the situational pipeline is NOT on the live route ***")
    print(f"2) SEAM      _apply_shot_head reached {stats['head_calls']} times, "
          f"changed logits {stats['head_changed']} times "
          f"(PLOFA_SHOT_HEAD is now {os.environ.get('PLOFA_SHOT_HEAD', '<unset>')})")
    if stats["errors"]:
        print(f"   *** {stats['errors']} decisions raised: {stats['first_error']} ***")

    if not rows:
        print("\nNo usable rows — the path raised on every state.")
        return

    flips = [r for r in rows if r[1] != r[2]]
    to_shoot = [r for r in flips if r[2] == "SHOOT"]
    away = [r for r in flips if r[1] == "SHOOT"]
    print(f"3) EFFECT    chosen intent changed on {len(flips)} / {len(rows)} states "
          f"({100.0 * len(flips) / len(rows):.1f}%)")
    print(f"             -> SHOOT  {len(to_shoot)}   (became the choice)")
    print(f"             -> other  {len(away)}   (SHOOT was the choice, no longer is)")
    if flips:
        scored = sorted(r[0] for r in flips)
        print(f"             shot_score on flipped states: "
              f"p10 {scored[len(scored)//10]:.2f}  "
              f"median {scored[len(scored)//2]:.2f}  "
              f"p90 {scored[min(len(scored)-1, 9*len(scored)//10)]:.2f}")

    before = sum(1 for r in rows if r[1] == "SHOOT")
    after = sum(1 for r in rows if r[2] == "SHOOT")
    print(f"\n    SHOOT chosen: OFF {before} -> ON {after}  (of {len(rows)})")

    # ── 4) LEAKAGE ───────────────────────────────────────────────
    bare = [r for r in rows if r[3] != r[4]]
    print(f"\n4) LEAKAGE   head effect with the critic PINNED OFF: "
          f"{len(bare)} / {len(rows)} states "
          f"({100.0 * len(bare) / len(rows):.1f}%)")
    print(f"             head effect with the critic LIVE:        "
          f"{len(flips)} / {len(rows)} states "
          f"({100.0 * len(flips) / len(rows):.1f}%)")
    extra = len(flips) - len(bare)
    if extra > 0:
        print(f"             *** {extra} additional flips come from the critic's "
              f"renormalisation, NOT from the read-out. ***")
        print("             The 'cannot reorder the other nine intents' argument "
              "holds only for the raw softmax;")
        print("             the consequence blend is not scale-invariant, so a "
              "SHOOT-only change can reach decisions")
        print("             the head was never meant to touch.")
    elif extra < 0:
        print(f"             the critic suppresses {len(bare) - len(flips)} of the "
              f"head's bare flips")
    else:
        print("             identical: the critic neither adds nor removes "
              "read-out flips on this sample.")

    bare_shoot_on = sum(1 for r in rows if r[4] == "SHOOT")
    bare_shoot_off = sum(1 for r in rows if r[3] == "SHOOT")
    print(f"\n    SHOOT chosen (critic off): OFF {bare_shoot_off} -> ON {bare_shoot_on}")


if __name__ == "__main__":
    main()
