"""Who actually decides during surrogate_collect?  (milliseconds, no match)

The collector's own docstring says "Run n real matches with the heuristic
brain".  Since the 2026-09-09 live swap, event_chain calls
``NeuralDecisionBrain.decide`` directly, and ``NeuralDecisionBrain._resolve_brain``
AUTO-LOADS ``BRAIN_DIR/<position>.json``.  Neither ``build_surrogate`` nor
``__main__`` pins the heuristic, so the documented intent and the code may have
silently diverged -- the project's standing pathology.

This probe answers it by construction: build the exact scratch squads the
collector builds and resolve each player's brain.
"""
import sys

sys.path.insert(0, ".")

from surrogate_collect import (  # noqa: E402
    _COLLECTION_TEMPLATES, _build_match_components,
)
import brain_integration as BI  # noqa: E402


def main():
    eng = _build_match_components("Probe FC", "Rival FC", "balanced", "fluid_counter",
                                  template=_COLLECTION_TEMPLATES[0])
    print("BRAIN_DIR =", BI.BRAIN_DIR)
    print()
    print(f"{'player':6} {'pos':4} {'resolved':>10} {'arch':>18} {'params':>8}  source")
    print("-" * 78)

    from football_brain import FootballBrain
    seen_random = 0
    seen_file = 0
    for team in ("Probe FC", "Rival FC"):
        starters = eng.squads[team]["starters"]
        for p in starters:
            name = getattr(p, "name", "?")
            pos = getattr(p, "position", "?")
            # Exactly what NeuralDecisionBrain does at call time.
            b = BI.get_brain(name)
            from_file = b is not None
            if b is None:
                b = BI._brain_for_position(pos)
                from_file = b is not None
            if b is None:
                b = FootballBrain.random()
                src = "RANDOM (no trained file)"
                seen_random += 1
            else:
                src = "trained file" if from_file else "registry"
                seen_file += 1
            arch = [int(b.w1.shape[0]), int(b.w1.shape[1]),
                    int(b.w2.shape[1]), int(b.w3.shape[1])]
            print(f"{name:6} {pos:4} {type(b).__name__:>10} "
                  f"{str(arch):>18} {b.param_count:>8}  {src}")

    print()
    print(f"trained brains engaged: {seen_file}   random fallbacks: {seen_random}")
    print()
    print("VERDICT:", end=" ")
    if seen_random:
        print("COLLECTION WOULD RUN ON RANDOM BRAINS -> invalid data.")
    else:
        print("collection runs on the TRAINED T1 brains, NOT the heuristic "
              "its docstring claims -> the pin below is required.")

    # ── the pin ────────────────────────────────────────────────────────
    print()
    print("─" * 78)
    print("_pin_on_ball_policy")
    print("─" * 78)
    from surrogate_collect import _pin_on_ball_policy  # noqa: E402

    original = BI.NeuralDecisionBrain.decide
    restore = _pin_on_ball_policy("heuristic")
    pinned = BI.NeuralDecisionBrain.decide
    from decision_brain import DecisionBrain  # noqa: E402
    print(f"  pinned is DecisionBrain.decide  : {pinned is DecisionBrain.decide}")
    print(f"  pinned differs from original    : {pinned is not original}")
    restore()
    print(f"  restore returns the original    : "
          f"{BI.NeuralDecisionBrain.decide is original}")
    noop = _pin_on_ball_policy("neural")
    print(f"  'neural' leaves it alone        : "
          f"{BI.NeuralDecisionBrain.decide is original}")
    noop()
    try:
        _pin_on_ball_policy("bogus")
        print("  bad policy refused              : False  <-- BUG")
    except ValueError:
        print("  bad policy refused              : True")

    ok = (pinned is DecisionBrain.decide) and (BI.NeuralDecisionBrain.decide is original)
    print()
    print("PIN VERDICT:", "OK" if ok else "BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
