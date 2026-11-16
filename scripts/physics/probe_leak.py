"""STEP 2: the same match, twice, in one process. What leaks?

`probe_determinism` shows the engine is byte-identical across fresh
interpreters (run 0 matches every time) but diverges by run 1 *within* one
process. So the global RNG is not the problem -- the harness reseeds it -- and
the leak is module-level state that a new `MatchEngine` inherits from the
previous match.

This narrows the leak by resetting one candidate at a time between two
identical runs and reporting which reset restores run 0's fingerprint.

Run:  $env:PYTHONHASHSEED="0"; python -m scripts.physics.probe_leak
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\PLOFA\plofa")
sys.path.insert(0, r"D:\PLOFA\plofa\tools\diag")

SEED = 1000


def _run():
    """One match, via the determinism probe's own fingerprint()."""
    from scripts.physics.probe_determinism import fingerprint
    return fingerprint(SEED)


def _sha(fp):
    import hashlib
    return hashlib.sha256("\n".join(fp).encode()).hexdigest()[:16]


def _reset_brains():
    import brain_integration as BI
    BI.clear_registry()
    for attr in ("_pos_brain_cache", "_brain_registry"):
        d = getattr(BI, attr, None)
        if isinstance(d, dict):
            d.clear()


def _reset_gates():
    from physics import adapter as A
    A._GATE_COUNTERS.clear()


def _reset_numpy():
    import numpy as np
    np.random.seed(0)


#: Already measured (2026-11-01) and all three FAILED to restore run 0, so
#: they are kept for the record but no longer re-run -- each costs a full
#: match. The live question is the engine-level singletons below.
CANDIDATES = {
    # "brains": _reset_brains,
    # "gate counters": _reset_gates,
    # "numpy global": _reset_numpy,
}


def _reset_engine_singletons():
    """Module-level brain/observer singletons in match_engine.

    These are loaded ONCE per process and shared by every match. If any of them
    carries recurrent state, a new match inherits it -- which is exactly the
    signature the probe measures.
    """
    import match_engine as ME
    for attr in ("_TEAM_PRESS_BRAIN", "_GAME_STATE_MIND", "_DEF_ACTION_BRAIN",
                 "_COGNITION_OBSERVER", "_DEFAULT_MEMORY_OBSERVER"):
        setattr(ME, attr, None)
    for attr in ("_TEAM_PRESS_LOADED", "_GAME_STATE_MIND_LOADED",
                 "_DEF_ACTION_LOADED"):
        setattr(ME, attr, False)
    rng = getattr(ME, "_COSMETIC_RNG", None)
    if rng is not None:
        rng.seed(0x5EEDC05)


def _reset_all():
    _reset_brains()
    _reset_gates()
    _reset_engine_singletons()


def main() -> int:
    base = _sha(_run())
    print(f"run 0 (fresh in this process)      {base}")

    print("-- one candidate at a time --")
    for label, reset in CANDIDATES.items():
        try:
            reset()
        except Exception as exc:                     # pragma: no cover
            print(f"run 1 after reset {label:<16} SKIPPED ({exc})")
            continue
        got = _sha(_run())
        verdict = "SAME" if got == base else "differs"
        print(f"run 1 after reset {label:<16} {got}   {verdict}")

    print("-- everything at once --")
    try:
        _reset_all()
        got = _sha(_run())
        print(f"run 1 after FULL reset         {got}   "
              f"{'SAME' if got == base else 'differs'}")
    except Exception as exc:                         # pragma: no cover
        print("full reset failed:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
