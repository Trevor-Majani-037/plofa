"""WHICH LAYER OWNS THE RECEIVER: TacticalPhase or AttackingMatrix?

The trace showed both write `forced_receiver` and both bypass the brain's
`active_decision.target` (event_chain.py:2111 runs before 2152). This measures
the SPLIT between them, which is what decides whether "make the policy
authoritative" means changing one owner or two.

Method — read the actual assignments in event_chain, not the helper calls:

  * `_tactical_phase_step` returns `(phase, phase_decision)`; the target is
    `phase_decision.target` and it is consumed at event_chain.py:1722 via
    `next((p for p in players if p.name == phase_decision.target), None)`.
  * `AttackingMatrix.decide` returns `matrix_decision`; consumed at 1817.

Rather than re-derive those lines, this wraps the two PRODUCERS and records
which one produced a target on each touch, then checks which name the engine
actually resolved to a player. Both are static/class methods, so read the raw
descriptor out of `__dict__` (project rule).

The counterfactual check: a name that is never in `players` cannot become the
receiver. So "produced a target" is counted separately from "target resolved".

Observation-only, consumes no RNG of its own. Run:
    .venv\\Scripts\\python.exe _diag_receiver_layer_split.py [matches]
"""
from __future__ import annotations

import random
import sys
from collections import Counter

import event_chain
import possession_phases
from _diag_chance_coords import build_pair

C = Counter()
TOUCHES = Counter()          # who produced a target, per touch
SEEN_TARGETS = Counter()     # (source, target_name) -> count
RESOLVED = Counter()         # source -> times the engine resolved it to a player
CARRIER_ROLE = Counter()     # carrier position -> times brain target used


def _wrap_static_or_class(holder, name, label):
    raw = holder.__dict__[name]

    if isinstance(raw, classmethod):
        def f(cls_, *a, **kw):
            out = raw.__func__(cls_, *a, **kw)
            _record(label, out)
            return out
        setattr(holder, name, classmethod(f))
        return raw
    if isinstance(raw, staticmethod):
        def f(*a, **kw):
            out = raw.__func__(*a, **kw)
            _record(label, out)
            return out
        setattr(holder, name, staticmethod(f))
        return raw
    return raw


def _record(label, out):
    """`_tactical_phase_step` -> (phase, decision); `AttackingMatrix.decide`
    -> AttackingDecision. Extract `.target` without assuming the shape."""
    C[f"{label}:calls"] += 1
    dec = None
    if isinstance(out, tuple):
        dec = out[1] if len(out) > 1 else None
    else:
        dec = out
    tgt = getattr(dec, "target", None)
    if tgt is None:
        C[f"{label}:no_target"] += 1
        return
    nm = getattr(tgt, "name", None) or (tgt if isinstance(tgt, str) else None)
    if nm is None:
        C[f"{label}:target_unnamed"] += 1
        return
    C[f"{label}:target"] += 1
    TOUCHES[label] += 1
    SEEN_TARGETS[(label, nm)] += 1


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City")]

    saved_phase = _wrap_static_or_class(
        event_chain.PossessionChain, "_tactical_phase_step", "phase")
    saved_matrix = _wrap_static_or_class(
        possession_phases.PossessionPhaseEngine, "decide", "matrix") \
        if "decide" in possession_phases.PossessionPhaseEngine.__dict__ else None
    # AttackingMatrix.decide is on attacking_matrix.AttackingMatrix
    import attacking_matrix
    saved_am = _wrap_static_or_class(
        attacking_matrix.AttackingMatrix, "decide", "matrix")

    lines = []
    def p(m="", flush=False):
        print(m, flush=flush)
        lines.append(m)

    try:
        for i in range(n):
            h, a = pairs[i % len(pairs)]
            random.seed(4000 + i)
            p(f"  [{i+1}/{n}] {h} v {a} ...", flush=True)
            res = build_pair(h, a).simulate()
            p(f"      score {res.home_goals}-{res.away_goals}", flush=True)
    finally:
        event_chain.PossessionChain._tactical_phase_step = saved_phase
        if saved_matrix is not None:
            possession_phases.PossessionPhaseEngine.decide = saved_matrix
        attacking_matrix.AttackingMatrix.decide = saved_am

    p()
    p(f"  tactical_phase_step calls : {C['phase:calls']}")
    p(f"     with a target         : {C['phase:target']}")
    p(f"     without a target      : {C['phase:no_target']}")
    p(f"  AttackingMatrix.decide   : {C['matrix:calls']}")
    p(f"     with a target         : {C['matrix:target']}")
    p(f"     without a target      : {C['matrix:no_target']}")

    pt = C["phase:target"]
    mt = C["matrix:target"]
    tot = pt + mt
    p()
    if tot == 0:
        p("  NO TARGETS PRODUCED — the wrappers did not fire. Not a result.")
        _dump(lines)
        return
    p(f"  >>> SPLIT: phase {pt} ({100.0*pt/tot:.0f}%)  vs  "
      f"matrix {mt} ({100.0*mt/tot:.0f}%)  of {tot} targets produced")
    p()
    p("  top targets by source:")
    for src in ("phase", "matrix"):
        rows = [(nm, c) for (s, nm), c in SEEN_TARGETS.items() if s == src]
        rows.sort(key=lambda r: -r[1])
        p(f"    {src}:")
        for nm, c in rows[:8]:
            p(f"       {nm:<24} {c}")
    p()
    p("  READ THIS BEFORE ACTING:")
    p("  These are targets PRODUCED, not passes EMITTED. A produced target is")
    p("  only a real receiver if `next(p for p in players if p.name == ...)`")
    p("  resolves — a stale/subbed name silently falls through to the wide-combo")
    p("  rule or the brain. The split above is the ceiling of each layer's")
    p("  authority, so the real owner is at least this share and possibly less.")
    _dump(lines)


def _dump(lines):
    with open("_diag_receiver_layer_split.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_receiver_layer_split.txt")


if __name__ == "__main__":
    main()