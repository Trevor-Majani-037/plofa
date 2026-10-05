"""SENSITIVITY: how many of AttackingMatrix's coefficients are load-bearing?

`_diag_receiver_layer_split.py` put 80% of receiver choices on one hand-written
expression (attacking_matrix.py:550):

    value = lane * (0.30*progress + 0.45*freedom + 0.20*depth
          + same_flank_bonus + winger_flank_bonus
          + midfield_coverage_bonus + false_nine_bonus)

Fitting those numbers to outcomes is only worth doing for the ones that
actually MOVE the receiver. This perturbs each coefficient and counts argmax
flips.

METHOD — recover the folded bonuses by algebra rather than reimplementing them.
The bonuses are inline literals folded into `_Option.value`, but from

    value = lane * (0.30*p + 0.45*f + 0.20*d + bonuses)

the bonus block is recoverable per option as

    bonuses = value/lane - (0.30*p + 0.45*f + 0.20*d)

so every coefficient can be perturbed and the expression recomputed in full.

FIDELITY GATE. Recomputing with the ORIGINAL coefficients must reproduce the
engine's own `_Option.value` exactly. If it does not, the algebra is wrong and
every number below would be fiction — so the gate runs first and aborts on
mismatch. This matters because a sensitivity study that silently measures a
reimplementation instead of the engine is worse than no study.

LIMITATIONS, stated up front:
  * This measures the sensitivity of the SCORING RULE. It is an UPPER BOUND on
    the sensitivity of the match: a flipped argmax can still be overridden
    downstream (wide-combo rule, phase directive), and the flipped receiver
    may barely change the outcome.
  * The four bonus terms are summed before recovery, so they are perturbed as
    ONE block. Their absence from the per-term table is NOT evidence they are
    inert.

Observation-only; consumes no RNG of its own. Run:
    .venv\\Scripts\\python.exe _diag_matrix_sensitivity.py [matches]
"""
from __future__ import annotations

import random
import sys

import attacking_matrix
from _diag_chance_coords import build_pair

W = {"progress": 0.30, "freedom": 0.45, "depth": 0.20}
ORDER = ("progress", "freedom", "depth")

CAPTURED: list = []          # (options, engine_target) per decide() call


def decompose(o):
    """-> (lane, progress, freedom, depth, bonus_block)"""
    lane = o.lane if o.lane else 1.0
    known = (W["progress"] * o.progress + W["freedom"] * o.freedom
             + W["depth"] * o.depth)
    return lane, o.progress, o.freedom, o.depth, o.value / lane - known


def rescore(o, weights, bonus_scale=1.0):
    lane, pr, fr, dp, bo = decompose(o)
    return lane * (weights["progress"] * pr + weights["freedom"] * fr
                   + weights["depth"] * dp + bonus_scale * bo)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    lines = []

    def p(m="", flush=False):
        print(m, flush=flush)
        lines.append(m)

    raw_opts = attacking_matrix.AttackingMatrix.__dict__["_build_options"]
    raw_decide = attacking_matrix.AttackingMatrix.__dict__["decide"]

    def build(*a, **kw):
        # `_build_options` is a @staticmethod: wrapping it as a classmethod
        # would pass `cls_` into `carrier` and desynchronise every later
        # positional (project rule: read the descriptor, branch on isinstance).
        opts = raw_opts.__func__(*a, **kw)
        CAPTURED.append([opts, None])
        return opts

    def decide(cls_, *a, **kw):
        out = raw_decide.__func__(cls_, *a, **kw)
        if CAPTURED:
            CAPTURED[-1][1] = getattr(out, "target", None)
        return out

    attacking_matrix.AttackingMatrix._build_options = staticmethod(build)
    attacking_matrix.AttackingMatrix.decide = classmethod(decide)

    pairs = [("Oxton", "Natrican"), ("Red Wolves", "Play City")]
    try:
        for i in range(n):
            h, a = pairs[i % len(pairs)]
            random.seed(4000 + i)
            p(f"  [{i+1}/{n}] {h} v {a} ...", flush=True)
            res = build_pair(h, a).simulate()
            p(f"      score {res.home_goals}-{res.away_goals}", flush=True)
    finally:
        attacking_matrix.AttackingMatrix._build_options = raw_opts
        attacking_matrix.AttackingMatrix.decide = raw_decide

    sets = [(o, t) for o, t in CAPTURED if o]
    multi = [(o, t) for o, t in sets if len(o) > 1]

    p()
    p(f"  option sets captured : {len(sets)}")
    p(f"  with >1 candidate    : {len(multi)}   (only these can flip)")

    # ---- FIDELITY GATE -------------------------------------------------
    worst = 0.0
    for opts, _t in sets:
        for o in opts:
            rebuilt = rescore(o, W, 1.0)
            worst = max(worst, abs(rebuilt - o.value))
    p()
    p(f"  FIDELITY  max |rebuilt - engine value| = {worst:.3e}")
    ok = worst < 1e-9
    p(f"  decomposition exact? {ok}")
    if not ok:
        p("  *** GATE FAILED — the algebra does not reproduce the engine.")
        p("      Every sensitivity number would be fiction. Stopping.")
        _dump(lines)
        return

    # ── ORACLE GATE ────────────────────────────────────────────────────
    # `AttackingMatrix.decide` does NOT take the global argmax: it selects from
    # SUBSETS (`best_far`, `best_close`, `target_opt`) chosen by a rule cascade
    # (counter / low_block / build_up / under_pressure...). Comparing a
    # perturbation against the global argmax therefore measures the wrong
    # thing -- and it showed up as 0/1195 agreement with the engine's pick.
    # The correct oracle: find the pool the engine actually chose FROM (by the
    # zone of the option it picked) and take the argmax within THAT pool.
    zone_of = {}
    for opts, t in sets:
        if t is None:
            continue
        for o in opts:
            if o.target is t or getattr(o.target, "name", None) == getattr(t, "name", None):
                zone_of[(id(opts), t.name)] = o.zone
                break

    def pool(opts, zone):
        sel = [o for o in opts if o.zone == zone]
        return sel if sel else list(opts)

    oracle_ok = oracle_tot = 0
    for opts, t in sets:
        if t is None:
            continue
        z = zone_of.get((id(opts), getattr(t, "name", None)))
        if z is None:
            continue
        oracle_tot += 1
        best = max(pool(opts, z), key=lambda o: o.value)
        if getattr(best.target, "name", None) == getattr(t, "name", None):
            oracle_ok += 1
    p()
    p(f"  ORACLE GATE  subset-argmax reproduces the engine's pick : "
      f"{oracle_ok}/{oracle_tot} ({100.0*oracle_ok/max(oracle_tot,1):.1f}%)")
    if oracle_tot == 0:
        p("  *** could not identify the engine's pool -- aborting ***")
        _dump(lines)
        return
    if oracle_ok / oracle_tot < 0.90:
        p("  *** ORACLE FAILS: the subset-argmax does NOT reproduce the engine's")
        p("      pick, so `decide` applies further selection beyond argmax within")
        p("      a zone. Every flip count below would be an UPPER BOUND of")
        p("      UNKNOWN correctness. Reporting the numbers as-is would repeat")
        p("      the error this gate exists to catch. ***")
        _dump(lines)
        return
    p("  (using the subset argmax as the perturbation baseline from here)")

    p()
    p("  SENSITIVITY — double one coefficient, count argmax flips")
    p("  (a doubling is a LARGE perturbation; 0% = inert at this scale)")
    p()
    rows = []
    for term in ORDER:
        w2 = dict(W)
        w2[term] = W[term] * 2.0
        flips = 0
        for opts, t in sets:
            if t is None:
                continue
            z = zone_of.get((id(opts), getattr(t, "name", None)))
            if z is None:
                continue
            base_name = getattr(t, "name", None)
            new = max(pool(opts, z), key=lambda o: rescore(o, w2, 1.0))
            if getattr(new.target, "name", None) != base_name:
                flips += 1
        rows.append((f"{term} ({W[term]:.2f}->{2*W[term]:.2f})", flips))
        p(f"    {term:<12} {W[term]:.2f} -> {2*W[term]:.2f}   flips "
          f"{flips:>4}/{oracle_tot} ({100.0*flips/max(oracle_tot,1):5.1f}%)")

    bflips = 0
    for opts, t in sets:
        if t is None:
            continue
        z = zone_of.get((id(opts), getattr(t, "name", None)))
        if z is None:
            continue
        new = max(pool(opts, z), key=lambda o: rescore(o, W, 2.0))
        if getattr(new.target, "name", None) != getattr(t, "name", None):
            bflips += 1
    rows.append(("ALL bonuses (x2 block)", bflips))
    p(f"    {'bonuses':<12} x2 (as one block)  flips "
      f"{bflips:>4}/{oracle_tot} ({100.0*bflips/max(oracle_tot,1):5.1f}%)")

    p()
    p("  LIMITATION: the four bonus terms (same_flank, winger_flank,")
    p("  midfield_coverage, false_nine) are summed before recovery, so they")
    p("  are perturbed as ONE block. Separating them needs the scoring line to")
    p("  expose them. Do NOT read their grouped number as per-term sensitivity,")
    p("  and do NOT read 'absent from the table' as 'inert'.")
    p()
    p("  This is the sensitivity of the SCORING RULE, an UPPER BOUND on match")
    p("  sensitivity: a flipped argmax can still be overridden downstream, and")
    p("  the new receiver may barely change the outcome.")
    p()
    inert = [name for name, f in rows if f == 0]
    live = [name for name, f in rows if f > 0]
    p(f"  >>> INERT at 2x : {inert if inert else 'none'}")
    p(f"  >>> LOAD-BEARING: {live if live else 'none'}")

    _dump(lines)


def _dump(lines):
    with open("_diag_matrix_sensitivity.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote _diag_matrix_sensitivity.txt")


if __name__ == "__main__":
    main()