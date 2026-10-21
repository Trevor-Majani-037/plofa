"""Is the FORBIDDEN value still a veto AFTER the per-position scale?

THE QUESTION
------------
`FORBIDDEN_VALUE = -1.0` was chosen because the RAW threat return's range is
about [-0.6, +1.0] — so -1.0 is below every real reward, and the genetic
algorithm treats a role-inappropriate action as UNAVAILABLE rather than
merely unattractive.

The per-position scale divides every value by that position's own std, so
the scaled real rewards spread over roughly [-3, +3] and a FIXED -1.0 is no
longer below all of them.  If a legitimate action scales to -1.49 and the
forbidden one stays at -1.0, the GA is being offered a REASON TO CROSS.

This probe measures, per position, on the real corpus:

  * the raw range of the fitted cells (is the [-0.6, +1.0] premise true?)
  * the raw min cell and its SCALED value (does the veto still win?)
  * how many cells fall below the veto (how much of the tail is affected)

    .venv\\Scripts\\python.exe tools\\diag\\_diag_forbidden_scale.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from surrogate_collect import (  # noqa: E402
    FORBIDDEN_VALUE, FitnessSurrogate, load_raw_corpus,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(ROOT, "brains", "surrogate_pos_v5.raw.json"))


def main():
    sur = FitnessSurrogate().fit(load_raw_corpus(RAW))
    print(f"corpus: {RAW}")
    print(f"FORBIDDEN_VALUE = {FORBIDDEN_VALUE} (a FIXED raw-scale number)")
    print()
    print(f"{'pos':<6}{'raw min':>11}{'raw max':>11}{'scale':>10}"
          f"{'scaled min':>12}{'below veto':>13}")
    worst_ratio = 0.0
    for pos in sorted(p for p in sur.table if p):
        vals = [float(v) for b in sur.table[pos].values() for v in b.values()]
        if not vals:
            continue
        scale = sur.scales.get(pos, 1.0) or 1.0
        scaled = [v / scale for v in vals]
        below = sum(1 for v in scaled if v < FORBIDDEN_VALUE)
        worst_ratio = min(worst_ratio, min(scaled))
        print(f"{pos:<6}{min(vals):>11.5f}{max(vals):>11.5f}{scale:>10.5f}"
              f"{min(scaled):>12.5f}{below:>13}")
    print()
    print(f"the most negative SCALED real reward, over every position: "
          f"{worst_ratio:.5f}")
    print(f"the fixed veto sits at {FORBIDDEN_VALUE}")
    if worst_ratio < FORBIDDEN_VALUE:
        print()
        print("VERDICT: the fixed veto is NOT below every real reward any "
              "more.")
        print("A legitimate action scoring below -1.0 makes the forbidden "
              "action look BETTER,")
        print("so the GA is handed a reason to perform it.  The veto must be "
              "expressed in the")
        print("position's own units, exactly like every other value.")
    else:
        print()
        print("VERDICT: the fixed veto is still below every real reward.")


if __name__ == "__main__":
    sys.exit(main())
