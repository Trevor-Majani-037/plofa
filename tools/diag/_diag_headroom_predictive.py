"""IS THE TABLE'S CONTEXT-DEPENDENCE PREDICTIVE?  (held out, on REALISED rewards)

WHY THIS FILE EXISTS
--------------------
Two earlier claims had to be thrown away before this one:

1. *"The headroom generalises, so the table's signal is real."*  The first
   version of the held-out check computed the oracle and the best constant
   from the TABLE's values on held-out states.  But the table's value for a
   bucket does not change when you change the state, so "held out" measured
   the bucket composition and nothing else.  **A held-out test that reads the
   model's own output is not a held-out test.**

2. *"The shrinkage is protecting the fit."*  The plan said not to touch it.
   The measured way to settle that is not to argue about it but to check
   whether the SHRUNKEN ordering predicts the REALISED reward on data the
   table never saw.

THE TEST
--------
Split the corpus in half.  Fit the table on half A.  Then, on half B:

  * the POLICY that follows A's table (pick the intent A's table prefers in
    this situation) — what did it actually collect?
  * the best CONSTANT policy on B — the intent with the highest REALISED
    mean reward on B (an upper bound on any constant, so the comparison
    cannot be rigged in the table's favour)
  * the headroom = policy minus constant

If the headroom is ~0 on held-out data, A's within-bucket ordering is NOISE,
the objective has no context-dependence to learn, and a GA that converges to
the best constant is not broken — it is CORRECT.  If the headroom is
positive, the signal is real and the GA is failing to find it.

Run with the shrinkage on (min_samples=3) and off (min_samples=1).

    .venv\\Scripts\\python.exe tools\\diag\\_diag_headroom_predictive.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from role_features import ROLE_INTENTS, _role_family  # noqa: E402
from surrogate_collect import FitnessSurrogate, load_raw_corpus  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "brains", "surrogate_pos_v5.raw.json")


def main():
    pairs = load_raw_corpus(RAW)
    half_a = pairs[0::2]
    half_b = pairs[1::2]
    print(f"corpus: {RAW}  ({len(pairs)} pairs)")
    print(f"fit half A: {len(half_a)} | held-out half B: {len(half_b)}")
    print()

    for min_samples in (3, 1):
        sur = FitnessSurrogate().fit(half_a, min_samples=min_samples)
        shrunk = 0
        total = 0
        for pos, buckets in sur.table.items():
            for bucket, intents in buckets.items():
                for ik in intents:
                    total += 1
                    if sur.counts.get(pos, {}).get(bucket, {}).get(ik, 0) < min_samples:
                        shrunk += 1
        print(f"=== min_samples={min_samples} "
              f"(cells shrunk toward the bucket mean: {shrunk}/{total}) ===")
        print(f"  {'pos':<6}{'policy B':>12}{'const B':>12}{'HEADROOM':>12}"
              f"{'agree':>8}{'n':>7}")
        for pos in sorted(p for p in sur.table if p):
            fam = _role_family(pos) or ""
            allowed = [ik for ik in ROLE_INTENTS.get(fam, ())]
            b_rows = [p for p in half_b if p[4] == pos]
            if len(b_rows) < 20:
                continue
            # the realised reward per intent, on B
            realised = {ik: [] for ik in allowed}
            policy_sum = 0.0
            policy_n = 0
            agree = 0
            for sensors, intent, reward, _ev, _pos in b_rows:
                ik = intent.value if hasattr(intent, "value") else str(intent)
                if ik in realised:
                    realised[ik].append(reward)
                # what A's table would have picked here
                key = sur.bucket(sensors)
                vals = []
                for cand in allowed:
                    v = sur._lookup(key, cand, pos)
                    if v is not None:
                        vals.append((v, cand))
                if not vals:
                    continue
                pick = max(vals)[1]
                policy_sum += reward if pick == ik else 0.0
                policy_n += 1
                # did A's ordering agree with B's realised ordering?
                order_a = [c for _, c in sorted(vals, reverse=True)]
                means_b = {c: (sum(realised[c]) / len(realised[c]))
                            for c in realised if realised[c]}
                if len(means_b) > 1:
                    order_b = [c for c in sorted(means_b, key=lambda c: -means_b[c])]
                    if order_a and order_b and order_a[0] == order_b[0]:
                        agree += 1
            # the constant policy's realised mean on B (the best one)
            const_means = {c: (sum(v) / len(v)) for c, v in realised.items() if v}
            if not const_means or policy_n == 0:
                continue
            const = max(const_means.values())
            policy = policy_sum / policy_n
            print(f"  {pos:<6}{policy:>12.5f}{const:>12.5f}"
                  f"{policy - const:>12.5f}"
                  f"{100.0 * agree / policy_n:>7.1f}%{policy_n:>7}")
        print()
    print("HEADROOM > 0 means A's table ordering predicts B's realised rewards:")
    print("the signal is REAL and the GA is failing to find it.")
    print("HEADROOM ~ 0 means the ordering is noise and the constant IS the")
    print("right answer — the objective is the ceiling.")


if __name__ == "__main__":
    sys.exit(main())
