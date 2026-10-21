"""DOES THE TABLE'S PER-BUCKET PREFERENCE PREDICT THE REALISED OUTCOME?

THE QUESTION, PRECISELY
------------------------
The GA converges to "always play the single best-average intent", and the
per-state oracle is worth +0.17 of fitness above it.  So either the table
encodes real, learnable situation-dependence and the GA cannot find it, or
the table's within-bucket ordering is NOISE and the constant is the correct
answer.

The earlier held-out check answered this wrongly (twice): it read the
TABLE's own values on held-out states, and a table's value for a bucket does
not change when the state changes.  **A held-out test that reads the model's
own output is not a held-out test.**

THE TEST THAT WORKS
-------------------
Split the corpus in half.  Fit the table on half A.  On half B, for each
(position, bucket) that has enough B samples to say anything:

  * what A's table PREFERS (its highest cell);
  * what B actually PAID for each intent (the realised mean per cell);
  * and therefore whether A's preference survives contact with B's data.

The statistics:

    agree      the share of buckets where A's preferred intent is also B's
               best-paid intent — against the CHANCE rate for the number of
               intents that have B samples, which is printed beside it
    paid       the realised mean on B of the intent A preferred, minus the
               realised mean on B of the intent B paid best
    n_buckets  how many buckets carried the comparison at all

The corpus is THIN — the table has ~2.5 samples per (position, bucket,
intent) cell — so "not learnable from this data" is a live possibility and
this is the measurement that settles it.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_bucket_predictive.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from role_features import ROLE_INTENTS, _role_family  # noqa: E402
from surrogate_collect import FitnessSurrogate, load_raw_corpus  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "brains", "surrogate_pos_v5.raw.json")

MIN_B_SAMPLES = 3


def main():
    pairs = load_raw_corpus(RAW)
    half_a = pairs[0::2]
    half_b = pairs[1::2]
    print(f"corpus: {RAW}  ({len(pairs)} pairs)")
    print(f"fit half A: {len(half_a)} | held-out half B: {len(half_b)}")
    print()
    sur = FitnessSurrogate().fit(half_a, min_samples=3)

    # the realised reward per (position, bucket, intent) on B
    cell = {}
    for sensors, intent, reward, _ev, pos in half_b:
        ik = intent.value if hasattr(intent, "value") else str(intent)
        cell.setdefault((pos, sur.bucket(sensors), ik), []).append(reward)

    print(f"{'pos':<6}{'buckets':>9}{'agree':>8}{'chance':>8}{'paid':>10}"
          f"{'n_int':>8}")
    print("-" * 52)
    tot_agree = tot_n = 0
    tot_paid = 0.0
    for pos in sorted(p for p in sur.table if p):
        fam = _role_family(pos) or ""
        allowed = set(ROLE_INTENTS.get(fam, ()))
        agree = 0
        n = 0
        paid = 0.0
        n_int = 0
        for bucket, intents in sur.table[pos].items():
            # what A prefers, among the intents this role may perform
            pref = None
            pref_val = None
            for ik, v in intents.items():
                if ik not in allowed:
                    continue
                if pref_val is None or v > pref_val:
                    pref, pref_val = ik, v
            if pref is None:
                continue
            # what B paid for, per intent, where there are enough samples
            paid_means = {}
            for ik in allowed:
                vals = cell.get((pos, bucket, ik))
                if vals and len(vals) >= MIN_B_SAMPLES:
                    paid_means[ik] = sum(vals) / len(vals)
            if len(paid_means) < 2:
                continue
            best_ik = max(paid_means, key=lambda k: paid_means[k])
            n += 1
            n_int += len(paid_means)
            if pref == best_ik:
                agree += 1
            if pref in paid_means:
                paid += paid_means[pref] - paid_means[best_ik]
        if n == 0:
            print(f"{pos:<6}{0:>9}{'-':>8}{'-':>8}{'-':>10}{'-':>8}")
            continue
        chance = sum(1.0 / len(cell.get((pos, b, ik), []) or [1]) * 0
                     for b in [] ) if False else 0.0
        # the chance rate: the mean over buckets of 1/(number of intents
        # that B could judge), i.e. what a blind table would score
        chance = 0.0
        nb = 0
        for bucket, intents in sur.table[pos].items():
            cnt = sum(1 for ik in allowed
                      if len(cell.get((pos, bucket, ik), [])) >= MIN_B_SAMPLES)
            if cnt >= 2:
                chance += 1.0 / cnt
                nb += 1
        chance = chance / nb if nb else 0.0
        tot_agree += agree
        tot_n += n
        tot_paid += paid
        print(f"{pos:<6}{n:>9}{100.0 * agree / n:>7.1f}%{100.0 * chance:>7.1f}%"
              f"{paid / n:>10.5f}{n_int / n:>8.1f}")
    print("-" * 52)
    if tot_n:
        print(f"ALL      {tot_n:>9}{100.0 * tot_agree / tot_n:>7.1f}%"
              f"{'-':>8}{tot_paid / tot_n:>10.5f}")
    print()
    print("`paid` is what following A's preference COST on B, in realised")
    print("reward.  Negative means A's preference was right more often than")
    print("not; ~0 means the ordering carries no information.")


if __name__ == "__main__":
    sys.exit(main())
