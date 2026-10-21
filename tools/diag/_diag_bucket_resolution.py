"""THE BUCKET RESOLUTION vs THE CORPUS SIZE — which scheme can be LEARNED?

THE CHAIN OF REASONING
----------------------
1. The per-position reward SCALE fixed the balance: the reward's swing is
   now 5x-16x the largest guard instead of 5x-20x smaller.
2. The GA still converges to "always play the best-average intent", and the
   per-state oracle is worth +0.17 of fitness above that.
3. The held-out test says the table's per-bucket ordering is only WEAKLY
   predictive of the realised outcome (38.6% agreement against a ~30%
   chance rate, on 44 held-out buckets).
4. And the reason is visible in the arithmetic: the corpus has 3,696 pairs
   spread over 11 positions x 64 buckets = 704 buckets, i.e. **5.25 pairs per
   bucket and ~2.5 per (bucket, intent) cell**.  `fit` shrinks every cell
   with fewer than `min_samples` samples toward its bucket mean, so MORE
   THAN HALF the cells are 75% bucket mean, and the ordering the GA is asked
   to learn is mostly the bucket's average.

So the bucket resolution is mismatched to the corpus: the table is asking
for 64 situations and being given 5 samples each.

WHAT THIS PROBE DOES
--------------------
Fit the table on half A with SEVERAL bucket schemes, and for each one
measure, on the held-out half B:

    shrunk     the share of cells pulled toward the bucket mean
    agree      the share of buckets where A's preferred intent is also B's
               best-paid intent, against the CHANCE rate for the number of
               intents B could judge
    paid       the realised mean on B of the intent A preferred, minus the
               realised mean on B of the intent B paid best (negative =
               A's preference was right more often than not)

A coarser scheme is only worth having if its held-out agreement beats the
finer scheme's.  The corpus is already collected, so every refit here is
FREE.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_bucket_resolution.py
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

#: (name, sensor indices used, one side each).  The current scheme uses all
#: six; the others drop features so the corpus can fill the cells.
SCHEMES = [
    ("6 features (current)", (16, 12, 13, 9, 15, 14)),
    ("5 features", (16, 12, 13, 9, 14)),
    ("4 features", (16, 12, 9, 14)),
    ("3 features", (16, 12, 14)),
    ("2 features", (16, 12)),
]


def _bucket(sensors, idxs):
    bits = "".join("1" if sensors[i] > 0.5 else "0" for i in idxs)
    return bits


class _Surrogate(FitnessSurrogate):
    """The real fit, with the bucket function swapped."""

    def __init__(self, idxs):
        super().__init__()
        self._idxs = idxs

    @staticmethod
    def bucket(sensors):  # replaced per instance below
        return ""

    def bucket(self, sensors):  # noqa: F811
        return _bucket(sensors, self._idxs)


def main():
    pairs = load_raw_corpus(RAW)
    half_a = pairs[0::2]
    half_b = pairs[1::2]
    print(f"corpus: {RAW}  ({len(pairs)} pairs)")
    print(f"fit half A: {len(half_a)} | held-out half B: {len(half_b)}")
    print()

    for name, idxs in SCHEMES:
        sur = _Surrogate(idxs)
        sur.fit(half_a, min_samples=3)
        cell = {}
        for sensors, intent, reward, _ev, pos in half_b:
            ik = intent.value if hasattr(intent, "value") else str(intent)
            cell.setdefault((pos, sur.bucket(sensors), ik), []).append(reward)

        shrunk = total = 0
        for pos, buckets in sur.table.items():
            for bucket, intents in buckets.items():
                for ik in intents:
                    total += 1
                    if sur.counts.get(pos, {}).get(bucket, {}).get(ik, 0) < 3:
                        shrunk += 1
        agree = n = 0
        paid_sum = 0.0
        chance_sum = 0.0
        nb = 0
        for pos in sorted(p for p in sur.table if p):
            fam = _role_family(pos) or ""
            allowed = set(ROLE_INTENTS.get(fam, ()))
            for bucket, intents in sur.table[pos].items():
                pref, pref_val = None, None
                for ik, v in intents.items():
                    if ik not in allowed:
                        continue
                    if pref_val is None or v > pref_val:
                        pref, pref_val = ik, v
                if pref is None:
                    continue
                paid_means = {}
                for ik in allowed:
                    vals = cell.get((pos, bucket, ik))
                    if vals and len(vals) >= MIN_B_SAMPLES:
                        paid_means[ik] = sum(vals) / len(vals)
                if len(paid_means) < 2:
                    continue
                best_ik = max(paid_means, key=lambda k: paid_means[k])
                n += 1
                agree += 1 if pref == best_ik else 0
                if pref in paid_means:
                    paid_sum += paid_means[pref] - paid_means[best_ik]
                chance_sum += 1.0 / len(paid_means)
                nb += 1
        if n == 0:
            print(f"{name:<22} shrunk {100.0*shrunk/max(1,total):>5.1f}%  "
                  f"NO HELD-OUT BUCKETS")
            continue
        print(f"{name:<22} shrunk {100.0*shrunk/max(1,total):>5.1f}%  "
              f"buckets {n:>4}  agree {100.0*agree/n:>5.1f}% "
              f"(chance {100.0*chance_sum/nb:>5.1f}%)  "
              f"paid {paid_sum/n:>+9.5f}")
    print()
    print("`paid` is what following the fitted preference COST on held-out")
    print("data: negative means the preference was right more often than not.")


if __name__ == "__main__":
    sys.exit(main())
