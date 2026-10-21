"""MEASURE the per-position reward spread, raw and scaled, on the real corpus.

WHY
---
The v5 census failed because the reward's spread is NOT comparable across
positions: the guards in `synthetic_fitness` carry FIXED weights (0.05 each),
so wherever the reward was flat the guards were ~15x the football and the GA
optimised peakedness instead of football (mean_conf 0.78-0.97 vs the shipped
0.19-0.38, and 8 of 11 positions collapsed to <=3 intents).

The fix is a per-position SCALE: the count-weighted std of that position's
FITTED cells, applied at read time in `expected_success`.

This probe answers the two questions the fix has to be judged on:

  1. how big is the RAW spread ratio between positions (is the diagnosis
     right at all?), and
  2. after the scale, how big is the READ spread ratio — the number the GA
     actually compares against its fixed-weight guards?

It prints, per position: the number of cells, the RAW weighted std, the
SCALE, the READ spread (the std of the values `expected_success` returns over
every cell that position can read, which is what the GA samples), and the
raw/scaled ratio between the flattest and the widest position.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_scale_spread.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from decision_brain import PlayerIntent  # noqa: E402
from surrogate_collect import (  # noqa: E402
    FORBIDDEN_VALUE, FitnessSurrogate, load_raw_corpus,
)

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "brains", "surrogate_pos_v5.raw.json")


def _wstd(vals, wts):
    tw = sum(wts)
    if tw <= 0:
        return 0.0
    m = sum(v * w for v, w in zip(vals, wts)) / tw
    var = sum(w * (v - m) ** 2 for v, w in zip(vals, wts)) / tw
    return var ** 0.5


def _read_spread(sur, pos):
    """The std of the values the GA can actually READ for this position.

    Every (bucket, intent) the position can read, through the real
    `expected_success` — so the family and global fallbacks are included,
    because the GA samples them too.
    """
    vals = []
    buckets = set(sur.table.get(pos, {}))
    buckets |= set(sur.family_table.get(_fam(pos), {}))
    buckets |= set(sur.table.get("", {}))
    for bucket in buckets:
        for ik in _readable_intents(sur, pos, bucket):
            v = sur.expected_success(_sensors(bucket), PlayerIntent(ik), pos)
            if v is None or v == FORBIDDEN_VALUE:
                continue
            vals.append(float(v))
    if len(vals) < 2:
        return 0.0, len(vals)
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return var ** 0.5, len(vals)


def _readable_intents(sur, pos, bucket):
    own = sur.table.get(pos, {}).get(bucket, {})
    if own:
        return list(own)
    fam = sur.family_table.get(_fam(pos), {}).get(bucket, {})
    if fam:
        return list(fam)
    return list(sur.table.get("", {}).get(bucket, {}))


def _fam(pos):
    from role_features import _role_family
    return _role_family(pos) or ""


def _sensors(bucket):
    import numpy as np
    p = int(bucket[1]); ft = int(bucket[4]); oh = int(bucket[7])
    s = int(bucket[9]); c = int(bucket[11]); g = int(bucket[13])
    v = np.zeros(24, dtype=np.float64)
    v[16] = 1.0 if p else 0.0
    v[12] = 1.0 if ft else 0.0
    v[13] = 1.0 if oh else 0.0
    v[9] = 0.8 if s else 0.0
    v[15] = 1.0 if c else 0.0
    v[14] = 0.2 if g else 0.9
    return v


def _within_bucket_spread(sur, pos):
    """THE quantity the GA can exploit: how much the intents DIFFER inside
    one situation.

    The GA chooses an intent for a given state (a bucket).  If the reward
    difference between the best and the worst intent in that bucket is tiny
    next to the fitness's fixed-weight guards, the GA cannot see the
    football at all and optimises peakedness instead — which IS a collapsed
    intent mix.

    So the spread that matters is NOT the overall spread of a position's
    cells (that is dominated by how different the SITUATIONS are); it is the
    spread ACROSS INTENTS inside a bucket.  Count-weighted mean over the
    position's buckets.
    """
    num = 0.0
    den = 0.0
    for bucket, intents in sur.table.get(pos, {}).items():
        vals = [float(v) for v in intents.values()]
        wts = [float(sur.counts.get(pos, {}).get(bucket, {}).get(ik, 1))
               for ik in intents]
        if len(vals) < 2:
            continue
        tw = sum(wts)
        if tw <= 0:
            continue
        m = sum(v * w for v, w in zip(vals, wts)) / tw
        var = sum(w * (v - m) ** 2 for v, w in zip(vals, wts)) / tw
        num += var ** 0.5 * tw
        den += tw
    return (num / den) if den > 0 else 0.0


def main():
    pairs = load_raw_corpus(RAW)
    sur = FitnessSurrogate().fit(pairs)
    print(f"corpus: {RAW}  ({len(pairs)} pairs)")
    print()
    print(f"{'pos':<6}{'cells':>6}{'raw sd':>12}{'scale':>12}"
          f"{'read sd':>12}{'read n':>8}{'INTENT sd':>12}{'scaled':>12}")
    raw_sds = {}
    read_sds = {}
    intent_sds = {}
    scaled_intent_sds = {}
    for pos in sorted(p for p in sur.table if p):
        vals, wts = [], []
        for bucket, intents in sur.table[pos].items():
            for ik, v in intents.items():
                vals.append(float(v))
                wts.append(float(sur.counts.get(pos, {}).get(bucket, {}).get(ik, 1)))
        raw_sd = _wstd(vals, wts)
        scale = sur.scales.get(pos, 1.0)
        rsd, rn = _read_spread(sur, pos)
        isd = _within_bucket_spread(sur, pos)
        raw_sds[pos] = raw_sd
        read_sds[pos] = rsd
        intent_sds[pos] = isd
        scaled_intent_sds[pos] = isd / (scale or 1.0)
        print(f"{pos:<6}{len(vals):>6}{raw_sd:>12.5f}{scale:>12.5f}"
              f"{rsd:>12.5f}{rn:>8}{isd:>12.5f}{isd / (scale or 1.0):>12.5f}")

    raw_r = _ratio(raw_sds)
    read_r = _ratio(read_sds)
    intent_r = _ratio(intent_sds)
    scaled_intent_r = _ratio(scaled_intent_sds)
    print()
    print(f"RAW spread ratio (widest / flattest):      {raw_r:.1f}x")
    print(f"READ spread ratio after the scale:          {read_r:.2f}x")
    print(f"INTENT spread ratio (within a bucket):      {intent_r:.1f}x"
          f"   <- the GA's gradient")
    print(f"INTENT spread ratio after the scale:        {scaled_intent_r:.2f}x")
    print()
    print("The claim under test: the scale makes the positions comparable, so")
    print("the GA's FIXED-weight guards are a tie-breaker everywhere instead")
    print("of the objective wherever the reward was flat.")


def _ratio(d):
    vals = [v for v in d.values() if v > 0]
    if not vals:
        return 0.0
    return max(vals) / min(vals)


if __name__ == "__main__":
    sys.exit(main())
