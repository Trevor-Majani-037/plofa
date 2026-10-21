"""IS THE TABLE'S CONTEXT-DEPENDENCE REAL, OR NOISE?  (held out)

THE QUESTION
------------
The census failed twice with the same shape: every position collapsed onto
one or two intents.  The scale fixed the BALANCE (the reward's swing is now
5x-16x the largest guard instead of 5x-20x SMALLER), and the 80-generation
smoke still came back 100% CARRY.  So either

  (a) the GA cannot find the state-dependent solution (a SEARCH problem), or
  (b) the table's within-bucket differentiation is mostly NOISE, so the
      state-dependent solution is not there to be found (an OBJECTIVE
      problem — "the objective is the ceiling").

The test that separates them is a HELD-OUT one.  Fit the table on half the
corpus and measure the headroom (the per-state oracle minus the best single
constant) on the OTHER half:

  * headroom on the FITTING half is inflated by the fit itself — the oracle
    takes the maximum over noisy cell values, so noise ADDS to it;
  * headroom on the HELD-OUT half is only real if the table's structure
    generalises.

And the shrinkage is the prime suspect for the noise: `fit` shrinks every
cell with fewer than `min_samples` samples toward its bucket mean, so a
cell with ONE sample is 75% bucket mean.  Measured: how many cells are
actually shrunk, and what the held-out headroom looks like with the
shrinkage on and off.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_headroom_cv.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from role_features import ROLE_INTENTS, _role_family  # noqa: E402
from surrogate_collect import FitnessSurrogate, load_raw_corpus  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "brains", "surrogate_pos_v5.raw.json")


def _headroom(sur, sensors_list, pos):
    """Oracle minus best constant over a list of sensor vectors."""
    fam = _role_family(pos) or ""
    allowed = [ik for ik in ROLE_INTENTS.get(fam, ())]
    sums = {ik: 0.0 for ik in allowed}
    oracle = 0.0
    n = 0
    for sensors in sensors_list:
        key = sur.bucket(sensors)
        vals = []
        for ik in allowed:
            v = sur._lookup(key, ik, pos)
            if v is None:
                continue
            vals.append((v, ik))
        if not vals:
            continue
        sums[max(vals)[1]] += max(vals)[0]
        oracle += max(vals)[0]
        n += 1
    if n == 0:
        return 0.0, 0.0, {}
    means = {ik: sums[ik] / n for ik in sums}
    best_const = max(means.values())
    return oracle / n - best_const, best_const, means


def main():
    pairs = load_raw_corpus(RAW)
    half_a = pairs[0::2]
    half_b = pairs[1::2]
    print(f"corpus: {RAW}  ({len(pairs)} pairs)")
    print(f"fit half: {len(half_a)} pairs | held out: {len(half_b)} pairs")
    print()

    for min_samples in (3, 1):
        sur_a = FitnessSurrogate().fit(half_a, min_samples=min_samples)
        sur_all = FitnessSurrogate().fit(pairs, min_samples=min_samples)
        shrunk = 0
        total = 0
        for pos, buckets in sur_all.table.items():
            for bucket, intents in buckets.items():
                for ik in intents:
                    total += 1
                    if sur_all.counts.get(pos, {}).get(bucket, {}).get(ik, 0) < min_samples:
                        shrunk += 1
        print(f"=== min_samples={min_samples} ===")
        print(f"  cells shrunk toward the bucket mean: {shrunk}/{total} "
              f"({100.0 * shrunk / max(1, total):.1f}%)")
        print(f"  {'pos':<6}{'fit-half':>12}{'HELD OUT':>12}{'best const':>12}")
        for pos in sorted(p for p in sur_a.table if p):
            sa = [p[0] for p in half_a if p[4] == pos]
            sb = [p[0] for p in half_b if p[4] == pos]
            if len(sb) < 10:
                continue
            h_in, bc, _ = _headroom(sur_a, sa, pos)
            h_out, bc_out, _ = _headroom(sur_a, sb, pos)
            print(f"  {pos:<6}{h_in:>12.5f}{h_out:>12.5f}{bc_out:>12.5f}")
        print()
        print("  (the headroom is in RAW reward units; divide by the "
              "position's scale for the fitness's units)")
        print()


if __name__ == "__main__":
    sys.exit(main())
