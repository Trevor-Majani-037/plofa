"""Inspect the reward distribution in a fitted surrogate.

Question (Next Move #2): the threat reward is small for ordinary passes
(+0.008 for a 35 m forward ball vs +0.35 for a shot on target).  If the
fitted table has almost no variance the GA has no gradient and the retrain is
wasted.  Read the distribution before evolving.

⚠️  THE FIRST VERSION OF THIS PROBE WAS WRONG, AND THE ERROR IS THE PROJECT'S
STANDING ONE.  It averaged the CELL MEANS per intent, unweighted:

    per_intent[intent].append(score)      # ← one entry per CELL

so a cell holding one sample counted exactly as much as a cell holding sixty,
and the "n" it printed was a count of CELLS, not of samples.  On the 6-match
v4 surrogate that made SHOOT read −0.0074 (n=14) when the sample-weighted
figure is a different story entirely — a 14-cell mean dominated by n=1 cells
is a statement about the table's sparsity, not about shooting.

A fitted surrogate stores {position: {bucket: {intent: mean_reward}}} plus a
PARALLEL {position: {bucket: {intent: count}}} table.  The sample-weighted
mean per intent is therefore exactly recoverable:

    mean_i = Σ_buckets count[bucket][intent] * value[bucket][intent]
             / Σ_buckets count[bucket][intent]

This probe reports BOTH numbers and labels each one, because they answer
different questions:
  * CELL mean  — how spread the table is (what the GA samples from)
  * SAMPLE mean — what the reward actually paid that intent (the football)

It also reports the per-intent sample count, so an intent resting on a
handful of samples is visible as such rather than hiding behind a mean.

Usage:  python _diag_reward_dist.py brains/surrogate_pos_v4.json [more.json ...]
"""
import json
import sys
import statistics as st


def _numeric_table(data):
    """Return the {pos: {bucket: {intent: value}}} table, whatever it is called."""
    table = data.get("table", data.get("positions_data", data))
    if isinstance(table, dict) and "positions" in table and isinstance(table["positions"], dict):
        table = table["positions"]
    return table if isinstance(table, dict) else {}


def _walk(table):
    """Yield (position, bucket, intent, value) for every numeric cell."""
    for pos, buckets in table.items():
        if not isinstance(buckets, dict):
            continue
        for bucket, intents in buckets.items():
            if not isinstance(intents, dict):
                continue
            for intent, value in intents.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    yield pos, bucket, intent, float(value)


def main(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        table = _numeric_table(data)
        counts_raw = data.get("counts")
        counts = counts_raw if isinstance(counts_raw, dict) else {}

        cell_values = []
        per_intent_cells = {}
        per_intent_samples = {}          # intent -> Σ count*value
        per_intent_n = {}                # intent -> Σ count
        global_samples = {}              # intent -> Σ count*value (global row)
        global_n = {}                    # intent -> Σ count (global row)

        for pos, bucket, intent, value in _walk(table):
            cell_values.append(value)
            per_intent_cells.setdefault(intent, []).append(value)

            n = 0
            if counts:
                bucket_counts = counts.get(pos, {})
                if isinstance(bucket_counts, dict):
                    intent_counts = bucket_counts.get(bucket, {})
                    if isinstance(intent_counts, dict):
                        raw = intent_counts.get(intent, 0)
                        n = raw if isinstance(raw, (int, float)) else 0
            if n:
                # `fit` writes every sample into BOTH its position table and
                # the global "" table (surrogate_collect.py:697-699), so the
                # global row double-counts.  Keep the two apart: the
                # position tables are what the GA reads first, the global
                # row is its fallback.
                if pos == "":
                    global_samples[intent] = global_samples.get(intent, 0.0) + n * value
                    global_n[intent] = global_n.get(intent, 0) + int(n)
                else:
                    per_intent_samples[intent] = per_intent_samples.get(intent, 0.0) + n * value
                    per_intent_n[intent] = per_intent_n.get(intent, 0) + int(n)

        print(f"\n=== {path} ===")
        if not cell_values:
            print("  no numeric scores found — check key layout")
            continue

        qs = st.quantiles(cell_values, n=10) if len(cell_values) >= 10 else []
        print(f"  CELLS: {len(cell_values)}")
        print(f"  cell mean {st.mean(cell_values):+.4f}  median {st.median(cell_values):+.4f} "
              f"sd {st.pstdev(cell_values):.4f}")
        print(f"  cell min {min(cell_values):+.4f}  cell max {max(cell_values):+.4f}")
        if qs:
            print(f"  cell p10 {qs[0]:+.4f}  cell p90 {qs[8]:+.4f}")
        nz = sum(1 for s in cell_values if abs(s) > 1e-9)
        print(f"  nonzero cells: {nz}/{len(cell_values)} ({100.0*nz/len(cell_values):.1f}%)")

        if not counts:
            print("  (no counts table — sample-weighted means unavailable)")
            continue

        total_n = sum(per_intent_n.values())
        print(f"  SAMPLES (position tables): {total_n} across {len(per_intent_n)} intents")
        print("  intent              pos_mean  samples   global_mean  g_n   cell_mean  cells")
        ranked = sorted(per_intent_n, key=lambda i: -per_intent_samples[i] / max(per_intent_n[i], 1))
        for intent in ranked:
            n = per_intent_n[intent]
            if n <= 0:
                continue
            smean = per_intent_samples[intent] / n
            gn = global_n.get(intent, 0)
            gmean = global_samples.get(intent, 0.0) / gn if gn else 0.0
            cells = per_intent_cells.get(intent, [])
            cmean = st.mean(cells) if cells else 0.0
            print(f"  {intent:<20} {smean:+.4f}  {n:>7}   {gmean:+.4f}  {gn:>4}   "
                  f"{cmean:+.4f}  {len(cells):>5}")

        # ── WHAT THE GA ACTUALLY READS ──────────────────────────────────
        # `FitnessSurrogate._lookup` is PER BUCKET: the position's cell for
        # that bucket, else the global row's cell for the SAME bucket, else
        # the neutral prior.  It never sees an aggregate over buckets — so
        # neither should the gate.  The collapse question ("in how many
        # situations does this intent look bad?") is a SIGN CENSUS over the
        # buckets, not an average.
        print()
        print("  PER-BUCKET SIGN CENSUS (what the GA reads):")
        print("  intent              buckets  negative  positive   worst      best")
        for intent in ranked:
            vals = []
            for pos, bucket, name, value in _walk(table):
                if name != intent:
                    continue
                if pos != "":
                    continue          # the global row is the fallback
                vals.append(value)
            if not vals:
                continue
            neg = sum(1 for v in vals if v < -1e-9)
            posi = sum(1 for v in vals if v > 1e-9)
            print(f"  {intent:<20} {len(vals):>7}  {neg:>8}  {posi:>8}   "
                  f"{min(vals):+.4f}  {max(vals):+.4f}")
        print("  (the GLOBAL row per bucket — a negative value here is what "
              "the GA\n   falls back to when a position has no cell of its own "
              "for that intent)")

        # How sparse is the table the GA will sample from?
        singles = sum(1 for i in per_intent_n if per_intent_n[i] <= 2)
        print(f"  intents resting on <=2 samples: {singles}/{len(per_intent_n)}")
        cell_ns = [per_intent_n[i] for i in per_intent_n]
        if cell_ns:
            print(f"  samples per intent: min {min(cell_ns)} median {st.median(cell_ns):.0f} max {max(cell_ns)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["brains/surrogate_pos_v4.json"])
