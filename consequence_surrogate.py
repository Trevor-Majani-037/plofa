"""Chain-return surrogate (v2): episode-payoff objective for brain evolution.

Instead of v1's decision-local label ("did this touch connect?"), each
cell holds the expected ATTACK-PAYOFF (3*goals + xg + 0.2*shots) of the
possession that followed every decision in that (position, bucket, intent)
cell — chain returns, not immediate-event weights.

Sparse-cell borrowing: same-intent cross-bucket mean (preserves intent
identity, pools rare-intent data).  Per-position linear rescaling to
[0.4, 1.5] so the evolution fitness scale stays comparable to v1.

Usage:
    python consequence_surrogate.py --dataset value_experiments/consequence_dataset.json
    python consequence_surrogate.py   # uses default dataset path
"""
from __future__ import annotations
import argparse, collections, json, os, sys
import numpy as np
from surrogate_collect import FitnessSurrogate

BUCKET = FitnessSurrogate.bucket  # static method: sensors[6] -> key
POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST", "CF"]

# per-position rescale targets (match v1 approximate reward range)
_RESCALE_LO, _RESCALE_HI = 0.4, 1.5
_PRIOR = 0.4


def _attack_payoff(record: dict) -> float:
    return 3.0 * record["goals"] + record["xg"] + 0.0 * record["shots"]


def _load_dataset(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("records", data)


def _build_raw_table(records: list, min_samples: int = 3):
    """Accumulate per-(pos, bucket, intent) mean attack payoff.

    Returns raw_table[pos][bucket][intent] = (sum, count), and
    intent_agg[pos][intent] = (sum, count) for cross-bucket borrowing.
    """
    raw_table: dict = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0])))
    intent_agg: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0]))
    for r in records:
        pos = r["position"]
        if pos not in POSITIONS:
            continue
        sensors = np.asarray(r["sensors"], dtype=np.float64)
        if len(sensors) < 20:
            continue
        ik = r["intent"]
        bucket = BUCKET(sensors)
        score = _attack_payoff(r)
        raw_table[pos][bucket][ik][0] += score
        raw_table[pos][bucket][ik][1] += 1
        intent_agg[pos][ik][0] += score
        intent_agg[pos][ik][1] += 1
        # also build global "" row
        raw_table[""][bucket][ik][0] += score
        raw_table[""][bucket][ik][1] += 1
        intent_agg[""][ik][0] += score
        intent_agg[""][ik][1] += 1

    # compute means and apply cross-intent borrowing
    table: dict = {}
    counts: dict = {}
    for pos in set(list(raw_table.keys()) + [""]):
        table[pos] = {}
        counts[pos] = {}
        for bucket, intents in raw_table[pos].items():
            table[pos][bucket] = {}
            counts[pos][bucket] = {}
            # bucket overall mean (for cross-intent borrowing)
            tot_s = sum(v[0] for v in intents.values())
            tot_c = sum(v[1] for v in intents.values())
            bucket_mean = tot_s / tot_c if tot_c else 0.0
            for ik, (s, c) in intents.items():
                if c >= min_samples:
                    table[pos][bucket][ik] = s / c
                else:
                    # cross-bucket same-intent borrowing
                    agg = intent_agg[pos].get(ik, [0.0, 0])
                    if agg[1] >= min_samples:
                        table[pos][bucket][ik] = agg[0] / agg[1]
                    else:
                        table[pos][bucket][ik] = bucket_mean if tot_c else _PRIOR
                counts[pos][bucket][ik] = c
    return table, counts


def _rescale_position(table: dict, pos: str) -> float:
    """Linearly rescale a position's cells to [_RESCALE_LO, _RESCALE_HI].
    Returns max_raw (for diagnostics).
    """
    cells = []
    for bucket in table.get(pos, {}):
        for ik in table[pos][bucket]:
            cells.append(table[pos][bucket][ik])
    if not cells:
        return 0.0
    raw_max = max(cells)
    raw_min = min(cells)
    span = raw_max - raw_min
    if span < 1e-9 or raw_max <= 0:
        # flat or zero; set everything to prior
        for bucket in table.get(pos, {}):
            for ik in table[pos][bucket]:
                table[pos][bucket][ik] = _PRIOR
        return 0.0
    for bucket in table.get(pos, {}):
        for ik in table[pos][bucket]:
            v = table[pos][bucket][ik]
            table[pos][bucket][ik] = round(
                _RESCALE_LO + (v - raw_min) / span * (_RESCALE_HI - _RESCALE_LO), 4)
    return raw_max


def build_surrogate_v2(records: list, out_path: str, report_path: str):
    raw_table, counts = _build_raw_table(records)
    # rescale per position
    rescale_info = {}
    for pos in POSITIONS + [""]:
        max_raw = _rescale_position(raw_table, pos)
        rescale_info[pos] = round(max_raw, 4)
    surrogate = {"prior": _PRIOR, "table": raw_table, "counts": counts}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(surrogate, f, indent=2)
    print(f"Wrote surrogate v2 -> {out_path}")
    # build report
    report = {"rescale_max_raw": rescale_info, "positions": {}, "prior": _PRIOR}
    # scoring buckets (final_third=1, goal_close=1)
    SCORING_BUCKETS = [b for b in sum((list(v.keys()) for v in raw_table.get("ST", {}).values()), [])
                       if "FT1" in b and "G1" in b]
    for pos in POSITIONS:
        buckets = raw_table.get(pos, {})
        intents_all = collections.Counter()
        for b, iv in buckets.items():
            for ik in iv:
                intents_all[ik] += 1
        report["positions"][pos] = {
            "n_buckets": len(buckets),
            "intents": dict(intents_all),
        }
        # show top intents in scoring buckets
        scoring = {}
        for b, iv in buckets.items():
            if "FT1" in b and "G1" in b:
                for ik, v in iv.items():
                    scoring.setdefault(ik, []).append(v)
        if scoring:
            means = {ik: round(np.mean(vs), 4) for ik, vs in scoring.items()}
            report["positions"][pos]["scoring_bucket_means"] = dict(
                sorted(means.items(), key=lambda x: -x[1]))
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report -> {report_path}")
    # print key comparison: ST/CM/CAM scoring bucket SHOOT/THROUGH_BALL/etc.
    print("\n=== Chain-return surrogate v2 key cells (scoring buckets FT1G1) ===")
    for pos in ["ST", "CM", "CAM", "LW", "RW"]:
        scoring = {}
        for b, iv in raw_table.get(pos, {}).items():
            if "FT1" in b and "G1" in b:
                for ik, v in iv.items():
                    scoring.setdefault(ik, []).append(v)
        if scoring:
            means = {ik: round(np.mean(vs), 4) for ik, vs in scoring.items()}
            top = sorted(means.items(), key=lambda x: -x[1])[:5]
            print(f"  {pos:4s}: ", "  ".join(f"{ik}={v:.3f}" for ik, v in top))
        else:
            print(f"  {pos:4s}: no scoring bucket samples")
    return surrogate


def main():
    p = argparse.ArgumentParser(description="Chain-return (v2) surrogate builder.")
    p.add_argument("--dataset", default="value_experiments/consequence_dataset.json",
                   help="consequence_probe output (list or {records: ...}).")
    p.add_argument("--out", default="brains_trainer/surrogate_v2.json")
    p.add_argument("--report", default="value_experiments/v2_surrogate_report.json")
    args = p.parse_args()
    records = _load_dataset(args.dataset)
    print(f"Loaded {len(records)} decision records from {args.dataset}")
    build_surrogate_v2(records, args.out, args.report)


if __name__ == "__main__":
    main()
