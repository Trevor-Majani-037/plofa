"""How much of the reward ladder the corpus can actually resolve.

WHY THIS EXISTS
---------------
The reward key became hierarchical on 2026-11-08 (rich goal-geometry cell, then
the 6-bit parent).  The design exists to avoid a cell explosion, but a design
that avoids explosion by never using the rich level is just the old key with
extra code.  So the question is empirical and has to be measured, not assumed:

  * how many rich cells are populated, and how many samples they hold;
  * what share of real lookups would hit the rich level, fall to the coarse
    parent, or admit ignorance;
  * whether the SHOOT cells now separate good chances from bad ones.

Run:  python tools/diag/_diag_reward_ladder.py [corpus.json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from surrogate_collect import FitnessSurrogate, _role_family  # noqa: E402
from decision_brain import PlayerIntent  # noqa: E402


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    out = []
    for r in raw:
        if isinstance(r, dict):
            s = r.get("sensors") or r.get("state") or r.get("x")
            it = r.get("intent")
            succ = r.get("success", r.get("reward", r.get("threat", 0.0)))
            pos = r.get("position", "")
            if s is None or it is None:
                continue
            try:
                intent = PlayerIntent(it) if not isinstance(it, PlayerIntent) else it
            except ValueError:
                continue
            out.append((np.asarray(s, dtype=np.float64), intent,
                        float(succ), str(it), str(pos)))
        elif isinstance(r, (list, tuple)) and len(r) >= 5:
            s, it, succ, _act, pos = r[:5]
            try:
                intent = PlayerIntent(it)
            except ValueError:
                continue
            out.append((np.asarray(s, dtype=np.float64), intent,
                        float(succ), str(it), str(pos)))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "brains_trainer/corpus.raw.json"
    if not os.path.exists(path):
        stale = "brains_trainer/corpus_20260920_stale.json"
        print(f"{path} missing; falling back to {stale}")
        path = stale
    data = load(path)
    print(f"corpus: {path}")
    print(f"samples: {len(data)}")
    if not data:
        return

    # slot sanity: the geometry bins are meaningless if these are broken
    a = np.array([d[0][2] for d in data])
    m = np.array([d[0][3] for d in data])
    z = np.array([d[0][0] for d in data])
    print(f"\nslot2 (goal_angle_norm): min {a.min():+.3f} max {a.max():+.3f} "
          f"std {a.std():.3f} | == slot0 in {int((a == z).sum())}/{len(a)}")
    print(f"slot3 (goal_mouth_norm): min {m.min():.3f} max {m.max():.3f} "
          f"std {m.std():.3f}")
    if (a == z).mean() > 0.5:
        print("  !! slot2 == slot0 for most rows: this corpus predates the "
              "2026-10-31 sensor change. Cell COUNTS below are indicative; the "
              "geometry VALUES are garbage until the corpus is recollected.")

    sur = FitnessSurrogate().fit(data)

    # occupancy, per level
    tot = rich = coarse = 0
    rich_samples = coarse_samples = 0
    for pos, buckets in sur.counts.items():
        if pos == "":
            continue
        for k, intents in buckets.items():
            n = sum(intents.values())
            tot += 1
            if FitnessSurrogate.is_rich_key(k):
                rich += 1
                rich_samples += n
            else:
                coarse += 1
                coarse_samples += n
    print(f"\ncells populated (all rows incl. families/global): {tot}")
    print(f"  rich   cells: {rich:5d}  ({rich_samples} samples)")
    print(f"  coarse cells: {coarse:5d}  ({coarse_samples} samples)")
    if rich:
        print(f"  mean samples per rich cell: {rich_samples / rich:.2f}")

    # what a real lookup would do
    hit_rich = hit_coarse = miss = 0
    for sensors, intent, _s, _a, pos in data:
        keys = FitnessSurrogate.bucket_keys(sensors)
        ik = intent.value
        if sur._lookup_cell(keys[0], ik, pos) is not None:
            hit_rich += 1
        elif sur._lookup_cell(keys[1], ik, pos) is not None:
            hit_coarse += 1
        else:
            miss += 1
    n = len(data)
    print(f"\nlookup outcomes over {n} real decisions:")
    print(f"  rich   hit: {hit_rich:5d}  {hit_rich / n:6.1%}")
    print(f"  coarse hit: {hit_coarse:5d}  {hit_coarse / n:6.1%}")
    print(f"  prior    : {miss:5d}  {miss / n:6.1%}")

    # does the geometry now order the shots correctly, per position?
    print("\nSHOOT value by geometry (ST/CF), richest available level:")
    by_pos = defaultdict(list)
    for sensors, intent, succ, _a, pos in data:
        if intent is PlayerIntent.SHOOT and _role_family(pos) == "ST":
            by_pos[pos].append((sensors, succ))
    for pos in sorted(by_pos):
        rows = by_pos[pos]
        print(f"  {pos}: {len(rows)} shots")
        groups = defaultdict(list)
        for sensors, succ in rows:
            groups[FitnessSurrogate.goal_geometry(sensors)].append(succ)
        for k in sorted(groups):
            v = groups[k]
            print(f"    {k}: n={len(v):3d} mean={np.mean(v):+.4f}")


if __name__ == "__main__":
    main()
