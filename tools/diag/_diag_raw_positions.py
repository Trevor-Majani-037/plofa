"""List the distinct position labels in a raw surrogate corpus.

The raw corpus is ``{"pairs": [...]}`` where each pair is a dict or list
produced by ``surrogate_collect``.  ``scales`` in ``FitnessSurrogate`` is
keyed by the EXACT position string passed to ``expected_success``, so the
question this answers is: which labels does the corpus actually carry, and
do they match the labels the evolver trains on (``ST``, ``CM``, ...) or the
live roster labels (``CM1``, ``CB2``, ...)?
"""
from __future__ import annotations

import json
import sys
from collections import Counter

RAW = sys.argv[1] if len(sys.argv) > 1 else r"brains/surrogate_pos_v5.raw.json"

with open(RAW, "r", encoding="utf-8") as fh:
    doc = json.load(fh)

pairs = doc.get("pairs", doc if isinstance(doc, list) else [])
print(f"corpus: {RAW}")
print(f"pairs : {len(pairs)}")
print(f"top-level keys: {sorted(doc.keys()) if isinstance(doc, dict) else 'list'}")
print(f"first pair type: {type(pairs[0]).__name__}")
print(f"first pair: {pairs[0]!r}"[:600])
print()

pos = Counter()
intent = Counter()
rewards_by_pos = {}

for p in pairs:
    if isinstance(p, dict):
        position = p.get("position", "?")
        act = p.get("intent", p.get("action", "?"))
        rew = p.get("reward", 0.0)
    else:
        # list form: locate the position by scanning for a known role string
        position = "?"
        act = "?"
        rew = 0.0
        for item in p:
            if isinstance(item, str) and item in {
                "GK", "CB", "CB1", "CB2", "LB", "RB", "CDM", "CM", "CM1",
                "CM2", "CAM", "LW", "RW", "ST", "CF",
            }:
                position = item
            elif isinstance(item, str):
                act = item
            elif isinstance(item, (int, float)):
                rew = float(item)
    pos[position] += 1
    intent[act] += 1
    rewards_by_pos.setdefault(position, []).append(rew)

print("POSITION LABELS IN CORPUS")
for k, n in pos.most_common():
    vals = rewards_by_pos[k]
    mean = sum(vals) / len(vals) if vals else 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
    print(f"  {k:<6} n={n:<5} mean_reward={mean:+.6f}  sd={var ** 0.5:.6f}  raw_spread={max(vals) - min(vals):.6f}")

print()
print("INTENTS IN CORPUS")
for k, n in intent.most_common():
    print(f"  {k:<24} {n}")
