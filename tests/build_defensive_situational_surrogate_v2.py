#!/usr/bin/env python
"""Build comprehensive situational surrogate for defensive actions from all data sources."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

ROLE_FAMILY_MAP = {
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "CDM": "DM", "CM": "DM", "CAM": "DM",
    "GK": "GK",
}

DEF_ACTIONS = ["tackle", "interception", "clearance", "block"]
ACTION_TO_IDX = {a: i for i, a in enumerate(DEF_ACTIONS)}

# Bucket features from 24-dim state
def bucket_from_state(state: np.ndarray) -> str:
    if len(state) < 24:
        return "Z0D0A0O0E0F0M0"

    pressure = state[16] if len(state) > 16 else 0
    zone = 2 if pressure > 0.7 else (1 if pressure > 0.3 else 0)

    final_third = state[12] if len(state) > 12 else 0
    danger = 2 if final_third > 0.7 else (1 if final_third > 0.3 else 0)

    own_half = state[13] if len(state) > 13 else 0
    attack_dir = 1 if own_half > 0.5 else 0

    space = state[9] if len(state) > 9 else 0
    opp_near = 2 if space < 0.3 else (1 if space < 0.7 else 0)

    central = state[15] if len(state) > 15 else 0
    engage = 1 if central > 0.5 else 0

    goal_close = state[14] if len(state) > 14 else 0
    foul = 1 if goal_close > 0.5 else 0

    momentum = 1 if pressure > 0.5 else 0

    return f"Z{zone}D{danger}A{attack_dir}O{opp_near}E{engage}F{foul}M{momentum}"


def family_from_position(pos: str) -> str:
    return ROLE_FAMILY_MAP.get(pos, "FB")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_situational_sequences(path: str) -> List[Dict]:
    """Extract (family, bucket, action, reward) from situational sequences."""
    data = load_json(path)
    sequences = data.get("sequences", [])
    results = []

    for seq in sequences:
        family = seq.get("family", "FB")
        if family not in ["CB", "FB", "DM", "GK"]:
            family = "FB"

        for step in seq.get("data", []):
            state = step.get("state")
            if state is None:
                continue
            state = np.array(state, dtype=np.float64)

            action = step.get("action", "tackle")
            if action not in DEF_ACTIONS:
                continue

            # Use a simple reward heuristic based on action and context
            # In reality we'd use episode_score from outcome
            reward = 1.0

            bucket = bucket_from_state(state)
            results.append({
                "family": family,
                "bucket": bucket,
                "action": action,
                "reward": reward,
            })

    return results


def process_legacy_samples(path: str) -> List[Dict]:
    """Extract (family, bucket, action, reward) from legacy samples."""
    data = load_json(path)
    results = []

    for sample in data:
        sensors = sample.get("sensors", [])
        if len(sensors) < 24:
            continue
        state = np.array(sensors, dtype=np.float64)

        action = sample.get("action", "tackle")
        if action not in DEF_ACTIONS:
            continue

        # Use episode_score as reward
        reward = sample.get("episode_score", 1.0)

        # Infer family from position (not in sample, so use site as proxy)
        site = sample.get("site", "contest")
        # Default to FB for now
        family = "FB"

        bucket = bucket_from_state(state)
        results.append({
            "family": family,
            "bucket": bucket,
            "action": action,
            "reward": reward,
        })

    return results


def build_surrogate(
    all_results: List[Dict],
    min_count: int = 1,
) -> Dict[str, Any]:
    """Build surrogate table from all results."""
    rewards: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for r in all_results:
        family = r["family"]
        bucket = r["bucket"]
        action = r["action"]
        reward = r["reward"]

        rewards[family][bucket][action].append(reward)
        counts[family][bucket][action] += 1

    # Global fallback
    global_rewards: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    global_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for family in rewards:
        for bucket in rewards[family]:
            for action, vals in rewards[family][bucket].items():
                global_rewards[bucket][action].extend(vals)
                global_counts[bucket][action] += len(vals)

    # Compute averages
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    count_table: Dict[str, Dict[str, Dict[str, int]]] = {}

    for family in ["CB", "FB", "DM", "GK"]:
        if family not in rewards:
            continue
        table[family] = {}
        count_table[family] = {}
        for bucket in rewards[family]:
            table[family][bucket] = {}
            count_table[family][bucket] = {}
            for action in DEF_ACTIONS:
                vals = rewards[family][bucket].get(action, [])
                if len(vals) >= min_count:
                    table[family][bucket][action] = float(np.mean(vals))
                    count_table[family][bucket][action] = len(vals)

    # Global fallback
    table[""] = {}
    count_table[""] = {}
    for bucket in global_rewards:
        table[""][bucket] = {}
        count_table[""][bucket] = {}
        for action in DEF_ACTIONS:
            vals = global_rewards[bucket].get(action, [])
            if len(vals) >= min_count:
                table[""][bucket][action] = float(np.mean(vals))
                count_table[""][bucket][action] = len(vals)

    return {
        "prior": 0.5,
        "table": table,
        "counts": count_table,
    }


def main():
    print("Loading data from all sources...")

    all_results = []

    # 1. Situational sequences (full 47-dim features)
    print("Processing situational sequences...")
    seq_results = process_situational_sequences("brains_def/situational_sequences.json")
    all_results.extend(seq_results)
    print(f"  Added {len(seq_results)} entries")

    # 2. Legacy samples (forced interventions)
    print("Processing forced samples...")
    forced_results = process_legacy_samples("brains_def/samples_forced.json")
    all_results.extend(forced_results)
    print(f"  Added {len(forced_results)} entries")

    # 3. Legacy samples (heuristic)
    print("Processing heuristic samples...")
    heuristic_results = process_legacy_samples("brains_def/samples_heuristic.json")
    all_results.extend(heuristic_results)
    print(f"  Added {len(heuristic_results)} entries")

    # 4. Legacy samples (natural)
    print("Processing natural samples...")
    natural_results = process_legacy_samples("brains_def/samples.json")
    all_results.extend(natural_results)
    print(f"  Added {len(natural_results)} entries")

    print(f"Total entries: {len(all_results)}")

    print("Building situational surrogate...")
    surrogate = build_surrogate(all_results, min_count=1)

    out_path = "brains_def/surrogate_situational.json"
    print(f"Saving to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(surrogate, f, indent=2)

    # Summary
    for family in ["CB", "FB", "DM", "GK"]:
        if family in surrogate["table"]:
            buckets = len(surrogate["table"][family])
            actions = sum(len(v) for v in surrogate["table"][family].values())
            print(f"  {family}: {buckets} buckets, {actions} action entries")

    if "" in surrogate["table"]:
        buckets = len(surrogate["table"][""])
        actions = sum(len(v) for v in surrogate["table"][""].values())
        print(f"  global: {buckets} buckets, {actions} action entries")

    print("Done!")


if __name__ == "__main__":
    main()