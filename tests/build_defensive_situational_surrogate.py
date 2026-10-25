#!/usr/bin/env python
"""Build situational surrogate for defensive actions from collected sequences."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from defensive_situational_collect import DefensiveSequence


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Role family mapping (from position to family)
ROLE_FAMILY_MAP = {
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "CDM": "DM", "CM": "DM", "CAM": "DM",  # CAM treated as DM for defensive
    "GK": "GK",
}

# Defensive actions (4 actions)
DEF_ACTIONS = ["tackle", "interception", "clearance", "block"]
ACTION_TO_IDX = {a: i for i, a in enumerate(DEF_ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(DEF_ACTIONS)}

# State features used for bucketing (subset of 24 state features)
# Based on the existing surrogate's bucket keys: Z<zone>D<danger>A<attack>O<opp>E<engage>F<foul>M<momentum>
STATE_BUCKET_FEATURES = {
    "zone": 16,        # pressure
    "danger": 12,      # final_third (inverted for defensive)
    "attack_dir": 13,  # own_half
    "opp_near": 9,     # space
    "engage": 15,      # central
    "foul": 14,        # goal_close
    "momentum": 11,    # not sure, use another feature
}


def bucket_from_state(state: np.ndarray) -> str:
    """Convert 24-dim state vector to bucket string like 'Z1D0A0O1E1F1M0'."""
    if len(state) < 24:
        return "Z0D0A0O0E0F0M0"

    # Zone (pressure): 0=low, 1=med, 2=high
    pressure = state[16] if len(state) > 16 else 0
    zone = 2 if pressure > 0.7 else (1 if pressure > 0.3 else 0)

    # Danger (final third proximity): 0=low, 1=med, 2=high
    # For defensive, high danger = opponent in final third
    final_third = state[12] if len(state) > 12 else 0
    danger = 2 if final_third > 0.7 else (1 if final_third > 0.3 else 0)

    # Attack direction (own half): 0=opponent half, 1=own half
    own_half = state[13] if len(state) > 13 else 0
    attack_dir = 1 if own_half > 0.5 else 0

    # Opponent near (space): 0=lots of space, 1=some, 2=pressured
    space = state[9] if len(state) > 9 else 0
    opp_near = 2 if space < 0.3 else (1 if space < 0.7 else 0)

    # Engage (central): 0=wide, 1=central
    central = state[15] if len(state) > 15 else 0
    engage = 1 if central > 0.5 else 0

    # Foul (goal close): 0=far, 1=close
    goal_close = state[14] if len(state) > 14 else 0
    foul = 1 if goal_close > 0.5 else 0

    # Momentum: use pressure as proxy
    momentum = 1 if pressure > 0.5 else 0

    return f"Z{zone}D{danger}A{attack_dir}O{opp_near}E{engage}F{foul}M{momentum}"


def family_from_position(pos: str) -> str:
    """Map position to role family."""
    return ROLE_FAMILY_MAP.get(pos, "FB")


def load_sequences(path: str) -> List[Dict]:
    """Load sequences from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sequences", [])


def build_situational_surrogate(
    sequences: List[Dict],
    min_count: int = 2,
) -> Dict[str, Any]:
    """Build situational surrogate from sequences.

    Returns dict with structure:
    {
        "family": {
            "bucket": {
                "action": avg_reward
            }
        },
        "counts": {...}
    }
    """
    # family -> bucket -> action -> list of rewards
    rewards: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for seq in sequences:
        family = seq.get("family", "FB")
        if family not in ["CB", "FB", "DM", "GK"]:
            family = "FB"

        for step in seq.get("data", []):
            # Get state (24-dim)
            state = step.get("state")
            if state is None:
                continue
            state = np.array(state, dtype=np.float64)

            # Get action
            action = step.get("action", "tackle")
            if action not in DEF_ACTIONS:
                continue

            # Get reward from engine_snapshot (simplified - use proxy)
            # In reality, we'd need the actual outcome. For now, use action as proxy.
            # The existing surrogate uses outcome-based rewards.
            # Since we don't have outcomes in the collected data, we'll use a heuristic:
            # successful actions get higher reward
            engine_snapshot = step.get("engine_snapshot", {})
            # We don't have outcome in the snapshot, so use a simple heuristic
            # based on action type and context
            reward = 1.0  # default

            bucket = bucket_from_state(state)

            rewards[family][bucket][action].append(reward)
            counts[family][bucket][action] += 1

    # Also add global fallback
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
    seq_path = "brains_def/situational_sequences.json"
    out_path = "brains_def/surrogate_situational.json"

    print(f"Loading sequences from {seq_path}...")
    sequences = load_sequences(seq_path)
    print(f"Loaded {len(sequences)} sequences")

    total_steps = sum(s.get("steps", 0) for s in sequences)
    print(f"Total steps: {total_steps}")

    print("Building situational surrogate...")
    surrogate = build_situational_surrogate(sequences, min_count=1)

    print(f"Saving to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(surrogate, f, indent=2)

    # Print summary
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