#!/usr/bin/env python
"""Wire the FB situational brain into _def_action_choice for real matches."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))
from distill_defensive_brains import SituationalDefensiveBrain

# Load FB brain
fb_brain = SituationalDefensiveBrain.load("brains_def/situational/FB.npz")

# Monkey-patch match_engine._def_action_choice
import match_engine

_original_def_action = match_engine._def_action_choice

# Cache for neural states per defender
_defensive_brain_cache = {}

def _build_situational_input(engine, defending_team, attacking_team,
                            danger_level, ctx_x, ctx_y, ball_aerial,
                            opponent_distance, attacks_right, ball_x, ball_y,
                            press_occurred, site):
    """Build 51-dim input for situational brain."""
    # Get defender info - we need to find who's making this decision
    # The engine's contest logic picks a defender; we approximate from context
    
    # State (24-d) - extract from engine state
    state_vec = np.zeros(24, dtype=np.float64)
    # pressure
    state_vec[16] = min(danger_level * 2.0, 1.0)
    # final_third proximity (for defensive: opponent in our final third)
    state_vec[12] = max(0.0, 1.0 - abs(ctx_x - 105.0) / 50.0) if attacks_right else max(0.0, 1.0 - abs(ctx_x) / 50.0)
    # own_half
    state_vec[13] = 1.0 if ((attacks_right and ctx_x < 52.5) or (not attacks_right and ctx_x > 52.5)) else 0.0
    # space
    state_vec[9] = min(1.0, opponent_distance / 20.0)
    # central
    state_vec[15] = 1.0 - abs(ctx_y - 34.0) / 34.0
    # goal_close
    state_vec[14] = 1.0 if ((attacks_right and ctx_x > 85) or (not attacks_right and ctx_x < 20)) else 0.0
    # other sensors from engine state...
    
    # Role block (13-d) - FB family
    role_block = np.zeros(13, dtype=np.float64)
    # FB vocabulary: RECYCLE, PROGRESSIVE_PASS, SAFE_PASS, PROTECT_POSSESSION, SWITCH, CROSS, DRIBBLE, CARRY, THROUGH_BALL
    role_block[1] = 1.0  # PROGRESSIVE_PASS
    role_block[2] = 1.0  # SAFE_PASS
    role_block[3] = 1.0  # PROTECT_POSSESSION
    role_block[4] = 1.0  # SWITCH
    role_block[5] = 1.0  # CROSS
    role_block[6] = 1.0  # DRIBBLE
    role_block[7] = 1.0  # CARRY
    role_block[8] = 1.0  # THROUGH_BALL
    
    # Memory block (8-d) - from NeuralStateStore (simplified)
    memory_block = np.zeros(8, dtype=np.float64)
    
    # Plan block (6-d) - team plan summary
    plan_block = np.zeros(6, dtype=np.float64)
    
    # Entity tokens (12-d) - simplified
    entity_tokens = np.zeros(12, dtype=np.float64)
    entity_mask = np.zeros(12, dtype=np.float64)
    
    # Concatenate: 24 + 13 + 8 + 6 = 51
    full_input = np.concatenate([state_vec, role_block, memory_block, plan_block])
    assert len(full_input) == 51
    
    return full_input, entity_tokens, entity_mask

def _brain_def_action(engine, defending_team, attacking_team,
                     danger_level, ctx_x, ctx_y, ball_aerial,
                     opponent_distance, attacks_right, ball_x, ball_y,
                     press_occurred=False, fallback=None, site="contest"):
    """Situational brain defensive action."""
    global _defensive_brain_cache
    
    # Build input
    try:
        full_input, tokens, token_mask = _build_situational_input(
            engine, defending_team, attacking_team,
            danger_level, ctx_x, ctx_y, ball_aerial,
            opponent_distance, attacks_right, ball_x, ball_y,
            press_occurred, site
        )
        
        # Get brain prediction
        action_idx = fb_brain.predict(full_input)
        action = ["tackle", "interception", "clearance", "block"][action_idx]
        
        print(f"[BRAIN] {defending_team} {site}: danger={danger_level:.2f} dist={opponent_distance:.1f} -> {action}")
        
        return action
    except Exception as e:
        print(f"[BRAIN ERROR] {e}, falling back")
        return _original_def_action(engine, defending_team, attacking_team,
                                   danger_level, ctx_x, ctx_y, ball_aerial,
                                   opponent_distance, attacks_right, ball_x, ball_y,
                                   press_occurred, fallback, site)

# Install patch
match_engine._def_action_choice = _brain_def_action
print("[WIRED] FB situational brain active in _def_action_choice")