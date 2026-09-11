# Match Simulation Refinements Bugfix Design

## Overview

This design addresses six interconnected match flow defects in the PLOFA 26/27 match simulation engine. The bugs span shot selection logic, restart mechanics, offside detection, and post-goal flow. Each defect breaks football realism—players shoot from impossible angles, balls go out without restarts, offside attacks count as valid goals, and goals lead directly to the next sequence with no celebration or proper center restart.

The fix introduces **intelligent geometry-aware shot selection**, **causal out-of-bounds detection with restarts**, **second-to-last-defender offside tracking**, **celebration sequences with time addition**, and **full formation resets** on center restarts. All changes preserve the existing causal event chain architecture and spatial continuity (state.last_ball_x/y) established in Checkpoints 5-7.

**Affected Modules:**
- `event_chain.py` — AttackChain shot selection logic, restart detection
- `match_engine.py` — MatchState offside tracking, celebration pause, kickoff reset
- `position_engine.py` — Formation reset method

**Testing Approach:** Exploratory PBT to surface counterexamples from unfixed code, then fix+preservation checking.


## Glossary

- **Bug_Condition (C)**: The set of inputs that trigger the six interconnected defects—unrealistic shots from behind the goal line, balls going out without restart events, offside attacks scoring, goals skipping celebration, kickoffs resetting ball but not players
- **Property (P)**: The expected correct behavior—geometry-aware shot/cross decisions, throw-in/goal-kick events on out-of-bounds, OFFSIDE events stopping attacks, celebration sequences adding time, full formation resets on kickoffs
- **Preservation**: All existing match flow (corners, saves, blocks, cards, substitutions) continues unchanged
- **Angle Geometry**: The mathematical model of goal visibility from a shooter's (x, y) position—posts at (105, 30.34) and (105, 37.66) define a visible opening that shrinks with acute angles
- **Offside Line**: The x-coordinate of the second-to-last defender (2nd closest to their own goal line) at the moment a forward pass is played
- **Celebration Duration**: Random pause between 10-30 seconds added to match clock after a goal, during which no simulation events occur
- **Center Restart**: The kickoff sequence after a goal—conceding team receives possession at (52.5, 34), all players reset to home_x/home_y, ball resets to center circle


## Bug Details

### Bug Condition 1: Unrealistic Shot Selection

**Current Defect:**  
The system allows shooting from any coordinates, including positions behind the goal line (x ≥ 105) and acute angles where the goal opening is barely visible (< 10° effective angle). A player at (105, 55)—2 meters behind the goal line, 21 meters off-center—can attempt and score a direct shot.

**Formal Specification:**
```
FUNCTION isBugCondition_UnrealisticShot(shot_x, shot_y)
  INPUT: shot_x (float), shot_y (float) — proposed shot coordinates
  OUTPUT: boolean
  
  behind_goal_line := shot_x >= 105.0
  distance_from_line := 105.0 - shot_x
  y_offset_from_center := ABS(shot_y - 34.0)
  
  // Acute angle: very close to goal line + wide y-offset
  acute_angle := (distance_from_line < 5.0) AND (y_offset_from_center > 15.0)
  
  // Goal opening angle (simplified: actual trigonometric threshold ~8-10°)
  IF distance_from_line > 0 AND y_offset_from_center > 0 THEN
    angle := ATAN2(y_offset_from_center, distance_from_line)
    // Goal width is 7.32m, visible width = 7.32 * COS(angle)
    impossible_angle := angle > RADIANS(75)  // < 2m visible opening
  ELSE
    impossible_angle := FALSE
  END IF
  
  RETURN behind_goal_line OR acute_angle OR impossible_angle
END FUNCTION
```

**Examples:**
- (105, 55): Behind goal line by 0m, 21m off-center → should reject shot, allow only pass/cross
- (103, 18): 2m from line, 16m off-center → ~83° angle, <1m visible goal → should heavily favor cross over shot
- (98, 12): 7m from line, 22m off-center → ~72° angle → should bias 80% cross, 20% shot
- (90, 60): 15m from line, 26m off-center → still wide but plausible → normal shot probability


### Bug Condition 2-3: Missing Restart Mechanics (Throw-In, Goal Kick)

**Current Defect:**  
When a possession sequence or attack chain results in the ball going out of bounds (y < 2 or y > 66 for throw-in, x ≥ 105 off a shot/clearance for goal kick), no restart event is generated. Play continues unrealistically as if the ball never left the pitch.

**Formal Specification:**
```
FUNCTION isBugCondition_MissingThrowIn(ball_x, ball_y, last_touch_team)
  INPUT: ball_x (float), ball_y (float), last_touch_team (string)
  OUTPUT: boolean
  
  went_out_sideline := (ball_y < 2.0) OR (ball_y > 66.0)
  NOT_goal_kick := ball_x < 105.0  // Ball didn't cross goal line
  
  RETURN went_out_sideline AND NOT_goal_kick AND NOT_throw_in_event_emitted
END FUNCTION

FUNCTION isBugCondition_MissingGoalKick(ball_x, ball_y, last_touch_team, defending_team)
  INPUT: ball_x, ball_y, last_touch_team, defending_team
  OUTPUT: boolean
  
  crossed_goal_line := ball_x >= 105.0
  not_between_posts := (ball_y < 30.34) OR (ball_y > 37.66)
  attacking_team_touch := last_touch_team != defending_team
  
  RETURN crossed_goal_line AND not_between_posts AND attacking_team_touch 
         AND NOT_goal_kick_event_emitted
END FUNCTION
```

**Examples:**
- Ball at (85, 1): Out for throw-in, attacking team last touch → defending team throw-in
- Ball at (105, 20): Crossed goal line, off a shot, outside posts → goal kick for defending team
- Ball at (105, 35): Crossed goal line BETWEEN posts → GOAL (existing logic, unchanged)


### Bug Condition 4: Missing Offside Detection

**Current Defect:**  
No mechanism tracks defender positions at the moment a forward pass is played. An attacker ahead of the second-to-last defender when receiving a pass in the attacking half is not flagged offside. Such attacks proceed to shots and can score.

**Formal Specification:**
```
FUNCTION isBugCondition_Offside(receiver_x, pass_x, defenders_positions, attacking_half)
  INPUT: receiver_x (float) — receiver's x at pass moment
         pass_x (float) — ball x when pass was played
         defenders_positions (list of float) — all defender x-coordinates
         attacking_half (boolean) — receiver_x > 52.5 for home team
  OUTPUT: boolean
  
  forward_pass := receiver_x > pass_x
  
  IF NOT forward_pass OR NOT attacking_half THEN
    RETURN FALSE  // Not offside (backward pass or own half)
  END IF
  
  // Sort defenders by x (distance from their own goal line at x=0 for home def)
  sorted_defenders := SORT(defenders_positions)
  
  // Second-to-last defender (goalkeeper typically last, CB second-last)
  IF LENGTH(sorted_defenders) >= 2 THEN
    offside_line := sorted_defenders[1]  // Second element (0-indexed)
  ELSE IF LENGTH(sorted_defenders) == 1 THEN
    offside_line := sorted_defenders[0]
  ELSE
    offside_line := 0.0  // No defenders (edge case)
  END IF
  
  // Offside if receiver ahead of offside line when ball was played
  is_offside := receiver_x > offside_line
  
  RETURN is_offside AND NOT_offside_event_emitted
END FUNCTION
```

**Examples:**
- Pass at x=70 to receiver at x=95, 2nd-last defender at x=88 → OFFSIDE, stop attack
- Pass at x=70 to receiver at x=85, 2nd-last defender at x=88 → ONSIDE, continue
- Pass at x=70 to receiver at x=50 (backward) → NOT offside regardless of defenders
- Pass in own half (x < 52.5) → NOT offside even if ahead of all defenders


### Bug Condition 5-6: Missing Celebration & Incomplete Center Restart

**Current Defect:**  
When a goal is scored, the system immediately sets `state.pending_kickoff_for = conceding_team` and continues to the next sequence. No celebration pause occurs, no time is added to the clock, and while the ball resets to center circle, player positions remain displaced from their pre-goal states.

**Formal Specification:**
```
FUNCTION isBugCondition_MissingCelebration(goal_scored, celebration_occurred)
  INPUT: goal_scored (boolean), celebration_occurred (boolean)
  OUTPUT: boolean
  
  RETURN goal_scored AND NOT celebration_occurred
END FUNCTION

FUNCTION isBugCondition_IncompleteKickoffReset(
    pending_kickoff, ball_reset, players_reset)
  INPUT: pending_kickoff (boolean), ball_reset (boolean), players_reset (boolean)
  OUTPUT: boolean
  
  RETURN pending_kickoff AND ball_reset AND NOT players_reset
END FUNCTION
```

**Examples:**
- Goal at 67:23 → should pause 18 seconds → resume at 67:41 (celebration added to clock)
- Goal at 67:23 → currently resumes immediately at 67:23+next_sequence_duration
- Kickoff after goal → ball at (52.5, 34) but striker still at (102, 35) from goal sequence
- Expected kickoff → ball at (52.5, 34), all players at home_x/home_y (formation reset)


## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Corner causality (`pending_corner_for` queue, consumed before sequence situation roll) remains exactly as implemented in Checkpoint 6
- Existing shot outcome logic (xG calculation, body_part/pressure multipliers, save evaluation) is untouched
- Goal events (`GOAL`, momentum shift, score update) continue as-is; only **post-goal flow** (celebration + full reset) is added
- Substitution mechanics (stamina drain, tactical/game-state subs, position_engine registration) remain unchanged
- Position Engine spatial state (record_touch, drift_minute, plausibility_at) operates identically
- Event timeline append + stamina drain in `_absorb_chain` continues for all events
- Red card and foul handling logic is preserved

**Scope:**  
All inputs that do NOT involve the six bug conditions (shots from realistic positions/angles, balls staying in bounds, onside attacks, existing corner/foul/sub flows) should produce identical event timelines and outcomes as the current unfixed system.


## Hypothesized Root Cause

### 1. Unrealistic Shot Selection (Bug 1)

**Root Cause:** `AttackChain._shot_location()` samples coordinates without geometric awareness of goal visibility. The subsequent angle penalty in `XGEngine.calculate()` reduces xG after-the-fact, but the **action selection** (shoot vs. cross vs. pass) ignores geometry entirely—a player 2m behind the goal line is offered "shoot" as an option.

**Missing Logic:**
- No pre-shot geometry check rejecting x ≥ 105 or acute angles
- No action biasing (cross vs. shoot) based on position angle/distance
- Wide players (LW/RW) near byline default to generic action weights

### 2-3. Missing Restart Mechanics (Bugs 2-3)

**Root Cause:** No out-of-bounds detection exists in `PossessionChain` or `AttackChain`. Event chains track ball coordinates (location_x, location_y, end_x, end_y) but never check if they cross pitch boundaries. The engine assumes the ball stays in play indefinitely.

**Missing Logic:**
- No boundary check: `(y < 2 or y > 66) → THROW_IN`
- No goal-line check: `(x ≥ 105 and not_goal) → GOAL_KICK`
- No restart event types linked to these conditions
- No possession award logic for restarts


### 4. Missing Offside Detection (Bug 4)

**Root Cause:** `MatchState` does not track defender positions at pass moment. `AttackChain.generate()` picks a receiver and emits a shot without checking if the receiver was ahead of the second-to-last defender when the chance-creation pass was played.

**Missing Logic:**
- No per-team defender position snapshot when a forward pass occurs
- No "offside line" calculation (2nd-last defender's x-coordinate)
- No comparison: `receiver_x > offside_line` in attacking half
- No `OFFSIDE` event type or attack-stopping logic

### 5-6. Missing Celebration & Incomplete Reset (Bugs 5-6)

**Root Cause:** `_absorb_chain()` detects `goal_scored` and immediately sets `pending_kickoff_for`, then returns `True` to break the sequence loop. No pause occurs between goal detection and the next minute's simulation. `pending_kickoff_for` resets ball to (52.5, 34) but doesn't reset player positions.

**Missing Logic:**
- No celebration duration sampled (10-30 seconds)
- No time addition to `state.minute` and `state.second`
- No `GOAL_CELEBRATION` event emitted
- `pending_kickoff_for` resets `last_ball_x/y` but not `position_engine` player states
- No `position_engine.reset_all_to_home()` call on kickoff


## Correctness Properties

Property 1: Bug Condition - Intelligent Shot Selection Geometry

_For any_ attacking position where the bug condition holds (x ≥ 105, or acute angle with < 10° goal visibility), the fixed AttackChain SHALL reject shooting as an action option and instead select cross (if wide) or pass (if central), ensuring no shots originate from behind the goal line or impossible angles.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Bug Condition - Throw-In Restart

_For any_ possession sequence or attack where the ball goes out of bounds via touchline (y < 2 or y > 66, x < 105), the fixed system SHALL emit a THROW_IN event, award possession to the team that did NOT touch it last, and restart play from (x_where_out, 0 or 68).

**Validates: Requirements 2.4, 2.5, 2.6**

Property 3: Bug Condition - Goal Kick Restart

_For any_ attack sequence where the ball crosses the goal line (x ≥ 105) outside the posts (y < 30.34 or y > 37.66) via attacking team touch, the fixed system SHALL emit a GOAL_KICK event with the defending GK as actor and restart play from (8-18, 34).

**Validates: Requirements 2.7, 2.8, 2.9**


Property 4: Bug Condition - Offside Detection

_For any_ forward pass in the attacking half where the receiver is ahead of the second-to-last defender at pass moment, the fixed system SHALL emit an OFFSIDE event, stop the attack immediately (no shot/goal), and award a free kick to the defending team at the offside location.

**Validates: Requirements 2.10, 2.11, 2.12**

Property 5: Bug Condition - Goal Celebration Sequence

_For any_ goal scored, the fixed system SHALL pause simulation for 10-30 seconds (random), add that duration to the match clock (minute:second), emit a GOAL_CELEBRATION event with duration metadata, then proceed to center restart.

**Validates: Requirements 2.13, 2.14, 2.15**

Property 6: Bug Condition - Full Center Restart Formation Reset

_For any_ goal scored, after celebration, the fixed system SHALL reset all player positions to home_x/home_y (PositionEngine formation reset), reset ball to (52.5, 34), emit KICKOFF event, ensure all players are in their own halves, then begin kickoff possession sequence.

**Validates: Requirements 2.16, 2.17, 2.18**

Property 7: Preservation - Existing Match Flow Unchanged

_For any_ input where the bug conditions do NOT hold (realistic shot positions, ball staying in bounds, onside attacks, existing corner/foul logic), the fixed system SHALL produce exactly the same event timeline, xG, goals, cards, and final score as the original unfixed system.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12**


## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

#### File 1: `event_chain.py` — AttackChain Modifications

**Class**: `AttackChain`

**Specific Changes**:

1. **Add Geometry-Aware Action Selector** (`_select_action_from_position`)
   - New static method before `generate()`
   - Inputs: `x`, `y`, `player_position` (LW/RW/ST/etc.)
   - Logic:
     ```python
     # Reject shooting behind goal line
     if x >= 105.0:
         return "pass" if 20 < y < 48 else "cross"
     
     # Calculate angle to goal posts
     dist_from_line = 105.0 - x
     y_offset = abs(y - 34.0)
     angle = atan2(y_offset, dist_from_line) if dist_from_line > 0 else pi/2
     
     # Acute angle logic
     if angle > radians(70):  # <2.5m visible goal opening
         return "cross" if y < 20 or y > 48 else "pass"
     if angle > radians(60):  # <4m visible
         # Bias toward cross: 80% cross, 20% shot
         return weighted_choice(["cross", "shot"], [0.80, 0.20])
     
     # Wide positions near byline
     if x > 95 and player_position in ["LW", "RW", "LB", "RB"]:
         if y < 20 or y > 48:
             return weighted_choice(["cross", "pass", "dribble", "shot"], 
                                     [0.65, 0.20, 0.10, 0.05])
     
     return "shoot"  # Realistic shooting position
     ```

