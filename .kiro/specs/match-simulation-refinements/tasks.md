# Implementation Plan

## Overview

This implementation plan addresses six interconnected match flow defects using the bug condition methodology:
1. Unrealistic shot selection from impossible angles
2. Missing throw-in mechanic
3. Missing goal kick mechanic
4. Missing offside detection
5. Missing goal celebration sequence
6. Incomplete center restart after goals

**Testing Approach:** Exploratory property-based testing BEFORE fix, then preservation testing.

---

## Task Dependency Graph

```
1 (Bug Condition Tests)
├─ 1.1 Unrealistic shot geometry test
├─ 1.2 Throw-in detection test
├─ 1.3 Goal kick detection test
├─ 1.4 Offside detection test
├─ 1.5 Goal celebration test
└─ 1.6 Kickoff formation reset test

2 (Preservation Tests - BEFORE fix)
├─ 2.1 Existing shot outcome logic
├─ 2.2 Corner causality
└─ 2.3 Position engine spatial state

3 (Implementation)
├─ 3.1 Geometry-aware shot selector
├─ 3.2 Out-of-bounds detection
├─ 3.3 Offside tracking
├─ 3.4 Celebration + formation reset
├─ 3.5 Verify exploration tests pass
└─ 3.6 Verify preservation tests pass

4 (Checkpoint)

---

## Bug Condition Exploration Tests (BEFORE Fix)

- [x] 1. Write bug condition exploration tests
  - **Property 1: Bug Condition - Unrealistic Shot Geometry** - Zero visible goal opening
  - **CRITICAL**: These tests MUST FAIL on unfixed code - failure confirms the bugs exist
  - **DO NOT attempt to fix the tests or the code when they fail**
  - **NOTE**: These tests encode the expected behavior - they will validate the fixes when they pass after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bugs exist
  - **Test Implementation:**
  
  **1.1 Unrealistic Shot Geometry**
  - Test that shots from position (105, 55) — behind goal line — are rejected
  - Test that acute angle positions (x > 100, y < 20 or y > 48) bias toward cross/pass over shot
  - Test that wide players near byline (x > 95, y < 20 or y > 48) select cross ≥65% of time
  - From Bug Condition: isBugCondition_UnrealisticShot pseudocode in design.md
  - Expected behavior: _select_action_from_position() returns "pass"/"cross" for impossible positions
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (proves bug exists - shots allowed from impossible positions)
  - Document counterexamples: e.g., "shot attempted from (105, 55) instead of pass/cross"
  
  **1.2 Throw-In Detection**
  - Test that ball at (85, 1) triggers THROW_IN event
  - Test that possession awards to team that didn't touch last
  - From Bug Condition: isBugCondition_MissingThrowIn pseudocode
  - Expected behavior: THROW_IN event emitted when y < 2 or y > 66, x < 105
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (no throw-in event generated)
  - Document: "ball out at (85, 1), no THROW_IN event, play continued unrealistically"

  **1.3 Goal Kick Detection**
  - Test that ball at (105, 20) off attacking touch triggers GOAL_KICK event
  - Test that defending GK is primary actor
  - Test that play restarts from GK position (x ≈ 8-18, y ≈ 34)
  - From Bug Condition: isBugCondition_MissingGoalKick pseudocode
  - Expected behavior: GOAL_KICK event when x ≥ 105, y < 30.34 or y > 37.66, attacking team touch
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (no goal kick event, play continues)
  - Document: "ball out at (105, 20), no GOAL_KICK, no restart"
  
  **1.4 Offside Detection**
  - Test forward pass at x=70 to receiver at x=95, 2nd-last defender at x=88 triggers OFFSIDE
  - Test that attack stops immediately (no shot/goal)
  - Test free kick awarded to defending team
  - From Bug Condition: isBugCondition_Offside pseudocode
  - Expected behavior: OFFSIDE event, attack stopped, no shot possible
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (offside attack proceeds to shot, potentially scores)
  - Document: "receiver at x=95 ahead of x=88 defender, no OFFSIDE, shot allowed"
  
  **1.5 Goal Celebration Sequence**
  - Test that goal at 67:23 adds 10-30 seconds to match clock
  - Test that GOAL_CELEBRATION event emitted with duration metadata
  - Test that simulation pauses during celebration
  - From Bug Condition: isBugCondition_MissingCelebration pseudocode
  - Expected behavior: time advances (e.g., 67:23 + 18s = 67:41), GOAL_CELEBRATION event logged
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (no pause, no time addition, no celebration event)
  - Document: "goal at 67:23, immediately resumed at 67:23+next_sequence, no celebration"

  **1.6 Kickoff Formation Reset**
  - Test that after goal + celebration, all players reset to home_x/home_y
  - Test that ball resets to (52.5, 34)
  - Test that KICKOFF event emitted
  - Test that all players in correct halves before play resumes
  - From Bug Condition: isBugCondition_IncompleteKickoffReset pseudocode
  - Expected behavior: PositionEngine.reset_all_to_home() called, players at formation positions
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (ball at center, players still displaced from pre-goal positions)
  - Document: "kickoff after goal, ball at (52.5, 34) but striker still at (102, 35) from goal sequence"
  
  - Mark task complete when ALL 6 tests are written, run, and failures documented
  - _Requirements: 1.1-1.15 (all bug conditions from bugfix.md)_

---

## Preservation Property Tests (BEFORE Fix)

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Existing Match Flow Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs
  - Write property-based tests capturing observed behavior patterns
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  
  **2.1 Existing Shot Outcome Logic**
  - Observe: shot from realistic position (90, 34) calculates xG, applies body_part multiplier
  - Observe: on-target shot calls GoalkeeperEngine.evaluate_save() with same params
  - Observe: woodwork events generate corner or rebound as existing logic dictates
  - Property: for all realistic shot positions (60 < x < 103, appropriate y funnel), xG calculation, save evaluation, and outcome logic remain unchanged
  - Verify test passes on UNFIXED code
  - From Preservation Requirements: 3.5, 3.6, 3.7 in bugfix.md

  **2.2 Corner Causality**
  - Observe: blocked shot sets pending_corner_for in unfixed code
  - Observe: corner consumed before next sequence situation roll
  - Observe: corner anchored to state.last_ball_x/y (no teleport)
  - Property: for all blocked shots or ineffective clearances, corner causality system continues unchanged
  - Verify test passes on UNFIXED code
  - From Preservation Requirements: 3.1 in bugfix.md
  
  **2.3 Position Engine Spatial State**
  - Observe: player touches ball → current_x/current_y updated via position_engine.record_touch()
  - Observe: minute elapses → uninvolved players drift toward home positions
  - Observe: substitution → incoming player registered with PositionEngine.register_substitute()
  - Property: for all events with real coordinates, spatial state updates continue unchanged
  - Verify test passes on UNFIXED code
  - From Preservation Requirements: 3.8, 3.9, 3.10 in bugfix.md
  
  - _Requirements: 3.1-3.12 (all preservation requirements from bugfix.md)_

---

## Implementation

- [ ] 3. Fix for match simulation refinements

  - [ ] 3.1 Implement geometry-aware shot selector in event_chain.py
    - Add `_select_action_from_position(x, y, player_position)` static method to AttackChain
    - Reject shooting behind goal line (x ≥ 105) → return "pass" if central, "cross" if wide
    - Calculate angle to goal posts: `atan2(y_offset, dist_from_line)`
    - Acute angle logic: angle > 70° → return "cross" or "pass"
    - Moderate angle (60-70°): weighted_choice(["cross", "shot"], [0.80, 0.20])
    - Wide positions near byline (x > 95, y < 20 or y > 48): weighted_choice(["cross", "pass", "dribble", "shot"], [0.65, 0.20, 0.10, 0.05])
    - Integrate into AttackChain.generate() before `_shot_location()`
    - _Bug_Condition: isBugCondition_UnrealisticShot(shot_x, shot_y) from design_
    - _Expected_Behavior: expectedBehavior(action) = action in {"pass", "cross"} when geometry prohibits shooting_
    - _Preservation: Existing shot outcome logic (xG, save evaluation, woodwork) from design_
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.2 Implement out-of-bounds detection with restarts
    - Add boundary check at end of PossessionChain.generate() and AttackChain.generate()
    - Throw-in detection: `(end_y < 2 or end_y > 66) AND end_x < 105`
    - Goal kick detection: `end_x ≥ 105 AND (end_y < 30.34 or end_y > 37.66) AND attacking_team_touch`
    - Emit THROW_IN event: award possession to team that didn't touch last, restart at (x_where_out, 0 or 68)
    - Emit GOAL_KICK event: defending GK as actor, restart from (8-18, 34)
    - Update ChainResult to include `restart_required: bool`, `restart_type: str`, `restart_team: str`
    - Update MatchEngine._absorb_chain() to consume restart flags and emit restart events
    - _Bug_Condition: isBugCondition_MissingThrowIn(ball_x, ball_y, last_touch_team) AND isBugCondition_MissingGoalKick(ball_x, ball_y, last_touch_team, defending_team)_
    - _Expected_Behavior: THROW_IN or GOAL_KICK event emitted, possession transferred, play restarts from correct coordinates_
    - _Preservation: Corner causality, existing possession flow_
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

  - [ ] 3.3 Implement offside detection system
    - Add `_check_offside(receiver_x, pass_x, defending_team_players, attacking_half)` to AttackChain
    - Extract defender x-coordinates: `[p.current_x for p in defending_team_players if p.position in ["CB", "LB", "RB", "CDM"]]`
    - Sort defenders by x (distance from their own goal line)
    - Calculate offside line: `sorted_defenders[1]` (second-to-last defender)
    - Check if receiver ahead of offside line when ball was played: `receiver_x > offside_line AND forward_pass AND attacking_half`
    - If offside: emit OFFSIDE event, set ChainResult.offside_detected = True, stop attack
    - Integrate check in AttackChain.generate() BEFORE shot selection
    - Add offside check to counter-attack flow in TransitionChain._generate_counter()
    - _Bug_Condition: isBugCondition_Offside(receiver_x, pass_x, defenders_positions, attacking_half)_
    - _Expected_Behavior: OFFSIDE event emitted, attack stopped, no shot/goal possible, free kick awarded_
    - _Preservation: Goal events, momentum shifts, xG accumulation for non-offside attacks_
    - _Requirements: 2.10, 2.11, 2.12_

  - [ ] 3.4 Implement goal celebration + full formation reset
    - **Celebration Sequence:**
      - In MatchEngine._absorb_chain(), after detecting goal_scored:
      - Sample celebration duration: `random.randint(10, 30)` seconds
      - Add duration to match clock: `state.minute += duration // 60`, `state.second += duration % 60` (handle minute overflow)
      - Emit GOAL_CELEBRATION event with duration in metadata: `{"duration": duration}`
      - Set `state.pending_kickoff_for = conceding_team`
    
    - **Full Formation Reset:**
      - In MatchEngine._simulate_minute(), before processing sequences:
      - Check if `state.pending_kickoff_for` is set
      - If set:
        - Call `position_engine.reset_all_to_home(state.pending_kickoff_for, home_team, away_team)`
        - Reset ball: `state.last_ball_x = 52.5`, `state.last_ball_y = 34.0`
        - Emit KICKOFF event with kickoff team as actor, location (52.5, 34)
        - Verify all players in correct halves: `home_team players.current_x < 52.5`, `away_team players.current_x > 52.5`
        - Clear flag: `state.pending_kickoff_for = ""`
    
    - **PositionEngine method:**
      - Add `reset_all_to_home(kickoff_team, home_team, away_team)` to PositionEngine
      - For all players in both teams: `state.current_x = state.home_x`, `state.current_y = state.home_y`
      - Validate formation positioning: all players within their own half
    
    - _Bug_Condition: isBugCondition_MissingCelebration(goal_scored, celebration_occurred) AND isBugCondition_IncompleteKickoffReset(pending_kickoff, ball_reset, players_reset)_
    - _Expected_Behavior: 10-30 second pause, time added to clock, GOAL_CELEBRATION event, all players at home positions, ball at center, KICKOFF event_
    - _Preservation: Score update, momentum shift, xG accumulation remain unchanged_
    - _Requirements: 2.13, 2.14, 2.15, 2.16, 2.17, 2.18_

  - [ ] 3.5 Verify bug condition exploration tests now pass
    - **Property 1: Expected Behavior** - Intelligent Shot Selection, Throw-In, Goal Kick, Offside, Celebration, Formation Reset
    - **IMPORTANT**: Re-run the SAME tests from task 1 - do NOT write new tests
    - The tests from task 1 encode the expected behavior
    - When these tests pass, it confirms the expected behavior is satisfied
    - Run all 6 bug condition exploration tests from task 1
    - **EXPECTED OUTCOME**: All tests PASS (confirms bugs are fixed)
    - Verify:
      - 1.1: Shot from (105, 55) now returns "pass"/"cross" action
      - 1.2: Ball at (85, 1) generates THROW_IN event
      - 1.3: Ball at (105, 20) generates GOAL_KICK event
      - 1.4: Offside at x=95 (defender at x=88) generates OFFSIDE, no shot allowed
      - 1.5: Goal at 67:23 adds time, emits GOAL_CELEBRATION
      - 1.6: Kickoff resets all player positions to home_x/home_y
    - _Requirements: 2.1-2.18 (Expected Behavior Properties from design.md)_

  - [ ] 3.6 Verify preservation tests still pass
    - **Property 2: Preservation** - Existing Match Flow Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run all 3 preservation property tests from task 2
    - **EXPECTED OUTCOME**: All tests PASS (confirms no regressions)
    - Verify:
      - 2.1: Realistic shot positions still calculate xG correctly, save evaluation unchanged
      - 2.2: Blocked shots still generate corners via pending_corner_for
      - 2.3: Position engine still updates current_x/y on touch, drift works
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: 3.1-3.12 (Preservation Requirements from design.md)_

---

## Checkpoint

- [ ] 4. Checkpoint - Ensure all tests pass
  - Run full test suite (exploration + preservation)
  - Verify all 6 bug fixes work correctly:
    - Unrealistic shots rejected (geometry-aware action selection)
    - Throw-ins and goal kicks generate proper restart events
    - Offside detection stops attacks in attacking half
    - Goal celebrations add time to clock
    - Kickoffs reset full formation (ball + all players)
  - Verify preservation requirements met:
    - Existing shot outcome logic unchanged
    - Corner causality system intact
    - Position engine spatial state accurate
  - Ask user if questions arise
  - _Requirements: All 42 requirements (1.1-3.12) validated_
