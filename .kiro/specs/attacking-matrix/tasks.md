# Implementation Plan

## Overview

Add an intelligent attacking decision layer: a High-Density Attacking Matrix that gives
every ball carrier spatial awareness of shooting windows, passing corridors, and teammate
strategic value, resolving into SHOOT / KEY_PASS / PROGRESSIVE_PASS / RECYCLE_PASS —
steered by match scenario and player DNA. Follow the property-based methodology used by
the defensive-awareness checkpoint: write property tests against the new module as it
lands, then verify preservation of existing behaviour.

## Task Dependency Graph

```
1 (Attacking Matrix module)
├─ 1.1 S_viability: distance decay × pressure tax × angle multiplier
├─ 1.2 L_clear: point-segment corridor clearance
├─ 1.3 V_strategic: progress × freedom × depth × lane
├─ 1.4 Network zones + scenario derivation
├─ 1.5 Decision resolution (priority cascade + DNA thresholds)
└─ 1.6 Unit properties: monotonicity, pressure tax, central>wide, lane blocking,
       strategic ordering, scenario decisions

2 (PossessionChain integration)
├─ 2.1 Per-touch matrix evaluation at the MAIN ACTION step
├─ 2.2 SHOOT decision → ChainResult hand-off fields + break
├─ 2.3 Matrix-selected pass targets + pass aimed at live receiver position
├─ 2.4 Fallback to existing _pick_receiver/_pass_destination without position engine
└─ 2.5 Properties: shot hand-off, no-PE preservation

3 (MatchEngine wiring)
├─ 3.1 Hand-off: shoot_decision → ChainDispatcher.attack anchored at (x, y)
├─ 3.2 No double-fire with independent shot_prob path
└─ 3.3 Properties: full-match smoke test with matrix decisions present

4 (Selector delegation + Checkpoint)
├─ 4.1 AttackChain._select_action_from_position delegates geometry to the matrix
├─ 4.2 Existing suite green (no new failures beyond known pre-existing probes)
└─ 4.3 All attacking-matrix property tests pass
```

## Implementation Tasks

- [x] 1. Build `attacking_matrix.py`
  - Constants: `GOAL_TOP_POST_Y`, `GOAL_BOTTOM_POST_Y`, `REFERENCE_ANGLE`,
    `CLOSE_NETWORK_M`, `PRESSURE_TAX_RANGE_M`, `LANE_BLOCK_RADIUS_M`, `FREE_RADIUS_M`
  - `AttackingDecision` dataclass: `action, shot_score, target, target_lane_clearance,
    target_strategic_value, zone, under_pressure, best_lane_clearance, best_strategic_value`
  - `nearest_defender_dist()`, `shot_score()`, `lane_clearance()`,
    `strategic_value()`, `network_zone()`, `scenario_for()`, `decide()`
  - Pure geometry + DNA; zero dependency on the simulation loop
  - All functions position-engine-gated: `position_engine=None` ⇒ fallback decision that
    never SHOOTs and never selects a target
  - _Requirements: 2.1, 2.3, 2.5, 2.8–2.11_

- [x] 2. Modify `event_chain.py`
  - `ChainResult`: add `shoot_decision/shoot_player/shoot_x/shoot_y/shoot_under_pressure`
  - `PossessionChain.generate`: at the MAIN ACTION step, when `position_engine` is wired,
    evaluate `AttackingMatrix.decide(...)`; SHOOT ⇒ set hand-off fields + `break`;
    pass actions ⇒ select the matrix target and aim the destination at the receiver's live
    position (new `_pass_destination_to_target` helper), set `is_long/is_prog` from the
    decision, tag `metadata["attacking_matrix"]`
  - Fallback: no position engine ⇒ existing `_pick_receiver` / `_pass_destination` unchanged
  - `AttackChain._select_action_from_position`: re-implement using the matrix's
    angular-width model, keeping signature and return contract
  - SHOOT hand-off carries a take-probability gate `clamp01((shot_score − 0.60)/0.40)`
    (see bugfix.md §5) so per-match shot volume stays near the pre-feature band
  - _Requirements: 2.2, 2.4, 2.6, 2.7, 3.1–3.4_

- [x] 3. Modify `match_engine.py`
  - `_simulate_minute`: after `_absorb_chain(poss_result)`, if `shoot_decision` and not
    `possession_lost` ⇒ `ChainDispatcher.attack(situation=OPEN_PLAY, context_x=shoot_x,
    context_y=shoot_y, position_engine=...)`, absorb, and mark the sequence as shot-taken
    so the independent `shot_prob` block does not double-fire
  - _Requirements: 2.2, 3.5, 3.7_

- [x] 4. Export + Checkpoint
  - Run `test_attacking_matrix.py` (new) + full existing suite
  - Run a short full-match simulation and confirm matrix decisions appear in the timeline
    and shots still resolve through the existing xG/GK pipeline
  - _Requirements: 2.11, preservation 3.1–3.7_

## Checkpoint

- [x] 5. All attacking-matrix property tests pass; the existing suite gains NO new
  failures beyond the known pre-existing bug-probe/stale-API failures; a short full-match
  sim completes with matrix decisions (SHOOT hand-offs and matrix-tagged passes) present
  in the timeline.
