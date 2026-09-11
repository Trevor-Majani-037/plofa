# Implementation Plan

## Overview

Add an intelligent defensive-awareness layer: a live per-team danger/threat level driven
by ball↔defended-goal geometry, headed vs foot clearances, spatially-aware defender
selection, and coordinated goal-side block positioning. Follow the property-based
methodology used by the match-simulation refinements checkpoint: write property tests
against the new modules as they land, then verify preservation of existing behaviour.

## Task Dependency Graph

```
1 (Threat Engine module)
├─ 1.1 DangerAssessment / assess() with monotonicity guarantees
├─ 1.2 Danger reduction on defensive success / goal peak+reset
├─ 1.3 Danger history + report for export
└─ 1.4 Unit properties: monotonicity, central>wide, zones, zero-threat

2 (DefensiveChain)
├─ 2.1 Headed vs foot clearance selection + attribute-driven success
├─ 2.2 Danger metadata on defensive events
├─ 2.3 Spatially-plausible defender pick
└─ 2.4 Properties: headed/foot, attribute sensitivity, relief

3 (PositionEngine + MatchEngine wiring)
├─ 3.1 defensive_block() coordination
├─ 3.2 threat engine lifecycle in MatchEngine
├─ 3.3 danger-scaled action weights in _simulate_minute
├─ 3.4 defensive_block hook in _run_minute
└─ 3.5 Properties: coordination block pull + zero-danger preservation

4 (Exporter + Checkpoint)
├─ 4.1 defensive-awareness JSON section
├─ 4.2 Full-match smoke test (timeline + goals + threat history sane)
└─ 4.3 All property tests pass; existing suite still green
```

## Implementation Tasks

- [ ] 1. Build `threat_engine.py`
  - `DangerAssessment` dataclass: `level, proximity, centrality, zone, zone_mult,
    pressure, momentum, shot_mult, dist_to_goal, ball_x, ball_y, attackers_near,
    defenders_near`
  - `ThreatEngine` with `assess()`, `observe_event()`, `danger_at()`, `on_goal()`,
    `classify()`, `history`, `report()`
  - `danger_after_clearance()` / defensive-win relief helpers
  - Pure geometry, zero dependency on the simulation loop
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 2. Modify `event_chain.py` `DefensiveChain`
  - New params: `danger_level=0.0`, `ball_aerial=False`, `own_goal_x=105.0`,
    `position_engine=None`
  - Clearance: choose headed vs foot; attribute-driven success; danger-aware panic;
    danger-aware destination (away from defended goal); `body_part` + `clearance_type`
    metadata; failure keeps ball in danger zone
  - `_pick_defender(action_type, position_engine, x, y)` spatial plausibility
  - Danger metadata (`danger_before`/`danger_after`) on all defensive events
  - _Requirements: 2.6, 2.7, 2.8, 2.9_

- [ ] 3. Modify `position_engine.py` + `match_engine.py`
  - `PositionEngine.defensive_block(...)` compact goal-side block (≥25 danger, out of
    possession, ball in defended half)
  - `MatchEngine.__init__`: create/register `self.threat`
  - `_absorb_chain`: observe events → live danger, relief on wins, goal peak+reset
  - `_simulate_minute`: danger-scaled action weights + aerial inference + pass danger
    into defensive chain
  - `_run_minute`: defensive_block hook after drift
  - `MatchResult.threat` passthrough
  - _Requirements: 2.3, 2.10, 2.11_

- [ ] 4. Export + Checkpoint
  - `export_json`: `"defensive_awareness"` section (per-team peak/avg danger, minutes in
    each band, headed/foot clearance split, danger timeline)
  - Run `test_defensive_awareness.py` (new) + full existing suite
  - Run a full match simulation and sanity-check the threat history
  - _Requirements: 2.5, preservation 3.1–3.7_

## Checkpoint

- [ ] 5. All property tests pass; existing suite (tests.py, test_sequence_engine.py,
  preservation tests) stays green; a full match sim completes with a sane danger
  history and headed/foot clearances appearing in the timeline.
