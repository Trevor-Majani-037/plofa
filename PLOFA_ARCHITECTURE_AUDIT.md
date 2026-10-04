# PLOFA ARCHITECTURE AUDIT
### Phase 0 deliverable — Next-Generation Football Intelligence Engine

Read-only audit. No production file in this repository was modified to produce this document.
Companion working state: a `PLOFA_ARCHITECTURE_AUDIT.md` (this file) added at repo root.

**Scope reviewed:** match entry points, season/fixture/roster pipeline, `MatchEngine`,
`event_chain`, `possession_*`, geometry/pitch-control/marking/threat, `FootballBrain` +
`DecisionBrain` + `brain_integration` + `brain_sensors`, the full learning/evolution/
surrogate stack, role behaviour modules, exporter/output layer, Streamlit dashboard, and
the test suites.

---

## 0. WORKING-TREE STATE (read before trusting any number)

The repository is **mid-refactor**. `git status` is dirty:

| Modified (tracked) | Meaning |
|---|---|
| `brain_evolution.py` | In-flight **batched-sensor** refactor: `synthetic_fitness()` signature changed from `(brain, position, n_states, seed, surrogate, pop_diversity, goal_bias)` to `(brain, position, batched_sensors, surrogate, pop_diversity)`. The sensor corpus is pre-generated per population member in `evolve()` (preserving the old RNG sequence); the forward pass is **still unbatched** (`brain.forward(batched_sensors[i])`), so the comment `# BATCHED` is aspirational. |
| `football_brain.py` | `_softmax` now supports 2-D arrays (`axis=-1`) for future batched forward passes. |
| `brains/GK.json` | **Overwritten by a 2-generation smoke run** (best_fitness 0.0104, seed 42, 400 states). This is *not* the T1 production GK brain. |
| `brains/_manifest.json` | Overwritten to match the GK smoke run (positions `["GK"]`, generations 2). T1 provenance lives in `brains/_manifest_all.json` (stale, 5/11 positions) and in `AGENTS.md`. |

Untracked: `brain_evolution_old.py`, `sys_patch.py`, `baseline/`, `scripts/`, and ~15
root-level scratch scripts (`test_baseline*.py`, `test_parity*.py`, `test_batch*.py`,
`test_timing*.py`, `test_unbatched_forward.py`).

**Consequences:**
1. `tests/test_football_brain.py::test_synthetic_fitness_accepts_surrogate` is **RED** —
   the committed test calls the old `n_states=` API. Observed: 36 passed / 1 failed.
2. `tests/tests.py::test_pass_matrix_sums_match_real_events` is **RED** (pre-existing,
   documented in README §6): `PassMatrix.build` counts successful crosses while the test
   tally does not. Observed 340 vs 337.
3. The stated "suites green 24/24 + 9/9" in `AGENTS.md` refers to the committed checkpoint,
   not the working tree.
4. `brains/GK.json` should be restored from `brains_backup_incumbent_20260911_g160swap/`
   (the T1-incumbent mirror) or from git before any real validation.

**Recommendation:** before any V2 work, create an immutable baseline tag/commit of the
*committed* checkpoint and stash or finish the batched-sensors refactor. A moving working
tree invalidates every A/B measurement from here on.

---

## A. CURRENT ARCHITECTURE DIAGRAM

### A.1 Season → match → result

```
PLOFA-2026-2027.xlsx  (roster + fixtures sheet, source of truth)
        │
        ▼
roster_loader.get_loader()  ── PlayerRecord, FORMATION_SLOTS (9 formations), SUB_TIMING
        │
        ├── auto_run_match.py  (PRODUCTION; agents MUST NOT run — writes season_state.json)
        ├── run_match.py       (scratch/manual)
        └── season_manager.py  (batch: round-robin fixture generator, run_matchday / run_full_season)
                 │
                 ▼
        availability resolution  ◄── THREE competing implementations:
                 │                  • SeasonState.is_available()          (JSON ledger; auto runner)
                 │                  • squad_manager.AvailabilityChecker    (reads prior exports)
                 │                  • roster_loader._filter_eligible()    (Excel eligibility cols)
                 ▼
        player_dna.SquadBuilder.build(team, starters)  ── PlayerProfile + DNA + personality + soul
                 │                                        (ManagerPool supplies ManagerProfile)
                 ▼
        MatchConfig ──► MatchEngine(config, home_profile, away_profile)      match_engine.py:1475
                 │
                 ▼
        MatchEngine.simulate()  :2454
                 │  _initialize_simulation → per-minute loop _run_minute → _simulate_minute :3099
                 │     • per-phase goal / card multipliers (0.70→1.50, 0.60→1.80)
                 │     • TacticalAI.adjust → EffectiveTactics (posture); tactical_shapes.formation_stance_for
                 │     • team-press cache: g = TeamPressBrain(team-state); prob = _PRESS_PROB[pos] * g
                 │
                 ▼
        ChainDispatcher  event_chain.py:7871  (the single chain entry point)
                 ├── possession()      → PossessionChain.generate()      :1054   (open play, main carrier chain)
                 ├── attack()          → AttackChain.generate()          :4890   (final-third shot selection)
                 ├── set_piece()       → SetPieceChain.generate()        :5829   (penalty/corner/freekick)
                 ├── transition()      → TransitionChain.generate()      :6671   (counter-attack)
                 ├── defensive_action()→ DefensiveChain.generate()       :6947   (tackle/int/clear/block)
                 ├── goal_kick()       → GoalKickChain.generate()        :7652
                 └── discipline()      → foul / card chain
                 │
                 ▼
        ChainResult  event_chain.py:215
                 │  goal_scored, xg_generated, corner_won, penalty_won, restart_required,
                 │  offside_detected, and the full ordered MatchEvent list (per-touch metadata)
                 ▼
        _absorb_chain  match_engine.py:4079  → timeline, stamina drain, possession, xG attribution (:4419), cards
                 │
                 ▼
        MatchResult  match_engine.py:4923 → exporter.export_all()  exporter.py:2117
                 │
                 ▼
        season persistence:  season_state.json  (authoritative ledger)
                             season_stats.json  (aggregates → build_app_data → plofa_streamlit/app.py)
                             manager_state.json / referee_state.json
```

### A.2 Per-touch decision flow (the only place the neural brain lives)

Inside `PossessionChain.generate()` (`event_chain.py:1054`), the loop over touches is
**layered**, and the neural brain is *not* the top authority:

```
touch begins
  │
  ├─ 1. pressing / counterpress gates            (deterministic + Bernoulli)
  ├─ 2. REGRESSION machine                       :1360–1465   forces receiver / blocks carries
  ├─ 3. AttackingMatrix SHOOT gate               :1466–1528   deterministic threshold (≥0.70 floor)
  │        └─ if fire: emit shot + break  →  the neural SHOOT intent can never fire a shot
  ├─ 4. matrix forced passes (winger ≤20 m cap)  :1529–1548
  ├─ 5. wide-combo override (LW/RW/LB/RB)        :1549–1573, 1754–1772
  ├─ 6. GK distribution pass                     :1576
  │
  ├─ 7. ►► NeuralDecisionBrain.decide(...)        :1599     ◄── the swapped-in layer
  │        returns PlayerDecision(intent, action, confidence, target, …)
  │        intent ∈ {PROGRESSIVE_PASS, SAFE_PASS, THROUGH_BALL, SWITCH, CARRY,
  │                  DRIBBLE, CROSS, SHOOT, RECYCLE, PROTECT_POSSESSION}
  │        action ∈ {"PASS","CARRY"}  (SHOOT/DRIBBLE collapse to PASS unless consumed)
  │
  ├─ 8. winger carry steering (drive/cut)        :1606–1636
  ├─ 9. carry gate: roll < max(0.55, confidence) AND can_carry AND not pressured
  │        AND regression_mode is None AND not GK             :1637–1639
  ├─ 10. ATTACKING_PROPHET scenario override     :1828–1845
  ├─ 11. brain target used ONLY for PROGRESSIVE_PASS/SWITCH with conf > 0.35
  │        everything else → heuristic _pick_receiver()       :1800–1807
  │
  ▼
resolve_carry / resolve_pass / through-ball Bernoulli → PossessionEpisode
  (possession_physics.py:413)
  │
  ├─ resolve_ground_pass  :777 / resolve_pass :794 / resolve_loose_ball :647
  ├─ geometry_engine.resolve_shot :1187 (GK dive geometry, magnus, rolling friction)
  └─ 10 Hz ball–player races (geometry_engine _race_motion)
  │
  ▼
event stamped with decision metadata :1715–1720
  {action, intent, confidence, reason, decision_quality, is_error}
  │
  ▼
next touch / chain end
```

**The world between touches is NOT a learned model.** Shot *outcomes*, offsides, corners,
miscontrols, penalties, loose-ball recoveries and press commitments are Bernoulli draws on
top of a deterministic geometry core. The neural brain picks an *intent*; the football
engine decides what actually happens.

---

## B. CURRENT BRAIN AUDIT

### B.1 `FootballBrain` (the on-ball brain) — `football_brain.py`

| Property | Value |
|---|---|
| Architecture | `INPUT_SIZE=24 → HIDDEN_1=32 (ReLU) → HIDDEN_2=32 (ReLU) → OUTPUT_SIZE=10 (Softmax)` |
| Parameters | **2,186** (all biases are exactly zero as serialized) |
| Init | He init `N(0, √(2/fan_in))`, biases zero; then DNA-seeded scalars: `vision_scale=0.8+vision·0.4` on `w1`, `composure_scale=1.2−composure·0.4` on `w2`, `decisions_scale=0.9+decisions·0.2` on `w3` |
| Forward | numpy only, deterministic per `(weights, sensors)` |
| Intent labels | `PROGRESSIVE_PASS, SAFE_PASS, THROUGH_BALL, SWITCH, CARRY, DRIBBLE, CROSS, SHOOT, RECYCLE, PROTECT_POSSESSION` (`football_brain.py:43`) |
| Serialization | JSON `{"arch":[24,32,32,10], "w1":[24×32],"b1", "w2":[32×32],"b2", "w3":[32×10],"b3"}` |
| **Metadata/versioning** | **NONE.** `brains/ST.json` has exactly the 7 keys above. No generation, fitness, seed, sensor-schema, DNA-schema, or training-method field. |
| Inference mode | Temperature-sampled softmax (never argmax) via `_sample_from_probs`; temperature from `_decision_temperature(player, under_pressure, fatigue)` (`brain_integration.py:307–310`) |
| Where it lives | `NeuralDecisionBrain.decide` (`brain_integration.py:252`) |

### B.2 `NeuralDecisionBrain.decide` contract — `brain_integration.py:252`

```python
@staticmethod
def decide(player, x, y, teammates, defenders, position_engine, team_profile,
           under_pressure, attacks_right, game_state, minute=45.0,
           soul=None, record_trace=False) -> PlayerDecision
```

- **Lookup:** registry by exact `player.name` → miss → `_brain_for_position(player.position)`
  auto-loads `BRAIN_DIR/<POSITION>.json` (env `PLOFA_BRAIN_DIR`, default `"brains"`,
  `set_brain_dir()` override), caches per position, registers under the player name →
  miss → `FootballBrain.random()` (silent random fallback).
- **Sensors:** `extract_sensors(...)` (24-d, see §C), `team_possession=True` hard-coded,
  `score_diff ∈ {−1,0,+1}` derived by substring-matching `game_state.name`.
- **Return:** `PlayerDecision(intent, action, confidence, target, reason, perceived_intents, trace)`;
  `action="CARRY"` iff intent ∈ {CARRY, DRIBBLE}, else `"PASS"`. Never returns a shot.
- **Trace:** optional `record_trace=True` yields `{player, fatigue, sensor_vector,
  output_probs, chosen_idx, choice_probability}` — this is the only first-class
  observability hook in the brain (Phase 20 substrate).

### B.3 Sibling brains

| Brain | Shape | Params | Status | Wiring |
|---|---|---|---|---|
| `OffBallBrain` ("conscience") | 24→32→32→1 sigmoid | 1,889 | **ABANDONED** | dead workstream |
| `TeamPressBrain` | 24→32→32→1, `kind="team_press_engagement"` | 1,889 | **LIVE** | shared XI-wide `g`; `match_engine.py:2229` |
| `DefensiveActionBrain` | 24→32→32→4 softmax `[tackle,int,clear,block]`, `kind="defensive_action"` | 1,988 | **LIVE** | 3 sites (contest, clearance, recovery) |

### B.4 DNA ↔ brain ↔ form ↔ tactics separation (current)

| Layer | Current role | Leak |
|---|---|---|
| **DNA** (`player_dna.py`) | (a) seeds initial weights, (b) enters sensors `[21..23]` plus fatigue estimate `[17]`, (c) is re-applied *after* evolution only at init (evolved weights replace it), (d) drives `_carry_speed`, `_carry_distance_advance`, control radii | After evolution, DNA's only influence is via the 3 sensor slots (`vision/composure/decisions`) + fatigue proxy. DNA no longer caps the brain's *policy* — a slow CB can carry exactly like a fast winger if the weights say so. |
| **Form** | `PlayerFormState` confidence/fatigue → temperature + carry gate threshold; **not a sensor** | Fatigue is an *estimate* (`_fatigue_estimate`), not the live stamina state |
| **Tactics** | `TeamProfile`/`EffectiveTactics` reshape geometry and upstream gates, not the brain input | No explicit tactical context vector in the 24-d sensors |
| **Manager** | influences posture/tempo through `TacticalAI` | **never reaches the brain directly** |

### B.5 Training / evolution

- **No gradient training anywhere.** Learning = genetic algorithm over weights.
- GA lives in `brain_evolution.py`: `evolve()` — population 32, elitism 20%, crossover from
  top 50%, mutation `rate = 0.15 · 0.97^gen` (floor 0.05, strength 0.3), fitness sharing
  (σ=10, L2), RNG seeded per brain `seed + gen·1000 + i`.
- Fitness = `synthetic_fitness()`: mean reward of **argmax** intent over synthetic states,
  weighted by `(0.5 + 0.5·mean_conf)`, minus penalties, scaled by behavioral-entropy bonus
  `(1+bdiv)/1.5`, clamped `[0,1]`.
- Two reward sources: `_context_reward` (hand table) or a `FitnessSurrogate` (see §D).
- Reproducibility: preserved *only* if the sensor corpus RNG sequence is preserved — the
  in-flight refactor is explicitly written to do so, but this should be tested.

---

## C. SENSOR AUDIT — the 24-d vector (`brain_sensors.py`)

All values are clamped to ~`[0,1]`; `x∈[0,105]`, `y∈[0,68]` metres (0=own goal line for
`attacks_right`, full pitch length 105, width 68).

| # | Name | Meaning / normalisation | Observable? | Leak? | Dup? | Future? | Used? |
|---|---|---|---|---|---|---|---|
| 0 | `ball_x` | carrier x / 105 | yes | no | **= #2** | no | yes |
| 1 | `ball_y` | carrier y / 68 | yes | no | **= #3** | no | yes |
| 2 | `player_x` | = #0 (carrier holds the ball) | yes | no | **dup** | no | yes |
| 3 | `player_y` | = #1 | yes | no | **dup** | no | yes |
| 4 | `nearest_defender_dist` | min non-GK def distance / 15 | **omniscient** | no | — | no | yes |
| 5 | `defenders_within_5m` | count / 5 | **omniscient** | no | — | no | yes |
| 6 | `defenders_within_10m` | count / 5 | **omniscient** | no | — | no | yes |
| 7 | `best_forward_dist` | best forward teammate progress / 40 | **omniscient** | no | — | no | yes |
| 8 | `best_forward_openness` | `(nearest_def − 1.5)/8.5` | **omniscient** | no | — | no | yes |
| 9 | `space_ahead` | `(near_def − 1.5)/8.5` | **monotone fn of #4** | no | **near-dup** | no | yes |
| 10 | `is_in_crossing_zone` | wide role AND (x>80 or x<25) | yes | no | static role marker | no | yes |
| 11 | `is_near_goal` | goal distance < 25 m | yes | no | — | no | yes |
| 12 | `is_final_third` | x>70 / x<35 | yes | no | — | no | yes |
| 13 | `is_own_half` | x<52.5 / x>52.5 | yes | no | — | no | yes |
| 14 | `goal_distance_norm` | `hypot(goal − pos)/70` | yes | no | correlated #11 | no | yes |
| 15 | `central_lane` | abs(y−34) < 18 | yes | no | — | no | yes |
| 16 | `under_pressure` | live press flag 0/1 | **event state, not player perception** | no | — | no | yes |
| 17 | `fatigue` | structural estimate from DNA stamina + minute + carry-over; **not live stamina** | estimate | no | — | no | yes |
| 18 | `score_diff` | quantised to {−1,0,+1} from `game_state.name` substring | yes | no | coarser than real GD | no | yes |
| 19 | `minute_norm` | minute / 90 | yes | no | — | no | yes |
| 20 | `team_possession` | hard-coded `True` → always 1.0 | yes | no | **constant** | no | yes (dead signal) |
| 21 | `player_vision` | DNA / 100 | yes | no | static | no | yes |
| 22 | `player_composure` | DNA / 100 | yes | no | static | no | yes |
| 23 | `player_decisions` | DNA / 100 | yes | no | static | no | yes |

**Verdict on the sensor layer:**
- **No futurity / no hidden-state leakage** — every sensor is derived from current geometry
  and the player's own DNA. Good. (This is the one property to protect in V2.)
- **Perfect information.** Sensors #4–#8 read *exact* coordinates of *every* opponent and
  teammate with no range/accuracy/awareness model. A player with 20 vision perceives
  identically to one with 99 (except the 3 static DNA scalars), which do **not** modulate
  perception — they are merely *inputs*.
- **No velocity / no acceleration / no body orientation / no predicted arrival / no
  ball-control difficulty / no closing speed / no marking relationship / no passing lanes /
  no space behind the line / no channels / no pitch-control value / no team phase / no
  tactical instruction / no manager intent / no support-run detection.**
- **Duplicates:** #0=#2, #1=#3, #9≈#4, #14≈#11. Roughly 3–4 of 24 inputs carry no new
  information.
- **Constant input:** #20 is hard-coded `False`-producing in practice; #21–23 are static per
  match.
- **Normalisation is crude:** counts divide by 5 (can exceed 1 → clamped), forward progress
  by 40, nearest defender by 15 — all magic numbers with no documented basis.
- **`extract_offball_sensors`** shares the layout (ball↔runner decoupled, slot 16 = def<4 m);
  feeds the abandoned conscience. `extract_team_sensors` and `extract_defensive_sensors`
  exist in `football_brain.py` for the two live auxiliary brains.

This is the single highest-leverage place to change. The brain is not capacity-limited
(2,186 params saturate easily); it is **perception-limited**.

---

## D. REWARD AUDIT — every human-authored term, classified

This is the core of Phase 6. The full pipeline is: real match → snapshot
`(sensors, chosen intent)` → match to next same-player execution event → `_event_success`
weight → bucket table → surrogate → GA fitness. **Every stage is hand-authored.**

### D.1 On-ball objective

| Term | Where | Value / form | Classification |
|---|---|---|---|
| `POSITION_REWARDS` per-position profiles | `brain_evolution.py` | ST SHOOT 1.0, CB SAFE_PASS/RECYCLE 1.0, etc. | **HEURISTIC** (football prior) |
| `_context_reward(intent, pos, sensors)` | `brain_evolution.py` | gates intents by sensor thresholds (near_goal→SHOOT, space→CARRY/DRIBBLE, crossing zone→CROSS, fwd→THROUGH/PROGRESSIVE, def_press=1−s[5]) | **HEURISTIC** (authored situational logic) |
| `_event_success` outcome weights | `surrogate_collect.py` | GOAL `4.0+min(xg·4,1)`; SHOT_ON/SAVE/BLOCKED `1.5+min(xg·4,2)`; SHOT_OFF `0.5+min(xg·4,1)`; completed PASS `2.0+advance+0.5·progressive`; CARRY `1.0+dist+0.5·progressive`; turnover `0.0`; default `0.5` | **LEARNING SIGNAL** (but constants are authored) |
| `FitnessSurrogate.expected_success` table | `brains/surrogate_pos.json` | `{"prior":0.5,"table":{pos:{bucket:{intent:score}}}}`, bucket = 6 boolean thresholds on sensors `[16,12,13,9,15,14]` | **LEARNING SIGNAL** (learned from real matches, hand-bucketed) |
| Global `""` fallback row | `surrogate_collect.py` | pooled across positions when a position row is missing | **STABILIZER / HEURISTIC** |
| Cross-intent borrowing vs neutral-prior fallback | `_lookup` | T1 (production) borrows a neighbouring intent's score; v3 (closed) substitutes neutral 0.5 | **HEURISTIC** — this single switch determines whether the brain keeps vertical intent (see AGENTS.md v3 failure) |
| Off-ball `_score` | `offball_probe.py` | recovery 1.0 / opp turnover 0.9 / shot off 0.6 / shot on 0.4 / goal 0.0 / none 0.5 | **LEARNING SIGNAL** (abandoned workstream) |
| `DefensiveActionSurrogate` | `brains_def/` | 568 real moments (191 natural + 377 forced) | **LEARNING SIGNAL** |
| `TeamPressSurrogate` | `brains_team/` | forced counterfactual sweep sit .619 / med .621 / press .636 | **LEARNING SIGNAL** |

### D.2 Stabilisers (must be separated from the football objective in V2)

| Term | Where | Effect | Classification |
|---|---|---|---|
| Bucket shrinkage (min_samples=3) | `FitnessSurrogate.fit` | pulls thin cells toward bucket mean | STABILIZER |
| Confidence weighting `mean_reward·(0.5+0.5·mean_conf)` | `synthetic_fitness` | rewards committing firmly | STABILIZER |
| Indecision penalty 0.1/state when top-2 gap < 0.02 | `synthetic_fitness` | discourages uniform output | STABILIZER |
| **Dominance penalty** `max(0, max_share−0.5)·1.5` | `synthetic_fitness` | blocks single-intent collapse | STABILIZER (v3) |
| **Effective-usage penalty** `shortfall·0.75` when <6 intents >5% | `synthetic_fitness` | blocks two-intent collapse | STABILIZER (v3) |
| Behavioral-entropy bonus `(1+bdiv)/1.5` | `synthetic_fitness` | rewards context-dependent spread | STABILIZER (v1 retained) |
| Fitness sharing (σ=10, L2 weight space) | `evolve()` | niching | STABILIZER |
| Evidence gates / `pop_diversity` arg | `evolve()` | `pop_diversity` is a **dead argument** in the refactor | DEAD |
| Promotion rule `Δ>+0.005 AND goal_diff≥0` | `brain_self_trainer.py` | accept/reject challenger | STABILIZER |
| `extract_team_fitness = 0.45·attack + 0.25·control + 0.30·decision` | `match_probe.py` | **the real-match arbiter**; attack=`min(1,(goals·0.6+xg·0.4)/5)`; decision quality sourced **only from CARRY-event metadata** (`active_brain` attached at `event_chain.py:1714`) | **HEURISTIC** (authored composite) |
| `verify_brains` static thresholds (55% / 75% / 3%) | `verify_brains.py` | static diversity gate | HEURISTIC — **must never be reported as a match outcome** (see AGENTS.md striker correction) |

### D.3 The critical finding

> **The on-ball brain has never been optimised against the actual match outcome.**
> It is optimised against a *surrogate table* learned from ~6 real matches (`v1`) whose
> cell scores were themselves computed from a hand-authored `_event_success` formula over
> the *heuristic* policy's behaviour. The real match is only ever used as a final **gate**
> (`validate_neural_xl` / `compare_striker` / trainer), never as the gradient/selection signal.
> This is imitation of a heuristic, filtered through a hand-bucketed table, then gated on a
> handful of matches.

The AGENTS.md history is the empirical proof: T2/V3 — richer data, penalties, and 4× the
generations — produced *better surrogate fitness* and *worse football* (T2G160: 0.6523
fitness but 19–23 goals; T1: 0.6208 and 18–18). The objective, not the budget or the data
volume, is the ceiling. Any V2 must make **actual simulated consequences** the objective.

---

## E. DECISION-SYSTEM AUDIT — where the neural brain is overridden/supplemented

The neural brain is a **minority voter** in its own chain. Ordered by the touch loop:

| # | Override / supplement | Location | Nature |
|---|---|---|---|
| 1 | Regression machine forces receiver, blocks carries | `event_chain.py:1360–1465` | deterministic structural order |
| 2 | **AttackingMatrix SHOOT gate** fires shot + `break` before neural call | `:1466–1528` | deterministic; the neural SHOOT intent is effectively **discarded** |
| 3 | SHOT-beats-REGRESSION re-check | `:1388–1438` | deterministic |
| 4 | Matrix forced passes; winger balls capped ≤20 m | `:1529–1548` | deterministic |
| 5 | Wide-combo override replaces receiver (LW/RW/LB/RB) | `:1549–1573, 1754–1772` | role override |
| 6 | GK distribution pass | `:1576` | role override |
| 7 | **Carry gate** `roll < max(0.55, confidence)` (+ pressure/can_carry/regression/GK) | `:1637–1639` | heuristic gate on neural CARRY |
| 8 | Winger drive/cut forces carry | `:1616–1636` | behaviour engine override |
| 9 | Brain target used only for PROGRESSIVE_PASS/SWITCH with conf>0.35; else `_pick_receiver` | `:1800–1807` | heuristic receiver selection |
| 10 | ATTACKING_PROPHET scenario replaces receiver | `:1828–1845` | scripted pattern |
| 11 | Team press: `prob = _PRESS_PROB[pos] * g` | `match_engine.py:2229` | brain *multiplies* a fixed table; never disables pressing |
| 12 | Defensive action feasibility grip (clear/block refused far from own goal); heuristic `_danger_scaled_action_weights` table is the no-brain fallback | `match_engine.py:150–153, 260–261, 3626/3672/3925` | physics constraint + heuristic fallback |
| 13 | `DecisionBrain.decide` pinned as the heuristic baseline by harnesses | `match_probe._pin_heuristic`, `validate_neural_xl._run_heuristic` | test-only |

**Implication:** to make the brain actually *decide*, Phase 5's separation must be
respected — the deterministic layers should become **action-resolution physics** (what
happens) rather than **intent vetoes** (what is allowed to be tried). The shot gate in
particular should become a shot-quality/payoff model the brain can choose to consult, not
a pre-emptive `break`.

---

## F. POSSESSIONEPISODE AUDIT — what the simulator already knows (and discards)

### F.1 What exists

- **`PossessionEpisode`** (`possession_physics.py:413`): `resolve_pass` :794,
  `resolve_ground_pass` :777, `resolve_loose_ball` :647, aerial resolution, dribble
  resolution, 10 Hz races via `geometry_engine.MovingPlayer`/`BallFlight`,
  `resolve_shot` :1187 (GK dive geometry, magnus, rolling friction).
- **`ChainResult`** (`event_chain.py:215`): per-possession record with
  `goal_scored`, `goal_scorer`, `goal_assistant`, `xg_generated` :227, `corner_won`,
  `penalty_won`, `restart_required`, `delayed_offside`, `offside_detected` +
  location/player/team :251–255, and `add(event)` :286 holding the full ordered
  `MatchEvent` list.
- **Per-touch decision metadata** stamped at `event_chain.py:1715–1720`:
  `action, intent, confidence, reason, decision_quality, is_error`.
- **`MatchEvent`** carries `location_x/y`, `xg`, `xa`, `body_part`, `is_shot` :481,
  `distance_from_goal` :500.
- **Post-hoc analytics**: `SequenceTracker` segments possessions into sequences
  (`build_up_attacks`, `direct_attacks`, `shot_ending_sequences`);
  `ChanceCreationLedger` (`chance_creation.py:164`) derives shot assists / goal assists /
  xA / big chances backward from the timeline.
  > **Correction, 2026-10-02 — THE LEDGER IS NOT THE ASSIST SOURCE.** Read this
  > before assuming the ledger governs assists. It does not:
  > `exporter.py`'s Goals sheet and `alltime_db` ASSISTS read the **engine's own**
  > `goal_assistant` / GOAL-event `secondary_player`, not `ChanceCreationLedger`.
  > The ledger is used for xA / chance-creation aggregates. Historically both
  > paths existed side by side and the ledger's own docstring claimed the engine
  > "used to" fabricate chances — which is not evidence that it stopped, and was
  > not true of the assist field at all.
  >
  > As of 2026-10-02 the engine path is causal and the two agree: the shooter is
  > the man `PossessionChain` last had the ball, and the assist is the previous
  > value of the carrier at the carrier change, with **no fallback**. A goal with
  > nobody passing to the scorer is genuinely UNASSISTED and reads as `""`.
  > Expect fewer assists than before; that is the honest answer, not a regression.
  >
  > This was the third instance in one session of the same trap — **a fixed field
  > is not a fixed feature.** After changing a field, follow it to the consumer
  > before claiming the feature works. The instances: the `CHANCE_CREATED` origin
  > (fixed, then the invented `record_touch` that propagated it downstream was
  > still shipping); `result.goal_assistant` (fixed, then the GOAL event's own
  > `secondary_player` — the column the Goals sheet actually reads — was still
  > `creator.name if creator else None`); and `world.ingest`'s `sub_controller`.

### F.2 What is discarded (the credit-assignment opportunity)

1. **No post-chain value target.** Each `ChainResult` is a self-contained episode;
   `_absorb_chain` appends events and accumulates xG but never writes a "value of this
   decision" back to the touch that caused it. There is no n-step return, no discount, no
   eligibility trace, no advantage estimate anywhere in the codebase.
2. **The decision metadata is consumed by exactly one thing:** the real-match decision
   quality in `extract_team_fitness`, and (per audit) only for **CARRY** events. The
   `intent`/`confidence`/`decision_quality` of passes, through-balls, crosses etc. is
   written to the timeline and then ignored by learning.
3. **The surrogate was collected once** by patching `NeuralDecisionBrain.decide` to snapshot
   `(sensors, intent)` and correlating with the next execution event. It is not a live
   learning loop.
4. **Ball path / player path** are recorded for the whole match (`full_match_ball_path`,
   `full_match_player_path`) and exported — but never fed to learning.
5. **xG is attributed to `TransitionChain`** with a special fix at `match_engine.py:4419–4441`
   — evidence that consequence attribution is subtle and currently patch-level.

### F.3 The substrate is sufficient

Everything needed for real credit assignment already exists: **the chain is an episode,
each touch is an action with a recorded intent, and the terminal outcome (goal / shot /
xG / turnover / progression / location) is in `ChainResult`.** What is missing is only the
mechanism that turns that into a learning signal. This is the strongest argument for the
Phase 7–11 design.

---

## G. CROSS-CUTTING PHASE-0 FINDINGS

### G.1 Duplicated / competing systems

1. **Three match entry points** feeding one `MatchEngine` (`auto_run_match.py` official,
   `run_match.py` scratch, `season_manager.py` batch).
2. **Three availability implementations**: `SeasonState.is_available`, `AvailabilityChecker`,
   `roster_loader._filter_eligible`.
3. **Two fixture sources**: Excel fixtures sheet vs `season_manager` circle-method generator.
4. **Two soul-player dictionaries**: `auto_run_match.py:123` and `run_match.py:43`.
5. **Three style layers**: `TEAM_CATALOG` overrides vs `auto_team_style()` vs
   `ManagerProfile.style_bias`.
6. **Two evolution modules**: `brain_evolution.py` (live) and `brain_evolution_old.py`
   (untracked, superseded — plus root scratch `test_baseline*.py` import it).

### G.2 Dead / unused code

- `OffBallBrain` "conscience" workstream (abandoned; `offball_probe`, `offball_evolution`,
  `evolve_offball`).
- `position_engine.drift_minute` + `_wide_run_step` / `_midfield_run_step` /
  `_striker_run_step` — **standalone-simulator/tests only**; live off-ball is
  `MatchEngine._offball_move_player`.
- `striker_behavior.decide_run` / `midfielder_behavior.decide_run` — only fire in the dead
  run steps.
- `active_play_brain.py` — keeps only a compat import.
- `virtual_gps.py` off by default; `pitch_replay.py` standalone; `weather_physics` disabled
  by default (`enabled=False`, all multipliers 1.0).
- `brains_retrain2/`, `brains_retrain2_g40/g80/g160/`, `brains_hybrid/`,
  `surrogate_pos_v2/v3.json` — reference-only.
- ~45 scratch test/debug files in `tests/` (no test functions: `check_*`, `dbg_*`, `diag_*`,
  `verify_*`, `fix_proposal.py`, `_*.py`) plus root scratch.

### G.3 Hard-coded football assumptions & magic numbers

- Pitch constants 105×68, goal y=34, own-goal line 0/105.
- `_PRESS_PROB` per-position table; team-press band thresholds 0.60/0.20.
- `shot_threshold` with **0.70 take-probability floor**; `_xg_quality_mult` scoreline
  governor 0.80–1.20; `POSSESSION_CARRY_PROB=0.90`.
- Offside call ramp `OFFSIDE_CALL_FLOOR=0.02 → OFFSIDE_CALL_PEAK=0.06`.
- SPOT/penalty conversion `0.60 + penalty_taking·0.30 − gk_reflex·0.05` clamped [0.55,0.92].
- Corner xG fixed 0.79; `_context_reward` sensor thresholds; forward progress /35, /40;
  defender counts /5; nearest-defender /15.
- `extract_team_fitness` weights (0.45/0.25/0.30) and `min(1,(goals·0.6+xg·0.4)/5)`.
- Trainer promotion thresholds (Δ>0.005, goal_diff≥0).

### G.4 Randomness substituting for simulation

| Site | Location | What is rolled |
|---|---|---|
| Event clock | `event_chain.py:371` `second = random.randint(0,59)` | no continuous clock; each event gets a random second within its minute |
| Goal outcome | `match_engine.py:1325` `XGEngine.does_goal_happen` | single Bernoulli on xG (capped 0.99); no post-shot model |
| xG jitter | `match_engine.py:1318` | ±5% random |
| Offside flag | `event_chain.py:4290–4366` | Bernoulli ramp |
| Through-ball success & jitter | `event_chain.py:4392–4408` | Bernoulli + uniform destination noise |
| Dribble / 1v1 | `event_chain.py:~2809–2828` | attempt roll + outcome |
| Miscontrol / loose ball | `event_chain.py:~2383`, `possession_physics.py:647` | Bernoulli + recovery roll |
| Corner side/zone | `event_chain.py:5957`, weighted-pick :717/:729 | random |
| Press commit | `match_engine.py:1996` | Bernoulli on `_PRESS_PROB[pos]·g` |
| Penalty | `event_chain.py:5906,5913` | Bernoulli + foot choice |

Note the *good* counterexamples: `resolve_shot`, `_gk_shot_motion`, `_race_motion`,
`PitchControlField.compute`, and the kinematic flight models are deterministic physics that
replace dice. V2 should expand that regime, not shrink it.

### G.5 Bottlenecks

1. **`synthetic_fitness` is per-state Python** and the forward pass is unbatched even after
   the "batch" refactor. Evolution is ~110–155 s/position × 11 positions; T2G160 took 29 min.
2. **Full match ≈ 19–23 s**; a 7-match validation run is minutes. The real-match gate is the
   bottleneck for outer-loop learning.
3. `_offball_move_player` runs a 10 Hz Python-wide shape integration for 22 players.
4. `PitchControlField.compute` is cached per minute/snapshot key — good, but a shared
   dependency for several modules.
5. The dummy-`PositionEngine` sensor construction inside `evolve()` (dict of names) dominates
   surrogate-guided fitness cost.

### G.6 Discarded causal information (summary)

See §F.2. In one line: **the engine computes a rich per-possession causal record and throws
away everything except a match-level xG total.**

### G.7 Where learned behaviour is trained against human-authored labels/rewards

Everywhere on the on-ball path: `POSITION_REWARDS`, `_context_reward`, `_event_success`
constants, the surrogate bucket thresholds and cell shrinkage, the neutral prior, the
dominance/usage penalties, and the `extract_team_fitness` composite. There is **no** learned
value function, **no** consequence-derived return, and **no** temporal credit assignment.

---

## H. RECOMMENDED ARCHITECTURE — PLOFA V2 (design only, not built)

### H.1 Target data flow

```
WORLD STATE (engine owns exact truth: positions, velocity, ball, DNA, tactics, phase)
      │
      ▼
OBSERVATION  (exact state for the engine; NO training signal)
      │
      ▼
PERCEPTION  (attribute-dependent, imperfect, role-specific, range/accuracy/noise model)
      │        • vision/anticipation/positioning/composure set range, precision, FOV, refresh
      │        • "I think there is space behind the fullback" — distributions, not coordinates
      ▼
INTENT      (FootballBrain V2 outputs an intent distribution; unchanged shape contract)
      │
      ▼
ACTION      (engine resolves physics: pass/carry/shot/tackle via PossessionEpisode)
      │
      ▼
CONSEQUENCE (ChainResult: terminal outcome + progression + xG + turnover location + who)
      │
      ▼
VALUE       (learned V(state); per-touch return V_after − V_before + terminal bonus)
      │
      ▼
LEARNING    (GA selection on episode return; V trained on engine rollouts)
```

### H.2 Component design

1. **Perception layer (`perception.py`, new).** Insert between `extract_sensors` and the
   brain. Produces a *perceived world* from the exact one:
   - **Range/accuracy:** each observable is sampled from a noise model whose σ shrinks with
     vision/anticipation/positioning; objects beyond a vision-scaled radius are unseen.
   - **Relevance filtering:** only the top-k most decision-relevant teammates/opponents are
     represented; k scales with vision/composure.
   - **Structured, not raw:** teammate *openness*, *lane availability*, *run-behind flag*;
     opponent *closing speed*, *marking relationship*; space *behind line*, *channels*,
     *pitch-control delta* — each derived from engine truth but degraded.
   - **No futurity rule enforced by test** (see §I).
2. **Sensor v2 schema.** Keep the 24-d v1 vector loadable (versioned), add a role-specific
   perception block. Suggested shared blocks: SELF (velocity, accel, orientation, live
   stamina, composure), BALL (velocity, height, predicted arrival, control difficulty),
   TEAM/OPP (perceived per-actor vectors), SPACE (a small grid of free-space / pitch-control
   / lane-risk values), CONTEXT (exact GD, match phase, tactical directive, manager intent).
   Do **not** jump to ~100 raw inputs; prefer ~40–60 structured, interpretable features.
3. **Value model (`value_model.py`, new).** `V(state)` learned (numpy; e.g. linear/GBM-free
   small MLP or even a tabular model over a discretised state) from real matches and/or fast
   rollouts. Reward = `γ·V(s') − V(s) + terminal`, with terminal = goal / shot / xG created /
   progression / turnover location. This directly implements Phases 7–8, 11.
4. **Objective vs stabiliser separation.**
   - *FOOTBALL OBJECTIVE:* episode return (consequences only).
   - *TRAINING STABILISERS:* diversity/role/exploration as **hard constraints or rejection
     sampling**, never as additions to fitness. Keep behavioral entropy as a *population*
     constraint, not a per-brain reward.
5. **Surrogate repositioning.** Keep `FitnessSurrogate` as a warm-start/curriculum and a
   diagnostic, but require the on-ball champion to pass a real-match gate before promotion.
   Treat the v1 surrogate explicitly as **heuristic-imitation data**.
6. **Brain versioning.** Add metadata to every serialized brain:
   `{arch_version, sensor_schema, normalization_version, dna_schema, training_method,
   generation, fitness, seed, created}`; loaders refuse incompatible schemas loudly.
   Prefer a v2 file (e.g. `brains_v2/`) and keep v1 loading for `brains/`.
7. **Observability.** Generalise `record_trace` into a decision journal: world state,
   perceived state, sensor values, DNA, tactics, probs, alternatives, chosen intent, result,
   immediate value, future value, final consequence. Add a `why(player, minute)` query over
   the journal.
8. **Roles & tactics.** Role-specific perception *blocks* (CB/CM/winger/ST/GK) selected by
   position; manager instructions injected as **context features**, so the brain can obey or
   deviate. No action is hard-coded by role.
9. **Anti-collapse as constraints.** Minimum behavioural diversity, role-consistency bounds,
   action-distribution bounds — enforced at the population level.

### H.3 Explicit non-goals

- **Do not replace the event/physics architecture.** `PossessionEpisode`, pass/shot
  resolution, geometry, pitch control, marking, pressing, DNA, form, fatigue, injuries,
  weather, match state and event chains are assets and stay.
- **Do not enlarge the network** until an experiment proves capacity is the limit. Current
  24→32→32→10 already saturates its objective.
- **Do not let a learned model resolve physics.** The brain chooses intent; the engine
  decides outcomes.

---

## I. MIGRATION PLAN (incremental, baseline-preserving)

**Invariant: `auto_run_match.py` and season persistence are never touched by V2 work.**
All changes sit behind flags (env vars / config) that default to current behaviour, and the
existing `validate_neural_xl.py` real-match gate is the acceptance test at every step.

0. **Freeze the baseline.**
   - Commit or stash the in-flight batched-sensors refactor.
   - Restore `brains/GK.json` + `brains/_manifest.json` to the T1 set (from
     `brains_backup_incumbent_20260911_g160swap/`).
   - Tag the committed checkpoint as `plofa-v1-baseline`; record the 7-match, seed-21 gate
     (`neural_fitness 0.6208 vs 0.5254, goals 18–18`) as the comparison anchor.
   - Get `tests/test_football_brain.py` green (update the changed-API test).

1. **Add perception behind a flag (no behaviour change).**
   - New `perception.py`; `PerceptionConfig(enabled=False)` returns the identity (v1 sensors).
   - Thread through `NeuralDecisionBrain.decide` → `extract_sensors`, preserving the exact
     array for v1 brains.
   - Tests: identity-mode is byte-identical to v1; range/FOV/no-futurity unit tests.

2. **Brain versioning + schema registry (no behaviour change).**
   - Add metadata to new files; loader accepts v1 (no metadata ⇒ v1) and rejects mismatches.
   - `brains_v2/` scratch namespace to avoid production writes.

3. **Value model + consequence logging (offline, no policy change).**
   - Instrument `ChainResult` → per-touch `(s, a, s', terminal)` records from real matches.
   - Fit `V`; measure correlation with terminal outcomes; report as a diagnostic. Do **not**
     change the objective yet.
   - Experiment: does `V_after − V_before` predict the next chain outcome better than the v1
     surrogate's `expected_success`? This is the kill/continue gate.

4. **Switch the on-ball objective to consequence-derived returns (the one risky step).**
   - GA selection on episode return + `V` bootstrap; stabilisers moved to constraints.
   - Evolve to `brains_v2/`; gate challenger vs incumbent with `validate_neural_xl` at
     identical seeds (same protocol as `brain_self_trainer`). Promote only on a real win
     (fitness up **and** goals/GD not worse) — mirroring the existing trainer rule.

5. **Perception richness + role blocks.**
   - Enable the imperfect perception model; add role-specific blocks; re-run the same gate.
   - Only increase capacity here if experiments show the network is capacity-limited.

6. **Tactics-as-context.**
   - Inject `EffectiveTactics`/`ManagerProfile` as context features; verify players obey and
     occasionally deviate without hard-coding.

7. **Observability + realism validation.**
   - Ship the decision journal + `why()` query.
   - Build the Phase-17 realism diagnostics (goals/shots/possession/pass completion/turnovers/
     xG/sequence length/progressive passes/crosses/through balls/pressing/formation integrity)
     as a **sanity boundary**, not a template.

### Test/experiment discipline (Phases 16, 22)

Every experiment records: seed, brain version, sensor version, population, generations,
training environment, reward/value method, mutation params, evaluation matches, opponent
policy, and results. Permanent regression additions: deterministic seed tests, physics
validity, sensor range, **no-information-leak**, possession conservation, action validity,
statistical sanity, performance benchmarks. No brain change may silently alter xG,
scorelines, timelines, league tables, player stats, or season persistence.

---

## J. PHASE 23 — SUCCESS CRITERIA (restated as the V2 acceptance checklist)

1. Imperfect but meaningful perception. 2. Same situation perceived differently by different
players. 3. DNA caps capability without dictating decisions. 4. Managers influence without
hard-coding. 5. Physics resolves actions. 6. Learning from consequences > authored labels.
7. Unscripted useful behaviours emerge. 8. Realistic mistakes occur. 9. Brains generalise.
10. Stable across matches/seasons. 11. Statistically plausible. 12. Decisions are explainable.
13. Computationally feasible.

---

## K. IMMEDIATE RECOMMENDATIONS (ordered, smallest-first)

1. **Restore T1 `brains/GK.json` + manifest** (working tree currently holds a 2-generation
   smoke brain) and finish/commit the batched-sensors refactor so the baseline stops moving.
2. **Fix `test_football_brain.py`** for the new `synthetic_fitness` signature (and decide the
   `test_pass_matrix` semantics).
3. **Guard the good property:** add a **no-information-leak** test asserting sensors never
   contain future/omniscient event data, so V2 perception cannot regress it.
4. **Prototype step 3 above (value model + consequence logging) offline** — it is read-only
   w.r.t. behaviour and is the decisive experiment for whether consequence-driven learning is
   viable on this engine. If it fails, V2's objective redesign is not justified.
5. Only then touch the objective and evolve `brains_v2/`.

---

*Audit produced from a read-only inspection of the working tree. All line numbers reference
the current (dirty) working copy; verify against a frozen baseline before acting.*

---

## STEP-7 DONE (from audit item 7 "Observability" -- decision journal + \`why(player, minute)\\)

**Delivered 2026-09-14.** Follows the item-6/	actics_context discipline: deterministic, no
\`input()\\, no RNG, assert-gated, and the journal is an append-only **sidecar** that is
byte-identical when disabled.

Files (all in the current dirty working tree; line numbers authoritative):
- \decision_journal.py\ (new, py_compile clean, 270 lines): \
ew_journal()\ factory;
  \DecisionJournal\ with \ecord(entry)\, \why(player, minute, *, tolerance)\,
  \justify\, \since(minute)\, \	o_records()\, \__len__\/\__eq__\/\__iter__\;
  module \journalize_trace(trace, *, world_state, perceived_state, dna, tactics,
  future_value, final_consequence)\. Full item-7 field contract per entry: world state,
  perceived state, sensor values, DNA, tactics, probs, alternatives, chosen intent, result,
  immediate value, future value, final consequence.
- \	ests/test_decision_journal.py\ (new, 6 tests, 6 passed): drives only the real module
  surface (no invented \ppend_from_trace\); \why()\ answers only at the precise
  (player, minute); \since()\/scroll honoured; determinism across equal traces.

**Seams (production byte-identical preserved):** \ecord_trace: bool = False\ default in
\decision_brain.py\ (line 677) and \rain_integration.py\ (line 274); the \if record_trace:\
blocks build the trace but are off by default, so no on-ball behaviour change.

**Realism diagnostics (Phase 17) -- kept as a sanity boundary, NOT a template** (item-7
\`` markdown, line 640): goals/shots/possession/pass completion/turnovers/xG/sequence
length/progressive passes/crosses/through balls/pressing/formation integrity are downstream
**consumers** of the journal's \	o_records()\/\since()\, deterministic, and re-use the
existing sidecars (\season_stats.py\, \_attack_realism_probe.py\) rather than duplicating
them. The journal never invents realism totals; it only answers what was recorded.

Relevant files: \decision_journal.py\, \	ests/test_decision_journal.py\,
\decision_brain.py:677/734\, \rain_integration.py:274/393\, \season_stats.py\,
\_attack_realism_probe.py\, \PLOFA_ARCHITECTURE_AUDIT.md\ (this file).
