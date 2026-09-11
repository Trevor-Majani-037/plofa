# PLOFA 26/27 — Neural Brain Project

## Objective
- Replace the heuristic `DecisionBrain` as the sole on-ball decision layer
  in `event_chain.py` with per-position feed-forward neural networks
  (`football_brain.py`, 24→32→32→10, 2186 params, pure numpy) evolved by a
  genetic algorithm against a learned fitness surrogate.
- Verify the neural XI competes with (ideally beats) the hand-calibrated
  heuristic baseline in real matches.

## Important Details
- USER CHOICE: "replace entirely" — the heuristic `DecisionBrain` is no
  longer called by the match engine. Since 2026-09-09 the call site in
  `event_chain.py` (~line 1599) routes through `NeuralDecisionBrain.decide`.
- No ML frameworks; numpy only. No gradient training — GA evolution.
- One real match ≈ 19–23 s. Synthetic surrogate evolution ≈ 110–155 s per
  position (300 states × 40 gen × 32 pop, 4 workers via `evolve_all_parallel.py`).
- Integration contract: `NeuralDecisionBrain.decide(...)` must return the
  same `PlayerDecision` the old site returned (same args, no `self`).
  Deterministic layers in `event_chain` (AttackingMatrix shot gate,
  TacticalPhase regression orders, wide-combo override) remain authoritative;
  the neural brain NEVER fires a shot itself.
- Auto-load: `brain_integration.NeuralDecisionBrain.decide` looks up the
  registry by exact `player.name`; on a miss it auto-loads
  `BRAIN_DIR/<POSITION>.json` (module `BRAIN_DIR`, default `"brains"`,
  per-position cache, `set_brain_dir()` to override) and registers it under
  the player's name. `FootballBrain.random()` only if no file exists. Every
  player in any match runner gets a trained brain with NO per-run wiring.
- BRAIN FILES: `brains/{GK,CB,LB,RB,CDM,CM,CAM,LW,RW,ST,CF}.json` +
  `brains/_manifest_all.json`. Surrogate: `brains/surrogate_pos.json`.
- Surrogate: `surrogate_collect.py` `FitnessSurrogate` — position-aware
  expected-success table `{position: {bucket: {intent: score}}}` + global ""
  fallback, learned from real matches (6 matches, 3286 samples, 3 opponent
  styles). Bucket from sensors `[16] pressure, [12] final_third, [13]
  own_half, [9] space, [15] central, [14] goal_close`. Outcome weights:
  GOAL 4.0+, SHOT_ON_TARGET/SAVE/BLOCKED 1.5+, SHOT_OFF_TARGET 0.5+,
  completed PASS 2.0+advance, retained CARRY 1.0+dist, turnovers 0.0,
  default 0.5. Threaded through `brain_evolution.synthetic_fitness`
  (`--surrogate` in `evolve_brains.py`), normalised by /1.5.
- HARNESS BUG (FIXED): `match_probe.ab_compare`/`run_neural_validation`
  registered brains under `{POS}{POS}` (e.g. "STST") but players are named
  "ST"/"CM1"/"CM2"/"CB1" — silent random-brain fallback invalidated old
  A/B numbers. Use `validate_neural_xl.py` (register_full_xi, position→brain
  map) for trustworthy comparisons.
- POST-SWAP HARNESS RULES: `event_chain` calls `NeuralDecisionBrain.decide`
  directly, so `DecisionBrain.decide` patching is DEAD. For a genuine
  heuristic baseline, pin `NeuralDecisionBrain.decide = staticmethod(DecisionBrain.decide)`
  (see `match_probe._pin_heuristic`/`validate_neural_xl._run_heuristic`).
- Validation methodology: neural XI and heuristic XI are SEPARATE matches vs
  the same opponent (Probe FC balanced vs Rival FC fluid_counter), not
  head-to-head. `extract_team_fitness` = 0.45 attack + 0.25 control + 0.30
  decision quality.
- Starter name template: GK, CB1, CB2, LB, RB, CDM, CM1, CM2, LW, ST, RW
  (subs SUB1 ST, SUB2 CM, SUB3 CB). Player objects expose `.name`/`.position`.
- Tests: `.venv\Scripts\python.exe test_football_brain.py` (24/24) and
  `test_decision_brain.py` (9/9, incl. a full real scratch match). Now
  28/28 (4 new DefensiveActionBrain/hook tests) + 9/9. Run from repo root
  with `PYTHONPATH=.` — the files live in `tests/`.

## Results So Far
- 7-match full-XI (seed 21): neural_fitness **0.7117** vs heuristic 0.5624;
  neural_goals 21 vs 23; possession 50.5 vs 49.2; neural conceded fewer;
  neural won fitness 6/7. Evolved brains (seed 123): GK .7318, RB .8707,
  LB .8367, CB .8988, CAM .8834, CM .9842, CDM .8315, LW .9450, RW .9499,
  ST .6978, CF .8063.
- Post-swap smoke (2 matches, seed 21): neural_fitness 0.686 vs heuristic
  0.690; neural_possession 52.45 vs 47.65; outscored 6–10 (heuristic more
  clinical in this small sample); bound=11 both neural matches.
- POST-PERMANENCE full-XI (2026-09-10, 7 matches, seed 21, ALL three brains
  active: neural on-ball + permanent team-press + defensive-action):
  neural_fitness **0.6208** vs heuristic 0.5254; goals **18 vs 18**
  (goal_diff 0 — smoke-test scoring lag closed); possession 51.99 vs 52.67;
  neural won fitness 5/7 (m3–m7; only losses are heuristic's 6–0 outlier m2
  and m1). Next-Move item 1 (goals lagging → evolve more) NO LONGER APPLIES.
- FAILED RETRAIN (2026-09-11, ROLLED BACK): surrogate v3 (merged 40-match
  dual-policy collection, 28,265 raw samples) was built to fill CAM/CF
  signature cells, `_lookup`'s cross-intent fallback was replaced with a
  neutral-prior fallback, and dominance + effective-usage penalties were
  added to `synthetic_fitness`. Re-running `evolve_all_parallel.py
  --surrogate brains/surrogate_pos_v3.json --states 300 --generations 40
  --population 32 --workers 4 --seed 123` (9.0 min) produced brains that
  PASS all static diversity thresholds (no single intent >55%, top-2 ≤75%,
  ≥5 intents >3%) but FAILED the real-match gate: `validate_neural_xl.py
  --matches 7 --seed 21` → neural_fitness 0.5451 vs 0.5397, goals 11 vs 21,
  goal_diff −10 (vs 0.6208 / 18–18 benchmark). ROOT CAUSE: penalizing
  collapse pushed each brain to a SECOND collapsed optimum (ST never shoots,
  SHOOT=0.0%; CM abandons through-balls) — surrogate-guided evolution on
  random states optimises rare-scoring-bucket intents out of the argmax.
  `brains/` was RESTORED byte-identical from
  `brains_backup_validated_20260911_084143/` (the 0.6208/18–18 set).
  `brains/surrogate_pos_v3.json` + `.raw.json` are REFERENCE-ONLY — NOT the
  production surrogate. DO NOT retrain the on-ball XI with surrogate v3
  without first fixing how rare scoring situations are represented in the
  fitness objective (situation-sampled states or bucket-weighted rewards).
- TRAINER GATE-GECKO EXPERIMENT (2026-09-11): does the T2 objective just need
  a bigger budget? Re-run v3 surrogate at 80 gens (`evolve_all_parallel.py
  --surrogate brains/surrogate_pos_v3.json --states 300 --generations 80
  --population 32 --workers 4 --seed 123` → `brains_retrain2_g80/`, 16.8 min),
  real-match striker gate vs incumbent at identical seeds (7 matches, seed 21,
  `compare_striker.py brains brains_retrain2_g80`): home goals 17 vs 21,
  ST goals 8 vs 9, ST shots 34 vs 36, possession 51.8 vs 53.2. Trajectory is
  real (G40: 13 goals/ST6 → G80: 17/ST8) — 40 gen was too few for the harder
  penalised landscape — but G80 still loses on every axis. CONCLUSION: budget
  matters but the objective is the ceiling; the old no-penalty + cross-intent
  fallback surrogates embed better football priors. Ranking stays
  **T1 > T2G80 > T2G40**. T2G80 ST is mechanically recovered (34 shots) —
  the residual gap is systemic across all roles, not one position.
- T2G160 FULL GATE (2026-09-11, 7 matches, seed 21, `validate_neural_xl.py`):
  `evolve_all_parallel.py --surrogate brains/surrogate_pos_v3.json --states
  300 --generations 160 --population 32 --workers 4 --seed 123` →
  `brains_retrain2_g160/` (29 min). All 11 positions hit fitness=1.0000 in
  the surrogate objective — SATURATION. Gate vs heuristic: neural_fitness
  **0.6523** vs 0.5373 (best fitness number the project has ever produced);
  neural_goals 19 vs 23 (goal_diff −4); possession 51.7 vs 51.0; neural
  won fitness 5/7. T2G160 was promoted to `brains/` (incumbent backup at
  `brains_backup_incumbent_20260911_g160swap/`). However: **USER RETAINED
  T1 AFTER 3 REAL MATCHES** — T2G160's passing output looked indistinguishable
  from the heuristic pass maps (the diversity penalty + neutral-prior _lookup
  fallback pushed every position toward safe lateral spread), whereas T1's
  old surrogate (no penalties, cross-intent borrowing) produced genuine
  verticality that sounded like football. `brains/` restored to T1 files;
  T2G160 kept as reference-only at `brains_retrain2_g160/`. Final ranking:
  **T1 (0.6208, 18–18, goal_diff 0) is the production brain set**; T2G160
  (0.6523, 19–23, −4) is better on the fitness metric but worse on the
  pitch. T2G80 (0.6451, 17–23, −6) and T2G40 (0.5451, 11–21, −10) are
  reference-only.
- SURROGATE-TO-PITCH LESSON: the v3 surrogate (dominance + effective-usage
  penalties, neutral-prior _lookup fallback) destroyed vertical intent by
  design — penalizing "rare intent dominance" on random states taxes the
  very passes that matter in real football (through-balls, progressive
  carries, wide crosses). The old surrogate (v1, 6 real matches, no
  penalties, cross-intent borrowing) embedded better football priors despite
  simpler data. The objective is the ceiling; more data + more generations
  cannot compensate for an objective that penalizes what the sport needs.
  DO NOT build a v4 surrogate without fixing this fundamental constraint.
- TRAINING-CAUSE MAP (for reference):
  - T1 (`brains/`): surrogate `brains/surrogate_pos.json`, 6 matches ~3286
    samples, one (heuristic) policy, no CAM/CF rows (trained on global ""
    fallback), old `_lookup` cross-intent borrowing, NO penalties,
    **80 gen** × 32 pop × 300 states, seed 123.
  - T2 (`brains_retrain2/`): surrogate `brains/surrogate_pos_v3.json`,
    40 matches 28,265 paired samples, dual-policy, CAM/CF filled,
    neutral-prior _lookup fallback, dominance + effective-usage penalties,
    **40 gen** (T2G40), **80 gen** (T2G80), **160 gen** (T2G160).
  - Per `brains/_manifest_all.json`: surrogate=brains\surrogate_pos.json,
    generations 80, population 32, states 300, seed 123.
- STRIKER DATA-QUALITY CORRECTION: the static-diversity check counted ARGMAX
  on random synthetic states. It is NOT a real-match outcome. T2 ST shows 0.0%
  argmax SHOOT yet still takes ~25% of team shots / ~6–8 goals per 7 matches
  in real gates (shots fire through the deterministic AttackingMatrix gate).
  Never report static argmax shares as "this player never shoots" — pair them
  with a real-match striker audit (`compare_striker.py`).

## Work State
- DONE: neural net + sensors + GA evolution; surrogate build + integration;
  pseudo-evo; evolved real brains; registration-bug fix (`validate_neural_xl.py`).
- DONE: LIVE SWAP — `event_chain.py` routes through `NeuralDecisionBrain`;
  auto-load-on-miss in `brain_integration`; harnesses/collector re-pointed
  to pin the neural entry point for heuristic baselines; suites green.
- DONE: team press controller — `TeamPressBrain` (1889 p), `TeamPressSurrogate`,
  intervention counterfactual sweep (sit .619/med .621/press .636), evolved
  `brains_team/XI.json`, live-drive validation (halved goals conceded, removed
  the loss; suites still 24/24 + 9/9). PERMANENT engine wiring since 2026-09-10:
  `prob = _PRESS_PROB[pos] * g` at match_engine.py:2229, `_TEAM_PRESS_AUTO=True`
  default auto-loads `brains_team/XI.json` in ANY runner. Permanent-path
  validation (6 matches, seed 21, `validate_team_press.py`): GA 8→3 (−5),
  GF 13→15, possession +2.9, wins 5/6 vs 3/6, mean drive-g 0.405.
- DONE: defensive-action controller (2026-09-10) — `DefensiveActionBrain`
  (24→32→32→4, 1988 params) replaces the hand-tuned `_danger_scaled_action_weights`
  table for the WHICH-ACT choice (tackle/interception/clearance/block) at all
  three engine sites (open-play contest ~line 3577, direct clearance ~line 3622,
  `_defensive_recovery` ~line 3878). ONE hook `_def_action_choice` with
  `site=` label + feasibility PHYSICS grip (clear/block refused far from own
  goal — a constraint, not a weight; feasibility is also sensor 19).
  Surrogate `DefensiveActionSurrogate` learned from 568 real-match moments
  (191 natural + 377 forced counterfactuals, `brains_def/samples_*.json`),
  GA in `defensive_action_evolution.py` → `brains_def/ACTION.json`.
  Live validation (6 matches, neural on-ball held constant): GA 5 vs 6,
  shots against 48 vs 65 (−17), tackles 99 vs 19 (wins the ball earlier),
  wins 5/6 vs 4/6. Suites now 28/28 + 9/9.
- DONE: production-roster sanity (2026-09-10) — `production_roster_sanity.py`
  runs a NON-PERSISTENT real-path match (real Excel roster →
  `loader.build_matchday_squad` → `SquadBuilder.build` → `MatchEngine`, real
  souls/stamina/availability glue) and proves every brain engages with ZERO
  wiring and ZERO season writes: 28 on-ball players all auto-loaded TRAINED
  per-position brains (0 random fallbacks), permanent TeamPressBrain engaged
  (live g<1.0), DefensiveActionBrain engaged (`brains_def/ACTION.json`),
  and a before/after hash snapshot shows `season_state.json`,
  `manager_state.json`, `referee_state.json`, `fixtures.json` and
  `plofa_output` untouched. `auto_run_match.py` is never executed — only its
  pure helpers are imported.
- DONE: scoring-situation objective fix (2026-09-11) — `goal_bias` in
  `brain_evolution.random_game_state_scoring`: a fraction of synthetic
  states are drawn from final-third/goal-close/central scoring situations
  (~50% clear-chance share vs ~5–24% for fully-random states across
  ST/CM/RW/CAM). Threaded through `synthetic_fitness`/`evolve`, exposed as
  `--goal-bias` in `evolve_brains.py` and `evolve_all_parallel.py`
  (default 0.0 = pre-fix behaviour). This is the prescribed fix for the v3
  collapse: rare scoring intents (ST SHOOT, winger crosses, CM
  through-balls) now survive the argmax because their buckets are actually
  represented in the fitness objective.
- DONE: post-match self-evolving trainer (2026-09-11) — `brain_self_trainer.py`
  closes the loop COLLECT → APPEND → REFIT → EVOLVE(scratch) → GATE →
  PROMOTE/REJECT. Real neural matches (targeted full-XI, default 6) append
  to a persistent corpus; the surrogate is refit fresh; the challenger XI
  evolves into a SCRATCH dir (`--goal-bias 0.25` by default); then real
  matches gate challenger vs INCUMBENT (challenger-vs-incumbent, not
  neural-vs-heuristic) at identical seeds; promotion happens ONLY if the
  challenger beats incumbent fitness AND holds the goal diff. A rejected
  challenger is kept for inspection and the incumbent is always the
  fallback. Safe by construction: scratch lives under `--work`
  (default `brains_trainer/`), promotion is a copy of 11 files, production
  surrogate files/surrogates (`surrogate_pos*.json`) are never overwritten,
  `auto_run_match.py`/season state are never touched. `--smoke` = fast
  end-to-end; `--no-collect --no-evolve` = re-gate an existing challenger.

## Next Move
1. Neural on-ball goal-scoring gap is CLOSED as of the 2026-09-10 full-XI
   run (goals 18 vs 18, goal_diff 0, fitness 0.6208 vs 0.5254). No further
   `--surrogate` evolution or temperature change required for the on-ball XI;
   re-run `validate_neural_xl.py --matches 7 --seed 21` if brains/permanent
   controllers change again.
2. **BRAIN SET IS FINAL (2026-09-11): T1 is the production set.** All T2
   variants (`brains_retrain2/`, `brains_retrain2_g80/`,
   `brains_retrain2_g160/`, `brains_hybrid/`) are reference-only. The v3
   surrogate path is CLOSED for the on-ball XI — its objectives (dominance +
   effective-usage penalties, neutral-prior fallback) demonstrably destroy
   vertical intent, and re-running it even at saturation produces better
   fitness numbers but worse football. Any future retrain must use the v1
   surrogate's design (no penalties, cross-intent borrowing) or implement
   situation-sampled scoring-bucket states FIRST. Incumbent backup for any
   accidental write: `brains_backup_incumbent_20260911_g160swap/` mirrors T1.
2. Team controller: engine wiring is now PERMANENT — `prob = _PRESS_PROB[pos] * g`
   at match_engine.py:2229, auto-load default in any runner; validated via
   `validate_team_press.py` (GA 8→3, wins 5/6 vs 3/6). Production-roster
   brain engagement proven by `production_roster_sanity.py` (zero-write).
3. Self-evolving trainer: `python brain_self_trainer.py` closes the loop
   COLLECT/APPEND/REFIT/EVOLVE(scratch)/GATE/PROMOTE. Defaults: 6 neural
   matches → append to `brains_trainer/corpus.raw.json` → refit
   `brains_trainer/surrogate_refit.json` → evolve challenger to
   `brains_trainer/challenger/` with `--goal-bias 0.25` → still never
   touches production `surrogate_pos*.json`; gate is 2 real matches
   challenger-vs-incumbent at identical seeds; promote only if challenger
   beats incumbent fitness (+0.005) AND holds the goal diff. Run
   `--smoke` first; on every real run AGENTS.md must be updated with the
   trained-brain gate history (like the validate_neural_xl results).

## Relevant Files
- `football_brain.py` — net, save/load, 10 intent labels; `OffBallBrain`
  (conscience, abandoned) and `TeamPressBrain` (shared XI engagement GATE).
- `brain_sensors.py` — 24-d sensor extraction (+ `extract_offball_sensors`, same
  layout; ball 0-1 / runner 2-3 decoupled. For the OFF-BALL conscience input).
- `offball_probe.py` — collection probe for the off-ball "conscience" (LIVE
  surface). Patches `MatchEngine._offball_move_player` observation-only (reads
  the chase `allow` transition 0→±1; consumes ZERO RNG, no engine writes,
  verified fresh-process score-neutral 5-1==5-1). Records ~13-20k press
  decisions/match; correlates with defensive episode outcome (recovery 1.0,
  opp turnover 0.9, shot off 0.6, shot on 0.4, goal 0.0, none 0.5).
- `brain_integration.py` — `NeuralDecisionBrain.decide`, registry, auto-load
  (`BRAIN_DIR`, `set_brain_dir()`), `_INTENT_BY_INDEX`.
- `team_offball_probe.py` — team-level collector + `TeamPressSurrogate`,
  `_wrap_offball` (0.5 s team dedup), `_correlate_team` (chronograph true
  clock, 20 s/25 m), `_G_CACHE` live controller drive; CLI flags
  `--intervene g` (forced counterfactual) and `--controller brains_team/XI.json`
  (live drive). ZERO engine edits; `_restore()` reverts.
- `team_offball_evolution.py`, `evolve_team_offball.py` — GA for the XI
  press controller; `--fast --seed 123` smoke, default 40 gen × 32 pop × 600
  states → `brains_team/XI.json`.
- `brain_evolution.py`, `evolve_brains.py`, `evolve_all_parallel.py` —
  GA + surrogate synthetic fitness. `random_game_state_scoring` +
  `--goal-bias`: fraction of sampled states drawn from scoring situations
  (the v3-collapse fix); default 0.0 = pre-fix behaviour.
- `brain_self_trainer.py` — post-match self-evolving XI: collect → append
  corpus → refit surrogate → evolve challenger to scratch
  (brains_trainer/) → gate challenger vs INCUMBENT on real matches →
  promote/reject with persistent `trainer_log.jsonl`; never touches
  `auto_run_match.py`, season state, or production surrogate files.
- `surrogate_collect.py` — collector + position-aware `FitnessSurrogate`.
- `match_probe.py` — `_pin_heuristic`/`_restore_neural`, `extract_team_fitness`,
  `_run_single_match`, `ab_compare`, `run_neural_validation`.
- `validate_neural_xl.py` — trustworthy full-XI neural-vs-heuristic harness;
  `_run_neural(brains_dir, seed, home_style, away_style)` is also the
  challenger/incumbent gate runner used by `brain_self_trainer.py`.
- `compare_striker.py` — real-match striker audit: runs two full-XI brain dirs
  at identical seeds (seeding exactly like `validate_neural_xl.main`, i.e.
  `random.seed(seed + m*100)` per match — leave this seeding OUT and the
  comparison is non-deterministic noise) and reports per-player goals/shots
  + ST line. Used for the T1-vs-T2G40-vs-T2G80 striker gates and the
  T1-ST-in-T2-skeleton hybrid. `brains_retrain2/` (T2, 40 gen),
  `brains_retrain2_g80/` (T2 objective, 80 gen), `brains_hybrid/`
  (T1 ST + 10×T2) are all reference-only scratch sets.
- `validate_team_press.py` — team-press controller-vs-baseline harness via the
  PERMANENT auto-load path (same seeds, neural on-ball + defensive held
  constant; `_pin_heuristic`'s press-off re-asserted per arm).
- `production_roster_sanity.py` — NON-PERSISTENT real-roster-path
  verification: real Excel squads → `SquadBuilder` → `MatchEngine`, proves
  trained on-ball/team-press/defensive brains engage with ZERO wiring and
  ZERO season-data writes (before/after hash snapshot). Matches the season
  manager pipeline read-only; never executes `auto_run_match.py`.
- `event_chain.py` — THE swap site (import line 106, call ~1599).
- `decision_brain.py` — heuristic (still the baseline; no longer called).
- `auto_run_match.py` — production runner (run() line 655). OFF-LIMITS:
  persists into `season_state`/`season_stats`; season fixtures cannot replay.
- LIVE OFF-BALL CAVEAT: the discrete run modes in `position_engine`
  (`drift_minute`→`_wide_run_step` byline/cut/box, `_midfield_run_step`
  drop/carry/orbit, `_striker_run_step`) are NOT wired into matches — they
  only run in the standalone simulator/tests. Live off-ball movement is
  `MatchEngine._offball_move_player` (match_engine.py:1904): 10 Hz shape
  integration where the ONLY stochastic decision is the chase-allow Bernoulli
  (`random.random() < _PRESS_PROB[pos]`, line 1996, table at 1826). The
  off-ball conscience must gate THAT, not the dormant run modes.

## Team-Level Press Controller (TeamPressBrain) — LIVE DATA 2026-09-09
- WHY: the individual off-ball "conscience" workstream is ABANDONED (probe
  evidence: one player's press/hold moves the team's defensive outcome by
  only ~0.02; pressing is a UNIT act, a per-player gate is the wrong lever).
- CONCEPT: ONE shared XI brain `TeamPressBrain` (24→32→32→1, 1889 params)
  maps a 24-d TEAM-state vector (ball zone vs OUR goal, danger, score state,
  our/opp unit density around the ball, shape compactness, block depth,
  attack direction, opp forwards near our goal) to an engagement scalar
  g∈[0,1]. Wiring (PERMANENT since 2026-09-10): `prob = _PRESS_PROB[pos] * g`
  at match_engine.py:2229 with per-team-per-tick cache; `_TEAM_PRESS_AUTO=True`
  auto-loads `brains_team/XI.json` for any runner; g=1.0 (controller off) is
  byte-identical to the pre-controller engine. Band thresholds: press ≥0.60,
  med ≥0.20, sit below.
- METHODS: `team_offball_probe.py` (collector + `TeamPressSurrogate`, bucketed
  `Z{zone}D{danger}O{ours}E{opp}C{compact}G{goalmouth}`, sample dedup 0.5 s
  per team, decision-local outcome attribution via chronograph true clock,
  20 s window / 25 m proximity, `_score` from offball_probe: recovery 1.0,
  opp turnover 0.9, shot off 0.6, shot on/save/blocked 0.4, goal 0.0,
  nothing 0.5). Evolution via `team_offball_evolution.py` +
  `evolve_team_offball.py` → `brains_team/XI.json`.
- COUNTERFACTUAL DATA: heuristic unit NEVER engages collectively (band
  distribution sit 11996/med 1495/press 5 over 6 matches; mean realised
  effort 0.065). The surrogate therefore CANNOT learn high-engagement value
  from observation alone → REQUIRED a controlled intervention experiment:
  `--intervene 0.15,0.5,0.9` forces per-match g by PRE-SETTING `cst['allow']`
  in the probe wrapper before `_offball_move_player` runs (original's
  Bernoulli skipped, RNG consumption identical → deterministic per seed,
  ZERO engine edits, fully reverted by `_restore()`). 6 matches × 3 levels:
  sit 0.619 / med 0.621 / press 0.636 — monotone, unit-press pays.
- LIVE-DRIVE VALIDATION: `--controller brains_team/XI.json` computes the
  brain's g once per tick per team (`_G_CACHE`) and injects it into every
  near-ball defender's Bernoulli — real match behaviour with the heuristic
  on-ball pinned identically in both arms. Result (6 matches, same opponents
  & seeds): controller scored 20–4 and won 6/6; baseline 26–10 with a 9–1
  and a 0–1; controller HALVED goals conceded (10→4) and removed the loss,
  GD identical +16. Mean controller effort 0.650 vs baseline 0.065. Band
  gradient (controller live): press 0.621 / med 0.636. Estimated on
  8.5k intervention moments: 137 bins / 2779 filled cells; 7.3k controller
  moments.
- LEAST-INTEGRITY RULES: `_restore()` fully reverts the patch; the controller
  never touches `auto_run_match.py`; on-ball neural wiring is untouched.
  Engine wiring is PERMANENT (`validate_team_press.py` = comparison harness;
  `_pin_heuristic`→`set_team_press_auto(False)` builds the heuristic arm).

- `test_football_brain.py`, `test_decision_brain.py` — regression suites.