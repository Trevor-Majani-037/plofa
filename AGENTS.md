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

## POSITIONAL PLAY (2026-09-29) — rest defence + live striker runs
A book-to-code audit against the positional-play model (numerical /
qualitative / positional / **socio-affective** superiority; play from the back;
La Salida Lavolpiana; rest defence; run-timing). Already present and often
under a different name: the four-phase possession machine
(`REGROUP_BUILD_UP → MIDFIELD_CIRCULATION → WING_ISOLATION → BOX_PENETRATION`),
keeper-as-eleventh release valve, free-man/third-man, `backline_spread_pressure`
(CB split into ball-side drop-in + deep/wide pair = the book verbatim), GK
sweeper socket, fullback overlap/underlap/**invert**, inverted-winger cut,
half-space coverage, `TRIANGLE SUPPORT RULE`, winger 1v1 isolation, La Pausa as
"invite the press", false-nine, `squad_chemistry.py` (socio-affective),
`perception.py` (scanning/FOV). Two gaps were closed.

- DONE: **REST DEFENCE** — the last superiority type with no representation, and
  the only one that is a CONSTRAINT rather than a preference, so it is
  implemented as one. Invariant: WHILE IN POSSESSION, ≥`REST_DEFENCE_MIN_BEHIND`
  (2) outfield players remain BEHIND the ball. Deliberately a BACKSTOP, not a
  force: in any normal shape the CBs and pivot are behind the ball and it
  returns None, so it costs nothing; it binds only when the shape has broken.
  `PositionEngine.rest_defence_violator()` (the TEAM decision, resolved ONCE PER
  TICK — positions move as the tick integrates, so a per-player resolve could
  pick a different victim mid-tick) names the SHALLOWEST man ahead of the ball,
  and `rest_defence_clamp()` pins him 12 m behind it. Depth only; y is left
  alone, because re-stamping width would fight CK35. Applied LAST in the target
  chain (CK37/CK38/triangle all steer the same depth axis) and as a clamp, not a
  blend, so no preference can outvote an invariant. Suppressed in a low block
  (the anchor substitution already covers it; the block wins). Kill switch:
  `PositionEngine.REST_DEFENCE_ENABLED`.
- DONE: **STRIKER RUNS ARE NOW LIVE.** `striker_behavior.py` was fully written
  and UNREACHABLE from a match: its only consumer, `_striker_run_step`, is called
  from `drift_minute`, and `MatchEngine` never calls `drift_minute`. So all three
  striker principles the book is most specific about (run in behind, drop to
  link, attack a post channel) had never affected a single match; the striker's
  only live contribution was the false-nine RECEIVE bonus in `attacking_matrix`
  — a pass-value term, not movement. Now wired as a TARGET steer
  (`PositionEngine.striker_run_targets` + `MatchEngine._striker_runs`), pace-
  capped by the 10 Hz integrator, cached per (team, minute, ball-zone).
- TWO LATENT BUGS surfaced by wiring it live, both about the offside line, both
  invisible while the module was dead: (1) the line was the DEEPEST defender, not
  the SECOND-deepest — measured against a sweeping CB, a striker with a 15 m
  channel behind read as five metres BEHIND the line and refused the run, i.e.
  the run was suppressed exactly when the space existed; (2) the axis maths
  saturated `last_line_gap` at 1.0 for an entire half attacking LEFT
  (`abs(x + 78)`), so in-behind was always allowed. Both fixed in
  `StrikerSpatialProfile.offside_line_nx` with a direction-normalised frame.
  `run_behind_target` now sites the run off the LINE (+2 m lead) instead of a
  fixed x=92, which was 14 m beyond a defence on 78 m (a permanent offside
  position) and needlessly short against a high line.
- **NO GLOBAL RNG CONSUMED.** The run decision needs a coin flip, and
  `random.random()` is the wrong coin: the project's own top open bug is that a
  match is not reproducible from `random.seed`, and a new consumer makes that
  worse. `PositionEngine._deterministic_rng(team, striker, key)` is a crc32
  counter stream (crc32 not `hash()` — builtin string hashing is PYTHONHASHSEED-
  salted, the same rule `world/squads.py` and `world/proof.py` follow). Asserted
  directly: an identical call sequence leaves the global stream in the identical
  position. Decision cadence is per (minute, ball-ZONE), not per tick and not
  per minute: per tick flickers ten times a second, per minute leaves a striker
  running in behind after the ball turned over at his feet.
- DEAD CODE REMOVED: `event_chain.py` imported `MidfielderBehaviorEngine` and
  never used it (exactly one occurrence in the file, the import itself — verified
  on raw bytes). It sat in the same import block as the behaviour engines and
  implied midfielder runs were live, which they are not. Removed.
- DONE (2026-09-29, second pass): **WIDE / MIDFIELD / CAM RUN MODES ARE NOW
  LIVE.** The audit above found the movement half of positional play was missing
  entirely, and the same `drift_minute` hole that hid the striker hid three more
  run-modes: winger byline/cut/box, **fullback overlap/underlap/tuck** (they
  lived in the same method, so the fullback advance modes were dead too — the
  book is explicit that full-backs are critical to pressing traps), CM
  drop/carry/late/orbit, and the #10 pocket roam. `attacking_matrix` was already
  PREFERRING to pass into half-spaces while nobody occupied them.
  Refactored rather than duplicated: each of `_wide_run_step`,
  `_midfield_run_step`, `_cam_pocket_roam` was split into a decision helper
  (`_wide_run_targets` / `_midfield_run_targets` / `_cam_pocket_targets`,
  returning `{name: (mode, tx, ty)}` and NO writes) plus the original write loop,
  which now delegates. Zero duplicated decision logic; the dormant path's
  behaviour is unchanged (verified file-by-file against a pre-refactor backup).
  Merged live by `PositionEngine.offball_run_targets` + `MatchEngine._striker_runs`
  on the same (team, minute, ball-zone) cache key as the striker layer.
- **PREFERENCE YIELDS, CONSTRAINT DOES NOT.** The run blend sits BEFORE
  CK35/CK36 in the target chain, the opposite of rest defence. Blending it after
  them silently disabled two older, tuned, test-covered shape rules — the
  triangle suite caught it on the first attempt ("Near CM receded from the wide
  cluster: 44.0 -> 43.9"). A run target is only a preference and must lose to a
  measured rule; a rest-defence clamp is an invariant and must not.
- **NO GLOBAL RNG, SECOND MECHANISM.** Unlike the striker, the winger / fullback
  / midfield engines call `random.random()` internally. Rather than thread an
  `rng` through three more modules, `offball_run_targets` runs the whole decision
  block inside a `random.getstate()` / `setstate()` pair: the engines still draw,
  so the decisions are still varied, but the football stream is left exactly
  where it was found and the same key yields the same draws. This is the class
  of thing that broke `test_cross_detector` the first time round.
- **FAR-SIDE CM MODE SET.** Of the four CM modes, `drop` (pocket between the
  centre-backs), `carry` (through the lines) and `late` (box arrival) all steer
  TOWARD the ball's side; only `orbit` (the open far channel) does not. Letting a
  far-side midfielder take any of the other three collapses the width CK35 exists
  to hold. So a midfielder more than `CM_DROP_SIDE_TOL_M` (18 m) from the ball's
  side may only orbit. A drop-only guard did NOT fix it — `carry` was doing the
  pulling. `test_triangle_support` measures this: the far CM now drifts 24.0 ->
  25.015, against 24.0 -> 25.4 at HEAD. It still fails its `+1.0 m` assertion by
  1.5 cm, and was ALREADY failing at HEAD, so it is a knife-edge pre-existing
  failure that this work improves rather than introduces. Do NOT tune the CM
  further to shave 1.5 cm — that is tuning real behaviour to pass an assertion.
- BUGS FOUND BY ITS OWN SUITE, all from the same class — a name that exists in
  the write path but not in the extracted helper: `NameError: minute` in
  `_midfield_run_targets` (the `last_active_minute` guard leaked into the
  decision body), and `NameError: awareness_bonus` in `_cam_pocket_targets`
  (the weight line was captured as a reference and never written). The second
  one CRASHED every full match; only the full-match tests reach it, because the
  dormant suites have no CAM in the squad. Dry-run-every-surgery and hard
  boundary assertions caught both before they landed.
- DONE: **RUN TYPES NOW REACH THE EXPORT** (the "calculated but not in any export"
  report, which was correct and worse than an oversight). `RunTracker` had been
  classifying six types live for months with per-type cooldowns and gain
  thresholds, exposed as `MatchEngine.get_run_profile()` — and the exporter
  never called it. The only run-ish counter, `_count_off_ball_runs`, walked
  `position_log` and incremented ONE flat `runs_without_ball` for any
  frame-to-frame jump >= 15 m, so a keeper shuffle, a half-time reset and a
  genuine diagonal run counted identically. Compounding it, my own live run
  layer discarded the run MODE at the position-layer boundary, so the decision
  that CAUSED a run was unrecoverable.
  - `run_tracking.IntendedRunRecorder` records the engine's own vocabulary
    (behind/hold/box, byline/cut, overlap/underlap/tuck, drop/carry/late/orbit,
    roam) on the cache MISS, so one count per real decision, not per 10 Hz tick.
  - The mode is now carried through `striker_run_targets` and
    `offball_run_targets` as a 4th tuple element. Both profiles are attached to
    `MatchResult` (`run_profile`, `intended_run_profile`) — the same shape trap
    `world.ingest` documented for `sub_controller`, where a post-match fact on
    the engine is silently lost by anything reading only the result.
  - `exporter._apply_run_taxonomies` fills 24 columns in the **Pass Profile**
    sheet: 6 observed + total, 14 intended + total, with the legacy
    `runs_without_ball` KEPT (a real exported column; removing it would change
    every historical export's shape) but relabelled "legacy, not a run count".
  - Verified end to end by writing a real workbook and reading the columns back
    (`_diag_run_export.py`), not just on the stat dict. First real match:
    observed 1502 (advance 456, support 706, far_side 171, overlap 67,
    underlap 47, forward 55), intended 1091 (roam 318, carry 178,
    post channel 137, in-behind 113, overlap FB 91, late 74, drop 70,
    underlap FB 53, orbit 22, byline 16, drop-to-link 13, cut inside 6).
  - **Two findings the new columns immediately exposed, both open:**
    (a) `Runs Intended: Tuck In (FB)` = **0** — inverted full-backs never tuck,
    and the tuck is a signature positional-play move (Zinchenko/Cancelo), so
    `fullback_behavior.choose_advance_mode` is not reaching that branch.
    (b) The intended counts are **not comparable across modes**: the CAM pocket
    roam is unconditional, so it counts every (minute, zone) cache miss as a
    "run" and returns 318, four times the next-largest. It is counting
    positionings, not runs. Any mode-weighted total needs a common bar first.
  - DONE: **the six observed types were degenerate and now are not.** Measured on
    a live match, `support` was 1421 of 1497 observed runs (95%) and `overlap`
    fired ONCE. Cause: the predicates were an `elif` chain ordered
    most-specific-first, so the loosest one (`support`: `behind_now and
    back >= 6m`) sat last and became a catch-all. Two compounding defects:
    (a) `gain` and `back` are independent accumulators that both survive until a
    run fires, so a player oscillating around the ball banked backward metres
    while also banking forward metres and got labelled by whichever crossed its
    line first — a NET-FORWARD segment could be filed as a support drop. Both
    general cases now require the net direction to match the label.
    (b) `support` claimed no more than backward DRIFT; a support run is a
    TRANSITION (advanced -> behind), so it now requires a new `ever_ahead` flag.
    Without that gate support stayed at ~80%.
  - DONE: **jump-aware filtering.** On-ball players are repositioned by
    possession-episode traces and keepers are written directly, so a position
    legitimately changes by up to ~99 m in one 10 Hz tick, and `RunTracker`
    accumulated gain between samples — a jump was indistinguishable from a
    sprint. `RunTracker.set_top_speeds()` now takes the engine's own speeds
    (`5.0 + pace*0.042` m/s, 7.3-8.7 in practice) and the guard refuses to
    OBSERVE a discontinuity: segment cleared, post-jump position adopted as the
    new reference with no gain banked. Controlled replay of one recorded feed:
    803 runs filter off, 750 filter on — **6.6% of observed runs were artefacts**,
    204 jumps caught. Live distribution after both fixes: far_side 46%,
    advance 24%, overlap 10%, underlap 7.5%, forward 4.5%, support 7% — six
    distinguishable categories instead of one.
  - **THREE CALIBRATION ERRORS OF MINE, recorded because they produced alarming
    false findings.** `top_speed_mpm` on the spatial state is a per-minute STEP
    DISTANCE, not a speed. Comparing it against `dt` in seconds gave a limit ~60x
    too generous; dividing by 60 gave one ~10x too strict; reading
    `_top_speed_cache` before `simulate()` populated it gave a third wrong
    answer. Those three mistakes successively produced "896 teleports",
    "49.7% of samples are teleports" and "98.9% of runs are artefacts" — all
    false. The real figure is 6.6%. Read the units from the source, not the field
    name. A fourth: the jump guard was inserted AFTER
    `self._last_t[name] = t`, so its elapsed time was always exactly 0 and the
    guard was INERT — which a missing `math` import was hiding. A controlled
    replay reporting `jumps_filtered == 0` with 200+ non-physical samples present
    is what caught it. A silently-no-op guard is worse than no guard.
  - ~~**AN A/B ACROSS TWO `simulate()` CALLS IS INVALID.**~~ — **RETRACTED
    2026-10-04.** Both arms scored differently (0-0 vs 1-0) and the cause was
    attributed to module-level brain caches. It is not: `_diag_seed_repro.py`'s
    `arm none` — two `simulate()` calls, same seed, **nothing cleared** — is
    byte-identical to a single-call reference. An in-process A/B is valid. What
    made those two arms differ was that they were run in two *processes* with
    different `PYTHONHASHSEED` values, which was the real and separate bug, now
    fixed (see *REPRODUCIBILITY*).
  - 22 tests in `tests/test_run_taxonomy_export.py` (19 fast + 3 match-level), all
    green. Includes one that EACH of the six types is reachable from a movement
    shaped to mean it, so a type that can never fire fails directly rather than
    hiding inside an aggregate share. The two vocabularies genuinely SHARE words
    "overlap"/"underlap" (the FB engine really does choose an overlap run, and
    the tracker really does observe an overlap); the export KEY namespace is
    what disambiguates them, and that is documented so nobody "fixes" the
    collision by renaming.
- MEASURED: 46/46 in `tests/test_positional_play.py` (incl. a full real match).
  26/27 REGRESSION (16 min): `tests/tests.py` + `test_chronography` +
  `test_possession_causality` + `test_checkpoint7_subsystems` = **50 passed,
  0 failed**. Note the documented pre-existing
  `test_pass_matrix_sums_match_real_events` did NOT reproduce here — it is
  NOT fixed. Its signature is matrix − manual = the number of successful
  crosses (a definitional mismatch in `PassMatrix.build`, README §6), so it
  fails or passes depending on how many crosses a match happens to contain:
  371 vs 370 at HEAD, 538 vs 535 with striker+rest defence, exact tonight.
  Do not read a green run as a repair.
- MOVEMENT, RENDERED (2026-09-29, `_diag_movement.py` → `_diag_movement.png`):
  the layers demonstrably REACH the pitch, and two realism metrics IMPROVE —
  the striker went from **0 sprints all match** to 63 (real is 20-40), and the
  winger's total fell 14.6 km → 12.1 km (real ~11-12; the old figure was over
  the band). The OFF baseline of a striker who never sprinted is plainly wrong.
  BUT the movement reads as JITTER, not runs: striker sd(x) 12.98 → 14.94 with
  high-frequency oscillation and constant transit, because the decision is
  re-rolled every 15 m of ball movement and a 0.30 blend never lets him
  arrive. Position-density panels are only marginally more structured and the
  hotspots sit at the same anchors, so the claim "the half-spaces are now
  occupied" is NOT supported by this evidence. The fix is COMMITMENT, not more
  behaviour: scope the decision to the possession episode, not the zone.
  No per-tick cost — four arms at 1.5–1.9 ms/tick are within noise of each other.
  **Cross probe** (6 seeds, `test_cross_detector`'s wing-play vs 10-CB shape):
  every arm produced a confirmed cross on 6/6 seeds — HEAD 19 attempts/11
  confirmed, rest-defence-only 24/18, striker-only 18/11, both 23/14. Rest
  defence INCREASES crosses; the striker layer leaves them flat. So the
  single-seed failure of `test_match_crosses_stamped_geometrically` was seed-luck,
  not a degradation. NOTE: that probe ran against the earlier stream-consuming
  version of the striker layer; the deterministic-RNG change then fixed the
  failing test outright (`test_match_crosses_stamped_geometrically` and
  `test_generic_passes_reclassified_as_crosses` both pass on the current code),
  because the failure mechanism WAS the stream shift. That is the argument for
  not drawing from the global stream, made concrete.
- PRE-EXISTING FAILURES, confirmed by reverting all four files to HEAD and
  re-running (not caused by this work): `test_triangle_support::
  test_live_10hz_integrator_moves_near_cm_into_flank_socket`,
  `test_touchline_wide::test_out_of_possession_touchline_press_gated_by_press_
  intensity`, `test_preservation_properties::test_realistic_shot_woodwork_and_
  rebound_preserved`. The first two read the live 10 Hz integrator directly and
  are worth fixing before trusting the integrator's numbers.
- ENVIRONMENT WARNINGS (not code): a real match takes **~90 s here, not the
  19–23 s** in "Important Details" — that figure is stale, and 2.5x swings
  between arms on one machine are warm-up noise, not signal. `tests/__pycache__`
  holds .pyc files compiled at the pre-move path
  `C:\Users\Trevor Majani\Downloads\plofa_checkpoint6\plofa`, so tracebacks point
  there; that copy of the project still exists. Search tooling in this shell
  (PowerShell `Select-String` globs, ripgrep, and identifiers passed through
  here-strings) returned contradictory counts for the same file — trust
  `read_bytes().count()` over everything else.

## SMALL GAME — the scenario harness (2026-09-30)

`small_game.py` + `tests/test_small_game.py` (7 tests, 6 green + 1 strict
xfail). A controlled picture, a few seconds of the REAL 10 Hz integrator via
`MatchEngine._offball_run`, traces out. **0.8 s per scenario against ~90 s for
a full match.** Real player DNA and real formation anchors; only the initial
coordinates are the scenario's, so the shape rules under test are shipping
ones. There is no second simulation path that could pass here and fail there.

It exists because the project's recurring pathology is a mechanism that
exists, is wired to something, and never runs — the striker layer sat dead in
`drift_minute` for months, the wide/FB/CM/CAM run modes died in the same
method, and `POLICY_INTENT_AUTHORITY` was gated at five sites and not the two
that force the delivery class. All 46 positional-play tests were unit tests of
a shape function or a full 90-minute match, with nothing in between, so
"does the striker get in behind?" could only be answered by staring at a full
match's aggregate. That is how four separate false findings happened in one
session.

**It asserts on where players ENDED UP, never on which functions were
called.** A test asserting "the run layer was invoked" would have passed
against the dead `drift_minute` wiring all day.

- `play()` returns `traces`, `final`, `start`, `intended` (what the engine
  decided) and `observed` (geometric), because a wrong outcome from a wrong
  decision needs a completely different fix from correct-decided-but-badly-
  executed. `intended_diag` carries the recorder's own diagnostics.
- `depth_series(trace, attacks_right)` NORMALISES for the half-time ends
  change. An un-normalised depth chart is how "the shape oscillates end to end
  every few minutes" was believed: it was one mirror at 45'.
- A position key may hold a LIST of coordinates, because two centre-backs
  share a position and an offside line cannot be expressed without saying which
  two defenders.
- Mirroring a scenario must mirror the FORMATION ANCHORS too, not just the
  initial coordinates.

- **OPEN FINDING — the striker does not get in behind.** Ball in midfield 24 m
  from a two-man line at 79/85, striker at 66, six seconds: he moves
  **BACKWARDS, 66.0 → 62.3**. Not a scenario artefact — the first version of
  the fixture placed the line 25 m from the ball and it dissolved 35 m in six
  seconds (correct engine behaviour), and fixing that only moved the failure
  from −4.7 m to −3.7 m. Narrowed to either the "behind" target sitting
  barely ahead of where he already stands, or `STRIKER_RUN_BLEND` (0.30)
  losing to the ball-side shape compaction, which seats the whole team at ~62
  when the ball is at 55. **NOT YET DIAGNOSED.** Marked `xfail(strict=True)`
  so the suite is green but a fix turns it RED and forces the note deleted.
- **A left/right asymmetry that is not yet believable.** With anchors properly
  mirrored, the left-attacking version of the identical picture PASSES and the
  right-attacking one fails. That is the REVERSE of the 2026-09-29 saturation
  bug (which made in-behind always-allowed when attacking left). Four
  plausible-but-wrong mechanisms were already ruled down tonight; do not
  believe this one until it is isolated.
- Fixture B (the tuck) PASSES on the real roster — independently confirming
  `Tuck In (FB) = 0` was the synthetic fixture's DNA, not a code gap.
- Fixture C (does he settle) PASSES: 15 s, ball 30 m behind him, he holds a
  line. The striker's shape IS stable.
- **THREE BUGS THE HARNESS FOUND IN ME, IN TWENTY MINUTES**, recorded because
  they are the argument for it: (1) I compared the engine's team names against
  the literal string `"home"`, so both clubs attacked left and fixture A failed
  for a reason unrelated to the striker; (2) the "mirrored" fixture did not
  mirror the anchors, so the away side was hauled 55 m the wrong way and the
  test **XPASSed for entirely the wrong reason** — the most dangerous way a
  test can be green; (3) the scenario dissolved its own defensive line, which I
  would have read as an engine bug. A harness that cannot itself be verified
  is the failure mode it exists to catch, so its liveness, its placement and
  its two-CB handling are asserted as tests.

## MEASUREMENTS (2026-09-30) — all on the real 26/27 roster, Oxton v Natrican

- **EVENT COUNT: 2634–3263, mean 2968, sd 346 over 12 seeds** (4 below 2900,
  8 in 2900–3300, none above 3300). A second 10-seed sweep gave 2721–3245,
  mean 2915. So it straddles 3000 roughly 60/40 and is NOT always above it.
  NOTE: the "~3000-3700" figure in the Checkpoint 32b note is NOT comparable
  to this — different roster and date — and the two should not be quoted side
  by side. The 1500–3400 in `event_chain`'s own docstring is a different claim
  again.
- **BRAIN-ONLY vs DECIDERS, 3 matches per arm, SEPARATE PROCESSES** (the
  stated reason — "an A/B across two `simulate()` calls in one process is
  invalid — the second inherits the first's module-level brain caches" — is
  **retracted 2026-10-04**, see *REPRODUCIBILITY*; separate processes are just
  habit here):

  | | deciders | brain only | real PL |
  |---|---|---|---|
  | events | 2866 | 2382 | — |
  | passes | 808 | 641 | — |
  | GK receptions | 37.7 | 25.0 | — |
  | median pass length | 9.7 m | 23.9 m | 15–18 m |
  | forward / square / backward | 27.9 / 41.0 / 31.1% | 35.8 / 27.0 / 37.2% | 35-40 / 45-50 / 10-15% |
  | **median displacement** | **−0.2 m** | **−0.07 m** | — |
  | progressive | 26.5% | 41.7% | 25–35% |
  | ends in final third | 15.1% | 15.6% | 15–20% |

  The brains play a recognisable game and over-correct in BOTH directions:
  length overshoots the band, progressive overshoots the band, GK receptions
  and event count collapse. **The deciders are a DAMPER, not a source of
  direction** — the distribution got wider, not more biased. Re-run with the
  fully-wired switch (4 receiver sites, not 2): events 2173–2612, gk 15–31,
  len 21.8–24.9, prog ~40.4%, fwd 33.6–38.0, back 35.4–40.3, **dx −0.2 to
  +0.1**. Conclusion unchanged, which strengthens it.
- **`pass_direction` metadata is uncorrelated with geometry.** Label
  "forward" → 125 fwd / 123 sq / 146 back; "backward" → 140 / 116 / 103;
  "sideways" is correct (73 of 83). Marginals match (365/371 labels vs 268/262
  geometry) while the assignment is random — the signature of a stale or
  shuffled value, not a wrong formula. **Untested hypothesis: it clusters
  within a possession or per player.** `event_chain.py:2482` already makes
  exactly this argument for crosses and long passes and `detect_cross` /
  `detect_long_pass` already exist; `pass_direction` was never converted.
  Until fixed, the exporter is stamping a field wrong two-thirds of the time.
  > ### ⛔ RETRACTED 2026-10-02 — THIS FINDING WAS A MEASUREMENT ERROR
  > **The code is correct. Do not "fix" it.** Re-measured by `_diag_pass_direction.py`
  > over 1,674 labelled passes in 2 real matches:
  >
  > | check | agree | disagree |
  > |---|---|---|
  > | **correct** (flip the sign for the away side) | **1,647 (98.4%)** | 27 (1.6%) |
  > | naive (no flip — what the original probe did) | 985 | **689 (41.2%)** |
  >
  > The naive row reproduces the original "~two-thirds wrong" figure almost
  > exactly, which is what identified the probe as the error rather than the
  > code. `event_chain.py` does it properly:
  > ```
  > pass_advance = end_px - x
  > if not attacks_right:
  >     pass_advance = -pass_advance
  > _pc = classify_pass(..., signed_dx=pass_advance, attacks_right=attacks_right)
  > ```
  > `classify_pass` computes the sign itself when `signed_dx is None`; the call
  > site passes it explicitly, correctly flipped.
  >
  > **Why the original probe got it wrong:** "sideways" is a `|dx|` band and
  > cannot be affected by a sign error at all — which is why it was the ONLY
  > category that agreed. That asymmetry is the signature of a one-sided sign
  > error, and it should have been read as such at the time. `PositionEngine.team_attacks_right`
  > is set once per team at `initialize_team` and **never flipped at half-time**,
  > so home is always `True` and away always `False`, both halves — confirmed by
  > the probe's per-half table.
  >
  > The residual 27 (1.6%) are all `sideways -> forward/backward` boundary
  > cases: the label is derived from the AIMED `end_px`, the event stores the
  > recorded ARRIVAL `end_x`. That is the open `end_px`-vs-`end_x` question
  > already on file under the export frame note, not a mislabelling.
  >
  > **Lesson, fifth instance of the same shape:** the probe's marginals agreed
  > with the truth while its *assignment* did not, and the project read that as
  > "a shuffled value" rather than "my check is in the wrong frame". When a
  > check disagrees with the code, verify the check's frame before the code.
- **THE PROGRESSIVE FLAG IS FINE.** An early reading of "7.3% progressive"
  was MY error: `EventType.PROGRESSIVE_PASS` requires `is_prog and success and
  abs(end_px - x) > 9.14`, a narrow subset. The engine's `is_progressive`
  metadata is 27.6%, inside the real 25–35% band. **Do not quote 7.3%.**
- **The turf collapses fast.** A backline placed 25 m from the ball travelled
  35 m upfield in six seconds under shape compaction. Correct engine
  behaviour, and the reason a scenario must be a picture that can actually
  occur.

## PLOFA MATCH EXPORT — our own format (2026-09-30)

`plofa_export.py` + `tests/test_plofa_export.py` (**20 passed, 2 strict xfail**).
Deliberately NOT StatsBomb-shaped: that schema answers StatsBomb's questions
and is lossy about the things that are none of its business. PLOFA's format
does three things a provider format cannot.

1. **ONE FRAME — "home attacks right", always.** Not per-event, not
   per-possession. `y` is never mirrored, so "left channel" means the same
   pitch side in both halves. The ends change is recorded per event as `half`
   and never baked into the numbers.
2. **DECISION IS NOT EXECUTION.** Every action carries `decision` (authority,
   intent, confidence, the brain's stated reason, and what the matrix and
   phase layers said) beside its physical result. A real record from a live
   match: brain chose `PROGRESSIVE_PASS` ("line-breaking option"), matrix said
   `RECYCLE_PASS`, phase ordered `recycle_backward`, and the ball went
   **backward 11.8 m**. That is `DECIDER_BOUNDARY.md` visible in one event,
   and no provider format can express it.
3. **THE FILE EXPLAINS ITSELF.** A `schema` block travels inside listing every
   enum and unit, so an export is still readable in six months without this
   repo. A `gaps` block states what is absent (`related_events`,
   `freeze_frame`, `off_camera`, `lineup`, `passing_network`); a
   `diagnostics` block carries the intended-run bar and the jump-filter count.
   Nothing is invented — absent fields are OMITTED, never zero-filled.

Own vocabulary throughout: 60 event types (vs StatsBomb's 24 in the reference
match), the 10 on-ball intents, 6 observed run types, 14 intended run modes, 4
possession phases, the matrix actions, plus the off-ball layer (press `g`, run
mode and target). Sample: `plofa_output/_format/Oxton_vs_Natrican.json`.

- **FRAME IS STILL AN OPEN QUESTION — do not "fix" it casually.** Got wrong
  twice. An intermediate version mirrored per ACTING team; a constructed case
  killed it (home shooting at raw 105 and away at raw 0 both landed on 105 —
  two sides cannot attack the same end). The current version is defined by the
  HOME side only, which is correct on constructed ground truth. BUT on a live
  match the away side's passes sit mostly at high x under it, which should not
  happen. Either the raw coordinates are in a per-attacking-direction frame
  after all, or the half-time flip double-counts a swap the engine does
  elsewhere. Pinned by
  `test_frame_which_half_the_team_operates_in` (strict xfail, full evidence in
  the reason). **Resolution needs `end_px` — the pass destination the engine
  AIMED at, as distinct from the recorded arrival `end_x`.** Exporting both
  would settle it immediately and is worth having regardless.
- **NEW FINDING: `end_x` is not populated on shot events.** On a live match,
  6 of 8 Oxton shots carried `end_x = 0.0` — their OWN goal — while the other
  2 correctly carried 105.0, and all 20 Natrican shots carried 0.0. A field
  that is 0.0 for three quarters of one team's shots and always-right for the
  other is an unpopulated default, not a coordinate. `validate()` therefore
  reports the shot check as a WARNING naming this cause, because failing on it
  would report a frame bug that is actually an engine bug — and a validator
  that cries wolf gets deleted.
- **9 coordinates off the pitch** (to −12.8 and 119.7) in a live export,
  exported verbatim and counted in `diagnostics.off_pitch_coordinates`, never
  clamped. From the simulation, not the transform.
- `decision_authority` was WRONG in the schema's first version: five values
  documented while the engine also emits `role_fallback` and
  `emergent_opportunity`. Found by a test asserting every authority in the
  file is documented. **`role_fallback` is the metric that matters: it is the
  count of on-ball actions the trained policy did NOT make.**

## SHOT MAP TRAJECTORY — the true flight exists; the exporter was faking it (2026-10-01)

**THE ANSWER TO "does the true shot trajectory exist": YES, and the shot map was
drawing a fabrication anyway.** `AttackChain.generate` builds a real `BallFlight`
(`aim_shot_flight`) and resolves it against the keeper's dive envelope
(`geometry_engine.resolve_shot`, 0.1 s integration); the terminus —
`ShotResolution.goal_point` — already rides on every open-play shot event as
`end_x`/`end_y`. `PossessionEpisode.resolve_shot` also traces the whole flight
per tick into `episode.ball_path` and `metadata["motion_trace"]` (`bx/by/bz`).

The exporter read NONE of it. All seven `shot_map.append` sites recorded only
origin + outcome, and the post-pass **overwrote** every endpoint with
`_shot_trajectory()` — a deterministic invention hashed off the origin
coordinates (`exporter.py` ~502). Everything downstream (PNG dotted line, Excel
"Shot Map" sheet) was drawn from that.

- MEASURED (`_diag_shot_endpoints.py`, 57 shot events, 2 real matches): **36/57
  (63%) carry a real physics endpoint**, 35 of them within 1.5 m of the correct
  goal line. Set pieces (corner / FK / penalty) carry none.
- **The `sx > 80` away-team mirror in `plot_shot_map` is DEAD and was WRONG.**
  Measured: the origin contradicted the team ONCE in 57 shots, at sx = 76.7 —
  below the threshold, so the branch never fired. It also flipped `ex`, so any
  shot it caught would have had its trajectory drawn pointing AWAY from the goal
  the ball finished at. **REMOVED.** Both teams are on one shared pitch (home
  right, away left) and every endpoint is already in that frame.
- The `goal_x` "safety net" inside `_shot_trajectory` is **KEPT but rescoped**.
  It now only ever runs on set-piece rows that have no physics, where the origin
  is the only evidence there is. It is load-bearing for penalties: measured a
  Natrican penalty at origin x=11 → endpoint x=0.0, correct.
- **Every row is now labelled.** `Trajectory` = `physics` | `reconstructed` in
  the export, and the PNG footnote says "resolved ball flight; set pieces
  inferred". A reconstructed row is never passed off as simulated. Also added a
  `Minute` column — the Shot Map sheet had no way to place a shot in a match.
- VERIFIED (`_verify_shot_traj.py`, reads the columns back off a real export):
  17/17 physics rows point exactly at the goal their team attacks, segment
  lengths 2.1–28.2 m, endpoint matches the originating event's own `end_x/end_y`,
  PNG renders. `tests/tests.py -k exporter` still passes.
- **TWO PRE-EXISTING BUGS IN THE CORNER CHAIN, surfaced by the new check and
  NOT FIXED (they change simulation/export data, so they need a gate):**
  (a) `event_chain.py:6874` clamps a corner header's origin to
  `x = max(85.0, min(102.0, ...))` with **no `mirror_x`** — every away-team
  corner is recorded at the WRONG END of the pitch (measured 4 rows/match, all
  Natrican corners landing at x≈91–94). Old and new exporter draw the identical
  endpoint there; nothing had ever looked at it.
  (b) `event_chain.py:6965` — the corner's `SHOT_OFF_TARGET` omits
  `situation=SituationType.CORNER`, so it defaults to OPEN_PLAY and **pollutes
  open-play shot statistics**. This is why 2 of the 5 flagged rows above were
  labelled `open_play`.
- CORRECTION to the 2026-09-30 `end_x = 0.0` note above: `end_x` is **not**
  unpopulated. Measured raw: 105.0 for the home team, 0.0 for the away team —
  populated and *correct* on both sides. The old reading was the home-frame
  artifact, not a default.
- STILL SYNTHETIC, deliberately out of scope: `full_match_ball_path` across shot
  windows is a constant `_FETCH_SPEED = 4.0` m/s lerp, because `AttackChain`
  never assigns `result.ball_path` so `_absorb_motion` falls through to
  `_synthesize_ball_path`. Measured by `_diag_shot_traj.py`: `v_in == v_out ==
  4.0`, `straight ≈ 0.01` on the flat rows, and the path end is within 3 m of
  the recorded `end_x` on **0/6** shots. The real per-tick flight is in the
  episode and dies there — so `pitch_replay` is still drawing a lerp through
  every shot. Wiring `result.ball_path = episode.ball_path` in `AttackChain`
  would fix it; the dotted line is one straight segment either way.
- `tests/fix_proposal.py` describes the away-team mirroring bug against a
  `plot_shot_map` that no longer exists (per-panel mirroring, pre-jointgrid). It
  is a print-only script, not a test. Obsolete — the code it warns about is now
  gone entirely.**

## CORNER BOX — from teleport to run (2026-10-02)

Step 1 (earlier) filled the box by WRITING 13 players into it with
`set_piece_place`. Step 2 makes them RUN there. The result is that a corner is
now a race that some players lose, rather than an arrangement that is always
already in place — and getting there required finding four separate no-ops,
every one of which looked correct in the source.

- **DISTANCE: the question was already answered, deliberately.** The instinct
  was to add `count_distance=False` to `record_touch` to stop the box
  placement banking ~250 m of fictional movement per corner (21/21 corners
  measured `sequence_duration_s = 0.0`, so the displacement was credited in
  zero seconds). There is already a purpose-built `set_piece_place`, and its
  docstring books the displacement ON PURPOSE: *"a man who jogs 20 m to the
  wall has covered 20 m, and hiding that would just move the lie to a
  different column."* The opt-out was reverted — `position_engine.py` was never
  modified by that attempt (`count_distance` count: 0). The counting question
  was already settled in the codebase; overriding it on the strength of my own
  reasoning is backwards. It self-corrects here anyway: with real elapsed time
  the movement becomes genuine and is counted ONCE, with real speed, by
  `record_physics_distance`. **Answer: counts as distance, does NOT count as a
  run.** Run sampling is suppressed for the whole dead-ball window
  (`_sample_run_tracking`), because the six observed types describe movement
  relative to the ball in LIVE play — `support` was 95% of all runs once
  already, and "centre-back shuffled to his marker" is the same error.
- **THE REAL STEP-1 BUG.** The box was placed with `record_touch`, which
  applies the wide-role flank hold and the GK own-box anchor.
  `set_piece_place`'s docstring records exactly what those corrections did to
  the free-kick WALL: *"13 of 14 walls were wider than the men in them could
  possibly produce ... individual men ended up 3.0 m and 4.0 m from the ball
  when the wall distance is 9.15 m."* An order-dependent arrangement was being
  placed with the method documented as dismantling them. The WALL still uses
  `set_piece_place` (both remaining call sites, 7489/7512).
- **FOUR NO-OPS, all measured, none visible in the source.**
  (1) The window opened but held a position already reached — `set_piece_place`
  had teleported everyone first, so the median gap from slot to player was
  **0.0 m** across 156 slot-assignments in 12 corners and **0.03 of them**
  closed any ground. Live, closing, and completely inert.
  (2) The dead-ball gate consumed the window with `tick_setpiece(duration_s)`
  at the TOP of `_offball_run`, but `_sample_run_tracking` runs LATER in the
  same function (3214/3260) — so the flag was always `False` by the time it
  was read. The same class as the jump guard that was inert because it was
  inserted after `self._last_t[name] = t`.
  (3) The window was consumed before the SUB-TICK loop, so a 5 s window was
  live for exactly one 0.1 s step: **0.93 m/s against a 39 m median gap**.
  Consumption is now per sub-tick (`tick_setpiece(DT)` inside `for i in
  range(ticks)`), which took it to **+27.4 m in 5.0 s = 5.48 m/s, 88% of
  assignments closing**.
  (4) The approach speed. The generic ladder hands out `jog` (1.5-1.9 m/s,
  ×0.30 when the ball is far) — a SHAPE-HOLDING integrator, correct for holding
  a shape you are already in and useless for reaching one you are not. Reused
  the low-block recovery branch's own shape and constants
  (`_SET_PIECE_APPROACH_TOP_FRAC = 0.75`, `_SET_PIECE_APPROACH_GAIN = 1.1`)
  because it is the same problem for the same documented reason.
- **THE ORDERING DEFECT — the reason a window was never going to be enough.**
  `_simulate_minute` calls `ChainDispatcher.set_piece(...)` at 4603 and
  `_absorb_chain(...)` (which integrates motion) at 4613. The corner chain
  resolves the header in the SAME call, so **anything the window does arrives
  after the ball has already been crossed**. With the placement removed, the
  header was being contested by an EMPTY box — a regression in the opposite
  direction from the teleport. Fixed by
  `PositionEngine.advance_to_slots(targets, seconds, exclude=...)`: a
  synchronous run-in, inside the chain, BEFORE the delivery, walking each man
  to his slot at ~6 m/s with ease-in and ease-out, booked once through
  `record_physics_distance` with real duration, speed and sprint count. The
  window is then opened anyway, to HOLD the box through the post-corner
  integration.
- **`SET_PIECE_JOSTLE_S = 5.0 -> 7.0`.** Not a guess: at 5.0 s the median gap
  at the cross was still **14.7 m** with only 24% arrived. The median gap from
  a player's own position to his slot is 41 m and a corner run-in is a sprint,
  so 5.0 s cannot cover it. A real corner is typically taken 5-10 s after being
  awarded; 7.0 s is inside that band.
- **MEASURED AT THE CROSS** (the only moment that counts — the earlier "13 in
  the D" figure came from the helper's RETURNED placements, which is not
  evidence about the pitch): median **8 outfielders in the box, range 7-13**
  (real corners 8-12), 62% of slot assignments arrived, **26% still >15 m
  away and genuinely did not make it**. Window opens and closes exactly once
  per corner, never left open at match end, run sampling suppressed ~2% of
  samples.
- **SUITES:** `test_positional_play` + `test_small_game` + `test_decision_brain`
  61 passed / 1 strict xfailed. `test_run_taxonomy_export` + `test_plofa_export`
  42 passed / 2 strict xfailed (the six-type taxonomy is INTACT — the dead-ball
  gate did what it was for). 26/27 regression 48 passed / 2 failed: the
  documented pre-existing `test_pass_matrix_sums_match_real_events` (446 vs
  445), and `test_pass_network_positions_stay_realistic` at
  `GK average x=25.11 is too advanced` — **non-deterministic**: three runs at
  current code gave pass, pass, fail (25.29). A threshold sitting inside the
  measured noise band, not a deterministic effect of this change; NOT bisected
  against HEAD, and worth fixing before trusting the integrator's numbers, as
  AGENTS.md already says of it.

## CORNER STEP 3 — the header is contested by the whole box (2026-10-02)

**The box was cosmetic.** Steps 1 and 2 spent a corner crowding thirteen men
into the box and then running them there, and the aerial duel was being
contested by **three players**:

    attackers = [attacker_mp] if attacker_mp else []      # the receiver
    defenders = [defender_mp, gk_mp]                      # one man + keeper

`resolve_aerial_delivery` scores EVERY contestant it is given and takes
`candidates[0]` — it has always supported N. We were offering it one. So the
crowding and the running changed the pitch and not the football.

- **NOW OFFERED:** everyone who actually ARRIVED, via
  `PositionEngine.arrived_setpiece_players(SET_PIECE_ARRIVED_M=3.0)` — measured
  against his assigned slot, not against the goal, because a man still 20 m away
  when the ball is crossed is not in the duel. Step 2 is what made this
  knowable. The keeper is still added unconditionally: he is on his line either
  way and the resolver charges him the real movement cost to the contact point.
- **A SECOND BUG THIS EXPOSED, WHICH HAD TO SHIP WITH THE FIRST.** `att_wins`
  required the winner to *be* the receiver:

      and getattr(aerial.winner.player,"name","") == getattr(attacker_mp.player,"name","")

  Harmless while one attacker competed. The moment a real box is contested, an
  attacker who wins the header other than the designated receiver falls through
  to "NO CLEAR WINNER → BALL FALLS LOOSE". Fixing the contest without fixing
  this would have turned every genuine attacker header into a loose ball. The
  shot is now taken by `headerer` — whoever won the aerial — and all 8 receiver
  references in the header block were retargeted.
- **MEASURED (dead-ball duels only — 46 aerial duels in a 7-corner match are
  mostly OPEN PLAY, and averaging them in hides the corner entirely):**

  | | before | after |
  |---|---|---|
  | attackers offered | 1, always | **median 3**, range 1–6 |
  | defenders offered | 2, always | **median 5**, range 4–7 |
  | corners with >1 attacker | 0 | **6 of 10** |
  | distinct winners across a match | 1 per corner | **3 across 7 corners** |

- **CORNER CONVERSION, 6 seeds in 6 SEPARATE PROCESSES** (mandatory at the
  time because of the now-retracted brain-cache rule; and note these numbers
  predate the crc32 fix so they carry hash-order noise): 3 goals from 58
  corners = **5.2%**. Real Premier League is 3–4%, so this is in the band, and
  it is up from step 1's 2.8%.
- **OPEN FINDING — attackers win 93-100% of corners.** Measured
  `won-aerial` by seed: 100/100/93/100/83/100. Real football has defenders
  clearing the large majority of corner deliveries, and a keeper claiming or
  punching a meaningful share; neither is happening at any rate worth
  reporting. This is the next thing to investigate and it is NOT fixed.
  Suspects, in order: the cross is aimed at the receiver's slot, so the whole
  attacking pack converges on the ball's path while the marking puts each
  defender 1.4 m goal-side of a man who is himself running at it; and
  `candidates[0]` may be ordering on something that favours the attack.
  Measure which before touching it — do not add a probability to hide it.
- **SUITES:** `test_positional_play` + `test_small_game` + `test_decision_brain`
  61 passed / 1 strict xfailed. 26/27 regression 48 passed / 2 failed — the
  same two as before step 3 and both already characterised (the documented
  crosses-signature mismatch, and the non-deterministic `GK average x` one).

## CORNER STEP 4 — the 93–100% was TWO defects, one of them mine (2026-10-03)

Asked whether conservation-of-energy physics would fix the attacker winning
93–100% of corners. **No**, and measuring it twice is what found the real cause.
Isolated probe first (`_diag_corner_race.py`, milliseconds, no match — the right
instrument here, since nothing about this needs a 90-minute simulation).

- **DEFECT 1: a dead heat was decided by ARGUMENT ORDER.**
  `resolve_aerial_delivery` built `candidates` from
  `attacking_players + defending_players` and ordered them with a **stable**
  sort on `(time_s, -reach)`. Two men on the *identical* spot: attacker passed
  first → ATTACKER wins; defender passed first → DEFENDER wins. Every call site
  passes attackers first, so every tie went to the attack.
- **DEFECT 2: the race was QUANTISED.** `if candidates: break` abandoned the
  flight at the first sample at which *anybody* could reach the ball, so a
  1.3 s cross was decided in 0.1 s buckets (`TICK_S`). The same pure tie
  returned ATTACKER at `sample_step` 0.1 and 0.05, and DEFENDER at 0.02 and
  0.01. **A duel that flips with the sampling resolution is not football.**
- **FIX (`geometry_engine.py`).** Score every contestant on `time_to_reach`,
  which is continuous, across a fixed `AERIAL_SCORE_STEPS = 240` grid;
  `sample_step` is now a FLOOR it can raise but never lower. Ties break on
  vertical reach, then on how far in front of the ball the man is **along the
  ball's own direction of travel** (a dot product — direction-agnostic, and it
  is the real tiebreak: in a dead heat the man the cross is travelling into
  gets it), then on **NAME**.
  - **`id()` is deliberately not the tiebreak.** It is not reproducible across
    processes, so a match would vary run-to-run for a tie — worse than the bias
    removed.
  - **My first fix MOVED the bug rather than removing it**, and the gate caught
    it: the per-player update was a strict `<`, so on an exact tie the first
    player *iterated* still won. The name term has to be inside the compared
    key, not merely present in the tuple.
- **GATE (`_diag_gate_aerial.py`, 0/5 failures on both properties):** argument
  order must not change the winner; sampling resolution must not either.
- **BUT THE REAL MATCH DID NOT MOVE — and that is what exposed defects 3 and
  4.** Seed 31: 8/8 attacker corners before the fix, 8/8 after. A real bug that
  changes nothing in the real number was not the dominant mechanism. The probe
  had said an attacker **4 m off his own slot** loses the duel, which pointed at
  the geometry.
- **DEFECT 3 (MINE, introduced by step 3): `att_wins` was overridden by its own
  fallback.** The `headerer` lookup set `headerer = receiver` whenever
  `headerer` was None — which is ALSO true when a **defender** won. So
  `att_wins` came out True for every controlled or contested corner and the
  `CORNER_TAKEN.outcome` delivery-quality flag read 93–100% attacker while the
  duel underneath was genuinely split. A fallback that cannot distinguish "no
  winner" from "the other side won" is not a fallback, it is an override.
  **This is the same trap as the fabricated assist: the computation was fixed
  and the field that REPORTS it was left wired to the fabrication.** The 93–100%
  headline was half this bug and half real geometry.
- **DEFECT 4: a man does not stop on his chalk mark.** `advance_to_slots` walked
  each man onto his mark and stopped *exactly*, so he stood on the ball's
  destination and scored ~0 arrival — nothing could beat him. It now ends each
  run-in short of the slot by a bounded residual
  (`SET_PIECE_ARRIVAL_ERROR_M = 1.8`, capped at `2.4` so it stays inside
  `SET_PIECE_ARRIVED_M = 3.0`). It models a human's positional error, not a
  football-outcome bias: it does not know which side a player is on.
- **MEASURED, 3 seeds in 3 separate processes, `CORNER_TAKEN.outcome`:**

  | seed | 31 | 57 | 73 | mean |
  |---|---|---|---|---|
  | corners | 14 | 8 | 13 | 35 |
  | attacker won the aerial | 64% | 50% | 46% | **~53%** |

  The duel agrees on its own terms — `_diag_race_in_match.py` logs every
  `arrival` inside a live match: **6 of 11 corner aerials won by the defence**,
  with the winner uniquely fastest in 8 of 11 and a median margin of 0.665 s
  over the next man. So the defence now wins on **arrival**, not on a
  tiebreak, which is the correct mechanism.
- **STILL OPEN, measured, NOT fixed:**
  1. **The keeper never wins a corner aerial — 0 across all three seeds.** Real
     keepers claim or punch a meaningful share. `gk_wins` is computed, but the
     keeper is added to `defenders` unconditionally and never gets there.
  2. **~53% is still attack-favouring**; defenders clear the large majority of
     real corners.
  3. The probe's sanity sweep is counter-intuitive and unexplained: with the
     defender placed progressively NEARER the ball, the attacker keeps winning
     until "nobody" is in reach at all. A 3 m advantage should not lose. **Do
     not act on this until it is understood** — it is the sixth rejected
     pairing of this kind in the audit.
  4. Delivery execution error (the user's "the kicker is not a god") is NOT
     done. The sweep showed it needs ~5.5 m of error before it flips a duel, and
     beyond that **nobody** contests at all — the 1.3 s flight is too short for
     a genuinely misplaced cross. The flight duration and the target have to
     move together. Short corners are also still absent.
- **ENERGY / BALL PHYSICS: still worth building, still not this bug.** It cannot
  change who wins a duel decided before the ball is struck. Its real target is
  recorded under *CORNER STEP 3*: corner headers still resolve through the
  probabilistic `GoalkeeperEngine.evaluate_save` while open-play shots use a
  real `aim_shot_flight` + `resolve_shot`. Genuine asymmetry, unaddressed.
- **PROBE NOTES worth not rediscovering:** `resolve_aerial_delivery` is imported
  **by name** into `possession_physics`, so patching the `geometry_engine`
  attribute intercepts nothing — the entry point is
  `PossessionEpisode.resolve_aerial`. And a set-piece window opens partway
  through the chain, so testing `setpiece_active()` when `generate` is ENTERED
  finds it closed and silently measures zero; test it where the duel resolves.
  Both mistakes produced "corners measured 0" here before being found.
- **SUITES:** `tests/test_set_piece_routines.py` 23 passed. The full suites were
  NOT run for this pass — the user's CPU was committed elsewhere — so this is
  **un-gated by the 26/27 regression** and should be treated as such.
- **`match_engine.py` LOST `class MatchResult` DURING THIS PASS** (a 223-line
  deletion, visible as `-class MatchResult:` in `git diff`, with nothing else
  missing). A match crashed at `match_engine.py:4005` with
  `NameError: name 'MatchResult' is not defined`. Restored from HEAD verbatim;
  the file is now correct and `MatchResult` is a `@dataclass` with 18 fields.
  Worth noting for the commit hazard above: **a deleted class at column 0
  still leaves a file that imports cleanly**, and my restore dropped the
  `@dataclass` decorator line above the class, which turned the crash into
  `TypeError: MatchResult() takes no arguments` — restore the decorator too.

## CORNER STEP 5 — the kicker is not a god (2026-10-03)

The user's objection: *"the kicker isnt a god that all his corners aim
perfectly, he is player who his hoping for someone to score, he just kicks."*

- **WHAT IT WAS.** `target_x = rx + random.uniform(-1.0, 1.0)` — the delivery
  was computed at the receiver's OWN placed position, plus a metre. The cross
  was aimed at the man, not at a zone, which is the geometry that made every
  corner unwinnable for the defence: the receiver stands on the ball's
  destination and scores ~0 arrival.
- **WHAT IT IS NOW — three terms, all from measurements the engine already
  has.** `CORNER_LEAD_M = 2.2`: how far in front of his runner a good crosser
  plays it, scaled by `cross_quality` and directed **toward the goal being
  attacked** (`_goalward = 1.0 if attacks_right else -1.0` — the same
  half-time mirror trap as the export frame, so it is explicit).
  `CORNER_ERROR_M = 2.0`: the miss, triangular (the sum of two uniforms, so
  errors bunch near the aim point instead of spreading flat), scaled by how
  badly he crosses. `CORNER_ERROR_MAX_M = 3.2`: a hard cap, so the ball always
  lands within reach of somebody.
- **THE CAP IS A CHOICE AND IS SAYS SO.** Measured, a cross that misses by more
  than ~5.5 m is contested by **NOBODY** — a 1.3 s flight gives a man 5 m away
  no time to get under it, so the duel simply drops. Capping below that cliff
  keeps corners playable rather than turning bad deliveries into dead balls.
  It is recorded as a choice, and the underlying cliff is still unexplained.
- **CALIBRATION, TWO PASSES.** First pass (`ERROR_M = 3.0`, cap 4.2) gave
  attacker-won **33% and 18%** — an overcorrection. Backing off to 2.0/3.2
  gives **40%, 57%, 20%**.
- **NET RESULT, 3 seeds in 3 separate processes:**

  | | attackers won the corner aerial |
  |---|---|
  | before this work | **93–100%** |
  | after the aerial fix + arrival residual | ~53% |
  | **after the delivery error (now)** | **~39%** (40 / 57 / 20) |

  Real football: roughly **30–40%** of corners are won by the attacker, with
  the defence clearing the large majority. That band is now reached, from a
  starting point 60 points above it.
- **STREAM: THIS SHIFTS THE MATCH RNG, DELIBERATELY.** Two `random.uniform`
  draws became four. Corner counts moved too (14/8/13 before, 5/7/10 after)
  and that is **stream noise, not a football effect** — the same matches are
  not reproducible run to run. Do not read the corner-count drop as "the
  delivery change removed corners". Accepted because this is an intentional
  change to the football; the root fix for comparability is seed
  reproducibility, not another parity shim.
- **STILL OPEN:** (a) the keeper wins **0** corner aerials across every seed —
  real keepers claim or punch a meaningful share, and `gk_wins` is computed but
  the keeper is offered as a defender and never arrives; (b) **short corners
  are still absent** — a quick pass to a nearby player who crosses, or a
  rehearsed circulation with the cross disguised, which is real and which would
  attack the same bias harder because the defence has committed to a near-post
  runner; (c) the **first man on the flight line** — the one positioning change
  the probe showed to be decisive (defender at (99.0, 22.0) → DEFENDER,
  *controlled*) is still not implemented; (d) the unexplained sweep where a
  defender 3 m nearer the ball still loses.
- **ENERGY / BALL PHYSICS remains unaddressed and remains the right next
  workstream** — not for this bias, which was never energetic, but because
  corner headers still resolve through the probabilistic
  `GoalkeeperEngine.evaluate_save` while open-play shots use a real
  `aim_shot_flight` + `resolve_shot`.
- **SUITES:** `tests/test_set_piece_routines.py` 23 passed.
  `tests/test_cross_detector.py` was **started and abandoned on timeout** —
  it is match-level and the user's CPU was committed elsewhere, so it is
  UNRUN, not passing. Neither is the 26/27 regression. **This pass is not
  gated**; run both before trusting it.

## CORNER OUTCOMES — what a corner now produces, and THREE wrong probe keys

Asked whether clearances have risen — headed ones especially — and whether
corners themselves should rise now that there is a real aerial contest.

- **THE LABEL IS `metadata["headed"]`, AND I READ THE WRONG KEY THREE TIMES.**
  `1.` `ball_aerial` — **not a `MatchEvent` field at all**. `2.`
  `clearance_kind` — a real helper `_clearance_kind(ball_aerial, ball_z)`
  returning `"headed"/"foot"` exists at `event_chain.py:9026`, but the
  set-piece chain **never calls it**, so no event carries that name;
  `3.` `headed` — **this is the field, and it was there the whole time.** The
  first run reported "0 headed clearances out of 25", which reads like a major
  football finding and was a probe reading the wrong key. Sixth instance of the
  audit's standing shape: *a column that reads cleanly is not a column that is
  right.* Note also that a helper existing is not evidence it is called — read
  the CALL SITE, not the helper.
- **COUNTING THE WHOLE MATCH HIDES THE CORNER.** A match has ~25 clearances
  and a corner accounts for at most one, so the corner's contribution was
  invisible in a match-level count. `_diag_corner_outcomes.py` now reads the
  `SetPieceChain` result directly and counts only what the CORNER produced.
- **THE A/B IS NOW LEGITIMATE, AND IT WAS NOT BEFORE.** `--arm off` zeroes the
  lead, the execution error and the arrival residual, which reproduces the
  pre-work behaviour ("aimed at the receiver's own feet, and he stops exactly
  on his mark"); `--arm on` ships. This is only trustworthy because matches now
  replay byte-identically across processes (see *REPRODUCIBILITY*), so the arms
  differ ONLY by those constants. Every earlier per-arm corner comparison
  carried hash-order noise and should not be quoted as a delta.
- **CAUTION ON PREVIOUS CORNER COUNTS.** The same seed 31 was measured at 6
  corners in one run and 12 in the next, days apart in code — that is code
  changing under the measurement, not noise. **Do not compare a corner count
  across a code change without re-running both arms.**
- **THE MULTI-SEED A/B WAS LAUNCHED AND THEN KILLED. DO NOT QUOTE IT.** The
  8-seed `_diag_corner_multi.py` run (arm ON; arm OFF never started) was
  backgrounded and terminated unfinished at the user's instruction — their CPU
  was committed elsewhere and they had already declined long runs. **There is
  no multi-seed corner-outcome number.** The only figures that exist are the
  single-match `on`/`off` pair below, and they rest on **3 and 4 corners**:
  clearances 25% → 67% of corners, headed clearances **1 in both arms**,
  shots from corners **3 → 0**, corners 4 → 3. Treat all of those as
  unproven directions, not results.
- **LESSON, and it is about conduct not code: I launched a long job and asked
  permission afterwards.** The job was backgrounded first and the "tell me when
  your CPU is freer" line came second, which is taking the decision and seeking
  cover for it. The user's instruction was to keep runtimes short and their
  machine was busy; a background job is still a load on their CPU. **Ask
  before scheduling, not after.**

## CORNER STEP 6 — THE FIRST MAN (2026-10-04)

Every other defender in `_corner_box_occupancy` is marked GOAL-SIDE of his man
(`at(depth - 1.4)`). That is correct for a ball contested AT A POINT and wrong
for a corner, because the delivery travels from the corner flag inward —
goal-side is **further along the ball's journey**, so every marker arrives
second by roughly the marking distance. The probe had already shown a defender
at `(99.0, 22.0)`, out on the flight line, winning *controlled* while the same
man tucked 1.4 m behind his attacker does not.

The first man is the defender who goes and **meets the ball**, placed on the
line from the corner flag to the delivery target, `FIRST_MAN_GOALSIDE_M = 1.0`
toward the goal. He goes into `out`, not into the position engine directly, so
`advance_to_slots` runs him there and `arrived_setpiece_players` offers him to
the duel — a man positioned but never offered changes nothing.

- **THE ORDER IS LOAD-BEARING, AND I GOT IT WRONG FIRST.** Written after the
  marking loops, it **never fired**: 3b pairs every unplaced defender with a
  loose attacker until the pool is empty, so by that point there was nobody
  left to be the first man. Moved before the loops and he claims his spot.
- **`FIRST_MAN_LINE_FRAC` HAD TO BE 0.62–0.78, NOT 0.30–0.42.** The flag sits
  **on** the goal line, so flag = goal-distance 0 and the target is 10.5–15 m.
  At 0.30–0.42 the first man landed **2.9 m from his own goal** — the corner
  quadrant, beside the taker.
- **MY PROBE'S VERDICT WAS INVERTED** and printed "every defender is BEHIND the
  target" while the first man was 5.5 m closer to the flag than the target,
  which is exactly what being a first man means. The flag being on the goal line
  inverts the intuition; the label has to say so.
- **GEOMETRY NOW VERIFIED, both corner sides** (`_diag_firstman.py`,
  milliseconds): designated first man **0.92 m off the flight line, 8.2–8.3 m
  from goal, in front of the target, on the pitch, 23 m from the flag.**
- **ONE MATCH, ONE SEED, AND IT MAY HAVE OVER-CORRECTED:**

  | seed 31 | attacker won the aerial |
  |---|---|
  | before the first man | 40% |
  | **after** | **14% (1 of 7 corners)** |

  Real football is **30–40%**, so this is now BELOW the band, where before it
  was above it. **One seed and seven corners is not a result** — it is the same
  sample size that made the earlier 3-vs-4 comparison unreadable, and I should
  not have run a single match and called it either way. What is established is
  the GEOMETRY; what is unestablished is the calibration. `FIRST_MAN_GOALSIDE_M`
  and the frac band are the two knobs, and the honest next step is a multi-seed
  sweep to find where 30–40% actually sits, not another single match.
- **STILL OPEN from step 4:** the keeper wins **0** corner aerials in every
  seed; short corners are absent; the unexplained sweep where a defender 3 m
  nearer the ball still loses.

## SHORT CORNERS — the routine existed, was SELECTED, and was never READ (2026-10-05)

Asked for short corners: *"short corner kicks dont exist at all, which are most
of the time just quick passes to a nearby player who immediately crosses or
sometimes it starts passes around the area but a cross is just a hidden tactic
in short kicks."* They existed as **vocabulary and nothing else.**

- **`event_chain.py` referenced `SHORT_CORNER` ZERO times.** The enum member
  existed, and it was in **all thirteen style pools**, so it was being *selected*
  routinely — and then played as a low cross aimed at the edge of the box,
  because `corner_delivery(SHORT_CORNER)` returned `target_zone: "edge"` and the
  chain honoured the zone. The word "short" was carried in the name and read
  nowhere. This is the audit's standing pathology in its purest form: the
  mechanism is selected, wired to nothing, and looks present in the source.
- **THE FIX is a flag plus a branch**, because a short corner is a different
  ACTION, not a different target:
  - `"short": True` on the routine's delivery params (and nowhere else;
    `_diag_short_corner.py` asserts exactly that, including that
    `routine=None` is unaffected);
  - **receiver** = the attacker NEAREST THE TAKER by tracked position, not
    `_pick_set_piece_target(zone)`, which picks 28-40 m out in the box;
  - **box NOT packed** — `if is_corner and not is_short`. A short corner exists
    *because* the box is empty; running thirteen men in would leave the short
    pass surrounded by our own attackers with nobody to receive it;
  - **`corner_height = 0.35`** — a ground ball, not a lofted delivery. Lofting
    it would hand the defenders a header they win every time and turn the
    routine back into the cross it replaced;
  - **target** = the receiver's TRACKED position, since `rx, ry` is zone-shaped
    and would aim the short pass at empty grass 30 m from its intended receiver.
- **⛔ RETRACTED 2026-10-05 — "the pool weighting is already correct at 13%".**
  That was **arithmetic about a dictionary, not a realised rate.** `SHORT_CORNER`
  is 7 of 52 pool entries (13%), which says nothing about how often the routine
  is chosen: `corner_routine_for` selects by a **chunk rotation**
  (`test_corner_routine_rotates_across_chunks`), not a proportional draw, so
  pool membership is not frequency. **Measured over 4 seeds / 33 corners, short
  corners fire at 48%** — against a real 10–20%. So honouring the routine
  *created* a new distortion: previously the routine was selected and ignored
  (visually neutral); now it is selected and played, at roughly 4x the real
  rate. **This is the first thing to fix, and the pool arithmetic must not be
  used to justify it again.** Note the style differentiation (`tiki_taka` 50%,
  possession styles 25%, every direct style 0%) remains a real property of the
  pools; it simply is not the realised rate.

- **⚠️ A BEFORE/AFTER ARM WAS BUILT BY MUTATION AND PRODUCED AN INVALID
  BASELINE.** `_diag_corner_ab.py` reconstructs the pre-work state by toggling
  six things (old aerial selection, zero arrival residual, zero delivery
  lead/error, no first man, no short corners). It cannot use `git stash` —
  `HEAD` is thousands of lines behind and a parallel session shares these
  files. The **before arm returned 0 on everything**: 0 clearances, 0 shots, 0
  recoveries across 27 corners. **Zero across every outcome is a broken engine,
  not a football result.** The suspect is `box_dead_first_man`, written in a
  hurry: it indexes names with `getattr(n2, "position", "")` where `n2` is
  already a string, and pops an arbitrary defender, so the duel never produces
  a winner and both `att_wins` and `def_wins` stay false.
  **Do not quote that arm, and do not rebuild it by mutation again** — a
  before-arm assembled from six edits measures the edits. `git stash` is unsafe
  here; the honest options are to revert the specific hunks by hand from a
  byte-exact backup, or to accept a partial before-arm with the un-reverted
  feature stated.
- **Measured, one match, seed 31** (`_diag_short_corner_match.py`): 27 set-piece
  chains, of which 8 actual corners — `far_post_outswing` 3, `penalty_spot_crowd`
  2, **`short_corner` 3**. The routine is **stamped in the `CORNER_TAKEN`
  metadata**, so the exporter and the chance ledger can see it; that was not
  true before and is what makes the mechanic countable at all.
  - **3 of 8 = 37% is NOT evidence of over-frequency.** Against a 13% rate,
    P(X>=3 | n=8) is about 0.30. One seed, eight corners. Do not "correct" the
    pool weight on this.
- **PROBE ERROR, the clean-wrong-table-above-a-right-one shape again.**
  `_diag_corner_pool.py` compared a bare `"SHORT_CORNER"` against
  `SetPieceRoutine.SHORT_CORNER` members and printed **0% for every style** — with
  the correct 13% aggregate sitting directly underneath it. A wrong table above
  a right number is the most dangerous layout a probe can have.
- **NOT VERIFIED, then MEASURED — see SHORT CORNERS: THE MECHANIC BELOW.** The
  routine is honoured but was not yet a short pass; three further defects had
  to be found before it was one.

## SHORT CORNER MECHANIC — three defects between "flagged" and "short" (2026-10-05)

The questions: is the short pass lost, is it worked back into the box, and how
does a 0.35 m duel resolve. Answered with the **routine FORCED**
(`_diag_short_corner_q.py`), because at 13% of ~8 corners a match yields roughly
ONE short corner and ten would need ten matches. **The forced arm measures the
MECHANIC, never the frequency** — `_diag_short_corner_match.py` measures the
unforced rate, and the two must not be quoted interchangeably.

- **DEFECT A: the height clamp was not routine-aware.** `corner_height = 0.35`
  was set, and **the next line** was
  `corner_height = max(2.0, min(3.0, ...))` — overwriting it. Measured with the
  flag correct: apex **3.4–9.1 m**. The clamp exists to keep a CROSS inside
  players' vertical reach, which is real and is kept, for crosses. A short pass
  has no such constraint.
- **DEFECT B: the receiver was 18–34 m away.** Picking "the attacker nearest the
  taker" is correct football and useless here, because in this engine the whole
  team is 20–40 m up the pitch. Measured pass length **17.7–34.2 m** against a
  real 8–18 m. The routine was "short" only in its name.
- **DEFECT C, the real one: NOBODY CAME SHORT.** The fix is that the receiver is
  **run to a support slot** 13 m from the flag (`SHORT_CORNER_SUPPORT_M`) via
  the same `advance_to_slots` the box uses. That is what a midfielder actually
  does — come short, offer, receive on the half-turn — and it is *why* a short
  corner works in real football and would not work here otherwise: it requires
  someone to have moved.

### THE THREE ANSWERS, forced arm, seed 31, 13 corners

| | before (flag only) | after (a real short corner) |
|---|---|---|
| pass length | 17.7–34.2 m | **11.3–12.7 m** (real 8–18) |
| target height / apex | 0.35 / 3.4–9.1 m | **0.35 / 0.67–0.81 m** |
| flight duration | 0.94–2.64 s | **0.61–0.70 s** |
| attacking side retained it | 8/8 = **100%** | **3/13 = 23%** |

- **Q1 — YES, IT IS USUALLY LOST.** 23% retained; `BALL_RECOVERY` 10/13. A
  short corner that worked 100% of the time was the tell that it was not short:
  an 18–34 m delivery is a cross, and crosses are won by the attack.
- **Q2 — SOMETIMES WORKED BACK IN.** 0.38 shots+goals per corner, so roughly one
  in three produces an attempt. That is the whole point of the routine and it is
  now plausible rather than 1.0+ per corner.
- **Q3 — THE DUEL IS CONTESTABLE, but it resolves as `drops`, not an
  interception.** 10 of 13 duels return `outcome="drops"` — no contestant
  satisfied `arrival <= time_s` — which the chain then turns into
  `BALL_RECOVERY` for the defence. The effect is right (the defence gets the
  ball) but the label is not: a dropped ball and an intercepted pass are
  different events, and the resolver never distinguishes them at 0.65 s over
  12 m, where only men within a metre or two can arrive in time.
- **⚠️ MY OWN PROBE MISLABELLED A COLUMN, and it is the one that would have
  carried the claim.** It printed `contestants QUALIFIED (could arrive)` from
  `len(attackers) + len(defenders)` — which is the number **OFFERED**, not the
  number that qualified. So "3 contestants qualified, 0/13 with ≤2" is a
  statement about nothing. The real figure is the outcome distribution above.
  A column whose heading asserts a stronger claim than its expression computes.
- **THE SHOT MAP / UNFORCED RATE HAS NOT BEEN RE-CHECKED** since defects A-C
  were fixed; the 13% pool weighting is unchanged by them but the *behaviour*
  of those 13% is now entirely different from what it was when that rate was
  measured.

- **TESTS, and why they are a NEW FILE.** `tests/test_set_piece_routines.py`
  **23 passed** — and is worth nothing against this work. It asserts
  `corner_delivery(SHORT_CORNER)["target_zone"] == "edge"` and nothing else:
  no `short` flag, no delivery height, no support run. **All three defects sail
  straight through it.** A suite that is green and silent about a feature is
  not evidence about that feature. So `tests/test_short_corner.py` (13 tests,
  **36 passed** with the routines file) pins the flag, the ground height, the
  support distance, that the chain actually READS `is_short`, and that the
  packed-box branch stays gated on `not is_short`.
- **THE GUARD THAT EARNS THE FILE IS AN AST ONE, and it needed two bugs in
  ITSELF before it worked.** Defect A was "the line I wrote was correct and
  the next line undid it", which no behavioural test on `corner_delivery` can
  see — the clamp lives in the chain. So `test_the_height_clamp_is_not_applied_
  to_a_short_pass` asserts the `[2.0, 3.0]` clamp is lexically inside
  `if not is_short:`, and it asserts **non-vacuity first** (that a clamp
  matching that shape still exists at all), so a rename cannot quietly turn the
  guard into a no-op.
  - **Bug 1: it walked `ast.iter_child_nodes(node)`, which walks DOWN.** It
    searched the clamp's descendants for an enclosing `if`, found none, and
    reported the clamp unguarded. The test failed on a *correct* pass.
  - **Bug 2: it re-parsed inside the helper**, so it compared node *identity*
    across two different trees — which can never match. Fixed by parsing once
    and caching. **Node identity only means anything within one parse.**
  - **Verified by negative control** (`_diag_short_guard.py`): restoring defect
    A makes the guard go **RED (1 failed, 12 passed)**, and the file is
    restored **byte-identical** (asserted, 533,419 chars). A green guard that
    has never been seen to fail is not a guard.

## KEY PASSES AND ASSISTS (2026-10-02) — the coordinates were real; the CAUSATION is not

Asked: "key passes / chance-creation events had end points but not start
points — a key pass and assist should always follow shot and goal with true
tracked coordinates." Measured on real matches before changing anything
(`_diag_chance_coords.py`). The rumour was right about the symptom and wrong
about the object — and the true cause is much worse than either.

- **ORDINARY PASSES WERE FINE.** `PASS` 655/680 carry a real destination and
  `secondary_player` = receiver. "Passes have no start point" is false.
- **THE `CHANCE_CREATED` ORIGIN WAS FABRICATED.** 15/15 sampled events had a
  start→end length inside [5, 20] m, which is literally
  `location_x = x - random.uniform(5, 20)` at `event_chain.py:5655` — a draw
  from the *global football RNG*, matching no pass that player ever made (0/15
  against every real pass origin by that player). Its END was the shot's taken
  location, so the pair asserted "the key pass ended exactly where the shot was
  struck", which is wrong whenever the receiver carried. That carry objection
  was the right one and the engine models carries explicitly elsewhere.
- **SET-PIECE DELIVERIES HAD NO END AT ALL.** `CORNER_TAKEN` 0/16,
  `FREEKICK_CROSS` 0/48. So the key passes the honest ledger actually finds
  were precisely the ones with no endpoint.

FIXED, in dependency order:

1. **The fabricated origin is gone.** `CHANCE_CREATED` now takes its start from
   `PositionEngine.tracked_position(creator.name)` — a real tracked coordinate.
   `MatchEvent.location_x` is typed `float`, not `Optional`, so an untracked
   creator cannot be represented as absent in memory; it falls back to the
   shot's own coordinates and stamps `origin_known` / `origin_source` so a
   consumer that ignores the flag sees a zero-length key pass rather than a
   plausible lie. New `PositionEngine.tracked_position()` exists because
   `get_position` answers (50.0, 34.0) — the centre spot — for a player it has
   no state for, which is a coordinate that reads as real. (This is the same
   trap that made `_diag_wall_shape.py` measure nothing.)
2. **The invented `record_touch(creator.name, x - 8, y)` is gone** (5632). It
   planted the assist-giver 8 m behind the shooter *and wrote it into the
   position engine*, so every downstream position read inherited the fiction.
   The creator is credited at his own tracked position, or not at all.
3. **Set-piece deliveries now carry a real destination and a real receiver.**
   `CORNER_TAKEN` is emitted without an end (the target is not known until the
   flight is built, the contact point not until the aerial resolves) and both
   are stamped onto the same event afterwards: the `aerial.contact_point` if
   anyone got to it, else the aimed target, plus `secondary_player` = the man it
   was aimed at, which is what lets the ledger link a corner to a shot by that
   player instead of guessing. `FREEKICK_CROSS` inherits the endpoint from the
   sub-chain's corner event (whose duplicate it discards). 0→100% on both types.
4. **A crossed free kick was being struck from a CORNER ARC.** `delivery_x /
   delivery_y` was computed at 6761 and documented as the Law 11 origin, then
   the flight ignored it: `make_ballistic_flight(Vec3(corner_x, corner_y, ...))`
   with `corner_y = random.choice([1.0, 67.0])`. A crossed free kick was
   offside-judged from the foul spot and struck from a *random* corner flag,
   and `corner_side` (which picks the taker, the marking grid and the
   in/out-swing sign) read off `corner_y` rather than the delivery spot. The
   two were in different frames at once — the single worst kind of bug, and it
   is invisible for a real corner because the values coincide. Real corners
   are byte-identical; the `random.choice` is still drawn unconditionally so
   the stream is unchanged.

MEASURED, 2 real matches, before → after:

| | before | after |
|---|---|---|
| ledger shots with a creator | 10/47 (21%) | 27/63 (43%) |
| zero-length key passes | 9/10 | **0/27** |
| `CORNER_TAKEN` with a destination | 0/16 | **100%** |
| `FREEKICK_CROSS` with a destination | 0/48 | **100%** |
| all deliveries with a destination | 92% | **97%** |
| `CHANCE_CREATED` origin inside the [5,20] draw band | 15/15 | 3/19 |
| pass-end → shot-origin gap | median 30.4 m | **median 4.4 m** |

The 4.4 m gap is the carry, measured. That is the user's objection answered
positively rather than argued away.

- **A FRAME BUG FOUND WHILE MEASURING, LARGER THAN THE ONE ASKED ABOUT.**
  Wrong-end shots (a shot taken on the shooting team's OWN half) went from
  **6/36 and 4/24 to 0/42 and 0/22**. Cause: `ChainDispatcher.set_piece` is
  declared `attacks_right: bool = True`, and `match_engine.py:4603` (the CORNER
  site) **omitted it**. Four of the five `set_piece` call sites already passed
  it; only the corner did. So every away-team corner was resolved as if that
  team attacked right: box grid, taker, delivery target, contact point and any
  loose-ball follow-up were all placed at the away team's own end. The
  bimodality was unmistakable — Play City's median shot x was exactly **85.0**,
  the hard-coded clamp. Fixed at 4603. This is the pre-existing corner-chain
  bug AGENTS.md recorded as "NOT FIXED" under *SHOT MAP TRAJECTORY*; it is
  fixed now and that note is stale.
  Also fixed: `BaseChain.clamp_attack_x()` — five sites clamped a live-frame x
  straight into an attacking-right band (`max(85.0, min(102.0, x))`), which
  drags an away-team contact at x=15 to x=85. Attacking-right behaviour is
  byte-identical.
- **A CLAIM I MADE AND HAD TO RETRACT.** I first read this as "`_shot_location`
  resamples a fresh coordinate and discards the ball's position". Wrong: with
  a position engine attached the shot is taken from the SHOOTER'S TRACKED
  position (5606-5613) and `_shot_location` is only the no-position-engine
  fallback. The coordinates were never the problem on that path.
- **THE REAL FINDING, and it is not a coordinates problem: THERE IS NO
  PER-PLAYER POSSESSION STATE.** `MatchState` records `possession_winner` (a
  TEAM) and `last_ball_x` (a coordinate). It never records WHICH PLAYER has the
  ball — that identity does not exist at this layer. So every chain that needs
  it re-derives it as a weighted draw over the squad: `_pick_shooter` weights
  by role (ST 6.0, CF 5.5, LW/RW 4.0, CAM 3.0, CM 1.5, CDM 0.5, CB 0.3,
  GK 0.0) × distance to the ball anchor; `_pick_creator` is the same idea. The
  man who actually received the ball is not an input, so he can be neither the
  shooter nor the assister. Measured goal run-up: ball carried to **3.4 m from
  the goal line by Allan Rodgers**, then **Tim Jonasi? shot from 25.2 m** 22 m
  away, credited to **Segan Ealong**, who appears nowhere in the sequence except
  in the fabricated marker. Second goal in the same match: credited assist
  Jony Rigann, whose last real event was a *tackle*.
  - This single gap explains every symptom in this section. The assist is
    fiction because possession is not tracked. The ledger cannot link a key
    pass to a shot because the shot is not caused by a pass. The
    `CHANCE_CREATED` end had to be *forced* onto the shot location to make the
    pair look coherent. And the ledger attributing 0 of 4 goals is the ledger
    being correct, not failing.
  - `chance_creation.py`'s own docstring claims the engine "used to" fabricate
    chances and that the ledger replaced it. The ledger is genuinely correct
    and genuinely unused for the causal question; the fabrication upstream is
    still there.
  - `TransitionChain` proves the shape is already known to be right: its solo-run
    path passes `att_players=[carrier]`, so the carrier shoots. The general
    open-play path has no such input.
- **STILL OPEN, measured, not fixed: shot-distance distribution.** Over all
  shots in the corrected frame: median 20.0 m (real PL ~17), but **14.8% inside
  4 m** (real ~0%) and **22.2% beyond 25 m** (real ~3%). `_shot_location`'s
  open-play draw is `min(32, -14·ln(1-u) + 3)` — a log-uniform tail that
  over-feeds long range, and the settle-from-range branch (30%, 20-28 m) adds
  more. Separate calibration question from the causation gap; do not conflate.
- **THE CAUSAL-SHOOTER FIX: `shoot_player` WAS ALREADY THERE AND WAS BEING
  DROPPED.** `PossessionChain` sets `result.shoot_player = last_player.name`
  at all three SHOOT sites (1645/1738/1858) — the engine has always known who
  decided to shoot. `match_engine.py:4989` passed `shoot_x`/`shoot_y` and
  dropped it, so `AttackChain` fell through to `_pick_shooter`, a role- and
  distance-weighted DRAW over the whole squad. Threaded now: `shooter_name`
  on `ChainDispatcher.attack` → `AttackChain.generate`, resolved by a new
  strict `_named_shooter` (must be an outfielder IN the squad passed in; a
  stale name returns None so it falls back rather than shooting with a ghost).
  Measured effect on a real goal run-up, before → after:
  - before: ball carried to (3.4, 35.8), a DIFFERENT player shot from
    (25.2, 42.8), credited to a third who appeared nowhere in the sequence.
  - after: `PASS Nashan Gwum -> Danso Potwemi`, `CARRY Francis Sesina` ×3,
    `SHOT_ON_TARGET Francis Sesina (5.3, 39.3)` — received, carried, shot.
    The scorer is now the man who had the ball.
  Note this also makes 4 of 5 measured goals correctly UNASSISTED: the scorer
  won the ball and dribbled it in, so there is no pass to be the key pass. A
  ledger that refuses to invent one is right, not broken.

- **THE ENGINE ASSIST IS STILL FABRICATED AND STILL SHIPPED.**
  `result.goal_assistant` = `_pick_creator` (weighted draw), written to the
  Goals sheet via `g.secondary_player` and to `alltime_db` ASSISTS. Making it
  honest needs the possession state above; until then the two sources cannot
  agree and the Goals sheet is not the ledger.

- **OPEN, AND I COULD NOT MEASURE IT CLEANLY IN THREE ATTEMPTS.** The obvious
  follow-up — "does the shot's recorded origin agree with where the shooter
  actually stood?" — needs the position engine sampled AT the shot.
  `_diag_shot_origin.py` tried, and rejected, three pairings:
  1. reading `position_engine.tracked_position` after `simulate()` returned —
     that is the FINAL WHISTLE position, so it reported a fictional "median
     39.5 m disagreement" that was pure end-of-match drift;
  2. keying snapshots by `(minute, team)` — a minute holds many sequences, so
     three shots in 11' all inherited one overwritten snapshot;
  3. keying by "the last AttackChain call before this shot's clock" — set-piece
     shots never reach AttackChain, so it returned an unrelated earlier
     possession, and reported "the shooter was at (50.0, 7.4)", a FORMATION
     ANCHOR rather than a live position.
  Version 4 tags the actual event objects by `id()` (the engine appends those
  same objects, so it is an exact join) and that DOES fix the man-vs-sequence
  confound. What it cannot yet settle is the BALL, because parking
  `shoot_x/shoot_y` for the following AttackChain call is keyed by TEAM, so a
  chain resolved between the two claims can attach a stale ball to a later
  shot. Until that handoff is keyed by sequence, treat any
  "|shot − ball|" number from this probe as unproven. Do not quote one.
- ~~**STREAM PARITY SHIM — DO NOT DELETE.**~~ Removing the fabricated origin
  also removed a `random.uniform(5, 20)` draw, and that shifted every later
  number in the global stream. Restoring that one discarded line turned
  `test_match_crosses_stamped_geometrically` from fail to pass with no
  behavioural change. It is named `_STREAM_PARITY_DRAW` and is assigned-and-never-
  read. **It kept itself alive on a premise that has now been retracted**: the
  stated reason it could not go was "the real fix is making a match reproducible
  from `random.seed`, which module-level brain/mind caches prevent". Matches are
  reproducible — see *REPRODUCIBILITY* — so the shim is no longer load-bearing
  for that reason. It is STILL an open decision, not a clean removal: it holds
  the historical stream, and every calibration number in this file was taken on
  the stream it preserves, so deleting it invalidates all of them at once. Do
  not delete it without deciding that deliberately.
- **THE DIRECT-FK WALL IS NOT A GROUND-PLANE ARTIFACT — my hypothesis was
  wrong, and it was wrong in a way worth recording.** I had assumed (and
  written down as an open question) that `MovingPlayer` was a 2D position with
  a scalar radius, so a wall man would have no jump term and the measured
  27–67% block rate would be a flat interception rather than a real wall.
  Checked: `MovingPlayer` is a full 3D body — `jump_height`,
  `standing_reach`, `dive_vertical_reach`, `body_mass_kg`, `balance`. And
  `resolve_shot` USES it. `geometry_engine.py:1245` calls
  `blocker.vertical_reach(point.z > 1.15)`, which is `min(2.65,
  standing_reach + jump_height)` airborne and `standing_reach` grounded, so
  the top corner of a direct free kick IS contestable over the wall by a man
  who jumps. Blockers also MOVE: reaction `reaction_time * 0.55`, then a
  block effort of `top_speed * (0.72 + 0.18 * balance)` toward the ball's
  current horizontal position, re-evaluated every tick of the flight, and the
  body test is `tackle_radius + 0.15` plus a `time_to_reach` feasibility
  check. So the wall is genuinely contested in three dimensions and there is
  nothing to build here. Do not "add a body to the wall".
- **SUITES:** `test_set_piece_routines` + `test_chance_creation` +
  `test_plofa_export` 59 passed / 2 strict xfailed. `test_cross_detector`
  23 passed (the stream-sensitive one — it survived the frame fix). New
  `tests/test_chance_provenance.py` 26 passed, incl. an AST guard asserting
  every `set_piece` call site in `match_engine.py` passes `attacks_right`
  (an invariant enforced by AST rather than by convention, because the default
  is `True` and one omission put every away-team corner at its own end).

## ASSIST CAUSATION + A DEAD GOAL PIPELINE (2026-10-02, second pass)

The user chose **true tracking** over leaving the fabricated assist, with the
observation that he had believed key passes were real since matchday 1 and they
were not. He was right about the symptom and wrong about the object, as before.

- **THE ASSIST IS NOW THE REAL PASSER.** `ChainResult.shoot_assister` records who
  passed to `shoot_player`, `""` if nobody did. It cost nothing to obtain: the
  passer is the PREVIOUS value of `last_player` at every carrier change, and
  `event_chain` has exactly FOUR assignments to `last_player` — three of them
  pass-derived (completed pass 2832, through ball 3107, attacker winning a cross
  ~3467) plus the sequence start. Crediting the **cross taker** on a headed goal
  is what the Laws award, so the cross case is not a special case.
  `_resolve_assister` requires the name to resolve to an outfielder in the squad
  and forbids it equalling the shooter.
- **`_resolve_assister` HAS NO FALLBACK, ON PURPOSE. DO NOT ADD ONE.** The old
  behaviour was `creator.name if creator else ""`, a role- and distance-weighted
  RANDOM DRAW over the whole squad. A goal by a man who received it from a
  team-mate is assisted; a goal by a man who won the ball and dribbled it in from
  40 yards is UNASSISTED, and that is common. **Expect FEWER assists than
  before** — 4 of 5 measured goals now come out correctly unassisted. That is the
  honest answer, not a regression, and it is the single most likely thing for a
  future agent to "fix" by reintroducing a fallback.
- **A FIXED FIELD IS NOT A FIXED FEATURE — I made this mistake myself.** I made
  `result.goal_assistant` honest and stopped. The **GOAL event's own
  `secondary_player`** — which is what the Goals sheet Assist column and
  `alltime_db` ASSISTS actually read — was still `creator.name if creator else
  None`. The feature the user asked for was still shipping the fabrication.
  This is the identical trap as the shot origin (`CHANCE_CREATED`) and of
  `world.ingest`'s `sub_controller`: **after changing a field, follow it to the
  consumer before claiming the feature works.** A field nothing reads is not a
  fix.
- **`_pick_creator` STILL SHIPS** — it is no longer the assist, but it still
  supplies `CHANCE_CREATED`'s `player` field (and `creation_type`). Removing the
  assist draw did not remove the creator draw.
- **THERE WAS A SECOND, ENTIRELY FABRICATED GOAL PIPELINE, AND IT WAS DEAD.**
  `match_engine._simulate_shot_sequence` picked the shooter as a random member of
  `['ST','CF','LW','RW','CAM']`, the creator as another random player, and the
  location by SAMPLING A ZONE NAME (`_shot_location(zone, ...)`) with no reference
  to any player or to the ball. `_register_goal` then called `_shot_location`
  TWICE — once for `location_x`, once for `location_y` — so the GOAL event's
  coordinate pair was a chimera of two independent draws. Measured over 2 real
  matches (`_diag_shot_pipeline_split.py`): it fires **0 times**, and the repo has
  **zero call sites**. So it was corrupting nothing — but it is the project's
  standing pathology in its worst form: a mechanism that reads as authoritative,
  is wired to nothing, and would fabricate goals the moment anyone routed to it.
  **Removed (265 lines) with `_register_goal` and `MatchEngine._shot_location`.**
  `event_chain._shot_location` is a SEPARATE, LIVE method (the no-position-engine
  fallback) and is untouched. Do not re-add or route to the engine one.
- **THE TWO `random.uniform` SITES THE USER ASKED ABOUT ARE NOT LOCATIONS.**
  `_STREAM_PARITY_DRAW` (the name is `_DRAW`, not "PROBE") is **assigned and never
  read** — pure RNG-stream ballast, so deleting it changes every later number in
  the match. `rx + random.uniform(1.0, 4.0)` starts from the receiver's REAL
  `position_engine.get_position(...)` and nudges him 1–4 m further toward goal so
  the cross is aimed where his run takes him. Same idea as the shot's
  `strike_pocket = random.uniform(0.5, 2.5)`. A small jitter around a real
  position is not a fabricated location.
- **OPEN AND UNVERIFIED — DO NOT QUOTE A FIGURE FOR IT.** The cross-receiver
  clamp at `event_chain.py:3404` reads the receiver's real x, clamps it into
  `[85, 100]` via `clamp_attack_x`, and then `record_touch`es the RESULT. A winger
  standing at x=40 is therefore written at x=85 and the gap is banked as distance
  covered. Aiming a cross inside the box is right; persisting the clamp as the
  man's position is not. `_diag_teleport_moves.py` measures the population —
  **1,040 single-call `record_touch` jumps ≥12 m in one match, 27,333 m total
  (~260 pitches)** — attributed 779 `unknown` (the engine's own restart /
  goal-kick / corner repositioning, legitimately discontinuous) / 195
  `possession` / 55 `set_piece` / 11 `attack_chain`. The clamp lives in
  `possession` so it is inside that 195, but **it was NOT isolated**, and this is
  the fourth rejected probe pairing of the session. Measure it before touching it.
- **HOW I BROKE `match_engine.py` — read this before deleting any line range.**
  To find where `_shot_location` ended I scanned for the next line matching
  `^    def `. That regex only sees METHODS. `match_engine.py` has a
  `class MatchResult:` at **column 0** after `MatchEngine` ends, and the scan
  sailed 85 lines past the method, through the module banner, `_terrs`, `_pcts`
  and the `@dataclass class MatchResult:` header. Everything after was left intact
  but **silently re-parented onto `MatchEngine`** — which is still valid Python,
  so `import match_engine` SUCCEEDED. The only symptom was
  `NameError: name 'MatchResult' is not defined` from deep inside `simulate()`.
  Worse, a duplicate `class MatchResult` then appeared further down the file and,
  being later, **shadowed the repaired one** — so the repair looked applied and
  still failed.
  **RULES, learned the hard way:**
  1. Never bound a deletion by a regex over `^    def `. Parse with `ast` and use
     `node.end_lineno`, or delete by matching the exact source text.
  2. Assert the class structure AFTER deleting (`ast` → `t.body` → count
     column-0 `ClassDef`s, and check the intended one exists).
  3. Deleting bottom-up while asserting against indices computed BEFORE the first
     `del` is how you get asserts that were true when written and false when used.
  4. A successful `import` proves nothing about class structure in this file.
- **`PlayerSpatialState.position` IS THE POSITION LABEL ("ST"), NOT A POINT.**
  Coordinates are `current_x` / `current_y`. Together with `get_position`
  returning `(50.0, 34.0)` — the pitch centre — for a player it has no state for,
  this is the single most productive trap in the spatial layer: both read as a
  real coordinate and neither is one. Use `tracked_position(name)`, which returns
  `None` for an untracked player.
- **`pass_direction` metadata — RETRACTED, THE CODE IS CORRECT.** Re-measured
  2026-10-02: **98.4% agree (1,647/1,674)**. The original "~2/3 wrong" figure is
  what you get by failing to flip the sign for the away side; "sideways" was the
  only category that agreed because a `|dx|` band cannot be affected by a sign
  error. Full evidence in the MEASUREMENTS section above. Do not change it.
- **STILL OPEN, unchanged:** the shot-distance distribution is still
  miscalibrated (median 20.0 m, 14.8% inside
  4 m against a real ~0%, 22.2% beyond 25 m against ~3%); and a match is still not
  reproducible from `random.seed`, which is the only reason
  `_STREAM_PARITY_DRAW` has to stay.
- **SUITES (2026-10-02, second pass):** `tests/test_set_piece_routines` +
  `tests/test_chance_creation` + `tests/test_plofa_export` + `tests/
  test_cross_detector` re-run green after the `MatchResult` repair and the dead-
  pipeline removal; `tests/test_chance_provenance.py` 26 passed. The full 26/27
  regression (`tests/tests.py`, `test_chronography`,
  `test_possession_causality`, `test_checkpoint7_subsystems`) was launched
  BEFORE the `secondary_player` fix, so it does **not** gate that fix — re-run
  it before claiming the second pass is clean. Expect the one known
  pre-existing failure, `tests.py::test_pass_matrix_sums_match_real_events`
  (matrix − manual = the number of successful crosses), and treat any OTHER
  failure as new until shown otherwise.
- **COMMIT `e76b644`** carries an explicit NOT-FIXED block in its message. Read
  it before assuming this work closed the class.
- **NEW PROBES:** `_diag_shot_pipeline_split.py` (which pipeline produced each
  shot/goal, by `id()`-tagging the emitted event objects — not by guessing from a
  timeline) and `_diag_teleport_moves.py` (single-call `record_touch` jumps,
  attributed by wrapping each chain's `generate` to push a context label).
  `_diag_chance_coords.py` is the primary chance-provenance probe and exports
  `build_pair(home, away)` that the others import. `_diag_shot_origin.py` is v4
  (`id()`-tagged) and its docstring records the three REJECTED pairings — read
  it before writing a fifth. **Write probes to a FILE, never an inline
  `python -c`**: PowerShell mangles the quotes, and both of my probe crashes
  today were real bugs surfaced by that mangling rather than by the probe.
- **PROBE-SCRIPT RULES (four false findings today, all the same shape):**
  1. `PlayerSpatialState.position` is the position LABEL (`"ST"`), not a point;
     coordinates are `current_x`/`current_y`.
  2. `position_engine.get_position(name)` returns `(50.0, 34.0)` — the centre
     spot — for a player it has no state for. That is a coordinate that reads
     as real and is not one. Use `tracked_position(name)`, which returns `None`.
  3. Reading position state AFTER `simulate()` returns attributes
     final-whistle state to a mid-match moment.
  4. An unattributed jump histogram cannot separate a deliberate placement
     from a clamp. Signature-test the mechanism instead of context-tagging it.
  5. `ChainDispatcher.attack` is a **staticmethod**, not a classmethod, so
     `getattr(f, "__func__", f)` silently returns the plain function and a
     wrapper that passes `cls` shifts every positional by one. It surfaced as
     `"got multiple values for argument 'position_engine'"` — a message that
     points at the wrong argument entirely. Read the descriptor out of
     `ChainDispatcher.__dict__`, which is a `classmethod`/`staticmethod`
     object for both cases.

## CHANCE CREATION FROM REAL TRACKED DATA (2026-10-03, third pass)

> **SEASON DATA BOUNDARY.** The season has played 5 matchdays. MD1–MD5
> chance creation, shot assists and assists are **fabricated**; MD6 onwards is
> the first honest data. Full statement, including what must not be done with
> MD1–MD5: **⛔ SEASON BOUNDARY** further down this file. Do not read the
> before/after tables in this section as a matchday trend.

The user rejected the residue list: *"i dont like this, like you have the
capability of finishing it and making true chance creation counts from real
tracked data"*, with one hard constraint: **do not destroy the relationship
between chances created, shot assists and assists.** So this was finished, not
listed.

- **`_pick_creator` IS DELETED.** `CHANCE_CREATED.player` resolves from
  `assister_name` via `_resolve_assister` + a new `_named_outfielder`.
  `_named_shooter` delegates to the same lookup. Because the key pass's ORIGIN
  is the creator's tracked position, the draw had fabricated the *geometry* as
  well as the name — so this closes both at once.
- **NO FALLBACK, AND THAT IS THE POINT.** An unassisted strike emits **no
  `CHANCE_CREATED` at all** rather than crediting the shooter. Crediting him
  would make the engine disagree with `ChanceCreationLedger._find_setup_pass`
  (which returns None for a dribble) about the SAME shot — i.e. it would break
  the exact relationship the user protected. The shot still appears.
  **Expect fewer assists than before. Do not "fix" that.**
- **THE STREAM SHIM MOVED OUT OF ITS GUARD.** `_STREAM_PARITY_DRAW` was
  inside `if creator and situation != PENALTY:`, safe only because `_pick_creator`
  always returned somebody. With a legitimately empty creator it would
  desynchronise the RNG stream for the rest of the match, so it is now drawn
  unconditionally at the same point.
- **THE REAL GAP WAS THE BALL CARRIER, AND IT WAS NOT WHERE I LOOKED
  FIRST.** `_shoot_assister`/`_shoot_player` are only set at PossessionChain's
  three SHOOT sites, which says nothing about a possession that ends with the
  ball at a player's feet and the shot decided LATER — by MatchEngine's
  per-minute shot funnel (`match_engine.py:5276`), which passes no names at
  all. That is the path that fires the BULK of a match's shots.
  `ChainResult.ball_carrier` / `ball_carrier_passed_by` now report who holds the
  ball and who gave it to him, and the funnel passes both.
- **`_absorb_chain` IS THE SINGLE POINT.** Its own docstring: *"This is the
  single point where chain outputs become match facts."* It is also the only
  place every chain's events pass through, so possession is stamped there from
  the last genuine ball touch — real events, not any chain's bookkeeping.
  `MatchState.possession_team` is a TEAM name and cannot answer "who takes the
  shot"; that is why every chain re-derived it and one of them drew it.
- **MEASURED, 3 real matches, `_diag_creator_truth.py`** (shooter = the man who
  had the ball, per section E, which walks back to the last real touch):

  | | before | after |
  |---|---|---|
  | shooter IS the ball-holder | *(see `_diag_creator_truth.py --no-thread`)* | **103 / 118 (87%)** |
  | engine CHANCE_CREATED events | 2 per 2 matches | 4 per 3 matches |
  | creator REALLY passed the ball | — | **4/4, fabricated 0** |
  | engine vs ledger, same player | — | 4 agree, **0 disagree** |
  | `chances == goal_assists + shot_assists` | — | 54 players ok, **0 broken** |
  | goals with an assist | 0 of 7 | **3 of 9** (6 genuinely unassisted) |

  The 15 remaining non-matching shooters are each printed with the player who
  actually had the ball; they are NOT assumed to be defects, because the
  back-scan's expected-predecessor rule is wrong for set-piece headers (it
  expects the man a corner was AIMED at, who is not necessarily the man who
  won it). Classify before touching.
- **`xA` WAS LAUNDERED BY THE DRAW.** `creation_event.xa = xg` is the correct
  definition, but it was being banked against the *drawn* creator, so it is
  real only now that the creator is. Same for `pass_network`'s (passer →
  receiver) xA.
- **A TEST THAT WAS RIGHT ABOUT ITS SUBJECT STILL HAD TO CHANGE.**
  `test_chance_created_origin_is_tracked_not_drawn` called the chain with no
  names and got an event anyway — because the creator was a draw that ALWAYS
  returned somebody. With no fallback it gets no event. The test now supplies
  the passer, which is what makes it exercise the mechanism its assertion is
  about. Fifth instance of **a fixed field is not a fixed feature.**
- `tests/test_chance_truth.py` (21 tests) pins the creator, the carrier hand-off
  and the three-quantity relationship. It includes an **AST guard that every
  `ChainDispatcher.attack` call site in `match_engine.py` passes
  `shooter_name`/`assister_name`** — a defaulted argument is a silent omission,
  the same reasoning as the `attacks_right` guard in `test_chance_provenance.py`.
- **TEST FIXTURES THAT ASSUME A CARRIER EXISTS ARE LYING.** A possession from
  x=62 against a full defensive block is lost ~82% of the time (measured: 49 of
  60 draws ended in a TURNOVER or MISCONTROL), so `ball_carrier` is correctly
  empty that often. The tests sweep 36 draws and assert both branches rather
  than reading one. Two more fixture bugs found the same way: a single-shot
  timeline that mutated a 3-event helper into 2 shots (so one pass correctly
  counted two chances), and an origin asserted against the coordinate handed to
  `record_touch` rather than read back — `record_touch` applies the wide-role
  flank hold, so an LW recorded at y=12 is stored at y≈7.6.
- **THE FULL CHAIN TO THE WAREHOUSE IS NOW HONEST END TO END** — this is what
  "true stats" actually required, and each link was checked rather than assumed:
  `PossessionChain.last_passer` → `ChainResult.shoot_assister` /
  `ball_carrier_passed_by` → `ChainDispatcher.attack(assister_name=…)` →
  `AttackChain` → **GOAL event `secondary_player`** → `exporter.py:845-848`
  (`assistant["assists"]`) → `world/ingest.py:172` (`assists`) → the warehouse.
  Note the exporter takes `chances_created` / `shot_assists` from
  `ChanceCreationLedger` (`exporter.py:463-465`, "single source of truth") but
  `assists` from the ENGINE event — so the engine's assist and the ledger's
  creator are two independent derivations of the same fact, and they were
  measured agreeing 3/3 with 0 disagreements. That agreement is the thing worth
  protecting; the ledger's docstring claim that the engine "used to" fabricate
  is corrected in `chance_creation.py`.
- **THE FUNNEL'S `poss_result` IS NOT STALE — verified, not assumed.**
  `match_engine.py:4958` does `if poss_result.possession_lost: … continue`, so
  the shot funnel is only reached when the attacking team KEPT the ball, and
  `poss_result` is rebuilt at the top of every sequence (4938). So
  `ball_carrier` is the man holding the ball at exactly the coordinate the
  funnel shoots from. Worth stating because "is this a stale variable?" is the
  obvious objection to the change and the answer is structural.
- **A FOURTH UN-THREADED SITE, AND A HOLE IN THE GUARD I JUST WROTE.**
  `TransitionChain`'s counter-attack calls `AttackChain.generate` twice with no
  `shooter_name`/`assister_name` — while having just emitted a PASS whose
  `secondary_player` IS the shooter (8271) or carrying the ball himself (8290).
  Worse, `TransitionChain` picks `carrier` and `shooter` as two INDEPENDENT
  draws and `exclude=`s the carrier from the shooter pick, so in the engine's
  own counters the man with the ball can never score. Also note the new AST
  guard walks `match_engine.py` ONLY — a site in `event_chain.py` is invisible
  to it, which is exactly the omission class being hunted. Fix pending.
- **SET-PIECE HEADERERS ARE NOT POSITION-DERIVED, BUT ARE DEFENSIBLE.**
  `_pick_aerial_threat` weights DNA `jumping + heading`, not geometry, so a
  corner is won by the best leaper rather than by whoever arrived. Given
  corner steps 2-3 now run 13 men into the box, "the best aerial threat wins it"
  is a legitimate football model. Recorded, not changed — do not "fix" it
  without a measurement.
- **THE CROSS-RECEIVER CLAMP IS NOW ISOLATED** (`_diag_cross_clamp.py`), which
  is item 2 of the residue and had been blocked on a probe that could not
  separate a clamp from a placement. Signature-testing `record_touch` (new x
  inside the attacking band, y in [22,46], old x >5 m outside it) matches
  **77 times / 2,766 m never covered (26 pitches) per 2 matches** — of which the
  cross-receiver site is only part, since corner box placement writes the same
  band. Not yet apportioned between the two, and not yet changed.

## COUNTER-ATTACKS AND CROSSES (2026-10-03) — the last two fabrications

The chance-creation work left two paths that still named the wrong man or the
wrong place. Both were found by walking the shot paths one at a time rather
than by trusting the one I had already fixed.

- **A COUNTER'S CARRIER WAS STRUCTURALLY INCAPABLE OF SCORING.**
  `TransitionChain` picked `carrier = _pick_fast_player(...)` and then
  `shooter = _pick_shooter(..., exclude=carrier.name)` — two INDEPENDENT draws,
  the second explicitly barred from choosing the man holding the ball. And
  because the branch that lets him shoot was gated on `if shooter != carrier`,
  **the solo-run-and-shot branch was UNREACHABLE DEAD CODE.** Dead code that
  reads as a design choice is this project's standing pathology; it survived
  because `shooter != carrier` was always true and therefore looked like a
  guard rather than a bug. Now ~55% of counters are squared to a team-mate
  (the previous rate) and otherwise the carrier goes himself.
- **THE COUNTER'S KEY PASS NAMED A RECEIVER WHO DID NOT SHOOT.** Both
  `AttackChain.generate` calls in the counter dropped `shooter_name` /
  `assister_name`, while the block immediately above had just emitted a PASS
  whose `secondary_player` *was* the intended shooter — and handed the chain
  the whole eleven-man squad, so `_pick_shooter` drew a THIRD, different
  player. The key pass and the shot named different men. Both branches now
  thread the pair, and the solo branch passes `assister_name=""` — he dribbled
  it in, so it is genuinely unassisted, which is the same rule
  `_resolve_assister` follows everywhere else.
- **MY OWN AST GUARD HAD A HOLE, ONE FILE WIDE.** The guard added earlier
  walked `match_engine.py` only, so a site in `event_chain.py` was invisible to
  it — which is precisely the omission class it was written for, and the
  omission was silent because nobody was looking at that file. It now walks
  both files. **A guard that protects one file cannot catch the class.**
- **NEGATIVE CONTROL, because a test that passes for the wrong reason is
  worse than no test.** `_diag_counter_guard.py` strips the two kwargs at the
  `AttackChain.generate` boundary and re-runs the real test: it goes RED with
  `counter pass names receiver 'A6' but 'A7' shot` — exactly the defect. Note
  the same descriptor trap as `_install_no_thread`: `generate` is a
  classmethod, so wrapping it as a staticmethod raises `missing 1 required
  positional argument: 'situation'`. Read `raw = Cls.__dict__["name"]` and
  branch on `isinstance(raw, classmethod)`.
- **`secondary_player` MEANS TWO DIFFERENT THINGS ON ONE FIELD NAME.** On a
  SHOT it is the goalkeeper who faced it (`event_chain.py:6109` and five
  sibling sites: `secondary_player=gk.name`); only on a GOAL is it the
  assister. My first counter test asserted the GOAL reading against a SHOT,
  failed with `'GK' == 'A9'`, and pointed at the counter chain for a defect
  that was in the test. Sixth instance of the same shape as the probe-sign
  errors: **one name, two frame-dependent meanings.**
- **A TEST OF MINE WAS FLAKY BECAUSE ITS FIXTURE ASSUMED AN OUTCOME — AND IT
  WAS TWO TESTS, NOT ONE.**
  `test_the_shot_itself_is_still_recorded_without_a_creator` called the chain
  once with no seeding and asserted a shot came out — but `AttackChain` is
  PROBABILITY-GATED, so whether a shot exists depends on whatever ran before
  it. It passed 4 runs of the file alone and failed next to
  `test_cross_detector.py`. Same class as the carrier tests that "assume a
  carrier exists": sweep the seeds and **assert the sweep was not vacuous**
  (40 seeds must produce at least one shot, or the loop proved nothing).
  Now stable across 6 consecutive runs.
  **The twin in `test_chance_provenance.py` had the identical defect and only
  surfaced in the FULL 53-minute regression** — 4 isolated runs and a 7-minute
  subset both passed it, and it came back as the single failure of 181. Its
  draw produced `['HIT_WOODWORK']`, which added a second dimension: `is_shot`
  is a property whose event-type list omits woodwork (see Next Move item 7),
  so "a strike was recorded" is not the same assertion as `any(e.is_shot)`.
  Both now sweep 40 seeds, assert non-vacuity, and accept woodwork.
  **Lesson: a probability-gated chain makes a single-draw fixture wrong by
  construction, and a shorter suite is not evidence that it is right — it is
  evidence that the RNG happened to land your way.**

### CROSS RECEIVER: TWO FABRICATIONS IN THREE LINES

`event_chain.py:3445-3462`, the open-play cross. It is the purest instance of
the project's pathology found so far, because each line looks reasonable alone:

    cross_receiver = cls._pick_aerial_threat(players, exclude=last_player.name)
    rx, ry = position_engine.get_position(cross_receiver.name)
    rx = cls.clamp_attack_x(rx + random.uniform(1.0, 4.0), 85.0, 100.0, attacks_right)
    ry = max(22.0, min(46.0, ry))
    position_engine.record_touch(cross_receiver.name, rx, ry, minute)

1. **The receiver was chosen by ABILITY ALONE** — `_pick_aerial_threat` weighs
   DNA `jumping + heading` and knows nothing about geometry, so it could
   return a player 30-50 m upfield.
2. **The clamp then teleported him into the box** and banked the gap as
   distance covered: 77 `record_touch` jumps / 2,766 m never walked (26
   pitches) over two matches.
3. **And because `_moving_player` reads the position engine, he was scored as
   ALREADY THERE**, so `resolve_aerial_delivery` charged him no movement cost
   and he won the header for free. A man who never ran into the box got the
   ball because he was moved there on paper.

The fix needed no new machinery — `pick_weighted_spatial` already existed at
line 892, documented as "multiplies the label-based weight by the player's
real-time positional plausibility for an action happening at (at_x, at_y)".
The receiver is now picked by ability × plausibility, **stays exactly where he
is**, and the resolver charges him the real distance. Aiming a cross inside the
box is correct football and is kept — it is only ever a TARGET now, and it
belongs to nobody's tracked position.

- **THE DEFENDER PLACEMENT WAS WRITE-ONLY.** The adjacent block computed
  `dx2, dy2`, clamped the defender's y into `[24,44]`, and `record_touch`ed
  him — and **nothing after it ever read either variable** (grep: five
  occurrences, all assignments; the target came from `rx`/`ry`). Its entire
  effect was to teleport a defender up to 14 m in y and bank the gap as
  distance he covered. A fabrication with no consumer, which is exactly why it
  survived: it read like it was placing a marker. Deleted.
- **THE NEGATIVE CONTROL EXPOSED A COVERAGE GAP IN MY OWN TEST, AND THE TEST
  NAME WOULD HAVE HIDDEN IT.** `_diag_cross_guard.py` restores the deleted
  defects (ability-only pick + the `record_touch` teleport), runs the two new
  tests, then restores the file from a byte-exact backup. Result: **only ONE of
  the two pins went red.** `test_the_open_play_cross_never_rewrites_a_position`
  correctly failed; `test_cross_receiver_pick_prefers_the_man_in_the_box`
  **still passed** — because it calls `pick_weighted_spatial` *directly*, so it
  pins the HELPER and never touches the call site. I had named it "the cross
  receiver", which would have read as coverage of the cross path that does not
  exist. Renamed to `test_the_spatial_aerial_pick_prefers_the_man_in_the_box`
  with the limitation written into its docstring: one test says the helper
  discriminates, the other says the cross uses it, neither alone does.
  **Running a negative control is what turns a test's name into a claim you
  have actually checked.**
- **A COMMENT QUOTING THE DELETED BUG FAILS A TEXT SCAN.** The source pin
  initially failed on the cross block's own explanatory comment, which quotes
  `record_touch(receiver, rx, ry, minute)` verbatim as part of documenting the
  fix. Stripping comment lines before scanning fixed it. This is the
  `_diag_teleport_moves.py` trap again — a name inside a comment is not a
  call site — and it is why the project's standing rule is to trust structure
  over text.
- **THE MEASUREMENT THAT ISOLATED IT.** `_diag_cross_clamp.py` signature-tests
  `record_touch`: new x inside the attacking band AND `y` in `[22,46]` AND old
  x more than ~5 m outside it. Note that signature also catches corner box
  placement (both write into the same band), so it is an upper bound — which
  is why the fix was made by reading the mechanism, not by apportioning the
  count. `_diag_teleport_moves.py` context attribution still cannot separate a
  clamp from a deliberate placement; that remains the fourth rejected pairing.
- **THE SIGNATURE DID NOT MOVE, AND THAT IS THE MORE IMPORTANT RESULT.** After
  deleting the clamp, `_diag_cross_clamp.py` still reports **79** (against the
  **77** recorded before, across different matches). So the cross-receiver
  clamp was a real defect and a SMALL contributor. Do not let a future agent
  read "clamp deleted" as "the jump problem solved" — it was one site among
  several, and the count is dominated by something else entirely.
- **`_absorb_chain` IS AN AMPLIFIER, WHICH IS WHY THE CLAMP MATTERED MORE THAN
  ITS ARITHMETIC.** The bulk of ≥12 m `record_touch` jumps is
  `match_engine.py:5764`/`:5772` inside `_absorb_chain`, which snaps each
  player to **the coordinates of the event they just performed**
  (`record_touch(event.player, event.location_x, …)` and
  `record_touch(event.secondary_player, end_x, end_y, …)`). That is the
  documented on-ball reposition and it is legitimate *when the event's
  coordinates are real* — but it means **any fabricated event coordinate
  becomes a real tracked position**, and is then read by the exporter, the
  shot map and the aerial resolver as fact. The cross clamp's clamped `rx/ry`
  became the CROSS event's `end_x/end_y`, so `:5772` persisted the fabrication
  for free. This is the general reason the "omit, never invent" rule matters
  more than the distance it banks: a fabricated coordinate does not stay local,
  it gets promoted to a tracked fact by the next line that trusts the event.
  (`event_chain.py:1336` is the same idea inside `PossessionChain`; `:5731` is
  the CHANCE_CREATED shooter snap, whose `x - 8` invented spot was removed
  earlier; `event_chain.py:7056`/`:7076` are the corner box placement, which
  is deliberate and documented under *CORNER BOX*.)

### MEASURED, `_diag_touch_sites.py` — WHERE the position writes actually come from

The earlier apportion attempts inferred INTENT from context labels and could
not separate a clamp from a deliberate placement. That was the wrong question.
The right one is *which line of code wrote this*, and the caller's stack frame
answers it exactly — no inference. 1,086 single-call `record_touch` jumps
≥ 12 m over 2 real matches, 28,972 m (**275.9 pitches**), bucketed by
`file:lineno` with the owning chain named one frame out:

| SITE | jumps | metres | pitches | box-band | what it is |
|---|---|---|---|---|---|
| `match_engine.py:5764` | 463 | 12,576 | 119.8 | 52 | `_absorb_chain`: snap actor to `event.location_*` |
| `match_engine.py:5772` | 260 | 6,943 | 66.1 | 30 | `_absorb_chain`: snap `secondary_player` to `end_*` |
| `event_chain.py:1336` | 235 | 5,761 | 54.9 | 27 | `PossessionChain`: snap ball-carrier to ball |
| `event_chain.py:7056` | 50 | 1,696 | 16.2 | **49** | corner box placement (deliberate, *CORNER BOX*) |
| `event_chain.py:7076` | 40 | 928 | 8.8 | 0 | corner box placement |
| `event_chain.py:5731` | 14 | 350 | 3.3 | 0 | `CHANCE_CREATED` shooter snap |
| 5 minor sites | 22 | 647 | 6.2 | 3 | set-piece restarts |
| **TOTAL** | **1,086** | **28,972** | **275.9** | **161** | |

Two things fall out, and they are not the same claim:

- **91% of the METRES (25,280 of 28,972) is the on-ball snap** — the three
  "snap the player to the coordinate of the event he just performed" lines.
  That is architecturally necessary: a man who performed an event at a
  coordinate must be recorded there. It is not a fabrication *in itself*.
- **30% of the BOX-BAND hits (49 of 161) is `event_chain.py:7056` alone**,
  which makes 49 of its own 50 jumps into the box band. That is the corner box
  placement, introduced deliberately by corner steps 2–3 and documented as
  booking the displacement on purpose. So the old 77/79 figure was never about
  the cross clamp at all.
- **THE CROSS-RECEIVER CLAMP WAS A SMALL CONTRIBUTOR.** It was a real defect,
  it fed the amplifier below, and it accounts for a handful of events. Do not
  read "clamp deleted" as "the jump problem solved".

**THE AMPLIFIER, WHICH IS THE PART THAT MATTERS.** `_absorb_chain:5772` writes
`record_touch(event.secondary_player, event.end_x, event.end_y)`. The cross
clamp's `rx/ry` *became* the CROSS event's `end_x/end_y`, so that line promoted
the fabrication into a tracked position for free — and from there the exporter,
the shot map and the next aerial all read it as fact. **A fabricated
coordinate does not stay local; the next line that trusts the event promotes it
to truth.** That is the general argument for "omit, never invent" being a hard
rule rather than a style preference: the cost of inventing a coordinate is not
the metres you banked, it is that everything downstream can no longer tell it
apart from a measured one.

**THE QUESTION THIS RAISES AND DOES NOT ANSWER.** When the on-ball snap moves a
man 29 m in one tick, there are two readings and this probe cannot separate
them: either the chain's coordinate is right and the position engine was simply
stale (the snap is a *correction*), or the chain's coordinate is invented and
91% of all position writes are fabrications. Both produce an identical jump.
Distinguishing them needs something this probe does not have — an independent
measure of where the player was before the chain ran (the 10 Hz integrator's
own trace at that instant, or a chain that is handed the engine's coordinate
instead of generating its own). **Do not guess.** This is now the single
highest-value open measurement in the audit, and it is far larger than anything
fixed today.

*Probe self-correction, recorded because the first version of it lied:* it
printed a heading `by signature (new_x in band, y in [22,46], old_x >5 m
outside)` and then summed **every** site — the filter was never implemented, so
the number under that heading was the ≥12 m total wearing a false label. `y` is
now captured and the filter is actually applied (hence the two columns above).
The probe also writes `_diag_touch_sites.txt` itself, because piping its table
through `Select-Object -Last N` truncated the **top** — the part worth reading.

## THE ON-BALL SNAP — correction or fabrication? (2026-10-04, ANSWERED)

The audit's highest-value open question, and it is now closed. The short
answer: **the snap is overwhelmingly a CORRECTION.** The coordinate it writes
is the ball's own tracked position, so it records a fact. One genuine
fabrication class was found on the way and is fixed.

**THE METHOD — no inference, two identity joins.** The earlier attempt failed
because it inferred INTENT from context labels, which cannot separate a
deliberate placement from a clamp. This one attributes every large snap to the
line of code that produced its coordinate:

1. Wrap `BaseChain.make_event`, record `id(event) -> "file:lineno"` of the
   CALLING frame (`make_event` is a `@staticmethod`, so wrap it AS one or the
   descriptor hands the wrapper `minute` as `self`).
2. Wrap `PositionEngine.record_touch`; for a jump ≥12 m, read
   `sys._getframe(1).f_locals["event"]` — `_absorb_chain` has the event object
   as a frame local, so the join is by **object identity**, not by matching
   coordinates. That distinction matters: a coordinate match would be ambiguous
   when two events share a position.
3. Look that event up in the `make_event` map. The answer is then read off the
   source, not argued.

**WHAT THE SNAP ACTUALLY WRITES** (`match_engine.py:5764` snaps the actor to
`event.location_*`; `:5772` snaps `secondary_player` to `end_*`):

| make_event site | event | snaps | metres | within 3 m of the ball |
|---|---|---|---|---|
| `event_chain.py:1556` | PRESS | 159 | 4,156 | **159/159** |
| `event_chain.py:2705` | PASS (actor) | 64 | 1,219 | **64/64** |
| `event_chain.py:2705` | PASS (`end_*`) | 59 | 1,332 | 18/59 |
| `event_chain.py:7315` | AERIAL_DUEL | 29 | 1,122 | **29/29** |
| `event_chain.py:1336` | PossessionChain own snap | 180 | 4,441 | n/a (see below) |

So the dominant sites write **the ball**, which is tracked state, and the snap
is a correction: the actor of an event happened where the ball was. The
18/59 row is the RECEIVER, which is correctly not at the ball — it is at the
pass's destination.

**MY OWN ORACLE WAS WRONG ON THE `event_chain.py` ROWS, AND IT ALMOST BECAME A
FINDING.** The `ATBALL` column reads `self.state.last_ball_x` from the caller's
frame, but for `event_chain.py:*` sites `self` is a **chain class**, not the
engine, so `last_ball_x` is `n/a` and the table printed a confident `0/180` for
the single largest site. That number was a probe artefact, not a measurement —
`event_chain.py:1336` is `record_touch(last_player.name, x, y, minute)`, the
possession chain placing the carrier on its own tracked ball. **A column that
reads cleanly is not a column that is right.**

**THE FABRICATION: `MatchEvent.location_x` DEFAULTS TO (50.0, 34.0).**
`match_engine.py:592` — `location_x: float = 50.0`, `location_y: float = 34.0`.
The centre spot. So an event built without a location does not read as absent,
it reads as a **measurement of the centre circle**, and `_absorb_chain:5764`'s
`if event.location_x is not None` guard cannot see it — 50.0 is not None.

This is the **third instance of the same trap** the project has already
documented twice: `PositionEngine.get_position` answers `(50.0, 34.0)` for an
untracked player, and `PlayerSpatialState.position` holds the position LABEL
("ST") rather than a point. All three read as real coordinates and none is one.

Measured, one match, by **kwarg presence** (exact — `make_event` takes them
via `**kwargs`, so their absence is knowable at the call, before the default is
ever applied):

- **64–68 of ~2,900 events per match (2.1–2.2%)** sat on the centre-spot default.
- **24 events/match produced ≥12 m snaps, 602–775 m banked** as distance nobody
  covered, and **every one** traced to a single line.

Sites fixed (8; `event_chain.py:8167`, `:9327`, `match_engine.py:3474`,
`:4210`, `:4221`, `:5948`, `:6008`; `match_engine.py:6156` `_emit_event` is a
`**kwargs` factory and is exempt):

- **`event_chain.py:8167`** — `TransitionChain`'s "player played through" PASS
  passed **no location at all**. The pressed man is on the ball, so it now takes
  the press location. This was the whole of the 602 m.
- **`event_chain.py:8107`** (found while reading 8167, not by the probe) — the
  PRESS location was **`random.uniform(55, 85)`**. A press happened in a zone no
  player or ball ever occupied, and the snap then teleported the presser there.
  Now `position_engine.tracked_position(pressed.name)`, since the pressed man
  holds the ball. The two press sites are why the table showed one at
  159/159 and this one at 16/30.
  - **THE SCALE OF THE PRESS FIX, MEASURED CORRECTLY AFTER A FAILED
    MEASUREMENT.** The first football A/B keyed its `before` arm on
    `event_type == "PRESS"` with no site check, so it rewrote **84 presses per
    match** that were already correct — `PossessionChain`'s press is `ball ± 3 m`
    — and produced a phantom "+44% deep presses" and "−6 km/match" from the
    restore alone. Both are **retracted in full**. Keyed on the emitting chain
    via the `cls` frame local instead (`_diag_football_sites.py`), the real
    figures are:
    | | before | after |
    |---|---|---|
    | snap distance, 3 matches | 29,428 m | 26,166 m (**−1,087 m/match, −11.1%**) |
    | TransitionChain press snaps | 95 / 2,946 m | 48 / 1,578 m (**−456 m/match**) |
    | governed band rewrites (`_diag_press_band.py`) | 21/match, all inside `[55,85]x[10,58]` | 0 |
  So the press fabrication was **~21 of ~315 presses per match (6.7%)** and
  **~1.1 km/match** of snap distance — real, but an order of magnitude smaller
  than first claimed. PossessionChain's press was never affected and is a
  standing control showing no systematic shift.
  - **THE BAND CONTROL NEEDED ITS OWN CONTROL.** The first band verdict reported
    **49 phantom violations** by testing PossessionChain presses against
    `[55,85]x[10,58]` — a rule that was only ever TransitionChain's. The three
    gates are: `PYTHONHASHSEED` now held constant (`_diag_repro_process.py`
    sweep), the emitting chain read from the `cls` frame local, and the verdict
    range restricted to the chain that owns the rule. A check that does not
    measure the thing under test is the same defect as a mutation with the
    wrong condition.
- **`match_engine.py:3474`** — `PENALTY_SCORED`/`PENALTY_MISSED` at the centre
  circle. A penalty is taken **11 m from the goal line**. The engine models no
  shootout goal (a real shootout has both teams at the SAME one), so the
  home-attacking end is used and `location_source="shootout_spot_convention"`
  is stamped. `PENALTY_SCORED` IS in `_ON_BALL_EVENTS`, so this one really did
  snap the taker.
- **`match_engine.py:4210` / `:4221`** — INJURY and SUBSTITUTION. Every change
  in every match was exported as happening on the centre circle. Now the
  outgoing player's tracked position (where he actually stood), computed
  **before** the injury block because the INJURY event is deliberately emitted
  first as the cause.
- **`match_engine.py:5948` / `:6008`** — VAR_DISALLOWED_GOAL and GOAL_CELEBRATION,
  both at the **goal mouth** via a new `MatchEngine._goal_mouth(team)`. A goal
  ruled out for offside, recorded on the centre circle, is football nonsense
  that exported as truth. The VAR event also now carries the offside
  coordinates (`pending_offside_fk_x/y`) so the two facts travel together.
- **`event_chain.py:9327`** — the `event_chain.py` substitution chain. **It has
  ZERO call sites**: the live path builds the SUBSTITUTION event directly at
  `match_engine.py:4221`, which is exactly why the same defect had to be fixed
  twice. Recorded rather than deleted, and the docstring says so, because a
  green guard on the dead path is not evidence the live one is covered.

**THE GUARD, AND IT EARNED ITS KEEP IMMEDIATELY.**
`test_every_event_builds_with_an_explicit_location` walks the AST over **both**
files and covers **both** construction routes — `make_event(...)` and direct
`MatchEvent(...)` — because `match_engine.py` builds four events directly and a
guard watching only `event_chain.py` could not see a single one of them. That is
the same one-file hole the `shooter_name` guard had, recurring; the lesson
generalises: **a guard that protects one file cannot catch the class.** It
flagged `match_engine.py:4210`/`:4221` on its first run, which the runtime
probe had not reported at all (no substitution fired in that match).

Re-measured after the fix: **0 events on the centre-spot default, 0 snaps.**

**WHAT IS STILL AN ASSUMPTION, AND DELIBERATELY NOT FIXED.** The receiver snap
(`match_engine.py:5772`) writes the pass's **aim** point. `end_x` is only ever
overwritten for corners and free kicks (`event_chain.py` CORNER_TAKEN /
FREEKICK_CROSS stamp `aerial.contact_point` or the aimed target afterwards); an
ordinary PASS keeps the aimed `end_px` forever. So the snap asserts "the
receiver arrived exactly where the pass was aimed". That is a **modelling
choice**, not an invention — the aim is derived from a teammate's tracked
position by `_find_target`, not drawn from `random` — and it is the same open
`end_px`-vs-`end_x` question already on file under the PLOFA export frame note.
Fixing it needs a recorded arrival, which is a real-physics change.

**`record_touch` DOES BANK THE JUMP** (`position_engine.py:907-910` → `state.
minute_touch_distance`), which is what makes the centre-spot default
consequential rather than cosmetic: a man recorded on the centre circle has the
gap from his real position credited as distance he covered.

### MEASURED, `_diag_creator_truth.py`, 3 real matches per arm — shooter honesty

Shooter = the man who actually had the ball (section E walks back to the last
genuine touch, independent of the ledger). **Each arm is a separate process** —
this was believed necessary because of the retracted "in-process A/B inherits
the brain caches" rule. It is not actually necessary (`arm none` in
`_diag_seed_repro.py` is byte-identical), but separate processes remain the
habit that caught the real bug, and they cost nothing.

| | before (no thread) | chain-local | **+ counter fix** |
|---|---|---|---|
| shooter IS the ball-holder | 64/103 (**62%**) | 103/118 (87%) | **98/108 (91%)** |
| drawn shooters | 39 | 15 | **10** |
| engine CHANCE_CREATED events | 0 | 4 | **9** |
| creator REALLY passed the ball | — | 4/4 | **9/9, fabricated 0** |
| engine vs ledger, same player | — | 4 agree / 0 disagree | **9 agree / 0 disagree** |
| `chances == goal_assists + shot_assists` | 54 ok, 0 broken | 54 ok, 0 broken | **54 ok, 0 broken** |

Denominators differ per arm (they are different matches: 113 / 126 / 119
shots), so read the RATES. The before-arm's `0` CHANCE_CREATED events are an
artefact of the probe stripping the names — `_pick_creator` is deleted, so with
no name there is legitimately no creator — and are not evidence about the old
code, whose draw always returned somebody.

- **ONE NEW DISAGREEMENT, RECORDED NOT HIDDEN.** Section D went from
  `3 agree / 0 disagree` to `7 agree / 1 disagree`. The engine's assist and
  the ledger's backward scan are two INDEPENDENT derivations of the same fact,
  and they now differ on one goal. That is worth chasing and must not be
  smoothed away: it is either a legitimate edge case (the ledger attaching the
  setup pass to a different delivery) or a real disagreement about causation.
  **Not yet diagnosed — do not report the two as agreeing.**
- **THE REMAINING 10 DRAWN SHOOTERS ARE NOT ALL DEFECTS.** Each is printed with
  the player who actually had the ball. They are not assumed to be wrong,
  because the back-scan's expected-predecessor rule is wrong for set-piece
  headers: it expects the man a corner was AIMED at, who is not necessarily
  the man who WON it. Classify before touching.
- **THE CARRIER PICK IS STILL A ROLE PICK, AND THAT IS DEFENSIBLE.**
  `_pick_fast_player` selects "who leads the break" by ability; it is not a
  false claim about where the ball is, and the CARRY event records him at the
  anchor coordinate. The proper fix is to thread `DefensiveChain`'s ball-winner
  in, not to guess at the call site.

## REPRODUCIBILITY — the whole cross-process story was `PYTHONHASHSEED` (2026-10-04, FIXED)

The project's single most-repeated open item was seed reproducibility, and it
had been carried as a claim with a confident wrong cause for five weeks. Three
things were believed. All three were false.

- **"An A/B across two `simulate()` calls in one process is invalid, because
  the second inherits the first's module-level brain caches."** Falsified by a
  negative control: `_diag_seed_repro.py`'s `arm none` — same seed, a second
  `simulate()`, **nothing cleared at all** — is byte-identical to the
  single-call reference (digest `0373f21a9ce5`, 2463 events). All five clearing
  arms match it too. Module caches do not contaminate the timeline.
- **"`np.random.seed()` does not help because the live path calls
  `np.random.default_rng()`, which is entropy-seeded."** `np.random.seed()`
  was the wrong lever, and for a different reason than stated: `default_rng()`
  is called 531,424 times per match and **every** call site passes an explicit
  seed. None of them reaches the event timeline.
- **"The proposed fix is three lines clearing the brain/mind caches at the top
  of `_initialize_simulation()`."** Not only unnecessary — **actively harmful.**
  `_minds` carries the season-hydrated temperament (`auto_run_match.py:1157`),
  and `_brain_registry` is how all 13 caller `register_brain` sites select a
  brain set for an A/B. No code change was made. Recorded so nobody re-derives
  it.

**WHAT IT ACTUALLY WAS.** Exactly three `hash()`-on-`str` calls sit on the match
path — `ball_vision.py:124` (recall noise for a player who has lost the ball)
and `perception.py:422` / `:519` (per-snapshot perception noise, two draw
paths). A `str` hash is salted per process, so the same fixture produced a
different match on every machine. `tests/test_determinism.py` had been pinning
the count at 3 "on purpose, because changing them changes the live 26/27
season".

**THE FIX.** `zlib.crc32(ident.encode("utf-8"))` at all three sites — the
pattern already in use at `position_engine.py:2481`, `world/squads.py:47` and
`world/proof.py:168`. The AST test now pins the count at **ZERO**, which is the
only version of this test worth keeping: a re-introduced `hash()` does not
crash, does not warn, and is invisible until someone re-runs a calibration on a
different machine.

**VERIFIED.** One match per process, `_diag_repro_process.py --seed 777`, three
separate processes:

| `PYTHONHASHSEED` | events | DIGEST |
|---|---|---|
| 0 | 2656 | `79857b22c6e2` |
| 1 | 2656 | `79857b22c6e2` |
| 12345 | 2656 | `79857b22c6e2` |

One digest. Before the fix, the same sweep gave three different matches (and
`=0` gave three identical ones, which is what pinned the cause).

**THE COST WAS ACCEPTED KNOWINGLY.** Every calibration number in this file
taken before today was taken under an unpinned hash seed. It means three
specific things:

1. Per-arm "swing" between measurements includes **hash-order noise**, which was
   never separated from real variance. "A real match takes ~90 s here, not
   19–23 s" and "2.5x swings between arms are warm-up noise" are both
   unestablished — neither was measured with the hash seed held constant.
2. `PYTHONHASHSEED=0` is now an optional hygiene measure rather than a
   requirement. Keep setting it (`world/` already does) but do not treat its
   absence as a defect.
3. `_STREAM_PARITY_DRAW` is no longer load-bearing *for reproducibility* — see
   its note above for why it is still a deliberate open decision.

**METHOD NOTES, so the next agent does not re-derive this.** The causation was
closed by exclusion before it was closed by measurement, which is what made it
fast: `tests/test_determinism.py` already pinned the count at 3, and a
set-iteration grep found none, so the search space was three lines. Then
`_diag_hash_divergence.py` found the first divergent event (event 37, a
`FOUL_COMMITTED` vs a `CARRY` — a proximity threshold between *perceived*
positions, which is exactly what a perception noise offset would move).
Running the sweep first and explaining it afterwards would have produced a
plausible guess instead of a located cause.

**STILL OPEN, AND IT IS A DIFFERENT BUG:** the *production roster* path does
not replay across processes — same seed, same hash seed, real
`PLOFA-2026-2027.xlsx` squads: 3,057 / 3,060 / 3,095 events. Player positions
diverge from minute 1:40 while the iteration *order* is identical, so it is
allocation-dependent (`id()`, or an unseeded generator). Not re-tested after
this fix, because the scratch path and the roster path are separate and this
fix only touches three lines of noise seeding.

> ### ⛔ AND THERE IS A THIRD, SEPARATE REPRODUCIBILITY DEFECT — THE CALENDAR (2026-10-21)
> **`roster_loader.PlayerRecord.age` uses `date.today()`.** `player_dna`
> multiplies every drawn attribute by the player's AGE CURVE, so a player
> whose birthday falls between two runs is a day older, their attributes
> move, and the match diverges completely (chaos: one changed probability is
> enough). **A real-roster match is reproducible for a FIXED roster and NOT
> across calendar days.** Measured while the export fix was being verified:
> the same seed and fixture gave 3,056 events on 2026-10-20 and 2,885 on
> 2026-10-21, and the workbook has three players born on that boundary —
> **Lewis Driutt (Oxton, the fixture's HOME XI, dob 1999-10-21)**, Joey
> Farńez (Telbey, 1995-10-21) and Nal Mahn (Telbey, 2002-10-20)
> (`_diag_roster_age_shift.py`).
>
> **THIS IS NOT THE SAME BUG AS THE ONE ABOVE** (that one is
> allocation-dependent and process-local; this one is the calendar and
> affects every process equally). **Rule for the next reader: before calling
> "same seed, different match" a determinism failure, check the roster's
> date dependency** — otherwise a date rollover will be diagnosed as
> nondeterminism.

## Work State
- **THE ROSTER WORKBOOK ALWAYS LOOKS DIRTY, AND IT IS NOT A WRITE.**
  `PLOFA-2026-2027.xlsx` is tracked and contains live formulas — `PLAYERS!R207`
  is `=DATEDIF(TODAY(),O207,"m")` ("Months Left") — so it shows as modified
  (and a different size) every time Excel recalculates and saves it, with the
  cached result drifting from 44 to 43. **Nothing in the repo writes that file**:
  `roster_loader.py` and every other reference use
  `load_workbook(read_only=True)` / `pd.read_excel`, and `squad_manager.py`'s two
  `.xlsx` sites read *exported* match files, not the roster. A diff of all
  11,853 non-empty cells at HEAD vs the working copy (`_diag_xlsx_diff.py`)
  found exactly one difference — that cached `TODAY()` value — so **a dirty
  roster is not evidence that something wrote it.** Verify with
  `_diag_xlsx_diff.py` (cells, not bytes) before treating it as a write;
  `_diag_xlsx_cell.py` / `_diag_xlsx_formula.py` identify the cell and say
  whether it is a formula. Do not stage the workbook to "clean" the tree.
- **COMMIT HAZARD, RECORDED BEFORE IT BIT: `git add -u` IS WRONG HERE.**
  Every substantive artefact of this session is either a MODIFIED tracked file
  or — worse — **an UNTRACKED new one**: `tests/test_chance_truth.py` (22
  tests, the only thing pinning the creator, the carrier hand-off, the
  three-quantity relationship and the AST guards) plus the probes
  (`_diag_creator_truth.py`, `_diag_carrier.py`, `_diag_pass_direction.py`,
  `_diag_cross_clamp.py`, `_diag_touch_sites.py`, `_diag_cross_guard.py`,
  `_diag_counter_guard.py`, `_diag_delete_pass_event.py`,
  `_diag_bound_delete.py`, `_diag_ast_structure.py`, `_diag_snap_origin.py`,
  `_diag_default_loc.py`) that AGENTS.md cites as
  evidence. `git add -u` stages **tracked files only**, so it would have
  committed the code changes and **silently dropped every test that justifies
  them** — the worst possible outcome, and invisible because the commit
  succeeds. Stage explicitly instead. Also exclude, deliberately:
  `plofa_output/**` (regenerated by `test_plofa_export.py` on every run) and
  the tracked `__pycache__/*.pyc`, which are noise. Note the `.quarantine_*`
  and `.corrupt-*` directories in the tree are pre-existing, not mine — leave
  them alone.
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

## RECEIVERS: THE BRAIN'S CHOSEN RECEIVER IS DEAD CODE (2026-10-05)

Started from two photos of RCB/LCB pass maps and the observation that the
passes "look right" in real football and wrong here. The trail ended somewhere
much larger than a missing sensor. Probes: `_diag_receiver_signal.py`,
`_diag_receiver_counterfactual.py`, `_diag_receiver_owner.py`,
`_diag_receiver_lever.py`, `_diag_lever_decisive.py`, `_diag_target_join.py`,
`_diag_receiver_layer_split.py`, `_diag_matrix_sensitivity.py`.

- **THE NEURAL BRAIN PICKS THE INTENT AND NOTHING ELSE.** `_find_target` runs
  ~896 times a match and returns a receiver, and that receiver governs the pass
  endpoint in **1 of 686 passes**. `event_chain.py:2111` reads
  `receiver = forced_receiver` — set earlier from `phase_decision.target`
  (1722) or `matrix_decision.target` (1817) — and only if that is `None` does
  line 2152 ever read `active_decision.target`.
- **FORCED 70 RECEIVERS, GOT A BYTE-IDENTICAL MATCH.** Arm `off` and arm `far`,
  same seed: 2827 events, digest `b075ecc2520f50ca`, 3-1, identical in every
  event field. A forcing probe that reports "no effect" is worthless unless the
  lever is proven live first (`team_offball_probe.py --intervene` sets the
  decision *before* the rule sees it), so `_diag_lever_decisive.py` re-ran it
  with no `importlib.reload` — the first version reloaded
  `brain_integration`, rebinding `NeuralDecisionBrain` while `event_chain`
  held the old class. A probe bug that reads exactly like an inert lever.
- **`_diag_receiver_owner.py` — the actual owner.** Endpoint from
  `forced_end` (phase/matrix) **1206**; wide-combo 225; `_pick_receiver` 13;
  from the brain's receiver **1**. `_pick_receiver` running 13 times in 685
  passes is what proves the policy path is live ~98% of the time *within* its
  branch — the branch itself is simply rarely reached.
- **SPLIT: AttackingMatrix 80%, TacticalPhase 20%** (1178 vs 296 of 1474
  targets produced; `_diag_receiver_layer_split.py`). The matrix barely
  abstains — 1178/1195. So there is effectively **one** owner, not two, and
  gating only the phase layer would move ~a fifth of passes.
- **NEITHER IS BRAIN-DERIVED.** `possession_phases._pick_circulation_target` is
  a role table × `1/(1+d/9)` × a marking factor × a 1.25 favored-flank pull,
  then a weighted-random draw. `attacking_matrix.py:550` is six hand-placed
  coefficients. **Emergent receiver choice is unreachable until one of them
  defers to the policy.**
- **SENSITIVITY (`_diag_matrix_sensitivity.py`, gates below).** Doubling one
  coefficient, count argmax flips: **bonuses block 13.9%**, progress 11.3%,
  freedom 4.7%, depth 4.7%. Nothing inert. The bonuses are the MOST
  load-bearing term — and `same_flank_bonus` lives in them.
- **THE HOLE, AND THE FIX.** `same_flank_bonus` fired **only** for LB→LW and
  RB→RW. A centre-back passing out of defence had **no same-side preference
  at all**. Added `_cb_same_side_corridor()` (`attacking_matrix.py:88`): a CB
  passing to a pivot on his own side **and ahead of the ball** gets the same
  0.06. The ahead-test is deliberate — `progress` already scores recycling, so
  a same-side bonus firing backwards would fight it. Fires **155x/match**.
  8/8 unit cases; `test_football_brain` 34/34; `test_positional_play` exit 0.
- **TWO GATES THAT EARNED THEIR KEEP.** *Fidelity:* the bonuses are inline
  literals folded into `_Option.value`, so they are recovered by algebra
  (`value/lane - (0.30p + 0.45f + 0.20d)`) and the recomputation must equal
  the engine's own value — **1.1e-16**, exact. *Oracle:* the first
  sensitivity table reported `unperturbed argmax == engine's pick: 0/1195`,
  because `decide` selects from **SUBSETS** (`best_far`/`best_close`/
  `target_opt`) via a rule cascade, not a global argmax. Fixing the oracle
  dropped `progress` from 19.3% to 11.3%. **A sensitivity study that measures
  its own reimplementation is worse than none** — the oracle disagreement is
  the only thing that caught it.
- **⚠️ POSITIONS ARE NUMBERED — this nearly shipped as a dead rule.** The first
  `_cb_same_side_corridor` tested `tpos not in ("CDM","CM","CAM")`. Live
  positions are `CM1`/`CM2`/`CB1`/`CB2` (see *Starter name template*), so it
  matched **nothing** — a guard that read as a working rule and was inert on
  arrival. Strip digits with `.rstrip("0123456789")`, exactly as
  `role_features._role_family` does. **Fourth instance this session of a
  condition that looks right and matches nothing**, after the wide-boost probe
  pairing, the atomic collector, and the phase detector.
- **WHAT IS STILL NOT KNOWN.** (1) The four bonuses are recovered as a **sum**,
  so `same_flank_bonus` was never measured INDIVIDUALLY — "the bonuses are
  most sensitive" does not prove `same_flank` is. (2) 0.06 is still a **guess**,
  fitted to nothing; the fix makes the corridor present, not correct. (3) Only
  **13%** of CBs in their own half have a same-side forward option available at
  all, so this improves the choice among men who are present and cannot create
  the ones who are not — the pitch is still 33-35 m wide against a real
  45-55 m (*WIDTH* below). (4) One match, one seed, throughout; every
  percentage is a single sample.
- **CONSEQUENCE FOR OPTION "A POINTER HEAD".** It is **dead**, not merely
  weak: a head driving `_find_target` would change ~1 pass in 686, so it cannot
  produce any pass-map difference. The real sequence, if emergent receiver
  choice is ever wanted, is (1) make the policy authoritative over
  `forced_receiver`, then (2) a learned target head. Weeks, not hours.
- **WHY NO RETRAINING AND NO NEURAL WORK WAS NEEDED.** The receiver is not
  chosen by the network at all, so nothing about the brains was relevant. The
  two CB sensors added to `role_features.MENU_NAMES["CB"]` (`same_side_lane_open`,
  `far_side_gap`) are **INERT against the shipped brain**: `brains/CB.json` is
  arch `[31,32,32,10]` = 24+7, and `brain_integration.py:524` truncates the
  33-wide vector back to 31. Measured: padded vector bit-identical pre/post,
  argmax differs on **0/4000** random inputs. They are harmless and are a
  prerequisite for (2) above, so they were KEPT — but **do not cite them as a
  fix.** ⚠️ That truncation is also **silent**: a schema change that
  invalidates every trained CB brain logs nothing and just degrades to the old
  behaviour. Unguarded.

## WIDTH: THE SHAPE CHAIN IS THE CAUSE, AND THE INTEGRATOR IS EXONERATED (2026-10-05)

Asked to stop the shape collapsing into a narrow block and to hold a 45–55 m
band. The diagnosis below **overturns** the hypothesis I formed from the first
measurement. Probe: `_diag_wide_target.py` / `_diag_wide_target.txt`.

- **THE INTEGRATOR IS NOT AT FAULT.** I expected a shape-holding jog to be
  failing to arrive — the same category of mistake as the low-block recovery
  branch at `match_engine.py:3001-3018`. Measured instead, on ~300k live
  samples: median lag between a wide player's chain target and his actual
  position is **1.2–1.4 m**, at every boost setting, in both phases. He walks
  to the target he is given. **Do not "fix" the approach speed.**
- **THE TARGET IS ALREADY NARROW.** ⛔ **the separation half of this claim is
  VOID and is withdrawn** — every separation number from `_diag_wide_target.py`
  was produced by a mispaired tick identity (the root cause is now known and is
  in *WIDTH GATE* below). The trustworthy figure is the target-side median
  **30.6 m** for wingers / **28.3 m** for full-backs, still nowhere near 45–55 m,
  so the claim's *verdict* survives on a different instrument. The per-player
  half STANDS, because it needs no pairing: distance to the touchline is
  15.2 m in possession / 18.3 m in a block, against a real 3–12 m.
  Instrument: `PositionEngine.live_spacing_redirect(team, cx, cy, tx, ty)` is
  called at `match_engine.py:2917` with the **finished** target of the whole
  chain, so its arguments are the answer; it receives no player name, so the
  wrapper reads `pname`/`pos`/`has_ball`/`_in_block` from the caller's frame
  locals (`sys._getframe(1)`) — the `_diag_touch_sites.py` join.
- **THE INTEGRATOR IS THE RED HERRING; `_JOG_SPEED` 1.75–1.90 m/s and the
  `arrive = 2.0` deadband are both fine at a 1.2 m lag.**
- **THE ONLY TWO CHAIN TERMS THAT TOUCH A WIDE ROLE** are the live run targets
  (`match_engine.py:2835-2843`) and the CK35 pitch stretch (`:2859-2862`).
  CK36/37/38 are midfielder and backline terms keyed by player name, so they
  **cannot** reach LW/RW/LB/RB — which is what narrowed the search to two.
  - **CK35 is dead out of possession.** `stretch_w = 0.0 if _in_block else …`,
    and the count confirms it: it was called on **139,991 of 301,955** samples —
    exactly the in-possession count (301,955 − 161,964 "out, in block" =
    139,991). Out of possession the block substitution owns the anchor.
  - CK35 fires on 65.8% of in-possession samples and is doing its job: when it
    is OFF the ball is already 10.2 m off a touchline (no stretch needed), when
    ON the ball is 26.5 m off one (stretch needed). The rule is not misfiring.
  - **The live run target fires on 18% of samples at a median 24.0 m off the
    line, blend 0.30.** A winger cutting inside is real football and is
    deliberately wired (2026-09-29); this is the source of the p90 excursions,
    not a defect.
- **THE KNOB WAS RAISED TO 0.55 AND 0.75 AND REVERTED.** In-possession actual
  distance to the touchline went **15.4 → 14.6 → 12.1 m** — monotone, and into
  the real 3–12 m band at 0.75. It was reverted anyway, and the reasons are the
  point:
  1. ⛔ **It does not fix the collapse** — team-width p05 4.2–6.6 m, p10 16–18 m
     at all three settings. **UNEVIDENCED, both figures VOID:** they are the
     same mispaired separation column as above, so the revert was justified at
     the time on a number that does not exist. The *reasoning* survives
     (a rule that raises the *mean* cannot move a *tail* — CORNER STEP 6's
     discipline, the thing measured is not the thing asked for) and is now
     independently supported by *WIDTH GATE*, but **the measurement offered for
     it was not evidence.**
  2. ⛔ **It does not reach the band** — median separation 36.2 → 38.9 m.
     **VOID**, same column. The trustworthy equivalent is the target-side median
     30.6 m, which is worse than 36.2 m and does not reach the band either.
  3. **It costs real football.** CK35 runs *after* the run targets, so a higher
     weight directly undoes the winger's cut-inside. Trading a deliberate cut for
     3 m of average width is a bad trade on its own terms.
  **So: the revert was right, and two of its three stated reasons were not
  evidence at the time.** That is worth more than the decision — the alternative
  reading, that a lever was abandoned because a probe lied, is the one to
  guard against next time.
  Full numbers kept as a comment block on `STRETCH_TARGET_BOOST`
  (`position_engine.py:143`) so the next attempt starts from evidence.
- **GATE:** `tests/test_positional_play.py` + `tests/test_touchline_wide.py`
  = **48 passed, 1 failed** at 0.75 and again at 0.30. The failure is the
  documented pre-existing `test_out_of_possession_touchline_press_gated_by_
  press_intensity` — an **x**-axis LB-forward-push assertion, and
  `STRETCH_TARGET_BOOST` only ever feeds `ty`. Not caused by this work.
- **PROBE ERRORS, three, all the documented class.** (1) `pname in TGT` against a
  `defaultdict` is **always False** for a first sighting — membership does not
  create the key — so the first run recorded zero samples and then divided by
  an empty list. A member test against the collection you are about to append to
  is not a filter. (2) I paired the two flanks by **list index**, then "fixed"
  it onto `MatchEngine._offball_tick_seq()` — and the separation numbers went
  degenerate (p10 **2.7 m**) at one setting and plausible (p10 38.7 m) at
  another with identical code. **ROOT CAUSE, FOUND LATER, AND IT IS WORSE THAN
  "the wrong tick identity": `_offball_tick_seq` is a plain `int` ATTRIBUTE
  (`match_engine.py:1990`), not a method.** The probe **called** it, so
  `int(fn())` raised `TypeError`, the probe's own `except Exception: return -1`
  **swallowed it**, and every wide player in the match collapsed into **one**
  tick group. The symptom was not a crash — it was `wide-ticks=1` and a p05
  computed over a single overwriting dict, printed beside a confident summary
  line. **A swallowed exception that silently degrades an instrument is worse
  than a crash.** Both separation figures from that probe are void; the
  trustworthy numbers are the per-player, index-free ones (target-to-line,
  actual-to-line, lag) at ~140k samples.
  `_diag_wide_channel.py` pairs correctly — one dict per `_offball_run` call,
  both flanks same-instant by construction — but samples only 414 times a match,
  and its absolutes disagree with the dense instrument by ~2–3 m. **Two
  instruments, different cadences, different absolute values: quote the dense
  one for medians and the sparse one for structure.** (3) **A third instrument
  that pairs correctly still produced a number nobody could use**, because a
  p05 over 409 frames is not a p05 — the reason `_diag_width_gate.py` uses the
  dense `live_spacing_redirect` call (~300k/match) rather than the sparse one.
- **THE OPEN QUESTION IS THE TAIL, AND IT IS NOT ISOLATED.** The collapse is
  both wide roles being central *at the same instant*. Candidates, in order,
  none measured: (a) the run target firing for **both** flanks off one cached
  `(team, minute, ball-zone)` key — a winger's cut is on the ball's side, so
  both flanks cutting in is one cache entry serving two players; (b) the block
  anchor substitution putting both wide roles in the same narrow slot band.
  Do not raise another constant until one of these is measured.

## TWO MORE DEAD MECHANISMS, BOTH FIXED (2026-10-05) — `geometric_awareness`, and form on the shot

The third and fourth instances of the project's standing pathology (a mechanism
that exists, is wired to something, and never runs), found in one pass and both
closed. Probes: `_diag_live_dna.py`/`.txt` (sections A–E),
`_diag_awareness_gates.py`/`.txt`, `_diag_drift_bisect.py`,
`_diag_dna_guard.py` (negative control).

### FIX 1 — `geometric_awareness` was never generated, so three LIVE shape rules were inert

- **THE DEFECT.** `MentalAttributes.geometric_awareness` is declared at
  `player_dna.py:80` with a `50.0` default. **Eight archetype templates have
  always specified a band for it** — `ball_winning_mid (60,72)`, `anchor
  (65,76)`, `box_to_box (68,80)`, `shadow_striker (72,84)`, `progressive_
  midfielder (74,86)`, `deep_playmaker (78,90)`, `regista (80,91)`, `classic_ten
  (82,93)`. `_build_mental` assigned **10 of its 11 mental fields and never this
  one**, so the templates were dead weight and the attribute read exactly
  `50.0` for every player ever created (measured `[50.0]` across 400).
- **THREE RULES CONSUMED IT AND NONE COULD WORK.** `position_engine.py:3292`
  `awareness_bonus` was **exactly 0.0**; `:3431` midfielder half-space coverage
  needs `>= 55.0` and **never fired once**; `:3519` attacker drift ran at a flat
  `awareness_factor = 0.091` for every player ever built. All three are live —
  `_midfielder_geometric_coverage` is reached from `position_engine.py:1739`.
- **THE FIX IS ONE LINE**, with a comment carrying the whole story:
  `cls._attr(arch, "mental.geometric_awareness", (50, 70), mental_age)`.
- **MY FIRST BAND WAS WRONG AND MY OWN PROBE KILLED IT BEFORE ANY GATE RAN.**
  I first used `(55, 74)` — `work_rate`'s band, on the argument that the eight
  template bands (60–93) mean "above average". Measurement: with a floor of 55
  **exactly equal to the `:3431` gate floor, 100% of 400 players cleared BOTH
  gates**, so the rules stopped selecting anyone and became *uniform* rules —
  the same inertness as the 50.0 constant, aimed the other way. **A gate has to
  reject.** Shipped `(50, 70)`, the house mental default five of that builder's
  other ten fields already use.
- **THE LESSON, and it generalises past this fix: A RULE'S GATE IS A STATEMENT
  ABOUT THE DISTRIBUTION IT WAS WRITTEN AGAINST.** Filling in a missing
  attribute is therefore a *coupled* change — you cannot generate the
  distribution and leave the thresholds untouched without re-deriving what they
  now mean.
- **AND THE GATES ARE ANYWAY CLOSE TO VACUOUS FOR THE TEMPLATED POPULATION** —
  a pre-existing property that only became *visible* once the attribute stopped
  being constant. The lowest template floor is **60**, already above the 55
  gate, so any player landing on one of the eight clears it on every draw.
  Sweeping five candidate default bands moved the combined midfielder clear
  rate only 52.8% / 64.4% / 67.1% / 74.9% / 97.0%, and only by dragging the
  template population down. **Whether `:3431` *should* select is a design
  question about a tuned threshold and is deliberately NOT answered here** —
  per the CORNER STEP 6 discipline, do not tune real behaviour to satisfy an
  assertion, and equally do not tune it to make a test pass.
- **MEASURED AFTER (400 players, 3 seeds):** median 61.1, 241 distinct values,
  min 46.0 max 90.9 — inside the same band the rest of the mental block
  occupies, not a new axis. `:3431` factor 0.000 → 0.317/0.410/0.663
  (CDM/CM/CAM); `:3519` 0.091 → 0.213–0.694; `:3292` bonus 0.000 → 0.095
  (blend 0.40 → 0.495).
- **SUITE `tests/test_dna_awareness_shots.py` (13 tests), and why it is a NEW
  FILE.** The existing suites are *green about this*: `test_positional_realism`
  and `test_shot_blocking` both build a `_FakeDNA` that sets
  `geometric_awareness` **by hand**, so they exercise the shape rules against
  a value the real builder never produced. They would pass identically if the
  builder emitted nonsense. A suite that is green and silent about a feature is
  not evidence about it — same conclusion the short-corner work reached for the
  same reason. It also carries an **AST pin** (the shot path must read
  `effective_*`, must NOT read the raw attributes), matching the
  `test_chance_provenance.py` precedent.
- **NEGATIVE CONTROL (`_diag_dna_guard.py`, 5 mutations, ALL CAUGHT, tree
  restored byte-identical at 72532 bytes).** This earned its keep immediately:
  mutation **M2 (restore the rejected `(55,74)` band) left the entire file
  GREEN** — the band, the single most consequential choice in the fix, was
  pinned by nothing. Hence `test_the_default_band_is_the_house_mental_default`.
  **A test that cannot fail is not a test, and the only way to find out is to
  run the negative control, not to reason about it.**
- **TWO OF MY OWN TEST BUGS, both the documented shape.** (1) A
  `test_a_constant_other_than_50_would_also_be_wrong` / band assertion set
  `min(vals) == 50.0` at **age 30**, which is `AGE_CURVE[range(30,33)] == 0.97`
  and therefore reads 48.7 — the peak bracket is `range(27, 30)`. The multiplier
  is not a rounding detail; the test now asserts the bracket it depends on.
  (2) The band-separation floor was set at `0.25`, which is **exactly**
  `Uniform(50,70)`'s theoretical `5/20 = 25.0%`, and a 300-sample draw crossed
  it (measured 24.3%). A threshold sitting on the theoretical value is a
  knife-edge — the same trap as the CM assertion in *WIDE PLAY* that fails by
  1.5 cm. Floor moved to 0.15 with the arithmetic in the comment.

### FIX 2 — the live shot read raw attributes, so form and fatigue could not reach it

- **`_shot_on_target_prob` (`event_chain.py`, `AttackChain`) read
  `dna.mental.composure` and `dna.technical.finishing` raw** while every other
  live consumer folds the same multipliers in (`get_shooter_quality`,
  `get_pass_accuracy`, `get_dribble_success_rate`). A tired striker in a slump
  shot identically to a fresh one in form.
- Added `PlayerDNA.effective_composure`, mirroring `effective_finishing`
  exactly (`player_dna.py:361-376` pattern), and switched the shot path to
  `effective_finishing` / `effective_composure`.
- **⚠️ SCOPE, AND IT IS NARROWER THAN IT SOUNDS. NO A/B IN THIS PROJECT CAN
  DEMONSTRATE IT.**
  - `live_performance_mult` has **NO WRITER** anywhere (dataclass default 1.0 at
    `player_dna.py:292`; one test at `tests/test_defensive_awareness.py:672`).
    So "live stamina" contributes exactly 1.0 — here and in **every**
    `effective_*` consumer, including the ones that predate this pass. **I did
    not invent a writer**: that means giving `SubstitutionController.stamina` a
    per-player, per-minute hook into the DNA, a real subsystem change.
  - `PlayerFormState.update_after_match` (`player_dna.py:212`) has **ZERO call
    sites**; `season_manager.py:632` reimplements the rating→confidence mapping
    inline. So **form cannot move within a match anywhere** — another instance
    of the pathology, recorded, not fixed here.
  - Only the production season path writes pre-kickoff fatigue
    (`auto_run_match.py:622`), and it clamps starting stamina to `[70, 100]`, so
    fatigue is capped at 30 → multiplier floor **0.955**. Form is the real
    lever at ±20%.
  - **Consequence: the scratch roster sees this fix as exact identity**
    (confidence 50 → `form_multiplier` exactly 1.00; fatigue 0 → exactly 1.00;
    live 1.0; and `x * 1.0` is bit-identical in IEEE-754). Every A/B harness
    here runs the scratch roster, so **no historical number moves** — a
    property worth stating, not celebrating, because it also means none of them
    can evidence the fix. The gate had to be a unit test plus the static
    call-site argument, not an A/B. **Do not quote a pitch impact for it.**
  - Magnitude where it *is* live: the attribute term is 0.05 weight on the SUM
    of two 0–100 values, so a ±20% form swing moves `base` by ~2.6% relative.
    Deliberately small — the right size for a state modifier.

### GATE — targeted files only, per the user's no-hour-regression rule

`test_positional_play` + `test_touchline_wide` + `test_small_game` +
`test_chance_provenance` + `test_set_piece_routines` + `test_full_wiring` +
`test_dna_awareness_shots` → **123 passed, 1 failed, 1 strict xfailed**
(3m57s). The single failure is the documented pre-existing
`test_out_of_possession_touchline_press_gated_by_press_intensity`, and a
second pre-existing (`test_preservation_properties::test_position_engine_
drift_toward_home`, x-axis) was found and **bisected, not assumed**.

- **THE BISECT IS THE PART WORTH COPYING.** Both "documented pre-existing"
  failures were last verified *before* this fix, so "documented" is not evidence
  that a later change did not cause them. `_diag_drift_bisect.py` applies the
  **exact pre-fix state** (`geometric_awareness = 50.0`) to the real file, runs
  the real pytest node **3× per arm**, and restores byte-exactly. Both nodes
  failed **3/3 in both arms** → pre-existing, and the fix is not the cause.
  The same argument retired a third failure (`test_preservation_properties::
  test_realistic_shot_woodwork_and_rebound_preserved`).
- **THE FIRST PROBE AT THIS QUESTION MEASURED NOTHING AND WAS DELETED.**
  `_diag_drift_regression.py` rebuilt the fixture on a one-player team and the
  player never moved at all (`start == final`, 4.27 → 4.27 in **both** arms),
  so it could not distinguish the two hypotheses. **A probe that reproduces
  nothing is not evidence in either direction** — deleted rather than left to be
  quoted. It also silently disagreed with the real test's numbers, which is the
  signature to watch for.
- **Likely overlap, recorded not chased:** `:3519` attacker drift steers the
  same axis as `winger_behavior.should_cut_inside`, which already pulls a wide
  player inward from archetype `byline_instinct`. **Two systems now pull the same
  man inward and neither knows about the other.** This is worth a probe before
  anything touches the band — and it is a candidate explanation for the
  un-isolated width-collapse tail above.
- **NOT GATED BY THE 26/27 REGRESSION**, by the user's standing instruction.

## WIDTH GATE (2026-10-05) — FIX 1 WIDENS THE SHAPE AND STILL DOES NOT FIX THE BAND

The user's complaint — *"i dont want it collapse into a narrow block sometimes
and should be in that 45-55 band"* — was **UNRESOLVED** when fix 1 landed, and
the reason it stayed unresolved is uncomfortable. All three revived rules
(`_midfielder_geometric_coverage`, `_cam_pocket_roam`, `_attacker_space_run`)
sit inside ONE `if in_possession:` block at `position_engine.py:1738`, and
`:3519` attacker drift steers the **same axis** as
`winger_behavior.should_cut_inside`. Inward is the direction of the collapse. So
regenerating `geometric_awareness` could have made the original complaint
**worse**, and no unit suite can see that. Probe: `_diag_width_gate.py` /
`_diag_width_gate.txt` (one command to re-run).

- **THE ARM IS THE PRE-FIX STATE BY EXACTLY ONE BIT OF LOGIC.** `PlayerDNA._attr`
  (`player_dna.py:1063`) calls `random.uniform`, so naively writing `= 50.0`
  into the pre-fix arm **removes one draw per player per match** and every later
  number in the match shifts. An A/B built that way measures STREAM SHIFT, not
  awareness — the same class as `_STREAM_PARITY_DRAW`, and the reason that line
  exists. The pre-fix arm here still makes the identical `_attr` call and
  **discards** the value, `(lambda _v: 50.0)(cls._attr(...))`. The arms differ
  in exactly one thing: whether the drawn value is used.
- **FOUR GATES BEFORE ANY VERDICT IS PRINTED**, because this probe's own first
  run produced a clean table out of a broken instrument:
  1. the anchor must exist in `player_dna.py` — **a non-matching anchor mutates
     nothing and then reports a clean PASS, the worst possible outcome**;
  2. the mutant must **compile** (the first two attempts died here: a nested
     f-string, then a bare statement inserted into a constructor *argument
     list*, which cannot accept one);
  3. the mutation must not be a no-op;
  4. the pre arm's `geometric_awareness` must be **exactly** `[50.0]`, the two
     arms' frame digests must **differ** (the lever reached the pitch), and the
     paired-tick count must clear 200 — **non-vacuity**, because a percentile
     over a collapsed tick grouping is a number about one dictionary, not a
     distribution.
- **⚠️ THE `_offball_tick_seq` TRAP.** `_offball_tick_seq` is a plain `int`
  **attribute** (`match_engine.py:1990`), not a method. `_diag_wide_target.py`
  **called** it, so `int(fn())` raised `TypeError`, the probe's own
  `except Exception: return -1` **swallowed it**, and every wide player in the
  match collapsed into **one** tick group. The symptom was not a crash — it was
  `wide-ticks=1` and a p05 over a single overwriting dict, printed beside a
  confident summary line. **A swallowed exception that silently degrades an
  instrument is worse than a crash.** The replacement reads the attribute and
  **raises** on a method-shaped value, so the mistake cannot recur quietly, and
  gate 4 above exists because of exactly this.
- **GUARD: `repo_digest` OVER THE ENGINE'S STATIC IMPORT CLOSURE (54 files),
  seeded from `match_engine.py`.** Two earlier scopes were both wrong in
  instructive ways. (a) *Every* `.py` in the repo root — too broad; it aborted
  because I edited a **probe** the match never imports, and **a guard that fires
  on irrelevant edits gets switched off, and then it fires on nothing.** (b)
  `import match_engine` + `sys.modules` — 23 files, and it **MISSED**
  `event_chain.py`, `player_dna.py` and `attacking_matrix.py`, because those are
  imported lazily *inside functions*. That is the "one-file hole" this project
  has now hit **four** times: the guard would have missed precisely the edits it
  exists to catch. A static `import` walk is an OVER-approximation of the
  runtime closure (it includes imports behind `if` branches), which is the safe
  direction for a guard, and costs microseconds instead of a match. **Proven in
  five directions**: catches an `attacking_matrix.py` edit, catches an
  `event_chain.py` edit, ignores probe edits, ignores its own mutation, and the
  arm mutation moves the inclusive digest but not the default. It earned its
  keep already: an earlier gate aborted on a `NameError: name 'W' is not
  defined` caused by a mid-save by the parallel session, and correctly reported
  **NO VERDICT** rather than a number.
- **THE RESULT — fix 1 DID WIDEN THE SHAPE, IN EVERY PERCENTILE, ON BOTH PAIRS:**

  | | arm | n | p05 | p10 | p50 | p90 | in 45–55 |
  |---|---|---|---|---|---|---|---|
  | wingers TARGET | pre | 47,960 | 0.6 | 1.6 | 26.6 | 35.0 | 2.1% |
  | wingers TARGET | post | 50,710 | **3.1** | **4.7** | **30.6** | 35.9 | 0.6% |
  | wingers ACTUAL | pre | 47,960 | 1.3 | 2.3 | 25.6 | 35.9 | 1.9% |
  | wingers ACTUAL | post | 50,710 | **3.0** | **5.2** | **31.0** | 36.5 | 1.5% |
  | fullbacks TARGET | pre | 43,711 | 0.6 | 1.4 | 18.5 | 42.3 | 6.3% |
  | fullbacks TARGET | post | 44,483 | 0.7 | 1.1 | **28.3** | 42.3 | 2.5% |
  | fullbacks ACTUAL | pre | 43,711 | 0.8 | 1.4 | 20.5 | 42.8 | 5.4% |
  | fullbacks ACTUAL | post | 44,483 | **1.4** | **2.4** | **24.3** | 42.1 | 3.3% |

  Every wide role also moved toward the touchline, measured per-player at
  48–53k samples each (so no pairing assumption enters it): **LW 19.4 → 17.6,
  RW 19.8 → 18.5, LB 15.9 → 15.3, RB 16.9 → 15.8 m**. **The collapse did not get
  worse; my worry that it would was wrong in direction** — and the source said
  so, because in `_attacker_space_run` the clamp `max(0.2, awareness_factor)` was
  **already binding** at the old flat 0.091, so the pre-fix value contributed
  nothing distinguishable; the post-fix median 0.29 moves `effective_pull`
  0.020 → 0.029, i.e. **+45%, not tripled**. For LW/RW the flank candidate wins
  by **6.25×** on `flank_weight` (3.0 vs 0.6/0.3), independently of awareness.
- **⚠️ AND THE BAND SHARE FELL IN ALL EIGHT ROWS. THAT IS THE ACTUAL FINDING.**
  The user's target is a 45–55 m *band*; median-lifting does not deliver a band,
  it **actively reduces the share inside it** (wingers 1.9% → 1.5%, fullbacks
  5.4% → 3.3%). The reason is the ceiling: **p90 is 36.5 m**, so with the top of
  the distribution 8.5 m below the band's floor, the band is *geometrically*
  unreachable. Only 1.5–5.4% of samples are in band at either arm. **Until the
  shape can be genuinely wide at its widest, any fix that raises the mean makes
  the stated symptom worse.** CORNER STEP 6's discipline, again: the thing
  measured is not the thing asked for.
- **THE FULLBACK ROW IS NOT A RESULT, IT IS A WARNING ABOUT THE METHOD.**
  Fullbacks moved **−8.7 m in possession** and **+13.4 m out of the block**, and
  fix 1 provably **cannot reach the out-of-block phase** — all three rules are
  `if in_possession:`. The control therefore FAILED. The honest reading: the two
  arms are *different matches* (2,525 vs 2,749 events) and per-position
  differences are entangled with trajectory divergence — the first divergence
  compounds into a different shape by minute 30. The wingers' movement is
  **consistent with** the mechanism; it is **not isolated** by this experiment.
  **One seed, one match per arm.** Direction is internally consistent across four
  percentiles; magnitudes are a single sample. Do not quote them as established.
- **SECTION F WAS PRINTED AS A CONTROL AND IT DOES NOT SUPPORT THE ATTRIBUTION
  IT WAS ASKED FOR.** CK35 64.0% → 66.1%, live run target 32.4% → 35.6% — both
  moved, so "it came from `:3519`" is unsupported. Recorded rather than quietly
  dropped: **the one line that would have identified the mechanism is the one
  line that moved.**
- **THE IN-BLOCK PHASE IS WHERE THE TAIL IS WORST, AND IS UNMEASURED.** 26,993
  fullback-pair observations out of possession, p50 36.6 m post-fix, band share
  3.1% — and fix 1 provably does not touch it. That is the next measurement, not
  another constant.
- **WIDTH IS STILL SHORT. THE USER'S OBJECTIVE IS NOT MET.** Median separation
  31.0 m against a real 45–55 m; wide players 15.3–18.5 m off the touchline
  against a real 3–12 m. **This section does not claim otherwise.**
- **CONDUCT: I OVERRAN THE APPROVED CPU BUDGET ON THIS, ~6.5 MATCHES
  (~10–12 min) AGAINST ~3 MIN, AND ~3.5 MIN OF THAT WAS MY OWN MISTAKE.**
  (a) I tested `--child` dispatch by **actually invoking it**, which runs a real
  match; (b) I **imported** `_diag_wide_target.py` to unit-test a helper, and
  that module runs two matches **at import**. Three zero-match cycles cost
  nothing. **Check what a command DOES before running it** — a probe file is not
  a library unless someone made it one.

## ⛔ SEASON BOUNDARY — MD1–MD5 CHANCE CREATION IS FABRICATED. MD6 ONWARDS IS NOT.

**The 2026-27 season has played 5 matchdays. Everything recorded in MD1–MD5
about chances created, shot assists and assists is FABRICATED and must never
be compared with anything from MD6 onwards.** MD6 is the first matchday
simulated by the honest code. That is a hard boundary in the data, not a soft
one, and it is the reason this note exists rather than a line in a changelog.

**WHAT WAS FABRICATED, precisely.** Up to and including MD5:

- `CHANCE_CREATED`'s **origin** was `location_x = x - random.uniform(5, 20)`
  — a draw from the **global football RNG**, matching no pass that player ever
  made (0/15 against every real pass origin by that player).
- `CHANCE_CREATED`'s **`player`** was `_pick_creator`, a role- and
  distance-weighted **RANDOM DRAW over the whole squad**. It had no
  connection to who actually passed the ball.
- The **assist on the GOAL event's `secondary_player`** — the field the Goals
  sheet and `alltime_db` ASSISTS actually read — was the same draw. So the
  headline assist number was fabricated too, not just the upstream event.
- The **shooter** on the bulk of shots came from `_pick_shooter`, the same kind
  of draw, because the chain that fires the shot was never told who had the
  ball.

**WHAT IT IS FROM MD6.** The passer is the **previous value of `last_player`
at the carrier change** — a real event, not a draw. The shooter is the man
who actually held the ball, threaded through `ChainDispatcher.attack(shooter_name=…)`
and resolved by a strict `_named_shooter` (an outfielder in the squad passed
in; a stale name returns `None` rather than shooting with a ghost).
`ChainResult.ball_carrier` / `ball_carrier_passed_by` carry it, stamped at
`_absorb_chain`, *"the single point where chain outputs become match facts"*.

**EXPECT FEWER ASSISTS FROM MD6, AND THAT IS THE CORRECT RESULT, NOT A
REGRESSION.** `_resolve_assister` has **no fallback, on purpose** — do not add
one. Measured on real matches, 6 of 9 goals are **genuinely unassisted** (the
scorer won the ball and dribbled it in, so there is no pass to be the key
pass). An unassisted strike emits **no `CHANCE_CREATED` at all** rather than
crediting the shooter, because crediting him would make the engine disagree
with `ChanceCreationLedger._find_setup_pass` (which correctly returns `None` for
a dribble) about the SAME shot — which would break the relationship the user
explicitly protected. **The shot still appears.** A rise in assists at MD6 is
a red flag, not a success.

**WHAT MUST NOT BE DONE WITH MD1–MD5 DATA:**

1. **No trend line across the boundary.** Pass-end → shot-origin gap fell from
   a median 30.4 m to 4.4 m (that 4.4 m is the carry, measured). Reading that
   as "the team improved at MD6" is reading a code change as a result.
2. **No xA / chances-created accumulation across the boundary.**
   `creation_event.xa` is `xg` by definition, but it was banked against the
   DRAWN creator, so it is only real from MD6. Season totals mixing the two are
   meaningless.
3. **No player leaderboard for chance creation, MD1–MD5.** The names in it came
   from a draw weighted by role and distance to the ball. Assists in
   `alltime_db` for those five matchdays are the same fiction and need purging
   or flagging before any career stat is published.
4. **The fabrication audit is NOT complete.** Do not report "chance creation is
   honest" — only "the shooter's pass is now real where the chain knew it".

**SEPARATE AND ALSO TRUE:** the *receiving* end was never the problem. `PASS`
events carried a real destination and a real `secondary_player` = receiver all
along (655/680 sampled). "Passes have no start point" was false; the fabricated
start point was on the `CHANCE_CREATED` event, not on the pass.

## WIDE PLAY — do the wingers and full-backs actually hold the touchline? (2026-10-04)

Asked after watching a match: *"do wingers/fullbacks stick to the touchline to
spread play like modern wingers? cause i see sometimes drifts inside, sometimes
its well, maybe they follow managers instructions i dont know."*

**Short answer: yes, the model exists and it does reach the pitch. The drift is
real but TRANSIENT, not resident. The manager CAN move them and, measured, DID
NOT in this match. And the pitch is under-stretched by about 8 m versus real
football.** Probe: `_diag_wide_channel.py` (writes `_diag_wide_channel.txt`).

**THE MANAGER CHANNEL IS REAL — my grep was wrong, and the probe caught it.**
`grep manager_brain.py 'width|wide|touchline|flank'` returns **no matches**, and
I nearly reported "the manager does not influence this". `manager_brain.py` is
the neural module; the tactical profile it feeds is a different object, and it
carries **`width`** (`match_engine.py:716`, `0=narrow, 1=wide`, 0.30–0.90 per
philosophy). There are **TWO** manager-side width channels that reach the wide
players' anchors:

1. `FormationEngine.compute_home(pos, profile, slot)` reads `profile.width` —
   **+3 m outward at width=1.0, −3 m inward at width=0.0**
   (`position_engine.py:362-371`). Both sides here ran BALANCED → width=0.5,
   the exact neutral midpoint, so channel 1 contributed **zero** this match.
2. `coach_instructions.instructions_for_manager(manager, pattern)` reads
   `manager._current_posture` and returns a **`width_cmd` (+1 stay wide, −1
   tuck in)**, applied through `PositionEngine.apply_coach_width` →
   `_recompute_homes`. The whole command is **±2.5 m / ±1.5 m** on wide roles
   (`position_engine.py:753-759`). Gated on `USE_MANAGER_BRAIN` (True) and on
   the posture being ATTACK or DEFEND.

**MEASURED: `width_cmd` was 0.0 in 3,272 of 3,272 wide-player samples — both
teams, all 409 samples.** Since `_ATTACK→+1`, `_DEFEND→−1` and BALANCED emits
none, the manager's posture never left BALANCED at any point the width command
is evaluated. **The user's instinct ("maybe they follow manager's instructions")
is the right question and the right subsystem — it just did not fire here.**

**THE DRIFT IS TRANSIENT, NOT RESIDENT — which is why it reads as "sometimes".**
Sampled live by wrapping `MatchEngine._offball_run`, reading `current_y` AND
`home_y` **at the same instant** (see the trap below). `in_ch%` = share of
samples within the engine's own `FLANK_CHANNEL_HALF_WIDTH_M` (10 m), i.e.
`winger_behavior.in_flank_channel`:

| player | pos | archetype | med drift | p90 | p99 | in_ch% | max | excursions off-flank | typical ON-flank spell |
|---|---|---|---|---|---|---|---|---|---|
| Jake Jivan | RW | traditional_winger | **2.7** | 15.0 | 20.8 | **80.7** | 22.1 | 24 | 8 samples ≈ 32 s |
| Moffart Randier | RB | attacking_fullback | 5.5 | 13.6 | 35.2 | 77.0 | 45.5 | 24 | 7 ≈ 28 s |
| Coupey-Daniels | LW | traditional_winger | 5.1 | 22.7 | 45.9 | 61.4 | 47.8 | 41 | 4 ≈ 16 s |
| Denis Doxel | RW | traditional_winger | 7.6 | 17.8 | 38.1 | 59.2 | 43.4 | 46 | 4 ≈ 16 s |
| Lewis Driutt | LB | attacking_fullback | 7.7 | 20.6 | 32.8 | 61.1 | 35.6 | 27 | 6 ≈ 24 s |
| Nashan Gwum | RB | attacking_fullback | 7.9 | 30.1 | 44.7 | 63.6 | 52.5 | 38 | 5 ≈ 20 s |
| Francis Sesina | LB | **inverted_fullback** | 7.5 | 27.5 | 40.6 | 64.3 | 42.7 | 36 | 5 ≈ 20 s |
| Pablo Antares | LW | **inverted_winger** | 7.8 | 20.6 | 34.4 | **55.0** | 38.7 | 45 | 3 ≈ 12 s |

He is wide **55–81% of the time**, in spells of 12–32 s, and goes inside **24–46
times a match** for a shorter spell each time. That is a player holding a line
and periodically cutting in — which is real football, and is a completely
different finding from a player parking in the half-space. **A median could not
have distinguished them; the run-lengths are what makes the verdict.**

**PER-PLAYER DNA EXPLAINS THE SPREAD, and it is per-player.** The archetype sets
`byline_instinct`, which sets `should_cut_inside`'s baseline
`(1 − byline_instinct) × 0.60`: traditional wingers land at **0.13**, the
inverted winger at **0.48** — and inverted_winger Antares has the worst
in-channel share on the sheet while traditional_winger Jivan has the best.
**So part of what the user sees as inconsistency is the engine correctly playing
two different player archetypes.** `inverted_fullback` Sesina carries
`flank_commitment = 0.50` against 0.75 for an attacking full-back, the weakest
commitment of the eight, and he has the worst p90.

**THE PITCH IS UNDER-STRETCHED — the one number that is a genuine gap.**
Side-to-side separation, sampled at the same instant for both flanks:

| | measured | real PL |
|---|---|---|
| winger-to-winger (LW–RW) | **37.8 / 38.3 m** | ~45–55 m |
| full-back-to-full-back | **34.8 / 38.6 m** | ~45–55 m |
| pitch | 68 m wide | 68 m |

A **~8–16 m shortfall**, and Natrican's p10 of **21.9 m** (full-backs **11.4 m**)
means the shape genuinely collapses into a narrow block sometimes. **Not
fixed** — this is a calibration question with a genuine trade-off (wider shape =
more space behind, fewer through-balls), not an oversight, and it needs its own
measurement across seeds before anything moves.

**THREE PROBE ERRORS OF MINE, all instances of documented traps, recorded
because each one produced a clean wrong answer.**

- **I assumed `_offball_run` was a 0.1 s tick.** It is called 409 times over 90
  minutes — **median gap 4.1 s** (min 0.1, max 114.0). Every duration the first
  version printed was wrong by ~40×. This is the `top_speed_mpm` trap verbatim
  (AGENTS.md MEASUREMENTS: *read the units from the source, not the field name*).
  The probe now **measures** the cadence from `state.match_clock_s` deltas and
  prints it, and run lengths are returned **in samples first** so the raw count
  stays visible if the cadence assumption ever changes.
- **`WingerSpatialProfile` / `FullbackSpatialProfile` do NOT store the DNA they
  were built from.** The first style table read `p.dna.archetype`, got `None`
  from `getattr`, and printed **`?` for all eight players** — which reads as
  "no archetype" rather than "wrong attribute". `PlayerProfile.dna.archetype` is
  the real source, read via `engine.squads[team]["starters"]`. (`FullbackSpatialProfile`
  has no `byline_instinct` at all, so `cutBase` is genuinely undefined for
  full-backs — left as `nan`, not filled in.)
- **Index-aligned pairing is not sample-aligned.** Team width was first
  reconstructed from per-player list **indices**, which silently misaligns the
  moment one player is skipped (a `None` coordinate shifts everyone after him in
  roster order). It reported *"never both flanks occupied at the same tick"* —
  a probe artefact that reads as a football finding. Fixed by keeping one dict
  per `_offball_run` **call**, so both flanks are same-instant by construction.
- **`home_y` is MUTATED mid-match** (`_recompute_homes`, called by
  `apply_coach_width` / `apply_attack_pattern` / every stance change), so
  reading it after `simulate()` returns attributes final-whistle state to a
  mid-match moment — and would have silently mislabelled every coach-width
  excursion. The probe reads `current_y` and `home_y` together inside the live
  loop. Same family as the seven existing instances of *read position state
  after the match*.

## CB ATTACKING CEILING — the clamp existed and was possession-gated (2026-10-11)

Reported from a real pass map: a centre-back (Juan Massey, Pearls) with **39/58
passes in the opponent half, 20 in the final third (x≥70 m), 22/26 carries in
the opponent half clustering at x=75–88 m, 2 shots, offside twice at x=86.2**.
A CB playing as a deep-lying playmaker, in the opponent's box.

- **THE MECHANISM ALREADY EXISTED.** `PositionEngine.CB_MAX_ADVANCE_NX = 58.0`
  (`position_engine.py:2499`) + `cb_advance_clamp` (`:2501`), called at exactly
  ONE site repo-wide: `match_engine.py:2961` inside `_offball_move_player`.
  Nothing new was added — no constant, no rule.
- **THE DEFECT WAS ONE WORD: `and has_ball`.** The gate read
  `if pos == "CB" and has_ball:`, so the ceiling bound only WHILE the CB was the
  ball carrier and was OFF for the other ~98% of ticks. Out of possession the
  CB's shape anchor is his own minute-start position (`_minute_start_snapshot`),
  and `MatchEngine` never calls `drift_minute`, so **no home-pull restoring
  force runs live** — the anchor creeps deeper each minute and never returns.
  A one-way ratchet. Fixed to `if pos == "CB":`, binding in every phase exactly
  as rest defence does (`position_engine.py:2963-2965` neighbour).
- **SET PIECES STAY EXEMPT, and for free.** The `if _sp_slot is not None:`
  branch sits ABOVE the clamp (`match_engine.py:2950-2953`), so a corner/FK slot
  is returned before the ceiling can touch it. No new exemption logic.
- **TWO HYPOTHESES KILLED BEFORE ANY CODE CHANGED**, both by reading rather than
  guessing. (H2) *The label is `CB1`/`CB2`, so the gate never matched* — FALSE
  for the production path: real-roster CBs normalise through `roster_loader
  .POS_MAP` (43–58) → `PlayerRecord.engine_pos` → `SquadBuilder` → `dna.position
  → `PlayerProfile.position` == **`"CB"`**. The clamp DID match. See *POSITIONS
  ARE NUMBERED* for why the same trap is real for `.rstrip(digits)`-style
  guards elsewhere. (H1) *On-ball players skip the whole integrator* — TRUE and
  expected: `_offball_move_player` returns early at `match_engine.py:2755`
  (`if pname in self._on_ball_this_minute: return`), populated from
  `_ON_BALL_EVENTS` at `:5773/5775`, reset per minute at `:3793`. That early
  return is why the `has_ball` arm of the gate was doubly redundant.
- **MEASURED** (one real match per arm, seed 4000, Pearls v Justice; probe
  `tools/diag/_diag_cb_ceiling3.py`, reusing `_diag_chance_coords.build_pair`):
  team CB passes in the final third **28 → 1**; opponent-half passes 91 → 65;
  CB deep position ticks (>58) 147,564 → 39,288 (**−73%**); deep
  out-of-possession 84,821 → 16,623 (**−80%**). Juan Massey: 57 pass / 24
  opp-half / **7 final-third** → 38 / 17 / **0**, carries 16 → 14, **no
  final-third carries**, max carry nx 62.6.
- **THE RESIDUAL, AND WHY IT IS NOT THE SYMPTOM.** After the fix, other CBs
  still show max pass nx 82.3 and max carry nx 93.6, with 4 carries + 1 pass in
  the final third. **Read the source before chasing it:** a PASS event's
  `location_x` is the chain anchor (`_starting_anchor` = `0.75*state.last_ball_x
  + 0.25*style_x`), and a CARRY event's `location_x` is `x` — the carry's START
  (`event_chain.py:2079`) — i.e. **the BALL's position, not the passer's or
  carrier's**. So a "CB carry at nx 93.6" means *the ball was at 93 and the CB
  was the chain's carrier, with his advance already capped at 60 on success*
  (`:2011-2012` main carry, `:1446-1447` micro-carry — the 60/45 pair that
  predates this work). The deep number is the ball wearing a CB's name; the CB's
  own tracked position is now bounded in every phase. Same lesson as *the
  pass-map origin is NOT the passer's position*.
- **ONE GENUINE LEAK LEFT OPEN DELIBERATELY.** `event_chain.py:2080` is
  `end_x = new_x if carry_success else cls.clamp_x(fail_x, attacks_right)` — the
  FAILED-carry branch is only `clamp_x`'d, never CB-capped, so a dispossessed
  CB's loose ball can land deep and become the next anchor (`last_ball_x`
  `match_engine.py:5869-5873`). Closing it changes `last_ball_x` and therefore
  simulation data, so it needs its own gate. **The user reviewed this and chose
  to STOP** (2026-10-11). Also open, and a different question: whether the chain
  should pick a CB as its carrier at a deep anchor at all — that is
  `_pick_builder`, not the position engine.
- **GATE (72.8 s):** `test_cb_ceiling` (new, 7) + `test_small_game` +
  `test_positional_play` + `test_touchline_wide` = **61 passed, 1 xfailed,
  1 failed**. The failure is the documented pre-existing
  `test_out_of_possession_touchline_press_gated_by_press_intensity` (an LB
  assertion in a test that never calls `_offball_move_player`); the change is
  provably CB-only, so it cannot be this work.
- **`tests/test_cb_ceiling.py`** pins `cb_advance_clamp` behaviourally (binds
  above the ceiling, passes through below, depth-only) **plus an AST guard that
  the live call site is not possession-gated** — because the gate lives one file
  away from the clamp and no behavioural test can see it. The guard asserts
  non-vacuity first. Same reasoning as the `attacks_right` / `shooter_name`
  guards: the omission is a defaulted or dropped condition, invisible at the
  behavioural layer.
- **CAVEAT ON THE NUMBERS:** the fix shifts the RNG stream, so the two arms are
  different matches (2967 vs 2657 events; 2-1 vs 1-1) — hash-order-free
  (crc32 since 2026-10-04) but not the same fixture. Direction is consistent
  across four surfaces and mechanistically explained; each delta is one sample.
  Multi-seed not run (cost, and matches are not cheap here).
- **⛔ SUPERSEDED 2026-10-15 — the clamp was never the problem, it was the
  SATURATION.** The CB ceiling only ever *bound* (70.4% of calls) because the
  shape chain was feeding it a target 30–38 m past the formation post; the
  centre-backs were parked ON the ceiling at p25/p50/p75 = 55.5/56.8/57.9, not
  holding a line. Both switches now report 0.0% binding. See *SHAPE ANCHOR +
  MIRRORED BLOCK* below. The clamp test above is still correct; its premise
  was not.

## SHAPE ANCHOR + MIRRORED BLOCK + STYLE-BLIND DEFENCE (2026-10-15, FIXED)

Commit `361cd0b`. Started from one user complaint — *"Left back plays super
high almost like a LW (same for RB and RW)"* — and produced **three defects**,
plus one retraction of my own reasoning that a future session should not
repeat. This is the single most consequential spatial finding in the project
and it **supersedes the WIDTH GATE and CB CEILING sections above** — both were
measuring symptoms of one cause.

### ⛔ READ THIS FIRST — MY BENCHMARK WAS WRONG AND I ALMOST SHIPPED TO IT

I used "real PL: LB 22–30, CB 20–26" throughout, from memory. **22–30 is a
deep-block line, not a fullback.** Actual data (Opta / The Analyst, Premier
League 2024-25, average full-back **touch** location):

| | metres from own goal |
|---|---|
| Man City | 60.1 |
| Tottenham | 59.4 |
| **league average** | **53.2** |
| Chelsea (deepest team) | 46.8 |

Robertson's touch position moved 56.4 → 51.5 m between two Liverpool seasons
and was written up as a style shift — the real range is narrow and HIGH. My
band was off by ~25 m, which is roughly why I first reported "a winger with a
different name". **Opta touch location is over EVENTS; a 10 Hz position median
is over TIME.** Same instrument, different aggregation. They are not
interchangeable and a benchmark that does not name its frame is not a
benchmark. Get real data before quoting a band. (Source: theanalyst.com
"Premier League full-backs", 2024-10-15.)

### DEFECT 1 — the shape anchor was a PERSISTENCE TERM (the root cause)

`match_engine._offball_move_player` built its target from:

```python
ax, ay = self._minute_start_snapshot.get(pname, (st.current_x, st.current_y))   # was 2757
tx    = ax + (ball_x - ax) * 0.14
```

and `_minute_start_snapshot` is rebuilt **every minute** from
`snapshot_positions(team)` — the player's **TRACKED position**, not
`state.home_x`. So depth was 86% *"wherever you already were"*, and because that
value became next minute's anchor, the recursion's **only fixed point is the
ball's x from any starting position** (`_diag_shape_fixed_point.py` prints the
algebra; every outfield role converges to the ball from any start).

**The decomposition is what proves it** (`_diag_chain_attrib.py`). The anchor
reached the chain already **30–38 m past the formation post**, while everything
downstream was correcting single metres:

| term | Δ nx (Frost) |
|---|---|
| **anchor itself, before any rule votes** | **+37.6** |
| compaction (0.14) | −1.3 |
| danger squeeze + cross crash | −4.6 |
| run blend | +5.6 |
| **depth anchor (new)** | **−7.4** |
| CK36 / CK37 / CK38 | 0.00 |

The chain started 37 m wrong and spent its whole budget arguing about 5.

**The y axis never had this hole**, and that is the whole explanation for why
*WIDTH* was only ~8 m short (WIDTH GATE, 2026-10-05) while *DEPTH* was fully
collapsed: `CK35` is the one live rule reading an **absolute** formation
reference (`st.home_y`). The x-axis twin existed once — `drift_minute`'s
forward x-anchor — and MatchEngine never calls `drift_minute`.

**FIX:** `PositionEngine.SHAPE_ANCHOR_FORMATION` (default `True`) makes the
chain compact from `state.home_x/home_y`. The compaction then reads as what it
always meant — a team shifting toward the ball — rather than a per-tick
ball-tracking recursion. `_minute_start_snapshot` is **NOT** removed: the
involvement gate at ~2981 still uses it, where *"where was this player when
the minute began"* is a legitimate question about persistence.

### DEFECT 2 — `defensive_block_target` had its depth axis MIRRORED

```python
tx = (self.PITCH_X_MAX - depth if attacks_right else depth)   # was WRONG in both branches
```

`depth` is documented as measured **from the player's own goal** (and
`LB_BACK_DEPTH = 19.0` means it). So a low block's back four stood at **x=86**
attacking right — inside the *opponent's* penalty area — and at **x=19**
attacking left. Verified 6/6 in both directions by `_diag_block_frame.py`,
which checks against the constant's own documented meaning.

**Why it survived, and this is the transferable lesson:** the comment directly
above that function *already recorded the symptom* — *"the shape reached 47.5 m
from its own goal when the block slot was 19 m"* — and attributed it to a blend
weight losing to a competing value. The blend weight was raised. The mirror was
left. **A tune applied on top of an arithmetic error looks like progress for
exactly as long as the error is small relative to the gain.**

### DEFECT 3 — the defensive shape was STYLE-BLIND

`formation_stance_for` takes a `profile` and its docstring claimed it read the
style. **The body reads `state.goal_difference` and `state.minute` and never
`.style`.** So the **attacking** half of the shape was style-driven
(`pattern_for` → `_STYLE_PATTERN_POOLS`) and the **defending** half was not: a
park-the-bus side and a gegenpressing side held an identical back line until
minute 60. Measured confirmation: two tactical frames produced per-role lines
within **0.3 m** of each other.

**FIX:** `tactical_shapes.style_defense_deltas`, threaded through
`apply_formation_stance(style=...)`, **stacked under** the game-state stance
(style → stance → man-down → fatigue, all additive). Four archetypes keyed to
the ordering the attack pools already imply:

| archetype | styles | LB | CB | ST |
|---|---|---|---|---|
| **sunk** | park_the_bus, ultra_defensive | −8.0 | −6.0 | **−3.0** |
| **low** | defensive, fluid_counter, route_one | −4.0 | −3.0 | −2.0 |
| **neutral** | balanced, wing_play | 0 | 0 | 0 |
| **high** | tiki_taka, gegenpressing, attacking, … | +4.0 | +3.0 | +2.0 |

Two deliberate choices: **magnitude under the stance tables** (game-state
stances run −20…+17 m on a fullback; a style bias able to outvote
`SEE_IT_OUT` would make a side abandon its identity to protect a lead), and
**the striker holds up** (`park_the_bus` moves LW −10 but ST only −3, copying
the principle `SEE_IT_OUT` already encodes — a block whose front pair all drop
has no outlet).

### ALSO SHIPPED — `DEPTH_ANCHOR_ENABLED`

A bounded pull toward the formation post on **x** (the mirror of `CK35`) for
CB/LB/RB, banded on depth drift. It **yields to a live run target** — an
overlap is a fullback getting to the byline to cross, not drift, so the
backstop's remaining job is drift with no decision behind it. It is now a
backstop doing ~a third of the work it did when the anchor was broken; **its
weights were deliberately NOT retuned** afterwards, because restoring the old
magnitude would reintroduce the servo through the back door.

### MEASURED (seed 777, correct frame, `_diag_depth_anchor_ab.py`)

Frame matters: Hartwell **ATTACKING/HIGH_PRESS** v Thornfield
**FLUID_COUNTER/COUNTER** — `run_match.py`'s own constants, i.e. the frame the
export was produced under. **The probe now defaults to that frame** via
`--home-style/--away-style/...`; the first version hard-coded BALANCED/MIXED.

| | before | after |
|---|---|---|
| `cb_advance_clamp` binding share | 70.4% | **0.0%** |
| CB median depth | 56.8 (on the ceiling) | **33.8** |
| Darius Frost (LB) median | 67.4 | **35.6** |
| Frost passes from final third | 65% | **20%** |
| Frost backward pass share | 61% | **20%** |
| Frost median pass origin | 74.8 m | **41.1 m** |
| ST / LW median in possession | 55.8 / 62.2 | **78.3 / 71.2** |
| width control (untouched axis) | — | flat |

### ⛔ RETRACTED — "the CAM's pass map got worse", and my wrong hypothesis

**Kwame Asante is NOT a defect and NO change was made.** I called 59.3 → 46.2 a
regression. That was wrong twice over:

1. It compared **two different tactical frames** (BALANCED/MIXED vs the
   FLUID_COUNTER/COUNTER his export actually ran). **Check the frame of a
   number before comparing it to anything.**
2. Even frame-matched, 60.6 and 48.4 were **two different matches**, not a
   controlled pair. No pre-fix arm was ever run under that frame.

The instrument that settles it is the **receipt/origin join** — a CAM's
position and his ball-contact depth are one question, not two:

| player | pos | receipt median | pass-origin median | gap |
|---|---|---|---|---|
| **Kwame Asante** | CAM | 46.2 | 48.4 | **−2.2** |
| Kofi Mensah | CAM | 51.2 | 51.6 | −0.4 |
| Nico Strauss | ST | 50.8 | 54.8 | −4.1 |

He plays from within 2.2 m of where he receives — the positioning is doing what
it is told — and sits **0.4 m off his team's own ball depth** (Thornfield 48.8).
His pass map *is* his team's. Both CAMs are at **0%** of their team's
final-third passes while the ST takes 26% and the RW 19%: that is the attack
selecting wide and forward players. And Asante has **8% of his receipts** in
the final third but **0% of his passes** from there — which is a #10 shooting
rather than passing once he is in the box, not an absence.

**My second hypothesis was also wrong.** I claimed the depth anchor was vetoing
legitimate overlaps. Gating it off during live runs moved Frost **36.1 → 33.9** —
runs only fire ~21% of the time. **The measurement killed it; I should have run
it before asserting it.** Recorded because it is the second time this session
that an asserted mechanism had to be walked back.

### ⛔ STILL OPEN — the `0.14` compaction coefficient is never fitted

Frost sits at **41.1 m** median pass origin against a real **53.2 m**. He is
now slightly on the *deep* side — I traded "never came back" for "never quite
arrives". Calibrating `COMPACTION` needs a **multi-seed sweep** against the
Opta benchmark (~10 matches) and is deliberately **not attempted**. It is a
refinement, not a defect, and it will **not** touch the three bugs above.

### REUSABLE INSTRUMENTS — use these before writing a new probe

| probe | what it answers | cost |
|---|---|---|
| `_diag_depth_anchor_ab.py --arm off\|on\|anchor\|anchor_only` | the master A/B: per-role depth, clamp saturation, target-vs-position lag, pass origins, **team-vs-player comparison (s7)**, **receipt/origin join (s8)** | 1 match |
| `_diag_chain_attrib.py` | term-by-term Δ of `tx` through the whole chain, per player per tick | 1 match |
| `_diag_block_frame.py` | is a mirror/arithmetic error in a geometry function? both directions | **ms** |
| `_diag_shape_fixed_point.py` | where a target recursion settles, in pure algebra | **ms** |
| `_diag_depth_anchor_control.py` | is a new lever actually REACHED (role scope, band non-vacuity, switch reversibility) | **ms** |
| `_diag_style_temp_control.py` | same, for the style temperament | **ms** |
| `_diag_wide_depth_live.py [--report-only]` | whole-team per-role shape + frame validation; caches samples so a re-report costs nothing | 1 match, then free |

**The reusable idea worth stealing: `_diag_depth_anchor_ab.py` §7/§8.** §7 asks
"is this player an outlier or is this his TEAM", and §8 asks "does the ball
ever reach him" by joining receipts to pass origins. Both turn *"this player's
number is wrong"* into a question about mechanism. A player-level figure with
no team context and no receipt join is not evidence.

**⚠️ DO NOT COMMIT `_diag_wide_depth_cache.json`.** It is a **166 MB** probe
output sitting untracked at the repo root. `git add -u` would also have dropped
all 16 new untracked tests while committing the code they justify (see the
*Commit hazard* note above). Both hazards were live on 2026-10-15.

### GATES

`tests/test_style_defensive_temperament.py` (16, **new file** — the stance
suite is green *about* the style's influence and silent on it) plus
`test_positional_play` 46 passed, `test_cb_ceiling` 7 passed. Two failures are
**pre-existing and characterised**: `test_out_of_possession_touchline_press_
gated_by_press_intensity` (bisected to fail identically with the switches
off; it drives `drift_minute`) and `test_lever_b_starve_squashes`
(**flaky by construction** — `sequence_length` calls an unseeded
`random.randint(3, 8)` and the test asserts an inequality on a stochastic
function; it passed and failed on consecutive runs, and never touches the shape
chain).

Two guards worth copying: an **AST** guard that `match_engine` actually passes
`style=` (a defaulted argument is a silent omission — every delta test still
passes with it dead), and that `style` is in `apply_formation_stance`'s
idempotency key (without it the second apply short-circuits and the team stays
on the previous style's line). Same reasoning as the `attacks_right` and
`shooter_name` guards.

## MATCH MOMENTUM — the chart measured the scoreboard, not the match (2026-10-16)

User complaint: *"a graph bar/line can rise even if no goal was scored"* —
and the chart could not, ever. They were right, and the cause was total.

- **THE OLD SERIES HAD THREE WRITERS.** `MomentumEngine.after_goal` (+18–30),
  `after_red_card` (+25–40), and `natural_decay` (×0.92 **per minute**, toward
  0). That was the complete set. So a match with no goals and no red cards was
  *mathematically obliged* to produce a flat line — and 45 flat minutes is
  what the exported PNG showed. Every visible feature was a goal spike with a
  decay tail. `after_save` was defined and **never called at all**.
- **THE THREE SYMPTOMS WERE ALL FORCED BY THAT.** (a) It could not rise
  without a goal — a team shredding a flank for four minutes registered
  nothing. (b) A goal was pinned at ±100 in the PLOT, so every goal looked
  like it arrived on a wave; a scrappy corner goal for an outpossessed side
  was **unreachable by construction**. (c) It was never a
  possession/shots/chances measure at all.
- **THE MODEL SHIPPED** (`momentum_model.py`, new): the published Opta
  algorithm with the xT substitution the independent literature uses when the
  proprietary possession-value model is unavailable (Kapich 2024, following
  Karun Singh). Threat grid (18×12, attacking frame) → per-action
  `xT = value(end_zone) − value(start_zone)` → clip to **[0, 0.10]** →
  **per-minute MAX per team** → **EWMA over 4 minutes, decay 0.25,
  normalised** → the difference. `XT_CLIP`/`WINDOW`/`DECAY` are Opta's
  published constants, named as such.
- **THE GRID IS DERIVED, NOT TYPED.** Every cell equals
  `XGEngine.calculate_geometric()` at that cell's CENTRE, so the threat model
  and the engine's own xG cannot disagree. It is still a **PRIOR, not a fit**,
  and says so: `fit_grid_from_matches()` does the real goals/shots fit with
  Laplace shrinkage toward the prior and **REFUSES to install** one whose
  median cell is under 25 shots, reporting why. `grid_source` travels into the
  export.
- **GOALS DO NOT MOVE THE CURVE.** Verified as a first-class property: a
  goal is valued **exactly** as the shot it was (0.20230 vs 0.20230 at the
  same spot), because a goal IS a shot; if it scored more, the old defect
  would return. Measured on a real match: Thornfield's 59' goal sat at
  **8.8% of the match's peak** — the user's exact example, and now reachable.
  A scrappy corner scramble is worth **4.0× less** than a clean finish, and a
  clean goal in the *same* minute of a turgid match reads **2.0× higher**.
- **HIT_WOODWORK IS IN `_SHOT_EVENTS` even though `is_shot` omits it** — that
  omission is a separate documented calibration question, and five hits off
  the frame is not zero threat.
- **THE FRAME TRAP, HANDLED ONCE.** `team_attacks_right` is never flipped at
  half time, so raw x is per-team. Every lookup goes through
  `to_attacking_frame()`. The strongest test is the mirror: an identical
  attack on either side gives momentum **exactly 0.000000000**, and a one-sided
  match flips sign to the last decimal. The 9 off-pitch coordinates a match
  are clamped for the lookup and **counted** in `clamped_coordinates`.
- **⛔ IT IS POST-HOC, AND THAT IS THE POINT.** The live `MomentumEngine` also
  feeds gameplay (`get_attacking_probability_modifier`; `state.momentum/100*
  0.08` at the shot funnel) and was **left completely untouched** — rewriting
  it changes every event in every match and invalidates every calibration
  number in this file. So the dependency is one-way and guarded:
  `match_engine ← momentum_model ← opta_analytics ← exporter`, and **no engine
  module may import `momentum_model`**.
  > **NAMING COLLISION, DELIBERATE.** `state.momentum` (live, gameplay) and
  > the exported "momentum" (threat) are different quantities under the same
  > name. `momentum_log` still exists and still feeds the live scalar; it is
  > simply no longer what the chart reads. Guarded by AST.
- **SUITE `tests/test_momentum_model.py` (33, new file)** + **negative control
  `_diag_momentum_guard.py`** (3 mutations, all caught, files restored
  byte-identically). A guard that has never been seen to fail is not a guard,
  so M1/M2 reintroduce each defect and confirm the RIGHT guard goes red and
  the other stays green; M3 confirms prose mentioning `momentum_log` does
  **not** trip the guard.

- **FOUR OF MY OWN ERRORS, all the documented class, all found by the probes:**
  1. **A real defect I introduced:** `_situation_bonus` first returned
     `end_value * 1.0` for *every* situation, so an open-play pass scored
     `2*end − start`. Both teams pinned at the clip on identical ground and the
     **difference — the only thing the chart plots — collapsed to zero for the
     whole match.** A term modelling the exceptional case must be zero
     everywhere else. Found by the "home must be positive" check.
  2. **Four false failures from one probe bug:** I asserted over all 90 rows
     when the fixture covered 30, and the series legitimately decays to
     `0.0000` past the EWMA window. Same cause as the documented CM
     knife-edge: *the thing measured is not the thing asked for.*
  3. **An off-by-one that silently measured the wrong minute** — an index
     expression asserting minute 24 while the label said 25. Replaced with an
     `at(result, minute)` lookup, because index arithmetic by eye is how you
     assert the wrong minute and never notice.
  4. **A dishonest LABEL:** one check compared "no goal at all" against "clean
     goal" and titled the no-goal arm "scrappy". A column whose heading
     asserts more than its expression computes.
- **A FIFTH, AND IT IS THE EXPENSIVE ONE — THE TREE WAS WIPED MID-SESSION.**
  `momentum_model.py`, `tests/test_momentum_model.py` and all three
  `tools/diag/_diag_momentum*.py` probes were **deleted by something outside
  this session**, along with the `opta_analytics.py` edit, while a stale
  `.pyc` for the deleted module remained and `git status` reverted to a single
  modified file. Everything was recreated from the conversation and
  re-verified green, but **this is the second recorded instance of a parallel
  session operating on these files** (see the *Commit hazard* note) and the
  third recorded instance of a **stale `.pyc` outliving its deleted source** —
  which makes a module look importable when `Test-Path` says it is not.
  **For a reader: verify with `Test-Path` / `read_bytes().count()`, not with
  "the import worked".**
- **PROBE-BUG THAT ALMOST REVERTED REAL WORK.** The negative control's
  `restore()` compared the file against its backup and raised *before*
  writing — and on a later run "restored" a **stale backup captured before the
  previous run's crash**, silently reverting `plot_momentum` to the old bar
  chart while the probe reported success. Fix: snapshot the file **inside**
  `mutate()`, immediately before the mutation, and return it. **A backup is
  only valid for the state it was taken from.**
- **THE EXPORTER GUARD IS AN AST GUARD, NOT A TEXT SCAN, for the documented
  reason.** `plot_momentum`'s docstring quotes `result.momentum_log` in order
  to explain what changed, so a substring scan matched the explanation —
  the fifth recorded instance of "a name inside prose is not a call site".
  Stripping `#` comments is not enough; the string was in a **docstring**.
- **GATES:** `test_momentum_model` 33 passed; `test_opta_analytics_runs_sprints`
  3 passed (a pre-existing test builds a `SimpleNamespace` result with **no
  `state`**, which `compute_momentum` now reads defensively — it is a pure
  function of a timeline and should not require one); `_diag_momentum.py`
  all checks passed; `_diag_momentum_guard.py` all controls passed; one real
  match rendered end to end (5–2, 7 goal icons all placed, the 40' goal on a
  genuine blue peak, the 83'/84' pair in a deep away trough).
  **NOT gated by the 26/27 regression** (per the standing no-hour-regression
  instruction).

## DISTANCE COVERED — 20 km was a 4.7x PHYSICS INFLATION (2026-10-17, FIXED)

Reported: *"players are covering 20km every match, something is making them
cover this massive distances."* Confirmed, located, fixed. It was **not** the
shape layer, **not** the integrator, and **not** the `record_touch` snap that
*SHAPE ANCHOR* blames for over-credit — it was the **PossessionEpisode race
trace booking the geometry of a linear interpolation as distance walked.**

- **THE PUBLISHED NUMBER IS NOT A MEASUREMENT.** `distance_covered` is
  assembled in `opta_analytics._compute_activity` as
  `distances[name] += baseline_dist + physics_dist_additive`, per minute.
  `baseline_dist` is a **DNA activity template** (`BASE_MINUTE_ACTIVITY` x
  `work_rate`, converted through `SPEEDS`) — an authored per-minute budget,
  not observed movement. `physics_dist_additive` is `physics_dist * 0.45`.
- **THERE ARE THREE WRITERS INTO `physics_distance_m`**, found by attributing
  every `record_physics_distance` call to its **call site** (the
  `_diag_touch_sites.py` frame join), not by inferring intent:

  | call site | what it is | one match |
  |---|---|---|
  | `match_engine.py:2732` `_record_offball_distance` | the real 10 Hz integrator | 73.7 km |
  | `event_chain.py:551` `_accumulate_physics_stats` | PossessionEpisode trace | **405.3 km** |
  | `position_engine.py:1076` `advance_to_slots` | set-piece run-in | 1.6 km |

  **The middle one was 5.5x the integrator that actually watched those players
  walk.** I instrumented only two writers on the first pass and the
  decomposition would not close — which is how the third was found. A derived
  residual that moves when a source changes is a measurement of your own
  subtraction, not of the code.
- **THE MECHANISM.** `possession_physics.resolve_ground_pass` /
  `resolve_dribble` place a defender by **geometric interpolation toward a
  MOVING target**:
  `pos = start + (ball_now - start) * dr`, `dr = ts / arrival`. His per-tick
  displacement therefore scales with the **ball's** speed, not his own, and
  `calculate_distance_stats` was booking `math.hypot(dx, dy)` from it.
  Measured on a real match (`_diag_trace_overcount.py`):

  | | |
  |---|---|
  | mean implied speed | **33.2 m/s (120 km/h)** |
  | metres above 8.5 m/s (30.6 km/h) | **70.1%** |
  | metres above 12 m/s (43.2 km/h) | **64.8%** (260 km) |
  | from the race notes | **98.2%** (intercept_race 53.5, pursuit 22.4, contact 22.3) |
  | geometry / the same segments x the engine's own speed label | **4.70x** |

- **⛔ THE SOURCE ALREADY KNEW, AND FIXED THE WRONG HALF.** The comment above
  the band classification read: *"the coarse per-tick interpolation makes
  receivers/defenders jump 1-2+ m per 0.1 s tick (an implied 10-20 m/s), which
  a real GPS would never see"* — then applied the fix to **sprint COUNT**,
  which reads the speed label, and left **DISTANCE** reading the geometry,
  under the line *"Distance accumulated remains geometry-real (seg_dist)."*
  **The label was always the honest quantity.** The comment also *understated*
  it: the real figure is 33 m/s mean, not 10-20.
- **MY FIRST HYPOTHESIS WAS DEAD ON ARRIVAL.** "Sparse trace rows / cross-action
  position gaps are booked as distance" is a real defect class in this repo and
  the code comment says so explicitly — but measured, those gaps are **1%** of
  writer (b). The 99% sits in **consecutive** ticks. A plausible mechanism that
  measures 1% is not the mechanism.
- **THE FIX** (`possession_physics.calculate_distance_stats`, 1 functional
  change): book `seg_dist = min(geometry, dt * speed_label)` under the
  consecutive-tick guard, and book **0** for a position gap. Speed, sprint and
  HSR banding already read the label, so **DISTANCE now reads the same
  quantity SPEED does**. The gap branch still zeroes `speed` so an open sprint
  run is closed — same rule and same precedent as `run_tracking.RunTracker`'s
  jump filter.
- **MEASURED, one match, seed 777, before -> after:**

  | | before | after |
  |---|---|---|
  | writer (b) episode trace | 405.3 km | **74.2 km** (5.46x down) |
  | exported total, 22 players | 364.5 km | **290.0 km** |
  | outfielders min/mean/max | 14.4 / 17.9 / **20.9** km | **10.7 / 14.2 / 16.5** km |
  | in the 10.0-13.5 km band | 0/20 | **8/20** |
  | match events / score | 3726, 3-1 | 3726, 3-1 (**identical**) |

  The decomposition now closes to **+0.014 km out of 290** (was unclosable).
- **THE FIX CANNOT CHANGE A MATCH.** `player_distance_stats` reaches exactly one
  consumer: `match_engine.py:6323` `self.gps.ingest_episode_stats`, and only
  when `enable_virtual_gps()` was called. Stamina is **not** driven by
  distance. Confirmed empirically — identical event count and scoreline. So
  this needed no 26/27 regression.
- **GUARD: `tests/test_distance_accounting.py` (10 tests), new file.** The
  existing suite is green *about* this and silent on it:
  `test_opta_analytics_runs_sprints.py` asserts per-minute values **accumulate**
  across a match and never that the magnitude is physical, and
  `calibrate_gps.py` — which carries the correct real band (10.5-13.0 km) — is a
  **print-only script** reading a CSV **last written 2026-09-02**, fed by
  `tools/diag/_gps_verify.py`, which does `import _repro as R` and so cannot
  run from the repo root. **Nothing ever compared `distance_covered` to a
  benchmark.** The file carries two AST guards (distance may only accumulate
  *under* the tick guard; the ceiling must exist) that assert **non-vacuity
  first**, plus a behavioural non-vacuity case proving physical geometry is
  still booked in full.
- **NEGATIVE CONTROL `_diag_distance_guard.py`: 3/3 caught**, file restored
  byte-identically, baseline green. **It earned its keep twice.** M2 was first
  written as a one-line edit that "passed" as a **collection error** — rc != 0,
  printed CAUGHT, and proved nothing about which assertion pins the defect, so
  it was rebuilt as a two-edit reconstruction of the actual pre-fix shape with
  a gate that the mutant must still **import**. And `test_virtual_gps` was
  bisected rather than assumed (`_diag_gps_bisect.py`): it fails at HEAD too
  (mean drift 118.0 m pre-fix vs 116.2 m post-fix, threshold 25 m), so it is
  **pre-existing** and the ceiling slightly improved it.
- **⚠️ WHY `test_gps_matches_engine_physics` COULD NEVER HAVE CAUGHT THIS.** The
  test's premise is that the GPS is an *independent* pass. It is not: GPS
  ingests the same `player_distance_stats` the engine just wrote, so both
  instruments carried the identical 4.7x defect and agreed with each other.
  **An oracle that shares the subject's defect is not an oracle.** The residual
  116 m disagreement is therefore a genuinely open question about the off-ball
  sampler path, and is *not* evidence about the trace.
- **THE RESIDUAL IS REAL, MEASURED, AND NOT A BUG IN THE FIX.** Outfielders now
  sit at 14.2 km mean against 10.0-13.5, max excess **+2.96 km**. It is
  entirely the flat additive, and the template is fine:
  - `template alone` (outfielders) **8.89 / 11.69 / 13.55 km** min/mean/max —
    already inside the band, and inside this module's own documented 9-14 km
  - `0.45 x physics` adds a **flat +2.53 km to every player**, unconditionally
  - **A STRUCTURAL INCONSISTENCY, measured not argued:** the
    `if real_dist > baseline_dist * 1.15: excess = real_dist - baseline_dist`
    logic **exists and is DEAD** — it lives in the `else` of
    `if physics_dist > 0`, so for any player with *any* physics trace (i.e. all
    of them) it never runs. The live branch adds a flat fraction of a
    full re-measurement instead of the *excess* the code's own comment says it
    is adding: *"adds high-intensity distance that the baseline undercounts."*
  - **CALIBRATION, NOT DEFECT — and deliberately NOT touched.** Making the
    additive excess-based changes every distance number in the project and in
    the warehouse, and needs a multi-seed sweep (~10 matches) against the
    Opta benchmark. Same rule as *SHAPE ANCHOR*'s `COMPACTION`: it is a
    refinement, it is a decision, and it does not touch the defect fixed here.
- **⚠️ SEPARATE PRE-EXISTING FINDING, same table, untouched:** published
  `top_speed` is **22.4-26.3 km/h** against `calibrate_gps`' own 28.8-34.6
  km/h band — every player ~25% low. The ceiling did not touch it (`top_speed`
  reads the label, which I did not change). Root cause is that the label peaks
  at `RACE_EFFORT(0.85) x top_speed`, so a published top speed can never exceed
  85% of the player's real maximum.
- **ALSO: keepers read 2.82-2.84 km against a real ~4-6.** `BASE_MINUTE_ACTIVITY`
  gives a GK `(0.80, 0.18, 0.02, 0.00, 0.00)` = 2.8 km. Same table, same
  pass — **do not fix it in this pass.**
- **PROBES** (all one match each, `PYTHONHASHSEED=0`):
  `_diag_distance_audit.py` (decomposition + G2/G3 gates), `_diag_trace_overcount.py`
  (implied-speed bands, trace-seconds, distance by `_snap` note),
  `_diag_distance_writers.py` (three writers by call site + independent
  template recomposition — the one that CLOSED the decomposition),
  `_diag_distance_final.py` (export-facing table vs `calibrate_gps` bands),
  `_diag_gps_bisect.py` (pre-existing-failure bisect), `_diag_distance_guard.py`
  (negative control).
- **THREE PROBE ERRORS OF MINE, all the documented class, recorded because each
  produced a clean wrong answer:**
  1. **`_snap` timestamps from `self.elapsed`, NOT from a `tick` argument.** A
     test helper that appears to pass the tick through writes every row at
     t=0.0; the consecutive-tick guard then discards all of them and **every
     distance assertion returns 0.0 and looks like a code defect.** Cost two
     debugging cycles inside the new test file.
  2. **I derived the "template" column by subtraction** (`raw − 0.45 x phys`)
     and it MOVED when the fix landed (6.77 -> 10.16 km/player). `baseline_dist`
     is computed before any physics term and cannot depend on it. A derived
     residual that moves when a source changes is measuring your own
     subtraction. Fixed by recomputing the template independently from
     `BASE_MINUTE_ACTIVITY` + `SPEEDS` — which is what closed the books.
  3. **The negative control counted a syntax error as a caught mutation.** See
     above.

## GOALKEEPING OWNERSHIP + GK WORK RATE + THE EXCESS ADDITIVE (2026-10-17)

Four items from the same session as *DISTANCE COVERED*, three of them code
changes. All measured on real matches, all guarded, all with a negative control.

### 1. GOALS PREVENTED WAS A TEAM TOTAL ON EVERY `position == "GK"` — FIXED

*"a bench gk who didn't play or entered as a sub gets the same goals
prevented."* Correct, and the cause was one line in `exporter.py`:

    s["xgot_faced"]      = self.team_xgot.get(team, 0.0)      # TEAM
    s["goals_prevented"] = xgot - goals_conceded               # TEAM

`saves` was already correctly per-player (SAVE events name the keeper), so the
pre-fix row for a bench keeper read **saves 0, save_pct 0.0%, and a non-zero
Goals Prevented** — and the rating block collected `min(gp*0.25, 0.75)` off it.
A substitute who came on at 70' inherited the 12' goal the same way.

- **ATTRIBUTION IS "WHO WAS IN GOAL AT THAT INSTANT", NOT `secondary_player`.**
  That field cannot be used: it is the keeper on a SHOT but the **ASSISTER** on
  a GOAL, and `PENALTY_SCORED` carries no keeper at all. New
  `StatAccumulator._in_goal_at()` resolves it from the starting XI plus the
  SUBSTITUTION timeline (`player` = off, `secondary_player` = on). One concept,
  no dual path, and it handles penalties and the scorer/assister ambiguity
  without either.
- **⛔ `e.team` MEANS OPPOSITE THINGS ON `GOAL` AND `OWN_GOAL`.** A GOAL is
  built on the **scoring** side (`AttackChain` passes `att_team`); an OWN_GOAL is
  built on the **conceding** side —
  `make_event(..., OWN_GOAL, defending_team, defender.name)` — because
  `goal_team` is the beneficiary while the scorer is the defender who put it in
  his own net. So the mirror rule applies to one and NOT the other. **My first
  version applied it to both and the reconciliation gate caught it** (Thornfield
  conceded 2, keepers accounted for 1). Second instance of the
  `secondary_player` trap in this file: one field, two frame-dependent
  meanings, and it fails *silently*.
- **RECONCILIATION IS AN INVARIANT, NOT A COMMENT:** Σ keeper xGOT ==
  `team_xgot`, and Σ keeper concessions == team goals conceded. Measured on a
  real match: **0.841 = 0.514 + 0.327**, conceded 1 = 1, both teams OK.
- **MEASURED, forced keeper swap at 70'** (the live match produced ZERO
  substitutions, so the swap branch was never exercised — a vacuum, and the
  keeper-who-came-on case is precisely what was broken):

  | | minutes | saves | conceded | xGOT faced | goals prevented |
  |---|---|---|---|---|---|
  | Keano Walsh (OFF at 70') | 0–70 | 4 | 1 | 0.514 | −0.486 |
  | Rafi Sandoval (ON at 70') | 70–90 | 0 | 0 | **0.327** | **0.327** |
  | a keeper who never played | — | 0 | 0 | **0.000** | **0.000** |

- `goals_conceded` and `clean_sheet` are now keeper-scoped too (they were
  team-scoped, so a substitute who came on with the team 2-0 up was credited
  with a clean sheet), and the GK rating branch reads the keeper's own figures.
  For every **non**-GK, `goals_conceded` stays team-level — it is a defensive
  figure there.
- **GUARD `tests/test_keeper_attribution.py` (14 tests) + `_diag_keeper_guard.py`
  (4/4 caught, files byte-identical).**
  > **⚠️ THE FIRST NEGATIVE CONTROL RUN WAS 3/4 AND THE MISS WAS THE HEADLINE
  > BUG.** M1 re-stamped the team total onto every goalkeeper and the suite
  > stayed GREEN — because every other test inspected `keeper_xgot` /
  > `keeper_conceded` (the *accumulator*) and none ran the *consumer* that
  > writes the stat row. So the accumulator was guarded and the defect's actual
  > location was not. **A guard written only against the producer cannot see the
  > consumer.** Fixed by extracting `_apply_keeper_stats` and adding
  > `KeeperStatRowTests`, which drives the row — plus a non-vacuity case that
  > asserts the team figure *is* available and *is* different, so the test
  > cannot pass because the team total happens to be zero. This is the
  > `_diag_cross_guard.py` lesson recurring verbatim: I had named the first file
  > as covering "the cross receiver" when it only covered the helper.

### 2. ARE THE ACTIVITY SECONDS MEASURED? — **NO. THEY ARE HEURISTIC.**

`standing_seconds` / `walking_seconds` / `jogging_seconds` / `running_seconds`
/ `sprinting_seconds` come from `opta_analytics._compute_activity`:

1. **the backbone is a table**, `BASE_MINUTE_ACTIVITY[position]`, scaled by
   `effort = 0.80 + 0.40 x work_rate/100` (jog/run/sprint only);
2. **telemetry then perturbs it** — touches, `peak_touch_jump`, physics sprint
   counts — moving time between bands;
3. **the result is renormalised to exactly 60 s per minute**, by draining
   `diff` out of standing→walking→jogging first and then out of running. So the
   five bands always sum to 60 by construction, and "the sum is 60" is not
   evidence of anything.

Measured, template vs published (seconds, whole match):

| player | pos | band | template | published | Δ |
|---|---|---|---|---|---|
| Percy | RW | standing | 941 | 675 | −266 |
| Percy | RW | running | 647 | 868 | +221 |
| Percy | RW | sprinting | 59 | 218 | +159 |
| Nils Werner | CM | standing | 1058 | 822 | −236 |
| Nils Werner | CM | running | 470 | 885 | +415 |
| Leo Mbeki | LW | sprinting | 59 | 290 | +231 |

**So: a heuristic shape, telemetry-shaped.** Telemetry consistently pulls time
*out of standing* and *into running/sprinting* — i.e. it makes everyone look
busier — by 200–400 s per player per match. The published **distance** is
correspondingly ~80–90% template. The genuinely **measured** quantities are
the `physics_*` fields and `distance_covered`'s trace term; the five
`*_seconds` are not among them. **Do not present the seconds as telemetry.**

### 3. KEEPERS COVERED 2.82 km — FIXED, BENCHMARKED FIRST

*"GKs in real life don't cover 2.88 km, they usually cover more, ~4, 5 or 6."*
Correct. Two independent GPS studies, both from the literature this project
already cites for its speed bands (`calibrate_gps.py`):

- **Di Salvo et al. 2008**, elite goalkeepers, GPS: **5,611 ± 613 m**
- **Kubayi et al. 2020**, Olympic goalkeepers: **4,819 m** (range 4,036–6,640)

`BASE_MINUTE_ACTIVITY["GK"]` was `(0.80, 0.18, 0.02, 0.00, 0.00)` = 29.2 m/min
= 2.80 km — about **half** of real. Now `(0.66, 0.17, 0.15, 0.02, 0.00)` =
54.2 m/min at effort 1.0, spanning **4.61–5.80 km** across the work-rate range,
inside both studies. **Measured on a real match: 5.27 and 5.50 km.**

> A goalkeeper's distance is **100% template** — `_offball_move_player` returns
> early for `position == "GK"`, so the integrator and the trace contribute
> ~0.04 km. Nothing about the GK number is measured, which is worth knowing
> before anyone treats it as evidence.

### 4. THE DEAD `excess` LOGIC IS NOW LIVE — MEASURED, AND THE RESIDUAL IS NAMED

The flat `physics_dist * 0.45` added a fraction of a **full re-measurement** on
top of a template that already represented the whole match. It now adds only the
**excess** over the template's budget for the seconds the trace actually covers:

    covered_frac = min(1.0, traced_s / 60.0)      # traced_s from the SAME row
    budgeted     = baseline_dist * covered_frac
    physics_dist_additive = max(0.0, physics_dist - budgeted) * 0.45

`traced_s` = `physics_walk_time_s + physics_jog_time_s + physics_sprint_time_s`,
already on the `position_log` row, so **no new field and no match-engine
change**. The dead `if real_dist > baseline_dist * 1.15` logic this replaces
lived in the `else` of `if physics_dist > 0` and had never run for anyone.

**MEASURED, outfielders who played (n=20):**

| | mean | max | in 10.0–13.5 |
|---|---|---|---|
| before the excess change | 14.22 | 16.46 | 8/20 (0 under) |
| **after** | **12.45** | **14.26** | 9/20 (**8 over, 3 under**) |

**The mean is now inside the band, but the spread is ±2.4 km and it straddles
BOTH edges.** That is the honest result, and the per-position table says why —
the residual is a clean positional bias **in the template**, not noise:

| pos | template | published | real (PL/90) | verdict |
|---|---|---|---|---|
| CB | 8.90 | 9.81 | ~10.5–11.0 | **~1 km LOW** |
| CDM | 10.21 | 11.41 | ~10.5–11.0 | good |
| ST | 10.51 | 11.77 | ~10.5–11.0 | good |
| CAM | 11.02 | 12.10 | ~11.5–12.0 | good |
| LB | 11.78 | 13.38 | ~11.0–12.0 | +1.4 over |
| RB | 11.78 | 13.30 | ~11.0–12.0 | +1.3 over |
| LW | 13.17 | 13.87 | ~11.5–12.5 | **+1.4 over** |
| RW | 13.17 | 14.12 | ~11.5–12.5 | **+1.6 over** |
| CM | 12.21 | 13.97 | ~11.5–12.5 | **+1.5 over** |

(GK shows 3.49/3.59 in that table only because the probe's injected substitute
has zero `position_log` frames; the two real keepers are 5.27 and 5.50.)

> ### ⛔ RETRACTED 2026-10-20 — THE `real (PL/90)` COLUMN IN THAT TABLE IS UNSOURCED AND WRONG
> Every band in it was written from memory. The published figures (CIES Football
> Observatory Monthly Report n°68, Oct 2021 — 7,855 matches, 31 leagues,
> SkillCorner tracking, five position modalities) are **lower and in a different
> order**: **centre backs 9.2 km/match, the LEAST of any position**; midfielders
> the most at **10.6**; wingers cover the most high-intensity (932 m) and
> sprinting (211 m), centre backs the least in both. So "CB ~10.5–11.0" is about
> **1.5 km too high**, and the "~1 km LOW" verdict it produced for the
> centre-back was a measurement against a number that does not exist. The table
> itself is also historical — it predates the 2026-10-19 measured-activity
> rewrite and its `template`/`published` columns describe the *excess-additive*
> form that has since been retired. **Read the numbers in it as a record of what
> was measured then, never as a benchmark.**

> **CALIBRATION, NOT A DEFECT — and deliberately NOT done here.** `CB` wants
> ~+0.9 km and `CM/LW/RW` ~−1.5 km. Per-position table edits are a **fit**, and
> per this file's own standing rules (`COMPACTION`, *SHAPE ANCHOR*) a fit needs
> a **multi-seed sweep** against the benchmark — ~10 matches, ~20 min — and is a
> deliberate decision because it moves every distance figure in the project and
> in the warehouse. The per-position evidence is above so that decision is one
> step away, not a fresh investigation. **One match, one seed, one roster.**

- **GUARDS:** `tests/test_keeper_attribution.py` carries the GK work-rate band
  (with a non-vacuity case proving the *old* template really is out of range, so
  the test cannot pass by accident if the table is reverted), a case pinning
  that a keeper still stands more than a CM, and two structural guards on the
  excess form — including `assertNotIn("physics_dist_additive = physics_dist *")`
  so the flat multiplicative form cannot come back unnoticed.
- **PROBE `tools/diag/_diag_keeper_distance_verify.py`** covers all of §1–§4 in
  one match, with the reconciliation invariant as a hard gate and a
  forced-keeper-swap scenario because the live match produced zero subs.
- **SUITES:** the two new files + `test_opta_analytics_runs_sprints` +
  `test_movement_and_body` + `test_plofa_export` + `test_chance_creation` +
  `test_alltime_db` → **99 passed, 2 strict xfailed, 0 failed** (5m26s).
  `tests/test_virtual_gps.py::test_gps_matches_engine_physics` remains the
  documented **pre-existing** failure (bisected in the section above).

## MEASURED ACTIVITY — the seconds were authored because standing time was UNMEASURABLE (2026-10-17)

The user asked to *"fix the activity seconds to come truly from raw match data
and also the multi seed sweep if it will bring true realsim not not handcoded
values"*. Both halves were done, and the second one's answer is **NO**.

### ⛔ THE MULTI-SEED SWEEP DOES NOT SATISFY THE STATED CONDITION

The sweep fits `BASE_MINUTE_ACTIVITY` — **a hand-authored per-position table** —
to the Opta benchmark. It makes a handcoded number *closer to real*; it cannot
make it **true**. The user's condition was "not handcoded values", so the sweep
is the **wrong tool for the goal they stated**. It is not refused as
unnecessary: it is wrong by construction.

And it is now nearly moot anyway, because the measurement change demotes the
table to a **residue**: it supplies the shape of the ~31% of seconds the
integrator did not watch, worth **~9% of the published distance**. Tuning a
table that carries 9% of a number is not a route to realism. **The instrument
the user actually wanted was the measurement, and that is what shipped.**

### TWO REFUTED HYPOTHESES, recorded so they are not re-derived

1. **"The engine simulates less time than 90'."** WRONG.
   `tools/diag/_diag_match_clock_budget.py`: **5,782 s simulated against 5,760 s
   nominal (100.4%)**, and the two integrators budget **5,570 s = 96.3% of the
   clock** (`_offball_run` 1,633 s live + `_continuous_offball_phase` 3,938 s
   dead). **The clock is fine.**
2. **"The gap is the `if moved <= 0: return` in the writer."** Initially
   *unmeasurable*, and my first probe of it was itself wrong — it counted calls
   to `_record_offball_distance` in a wrapper placed **before** the method's own
   early return, so the gate could never fire and reported "premise is wrong,
   stop" out of a broken instrument. **A gate that cannot fail is worse than no
   gate.** Rewritten to wrap the ACCUMULATOR (`record_physics_distance`, the
   only thing that proves a measurement was booked), it is confirmed below.

### THE HEADLINE DEFECT

`MatchEngine._record_offball_distance` opened with

    if moved <= 0:
        return

`_offball_move_player` has an `arrive = 2.0 m` dead band, so a player settled AT
his shape target moves 0 m on the tick — **and his time was discarded before any
measurement.** Measured (`_diag_standing_gap.py`, one real match):

| | ticks |
|---|---|
| integrator stepped | 666,476 |
| booked to the accumulator | 364,160 |
| **dropped by `moved == 0`** | **306,504 (46.0%)** |

**≈1,533 s per player — 26.6% of a nominal match — thrown away.** That is not a
cosmetic gap: it is *why* the table had to invent a **16–83% standing share**. The
largest single band had no data, so it was authored.

Separately measured: the on-ball exclusion (`_on_ball_this_minute`, a sticky
per-minute set) removes **447,544 ticks (42.8%)** — the other structural hole,
and a **simulation** question, not an accounting one (see the residual below).

### WHAT SHIPPED

- **`moved == 0` now books its time.** The sprint-segment state machine is
  deliberately NOT touched — it only ever ran on `moved > 0` and still does, so
  sprint counts and cooldowns are byte-identical.
- **Five measured band accumulators** on `PlayerSpatialState`
  (`physics_{standing,walk,jog,running,sprint}_time_s`) + the `position_log` row,
  banded **once, at the writer**, from the integrator's own per-tick speed.
- **The band edges are Di Salvo's published GPS thresholds** — the literature
  `calibrate_gps.py` already cites — so the buckets mean something outside this
  repo rather than being tuned to make a total come out right:
  `standing < 0.5`, `walking 0.5–2.0`, `jogging 2.0–4.2`, `running 4.2–7.0`,
  `sprinting ≥ 7.0 m/s` (`>25.2 km/h`). `position_engine.ACTIVITY_BAND_EDGES`.
- **Measured first, template only for the gap.** For every second the integrator
  watched, the published band seconds and the published metres ARE the
  measurement. The table supplies the shape of the `uncovered_s` seconds only,
  scaled by `work_rate`, and telemetry may shape **that part only** — a measured
  second is a fact and is never overwritten.
- **`distance_covered = physics_dist + template_dist`** — measured metres plus a
  template budget for the unwatched seconds. This **retires the flat/excess
  `physics_dist_additive` entirely** (yesterday's excess form is superseded, not
  merely tuned): the trace is no longer an *addition* to a full-match template,
  it is that template's **replacement** on the seconds it covers.
- **`activity_measured_pct` + `activity_source` are published per player**, so a
  blended figure can never be mistaken for an observation. Nothing is
  mislabelled.

### MEASURED, one real match, seed 777 (`_diag_measured_activity_verify.py`)

| | before | after |
|---|---|---|
| outfield measured coverage | 42.1% | **68.9%** (min 44.3, max 88.5) |
| `standing` seconds measured | 0% (no such measurement existed) | **100%** |
| `walking` measured | — | **100%** |
| `jogging` / `running` measured | — | **96–99% / 92–97%** |
| template share of published distance | effectively 100% | **~9%** |

### ⛔ THE HONEST RESIDUAL — IT IS NOW *TRUE* AND NOT YET *REALISTIC*

Publishing the measurement **lowered** `distance_covered` to **~6.6–8.1 km**
against the real 10.5–13.0 km band, because the measured movement really is
6.1–7.6 km. **Two independent gaps, both measured, neither an accounting
artefact:**

1. **COVERAGE 69%** — the 31% unwatched is dominated by the per-minute sticky
   on-ball exclusion. Closing it changes positions on the pitch, so it needs its
   own gate and is **not** done here.
2. **SPEED.** While watched, players move at **≈1.56 m/s average**, against a
   real ≈2.13 m/s (11.5 km / 5,400 s). Even at 100% coverage and 1.56 m/s the
   total is ~9.0 km, still short of 11.5. The cause is the shape-holding jog
   (`_JOG_SPEED` 1.75–1.90 m/s) plus the measured **standing share of 20–32%**
   (real football is ~10–15%): the integrator's players **arrive and stop**.

**So the deliverable is a measurement, and the measurement says the engine's
locomotion is about 61% of a real player's. Publishing 12 km would have been
fiction; publishing 7 km is the truth about the simulation.** Closing that gap
is a *pitch* change (integrator coverage and speeds), not an analytics one, and
it is the next workstream — not another table edit.

### GUARDS

- **`tests/test_measured_activity.py` (13, new file).** `test_opta_analytics_
  runs_sprints` is green *about* `distance_covered` and **silent on what it is
  made of** — it asserts per-minute values ACCUMULATE, never that the magnitude
  is a measurement. A suite green and silent about a feature is not evidence
  about it. The new file pins the band edges, that time lands in exactly one
  band, that a stationary tick is booked rather than dropped, that **full
  coverage publishes the measured seconds and metres exactly** (no template left
  to hide behind), that partial coverage adds only the unwatched residue, and
  that `activity_measured_pct`/`activity_source` are published. The
  bare-early-return pin is a **structural** scan because a behavioural test
  cannot see a re-introduced discard.
- **THE TESTS CAUGHT A REAL BUG IN MY OWN IMPLEMENTATION ON THE FIRST RUN:
  `tpl_s` lost its ×60** (`prof[b] / 60.0 * uncovered_s` → `prof[b] /
  60.0 * 60.0` = a *fraction*, not seconds). At zero coverage the published
  bands summed to **1.0 s instead of 60**. Fixed to `prof[b] * uncovered_s`.
  That test was the only thing standing between the user and a silently broken
  fallback path.
- **NEGATIVE CONTROL `tools/diag/_diag_activity_guard.py`: 4/4 caught**, files
  restored byte-identically, baseline green. M1 re-discards `moved == 0` (the
  headline defect), M2 publishes the table for every minute, M3 reverts to
  `template_distance + fraction_of_trace`, M4 drops the standing band. Each
  mutant is required to **still import** — a syntax error counted as a caught
  mutation proves nothing about which assertion pins the defect.
- **A GUARD HAD TO BE REPOINTED, NOT DELETED.** `test_keeper_attribution.
  PhysicsAdditiveIsAnExcessTests` (added yesterday) pinned the *excess* form
  that today's blend supersedes. Left alone it would have gone **RED on a
  strictly better implementation**, which is how a guard trains people to delete
  it instead of reading it. It now asserts the superseding invariant
  (`baseline_dist == physics_dist + template_dist`, the residue scoped to
  `uncovered_s`, and no flat/excess additive) — and it is an **AST** check,
  because the comment explaining the retirement *quotes* `physics_dist_additive`
  and a substring scan matches the prose. **A name inside a comment is not a
  call site** — the sixth recorded instance of that shape in this file.
- **PROBES:** `_diag_activity_coverage.py` (coverage %), `_diag_match_clock_budget.py`
  (refuted the clock hypothesis), `_diag_standing_gap.py` (the 46% drop),
  `_diag_measured_activity_verify.py` (all three post-change gates:
  coverage > 42.1%, published == measured, distance = measured + residue).

### GATES

`test_measured_activity` (13) + `test_keeper_attribution` (15) +
`test_distance_accounting` + `test_opta_analytics_runs_sprints` +
`test_movement_and_body` + `test_plofa_export` + `test_chance_creation` +
`test_alltime_db` + `test_positional_play` → **113 passed, 2 strict xfailed,
0 failed** (1m31s). Both negative controls **4/4**.

**Not gated by the 26/27 regression** (standing no-hour-regression
instruction). **One match, one seed, one roster** — every percentage above is a
single sample; the *mechanism* is what is established, and none of the distance
figures should be quoted as calibrated.

## LOCOMOTION — players stood because three separate things booked "still" (2026-10-19)

Started from the two complaints that follow on from the section above: *players
stand too much / move too little vs real football*, and *are the activity
seconds real or hand-coded?* Both were answered by attributing every number to
the line that produced it. Probe: `tools/diag/_diag_locomotion.py` (phase-
instrumented: wraps `_offball_run` and `_continuous_offball_phase`, and its
section `A0` sums the raw `position_log` bands per player beside the published
table — that section is what located the standing).

### Q2, ANSWERED: the published seconds are a BLEND, and 3 bands leaked

`BASE_MINUTE_ACTIVITY` is a hand-authored per-position table. After the
2026-10-17 work it supplies only the seconds no instrument watched, but that
residue is not nothing:

| | before 2026-10-19 | now |
|---|---|---|
| outfield seconds carrying a measurement | 42.1% | **91.1%** (80–94%) |
| template share of the published DISTANCE | ~100% | **11%** |
| GK | 0.4–0.6% measured — **100% table** | unchanged (the integrator returns early for GKs) |

- **THE TRACE HALF USED THREE BANDS AND A MISLABELLED EDGE.** The possession
  episode trace classified walk/jog/sprint with `<2.0 m/s → walking`, so a man
  at 0.0 m/s booked WALKING and a man at 6.9 booked JOGGING. It now emits the
  same five Di Salvo bands (`STANDING <0.5`, `WALK <2.0`, `JOG <4.2`,
  `RUN <7.0`, `SPRINT ≥7.0`), and `event_chain._accumulate_physics_stats`
  passes all five.
- **A VERDICT ON THE SECONDS, for a future reader:** the five
  `*_seconds` fields are measured for ~90% of an outfield player's match and
  table-shaped for the rest; the two goalkeepers are still the table (brass
  tacks: the integrator skips `position == "GK"`). Do not present the GK
  activity seconds as telemetry. `activity_measured_pct` and
  `activity_source` say which on every row.

### THE THREE "STILL" SOURCES, all measured

Phase decomposition before the fixes (`_diag_locomotion.txt`), ticks at 10 Hz:

| phase | ticks | still | mean speed |
|---|---|---|---|
| dead / block | 239,409 | **98%** | 0.00 |
| dead / open | 204,803 | 0% | 1.42 |
| live / block | 123,794 | 58% | 2.20 |
| live / open | 98,540 | 0% | 1.50 |

1. **DEAD-PHASE ON-BALL EXCLUSION.** `_on_ball_this_minute` is a STICKY
   per-minute set, added for the LIVE window (where the episode trace covers
   the man). In the DEAD window nothing covers him, so one touch at second 5
   froze him for the rest of the minute: measured, **27,773 of a CM's 36,561
   excluded ticks were dead-phase**. `_offball_move_player` now takes
   `include_on_ball=False`, and ONLY `_continuous_offball_phase` passes
   `_DEAD_TIME_ON_BALL` (True). The live phase keeps the exclusion — the
   distance written is `speed x DT`, not a position delta, so including him
   live would double-book the trace.
2. **BLOCK HOLD WAS A FREEZE.** The arrived branch computed a step and then
   hard-zeroed `moved`, so **234,157 of 239,409 dead-phase block ticks (98%)**
   were booked standing while the line never moved. It is now a bounded
   along-the-line shuffle: `_block_shuffle_target` (a triangle wave, ±1.0 m of
   the slot, 0.637 m/s = Di Salvo WALKING), which leaves the line's DEPTH and
   SPACING untouched — the property the pre-arrival patience rule exists to
   protect (the unit once sat 53 m from its own goal). It is bounded inside the
   2.0 m `arrive` band on purpose.
3. **THE PACE MUST NOT COME FROM THE BALL-SIDE GATE.** First attempt reused
   `jog`, which is `x0.30` for a far-side player: a centre-back measured
   **0.2475 m/s and booked standing while moving**. The pace now comes from the
   ROLE's `_JOG_SPEED`.

### THE FIRST REPLACEMENT WAS ALSO WRONG, AND THE MEASUREMENT SAID SO

Chasing a POINT that circles the slot at 1 m/s while the step cap is 0.825 m/s
makes the man crawl whenever the point swings back toward him — measured, that
alone booked **25% of a centre-back's block time at 0.1–0.5 m/s** (moving, but
below the walking edge). Replaced with a triangle wave: constant pace, instant
reversal. The reversal tick is the only slow one (1 tick per 3.1 s, <2% of the
ground covered) and it is real — a defender turns.

**NO GLOBAL RNG.** The phase is a deterministic name hash
(`sum(ord(c)) % 628 / 100`), and the helper consumes nothing (pinned by test).

### TIME IS A UNION, NOT A SUM (the overlap)

Two instruments write the same minute: the 10 Hz integrator and the possession
trace. Their seconds OVERLAP, and the row carried their sum, so published
whole-match totals reached **6,195 s against 5,820 s of clock (106%)** and
`activity_measured_pct` could exceed 100. Fixed in two places, both published
rather than silent:

- per minute, merged measured bands above 60 s are scaled back into the minute
  (ratios preserved) and the correction is published as **`activity_overlap_s`**
  (160–470 s per player per match here);
- the TEMPLATE residue is redistributed to exactly `uncovered_s`. It had to be:
  `running = min(running, running + positive)` and the same for `jogging` were
  **no-ops**, so the "telemetry shaping" never fired — and turning them into
  real additions (which they now are) would otherwise let the residue create
  time. Measured before the redistribution: +20–70 s per player above the
  clock even after the overlap cap.
- Every published total is now exactly **5,820 s** for a player who played the
  whole match (5,160 for the man sent off at 87'). Gate G2 in the probe.

### MEASURED AFTER (one real match, seed 777, `_diag_measured_activity_verify.py`)

| | before | after |
|---|---|---|
| outfield measured coverage (net) | 42.1% | **91.1%** (min 80.1) |
| published total vs clock | 106% | **100%** |
| outfield distance | 6.6–8.1 km | **8.2–11.4 km** |
| template share of distance | ~100% | **11%** |
| standing share, by position | CB 33%, others 15–25% | **CB 16.9%, LB 5.8, RB 7.1, CDM 4.9, CM 5.9, CAM 6.3, LW 4.8, RW 3.8, ST 5.5** |

Per-position distance (mean, km): CB 8.41 · LB 9.66 · RB 10.14 · CDM 9.80 ·
CM 11.10 · CAM 10.57 · LW 10.54 · RW 10.71 · ST 10.04.

- **THE STANDING PROBLEM IS CLOSED, and it is not over-corrected.** Real GPS
  position bands are roughly CB 20–30%, midfield 10–15%, wide 8–12%; the engine
  now sits at or below the low end everywhere, with centre-backs the highest at
  16.9% (Crane ~23% individually — a `stopper_defender`). Note 23% of block time
  is a legitimate reading for a CB, so the earlier "Crane is still 23%" reading
  was NOT a residual defect; forcing it to zero would have been tuning real
  behaviour to an assertion.
- **~~THE REMAINING GAP IS PACE, AND IT IS POSITIONAL.~~** CBs covered 8.4 km
  against a real "~10.5" — **that band was unsourced and is wrong; the real
  figure is 9.2 km** (CIES n°68). The gap was ~0.8 km, not ~2.3. **FIXED
  2026-10-20**: see *CB PACE — THE ROW WAS AN OUTLIER* below. It was still a
  PITCH change (the `_JOG_SPEED` table), not an accounting one, exactly as this
  bullet argued.
- **ONE MATCH, ONE SEED, ONE ROSTER.** Direction is consistent across every
  surface the probe reports; magnitudes are a single sample.

### TWO PRE-EXISTING SUITE FAILURES, BISECTED NOT ASSUMED

`tests/test_positional_play.py` + `tests/test_small_game.py` → **48 passed,
2 failed, 1 skipped** (9m45s). Both were checked against `HEAD` with a
byte-exact copy-and-restore (`git show HEAD:match_engine.py` written with
`sys.stdin.buffer`, NOT PowerShell `>` — that emits UTF-16 and the suite dies
on *"source code string cannot contain null bytes"*):

- `test_fullback_can_tuck_inside` — **identical number at HEAD and with the
  changes (0.12435051023646793)**. Pre-existing, not caused by this work.
- `test_striker_gets_in_behind_not_offside` — **XPASS(strict) at HEAD too**, so
  the documented open finding from 2026-09-30 now passes and the `xfail`
  marker is stale. Pre-existing; the note in *SMALL GAME* should be re-read
  before trusting it.
- Also re-confirmed unchanged by these edits: the documented
  `test_out_of_possession_touchline_press_gated_by_press_intensity` family.

### GUARDS

`tests/test_measured_activity.py` grew 13 → **24**, with three new classes:
`BlockLineIsHeldNotFrozenTests` (behavioural on the extracted
`_block_shuffle_target`: constant pace, no crawl, bounded to the slot radius,
depth-only closure, step cap, **zero global RNG**, plus a structural pin that
the branch does not hard-zero `moved`), `DeadPhaseStillStepsTheOnBallManTests`
(an AST guard on the CALL SITES: the dead phase opts in, the live phase does
not, and the flag defaults off) and `TimeIsAUnionTests` (overlap scaling,
residue cannot create time, no minute above 60 s).

**NEGATIVE CONTROL `tools/diag/_diag_activity_guard2.py`: 4/4 CAUGHT**, files
restored byte-identically (hash asserted), baseline green afterwards: M1 block
freeze, M2 dead-phase opt-out, M3 overlap cap removed, M4 residue redistribution
removed. M1 initially reported MISS because the structural test split on the
FIRST `if _in_block:` — which is the earlier "move toward the slot" gate — so
the assertion was reading the wrong branch. **A test that passes by reading the
wrong region is the same defect as a probe reading the wrong key.** Fixed to
`rsplit` the last occurrence and assert the helper is inside that slice
(non-vacuity), and it then failed correctly.

**Gates:** `test_measured_activity` (24) + `test_distance_accounting` +
`test_keeper_attribution` + `test_opta_analytics_runs_sprints` → **52 passed**.

## CB PACE — the row was an outlier, and the benchmark was invented (2026-10-20)

Started as "close the CB pace gap quickly, then go to the on-ball neural layer".
The gap was real, much smaller than the note said, and the note's benchmark was
wrong. The order of work mattered: **measuring where a centre-back's metres come
from is what made the wrong benchmark visible**, because the branch decomposition
showed the shape was already identical to a full-back's and only the pace
differed — which is not what a ~2.3 km deficit looks like.

### ⛔ THE BENCHMARK WAS WRONG. IT IS THE SAME ERROR CLASS AS *SHAPE ANCHOR*'s.

`AGENTS.md` carried **"real centre back 10.5–11.0 km"** from memory, in a table
whose whole `real (PL/90)` column was written the same way. The published
figures, verified before fitting anything to them — **CIES Football Observatory
Monthly Report n°68 (Oct 2021), 7,855 matches across 31 leagues, SkillCorner
broadcast tracking, five position modalities**:

| position | distance | note |
|---|---|---|
| **centre backs** | **9.2 km** | the **LEAST** of any position |
| midfielders | 10.6 km | the most |
| wingers | — | most high-intensity (**932 m**) and sprinting (**211 m**) |
| centre backs | — | least in high-intensity and sprinting too |

So the real deficit was **~0.8 km, not ~2.3 km**, and the engine's role ordering
was already correct (CB lowest, midfielders highest — it matched the source
before anything was changed). This is the third benchmark error of the same
kind on file, after *SHAPE ANCHOR*'s "real PL: LB 22–30, CB 20–26" (which sent a
fullback fix ~25 m the wrong way) and the same note's Opta-touch-vs-time-median
confusion. **A band with no source and no aggregation is not a benchmark.**

Why it mattered practically: hitting 10.5–11.0 from 8.4 would have required
raising CB jog to ~2.5 m/s, above every midfielder, **inverting the role
ordering** — the exact failure the earlier fullback fix produced. The wrong
number would have produced a plausible-looking change that made the model worse.

### THE DEFECT: ONE OUTLIER ROW, NOT A MISSING MECHANISM

`tools/diag/_diag_cb_pace.py` (new) attributes every metre to the branch of
`_offball_move_player` that produced it, by reading the caller's frame locals
(`sys._getframe(1).f_locals`) — the `_diag_touch_sites.py` join, not an
inference. One real match, seed 777, per centre-back:

| branch | metres | mean speed |
|---|---|---|
| `to_target` | 1,801 | 1.13 m/s |
| `block_recover` | 1,579 | 4.94 |
| `block_shuffle` | 1,628 | 0.70 |
| `orbit` | 997 | 1.04 |
| `set_piece` | 17 | — |
| **total** | **6,021** (integrator) | **1.16** over 5,200 s |

**And the control table is what turned a CB question into a role-table
question.** Run for every role, the branch SHAPE is the same for all of them —
`block_shuffle` is 40–60% of stepped seconds for everyone, at 0.67–0.90 m/s —
and the CB is merely slower *in every branch* (`to_target` 1.13 vs LB 1.47–1.75,
`orbit` 1.04 vs 1.32–1.59). That is the signature of the pace table, not of a
centre-back-specific rule. The table had every outfield role at **1.75–2.0**
except `CB`/`DC`/`LCB`/`RCB` at **1.5** — inside the **WALKING** band by Di
Salvo's own edges (walk 0.5–2.0, jog 2.0–4.2), and the only outfield value below
the rest. The role with the least target movement compounded with the slowest
pace, which is why it showed worst there.

### THE CHANGE

`_JOG_SPEED`: `CB`/`DC`/`LCB`/`RCB` **1.5 → 1.9**, the full-back value, with the
CIES citation and the whole argument in a comment above the table. A CB's target
covers less ground per minute than a full-back's (deeper formation post, and
`COMPACTION` only shifts it 14% of the ball's travel), so equal pace still lands
him below a full-back's total — the real ordering — while no longer being a walk.

Also fixed in passing, and it is the alias trap this project has hit repeatedly:
**`DMC` sat at 1.5 in the centre-back row** while the live defensive-midfield
label is `CDM` at 2.0. `DMC` is emitted **nowhere in the repo** (its only two
occurrences are these two tables in `match_engine.py`), so the fix is
zero-risk — but it was a latent walk for a midfielder waiting on a future roster
change, and nothing behavioural would have revealed it.

### MEASURED — 3 seeds × 2 arms, ONE VALUE DIFFERENT

`tools/diag/_diag_cb_pace_ab.py` (new) switches the arm **at runtime**
(`MatchEngine._JOG_SPEED["CB"] = …`), so the arms differ in exactly one value.
This matters: the note's original before-figure came from a *different match*,
which is not a controlled pair.

| seed | off | on | delta |
|---|---|---|---|
| 777 | 8.41 | 8.69 | +0.28 |
| 57 | 8.26 | 9.25 | +0.99 |
| 31 | 8.94 | 9.22 | +0.75 |
| **mean** | **8.54** | **9.05** | **+0.52** |

**All three seeds move up, mean −0.15 km from the CIES 9.2 figure.** The arm
switch is a real difference in the football, so the report prints per-seed
per-position too, and that table is the important one: the **control roles swing
±1.5–2.0 km between arms in BOTH directions with no mechanism** (LB −2.06 at one
seed while RB +1.6 at the same seed) — the divergence signature documented in
*WIDTH GATE*, not an effect. Only CB has a mechanism, and only CB moves one way
every time.

**THE ACCOUNTING INVARIANT HELD IN ALL SIX MATCHES:** the maximum published
activity total equalled the match clock exactly (5820/5820/6000/5640/5760/5760).
That is the `test_measured_activity` guarantee, and it is the thing a pace change
must not break — **a pace change that makes the published totals wrong is a
regression, not a calibration.**

### WHAT THE REMAINING SHORTFALL IS, AND WHY IT WAS NOT CHASED

- **`block_shuffle` IS WAVE-BOUND, NOT JOG-BOUND — measured, and it is the one
  surprise.** It is 40–60% of every role's stepped seconds. Raising CB jog
  1.5→1.9 moved `to_target` **+174 m** and `orbit` **+158 m** but left
  `block_shuffle` at **1,629 m, unchanged**, because `_BLOCK_SHUFFLE_RATE ×
  _BLOCK_SHUFFLE_R_M` (1.0 m/s) sets that channel's pace and the jog-derived cap
  (`jog × _BLOCK_SHUFFLE_FRAC × DT`) never binds at either setting. So **no
  `_JOG_SPEED` edit can reach that channel at all** — it is ~44% of a
  centre-back's time. My first arithmetic assumed it scaled with jog and
  predicted +0.7 km; the measurement said +0.52 and explained the difference.
  **Do not predict a distance change from a constant's ratio; run the branch
  table.**
- **Raising it globally was estimated and rejected on measurement.** At
  ~1.2 m/s the lift is the right size for CB but pushes CAM/ST/LW **over** their
  CIES anchors (ST would land ~11.0 against 9.9), because the roles that are
  already correct share the same channel. That is a `COMPACTION`-class decision
  — global, multi-seed, moves every distance number in the project and the
  warehouse — and it is deliberately **not** in this change.
- **LB (~−0.8) and CDM (~−0.75) remain short against their anchors**, and the
  `off` arm shows LB at 9.81 *before* this work too, so this is pre-existing and
  not collateral. CB is no longer the outfield outlier, which was the item.
- **Centre-backs standing 16.9% is NOT a residual** — real CBs sit at 20–30%,
  the highest of any role. The engine is at the low end of correct, and forcing
  it lower to chase distance would be tuning real behaviour to an assertion.

### GUARDS

- **`tests/test_role_pace_calibration.py` (7 tests, new file)** — pins the
  mechanism in milliseconds: no outfield role below a 1.75 floor; the CB is not
  strictly the slowest outfielder; **all four CB aliases share one pace** (a
  roster emitting `LCB` after an edit to `CB` would silently keep the old walk,
  and nothing behavioural would show it); a **non-vacuity** case proving the
  pre-fix 1.5 really is below every other outfield value, so the assertions
  would have caught it; and two **AST** checks that the table is defined at class
  scope and has **≥2 load sites**, so it cannot become decorative. It does
  **not** assert a distance band — that needs a real match, one seed is a
  direction, and a single-draw match fixture is the flaky class this project has
  been burned by.
- **`tools/diag/_diag_pace_guard.py` — 3/3 CAUGHT**, `match_engine.py` restored
  byte-identically (sha256 asserted), baseline green before and after. It names
  *which* assertion fired, not just "something failed": M1 restores 1.5 → the
  floor test; M2 puts `DMC` back to 1.5 → the floor and non-vacuity tests; M3
  desynchronises `LCB` → the alias test. **M3 is the one that matters** — it is
  the failure mode no behavioural test in this repo can see.
- **Gates:** `test_role_pace_calibration` (7) + `test_measured_activity` (24) +
  `test_distance_accounting` + `test_keeper_attribution` +
  `test_opta_analytics_runs_sprints` → **59 passed (18.8 s)**.

## THE REWARD, THE SCHEMA AND THE MISSING INPUT (2026-10-20) — items 1, 2 and 4 of the neural plan

The dependency order in *THE ON-BALL NEURAL LAYER* was **reward → inputs →
receiver authority → truncation guard**. Items 1, 2 and 4 are DONE and gated
without a single match. Item 3 (receiver authority) is untouched — it is the
one that invalidates every calibration number in this file and must be its own
commit.

**Nothing here has been validated on the pitch yet.** The surrogate in `brains/`
is still the T1 table fitted to the RETIRED objective, and `brains/*.json` are
still T1. **Every number below is a unit-level or structural measurement, not a
football result.** The retrain and the pass-map gate are the next step and they
need CPU permission.

### 1. `threat_reward.py` — the objective, replaced

The retired reward scored **engine success**: completed pass 2.0 + advance,
retained carry 1.0, turnover 0.0. A square ball that completes 90% beats a
through ball that completes 55%, honestly, on that objective — which is why the
GA converged to lateral spread and why T2G160 could score the best fitness in
project history (0.6523 vs 0.5373) while being rejected on the pitch.

The replacement is **action value as a potential difference**:

    r = Φ(end) − Φ(start)

where `Φ` is the threat of a position, read from `momentum_model`'s xT grid —
which is itself **derived** from `XGEngine.calculate_geometric` at every cell
centre, so the reward's threat model and the engine's own xG cannot disagree.

- **Why a potential difference and not a possession return.** It is LOCAL. It
  needs no segmentation of a possession into a chain, and it **telescopes**:
  the sum of the differences along a sequence equals the terminal return — verified
  exactly (difference `0.00e+00`), not approximately.
- **Why not `xG`-at-cell as the potential.** For a shot, the potential difference
  from "here" to "goal" is ≈ the shot's own xG, which would make shooting worth
  exactly zero. So shots are special-cased: a shot scores its **absolute `xg`**,
  a goal `GOAL_VALUE = 1.0`, and a miss is charged by the **opponent's** threat
  at the point the ball dies (`LOSS_WEIGHT = 0.6`).
- **Failures are CHARGED, not scored 0.0.** A 0.0 for a turnover makes every
  risky action free in expectation, and the brain-only arm has already measured
  what that looks like: **41.7% progressive against a real 25–35%**, i.e. the
  mirror-image degenerate optimum. The charge is `-LOSS_WEIGHT ×` the
  opponent's threat at the loss point, so losing it in your own box is expensive
  and losing it in theirs is nearly free — which is true football.
- **`FitnessSurrogate.prior` had to become `0.0`.** With a signed reward whose
  median is ≈0, a `0.5` prior makes every UNOBSERVED intent (including SHOOT)
  beat every observed one. The prior is not a smoothing constant here; it is an
  assertion about the mean of the reward, and the measured mean is zero.
- **`EXECUTION_TYPES` (20 types) is exported by `threat_reward` and imported by
  the collector.** The collector's pairing set contained only `PASS`, so
  `PROGRESSIVE_PASS`, `SWITCH_OF_PLAY`, `THROUGH_BALL`, `CROSS_ATTEMPT` and
  `DRIBBLE_SUCCESS`/`FAIL` were **never paired**, and each unmatched pass
  permanently shifted that player's intent→outcome cursor. One concept, one
  definition, imported — not a second list that can drift.
  `CROSS_SUCCESS` is deliberately excluded (it is emitted **in addition to**
  `CROSS_ATTEMPT`, so pairing it would double-count); `SHOT_ATTEMPT` /
  `DRIBBLE_ATTEMPT` are never emitted at all.
- **`collapse_penalties` is now a keyword argument, default `False`.** The
  dominance penalty and the `n_effective < 6` penalty were **still live** in
  `synthetic_fitness` despite AGENTS.md recording the v3 rollback. They are the
  documented cause of T2's SECOND collapsed optimum (ST never shoots, CM abandons
  through-balls). Off by default; a retrain that leaves them on reproduces T2.
- The surrogate path re-bases before weighting:
  `base = 0.5 + 0.5·clip(mean_reward, −1, 1)`, then
  `fitness = base · (0.5 + 0.5·mean_conf)`. Without it a negative mean clamps to
  zero and the confidence weighting erases the gradient. It is a **monotone
  transform of the aggregate mean**, so brain ranking is preserved. The legacy
  `_context_reward` path keeps its old `scale = 1.5` and is untouched.

**One known weakness, recorded before any match is run.** The reward is
geometrically correct but **small for ordinary passes** — a 35 m forward ball is
worth ≈ +0.008, against ≈ +0.09 for a through ball into x=88, ≈ +0.19 for a
successful cross and ≈ +0.35 for a shot on target. That is xT being convex, and
it is the right shape; but it means the midfield-circulation signal is weak and
the fitted surrogate must be inspected for variance **before** evolution is
spent on it. A failed through ball deep in their half is also nearly free
(charge ≈ 0), so expected value favours trying — the brake is the bucket, since
high-pressure buckets contain mostly failures. That is a real limitation of an
immediate-reward bandit and it is why item 3 (receiver authority) will matter
more than any further reward tuning.

### 2. Schema alignment BY NAME — the silent truncation is gone

`brain_integration.py:524` truncated the padded sensor vector to the brain's
`w1` width with **no check and no log**. `MENU_NAMES["CB"]` grew 7 → 9 on
2026-10-05 and `brains/CB.json` is 31 wide, so the two new CB sensors were
**inert** (bit-identical vector, argmax differs on 0/4000 inputs) — a schema
change that invalidates every trained brain of a position degraded silently.

- `role_features.align_role_tail(family, role, tail_width, want_names=None,
  brain_label="")` is now the single place a role tail is produced. It is a
  **name-based selection with a width-only fallback**, not a hard break: a brain
  that declares `meta.role_features` gets exactly those features **by name**; a
  legacy brain gets the first N by width and **warns once per (family, width)**,
  naming the brain and saying the extra features are INERT.
  Verified byte-identical for every shipped brain — which is why it was allowed
  to land without a retrain.
- **`len(declared) != arch_tail_width` RAISES** (`RoleSchemaError`), in both
  directions and also for a name this build no longer emits. Zero-padding a
  short declaration would be the same silent degradation wearing a new coat.
- `brain_evolution.evolve()` now **stamps** `meta.role_features` on every new
  brain, clamped to the brain's **actual** tail width (minus
  `TACTICS_CONTEXT_D` for v3). Proven end to end: a fresh
  `evolve_brains.py --position RW` writes `arch [33, 32, 32, 10]` with all nine
  WING names declared, and `_aligned_tail` round-trips all nine with no warning.
- **⚠️ 5 of the 11 shipped brains now carry a warn-once inert input**: CB (7 of
  9), LB and RB (7 of 8 — FB grew with §3 below), LW and RW (8 of 9). That
  is the correct, honest state: the declaration mechanism reports exactly what
  the retrain will fix. **Do not silence those warnings to tidy the log.**
- The load-bearing test is
  `test_inserting_a_feature_in_the_middle_shifts_a_legacy_brain_but_not_a_declared_one`
  — a synthetic mid-schema insertion, asserting the legacy path returns
  `[10,20,30,40]` where the declared path returns `[10,30,40,50]`. Every growth
  so far has appended, so the by-name path's whole value is the case that has
  not happened yet.

### 3. The missing input — `pass_lane_open`, on FB and WING only

The plan said "add the two missing inputs: signed forward progress available to
the best forward option, and the rank of the best matrix option". **Measurement
narrowed the first and killed the second.**

- **The gap is narrower and more specific than the plan assumed.**
  `pass_lane_open` (openness of the best forward teammate) has always been in
  DM/CM/AM. CB has `same_side_lane_open`. GK has `pass_open`. **FB and WING were
  the only two families with no forward-option signal at all** — and they are
  the two roles that live on the touchline where that option is the whole point.
- **`overlap_option` LOOKS like it fills the gap and does not.** Measured, in
  both the FB and WING menus it computes
  `clamp(distance_to_nearest_forward_teammate / 30)` — a **DISTANCE** sensor,
  not an openness one. Worse, it returns `0.0` both when **nobody** is ahead
  and when a teammate is **standing on top of you**: two opposite facts, one
  value. It is misnamed, and the name is what stopped anyone looking.
  (`short_cut` is the same shape: `clamp(nearest_teammate_dist / 40)`, named as
  if it meant "a short reset exists". Recorded, not renamed — renaming a live
  sensor changes nothing functionally and would invalidate nothing, but it is a
  separate decision.)
- **APPENDED, not inserted.** The legacy loader takes "the first N", so
  appending leaves every shipped brain's first 7 (FB) / 8 (WING) numbers meaning
  exactly what they meant. Inserting anywhere else would silently relabel every
  feature after the insertion point — the precise failure the by-name schema
  exists to prevent.
- **One definition.** `_forward_options()` is a new shared helper, and
  `_mid_block` now calls it too, so `pass_lane_open` means the same thing in all
  five families. A test asserts `_mid_block` contains no second copy of the
  `max(...)` computation. A second implementation is how the same feature NAME
  comes to mean two things depending on position — the schema defect one level
  down.
- **ST is deliberately excluded**: a striker's best forward option is usually
  nobody, so the sensor would be a near-constant `0.0` — training budget spent
  on a feature with no variance.
- The **discriminating test** (the reason a new sensor was justified rather than
  a rename): two frames identical except the opposition marker moves 1 m → 20 m
  off our teammate. `overlap_option` **cannot move** (it measures our own
  teammate). `pass_lane_open` goes 0.0 → 1.0. A sensor that correlates with
  something already in the vector is not a new input.
- Widths, derived not typed: FB 31 → 32, WING 32 → 33; `V3_INPUT_D` is now
  `{43, 44, 45}`. **`percy_the_chosen.PERCY_INPUT_SIZE` is therefore 33**; his
  saved `brains/percy*.json` are 32 wide and now load through the legacy path
  with a warn-once until retrained.

**The plan's item 2 also called for a second sensor — "rank of the best matrix
option" — and it is deliberately NOT implemented.** It would couple perception
to `AttackingMatrix`, duplicate work the chain does immediately afterwards, and
amount to teaching the policy to imitate the matrix, which is the opposite of
the emergent play the item exists to enable. This is a reasoned
non-implementation and it is recorded so it is not re-proposed as an oversight.

### 4. Two stale snapshot tests that had been failing since 2026-10-05

`tests/test_tactics_context.py` asserted `sorted(set(V3_INPUT_D.values())) ==
[43, 44]` and listed `45` among the widths the schema must **reject**. The
2026-10-05 CB sensor addition made 45 legitimately valid, so both checks had
been failing for ~15 days without anyone noticing — the project's recurring
"a hard-coded snapshot pinned as an invariant" defect, this time in a test.

Both are now **derived**: the width set is checked against
`{V2_INPUT_D[f] + TACTICS_CONTEXT_D}`, and the bad-width set is computed as
neighbours of the valid set, so an off-by-one role block stays the failure mode
being tested and the list cannot go stale again. 86/86 pass.

### Gates and the negative control

- **`tests/test_threat_reward.py` (47, new file).** Verticality ordering
  (forward > square > backward, backward **negative**), longer gain worth more,
  the through-ball margin floor, goal > best shot, penalty == `GOAL_VALUE`, shot
  == absolute xg, on-target > off-target > 0, missed penalty negative, `SAVE`
  scores as a shot on target, the loss-charge geometry for all five loss types,
  neutral events exactly `0.0`, telescoping exact, mirror symmetry both
  directions, 3,000-sample bounds in `[−1, 1]`, `prior == 0.0`, the reward
  definition travelling through `save`/`load`/`report()`, **and a non-vacuity
  case that reproduces the RETIRED reward and shows it ranks the square ball
  above the through ball while the new one reverses it.** That last test is the
  file's reason to exist: without it, "the new reward is better" is an assertion.
- **`tests/test_brain_schema_alignment.py` (39, new file).** By-name vs legacy,
  the mid-schema insertion proof, warn-once per (family, width), every shipped
  brain scanned for silent truncation (11 found, 5 warn), the raises, the
  evolver's stamping, and AST guards that **both** padded branches route
  through `_aligned_tail` with no bare `role[:` slice left anywhere.
- **`tools/diag/_diag_neural_guard.py` — 11/11 CAUGHT**, files restored
  byte-identically (sha256 asserted), baseline green before and after. M1
  reverses the potential difference, M2 restores `prior = 0.5`, M3 narrows
  `EXECUTION_TYPES` to `{"PASS"}`, M4 removes the re-base, M5 defaults
  `collapse_penalties` True, M6 removes the missing-name raise, M7 removes the
  count check, M8 silences the legacy warning, **M9 makes FB's forward option
  the distance sensor it replaced**, **M10 makes `_mid_block` recompute the lane
  instead of sharing the helper**, **M11 declares the feature in the menu while
  the FB block never appends it**. Every mutant is required to still compile —
  a syntax error counted as a caught mutation proves nothing about which
  assertion pins the defect.
- **Combined gate: 213 passed** (threat_reward, brain_schema_alignment,
  momentum_model, football_brain, role_pace_calibration, measured_activity,
  distance_accounting, keeper_attribution, opta_analytics_runs_sprints,
  tactics_context) + `test_role_features.py` ALL PASS.
- `tests/test_momentum_model.py` gained `PURE_TRAINING_CONSUMERS =
  {"threat_reward.py"}` — **one asserted, documented exemption** from "no engine
  module may import `momentum_model`" — with a non-vacuity check that the guard
  really reached that file and a companion test that the training reward is a
  pure reader of the grid. Hiding the import behind `importlib` was rejected as
  dishonest; an exemption you can read is better than a guard you can dodge.
- The import cycle this creates (`match_engine → event_chain →
  brain_integration → threat_reward → momentum_model → match_engine`) was
  checked, not assumed: `match_engine` never statically imports the chain, so
  `momentum_model`'s `from match_engine import …` always finds a fully
  initialised module.

### What is NOT done

- **The surrogate has not been rebuilt and no brain has been re-evolved.** That
  needs ~6 real matches for collection and 11 × 110–155 s for evolution.
- **The reward's variance on real matches is unmeasured.** It must be inspected
  before evolution is spent: if the ordinary-pass signal is too weak in the
  fitted table, the GA has no gradient and the retrain is wasted.
- **Item 3 (policy authoritative over `forced_receiver`) is untouched.** Until
  it lands, none of this is visible on a pass map, because the receiver governing
  the pass endpoint is the matrix (1178) / phase (296) / wide-combo (225), not
  the policy (1).
- **The acceptance gate is pass maps and the eye, never fitness.**

## POLICY TARGET DEFERENCE — who names the receiver, and an experiment allowed to see it (2026-10-19)

Started from `_diag_receiver_owner.py`'s measurement: on an ordinary pass the
receiver is named by **AttackingMatrix 1178 / TacticalPhase 296 / wide-combo 225
/ heuristic 13 / policy 1**. Two wrong readings of that number had to be
discarded before anything was built.

- ⛔ **IT IS NOT "98% OF PASSES ARE WRONG."** The pass ENDPOINT is not the
  decision. The policy owns the ACTION (~98% of on-ball actions carry
  `role_fallback` on only a handful), the pass CLASS, the cross, the carry and
  the through ball **including its destination**. The structure owns the NOUN on
  ordinary circulation. The user's framing is the correct one and it is now the
  project's: **the player picks the verb, the structure picks the noun.**
  Positional play IS a constraint system, so a receiver chosen by the matrix is
  a coherent philosophy rather than a gap.
- ⛔ **AND THE "1 IN 686" FIGURE WAS AN AUTHORITY MEASUREMENT READ AS A QUALITY
  MEASUREMENT.** Nobody had ever measured whether the policy's own receiver is
  *good*, because it is computed every touch by `_find_target` and then
  discarded. That is the only open question here, and it is a small one.

### What the export could not say, and now can

`decision_authority` has always answered **who chose the action**, and on an
ordinary pass it reads `player_policy` unconditionally. Nothing in the export
distinguished that from the ~98% of ordinary passes whose RECEIVER came from the
matrix. New, additive, present with the experiment OFF:

- pass metadata `target_authority` ∈ `matrix | phase | wide_combo |
  gk_distribution | policy | policy_deference | heuristic | soul | unknown`,
  stamped from a `receiver_authority` local set at every assignment in the
  receiver ladder (including the soul override, which is easy to miss because it
  happens after the ladder and silently replaces the receiver).
- `plofa_export._decision` surfaces it beside `authority`, the `schema` block
  documents every value and states the distinction, and `_index` publishes a
  `target_authority` histogram — so "who names the receiver?" is answerable from
  a file rather than from a probe that wraps internals.

### The experiment: `POLICY_TARGET_DEFERENCE`, default **OFF**

`PossessionChain._policy_target_deference()` — a pure classmethod, no writes, **no
RNG** — compares the policy's target against the option the matrix ACTUALLY
CHOSE, **in the matrix's own units**, and defers only inside a relative margin.

- **Why bounded.** Running the policy alone is Checkpoint 32b: GK receptions
  37.7 → 25.0, events 3000-3700 → 1900-2000. The margin opens only where the
  matrix's score carries no information.
- **Why `value` and not `max(scores)`.** The cascade deliberately selects from
  SUBSETS (`best_far` / `best_close` / `target_opt`), so the global max is a
  number about a menu nobody ordered. The chosen option's own score is the
  decision that was made.
- **Exclusions, each a non-preference:** `forced_source != "matrix"` (a phase
  directive is a team order — "even a genius obeys the reset"; the wide
  combination is a role rule; an already-applied deference is not re-decided),
  and any `back_to_keeper_*` reason (safety, and those options are scored outside
  the candidate list so `scores` cannot compare them honestly).
- **`AttackingDecision` gained `value` + `scores`** (appended, defaulted), because
  without them "nearly indifferent" is unevaluable. `_pass` populates both.
- **The margin (0.05) is NOT FITTED and says so.** `target_deference` is stamped
  on **every pass the flag inspects**, fired or not, carrying `matrix_value`,
  `policy_value` and `gap_frac`; the export aggregates the distribution. One
  match is meant to set the constant, not an opinion.

### A gate that would not have fired, found by its own test

The first version returned early on `matrix_value <= 0.0`, to keep `gap_frac`
from dividing by zero. But **0.0 is what the "no clear option" branch produces** —
every candidate the matrix weighed scored nothing, while an UNCHOSEN far option
scored 0.199. That early return made the rule **inert precisely where the
structure has no preference**, which is a large share of real passes and the case
the rule exists for. Now: `matrix_value <= 0` means the matrix is silent, not
indifferent, and the policy may speak (`gap_frac = 0.0`, no division).

### Evidence

- **`tests/test_policy_deference.py` (18)** — the payload, every exclusion, the
  zero-preference case, the relative reference (a 0.20 chosen value against a
  0.90 unchosen option), agreement, an unknown policy target (refused, same rule
  as `_named_shooter`), the no-RNG property, the end-to-end delivery, and the
  two export-facing fields.
- **OFF-path inertness is proved, not asserted**: 60 seeded possessions run twice
  on a freshly-built world each time — once with the rule present, once with it
  replaced by a no-op — compared on every field that decides football. Identical;
  the only payload difference is the new descriptive keys. (A fresh world per arm
  matters: a shared `PositionEngine` would let arm A's touches change arm B's
  geometry, which is how a "no difference" result gets manufactured.)
- **Negative control `tools/diag/_diag_deference_guard.py`: 5/5 CAUGHT**, file
  restored byte-identically (sha256 asserted), suite green before and after: M1
  provenance guard dropped, M2 keeper-release guard dropped, M3 global-max
  reference, M4 RNG draw, M5 `target_authority` not stamped.
  Its first run reported **0/5 with "ANCHOR MATCHED 0 TIMES"** because
  `event_chain.py` is CRLF here and the anchors were written with LF — the exact
  no-op-mutation failure the control exists to catch, reproduced in the control.
  Fixed by converting to the file's own line ending before matching.

### ⚠️ TWO PRE-EXISTING FAILURES FOUND BY THIS WORK, BISECTED AT HEAD

Both fail identically with `event_chain.py` and `attacking_matrix.py` restored to
HEAD bytes (`_diag_head_bisect.py`, sha256-asserted restore):

1. **`test_attacking_matrix.py::test_possession_chain_policy_target_beats_matrix_target`**
   asserts *"the matrix may score alternatives, but it cannot replace the
   receiver selected by a pass-like player policy."* **The shipped engine does
   the opposite** and has since the matrix was wired. The test encodes an
   ASPIRATION, not the design — and it has presumably been failing ever since.
   Do not "fix" the code to satisfy it; either delete it or rewrite it to assert
   the measured ownership. This is also the clearest evidence that the
   player-verb / structure-noun split was never written down: a test was asserting
   the other model.
2. **`test_attacking_matrix.py::test_shot_origin_binds_to_shooter_real_position`**
   — asserts a shot event exists from a no-context origin and draws none. Same
   class as the documented probability-gated fixtures (`test_chance_provenance`):
   a single draw is wrong by construction. Needs a seed sweep with a
   non-vacuity assertion, not a code change.

### MEASURED — THE FIRST LIVE RUN OF THE RULE (2026-10-19, seed 777)

`tools/diag/_diag_deference_ab.py --arm both --seed 777`, real 26/27 roster
(Hartwell City HIGH_PRESS v Thornfield United FLUID_COUNTER), one match per arm.
This is the measurement the margin was supposed to come from. It is **one match
and one seed**, and the arms diverge after the first applied deference, so every
number below is a single sample and the pass-map comparison is NOT a controlled
per-pass causal effect. What it establishes is the rule's *mechanism and dose*,
which is what was missing.

| | OFF (shipped) | ON |
|---|---|---|
| events | 3591 | 3433 |
| score | 9–2 | 3–0 |
| `target_authority` | matrix 746 (87.8%) · `?` 64 · wide_combo 34 · heuristic 3 · soul 2 · phase 1 | matrix 746 (87.8%) · policy_deference **58** · `?` 50 · wide_combo 34 · heuristic 3 · soul 2 · phase 1 |
| rule inspected | — | **347** |
| fired / applied | — | 172 / **58 (16.7% of inspected, 7.5% of passes)** |

**The margin was NOT the binding parameter, and that is the finding.**

| `gap_frac` set | n | p10 | p50 | p90 | min | max |
|---|---|---|---|---|---|---|
| fired, **matrix-silent** (`matrix_value <= 0`) | ~114+ | — | 0.000 | 0.000 | −1.058 | 0.050 |
| fired, comparable | ~58 | — | — | — | — | 0.050 |
| unfired | 175 | +0.146 | +0.773 | +1.000 | **+0.050** | — |

The `fired` set is **dominated by `gap_frac == 0.000`**, i.e. by the
`matrix_value <= 0` branch — the one case where the relative margin is not
applied at all. So the headline "the rule fires on ~50% of inspected passes"
is mostly the rule firing *unconditionally on silence*, and `gap_frac` as a
single number **conflated a comparison with its absence**. On the genuine
comparison (`matrix_value > 0`) the split sits exactly at the margin
(unfired min = +0.050 = the constant), so 0.05 is *internally consistent* with
the data — but it is still fitted to **one match**, and it is not the number
that governs most firings. **Do not quote 0.05 as calibrated.**

Consequence, shipped: the payload now separates the two modes —
`matrix_silent` (bool) and `matrix_reason` (str, e.g. `no_clear_option`) ride
beside `gap_frac`. `"chosen value is 0.0"` is **not** by itself evidence that the
structure had no preference: the matrix's cascade can choose an option that
scored 0.0 while better-scoring options exist (a live sample: matrix=0.0,
policy=0.2379). Which is why the reason is recorded rather than inferred.
`tests/test_policy_deference.py` 18 → **19**: one new test pins that two passes
with the SAME `gap_frac` of 0.0 are nevertheless distinguishable in the payload.

**Dose without direction.** The rule changed **58 receivers** and the pass map
did not move toward verticality on either team: geometry forward share 27.3% →
27.3% (Hartwell) and 27.5% → 28.5% (Thornfield); the **band-free** statistic —
median signed advance in the attacking frame — went from **−0.7 → −0.9 m** and
**−0.2 → −0.4 m**, i.e. very slightly MORE lateral, not less. Median pass length
10.3 → 11.3 and 10.9 → 10.4. **So there is no evidence here that the policy's
receiver is more vertical than the structure's.** The honest conclusion is:
the rule is live, measurably dosed, and directionally inert in this sample.

**A probe error of mine, eighth instance of the documented shape.** The first
version of the probe used its own `SQUARE_M = 5.0` dead band and then reported
**"30–35% of passes disagree with `pass_direction`"** — a clean, alarming number
under a heading that asserted more than its expression computed. The engine's
band is `pass_classifier.classify_direction`: `|signed_dx| <= 1.0` is sideways.
The disagreement was **my threshold, not the code**. Fixed to apply the ENGINE'S
classifier to the arrival point; what remains is the already-documented
aim-`end_px`-vs-arrival-`end_x` question, now measurable instead of inflated.
**When a check disagrees with the code, verify the check's frame — or its
thresholds — before the code.**

**A real, band-free finding falls out of the same run, and it is not about the
deference.** The engine's median pass goes *slightly backwards*: **−0.7 m and
−0.2 m** in the attacking frame across 461 + 349 passes. The event's
`pass_direction` label (computed from the AIMED point) reads fwd ~40 / square
~10 / back ~48%, against a real-football ~35–40 / 45–50 / 10–15. Under ANY dead
band the engine is backward-heavy; the band-free median is the statement that
does not depend on the band, and it corroborates the long-standing *MEASUREMENTS*
note "median displacement −0.2 m". **Recorded, not changed** — it is the
`COMPACTION` / shape-anchor calibration question, not a deference question.

### Still open

- **The margin is fitted to ONE match, and it is not the binding parameter for
  most firings.** Two things need a second run: the precise split of
  silent-vs-comparable firings (now recorded, not yet counted), and whether
  firing *unconditionally* on `matrix_value <= 0` is wanted at all. The matrix's
  cascade can pick a zero-scoring option while better ones exist, so "the chosen
  option scored 0" may be a deliberate safety default rather than silence.
  `matrix_reason` is what will settle it. **Do not change the silent-case
  behaviour on the current evidence.**
- **`carry_prob` is computed and never read** (`event_chain.py` ~1997/2006,
  raised to 0.80, referenced nowhere), so the comment claiming the winger's
  instinct gates carry-vs-pass is false; `winger_bias`/`fb_bias` direction
  steering (2023-2049) is the part that does work. Recorded, not changed.
- **The stale comment at ~2172** ("The policy owns the intended receiver for
  every pass-like intent") was REWRITTEN in place to describe the fallback ladder
  it actually implements, with the ownership counts. A comment that overstates
  the code is how the next agent inherits a false model.
- **`target_authority` covers ~93% of PASS events, not 100%.** The remainder is
  PASS events built outside `PossessionChain` (transition and set-piece chains),
  which report `?`. The schema documents this; do not read `?` as a defect.
- **T2G160's pass-map verdict is contaminated** and should be re-read before it
  is treated as final: `validate_neural_xl` runs the shipped default
  (`POLICY_INTENT_AUTHORITY` off), so BOTH arms' ordinary pass endpoints were
  matrix-chosen. Class mix, action mix and through-ball targets were genuinely
  brain-driven; the arrows were not.

## THE ON-BALL NEURAL LAYER — THE OBJECTIVE IS THE CEILING (2026-10-19)

**WRITTEN SO IT CAN BE TACKLED LATER. Nothing here was changed on the day it was
written.** A design review of the on-ball policy, assembled from measurements
already in this file plus three probes (`_diag_receiver_owner.py`,
`_diag_receiver_signal.py`, `_diag_brain_only.py`). Read this before spending
another hour of CPU on evolution.

### THE SHORT VERSION

The component is well built — numpy, GA, a learned surrogate, an auto-load
registry, a self-training loop, a real harness. **The weakness is placement and
objective, not capacity.** `NeuralDecisionBrain` chooses one of ten intent
LABELS; everything consequential after that — who receives, where the ball goes,
whether a shot fires — is hand-written. And it is evolved against a surrogate
whose rewards are **engine success**, so it is being asked to optimise a model of
the physics, not a model of football. 2,186 params is a reasonable size for a
10-class label from 24 inputs. That was never the problem.

### "THE NEURAL SYSTEM" IS FOUR THINGS, AND ONLY ONE IS WEAK

| component | params | chooses | load-bearing? | evidence |
|---|---|---|---|---|
| `NeuralDecisionBrain` | 2,186 | 1 of 10 intent labels | **headline only** | its chosen receiver governs **1 pass in 686** |
| `TeamPressBrain` | 1,889 | scalar `g` in [0,1] | **yes**, permanent wiring | conceded 8→3, wins 5/6 vs 3/6 |
| `DefensiveActionBrain` | 1,988 | tackle/intercept/clear/block | **yes**, three sites | shots against 48 vs 65, tackles 99 vs 19 |
| Percy (per player) | 6,922 | 1 of 10 labels + 8 WING dims | one player | xG 0.838 vs 0.356; **points 13 vs 16** |

The two controllers are NOT weak. They replace hand-tuned tables with something
that measurably moved outcomes, on matched seeds with the on-ball policy pinned.
A small net doing one narrow job is the correct use of a small net. **Do not
lump them into this critique.**

### FIVE REASONS, IN INCREASING ORDER OF IMPORTANCE

**1. IT DOES NOT CHOOSE THE PASS, IT CHOOSES THE VERB.**
Output is `_INTENT_BY_INDEX[argmax]`. `_find_target` then runs a hand-written
geometric lookup; the matrix scores six hand-placed coefficients; the phase layer
orders a circulation target. Only if BOTH abstain is the brain's own `target`
read — and that happened on **1 of 686 passes**. Ownership measured:
AttackingMatrix **1,178** / TacticalPhase **296** / brain **1**
(`_diag_receiver_owner.py`). `_find_target` runs ~896 times a match and is
effectively decorative. **The arrow's endpoint on every pass map has ~zero neural
contribution.** A forcing probe that overrode 70 receivers produced a
byte-identical match (2,827 events, digest `b075ecc2520f50ca`) — and was only
believed because `_diag_lever_decisive.py` first proved the lever was live.

**2. THE SURROGATE IS A MODEL OF THE ENGINE, NOT OF FOOTBALL — and this is the
whole problem.** `{position: {bucket: {intent: score}}}` learned from 3,286
recorded outcomes over 6 matches. Rewards are engine success: completed PASS 2.0
+advance, retained CARRY 1.0 + distance, turnover 0.0, GOAL 4.0, SOT 1.5, default
0.5. So it answers *"which intent does this engine make work?"* — and the engine's
physics has no concept of a good pass. If a lateral square ball completes 90% and
a through-ball 55%, the surrogate honestly reports square is better, and the GA
converges to square. **That is T2G160 exactly**: the best fitness number the
project has ever produced (0.6523 vs 0.5373 heuristic, all eleven positions
saturated at 1.0000) and pass maps indistinguishable from the heuristic, rejected
after three real matches. The v3 penalties did not fix it — they forbade one
collapse and the GA found a second (ST never shooting, CM never playing
through-balls). **A constraint on a bad objective relocates the failure; it does
not correct it.**

**3. THE FITNESS SIGNAL HAS SATURATED WHERE THE SEARCH STOPS.** T1 per-position
surrogate fitness is **0.6978–0.9842** (ST .6978, GK .7318, CF .8063, CDM .8315,
LB .8367, RB .8707, CAM .8834, CB .8988, LW .9450, RW .9499, CM .9842) against a
real-match team fitness of **0.6208**. An objective that scores a CM at 0.9842
while the pitch shows a damper has stopped discriminating — there is no gradient
left at the top of the search, which is why more generations produce *better
numbers* and *worse football*. Budget was never the binding constraint.

**4. NO CREDIT ASSIGNMENT ACROSS A POSSESSION.** Each action is scored by its
IMMEDIATE bucket outcome. A through-ball that creates a 3v2 is scored the same as
a square ball that completes. Nothing carries the value of what the action
*enabled*. Measured consequence (`_diag_brain_only.py`, 3 matches per arm,
separate processes):

| | deciders | brain only | real PL |
|---|---|---|---|
| events | 2,866 | 2,382 | — |
| passes | 808 | 641 | — |
| GK receptions | 37.7 | 25.0 | — |
| median pass length | 9.7 m | **23.9 m** | 15–18 m |
| forward / square / backward | 27.9 / 41.0 / 31.1 | **35.8 / 27.0 / 37.2** | 35–40 / 45–50 / 10–15 |
| progressive | 26.5% | **41.7%** | 25–35% |
| median displacement | −0.2 m | −0.07 m | — |

Read that as the file already does: **the deciders are a DAMPER, not a source of
direction — the distribution got wider, not more biased.** An immediate-reward
bandit widens a distribution. It cannot express "play this way".

**5. THE INPUTS CANNOT CONTAIN THE DISTINCTION.** 24 dims of pressure, space,
thirds, centrality, goal-closeness. Nowhere is *"is there a forward option
available and how good is it?"* The net literally cannot know a progressive pass
exists — which is why a forwardness HEAD cannot work on top of inputs that do not
carry the fact. The surrogate's bucket uses 6 of the 24 dims, so the REWARD sees a
6-dim discretisation of a 24-dim state.

**Percy is the only positive evidence** and it is worth keeping in view: 8 WING
role dims + 64-wide layers produced a genuinely different intent mix (SWITCH 37
vs 18, CARRY 9 vs 39), double the xG, +1 SOT, decision quality 0.801 vs 0.705 —
**and three fewer team points (13 vs 16).** More role-relevant inputs DID move the
football. They did not move the results. That is the trap in one line.

### TWO MAINTENANCE LANDMINES (fix regardless of the redesign)

- **`brain_integration.py:524` SILENTLY TRUNCATES a 33-wide vector back to 31.**
  The two CB sensors added 2026-10-05 are inert against the shipped brain for
  exactly this reason (argmax differs on **0/4000** inputs). A schema change that
  invalidates every trained CB brain logs **nothing** and degrades to the old
  behaviour. Make a width mismatch RAISE.
- **`POLICY_INTENT_AUTHORITY` WAS HALF-WIRED until 2026-09-30** — it gated the
  five receiver sites but NOT the two that force the delivery class, so
  **Checkpoint 32b is a PARTIAL result** and the "brain decides alone" experiment
  had never actually been run. Also note matches are now reproducible across
  processes (crc32, 2026-10-04), so the old "A/B across two `simulate()` calls is
  invalid" premise no longer applies.

### WHAT TO CHANGE, IN DEPENDENCY ORDER

**1. REPLACE THE REWARD BEFORE TOUCHING THE NETWORK. (≈1 day + a gate.)**
The instrument already exists: `momentum_model`'s xT grid is **derived** from
`XGEngine.calculate_geometric` at each cell centre, and is already verified to
value a goal EXACTLY as the shot that produced it (0.20230 vs 0.20230). Score a
possession by its terminal xT/xG return and attribute the action's contribution
along the chain, instead of scoring the action by whether the engine completed it.
That is the difference between a contextual bandit and something with a directional
preference. `brain_self_trainer.py` already runs collect → append → refit →
evolve(scratch) → gate → promote, so the loop exists.

**2. ADD THE TWO MISSING INPUTS. (Same pass as 1 — it is a retrain either way.)**
Signed forward progress available to the best forward option, and the rank of the
best matrix option. Without these no objective can be *expressed*, only *sensed*.
Any input-width change must also fix landmine 1 or it will silently truncate.

**3. MAKE THE POLICY AUTHORITATIVE OVER `forced_receiver`. (Weeks.)** Until it is,
items 1–2 are unmeasurable on a pass map because the brain still does not name the
endpoint. Build behind a flag, default OFF, gate on pass maps AND the eye. This
invalidates every calibration number in this file when it lands, which is why it
must not be mixed into the same commit as anything else.

**4. KILL THE SILENT TRUNCATION.** Cheap, independent, do it first if you touch
nothing else.

### WHAT NOT TO DO

- **DO NOT run more generations on the v3 surrogate.** G40 → 11 goals, G80 → 17,
  G160 → 19 against a heuristic's 21–23, fitness rising monotonically the whole
  way. This is the most expensive lesson in the project and it is already paid for.
- **DO NOT start with a bigger network.** Capacity is not the binding constraint;
  the CM at 0.9842 surrogate fitness proves it. Percy gained from capacity only
  because he ALSO gained role-relevant inputs.
- **DO NOT add penalties to a bad objective.** See T2G160.
- **DO NOT define "better football" by fitness.** The project's own history says a
  metric can improve while the pitch gets worse.

### ACCEPTANCE GATE (fitness is NOT part of it)

Matched seeds, matched tactical frame, three things at once:
1. forward / square / backward moves from 27.9 / 41.0 / 31.1 toward
   35–40 / 45–50 / 10–15;
2. median pass length lands in **15–18 m with the deciders ON** (currently 9.7
   against the brain-only 23.9);
3. the eye: three pass maps that read vertical, with a reason for each arrow,
   rather than lateral spread wearing a neural badge.

### OPEN QUESTIONS / UNKNOWNS (do not pretend these are settled)

- Whether xT contribution along a chain is **identifiable** from a single
  possession, or whether it needs the paired counterfactual collection the v3
  corpus used.
- Whether policy-authoritative receivers can be gated WITHOUT shifting the match
  stream in a way that invalidates the comparison (the corner-delivery work
  accepted a deliberate stream shift; this one needs a matched-seed plan).
- Whether role-conditioned nets (the Percy path) scale to eleven players without
  their own corpus each.
- **One match / one seed** for most numbers quoted here. The mechanisms are
  established; the magnitudes are single samples.

## THE REWARD, THE SCHEMA AND THE MISSING INPUT (2026-10-20) — items 1, 2 and 4 of the neural plan

The dependency order in *THE ON-BALL NEURAL LAYER* was **reward → inputs →
receiver authority → truncation guard**. Items 1, 2 and 4 are DONE and gated
without a single match. Item 3 (receiver authority) is untouched — it is the
one that invalidates every calibration number in this file and must be its own
commit.

**Nothing here has been validated on the pitch yet.** The surrogate in
`brains/` is still the T1 table fitted to the RETIRED objective, and
`brains/*.json` are still T1. **Every number below is a unit-level or
structural measurement, not a football result.** The retrain and the pass-map
gate are the next step and they need CPU permission.

### 1. `threat_reward.py` — the objective, replaced

The retired reward scored **engine success**: completed pass 2.0 + advance,
retained carry 1.0 + distance, turnover 0.0. A square ball that completes 90%
beats a through ball that completes 55%, honestly, on that objective — which is
why the GA converged to lateral spread and why T2G160 could score the best
fitness in project history (0.6523 vs 0.5373) while being rejected on the
pitch.

The replacement is **action value as a potential difference**:

    r = Φ(end) − Φ(start)

where `Φ` is the threat of a position, read from `momentum_model`'s xT grid —
which is itself **derived** from `XGEngine.calculate_geometric` at every cell
centre, so the reward's threat model and the engine's own xG cannot disagree.

- **Why a potential difference and not a possession return.** It is LOCAL. It
  needs no segmentation of a possession into a chain, and it **telescopes**:
  the sum of the differences along a sequence equals the terminal return —
  verified exactly (difference `0.00e+00`), not approximately.
- **Why not `xG`-at-cell as the potential.** For a shot, the potential
  difference from "here" to "goal" is ≈ the shot's own xG, which would make
  shooting worth exactly zero. So shots are special-cased: a shot scores its
  **absolute `xg`**, a goal `GOAL_VALUE = 1.0`, and a miss is charged by the
  **opponent's** threat at the point the ball dies (`LOSS_WEIGHT = 0.6`).
- **Failures are CHARGED, not scored 0.0.** A 0.0 for a turnover makes every
  risky action free in expectation, and the brain-only arm has already measured
  what that looks like: **41.7% progressive against a real 25–35%**, i.e. the
  mirror-image degenerate optimum. The charge is `-LOSS_WEIGHT ×` the
  opponent's threat at the loss point, so losing it in your own box is expensive
  and losing it in theirs is nearly free — which is true football.
- **`FitnessSurrogate.prior` had to become `0.0`.** With a signed reward whose
  median is ≈0, a `0.5` prior makes every UNOBSERVED intent (including SHOOT)
  beat every observed one. The prior is not a smoothing constant here; it is an
  assertion about the mean of the reward, and the measured mean is zero.
- **`EXECUTION_TYPES` (20 types) is exported by `threat_reward` and imported by
  the collector.** The collector's pairing set contained only `PASS`, so
  `PROGRESSIVE_PASS`, `SWITCH_OF_PLAY`, `THROUGH_BALL`, `CROSS_ATTEMPT` and
  `DRIBBLE_SUCCESS`/`FAIL` were **never paired**, and each unmatched pass
  permanently shifted that player's intent→outcome cursor. One concept, one
  definition, imported — not a second list that can drift.
  `CROSS_SUCCESS` is deliberately excluded (it is emitted **in addition to**
  `CROSS_ATTEMPT`, so pairing it would double-count); `SHOT_ATTEMPT` /
  `DRIBBLE_ATTEMPT` are never emitted at all.
- **`collapse_penalties` is now a keyword argument, default `False`.** The
  dominance penalty and the `n_effective < 6` penalty were **still live** in
  `synthetic_fitness` despite AGENTS.md recording the v3 rollback. They are the
  documented cause of T2's SECOND collapsed optimum (ST never shoots, CM
  abandons through-balls). Off by default; a retrain that leaves them on
  reproduces T2.
- The surrogate path re-bases before weighting:
  `base = 0.5 + 0.5·clip(mean_reward, −1, 1)`, then
  `fitness = base · (0.5 + 0.5·mean_conf)`. Without it a negative mean clamps
  to zero and the confidence weighting erases the gradient. It is a
  **monotone transform of the aggregate mean**, so brain ranking is preserved.
  The legacy `_context_reward` path keeps its old `scale = 1.5` and is
  untouched.

**One known weakness, recorded before any match is run.** The reward is
geometrically correct but **small for ordinary passes** — a 35 m forward ball
is worth ≈ +0.008, against ≈ +0.09 for a through ball into x=88, ≈ +0.19 for
a successful cross and ≈ +0.35 for a shot on target. That is xT being convex,
and it is the right shape; but it means the midfield-circulation signal is weak
and the fitted surrogate must be inspected for variance **before** evolution is
spent on it. A failed through ball deep in their half is also nearly free
(charge ≈ 0), so expected value favours trying — the brake is the bucket,
since high-pressure buckets contain mostly failures. That is a real limitation
of an immediate-reward bandit and it is why item 3 (receiver authority) will
matter more than any further reward tuning.

### 2. Schema alignment BY NAME — the silent truncation is gone

`brain_integration.py:524` truncated the padded sensor vector to the brain's
`w1` width with **no check and no log**. `MENU_NAMES["CB"]` grew 7 → 9 on
2026-10-05 and `brains/CB.json` is 31 wide, so the two new CB sensors were
**inert** (bit-identical vector, argmax differs on 0/4000 inputs) — a schema
change that invalidates every trained brain of a position degraded silently.

- `role_features.align_role_tail(family, role, tail_width, want_names=None,
  brain_label="")` is now the single place a role tail is produced. It is a
  **name-based selection with a width-only fallback**, not a hard break: a
  brain that declares `meta.role_features` gets exactly those features **by
  name**; a legacy brain gets the first N by width and **warns once per
  (family, width)**, naming the brain and saying the extra features are INERT.
  Verified byte-identical for every shipped brain — which is why it was allowed
  to land without a retrain.
- **`len(declared) != arch_tail_width` RAISES** (`RoleSchemaError`), in both
  directions and also for a name this build no longer emits. Zero-padding a
  short declaration would be the same silent degradation wearing a new coat.
- `brain_evolution.evolve()` now **stamps** `meta.role_features` on every new
  brain, clamped to the brain's **actual** tail width (minus
  `TACTICS_CONTEXT_D` for v3). Proven end to end: a fresh
  `evolve_brains.py --position RW` writes `arch [33, 32, 32, 10]` with all
  nine WING names declared, and `_aligned_tail` round-trips all nine with no
  warning.
- **⚠️ 5 of the 11 shipped brains now carry a warn-once inert input**: CB
  (7 of 9), LB and RB (7 of 8 — FB grew with §3 below), LW and RW (8 of 9).
  That is the correct, honest state: the declaration mechanism reports exactly
  what the retrain will fix. **Do not silence those warnings to tidy the log.**
- The load-bearing test is
  `test_inserting_a_feature_in_the_middle_shifts_a_legacy_brain_but_not_a_declared_one`
  — a synthetic mid-schema insertion, asserting the legacy path returns
  `[10,20,30,40]` where the declared path returns `[10,30,40,50]`. Every growth
  so far has appended, so the by-name path's whole value is the case that has
  not happened yet.

### 3. The missing input — `pass_lane_open`, on FB and WING only

The plan said "add the two missing inputs: signed forward progress available
to the best forward option, and the rank of the best matrix option".
**Measurement narrowed the first and killed the second.**

- **The gap is narrower and more specific than the plan assumed.**
  `pass_lane_open` (openness of the best forward teammate) has always been in
  DM/CM/AM. CB has `same_side_lane_open`. GK has `pass_open`. **FB and WING
  were the only two families with no forward-option signal at all** — and they
  are the two roles that live on the touchline where that option is the whole
  point.
- **`overlap_option` LOOKS like it fills the gap and does not.** Measured, in
  both the FB and WING menus it computes
  `clamp(distance_to_nearest_forward_teammate / 30)` — a **DISTANCE** sensor,
  not an openness one. Worse, it returns `0.0` both when **nobody** is ahead
  and when a teammate is **standing on top of you**: two opposite facts, one
  value. It is misnamed, and the name is what stopped anyone looking.
  (`short_cut` is the same shape: `clamp(nearest_teammate_dist / 40)`, named
  as if it meant "a short reset exists". Recorded, not renamed — renaming a
  live sensor changes nothing functionally and would invalidate nothing, but
  it is a separate decision.)
- **APPENDED, not inserted.** The legacy loader takes "the first N", so
  appending leaves every shipped brain's first 7 (FB) / 8 (WING) numbers
  meaning exactly what they meant. Inserting anywhere else would silently
  relabel every feature after the insertion point — the precise failure the
  by-name schema exists to prevent.
- **One definition.** `_forward_options()` is a new shared helper, and
  `_mid_block` now calls it too, so `pass_lane_open` means the same thing in
  all five families. A test asserts `_mid_block` contains no second copy of
  the `max(...)` computation. A second implementation is how the same feature
  NAME comes to mean two things depending on position — the schema defect one
  level down.
- **ST is deliberately excluded**: a striker's best forward option is usually
  nobody, so the sensor would be a near-constant `0.0` — training budget spent
  on a feature with no variance.
- The **discriminating test** (the reason a new sensor was justified rather
  than a rename): two frames identical except the opposition marker moves 1 m
  → 20 m off our teammate. `overlap_option` **cannot move** (it measures our
  own teammate). `pass_lane_open` goes 0.0 → 1.0. A sensor that correlates
  with something already in the vector is not a new input.
- Widths, derived not typed: FB 31 → 32, WING 32 → 33; `V3_INPUT_D` is now
  `{43, 44, 45}`. **`percy_the_chosen.PERCY_INPUT_SIZE` is therefore 33**;
  his saved `brains/percy*.json` are 32 wide and now load through the legacy
  path with a warn-once until retrained.

**The plan's item 2 also called for a second sensor — "rank of the best matrix
option" — and it is deliberately NOT implemented.** It would couple perception
to `AttackingMatrix`, duplicate work the chain does immediately afterwards, and
amount to teaching the policy to imitate the matrix, which is the opposite of
the emergent play the item exists to enable. This is a reasoned
non-implementation and it is recorded so it is not re-proposed as an oversight.

### 4. Two stale snapshot tests that had been failing since 2026-10-05

`tests/test_tactics_context.py` asserted `sorted(set(V3_INPUT_D.values())) ==
[43, 44]` and listed `45` among the widths the schema must **reject**. The
2026-10-05 CB sensor addition made 45 legitimately valid, so both checks had
been failing for ~15 days without anyone noticing — the project's recurring
"a hard-coded snapshot pinned as an invariant" defect, this time in a test.

Both are now **derived**: the width set is checked against
`{V2_INPUT_D[f] + TACTICS_CONTEXT_D}`, and the bad-width set is computed as
neighbours of the valid set, so an off-by-one role block stays the failure
mode being tested and the list cannot go stale again. 86/86 pass.

### Gates and the negative control

- **`tests/test_threat_reward.py` (47, new file).** Verticality ordering
  (forward > square > backward, backward **negative**), longer gain worth more,
  the through-ball margin floor, goal > best shot, penalty == `GOAL_VALUE`,
  shot == absolute xg, on-target > off-target > 0, missed penalty negative,
  `SAVE` scores as a shot on target, the loss-charge geometry for all five loss
  types, neutral events exactly `0.0`, telescoping exact, mirror symmetry both
  directions, 3,000-sample bounds in `[−1, 1]`, `prior == 0.0`, the reward
  definition travelling through `save`/`load`/`report()`, **and a non-vacuity
  case that reproduces the RETIRED reward and shows it ranks the square ball
  above the through ball while the new one reverses it.** That last test is
  the file's reason to exist: without it, "the new reward is better" is an
  assertion.
- **`tests/test_brain_schema_alignment.py` (39, new file).** By-name vs legacy,
  the mid-schema insertion proof, warn-once per (family, width), every shipped
  brain scanned for silent truncation (11 found, 5 warn), the raises, the
  evolver's stamping, and AST guards that **both** padded branches route
  through `_aligned_tail` with no bare `role[:` slice left anywhere.
- **`tools/diag/_diag_neural_guard.py` — 11/11 CAUGHT**, files restored
  byte-identically (sha256 asserted), baseline green before and after. M1
  reverses the potential difference, M2 restores `prior = 0.5`, M3 narrows
  `EXECUTION_TYPES` to `{"PASS"}`, M4 removes the re-base, M5 defaults
  `collapse_penalties` True, M6 removes the missing-name raise, M7 removes the
  count check, M8 silences the legacy warning, **M9 makes FB's forward option
  the distance sensor it replaced**, **M10 makes `_mid_block` recompute the
  lane instead of sharing the helper**, **M11 declares the feature in the menu
  while the FB block never appends it**. Every mutant is required to still
  compile — a syntax error counted as a caught mutation proves nothing about
  which assertion pins the defect.
- **Combined gate: 213 passed** (threat_reward, brain_schema_alignment,
  momentum_model, football_brain, role_pace_calibration, measured_activity,
  distance_accounting, keeper_attribution, opta_analytics_runs_sprints,
  tactics_context) + `test_role_features.py` ALL PASS.
- `tests/test_momentum_model.py` gained `PURE_TRAINING_CONSUMERS =
  {"threat_reward.py"}` — **one asserted, documented exemption** from "no
  engine module may import `momentum_model`" — with a non-vacuity check that
  the guard really reached that file and a companion test that the training
  reward is a pure reader of the grid. Hiding the import behind `importlib`
  was rejected as dishonest; an exemption you can read is better than a guard
  you can dodge.
- The import cycle this creates (`match_engine → event_chain →
  brain_integration → threat_reward → momentum_model → match_engine`) was
  checked, not assumed: `match_engine` never statically imports the chain, so
  `momentum_model`'s `from match_engine import …` always finds a fully
  initialised module.

### What is NOT done

- **The surrogate has not been rebuilt and no brain has been re-evolved.**
  That needs ~6 real matches for collection and 11 × 110–155 s for evolution.
- **The reward's variance on real matches is unmeasured.** It must be
  inspected before evolution is spent: if the ordinary-pass signal is too
  weak in the fitted table, the GA has no gradient and the retrain is wasted.
- **Item 3 (policy authoritative over `forced_receiver`) is untouched.** Until
  it lands, none of this is visible on a pass map, because the receiver
  governing the pass endpoint is the matrix (1178) / phase (296) / wide-combo
  (225), not the policy (1).
- **The acceptance gate is pass maps and the eye, never fitness.**

## THE PAIRING DEFECT — THE SURROGATE WAS LEARNING FROM MIS-PAIRED DATA (2026-10-20)

**This is the most consequential finding of the reward work, and it was found
by the distribution gate, not by reasoning.** Everything above it is about the
OBJECTIVE; this is about the DATA the objective was fitted to, and it is the
difference between a retrain that produces football and one that reproduces
T2's collapse.

### THE SYMPTOM

The 6-match v4 collection (4,011 pairs) fitted cleanly and its per-intent
reward came out **backwards for the intents the new reward exists to favour**:

| intent | sample-weighted reward | samples |
|---|---|---|
| THROUGH_BALL | +0.0152 | 17 |
| RECYCLE | +0.0023 | 471 |
| PROGRESSIVE_PASS | +0.0012 | 457 |
| PROTECT_POSSESSION | +0.0008 | 384 |
| SAFE_PASS | +0.0001 | 1245 |
| DRIBBLE | −0.0004 | 406 |
| SWITCH | −0.0007 | 496 |
| CARRY | −0.0022 | 481 |
| **CROSS** | **−0.0041** | 35 |
| **SHOOT** | **−0.0079** | 19 |

**SHOOT was the WORST of all ten intents.** A GA trained against that table
would learn never to shoot — which is *exactly* T2's documented second
collapsed optimum ("ST never shoots, CM abandons through-balls"). And the
1-match smoke had said the opposite (SHOOT +0.024, the highest), so the
smaller sample had read as a confirmation. **It was noise: n=2.**

### THE DEFECT: THE COLLECTOR JOINED DECISIONS TO THE WRONG OUTCOMES

`surrogate_collect._correlate_outcomes` attached each decision record to "the
first execution event by that player at or after this record's **minute**",
consuming **one event per record**. Measured over 3 real matches, the
self-consistency of the resulting pairs (does the paired event actually
correspond to the intent?) was:

| intent | self-consistent pairings |
|---|---|
| **SHOOT** | **1/8 (12.5%)** |
| **CROSS** | **0/18 (0%)** |
| **THROUGH_BALL** | **1/10 (10%)** |
| DRIBBLE | 3/212 (1.4%) |
| SAFE_PASS | 408/630 (64.8%) |

A shot decision was charged for somebody else's carry. The mechanism: **a
decision can emit MORE THAN ONE execution event** — a failed pass emits PASS
*and* TURNOVER / DISPOSSESSED / MISCONTROL — and a one-event-per-record cursor
therefore mis-consumes the extra event against the NEXT decision of that
player, after which the alignment slides for the rest of the match.

### THE FIX: A CAUSAL JOIN, NOT A CLEVER ONE

`ChainDispatcher.possession` / `.attack` / `.set_piece` are the units of work:
**the chain that makes the on-ball decision also produces the event**, and it
emits the ACTION before its consequences. So a decision joins to the first
ACTION event of its OWN chain, by the same player, each event claimed at
most once. Two kinds of event are not claimable:

- **not the player's action at all** — a `BALL_RECEIPT` is the pass ARRIVING
  at the new carrier (emitted *by* the carrier, before the action he then
  takes; it was the FIRST event by that player in 219 of 630 SAFE_PASS
  pairings), and a PRESS is not an on-ball action;
- **the CONSEQUENCES of a failed action** — a TURNOVER / DISPOSSESSED /
  MISCONTROL / GROUND_DUEL. The failed action's OWN event already carries the
  loss charge in `threat_return` (a PASS with `outcome=False` is charged the
  opponent's threat where the ball died), so claiming the consequence as well
  **double-charges the same loss** against whichever decision came next.

**Measured on the SAME 3 matches, same decisions, same rewards:**

| intent | proximity join | causal join |
|---|---|---|
| SAFE_PASS | `BALL_RECEIPT:219, PASS:162, PRESS:78` | **`PASS:480, CARRY:146`** |
| SWITCH | `BALL_RECEIPT:93, …` | **`SWITCH_OF_PLAY:204`** |
| CARRY | `BALL_RECEIPT:122, …` | **`CARRY:183`** |
| DRIBBLE | `CARRY:58, BALL_RECEIPT:49, …` | **`CARRY:198`** |
| PROGRESSIVE_PASS | `BALL_RECEIPT:96, …` | **`PASS:119, PROGRESSIVE_PASS:69`** |
| **SHOOT** | mean **−0.0001** | mean **+0.0372** (highest of all intents) |
| drop rate | — | **0.0%** (every decision finds its own chain's action) |

The two exclusions were measured separately: excluding the consequences moved
26 of 630 SAFE_PASS pairings off a `GROUND_DUEL` and took SHOOT from +0.0201
to **+0.0372** — the double-charge was penalising exactly the intent the
objective was built to favour.

### WHY A SHOOT INTENT IS NOT A SHOT EVENT — AND WHY THAT IS CORRECT

`SHOOT` pairs with a carry into shooting position, not with a shot, and the
cross-tab confirms it (0/8 shot-ish events under *both* joins). **The shot is
the deterministic `AttackingMatrix` gate's decision — the policy never fires a
shot in this engine.** So the reward the policy is credited with is the threat
IT actually created, which is the honest number. It is also the largest of
all the intents, which is the opposite of what the broken join produced.

### MY OWN INSTRUMENT WAS WRONG FIRST, AND IT MATTERED

`_diag_reward_dist.py` averaged the **cell means** per intent, unweighted:

```python
per_intent.setdefault(intent, []).append(score)   # ← one entry per CELL
```

so a cell holding one sample counted exactly as much as a cell holding
sixty, and the "n" it printed was a count of CELLS, not of samples. On the
6-match table that made SHOOT read −0.0074 (n=14) — a statement about the
table's sparsity, not about shooting. **A column that reads cleanly is not a
column that is right.** The probe now reports the SAMPLE-weighted mean
(counts-weighted, with the position tables and the global `""` row kept apart,
because `fit` writes every sample into both) alongside the cell mean,
labelled.

### THE STATE OF THE v4 SURROGATE FILES

- `brains/surrogate_pos_v4_brokenpair.json` — the **broken** fit, kept as the
  reference for this defect. **REFERENCE-ONLY. NEVER TRAIN ON IT.**
- `brains/surrogate_pos_v4.json` — the refitted surrogate under the causal
  join.

### GATES

- **`tests/test_surrogate_pairing.py` (15, NEW file).** The existing suite was
  green *about* the reward and **silent about the join** — it tested
  `_correlate_outcomes` on synthetic timelines that happened to pair
  correctly, so it could not see the defect. The new file pins the join
  itself: the action comes from the decision's own chain, a receipt and a
  press are not actions, a consequence is not claimed (so the loss is not
  double-charged), no event is claimed twice, an unexecuted decision is
  NEUTRAL rather than charged with another player's event, and the attacking
  direction is respected. It also pins the **wiring**, which lives in a
  different file from the join: the chain entry points are installed AND
  restored, the wrapper is an observer that returns the chain result
  unchanged, and the chain-call buffer is cleared per match.
  **The load-bearing case is `TheOldJoinIsProvenWrongTests`** — it pins that
  the legacy proximity join STILL mis-pairs the failed-pass fixture, so the
  defect cannot be re-introduced silently, plus a **non-vacuity** test that
  the two joins disagree on that fixture.
- **`tools/diag/_diag_pairing_guard.py` — 8/8 CAUGHT**, file restored
  byte-identically (sha256 `b81f98e2ab7cdea8`), baseline green before and
  after: M1 consequences claimable, M2 the receipt claimable, M3 an event
  claimed twice, M4 the buffer not cleared per match, M5 the entry points not
  restored, M6 a fabricated 0.5 instead of NEUTRAL, M7 the attacking
  direction ignored, M8 another player's event claimed. Every mutant must
  still **compile** — a syntax error counted as a caught mutation proves
  nothing about which assertion pins the defect.
  **Its first run reported 2 MISSES, both anchor failures** (an indented
  anchor against a module-level line, and an anchor that matched twice). That
  is the `_diag_deference_guard.py` no-op-mutation lesson recurring in a new
  shape, and it is why the control refuses to report a mutation it did not
  actually apply.
- Probes: `_diag_intent_pairing.py` (intent × paired-event cross-tab),
  `_diag_pairing_causal.py` (all three joinings over the SAME matches),
  `_diag_reward_dist.py` (the distribution gate).

### THE GATE, READ THE WAY THE GA READS IT — AND A SECOND INSTRUMENT ERROR

The first reading of the refit was an AGGREGATE per intent, and it was the
wrong instrument twice over.

`FitnessSurrogate._lookup` is **per bucket**: the position's own cell for that
bucket, else the global row's cell for the SAME bucket, else the neutral
prior. **It never sees an average over buckets.** So "SHOOT's aggregate is
−0.0039" is not a number the GA can read, and the collapse question is not
"is this intent's average positive" — it is **"in the buckets where this
intent exists, is it competitive with the alternatives in that same bucket?"**

The per-bucket reading, on the same 6 matches, **the ST's own cells**:

| ST, bucket `P1FT1OH0S1C1G1` (final third, central, goal-close, **with space**) | broken join | causal join |
|---|---|---|
| CARRY | +0.0092 | +0.0092 |
| PROTECT_POSSESSION | −0.0081 | −0.0041 |
| **SHOOT** | **−0.0228** (3rd worst of 6) | **−0.0051** (3rd best of 6) |
| RECYCLE | −0.0101 | −0.0178 |
| SAFE_PASS | −0.0278 | −0.0278 |
| SWITCH | −0.0319 | −0.0051 |

And the **global fallback** — what the GA uses whenever a position has no cell
of its own for that intent, which for an intent with 14 cells is MOST of the
time:

| global row, same bucket | broken join | causal join |
|---|---|---|
| best intent in the bucket | THROUGH_BALL +0.0306 | **SHOOT +0.0243** |
| **SHOOT** | **−0.0104** (below everything) | **+0.0243 (the top)** |
| SHOOT in the NO-SPACE bucket | −0.0408 | −0.0270 (still negative — correctly) |

**THE FOOTBALL STATEMENT, PRECISELY:** the broken table said *"with room in
the box, a switch of play beats a shot"* — i.e. never shoot. The fixed table
says *"shoot when you have room"* (SHOOT is the top intent in that bucket)
and still says *"do not force it when you are surrounded"* (the no-space
bucket stays negative). **The mis-pairing could not express that
distinction** because it charged shots with other players' turnovers alike in
every situation; the causal join can, because it charges the shot with what
the shot actually did.

**THE SECOND INSTRUMENT ERROR, and it is the same shape as the first.** The
distribution probe reported a per-intent AGGREGATE, and the gate then judged
the fit on that number. The probe now also reports the **per-bucket sign
census** — how many buckets each intent is negative in, and its worst/best
bucket — because that is what the GA reads. Both numbers are printed, each
labelled, and the AGGREGATE is explicitly marked as not-what-the-GA-reads.
`_diag_position_table.py` prints a whole position's table for the
bucket-by-bucket comparison that decides a retrain.

**THE GATE PASSES.** SHOOT is competitive with the alternatives in the ST's
own shooting buckets and is the TOP intent in the global fallback for the
clear-chance-with-space bucket; CROSS is the top intent in the
clear-chance-without-space bucket (+0.0420); and no intent is negative in
every bucket. The retrain is safe to run.

**WHAT WAS NOT DONE, and why it does not matter:** a 6-match RAW per-intent
measurement (the decision→outcome cross-tab over the same matches) was
launched to confirm the fitted numbers against the un-shrunk samples, and the
server restart killed it. **It is not needed:** the fitted table IS what the
GA reads, so the fitted per-bucket values are the correct gate instrument.
The raw figures remain unmeasured on 6 matches (the 3-match raw figures are
in the table above).

### WHAT THIS DOES NOT CHANGE

The reward itself is unchanged and still correct — the defect was in the
DATA, not the objective. And the acceptance gate is unchanged: **pass maps and
the eye, never fitness.**

## THE CROSS-ROLE LEAK — THE RETRAIN BUILT A STRIKER WHO NEVER SHOT (2026-10-20)

**This is what the retrain actually produced, and it is why the run was
stopped before it was extended to eleven positions.** The objective was
fixed, the data was fixed, the gate passed — and the first evolved brain was
still wrong. The defect this time is in the SURROGATE'S FALLBACK, not in the
reward and not in the pairing.

### THE COLLAPSE, MEASURED

One position was evolved to test the chain end to end (ST, 80 gen × 32 pop ×
300 states, seed 123, `brains_smoke_st/ST.json` — **REFERENCE-ONLY, NEVER
SHIP IT**; it is the evidence of the collapse). Its intent mix over 2,000
synthetic states:

| intent | retrained ST | shipped T1 ST |
|---|---|---|
| **CROSS** | **43.3%** | 0.1% |
| SAFE_PASS | 29.4% | 35.0% |
| RECYCLE | 19.7% | 5.5% |
| PROGRESSIVE_PASS | 7.5% | 12.1% |
| **SHOOT** | **0%** | 0% |
| PROTECT_POSSESSION | 0% | 29.9% |
| CARRY / THROUGH_BALL / SWITCH / DRIBBLE | 0% | 6.8 / 6.2 / 4.2 / 0.1% |

**A striker who crosses 43% of the time, has four intents instead of nine,
and never shoots.** The shipped brain is broad (nine intents, max share
35%). The retrain made it *more* concentrated, not less — on an intent the
striker never performs.

### THE MECHANISM: THE SURROGATE LENT THE STRIKER THE WINGERS' CROSSING VALUE

`FitnessSurrogate._lookup` fell back from the position's cell to the **GLOBAL**
row's cell for the same bucket and intent. The global row pools EVERY
position, so a **role-specific** action's value leaked across roles. Read
straight out of the fitted table (`_diag_lookup_leak.py`):

| ST, bucket `P1FT1OH0S0C1G1` (final third, central, goal-close, no space) | value |
|---|---|
| **CROSS** — own cell: none; global: **+0.0420** (the WINGERS') | **lookup = +0.0420** |
| SHOOT — own cell: −0.0017; global: −0.0270 | lookup = −0.0017 |

The striker was offered **+0.0420 for crossing** against his own **−0.0017
for shooting**, and the GA took the +0.0420. In the other clear-chance
bucket the same thing: global CROSS −0.0003 was still above the ST's own
SHOOT −0.0051.

**The ST's own table contains NO CROSS cell in ANY bucket** — in 4,011 real
decisions the striker never chose to cross, so the only crossing value
available to him was somebody else's.

### THE FIX: ROLE-SCOPED POOLING, WITH A DATA-DERIVED VOCABULARY

The fallback chain is now **position cell → ROLE FAMILY cell → the global
row, and the global row may only lend an intent the ROLE FAMILY actually
performed.**

- The family map is the project's own (`role_features.ROLE_FAMILY`): ST/CF,
  LW/RW, LB/RB, CM, CAM, CDM, CB, GK — so a winger's crossing value informs
  the other winger and never a striker.
- The **role-appropriate intent vocabulary is derived from the DATA**, not
  invented: the ST family performed SHOOT but never CROSS, so the strikers
  can never borrow a crossing value. No hand-written "which intents may a
  striker have" table exists in the codebase and none was added.
- The vocabulary is fitted and saved with the table (`family_intents`), so
  it travels with the surrogate file the trainer and the harnesses read.

**THE FITTED VOCABULARIES (6 matches, 4,011 pairs, read straight out of the
refitted table):**

| family | intents the family performed |
|---|---|
| **ST** | CARRY, DRIBBLE, PROGRESSIVE_PASS, PROTECT_POSSESSION, RECYCLE, SAFE_PASS, **SHOOT**, SWITCH, THROUGH_BALL — **no CROSS** |
| **WING** | … **CROSS**, no SHOOT |
| **FB** | … **CROSS**, SHOOT, THROUGH_BALL |
| CB | … SHOOT (2 samples — noise, not a role truth) |
| CM | … **no THROUGH_BALL** (the recorded data gap) |
| AM | … no SHOOT |

**THE LEAK IS CLOSED ON THE REAL FILE.** `_diag_lookup_leak.py` on the
refitted `brains/surrogate_pos_v4.json`: the ST's CROSS lookup now returns
**the neutral prior (0.0)** in every bucket, where the broken surrogate
returned **+0.0420** (the wingers'), and the ST's own cells are still read
as his own.

**A REFIT IS NOW FREE.** The collector's `--raw-out` wrote the raw pairs
(`brains/surrogate_pos_v4.raw.json`, 4,011 pairs), and
`tools/diag/_diag_refit.py` rebuilds the table from them in seconds — so
every future change to the fit costs no match time. This refit was rebuilt
that way, which also exercises the new load path.

**AND ONE TRAP CAUGHT BEFORE IT SHIPPED:** the family lookup must strip the
digits from a NUMBERED position (`CM1`, `CB2` are the live roster's names).
`ROLE_FAMILY.get("CM1")` returns `None`, so a lookup that did not strip
would have been silently inert on every numbered position — the exact trap
AGENTS.md records from `_cb_same_side_corridor`, where a guard that looked
right matched nothing. The canonical `role_features._role_family` (which
strips) is used instead, and mutation M8 in the negative control re-breaks
it to prove the test sees it.

### THE GATES, AND TWO MISSES THAT WERE MY OWN FIXTURE'S FAULT

- **`tests/test_surrogate_role_pooling.py` (13, NEW file).** Pins the chain,
  the leak, the vocabulary, the save/load round trip, and the real v4
  surrogate (the ST cannot read a crossing value; the winger still can).
- **`tools/diag/_diag_role_pooling_guard.py` — 8/8 CAUGHT**, file restored
  byte-identically, baseline green before and after: M1 the global row lent
  unconditionally, M2 the vocabulary guard removed, M3 the family row
  ignored, M4 the family tables never fitted, M5 the vocabulary built from
  the global row, M6 the family table not saved, M7 not loaded.
- **THE FIRST CONTROL RUN REPORTED 2 MISSES (M3, M4) and the reason was my
  own test fixture, not the guard.** The fixture hand-built
  `family_table` and `family_intents` instead of going through `fit()` — so
  the two mutations that change `fit` itself could not be seen, and the
  family row and the global row happened to hold the SAME number so the
  test could not tell them apart. **A suite that builds the thing under test
  by hand is evidence about nothing.** The fixture now goes through the real
  `fit()` and the family/global rows are made to differ.
- **A NOTE ON MY OWN EDITING, because it cost real time:** every multi-line
  edit of one comment block failed for ten minutes. The tool was RIGHT and
  I was wrong: the file says "as if it **were** this intent's value" and I
  transcribed "was" into every attempt, so the anchor never matched. The
  byte-level diff (`_diag_linediff.py`) settled it in one run. **A refused
  edit is evidence about the edit, not about the file.**

### WHAT IS STILL OPEN, AND IT IS NOT A DEFECT

**The centre midfielder has NO through-ball samples in the corpus, and
therefore no through-ball value.** Neither the heuristic CM nor the shipped
T1 CM brain ever chooses THROUGH_BALL (measured: 0% over 2,000 states, and
0 in 4,011 real decisions). Under role-scoped pooling the CM cannot borrow
one, so the CM will not learn a through-ball.

This is the SAME gap AGENTS.md records as T2's second collapsed optimum
("CM abandons through-balls"), and it is a **DATA** gap, not a code gap: no
policy in this project ever plays it, so no data exists. Closing it needs
**ε-exploration collection** — a policy that occasionally picks an
unvisited intent so the corpus covers the role-appropriate vocabulary. That
is a scoped piece of work with its own gate, and it is deliberately NOT
started in this pass.

### THE RE-EVOLVE AFTER THE FIX — CONTROLLED PAIR, AND IT PASSES THE CHECK

The ST was re-evolved with **identical parameters** (80 gen × 32 pop × 300
states, seed 123) against the role-pooled surrogate, so the surrogate is the
ONLY difference. The check is the intent mix, never the fitness.

| intent | collapsed (broken surrogate) | **after the fix** | shipped T1 |
|---|---|---|---|
| **CROSS** | **43.3%** | **2.5%** | 0.1% |
| **SHOOT** | **0%** | **7.2%** | 0% |
| SAFE_PASS | 29.4% | 31.5% | 35.0% |
| DRIBBLE | 0% | 31.5% | 0.1% |
| SWITCH | 0% | 24.5% | 4.2% |
| THROUGH_BALL | 0% | 2.8% | 6.2% |
| intents used / max share | 4 / **43%** | 6 / **31.5%** | 9 / 35% |

**THE COLLAPSE IS GONE AND THE STRIKER SHOOTS FOR THE FIRST TIME.** No ST
brain had ever chosen SHOOT on synthetic states (not T1, not the collapsed
one); this one does, at 7.2%. And the brain is LESS concentrated than the
shipped T1 (max share 31.5% against 35%).

**BUT THE MIX IS BROAD, NOT DEMONSTRABLY BETTER FOOTBALL** — a striker who
dribbles 31% and switches 24% is not obviously better than T1's. The reason
is in the ST's own table: **most of his cells sit within ±0.0002 of zero**
(the shrinkage pulls a sparse cell to its bucket mean, and the bucket means
are near zero), so the mix is decided by differences in the fourth decimal.

**AND THE PRIOR STILL LEAKS A LITTLE.** The neutral prior (0.0) outranks a
slightly-negative REAL cell, so an intent with **no data at all** still
takes a share: CROSS is at 2.5% with no samples of its own, purely because
0.0 > −0.0051. **"Ignorance is attractive" is the residual defect in the
objective's use of the prior**, and it is a VALUE question, not a data one.

**ONE ROOT CAUSE FOR BOTH OPEN GAPS.** The ST's SHOOT cells are n=1 and the
CM has no THROUGH_BALL samples at all — because **the corpus does not cover
the intents that matter, because the policy that generated it never chooses
them.** The same ε-exploration collection that would give the CM its
through-ball would give the striker real shooting samples (n≫1 instead of
n=1) and would stop the mix being decided by fourth-decimal differences. It
is the single highest-value next measurement, and it needs CPU permission.

### ALSO SHIPPED: A RAW CORPUS, SO A REFIT NO LONGER COSTS A COLLECTION

A fitted table **cannot be refitted**: the shrinkage hides the raw sums, so
changing the fit used to mean re-running every match (12 minutes). The
collector now writes the raw pairs (`--raw-out`), exactly as
`brain_self_trainer` already keeps, and `load_raw_corpus` reads them back.
Every refit after this point is free.

### WHAT THIS DOES NOT CHANGE

The reward is unchanged and still correct. The pairing fix is unchanged. And
the acceptance gate is unchanged: **pass maps and the eye, never fitness** —
which is exactly why this collapse was caught by reading the brain's INTENT
MIX rather than by its fitness number (0.6487, unremarkable).

## THE THIRD ROUTE TO T2's COLLAPSE — THE POLICY WAS NEVER CREDITED WITH THE SHOT (2026-10-20)

**This is the deepest finding of the reward work, and it was only visible
because the intent mix was read instead of the fitness.** Everything above it
is about the join and the pooling; this is about the ENGINE'S OWN STRUCTURE.

### THE DEFECT: A SHOOT DECISION WAS WORTH EXACTLY 0.0, ALWAYS

A policy SHOOT intent makes the possession chain create a **zero-length
marker carry**:

```python
cls.make_event(minute, EventType.CARRY, attacking_team, last_player.name, …,
               location_x=x, location_y=y, end_x=x, end_y=y,   # ← no movement
               outcome=True, metadata={"shot_intent": True, …})
```

`threat_return` scores a completed ball-mover as `Φ(end) − Φ(start)`, and
`end == start`, so **every shoot decision in the corpus was worth precisely
0.0** — not "small", not "noisy": exactly zero, every time.

And the shot itself — worth its xG, which is the entire value of the
chance — is fired by the **ATTACK chain**, a different chain, and was
credited to nobody. The policy chose to shoot; the engine checked the
geometry (`policy_shot_score >= 0.20`); the shot happened; and the surrogate
saw a zero-length carry.

**So the surrogate could not express "shooting is worth it".** The striker's
SHOOT cell values in the fitted table were not shooting values at all: with
n=1 and the shrinkage toward the bucket mean, they were the BUCKET's mean
scaled by 3/4 — pure artefact. This is T2's "ST never shoots" collapse
reached through the DATA, the third distinct route to the same failure
(the other two being the mis-paired join and the cross-role leak).

### THE FIX: A SHOOT DECISION IS ANSWERED BY THE SHOT IT CAUSED

The causal join now looks forward in the TIMELINE for a SHOOT decision: the
first shot-ish event by the same player, at or after that chain's first
event, **in the same minute** (the possession chain and the attack chain run
inside one minute of the match). The minute bound is what keeps this from
becoming the proximity join again. Only ONE policy decision site exists in
the engine (`event_chain.py:2048`, the possession chain), so the shot cannot
be claimed twice.

**Why this makes the objective able to express football:** the engine's own
feasibility gate does the geometric work. A shoot decision only becomes a
shot when the geometry allows it, so the reward is earned exactly where the
chance was real:

- **shoot with room** → the shot fires → the policy is credited **+xG**;
- **shoot from halfway** → no shot is possible → the decision is worth
  **0.0** → the safe pass's own value wins.

That is the distinction the table could not previously make, and it is the
difference between "shooting is always worth it" and "shooting is worth it
where a shot is on".

### THE SECOND HALF: "IGNORANCE IS ATTRACTIVE"

With the shot credit fixed, the remaining residual was the PRIOR. The neutral
prior (0.0) outranks a slightly-negative REAL cell, so an intent with **no
data at all** took a share: the striker's crossing was at **2.5%** purely
because 0.0 > -0.0051, and the striker's 7.2% SHOOT was the same effect
rather than a football improvement.

**The fix is a role-specific FORBIDDEN value.** Two actions are genuinely
role-specific and the data proved it:

| action | roles that may perform it | why |
|---|---|---|
| **CROSS** | WING, FB | the ST family performed it **ZERO** times in 4,011 real decisions — every crossing value the striker could read was the wingers' |
| **SHOOT** | ST, AM, WING | the CB and GK tables carry stray SHOOT samples (the heuristic's noise), not a role truth |

Everything else — a safe pass, a recycle, a carry, a switch, a through ball
— is GENERIC and any role may perform it. **It is deliberately NOT a full
per-role table**: a table that forbade the keeper's progressive pass, for
instance, would teach the keeper to hoard the ball, and a wrong table is
worse than no table.

A forbidden action now returns `FORBIDDEN_VALUE = -1.0` — far below any
real reward (the threat return's range is about [-0.6, +1.0]) — and the
check comes FIRST, so it cannot be read from the position, the family or
the global row. **Ignorance must not outrank evidence.**

### THE THIRD PIECE: ε-EXPLORATION, SO THE CORPUS COVERS THE ACTIONS

The corpus is generated by a policy, and the policy never chooses the rare
actions: the CM family performed NO through-ball in 4,011 decisions, and the
striker performed exactly one shot. No amount of re-fitting fixes a hole
in the data.

So the collector can now **replace a decision with a random action the
player's ROLE can perform** (`--explore EPS`), and the engine really
executes it, so the reward is measured on what really happened. The
vocabulary is `role_features.ROLE_INTENTS` — the football truth about what
the role can do, not a list of what happened to happen. **The striker is
never asked to cross; the midfielder IS asked for a through ball.**

The exploration is sound rather than random because the engine gates each
action on its own geometry (a through ball needs a runner in behind; a shot
needs a shooting position), so the bucket structure separates the good
situations from the bad ones exactly as it does for the policy's own
choices.

### THE GATES

- **`tests/test_surrogate_pairing.py` (20)** — now includes
  `ShootDecisionIsAnsweredByItsShotTests`: the shoot decision takes the
  shot's value; the marker carry would have scored exactly 0.0 (the defect,
  stated as a measurement); a shot from another minute is NOT claimed (the
  bound that keeps it from being the proximity join); another player's shot
  is not claimed; a non-SHOOT decision is never answered by a later shot.
- **`tests/test_surrogate_role_pooling.py` (18)** — now includes
  `ExplorationCoverageTests` (off by default; the striker is never asked to
  cross; the midfielder CAN be asked for a through ball; an unknown role is
  left alone) and the forbidden-value rules.
- **`tools/diag/_diag_pairing_guard.py` — 9/9 CAUGHT** (M9 re-breaks the
  shot pairing), **`tools/diag/_diag_role_pooling_guard.py` — 9/9 CAUGHT**
  (M9 removes the forbidden rule). Both restored byte-identically, baseline
  green before and after.
- **THE CONTROLS CAUGHT THEIR OWN STALE ANCHORS.** After the join was
  rewritten, three mutations reported "ANCHOR MATCHED 0 TIMES" — the guard
  was pointing at the old indentation. **A mutation that silently does
  nothing is the failure this control exists to catch, and it caught it in
  the guard rather than in the code.**

## THE RETRAIN'S CENSUS — EVERY POSITION, AND THE FITNESS WAS THE JUDGE (2026-10-20)

**The 11-position retrain ran (80 gen × 32 pop × 300 states, seed 123, the
production config), and the intent-mix census of the result is the most
important measurement in this whole section.** Every fitness number was
unremarkable — 0.5958 (CM) to 0.6699 (RB) — and not one of them would have
told you anything.

| pos | intents | max share | top intent | SHOOT | CROSS | the failure |
|---|---|---|---|---|---|---|
| GK | 7 | 27.9% | THROUGH_BALL | **17.2%** | **14.1%** | role-inappropriate |
| CB | 6 | **44.2%** | **SHOOT** | **44.2%** | 0% | role-inappropriate |
| LB | **2** | **81.9%** | RECYCLE | 0% | 0% | collapsed |
| RB | 6 | 27.9% | SWITCH | 6.7% | 3.5% | — |
| CDM | **2** | **61.5%** | DRIBBLE | 0% | 0% | collapsed |
| CM | **2** | **67.3%** | SAFE_PASS | 0% | 0% | collapsed |
| CAM | 4 | 43.2% | CARRY | 0% | 0% | collapsed |
| LW | **2** | **61.5%** | CARRY | 0% | 0% | collapsed |
| RW | 5 | 31.4% | SWITCH | **18.1%** | 0% | shooting worth 0.0 |
| ST | 5 | 46.5% | SAFE_PASS | **20.4%** | 0% | shooting worth 0.0 |
| CF | **3** | **47.7%** | SWITCH | 10.8% | 0% | collapsed |

**A goalkeeper who crosses 14% of the time, a centre-back who shoots 44%,
a right winger who shoots 18%, and four positions collapsed to TWO intents.**
All of it hidden behind fitness numbers between 0.60 and 0.67.

### WHY THE FITNESS COULD NOT SEE IT: THE CONFIDENCE TERM OUTRANKED THE REWARD ~50x

The surrogate fitness was:

```
base    = 0.5 + 0.5 * clip(mean_reward, -1, 1)
fitness = base * (0.5 + 0.5 * mean_conf) - penalties / n
scaled  = (fitness - dom_pen) * (1.0 + 1.0 * bdiv)
```

The confidence term and the entropy term are MULTIPLICATIVE factors with a
range of about 2. The threat reward's within-position spread is about
**0.01** — a forward pass is worth ~+0.008, while the OLD reward (engine
success: 2.0 / 1.0 / 0.0) had a spread of ~1.0. **So replacing the
objective inverted the balance by roughly fifty times**, and the GA
faithfully optimised PEAKEDNESS instead of football. Measured on the retrained
brains against the shipped ones, on the same corpus:

| pos | new mean_conf | shipped mean_conf | new mean_reward |
|---|---|---|---|
| GK | **0.453** | 0.194 | −0.253 |
| CB | **0.625** | 0.247 | −0.361 |
| LB | **0.960** | 0.212 | +0.006 |
| CDM | **0.966** | 0.230 | +0.001 |
| CM | **0.889** | 0.227 | +0.002 |
| LW | **0.887** | 0.380 | −0.000 |

**A goalkeeper whose reward was -0.253 WON, because it was more confident.**
And a peaked network IS a collapsed intent mix — so the confidence term was
not a tie-breaker, it was the objective.

**THE FIX: both guards are demoted to small ADDITIVE terms** (0.05 each):
they can break a tie and they can veto a collapse, but they cannot decide.
The reward's range is 1.0; the guards' combined weight is 0.10.

### AND THE ROLE-INAPPROPRIATE ACTIONS ARE THE FORBIDDEN RULE

The GK's 14% crossing and the CB's 44% shooting are exactly what the
role-specific FORBIDDEN value stops: CROSS is a wide-role action and SHOOT
is a forward-role action, and a keeper or a centre-back asking for one is
told **−1.0**, which no real reward can reach.

### AND THE SHOOTING SHARE IS THE ZERO-LENGTH MARKER CARRY

The RW's 18% and the ST's 20% shooting are not the striker's eye for goal —
they are the "ignorance is attractive" residual, because a shoot decision
was worth exactly 0.0 and 0.0 outranks a slightly-negative real cell.
Crediting the shoot decision with the shot it caused fixes the sign AND the
scale: a shot is worth its xG, which is an order of magnitude above a
forward pass, so the objective regains a real dynamic range.

### WHAT THIS MEANS FOR THE ORDER OF WORK

**Every one of the three defects was found by reading the intent mix, and
none of them was visible in the fitness.** The census is now a gate, not a
diagnostic: a brain set that collapses, or that has a role performing an
action it never performs, is REJECTED regardless of its fitness. The
acceptance gate was always "pass maps and the eye, never fitness"; this
adds the sharpest cheap instrument available before a single match is run.

`_diag_intent_mix_all.py` is that instrument: it reads every position's mix
on a fixed corpus, against the shipped brain, and flags a collapsed mix, a
single action over 40%, or a role-specific action out of role.

## THE v5 CORPUS — THE HOLES CLOSED, MEASURED IN THE RAW PAIRS (2026-10-20)

The v5 collection (`--matches 6 --seed 123 --styles fluid_counter,attacking,
tiki_taka --policy heuristic --explore 0.25`) produced **3,696 pairs, 962 of
them explored (26%)**, with the shot-aware join. The coverage is the
measurement that matters, and it is read from the RAW corpus (before any
shrinkage):

| what was missing | v4 (4,011 pairs) | **v5 (3,696 pairs, 26% explored)** |
|---|---|---|
| **CM through-ball** | **0 samples** | **32 samples, mean +0.00278** |
| CDM through-ball | 0 | 7 samples, mean **+0.02674** (max +0.2175) |
| CB through-ball | 0 | 31 samples |
| LB / RB / ST through-ball | 0 | 8 / 6 / 4 |
| **ST shot** | **1 sample, and worth exactly 0.0** | **6 samples, mean +0.00673, max +0.0382** |
| LW shot | 0 | 4 samples, mean −0.09367 (a winger's shots mostly miss) |
| CAM shot | 0 | 2 samples, mean +0.01880 |
| LB / RW shot | 0 | 1 / 3 |

**THE SHOT PAIRING IS CONFIRMED BY THE DATA:** a shoot decision used to be
worth exactly 0.0 in every single case; now it carries the shot's real
value. And the per-bucket census — the thing the GA actually reads — has
**SHOOT positive in 7 of its 11 buckets** (best +0.0303), where the broken
table had it negative in every one.

**WHAT IS STILL THIN, AND IT IS STATED RATHER THAN HIDDEN:** shooting has 24
samples in the position tables (the ST's 6, the winger's 4, and one each of
noise from the CB / GK / CM). The exploration gives COVERAGE, not volume; a
corpus with more matches, or a higher ε, would thicken it. That is a
measurement to take, not a defect to fix in the code.

**AND THE NOISE THE EXPLORATION DID NOT CAUSE:** the stray shoot samples on
the CB (1), the GK (3) and the CM (2) are the HEURISTIC's own occasional
choices, not the exploration — the exploration draws from the role's
vocabulary, and a keeper's vocabulary contains no shot. Those three are
exactly what the role-specific FORBIDDEN value now handles.

## THE HEURISTIC IS NOT THE ENEMY — A CORRECTION FROM THE USER (2026-10-20)

**I framed the on-policy collection as if the hand-written heuristic were
something to be phased out. That was wrong, and the user corrected it: the
heuristic is a GOOD football model, and the exported realism stats (~95% on
the export metrics) are a hard-won number, not a joke.**

**What the retrain actually touches, stated plainly: ONE layer.** Which of the
ten intent labels the on-ball decision picks. Everything that produces the
realism — the pass destination logic (the matrix), the role behaviour, the
shape, the distances, the event structure — is the heuristic's and the
engine's, and **none of it moves**. The intent label then nudges that
football: it sets the delivery class (ground / switch / cross) and the pass
length (~14 m safe, ~19 m progressive, ~39 m switch). So the retrain is an
**increment on top of a good model**, not a replacement of it.

**The only thing that was arguably mis-calibrated is the PRICES, not the
football**: the loss charge (`LOSS_WEIGHT × the opponent's threat where you
lost it`) was measured against a heuristic opponent, so the price of a
turnover is calibrated to how dangerous the heuristic makes you. That is a
refinement to a table, not a defect in the model.

**PARKED 2026-10-20 by user decision (no more CPU tonight):** the
on-policy collection (`--policy neural` plus a `--brains-dir`, with the
ε-exploration still on for coverage). It would recalibrate the prices and the
state distribution to the football we actually play, and it is the one
measurement that would quantify how much the calibration is worth. **It is
NOT needed to finish the retrain, and it is NOT needed to justify the
retrain's value.** The order of work, when it is picked up:

1. the v5 census (free) — the gate on the intent mix;
2. the real-match gate (pass maps and the eye) — the only thing that decides
   whether a retrained set SHIPS;
3. then, and only then, the on-policy collection as a refinement.

**AND THE ORDER IS SET BY A REAL DEPENDENCY, NOT A PREFERENCE:** the shipped
T1 brains are **stale w.r.t. the schema** — the CB is 31 wide against a
33-wide schema, so it runs on 7 of its 9 role sensors (`same_side_lane_open`
and `far_side_gap` silently discarded). So an on-policy collection with *T1*
would train against a blind centre-back. **The v5 brains are the first ones
with the current schema**, so they are the right set to collect with — which
is why the on-policy collection comes AFTER the retrain, not before.

## THE v5 CENSUS FAILS — AND THE CAUSE IS THE REWARD'S SCALE, NOT THE FIT (2026-10-20)

**The v5 retrain (11 positions, production config, `brains/surrogate_pos_v5.json`)
FAILED the census, and it failed differently from v4.** The forbidden rule
worked exactly as intended:

| | v4 | **v5** |
|---|---|---|
| GK shooting | 17.2% | **0.0%** |
| GK crossing | 14.1% | **0.0%** |
| CB shooting | 44.2% | **0.0%** |
| positions collapsed to <=3 intents | 4 | **8** |

So the role-inappropriate actions are GONE (the forbidden rule), and the
mixes got **worse**. The census, in full:

| pos | intents | max share | top | SHOOT | CROSS |
|---|---|---|---|---|---|
| GK | 5 | 41.8% | THROUGH_BALL | 0.0% | 0.0% |
| CB | 4 | 59.5% | PROTECT_POSSESSION | 0.0% | 0.0% |
| LB | 3 | 43.4% | CARRY | 0.0% | 35.3% |
| RB | **2** | 58.6% | SWITCH | 0.0% | 0.0% |
| CDM | **2** | 62.2% | DRIBBLE | 0.0% | 0.0% |
| CM | **2** | 57.7% | THROUGH_BALL | 0.0% | 0.0% |
| CAM | **3** | 47.8% | PROGRESSIVE_PASS | 0.0% | 0.0% |
| LW | **2** | 66.5% | SWITCH | 0.0% | 0.0% |
| RW | **2** | 83.5% | THROUGH_BALL | 0.0% | 0.0% |
| ST | 6 | 67.3% | CARRY | 1.5% | 0.0% |
| CF | **2** | **96.2%** | SWITCH | 3.9% | 0.0% |

**THE v5 SET MUST NOT SHIP.** The shipped T1 set stays.

### THE SURROGATE IS NOT THE PROBLEM — MEASURED

The bucket winners are SPREAD, so the table is not telling the brain to pick
one action everywhere:

| pos | buckets | winner(s) across them |
|---|---|---|
| CB | 27 | SAFE_PASS 9, PROGRESSIVE_PASS 5, THROUGH_BALL 3, PROTECT 3 |
| CM | 25 | PROGRESSIVE_PASS 8, SAFE_PASS 8, PROTECT 5, CARRY 3 |
| ST | 17 | RECYCLE 6, PROTECT 4, SAFE_PASS 3, PROGRESSIVE_PASS 2 |

**The brain is not following a table that says "one action everywhere".**

### THE CAUSE: THE REWARD'S SPREAD VARIES 45x BY POSITION, AND THE GUARDS HAVE A FIXED WEIGHT

Measured on the same corpus, the v5 brain against the shipped one
(`_diag_fitness_parts.py`):

| pos | v5 mean_reward | shipped mean_reward | v5 mean_conf | shipped mean_conf | intents |
|---|---|---|---|---|---|
| GK | +0.00033 | -0.05573 | 0.785 | 0.194 | 5 |
| CB | +0.00130 | -0.09136 | 0.908 | 0.247 | 4 |
| CF | +0.00082 | -0.02798 | 0.964 | 0.225 | **2** |
| **CM** | **+0.00015** | **+0.00196** | 0.833 | 0.227 | **2** |
| **CDM** | **+0.00067** | **+0.00192** | 0.794 | 0.230 | **2** |
| **LB** | **-0.00822** | **-0.00064** | 0.878 | 0.212 | **3** |
| **RW** | +0.00136 | +0.00019 | 0.938 | 0.240 | **2** |
| ST | +0.00756 | +0.00086 | 0.779 | 0.215 | 6 |

**THE ARITHMETIC:** the guard's contribution to the fitness spread is
`0.05 x (0.9 - 0.3) ~ 0.03` for EVERY position. The reward's contribution is
the DIFFERENCE of the two mean_rewards, mapped through `0.5 + 0.5 x reward`:

| pos | reward's contribution | guard's contribution | who wins |
|---|---|---|---|
| GK | 0.028 | 0.030 | a tie |
| CB | 0.046 | 0.033 | the reward |
| CF | 0.014 | 0.037 | **the guard** |
| CM / CDM / LB | **0.001** | **0.032** | **the guard, by 30x** |

**So the guards still decide exactly where the reward is flat, and the
collapses are exactly there.** And `mean_conf` of 0.78-0.97 means the
softmax is SATURATED: the network cannot express a state-dependent choice at
all. **The confidence term is still the collapse generator** — demoting it
from 0.5 to 0.05 was not enough, because the reward's scale is not comparable
ACROSS POSITIONS.

### THE FIX (DESIGNED, NOT YET RUN): NORMALISE THE REWARD PER POSITION

A fixed-weight guard cannot be balanced against a reward whose spread varies
45x between positions. So the reward must be expressed **in units of its own
position's typical spread** — the surrogate stores a per-position scale at
fit time (the reward's spread over that position's cells) and the fitness
divides by it:

    base = 0.5 + 0.5 * clip(mean_reward / scale[position], -1, 1)

Then every position's reward has an O(1) spread, the guards' fixed 0.05 is a
genuine 5% tie-breaker everywhere, and the balance no longer depends on
which position is being trained. **A fixed global gain will NOT do this**: a
gain that makes the CM's 0.002 spread visible would saturate the CB's 0.09.

**THIS IS THE NEXT PIECE OF WORK, and it is NOT started.** It needs its own
gate: the same census, on the same corpus, against the same baseline.

## THE COVER, THE SCREEN, AND THE POINT — a football gap the user found by watching (2026-10-20)

**The user watched matches and reported a specific gap:** *when a CB has the
ball, the other CB does not come across to cover him; and where are the CDM
and the CM at that moment, and what is their plan — do they communicate,
point to tell the others what to do?* He also asked for the football to be
researched rather than guessed.

### WHAT THE FOOTBALL SAYS (SOURCED, NOT ASSUMED)

**The first / second / third defender framework** (Wikipedia, *Association
football tactics*, citing the standard coaching texts) is the canonical
answer, and it names the communication explicitly:

- **First defender** — the first respondent, the one on the ball. *"The first
  defender may also indicate to the rest of the team whether to **stay high
  or back off from pressure**."* And: *"a common tactic for a first defender
  is to channel the opposing player to an area where a second or third
  defender is located."*
- **Second defender** — *"provides immediate support for the first defender.
  If the first defender loses their marking, the second defender takes over as
  first defender, and ideally one of the third defenders takes over as second
  defender... **A typical distance between the second and first defender is
  about six metres.**"* And: *"A second defender also adjusts their
  positioning relative to where teammates of the defensive side are
  positioned in order to uphold defensive structure."*
- **Third defender** — *"provides deeper cover... they have a view of the
  entire defence... **the third defender is usually tasked with informing
  teammates** (including but not limited to first and second defenders) about
  potential threats."* (the sweeper/libero)
- **Line depth** — *"a good rule of thumb for distances between defensive and
  midfield lines is about **16 metres**."*
- **Zone defence** — *"defensive players mainly move in **synchronicity with
  teammates**"*, and the second/third defenders and the midfielders sit in
  **two lines** (the defender line and the midfielder line).

**THE COMMUNICATION IS MEASURED, NOT ANECDOTAL.** Lian, Gjesdal, Fletcher &
Jordet, *Frontiers in Psychology* 16:1699943 (2026) — 18,031 non-verbal
behaviours coded across 143 players and 33 matches of the 2022 World Cup:

- **Tactical non-verbal behaviour is the dominant kind**: 2.03/min against
  0.73/min emotional (p<0.001).
- **CENTRE BACKS COMMUNICATE THE MOST OF ANY ROLE: 3.05 tactical NVB/min**,
  significantly above wide attackers (mean difference 1.36, p=0.028) and
  strikers (1.26, p=0.033). Centre midfielders sit between.
- Their coding taxonomy is a specification:
  - **Directing teammates** — *"pointing to or waving one's hand to where a
    teammate should position themselves (offensively or defensively)"*;
  - Asking for the ball; Confirming message (*arms out, palm down — a "hold"
    signal*); Influencing the referee.
- The functional reading is the one that matters for an engine: from the
  ecological-psychology literature cited in that paper, **tactical NVB is
  "actions aimed at influencing the behaviour of other individuals by
  altering their field of affordances"**. A point does not move a body — it
  changes what a teammate can do.

### WHAT THE ENGINE DOES TODAY, AND WHERE THE GAP IS

Traced through the shape chain in `match_engine._offball_move_player`:

| rule | what it does | the gap |
|---|---|---|
| CK37 `backline_build_up_support` | the **ball-side CB** drops toward a goal-side socket for a short outlet | it assumes the ball is with someone ELSE; it says nothing about the CB who HAS it |
| CK38 `backline_spread_pressure` | under a live press the back line's y is pushed AWAY from the ball and the **far-side CB deepens to 24-30 m** | the far CB goes DEEPER AND WIDER — the opposite of coming across to cover |
| the default 10 Hz compaction | pulls everyone toward the ball | not a measured support distance |
| `rest_defence_clamp` | >=2 outfielders behind the ball | a CONSTRAINT, and the right precedent |
| — | — | **there is no cover rule, no screen rule, and no communication channel at all** |

**THE THREE MISSING PIECES, precisely:**
1. **No cover.** Nothing says the second CB comes ACROSS to the pressed CB's
   side, at the football's ~6 m. (The user's own observation — the CBs
   staying level in x within +/-3 m — is the same thing seen from the other
   side.)
2. **No screen.** Nothing says the CDM drops between/inside the pressed CB to
   deny the inside lane, and nothing holds the football's ~16 m CB-CDM line
   distance.
3. **No communication channel.** Every player's target is a function of the
   ball, the shape, the opponents and the rules — **never of a teammate's
   intent**. So "the CB points and the CDM comes across" cannot be
   expressed, however good the shape rules are.

### THE ANSWER TO "COULD WE TRAIN THIS?" — IT SPLITS IN TWO, AND ONLY HALF IS LEARNABLE

**THE SHAPE MUST NOT BE TRAINED.** The cover distance (6 m), the line depth
(16 m) and the level back four are **invariants with numbers from the
literature**, not preferences. The project's own precedent is
`rest_defence`: a CONSTRAINT applied LAST, after every preference has had its
say, so that a state which violates it is *unreachable*. The cover is the
same kind of thing:
- *"there must exist a team-mate within ~6 m of the pressed CB, on the ball
  side, screening the inside lane"*; and
- *"when the ball-side CB is the carrier, the CB pair must be within ~+/-3 m
  in x"* (the user's observation, and the zone defence's "synchronicity").

**THE CALL CAN BE TRAINED, and it is the genuinely new capability.** The
mechanism:
- a second output on the brain — a **CALL** (COME_ACROSS / DROP / HOLD /
  PUSH_UP / SWITCH), separate from the ten on-ball intents;
- the CB's role block already carries the right sensors for it:
  `space_behind`, `marking_coverage`, `runner_danger`, `own_line_height`;
- the call's reward is **the threat it denies** — the opponent's Phi change
  over the seconds that follow, which is measurable with the xT grid that
  already exists, signed the other way.

**AND THE HONEST CAVEAT:** the call's effect is DELAYED and MULTI-STEP (call
-> teammate moves -> space covered -> chance denied), which is exactly the
credit-assignment gap AGENTS.md records for the objective ("No credit
assignment across a possession"). A per-decision delayed reward is
measurable, but it is a different objective from the current one, and it
would need its own surrogate, its own gate, and — crucially — a way to
attribute the denied threat to the call rather than to the teammate who
would have moved anyway. **That attribution is the hard part, and it should
be measured before it is modelled.**

### AND THEN IT WAS BUILT: THE COVER AND THE SCREEN (2026-10-20)

**The two shape constraints were implemented, with the football's numbers.**
No training, no learned anything — two invariants, applied where the project
already applies its invariants.

**THE MEASUREMENT BEFORE (one real match, `_diag_cover.py`, the case the user
described — our CB has the ball and an opponent is pressing him):**

| | before | the football |
|---|---|---|
| the other CB within 6 m of the pressed CB | **15.4%** | ~always, at ~6 m |
| the other CB level in x (within 3 m) | **15.4%** | the user's ±3 m |
| the CB pair's x-gap | median **7.3 m** | a couple of metres |
| the cover's distance from the pressed CB | median **33.7 m** | ~6 m |
| the CB line to the CDM | median **10.2 m** | ~16 m |

**THE CAUSE WAS CK38.** "Spread the pitch under pressure" pushes every
back-line player's y AWAY from the ball and the far-side CB DEEPER
(24-30 m). That split is a BUILD-UP pattern — a carrier in the middle needs
two escape lanes — and it is simply wrong when the carrier is one of our own
centre-backs: then the ball-side CB is the football's first defender and his
partner is the second, who must be ~6 m away, inside the presser.

**WHAT WAS BUILT (all in the position layer, all with kill switches):**

1. **`PositionEngine.cover_target`** — the second defender's socket: the
   football's ~6 m (`COVER_DIST_M`), 2.5 m goal-side and 5.5 m across, so he
   is roughly LEVEL in x (the user's ±3 m) and inside the presser. It fires
   only when our CB is the carrier and an opponent is within 8 m of him.
2. **CK38's amend** — the split does not fire at all when a centre-back holds
   the ball. The build-up case is untouched (the split still fires there, and
   a test pins that).
3. **`PositionEngine.screen_target`** — the midfield line's floor: the CDM
   may not collapse onto the back line (the football's ~16 m, floored at 12 m
   so the build-up support is not outlawed).
4. **THE COVER SPRINTS.** The constraint set the right target and the cover
   still sat a median **22.6 m** from it — because the generic ladder is a
   SHAPE integrator (1.8 m/s jog) and never arrives. That is the failure the
   code already documents twice (the block recovery and the set-piece approach
   both raise the speed for exactly this reason), so the cover uses the same
   form: `min(0.75 × top_speed, 1.1 × distance)`.

**MEASURED AFTER — and the measurement was redone properly, because the
first one was on n=21-32 in a single match and its medians bounced.** Three
seeds (777, 31, 57), the SAME instrument, the feature off and on
(`_diag_cover.py --off` vs not), so it is a controlled pair:

**The case the user described — our CB has the ball and an opponent is
pressing him (n=61 before, 75 after):**

| | before | **after** | the football |
|---|---|---|---|
| the CB pair's x-gap | 19.8 m | **6.1 m** | the user's ±3 m |
| the CB pair's y-gap | 16.4 m | **11.1 m** | |
| the cover's distance from the pressed CB | 29.4 m | **17.3 m** | ~6 m |
| the cover within 6 m | 4.9% | **13.3%** | |
| the cover level in x (≤3 m) | 13.1% | **28.0%** | the user's ±3 m |
| the cover came ACROSS | 60.7% | **84.0%** | |
| the CDM to the carrier | 19.2 m | **16.4 m** | |
| the CB line → the CDM | 6.0 m | **8.3 m** | ~16 m |

**And the unpressed CB-carrier case:** the cover's distance 22.9 → **16.9 m**,
came across 81.8% → **91.4%**, level in x 41.6% → **45.7%**. (One metric went
the other way — within 6 m, 14.3% → 8.6% — because the constraint only fires
when there IS a press, and the CK38 amend changes where the pair stands.)

**THE GAP IS NARROWED, NOT CLOSED, AND THE REASON IS FOOTBALL.** The cover
is 17.3 m away, not 6 m. Two reasons, both real:
1. **a real back four IS ~15 m wide**, so the cover starts 15 m from his
   team-mate, and
2. **the press window is only a few seconds** — the moment the CB passes, the
   first defender is someone else and the constraint stops. The cover's
   measured lag from his own socket is **13.3 m**, so he is still travelling
   when the ball leaves.

So the 6 m is not reached, and the honest reading is that the constraint
buys the DIRECTION and the SYNCHRONITY (the zone defence's principle) rather
than the final metre. **The remaining metre needs the press to last longer,
or the pair to start closer — and the pair's width is a FORMATION choice, not
a rule.** That is a separate, deliberate decision and it is not taken here.

**ZERO REGRESSIONS, BISECTED NOT ASSUMED.** The shape suite
(`test_positional_play` + `test_small_game` + `test_touchline_wide`) reports
3 failures. `tools/diag/_diag_cover_bisect.py` ran the same tests with the
WHOLE feature switched off and got **the same three failures**, and
`_diag_cover_suite_off.py` ran the full suite in the same order with the
feature off and got the same three. So all three are independent of this work
— which is what AGENTS.md claimed, now shown rather than assumed. (The
touchline one does not reproduce in isolation; it needs the wider suite.)

**GATES:** `tests/test_cover_and_screen.py` (14, new) and
`tools/diag/_diag_cover_guard.py` (**6/6 CAUGHT**, restored byte-identically):
M1 the socket is 20 m instead of 6, M2 the cover does not come across, M3 the
cover is not level, M4 CK38's split fires again, M5 the CDM's line gap is not
held, M6 the cover jogs instead of sprinting.

**WHAT THIS IS AND IS NOT:** it is the SHAPE half of the user's observation —
the cover and the screen, with the football's numbers. It is NOT the
communication half: "the CB points and the CDM comes across" is still not
expressible, because the engine has no channel from one player's decision to
another player's target. **That remains the trainable piece, and it is
recorded above with its reward and its hard part.**

## Next Move

**⏭ NOW — THE DEFERENCE RUN IS DONE; IT IS RECORDED AND PARKED.**
`tools/diag/_diag_deference_ab.py --arm both --seed 777` has been run (2 matches,
~3 min) and is written up under *POLICY TARGET DEFERENCE → MEASURED*. Short
version: the rule is live and dosed (**58 applied receivers, 7.5% of passes**),
the pass map did **not** move toward verticality in either direction, and the
margin is **not the binding parameter** because most firings are the
`matrix_value <= 0` "silent" mode where the margin does not apply. The payload
now separates those modes (`matrix_silent` / `matrix_reason`); 19 tests; negative
control re-run green **5/5**.

**USER DECISION 2026-10-19: no further matches on this — record and stop.**
The follow-ups below are therefore OPEN AND UNSCHEDULED, not pending work. Do not
re-run them without asking; the live rule cost the CPU budget for this item and
the default stays OFF.

Open decisions, if this is ever picked up (each needs a match, so **ask first**):
1. **Read the silent-vs-comparable split** — one ON-arm run with `--dump` writes
   per-pass JSON; this needs a re-run only because the fields did not exist when
   the first match was played. From now on the split is recorded on every run, so
   a future ON arm answers it for free.
2. **Decide whether unconditional firing on `matrix_value <= 0` is wanted.** If
   the matrix's `no_clear_option` cascade is a deliberate safety default, an
   unbounded override there is the one place the rule is NOT bounded. Measure
   `matrix_reason` first; **do not change behaviour on the current evidence.**
3. **Do NOT promote the flag on this sample.** No direction change and no eye
   evidence. The default stays OFF.

**✅ DONE 2026-10-20 — THE CB PACE GAP. Closed, and the benchmark it was
measured against was wrong.** See *CB PACE — THE ROW WAS AN OUTLIER* below.
Short version: the "real CB ~10.5–11.0 km" this note carried was unsourced;
CIES n°68 puts centre-backs at **9.2 km** (the least of any position) and
midfielders at **10.6**. So the gap was ~0.8 km, not ~2.3, and the cause was a
single outlier row in `_JOG_SPEED` (`CB` = 1.5, inside the WALKING band, while
every other outfield role was 1.75–2.0). Raised to the full-back value 1.9.
Measured 3 seeds × 2 arms: **CB 8.54 → 9.05 km, all three seeds up, −0.15 from
the CIES figure.** The accounting invariants held in all six matches (max
published activity total == the clock, exactly), so `_diag_activity_guard2.py`'s
subject matter is provably untouched. Guard: `tests/test_role_pace_calibration.py`
(7) + `_diag_pace_guard.py` **3/3 caught**.

Residuals, measured and deliberately NOT chased (all PRE-EXISTING, none
CB-specific now — CB is no longer the outfield outlier):
- **LB ~−0.8 km** and **CDM ~−0.75 km** against their CIES anchors. The `off` arm
  shows LB 9.81 before this work too, so it is not collateral; it is the same
  uniform-gait question below.
- **The holding gait is where the rest of the shortfall lives, and it is
  GLOBAL.** `block_shuffle` is 40–60% of every role's stepped seconds at
  ~0.8 m/s, and it is **wave-bound, not jog-bound** — measured: raising CB jog
  1.5→1.9 moved `to_target` +174 m and `orbit` +158 m but left `block_shuffle`
  at 1,629 m, unchanged, because `_BLOCK_SHUFFLE_RATE × _BLOCK_SHUFFLE_R_M`
  (1.0 m/s) sets its pace and the jog-derived cap never binds. So that channel
  is untouched by any `_JOG_SPEED` edit. Bumping it globally was estimated and
  **rejected on measurement**: it pushes CAM/ST/LW over their CIES anchors while
  lifting CB. Same class as `COMPACTION` — a separate, deliberate fit.
- **`COMPACTION = 0.14` remains unfitted.** Untouched here; still queued below.

**⏭ NOW — THE PAIRING DEFECT WAS FOUND AND FIXED; THE REFIT IS RUNNING.**
See *THE PAIRING DEFECT (2026-10-20)* above. The summary:

| item | state | evidence |
|---|---|---|
| 1. reward replaced | **DONE** | `threat_reward.py` + 47 tests, 11/11 negative control |
| 2. missing inputs | **DONE, narrowed by measurement** | `pass_lane_open` on FB/WING only; the "matrix rank" half deliberately not built |
| 4. silent truncation | **DONE** | schema alignment by name, 39 tests, raises instead of padding |
| **1b. THE PAIRING (new)** | **DONE — and it was the real blocker** | the surrogate was fitted from MIS-PAIRED data and ranked SHOOT **worst of all ten intents**; the causal in-chain join takes SHOOT to the **highest**. 15 tests + **8/8** negative control |
| 5. documentation | **DONE** | this file |
| 3. receiver authority | **NOT STARTED** | own commit; invalidates every calibration number |

**THE DISTRIBUTION GATE EARNED ITS PLACE.** The 1-match smoke said "SHOOT
+0.024, the highest — the ordering has flipped as designed". The 6-match fit
said **SHOOT −0.0079, the WORST**. The smoke was n=2; the 6-match figure was
the defect. **Reading the fitted table before spending evolution generations
was the single decision that saved this retrain** — without it the GA would
have been pointed at a table that teaches "never shoot", and T2's second
collapsed optimum would have been reproduced with a better objective and
better inputs.

**REFIT DONE, AND THE GATE PASSES.** `brains/surrogate_pos_v4.json` — 6 matches,
4,011 pairs, same seed/styles as the broken fit, now on the causal join. Read
the way the GA reads it (per bucket, not per aggregate):

- **SHOOT is the TOP intent in the global fallback for the clear-chance
  bucket with space** (+0.0243; the broken table had it at −0.0104, BELOW
  everything, so the GA would have never shot);
- in the **ST's own** cells, SHOOT moved from −0.0228 (3rd worst of 6) to
  −0.0051 (3rd best of 6) in his shooting bucket — above the safe pass
  (−0.0278) by 5×;
- **shooting without space stays negative** (−0.0270) — "do not force it when
  you are surrounded" — which is the football distinction the broken table
  could not express at all;
- **CROSS is the top intent in the clear-chance-without-space bucket**
  (+0.0420) and is no longer the worst intent overall (was −0.0041);
- no intent is negative in every bucket.

The broken fit is preserved as `brains/surrogate_pos_v4_brokenpair.json` —
**REFERENCE-ONLY, NEVER TRAIN ON IT**.

**NEXT: re-evolve all eleven** (`evolve_all_parallel.py --surrogate
brains/surrogate_pos_v4.json --states 300 --generations 80 --population 32
--workers 4 --seed 123`, 11 × 110–155 s ≈ **25–30 min**), then gate
challenger-vs-incumbent on real matches. **This needs CPU permission** — say
the word and I will run it. The acceptance gate is pass maps and the eye,
never fitness.

**⏭ NOW — THE v5 RETRAIN IS RUNNING; THE CENSUS IS THE GATE.**
See *THE RETRAIN'S CENSUS* and *THE v5 CORPUS* above. Where the work stands:

| step | state | evidence |
|---|---|---|
| reward replaced | **DONE** | `threat_reward.py`, 47 tests |
| the PAIRING fixed | **DONE** | causal in-chain join; 20 tests + **9/9** control |
| the CROSS-ROLE LEAK fixed | **DONE** | role-scoped pooling + data-derived vocabulary |
| the SHOT CREDIT fixed | **DONE** | a shoot decision is answered by the shot it caused |
| the FORBIDDEN rule added | **DONE** | CROSS is a wide-role action, SHOOT a forward-role one |
| **the FITNESS BALANCE fixed** | **DONE** | the confidence term outranked the reward ~50×; both guards are now small additive terms |
| the CM through-ball gap | **DONE** | ε-exploration: 32 CM through-ball samples, mean +0.00278 |
| the v5 corpus | **DONE** | 3,696 pairs, 26% explored, shot-aware join |
| **the v5 retrain (11 positions)** | **RUNNING** | `brains_v5/` |
| **the census of v5** | **NEXT, and it is the GATE** | `_diag_intent_mix_all.py brains_v5` |
| a real-match gate | **AFTER the census** | pass maps and the eye, never fitness |
| receiver authority (item 3) | **NOT STARTED** | own commit; invalidates every calibration number |

**THE CENSUS IS NOW A GATE, NOT A DIAGNOSTIC.** The v4 census showed a keeper
crossing 14% of the time, a centre-back shooting 44%, and four positions
collapsed to two intents — every one of them behind a fitness number between
0.60 and 0.67. So a brain set is **REJECTED** if any position collapses, if
one action takes more than 40%, or if a role performs a role-specific action
out of role, **regardless of its fitness**.

**WHAT THIS WORK DID AND DID NOT CHANGE, HONESTLY:**
- It DID make the objective able to say *"shoot when there is room, don't
  shoot when you are surrounded"*, and it DID give the midfielder a
  through-ball to learn.
- It did NOT make a pass map move, and it is not going to: the receiver of an
  ordinary pass is still the matrix's (AttackingMatrix 1178 / TacticalPhase
  296 / wide-combo 225 / policy 1). The retrain changes the **verb** — the
  intent mix, the pass length, the through-ball and cross destinations.
  Item 3 is what changes the **noun**, and it is weeks away.
- **NOTHING IN `brains/` (the production set) HAS BEEN TOUCHED.** Every
  retrained brain lives in a scratch directory (`brains_v4/`, `brains_v5/`,
  `brains_smoke_st*/`), and `brains_smoke_st/` is marked REFERENCE-ONLY
  because it is the evidence of the collapse.

**⏭ NOW — THE COVER AND THE SCREEN ARE IN; THE v5 RETRAIN IS STILL BLOCKED.**
Two separate threads are open, and they do not block each other.

**THREAD 1 — THE SHAPE (the user's gap). DONE, gated, and the numbers are
honest:**

| item | state | evidence |
|---|---|---|
| the research | **DONE** | the first/second/third-defender frame, and Lian et al. (Frontiers in Psychology 2026): **centre backs make the most tactical signals of any role, 3.05/min** |
| the "before" measurement | **DONE** | the cover was within 6 m of the pressed CB **15.4%** of the time; the CB line → CDM was 10.2 m against the football's ~16 m |
| the cover constraint | **DONE** | `PositionEngine.cover_target` — the second defender's ~6 m socket, level in x, inside the presser |
| CK38's amend | **DONE** | the two-escape-lane split does not fire when a centre-back holds the ball (it is a build-up pattern) |
| the screen | **DONE** | `PositionEngine.screen_target` — the CDM may not collapse onto the back line |
| the cover SPRINTS | **DONE** | the shape ladder jogs at 1.8 m/s and never arrives (22.6 m from his own socket); the cover runs, like the block recovery and the set-piece approach do |
| the gates | **DONE** | 14 tests + **6/6** negative control, restored byte-identically |
| **a multi-seed measurement** | **OWED** | the pressed case is n=21-32 in one match, so its medians bounce; the two non-noisy numbers (the cover's lag 22.6 → 11.0 m, the unpressed cover distance 27.7 → 13.0 m) both moved decisively |

**THREAD 2 — THE NEURAL LAYER. The v5 retrain FAILED its census and the
next piece of work is the per-position reward normalisation** (the reward's
spread varies 45x between positions, so the fixed-weight guards decide where
the reward is flat, and the collapses are exactly there). See *THE v5 CENSUS
FAILS* above. The v5 set does NOT ship; the shipped T1 set stays.

**AND THE THIRD THREAD — THE CALL.** The communication half of the user's
observation is NOT built: "the CB points and the CDM comes across" is still
not expressible, because the engine has no channel from one player's decision
to another player's target. It is the trainable piece, and its reward and its
hard part (the delayed, multi-step attribution) are recorded under *THE COVER,
THE SCREEN, AND THE POINT*.

## THE FOV — WHY THE CALL IS NOT OPTIONAL, MEASURED (2026-10-20)

**The user named the feature they value most: "no player sees everything,
they have real like FOVs — man that was phenomenal to me."** It is, and it is
worth measuring, because a feature nobody can quantify is a feature nobody can
protect. `_diag_fov.py` wraps the live perception path of a real match and
records what each role actually perceives.

**THERE ARE TWO LAYERS, and which one is live depends on the runner:**

| layer | where | the rule |
|---|---|---|
| `perception.perceive` | the SCRATCH runner | a DNA-scaled radius between `range_min 8.0` and `range_max 45.0`, a **150-degree horizontal cone centred FORWARD** (an actor behind the player is UNSEEN), positional noise up to `noise_base 3.0`, and a per-role relevance top-k |
| `cognition.senses.VisionSystem` | the **PRODUCTION** runner (`auto_run_match.py:1242` registers a mind per player) | a fixed **25 m** radius, **360 degrees — NO cone**, the nearest **2 teammates + 2 opponents** only, goalkeepers excluded |

**⚠️ CORRECTION TO MY OWN FIRST NOTE (2026-10-20): I wrote that the
production layer was "the same rule, tighter". IT IS NOT THE SAME RULE.**
The scratch path limits by DIRECTION (a forward cone: what is behind you is
unseen); the production path limits by **ATTENTION** (no cone at all — you
see in all directions, but you only notice the nearest two of each). Same
conclusion, different mechanism, and the mechanism decides what the CALL can
be aimed at.

**AND THE DEEP DIVE FOUND THE SEAM THE CALL SHOULD PLUG INTO.**
`SquadChemistry.positional_understanding_mult(p1, p2)` — *"affects defensive
shape / off-ball coordination (e.g. a CB pairing's joint tackle_success /
press-trap coordination)"*, 0.92-1.10 — **is defined and has NO CALLER**
(the grep finds only the definition; the sibling `pass_chemistry_mult` IS
used, at `event_chain.py:4941`). So the project already has the
socio-affective PAIR channel, and the call's effect can be scaled by how
well two players understand each other: **the CB points, and whether the CDM
acts on it depends on their chemistry.** That is the same shape as the
research's finding that communication's effect depends on shared
understanding, and it needs no new concept.

**WHAT EACH LAYER KNOWS (the dive's output, in one place):**

| layer | knows | does NOT know |
|---|---|---|
| `perception.perceive` | the scene filtered by range, forward cone, noise, top-k | what is behind him; distant players; the true ball (that is `ball_vision`) |
| `cognition.senses.VisionSystem` | the nearest 2 teammates + 2 opponents within 25 m, 360 degrees | everyone else — **including the 5th teammate** |
| `ball_vision` | a STALE ball off-screen, its uncertainty growing with time | the ball's true position when he cannot see it |
| `cognition.mind.PlayerMind` | his senses + MEMORY (event->tag: foul_in_box, clean_tackle_win, lost_duel, big_chance_missed, big_chance_scored, conceded) + experience + temperament | anything he never perceived |
| `squad_chemistry` | the PAIR's chemistry (pass 0.90-1.14, positional understanding 0.92-1.10) | — |
| `brain_integration` | routes to the cognition brain when a mind is registered, and derives the role tail from the SAME perceived scene, so the shared 24-d and the role block always agree about what he can see | — |

**MEASURED, the scratch path (the DNA-scaled cone), one real match, 1,070
perception samples:**

| pos | decisions | perceived | of | share | **unseen because BEHIND him** |
|---|---|---|---|---|---|
| GK | 32 | 11.3 | 21 | 53.7% | 1.2 |
| CB | 109 | 7.4 | 21 | 35.3% | 7.6 |
| RW | 107 | 6.8 | 16 | 41.5% | 6.3 |
| LB | 93 | 5.8 | 21 | 27.5% | 9.8 |
| CDM | 107 | 5.7 | 21 | 27.2% | 9.6 |
| RB | 79 | 5.5 | 21 | 26.4% | 8.8 |
| CM | 214 | 5.2 | 21 | 24.9% | 11.2 |
| CAM | 110 | 4.9 | 21 | 23.2% | 10.9 |
| **ST** | 168 | **3.3** | 21 | **15.6%** | **14.7** |
| **LW** | 51 | **2.7** | 21 | **12.8%** | **13.2** |
| **OVERALL** | | **5.4 of 21** | | **26.5%** | **10.2** |

**THE PRODUCTION PATH (the cognition gate): a player perceives 3.9 actors**
— the VisionSystem's cap of 2 teammates + 2 opponents inside 25 m, hit on
almost every decision.

**THE STRIKER SEES 3.3 OF 21 PLAYERS, AND HAS 14.7 BEHIND HIM. THE
WINGER SEES 2.7 AND HAS 13.2 BEHIND HIM.** And the goalkeeper, facing the
whole pitch, sees 11.3. **The further up the pitch you are, the less you
see** — because the cone is centred FORWARD, and at the top your back is to
your own goal. That is not a limitation to be wished away; **it is the reason
the call exists.**

**AND THE LITERATURE AGREES ON WHO CALLS.** The CB is the second-most-visible
outfield role (35.3%) and makes the most tactical signals of anyone
(3.05/min). **The player with the most information to share is the one who
shares the most** — and the thing he shares is exactly what his team-mates
cannot see.

**WHAT THIS MEANS FOR THE CALL, AND FOR THE COVER WORK:**
- The call's inputs must be the CALLER's FOV-limited view, and its reward
  (the threat it denies) must be judged on what his call could actually
  achieve from what he can actually see.
- **AND A NOTE ON THE COVER RULE I BUILT: it is in the sanctioned class, but
  it should be said out loud.** The shape layer is the TEAM's model and the
  engine's truth — the rest-defence invariant is explicitly "a TEAM
  judgement... made once per tick", and the cover follows the same shape (the
  engine identifies the situation, the per-player effect is pure geometry).
  The FOV constrains the BRAIN's inputs, not the shape model's.
  **But the cover's TRIGGER ("an opponent within 8 m of our CB") is information
  the CB may not have** — the presser can be behind him, outside his cone —
  and a FOV-honest trigger ("the presser is inside the carrier's own view")
  is a small change with a real meaning: **a press the CB cannot see is a
  press the team does not react to**, which is a football truth in its own
  right.

## THE EXPORT'S TIME AXIS IS A RANDOM NUMBER — and the truth already exists (2026-10-20)

**The user asked: in the export, is the timeline in seconds or milliseconds
(StatsBomb style), and why does the first ten seconds not flow?** Both
answered, and the second one is a defect.

### THE UNITS: SECONDS. There is no millisecond field.

The export's timeline entries carry `minute` (int) and `second` (int) — no
`timestamp` in milliseconds, so the file is not StatsBomb-shaped on time.
StatsBomb's `timestamp` is **ms within the period**.

### WHY IT DOES NOT FLOW: `MatchEvent.second` IS A RANDOM PLACEHOLDER

`event_chain.py:466`:

```python
second=kwargs.pop("second", random.randint(0, 59)),
```

**Every event's second is a random draw from 0-59** unless the caller passes
one explicitly (only two call sites do). The engine's own docstring on
`MatchEngine.chronograph` says so: *"The timeline's `MatchEvent.second` is
only a bucket (mostly a placeholder random value)"*.

Measured, the exported second against the truth (`_diag_timeline_clock.py`):

- **median |difference| 16.6 s, max 59.9 s**;
- **only 3.2% of events (114 of 3,547) are within 1 s of the truth.**

So the timeline's *minute* is right (the chain's minute) and its *order* is
right (the events are appended in chain order) — only the within-minute axis
is noise. That is why it looks like it "jumps".

### THE TRUTH EXISTS, IT FLOWS PERFECTLY, AND IT IS NOT USED

`MatchEngine.chronograph()` rebuilds every event's real time from the single
global clock (`state.match_clock_s`) and the `_chain_clock_marks` (the window
each chain actually occupied), and it is carried on `MatchResult.chronology`
as `TimedEvent`s with a **fractional** `second`, a true `match_clock_s` and a
`duration`. Measured on a real match:

| | measured |
|---|---|
| chronology events | 3,547 (against the timeline's 3,555 — the join is by `source_event_index`, NOT by position) |
| span | 0.00 → 5,782.69 s (96.4 minutes) |
| **out-of-order steps** | **0** |
| delta between chains | 1.06 s |
| delta WITHIN a chain | **0.15 s** |
| as StatsBomb-style milliseconds | 0 → 5,782,690 |

**That is a properly flowing football timeline: 0.15 s between the events of
one possession, ~1 s between possessions, nothing out of order.**

### AND THE EXPORTER IS ALREADY WRITTEN TO WANT IT

`plofa_export.convert_event` reads:

```python
clock_s = getattr(e, "match_clock_s", None)
if clock_s is None:
    clock_s = (minute * 60) + second      # ← the random placeholder
```

**`MatchEvent` has no `match_clock_s` field**, so that lookup always fails
and the exporter falls back to the placeholder. And the JSON timeline
(`exporter.py::_timeline_json`, which is the file the user reads) does not even
try — it writes `"second": e.second` directly, with no `clock`, no `clock_s`
and no millisecond timestamp.

### ✅ THE FIX IS SHIPPED (2026-10-21) — and the probe found THREE things the plan did not

The plan offered two routes ("stamp the field" or "have the exporters join
the chronology"). **The stamp won**, and it won for a reason the plan did not
state: the shot map is a THIRD consumer that no join could have reached,
because it sorts and positions shots by `minute * 60 + second` — i.e. it was
placing every shot at a random position inside its minute.

| where | what |
|---|---|
| `MatchEvent` | a new `match_clock_s: Optional[float]` field — the TRUE seconds from kickoff. The placeholder `second` is untouched, so nothing that reads it changes behaviour; the truth is ADDED, never substituted. |
| `MatchEngine.chronograph()` | writes the truth back onto every chain event, spread across the window the chain actually occupied. |
| `MatchEngine._emit_event()` | stamps the engine's direct events — no chain mark covers them. |
| the KICKOFF event | stamps its own clock (it is constructed directly in `match_engine.py`, NOT through `_emit_event`). |
| the goal celebration | stamps its own clock. |
| `exporter.py::_timeline_json` | reports the true `second`, `clock`, `clock_s` and a **StatsBomb-style millisecond `timestamp`**. |
| `exporter.py`'s shot map | sorts and places shots by the TRUE clock. |
| `plofa_export.convert_event` | already preferred `match_clock_s`; the field now exists, so its `clock` is real, its `second` follows the clock, and it gains a ms `timestamp`. |

**THE PROBE FOUND THREE THINGS THE PLAN DID NOT, and each would have shipped
a second truth.** `_diag_export_timeline.py` runs the match ONCE and dumps the
whole clock sequence, so the answer could be READ rather than guessed at.

1. **THE KICKOFF HAD NO CLOCK AT ALL — AND THE TIMELINE WENT BACKWARDS
   BECAUSE OF IT.** Three KICKOFF events are constructed directly in
   `match_engine.py`, so neither the chronology nor `_emit_event` stamped
   them. The exporter's fallback for a missing clock is `minute * 60 +
   second` — which for the kickoff is **60 s** — and the next event's real
   clock is **0.0 s**. So the exported timeline opened with a 60-second jump
   BACKWARDS. The user's exact symptom, and it was not the random second: it
   was a MISSING FIELD. Fixed by stamping the kickoff.

2. **THE STAMP AND THE CHRONOLOGY DISAGREED ON 2,648 OF 3,052 EVENTS — ON
   ROUNDING ALONE, AND AT A MINUTE BOUNDARY IT MOVED THE SECOND FROM 0.0 TO
   59.999.** The first version stamped `round(at, 3)` while the chronology
   carried `round(at, 2)`, so the two were different numbers for the same
   event. On an event sitting exactly on a minute boundary the
   floating-point hair decided which minute it belonged to: stamped 3359.999
   → second 59.999; chronology 3360.0 → second 0.0. **A float decided the
   minute.** Now `at` is rounded ONCE and the same value goes to both the
   event and the `TimedEvent`, and both exporters use the chronology's own
   convention — including its `min(..., 59.9)` clamp — so the two can never
   disagree about which minute an event is in.

3. **THE TIMELINE'S ORDER WAS NEVER THE PROBLEM.** Measured on the live
   match: **0 backwards steps** in the timeline's own order and 0 in the
   chronology's. The engine appends events in clock order; the "jump" the
   user saw was entirely the within-minute axis. (A timeline that does not
   flow is not evidence of an unordered timeline — the thing measured has to
   be the thing asked for.)

**GATE: `tests/test_export_timeline_clock.py` (6 tests, NEW file).** It
drives a REAL match, exports it, and asserts the properties the user asked
for: the timeline carries a millisecond timestamp, the first event is at
kickoff, **the timeline FLOWS (no backwards steps)**, every exported second
equals the engine's own chronology second for the same event, and the
provider-style export's `clock_s`/`timestamp` agree. Plus a **structural**
guard on the shot map (a PNG, so nothing behavioural can see its axis) with a
**non-vacuity** case: the guard must FAIL on the pre-fix shape, or it is not
checking anything. (Its first version looked for an `ast.Attribute` and went
red on the FIXED code, because `getattr(e, "match_clock_s", None)` is a
string literal — the mirror of "a name inside a comment is not a call
site".)

**POST-FIX MEASURED** (one real match, `_diag_export_timeline.py`):

| | pre-fix | post-fix |
|---|---|---|
| events with NO clock | 3 (the kickoffs) | **0** |
| stamp vs chronology disagreements | **2,648 of 3,052** | **0** |
| backwards steps in the timeline's order | 0 | **0** |
| backwards steps in the chronology | 0 | **0** |
| the first event's clock | none (fallback 60 s) | **0.0 s** |

**THE `sequences` BLOCK NEEDED NOTHING.** It is an aggregate (build-up
attacks, direct attacks, counts) with no per-event time axis — checked, not
assumed.

### ⛔ THE TWO PROBE RUNS WERE DIFFERENT MATCHES — and the cause is the CALENDAR, not the code

The pre-fix probe reported **3,056 events** and the post-fix probe **2,885**,
from the same seed and the same fixture. The export fix consumes no RNG, so
that difference needed an explanation before any number above could be
quoted.

**It is `date.today()`.** `roster_loader.PlayerRecord.age` computes each
player's age against today, and `player_dna` multiplies every drawn
attribute by the player's AGE CURVE — so a player whose birthday falls
between two runs is a day older, their attributes move, and the match
diverges completely (chaos: one changed probability is enough). **The
session crossed 2026-10-20 → 2026-10-21 mid-run**, and the workbook has
**three** players born on that boundary — checked, not assumed
(`_diag_roster_age_shift.py`):

| club | player | dob | |
|---|---|---|---|
| Oxton | **Lewis Driutt** | 1999-10-21 | **in the fixture's HOME XI** |
| Telbey | Joey Farńez | 1995-10-21 | |
| Telbey | Nal Mahn | 2002-10-20 | |

**SO: a real-roster match is reproducible for a FIXED roster, and NOT across
calendar days.** AGENTS.md's crc32 determinism claim is about the *match*
given its inputs; the roster's inputs drift with the date. This is the same
family as the open "production roster path does not replay across processes"
item in *REPRODUCIBILITY* — and it is a **new, separate** one: that one is
allocation-dependent, this one is the calendar. **Do not read "same seed,
different match" as a determinism failure until the roster's date
dependency is ruled out.**

**WHAT THIS DOES NOT AFFECT:** the fix's properties are STRUCTURAL, not
sample-dependent — the clock is monotonic because the engine's clock is, the
stamp and the chronology are the same number because one value is written to
both, and every event has a clock because the kickoff and the direct events
are stamped too. The gate asserts them on whatever match the fixture
produces.

## THE PLAN — WHAT TO DO NEXT, IN ORDER (2026-10-20)

**A PROCESS NOTE FIRST, because the user was right about it.** I edited the
shape chain (the cover and the screen) having read the shape chain and
CK37/CK38 but **NOT the perception layer**, and I wrote a rule whose trigger
depends on information the FOV says the CB does not have. It was caught only
because the user mentioned the FOV. **The dive must come before the edit**,
and the dive is now done and written up above (the table of what each layer
knows). Everything below is designed against that table.

**THREAD A — THE NEURAL LAYER (it blocks the retrain; it is not a small
thing).**
1. **Express the reward in units of its own position's spread.** The
   surrogate stores a per-position scale at fit time and the fitness divides
   by it, so every position's reward has an O(1) spread and the guards'
   fixed 0.05 is a genuine 5% tie-breaker everywhere. *Why:* the reward's
   spread varies **45x** between positions (0.09 for the CB, 0.002 for the
   CM), so the fixed-weight guards decide exactly where the reward is flat,
   and the collapses are exactly there. *Cost:* a small fit change, a free
   refit from the raw corpus, one evolve per position (25-30 min), the
   census.
2. **Re-run the census** — it is the gate: a set that collapses, or that has
   a keeper crossing, is rejected regardless of its fitness.
3. **Only then the real-match gate** (pass maps and the eye).

**THREAD B — THE CALL (the user's gap; now designed against the dive).**
1. **A second brain output: a CALL** (COME_ACROSS / DROP / HOLD / PUSH_UP /
   SWITCH), on the CB first — the role the research says calls most, and the
   one whose sensors already carry what a call needs (`space_behind`,
   `marking_coverage`, `runner_danger`, `own_line_height`).
2. **The call's inputs are the CALLER's FOV-limited view** — the same
   perceived scene the shared 24-d and the role block already read, so the
   call cannot know anything the caller's own brain cannot see. **In
   production the caller only sees the nearest 2 teammates, so a call CANNOT
   be aimed at a specific teammate he does not notice** — the call must be
   generic ("drop") and the engine resolves the recipient, or the caller's
   attention must be widened to include the intended recipient. **That
   decision is a design choice and it is recorded here rather than decided
   silently.**
3. **The effect is scaled by the EXISTING, currently-uncalled seam**:
   `SquadChemistry.positional_understanding_mult(caller, callee)` (0.92-1.10).
   **The CB points, and whether the CDM acts on it depends on how well the two
   understand each other.** No new concept needed.
4. **The reward is the threat it denies** — the opponent's Phi change over
   the seconds that follow, signed the other way. **THE HARD PART, to be
   measured before it is modelled:** the effect is delayed and multi-step
   (call -> teammate moves -> space covered -> chance denied), and the
   denied threat must be attributed to the CALL rather than to the teammate
   who would have moved anyway. **Measure the attribution first:** run a
   matched pair of possessions, one where the callee was called and one where
   he was not, and see whether the threat differs. If it does not, there is
   no signal to learn and the call must be a rule, not a trained output.

**THREAD C — THE COVER TRIGGER (one line, and a real choice).** The shape
layer is the TEAM's model and the engine's truth (the rest-defence invariant
is explicitly "a TEAM judgement... made once per tick"), so an omniscient
trigger is defensible for a shape invariant. **But the trigger can be made
FOV-honest** — "the presser is inside the carrier's own view" — which means a
press the CB cannot see goes unanswered. That is football, and it is the
user's call which one the engine should play.

**WHAT IS ALREADY DONE AND GATED (so it is not lost):** the reward, the
pairing, the cross-role leak, the forbidden rule, the ε-exploration, the
fitness balance, the cover, the screen, the CK38 amend, the cover's sprint —
with 220+ tests green, three negative controls at 9/9, 9/9 and 12/12, and
the shape suite's three failures bisected as pre-existing.

## ⏭ TOMORROW (2026-10-21) — THE PLAN IS WRITTEN DOWN: `PLAN_2026-10-21.md`

**THE DETAILED PLAN IS IN `PLAN_2026-10-21.md`** — every open thread, in
order, with the reason, the files, the exact command, the gate and the CPU
cost, plus the decisions that are the user's and the traps this project has
actually fallen into. Read that file first; it supersedes this section.

**THE DAY'S WORK, IN ONE LIST:**

1. **The reward's per-position scale** (the neural layer's blocker): the
   surrogate stores a per-position scale at fit time and `expected_success`
   divides by it, so the reward has an O(1) spread for every position and the
   guards' fixed 0.05 is a real tie-breaker everywhere. **Why:** the reward's
   spread varies 45x between positions (0.09 CB, 0.002 CM), so the guards
   decide exactly where the reward is flat, and the collapses are exactly
   there. Small fit change + a free refit from the raw corpus.
2. **Re-evolve all eleven** (25-30 min, **needs CPU permission**) against the
   refitted surrogate.
3. **The census** — the gate: <= 3 intents, any action over 40%, or a
   role-specific action out of role = REJECTED, regardless of fitness.
4. **The real-match gate** — pass maps and the eye, never fitness.
5. **The call** (the user's gap), in increasing order of ambition: (a) the
   cover's trigger, FOV-honest or omniscient — **the user's call**; (b) the
   cover's effect scaled by the existing, uncalled
   `SquadChemistry.positional_understanding_mult` seam; (c) the call as a
   LEARNED output — **not until the attribution is measured**.

**AND THE FOUR DECISIONS THAT ARE THE USER'S:** the cover trigger; the
call's architecture (extend the output layer, or a parallel head); the
call's targeting (a generic call, or a widened attention); and the CPU for
the evolve and the on-policy collection.

**A PROCESS NOTE, because the user was right about it:** I edited the shape
chain having read the shape chain but **NOT the perception layer**, and wrote
a rule whose trigger the FOV forbids. It was caught only because the user
mentioned the FOV. **The dive comes before the edit** — and the dive's output
(the table of what each layer knows) is recorded above, together with the
correction that the two perception layers are NOT the same rule (the scratch
path limits by DIRECTION, production by ATTENTION).

**A retrain alone will NOT make a pass map move.** The receiver governing the
pass endpoint is AttackingMatrix 1178 / TacticalPhase 296 / wide-combo 225 /
heuristic 13 / policy **1**. Item 3 is what makes any of this visible, and it
is weeks, not hours.

**NEW 2026-10-15 — from the SHAPE ANCHOR work, three items in priority order
before anything spatial is tuned further:**
1. **Calibrate `COMPACTION` (currently the literal `0.14` in
   `match_engine._offball_move_player`).** It was never fitted to anything.
   Frost now sits at 41.1 m median pass origin against a real 53.2 m, so the
   fullbacks are slightly on the DEEP side. **Needs a multi-seed sweep (~10
   matches) against the Opta benchmark, and it will not touch any of the three
   defects already fixed.** See *SHAPE ANCHOR* below for the full method.
2. **Style's influence on the ATTACKING lines is still only indirect** — it
   arrives via `pattern_for`'s pools, and `formation_stance_for` still does not
   read the profile (by design now, documented). If a style ever needs to move
   a line directly, `style_defense_deltas` is the seam.
3. **Re-verify the WIDTH band.** The width findings above predate the anchor
   fix; the anchor changed *depth*, and CK35's home-`y` pull is now the only
   absolute lateral reference in the chain. `_diag_width_gate.py` predates this
   work — **do not trust its numbers without re-running it.**

**ACTIVE — THE FABRICATION AUDIT IS NOT FINISHED.** The assist/shooter work closed
one path, not the class. Do not report "fabrication eliminated". Remaining, in
the order I would take them:

1. ~~**`_pick_creator` is STILL LIVE**~~ — **DONE 2026-10-03.** Deleted; the
   creator is now the real passer (`assister_name` → `_resolve_assister`), and
   a drawn creator had been fabricating the key pass's ORIGIN geometry too.
2. ~~**The cross-receiver clamp at `event_chain.py:3404`**~~ — **DONE
   2026-10-03.** See *CROSS RECEIVER: TWO FABRICATIONS IN THREE LINES* below.
3. ~~**`pass_direction` metadata is uncorrelated with geometry**~~ —
   **RETRACTED 2026-10-02, DO NOT TOUCH IT.** Measured 98.4% correct; the
   original finding was a sign error in the probe (it never flipped for the
   away side). See the retraction block in MEASUREMENTS.
4. **Shot-distance distribution** — median 20.0 m (real ~17), 14.8% inside 4 m
   (real ~0%), 22.2% beyond 25 m (real ~3%). `_shot_location`'s open-play draw
   `min(32, -14·ln(1-u) + 3)` is a log-uniform tail that over-feeds long range.
   This is calibration, NOT a fabrication — do not conflate it with 1-3.
5. ~~**Seed reproducibility** remains the root blocker for every A/B in this
   project~~ — **RESOLVED 2026-10-04, and the premise was wrong in two
   separate ways.** See *REPRODUCIBILITY* below. Short version: matches now
   replay byte-identically ACROSS processes with a varying `PYTHONHASHSEED`
   (verified 0/1/12345 → one digest `79857b22c6e2`), and the "two
   `simulate()` calls in one process are not comparable" rule is **FALSE** —
   a negative control proved it. `_STREAM_PARITY_DRAW` is no longer
   load-bearing *for reproducibility*, but deleting it is still a deliberate
   decision, not a cleanup: it holds the historical stream and every
   calibration number in this file was taken on the stream it preserves.
6. ~~**is the on-ball snap a correction or a fabrication?**~~ — **ANSWERED
   2026-10-04. It is overwhelmingly a CORRECTION, plus one confirmed
   fabrication class that is now closed.** See *THE ON-BALL SNAP* below for the
   method, the numbers and the fix. Short version: the coordinate the snap
   writes is the ball's own tracked position (PRESS 159/159 exactly at the
   ball; PASS passer 64/64), so the snap records a fact rather than inventing
   one; the receiver snap writes the pass's AIM point, which is a modelling
   assumption ("passes are executed as aimed") and not an invention; and the
   genuinely fabricated coordinate was `MatchEvent.location_x/y` DEFAULTING to
   (50.0, 34.0), the centre spot, at eight construction sites — now fixed and
   guarded by AST. **Remaining from the question:** whether the *receiver* snap
   should write the aim or a recorded arrival. `end_x` is only ever overwritten
   for corners and free kicks (`event_chain.py` CORNER_TAKEN / FREEKICK_CROSS),
   so an ordinary PASS keeps the aimed `end_px` forever. That is the one
   modelling assumption here, it is documented rather than fixed, and it is the
   same open `end_px`-vs-`end_x` question already on file under the PLOFA
   export frame note.
7. **`HIT_WOODWORK` IS NOT A SHOT, AND THAT IS A CALIBRATION QUESTION, NOT A
   BUG.** Found by the full regression: the only failure was a fixture that
   asserted `any(e.is_shot)` and drew `['HIT_WOODWORK']`. `MatchEvent.is_shot`
   is a **property** over a fixed list (`match_engine.py:616`) — SHOT_ON_TARGET,
   SHOT_OFF_TARGET, SHOT_BLOCKED, GOAL, PENALTY_SCORED, PENALTY_MISSED — and
   woodwork is absent, so a ball on target that hits the frame is not a shot.
   **Measured as internally CONSISTENT, not as an oversight**: the exporter's
   "Total Shots" is `shots_on_target + shots_off_target + shots_blocked`
   (`exporter.py:1758`, `:3058`, `:4183`) and it tracks woodwork as its own
   `hit_woodwork` column (`exporter.py:954`), so flag and export agree. The
   disagreement is with real football, where a woodwork strike IS a shot on
   target. **NOT CHANGED HERE** — adding it moves every shot and shot-on-target
   number in the project and needs its own gate. Do not "fix" it inside a
   fabrication pass; it is a separate calibration decision, and the finding is
   only that the question exists.
8. **ONE NEW ENGINE-vs-LEDGER DISAGREEMENT, UNDIAGNOSED.** Section D of
   `_diag_creator_truth.py` went from `7 agree / 0 disagree` to `7 agree / 1
   disagree`. The engine's assist and the ledger's backward scan are two
   INDEPENDENT derivations of the same fact and now differ on one goal. It is
   either a legitimate edge case (the ledger attaching the setup pass to a
   different delivery) or a real disagreement about causation. **Do not report
   the two as agreeing until it is explained.**
9. **The 10 remaining drawn shooters are not yet classified.** Each is printed
   with the man who actually had the ball. Do NOT assume they are defects: the
   back-scan's expected-predecessor rule is wrong for set-piece headers, since
   it expects the man a corner was AIMED at rather than the man who WON it.
   Classify before touching.

Then the wider set-piece work, still in the user's non-negotiable order:
corner box crowding via the pre-corner shape (block drops before the ball goes
out) → geometric shot resolution → exporter.

0. **PASS DIRECTION IS HAND-CODED, AND THAT IS A DECISION, NOT A BUG.**
   Read `DECIDER_BOUNDARY.md` before touching `_pass_destination` or
   `_best_forward`. The evolved brain's entire output is one of ten intent
   LABELS (`brain_integration.py:550`); which teammate receives, and therefore
   which way the ball goes, is a hand-written lookup dispatched on that label
   (`_find_target`, whose own docstring says "a lightweight geometric lookup,
   not a scoring system"). So direction is 100% hand-coded and 0% learned —
   while the surrogate has paid for forward progress since 2026-09-11
   (`2.0 + min(pass_advance/25, 1.5)`, correctly signed). The policy had no
   output that could claim it. Measured: median displacement −0.2 m over a
   whole match; backward passes 31% against a real 10–15%. The author reviewed
   this on 2026-09-30 and accepted the hand-coded design deliberately.
   `DECIDER_BOUNDARY.md` records the three ways to revisit it (forwardness
   head / pointer head / accept-as-a-written-rule) so it is not rediscovered
   as a phantom bug. ALSO NOTE: `POLICY_INTENT_AUTHORITY` was half-wired until
   2026-09-30 — it gated the five receiver-selection sites but NOT the two
   that force the delivery class, so **Checkpoint 32b's result is a PARTIAL
   result** and the "brain decides alone" experiment had never actually been
   run. Fixed in two conditions; default False so the shipped engine is
   byte-identical. Reproduce with `_diag_brain_only.py --arm deciders|brain`
   in SEPARATE PROCESSES.
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
**`PLAN_2026-10-21.md` (NEW)** — the detailed plan for the next working day:
every open thread in order, with the reason, the files, the exact command,
the gate and the CPU cost; the four decisions that are the user's; the open
measurements owed; and the traps this project has actually fallen into (a gate
that has never failed, a fixture that builds the thing under test by hand,
"documented pre-existing" as evidence, an aggregate where the GA reads a
bucket, a single-match median, the dive before the edit, and the commit
hazard). **Read it first when picking this up.**

**THE EXPORT'S TIME AXIS (2026-10-21) — read before touching `second`,
`minute`, the timeline, the shot map, or anything that sorts events:**
- `match_engine.py` — `MatchEvent.match_clock_s` (the TRUE seconds from
  kickoff). Stamped by `chronograph()` (every chain event, spread across the
  window the chain occupied), by `_emit_event()` (the engine's direct
  events), and by the KICKOFF and the goal celebration (both constructed
  directly, NOT through `_emit_event` — **that is why the first version
  shipped a timeline that opened with a 60-second jump backwards**).
  `chronograph()` now rounds `at` ONCE and writes the same value to the
  event and to the `TimedEvent`; two roundings at two precisions is a second
  truth.
- `exporter.py` — `_timeline_json` reports the true `second` / `clock` /
  `clock_s` / ms `timestamp`; the **shot map** sorts and places shots by the
  true clock (it used `minute * 60 + second`, i.e. a random position inside
  the minute).
- `plofa_export.py` — `convert_event` already preferred `match_clock_s`; the
  field now exists, so its clock is real and it gains a ms `timestamp`.
- `tests/test_export_timeline_clock.py` (6, NEW) — real match, real export:
  the timeline FLOWS, the first event is at kickoff, every exported second
  equals the engine's own chronology second, and a structural guard (with a
  non-vacuity case) on the shot map's axis.
- `_diag_export_timeline.py` — one match, then the whole clock sequence
  dumped so the answer can be READ. **Its first run found the two things
  the plan did not** (the unstamped kickoff, and the 2-dp/3-dp split).
- `_diag_roster_age_shift.py` — **why "same seed, different match" is often
  the CALENDAR**: `roster_loader`'s age is `date.today()`, and the workbook's
  boundary birthdays are printed. Read this before diagnosing
  nondeterminism.
- `_diag_matchevent_positional.py` — AST sweep for POSITIONAL
  `MatchEvent(...)` constructions. **Required reading before adding a field
  to that dataclass**: the `match_clock_s` field was inserted BEFORE
  `metadata`, and a positional construction would bind `metadata` to the
  wrong slot. The sweep is green (0 of 684 files).

**NEURAL REWARD + SCHEMA + INPUTS + THE PAIRING (2026-10-20) — read before
touching the objective, the sensor vector, a brain file, OR the collector:**
- `surrogate_collect.py` — `_correlate_outcomes_causal` is the PRODUCTION
  join (decision → the action event of its OWN chain; `_UNCLAIMABLE_EVENTS`
  excludes receipts, presses and failure consequences so the loss is not
  double-charged). `_correlate_outcomes` is the **legacy proximity join** —
  kept only for the tests that pin the defect and for `surrogate_rebuild.py`'s
  history. **Never wire a new collector to it.**
  `_make_chain_wrapper` / `_install_dispatcher_capture` / `_restore_dispatcher_capture`
  are the observer wiring; `_CHAIN_CALLS` must be cleared per match.
- `tests/test_surrogate_pairing.py` (15, new) — pins the join itself and the
  wiring. `TheOldJoinIsProvenWrongTests` is the load-bearing class: it pins
  that the legacy join STILL mis-pairs, so the defect cannot return silently.
- `tools/diag/_diag_pairing_guard.py` — 8-mutation negative control, 8/8
  caught, sha256-verified restore. Copy its structure for any new join work:
  anchors are escaped literals joined with an explicit `\r?\n`, and every
  mutant must still **compile**.
- `_diag_reward_dist.py` — the distribution gate. It reports the
  **sample-weighted** per-intent mean (counts-weighted; position tables and
  the global `""` row kept apart) NEXT TO the cell mean, because the cell mean
  is what the GA samples and the sample mean is what the reward paid. Read
  the sample-weighted one for "is this intent worth anything"; read the cell
  spread for "is there a gradient". **The aggregate is NOT what the GA
  reads** — `_lookup` is per bucket — so the per-bucket sign census it also
  prints is the collapse detector.
- **The new probes (2026-10-20), all read-only and all in the repo root
  unless noted:** `_diag_intent_mix.py` (a brain's intent mix over the
  synthetic corpus, against another brain on the SAME corpus — the collapse
  detector), `_diag_lookup_leak.py` (which value `_lookup` actually returns
  per bucket/intent/position, and whether it is borrowed across roles),
  `_diag_position_table.py` (one position's whole table, bucket by bucket),
  `_diag_global_row.py` (the global row cell by cell, with counts and
  contributions), `_diag_intent_pairing.py` (intent × paired-event
  cross-tab), `_diag_pairing_causal.py` (all three joinings over the same
  matches), `tools/diag/_diag_refit.py` (rebuild a surrogate from the raw
  corpus in seconds).
- **Four small EDITING TOOLS, because the editor's refusals cost real time
  and were worth diagnosing:** `_diag_show_lines.py` (a line range with
  `ascii()`, so a console cannot mangle what you are reading),
  `_diag_bytes.py` (one line's raw bytes), `_diag_linediff.py` (the first
  byte that differs between what you typed and what is on disk),
  `_diag_eol.py` (line endings, line by line), `_diag_anchor_count.py`
  (how many times an anchor string occurs, before you rely on it).
  **A refused edit is evidence about the edit, not about the file** — twice
  this session the tool was right and I was wrong (I re-flowed a line break,
  and I transcribed "was" where the file said "were").
- `threat_reward.py` (new) — the objective. `r = Φ(end) − Φ(start)` on
  `momentum_model`'s xT grid; shots score absolute `xg`, goals `GOAL_VALUE`,
  losses charged `LOSS_WEIGHT ×` the **opponent's** threat at the death point.
  `EXECUTION_TYPES` (20) is the ONE pairing list, imported by the collector —
  do not add a second one. `describe()` travels into every saved surrogate.
  Pure, no RNG. **Read the module docstring's "retired objective" block before
  proposing a reward change** — it records why engine-success scoring is a
  trap that looks correct.
- `tests/test_threat_reward.py` (47) — including
  `test_the_retired_objective_really_did_prefer_the_square_ball`, the
  **non-vacuity** case that reproduces the old reward and shows the new one
  reverses its ranking. Without that test, "the new reward is better" is an
  assertion rather than a measurement.
- `role_features.py` — `MENU_NAMES` (FB 8 / WING 9 / CB 9), the shared
  `_forward_options()` helper (one definition of `pass_lane_open` for **five**
  families), `align_role_tail()` (the single place a role tail is produced),
  `RoleSchemaError`, `reset_alignment_warnings()`. `V2_ROLE_D`/`V2_INPUT_D` are
  **derived** — never type a width by hand.
  ⚠️ `overlap_option` in FB and WING is `clamp(distance_to_nearest_forward_
  teammate / 30)`: a DISTANCE sensor whose name reads as an openness one, and
  it returns 0.0 for both "nobody ahead" and "somebody on top of you".
  **Do not read it as a lane-openness signal.** `short_cut` is the same shape.
- `tests/test_brain_schema_alignment.py` (39) — the load-bearing case is
  `test_inserting_a_feature_in_the_middle_shifts_a_legacy_brain_but_not_a_declared_one`;
  every growth so far has appended, so the by-name path's value is precisely the
  case that has not happened yet.
- `brain_integration.py` — `_brain_label()` / `_aligned_tail()`; both padded
  branches raise rather than truncate. If you see a `RuntimeWarning` naming a
  brain, the fix is a **retrain**, not silencing the warning.
- `brain_schema.py` — `BrainMeta.role_features` (None = legacy, `[]` is a
  declaration of zero and is refused) and `brain_meta_dict(role_features=…)`.
- `brain_evolution.py` — `synthetic_fitness` (`collapse_penalties=False`;
  re-base `0.5 + 0.5·clip(mean_reward, −1, 1)` then confidence-weighted,
  `scale` split between the surrogate and legacy paths) and `evolve()`'s
  `role_features` stamping, clamped to the brain's actual tail width.
- `surrogate_collect.py` — `FitnessSurrogate.prior = 0.0` (an assertion about a
  signed reward's mean, **not** a smoothing constant);
  `_correlate_outcomes(..., grid=…)`; imports `EXECUTION_TYPES` rather than
  keeping its own list.
- `tools/diag/_diag_neural_guard.py` (new) — 11-mutation negative control, all
  caught, sha256-verified restore. Copy this file's structure for any new
  reward or schema work: anchors are escaped literals joined with explicit
  `\r?\n`, every mutant must still **compile**, and it names the assertion that
  fired rather than just "something failed".
- `tests/test_momentum_model.py` — `PURE_TRAINING_CONSUMERS = {"threat_reward.py"}`,
  the one asserted exemption from "no engine module may import `momentum_model`",
  with a non-vacuity check that the walk reached the file.
- `tests/test_tactics_context.py` — the two schema-snapshot checks are now
  **derived**; they had been failing since 2026-10-05 because a hard-coded
  `[43, 44]` was treated as an invariant. Do not hard-code a width set again.

**CB PACE / ROLE TABLE (2026-10-20):**
- `match_engine.py` `_JOG_SPEED` (~2622) — the sustained shape-holding pace that
  drives almost every off-ball metre. `CB`/`DC`/`LCB`/`RCB` were the only
  outfield row inside the WALKING band (1.5) and are now 1.9; `DMC` (a label
  emitted nowhere) was aligned to `CDM`. **The comment above the table carries
  the CIES citation and the whole argument — read it before editing a value.**
- `tools/diag/_diag_cb_pace.py` — attributes every metre to the
  `_offball_move_player` branch that produced it (frame-local join), prints the
  per-role control table and the published-vs-integrator comparison. **Run this
  before believing any "role X covers too little" claim.**
- `tools/diag/_diag_cb_pace_ab.py` — the multi-seed A/B; switches the arm at
  RUNTIME so the two matches differ in exactly one value. Prints per-seed
  per-position, which is what separates a mechanism from trajectory divergence,
  plus the accounting invariant (max published seconds == clock).
- `tools/diag/_diag_pace_guard.py` — negative control, **3/3 caught**, names the
  assertion that fired, sha256-verified restore.
- `tests/test_role_pace_calibration.py` — 7 tests; includes the alias guard
  (`LCB` vs `CB`) that no behavioural test in this repo can see.

**SHAPE / DEPTH AXIS (2026-10-15) — read before tuning anything spatial:**
- `match_engine.py:2757` — `ax, ay = st.home_x, st.home_y` under
  `position_engine.SHAPE_ANCHOR_FORMATION`. **This one line is the fix for
  "the left back plays like a winger"** — it was `_minute_start_snapshot`
  (tracked position, refreshed every minute), a persistence term whose only
  fixed point is the ball. ~2981 still uses the snapshot deliberately.
- `position_engine.py` `SHAPE_ANCHOR_FORMATION` (default `True`) — what `ax`
  means.
- `position_engine.py` `DEPTH_ANCHOR_ENABLED` / `depth_home_anchor_blend` /
  `DEPTH_ANCHOR_*_W` — the CK35 mirror on x for CB/LB/RB. **Backstop only**;
  yields to a live run target; weights deliberately NOT retuned after the
  anchor fix.
- `position_engine.py` `defensive_block_target` — **the depth axis was mirrored
  in both branches and is now fixed.** `depth` is from the player's OWN goal.
- `tactical_shapes.py` `style_defense_deltas` + `_STYLE_DEFENSE_ARCHETYPE` —
  the defensive-shape style temperament. `formation_stance_for` does NOT read
  the profile, by design and by documentation now.
- `position_engine.py` `apply_formation_stance(style=...)` — the style is part
  of the idempotency key; dropping it from either is a silent no-op.
- `tools/diag/_diag_depth_anchor_ab.py` — **the master A/B.** 4 arms,
  frame-selectable (`--home-style/--away-style`, defaults to `run_match.py`'s
  own constants). §7 = is-this-player-an-outlier-or-his-team; §8 = the
  receipt/origin join. **Use §7/§8 before concluding anything about a single
  player.**
- `tools/diag/_diag_chain_attrib.py` — term-by-term Δ of `tx` through the
  whole chain. This is what found the 37 m anchor error. Read its per-stage
  output before blaming any single rule.
- `tools/diag/_diag_block_frame.py` — milliseconds; catches an
  `attacks_right` mirror or any depth arithmetic error in a geometry function.
  **Highest value per second in the repo — run it first whenever a shape looks
  wrong.**
- `tools/diag/_diag_shape_fixed_point.py` — milliseconds; where a target
  recursion settles, in pure algebra.
- `tools/diag/_diag_depth_anchor_control.py`,
  `tools/diag/_diag_style_temp_control.py` — milliseconds; **is a new lever
  actually REACHED?** (role scope, band non-vacuity, switch reversibility).
  Copy this pattern for any new rule.
- `tools/diag/_diag_wide_depth_live.py [--report-only]` — whole-team per-role
  shape + frame validation (keeper/CB check). Caches samples, so a re-report
  costs nothing. **Do not commit its `_diag_wide_depth_cache.json` — 166 MB.**
- `tools/diag/_diag_frost_asante.py`, `_diag_cam_passmap.py`,
  `_diag_asante_frame.py` — the pass-map / CAM analysis, all reading the
  existing export (instant, no match).
- `tests/test_style_defensive_temperament.py` — 16 tests. Includes two AST
  guards (the call site actually passes `style=`; `style` is in the
  idempotency key). Model these for any new rule.
- `event_chain.py` — POLICY TARGET DEFERENCE: `POLICY_TARGET_DEFERENCE` /
  `POLICY_TARGET_DEFERENCE_MARGIN` (default OFF / 0.05, **margin fitted to ONE
  match, and not the binding parameter**) on `PossessionChain`;
  `_policy_target_deference()` (the pure, RNG-free rule);
  `forced_source` / `receiver_authority` locals; `target_authority` +
  `target_deference` stamped on the PASS event inside the `metadata` dict at
  ~2930; the deference meta now also carries **`matrix_silent`** (bool: the
  chosen option scored <= 0, so the margin was not applied) and
  **`matrix_reason`** (which branch produced the choice) — because
  `gap_frac == 0.0` otherwise means both "the matrix is silent" and "a genuine
  near-tie", and on the first live run the silent mode was the majority of the
  `fired` set. Through balls bypass `forced_receiver` entirely
  (`policy_through_target` ~3111), so the policy already owns that destination
  and this flag cannot reach it.
- `attacking_matrix.py` — `AttackingDecision.value` / `.scores` (appended,
  defaulted) populated by `_pass`, so "the matrix is nearly indifferent" can be
  evaluated in the matrix's own units. `value` is the CHOSEN option's score, not
  `max(scores.values())` — the cascade selects from subsets on purpose.
- `plofa_export.py` — `_decision` surfaces `target_authority` /
  `target_deference`; `schema.target_authority` (+ meaning/note) documents every
  value and states the difference from `decision_authority`; `_index` publishes
  the `target_authority` histogram and the `target_deference` gap distribution
  (absent when the experiment is off — absence is a fact, not a zero).
- `tests/test_policy_deference.py` — 19 tests: the payload, every exclusion, the
  zero-preference case (and that silence is distinguishable from a near-tie in
  the payload), the relative reference, agreement, unknown target, the
  no-RNG property, end-to-end delivery, and the OFF path proved decision-identical
  over 60 seeded possessions on freshly-built worlds.
- `tools/diag/_diag_deference_guard.py` — negative control, **5/5 caught**,
  byte-identical restore. Its CRLF-vs-LF lesson: an anchor written with `\n`
  against a CRLF file matches NOTHING and every mutation silently becomes a no-op
  — the exact failure the control exists to catch.
- `tools/diag/_diag_deference_ab.py` — the live A/B: one match per arm
  (`--arm both`), reports the `target_authority` histogram, the `gap_frac`
  distribution **split into matrix-silent vs comparable** with the matrix
  `reason` histogram, the pass map as BOTH the engine's `pass_direction` label
  and `pass_classifier.classify_direction` applied to the arrival point, and the
  band-free median signed advance. `--dump PATH` writes per-pass records as JSON
  (an `_on`/`_off` suffix is added for `--arm both`) so the analysis never needs
  a second match. ⛔ Its first version used a probe-local `SQUARE_M = 5.0` band
  and reported a false "30-35% disagreement" — use the engine's classifier.
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
- `striker_behavior.py` — striker spatial profiles + `decide_run` (behind /
  hold / box). **Live since 2026-09-29** via `PositionEngine.striker_run_targets`
  + `MatchEngine._striker_runs`; was dead code reachable only through
  `drift_minute`. `offside_line_nx` is the Law-11 second-last-defender line and
  is direction-normalised; `run_behind_target` sites off the line (+2 m).
  `rng` is injectable — the live path passes a crc32 stream, never
  `random.random()`.
- `position_engine.py` — spatial layer. Positional-play additions:
  `rest_defence_violator` / `rest_defence_clamp` (the one superiority type
  implemented as a CONSTRAINT, depth-only, backstop-only),
  `striker_run_targets`, `_deterministic_rng`, `_ShapeShim`,
  `REST_DEFENCE_*` / `STRIKER_RUN_*` kill switches.
- `match_engine.py` — `_offball_move_player` is where both new layers land
  (striker run blended into the target after CK37/CK38; rest defence clamped
  LAST, after `live_spacing_redirect`). `_offball_tick_seq` gives the 10 Hz loop
  a tick identity — `match_clock_s` cannot, because both tick loops advance it
  outside. Per-tick team decisions are cached in `_rest_defence_cache`,
  per-(team, minute, zone) in `_striker_run_cache`. `_initialize_simulation()`
  ~4299 is a TRAP — do not add brain/mind cache clearing there.
- `ball_vision.py:124`, `perception.py:422`/`:519` — the three former
  `hash(ident)` sites, now `zlib.crc32`. `tests/test_determinism.py` pins the
  count at **zero**, which is the guard that keeps it there.
- `_diag_repro_process.py` — the cross-process instrument (one match per
  process, prints `DIGEST`). `_diag_seed_repro.py` — the in-process check and
  its `arm none` negative control that falsified the brain-cache claim.
  `_diag_hash_divergence.py` — first divergent event between two hash seeds.
  `_diag_football_sites.py` / `_diag_press_band.py` — the corrected press A/B.
- `tests/test_positional_play.py` — 46 tests: the offside-line regressions, the
  rest-defence invariant (and that it stays silent in a normal shape), the
  depth-only clamp, the no-global-RNG guarantee, and a full real match.
- `_diag_posplay_speed.py`, `_diag_cross_probe.py` — throughput and
  cross-detection probes (diagnostics, not tests).
- `event_chain.py` — ALSO the chance-provenance site: `ChainResult.shoot_player`
  / `shoot_assister`, `_resolve_assister` (NO FALLBACK, by design),
  `_pick_creator` (STILL LIVE — feeds `CHANCE_CREATED.player`), `_named_shooter`,
  `clamp_attack_x` (~1030), `_STREAM_PARITY_DRAW` (~5719, DO NOT DELETE),
  cross-receiver clamp (~3404, unverified).
- `tests/test_chance_provenance.py` — 26 tests. Contains an AST guard asserting
  every `set_piece` call site in `match_engine.py` passes `attacks_right`, which
  is why that class of omission cannot recur silently.
- `_diag_chance_coords.py` — primary chance-provenance probe; exports
  `build_pair(home, away)` reused by the other probes.
- `_diag_shot_pipeline_split.py` — proves which pipeline emitted each shot/goal
  by `id()`-tagging the event objects at emission. This is what established
  that `_simulate_shot_sequence` fires zero times.
- `_diag_teleport_moves.py` — single-call `record_touch` jump audit. Measures
  the population; CANNOT isolate the cross-receiver clamp. Do not quote a
  clamp figure from it.
- `_diag_shot_origin.py` — v4, `id()`-tagged. Docstring records three rejected
  pairings; read before writing another.
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

---

# WORLD LAYER (`world/`) — running log

Separate workstream from the brain project above. Governed by
`PLOFA_WORLD_LAYER_AUDIT.md` and the World Football plan
(`Documents/# PLOFA FOOTBALL WORLD.txt`). Full documentation in `README.md` §11.

## Hard rules
- **`auto_run_match.py` is still off-limits** (README §2). The world layer never
  touches it, `season_state.json`, `season_stats.json`, `manager_state.json`,
  `referee_state.json`, or `plofa_output/`.
- **No competition runs its own engine** (plan §10). The world layer chooses
  which match, when, and under which rules; `MatchEngine` is unchanged and is
  reached only at call time.
- `world/` must be imported by NOTHING in the live 26/27 pipeline. Both this and
  the no-26/27-imports rule are enforced by tests, not convention.
- Keep the `PYTHONHASHSEED=0` discipline of README §8 in anything that places or
  orders fixtures. `test_build_is_deterministic_across_processes` runs the
  calendar in two interpreters under different hash seeds and requires identical
  output.

## State — 2026-09-26: Phases 1–8 + phase 10 delivered; phase 9 outstanding

Phases 1–3 (`ids`/`model`/`competition`) had shipped on 2026-09-23 with **five
failing tests and no suite at all for `model.py` or `competition.py`**. Fixed
before building on them:

- **`mint_id` did not canonicalise its key** (real bug). It hashed the raw
  string, so `mint_id("player","Alan Shearer") != mint_id("player","ALAN
  SHEARER")`, contradicting the module's own "same canonical key ⇒ same ID"
  contract. `NameAdapter.register` masked it by pre-canonicalising. Now
  `mint_id` normalises internally; verified **zero existing IDs change** (every
  key the adapter mints was already canonical), so no world ledger is invalidated.
- **`id_override` bypassed the ownership check** (real bug). The guard read
  `elif ent.canonical_key != key and id_override is None`, so a caller supplying
  an explicit ID could silently merge two identities — the exact failure the
  module docstring promises never happens. Ownership is now enforced
  unconditionally, and the legitimate rename workflow moved to an explicit
  `recannonicalise()` so the capability survives without the escape hatch.
- **`CompetitionRules` did not survive a JSON file round trip** (real bug).
  `to_dict` coerced only top-level tuples and `from_dict` only top-level ones
  back, so `from_dict(json.loads(json.dumps(r.to_dict()))) != r` for any nested
  tuple. Added recursive `_json_safe` / `_tuples`; round trip is now exact.
- **Validation order** (real bug). `LeagueCompetition`/`KnockoutCompetition` read
  `rules.type` before `super().__init__` validated the type, so a dict argument
  produced an `AttributeError` instead of the documented `TypeError`. Added
  `_require_rules`.
- **Dead code removed**: `world/model.py::_COUNTRY_REF_FIELDS` (defined, never
  used).
- `tests/test_world_ids.py` asserted a fictional world (20 clubs, "Hartwell
  City", "Westside Heroes", "Port Colborne United") against the real roster of
  18 clubs / 433 players, contained two mutually contradictory tests, had a
  dead placeholder line, and its "digest drift" guard asserted nothing. Rewritten
  against reality, with the club digests now genuinely frozen.
- `tests/test_world_competition.py` **written from scratch** — Phase 3 shipped
  with no suite despite the audit's file map listing one.
- `tests/test_world_model.py` **written from scratch** — `world/model.py` had
  zero coverage despite being the source-data contract every other world module
  reads. It pins the ID-only reference graph per class, so adding a stray
  `home_club_name` field now fails the suite, and it pins the per-season stadium
  capacity rule (plan §6: a historical season must never inherit today's
  capacity — an unknown season returns `None`, never a guess).

**Phase 4 delivered**: `world/calendar.py` (multi-competition scheduler) +
`tests/test_world_calendar.py` (41 tests, adversarial — fixtures are handed to
the calendar *wanting* to collide and the invariant is asserted afterwards) +
`world/proof.py` (the audit §20 proof harness).

**Phase 6 delivered**: `world/ledger.py` + `tests/test_world_ledger.py` (41).
This is the module plan §18's "CRITICAL INTEGRATION TEST" exists for. Two real
bugs were found by its own suite while writing it:

- **Single yellow cards were never recorded** (real bug). The accumulation rule
  counted entries in `suspensions`, but a single yellow only ever produced a
  suspension at TWO yellows — so a five-in-six rule could never see the cards
  that trigger it. Every yellow is now stamped into `yellow_card_events` with
  the `match_index` at which it was earned, because the window is counted in
  MATCHES, not days, and "a card that is not recorded is a card the rule can
  never see".
- **Injuries and suspensions never expired** (real bug, the most dangerous kind
  here). Nothing decremented `matches_remaining`, so an injured player stayed
  unavailable FOREVER, silently ending a season. Added `PlayerState.settle()` +
  `WorldLedger.advance_matchday()`, with `record_matchday()` bundling
  record-then-advance as the safe default. A test asserts an ACL eventually
  clears and that the ledger holds no expired events afterwards.
- Scoped-vs-carry-across field names differ on purpose ("minutes" on a season
  line, "minutes_played" on the continuous state), so the relationship is
  declared in `SCOPED_TO_CARRY_ACROSS` and asserted, rather than inferred from
  a name match.

**Phase 7 delivered**: `alltime_db_competitions.py` +
`tests/test_alltime_db_competitions.py` (22) — the warehouse competition keying
migration, plus the `migrate-competitions` / `competition-status` CLI. **Applied
to the live warehouse 2026-09-25** with Trevor's explicit approval: 3 tables
rebuilt, 2 columns added, 19,467 rows backfilled to `CMP-PLOFA`, row counts and
content checksums preserved, 0 FK violations, `integrity_check: ok`. Backup at
`alltime.db.pre-competition-keying`. The rebuild left 138 MB of free pages, so
run `VACUUM` after any future rebuild. The pre-existing `test_alltime_db.py`
(10 tests) still passes.

**The crossing delivered**: `world/ingest.py` + `tests/test_world_ingest.py` (21),
which is where the world layer's two halves finally meet. Its governing rule is
that the ledger NEVER re-derives what the engine decided (plan §13): per-player
lines come from the exporter's own `stats` dict, fatigue/fitness/injuries from
`SubstitutionController.stamina` verbatim. Two real findings, both from the
real-match test rather than from reasoning:

- **`MatchResult` does not carry `sub_controller`.** It has config/state/timeline/
  squads and nothing else, so post-match stamina lives on the controller the
  caller created and passed to `set_stamina_controller`. An adapter reading the
  result alone silently loses every fatigue and injury value. `apply_result` now
  takes `sub_controller=` explicitly, and the shape fact is pinned by a test so
  it cannot regress quietly.
- **`shots` has no single key in the exporter** — it is split three ways, so the
  ledger's one field is a definitional sum, pinned in `world.ingest.SHOT_PARTS`.

**Pre-existing bug fixed (not from this work)**: `alltime_db.py report` crashed
with `UnicodeEncodeError` on real player names containing 'ć' / 'ķ' (e.g.
"Perćy Luka") because the default Windows console codec is cp1252. Verified
present in the pre-migration backup, so it predates the competition keying. Fixed
once at the CLI entry point (`_make_console_utf8_safe`), so every command in the
file can now print the names the warehouse actually holds.

Design notes worth keeping:
- Placement is greedy in priority order over a finite horizon. Unplaceable
  fixtures become `unscheduled` violations; they are never dropped and never
  allowed to break an invariant.
- `validate()` re-derives the invariants from the placed fixtures **independently**
  of the placement code (audit §18 risk 3: an invariant enforced by convention is
  not enforced).
- `postpone()` searches forward for the first fully legal date. A naive
  "roll to the next matching weekday" can never succeed in a league where every
  club plays every week — that was found by a test, not assumed.
- `reschedule()` refuses a move that would collide rather than applying it with a
  warning.
- **Rest floors are resolved as the strictest claim on the club** (real bug,
  found by a test written after the fact). Placement originally enforced only
  the *incoming* fixture's competition floor. Because the cup is placed first on
  priority, a lenient league fixture could then be dropped right beside it,
  silently defeating the cup's stricter rest requirement — `validate()` flagged
  5 real violations the placer had created. A gap now has to satisfy
  `max(floor(incoming competition), floor(each neighbouring fixture's
  competition))`, and `build()` and `validate()` resolve floors through the same
  `_min_rest_for` helper so the independent check can never verify a weaker rule
  than the placer enforced.
- The proof harness seeds its synthetic scorelines with `zlib.crc32`, never
  `hash()` — builtin string hashing is salted per process and made the printed
  table differ between runs.

**Phase 8 delivered**: `world/qualification.py` + `tests/test_world_qualification.py`
(32). Promotion/relegation and continental entry are pure data (plan §14) and the
entrants are read from the real competition-keyed warehouse, so plan §15's
"generated from actual previous-season results" is literal, not aspirational.
Real bug found by its own suite: a relegated club was being reported in
`unqualified` — but "qualified for nothing" must mean *still in the division with
no berth anywhere else*; a relegation is not a failure to qualify. Playoff ties
are deliberately NOT simulated here (that is phase 5); an unplayed tie leaves the
position empty, and an aggregate that identifies no winner is refused.

**Phase 5 delivered**: `match_engine.py` + `tests/test_extra_time.py` (18). The
only phase that touches the engine, so the gate is the strictest in the project.
Extra time runs through the SAME `_run_minute` closure as the first 90 — no
simplified "cup mode" — and ends at a two-goal lead or 30 minutes, whichever
comes first. The shootout draws from `_COSMETIC_RNG`, never the football RNG, so
a tie decided from the spot cannot perturb the seeded sequence (a test asserts
the football stream is untouched). `winner_team` is deliberately `""` for a league
match, so knockout semantics never leak into a warehouse row.

**IMPORTANT — pre-existing finding, not a phase-5 defect:** the live match is
**not seed-reproducible**. Two fresh processes with the same `random.seed(4242)`
and `PYTHONHASHSEED=0` produce different event timelines (verified 2026-09-25:
2873 events / 1-4 vs 3170 events / 1-2). Adding `np.random.seed()` does not fix
it either, because the live path calls `np.random.default_rng()` (entropy-seeded,
ignores `np.random.seed`) — the known instances are in training/evolution code
(`football_brain.mutate/crossover`, `manager_brain`), so the live culprit is
still unidentified. **Consequence: every calibration number taken before this is
suspect for the same reason, not just the world layer.**

> ### ⛔ RETRACTED 2026-10-04 — THE DIAGNOSIS WAS WRONG, ALL THREE CLAUSES
> The observation (two processes, same seed, different matches) was real. Every
> proposed cause was false, and the two "fixes" that followed from them were
> aimed at code that was never involved.
>
> - **Not `np.random.default_rng`.** It is called 531,424 times per match and
>   **every** call site passes an explicit seed. None reaches the event timeline.
> - **Not the module-level brain/mind caches.** `_diag_seed_repro.py`'s `arm none`
>   — same seed, a second `simulate()`, **nothing cleared at all** — is
>   byte-identical to the reference (digest `0373f21a9ce5`, 2463 events). All five
>   clearing arms match it. The caches do not contaminate.
> - **It was `PYTHONHASHSEED`, and it was the whole story.** Exactly three
>   `hash()`-on-string sites sit on the match path. They are now `zlib.crc32`, and
>   one match per process under `PYTHONHASHSEED` 0 / 1 / 12345 gives **one
>   digest** (`79857b22c6e2`, 2656 events). It was `hash(ident)` in
>   `ball_vision.py:124` and `perception.py:422,519` — all three feeding
>   `np.random.default_rng(...)`, which is why the two symptoms looked like
>   unrelated bugs.
> - **A 26/27 caution that survives.** The numbers in this file taken before
>   2026-10-04 were taken under an *unpinned* hash seed, so they are not
>   byte-reproducible and the per-arm "swing" between measurements includes
>   hash-order noise. "2.5x swings between arms on one machine are warm-up
>   noise" is not established — it was never separated from hash order.
>
> This is why the phase-5 gate counts **executions** of the new code rather
> than comparing simulated output: a fingerprint gate would now report
> "regression" for a change nobody made.

## Next Move — world layer

**Phase 10 delivered** (`world/realworld.py`, `world/testworld.py`,
`world/month.py`, `world/runmonth.py`, `tests/test_world_month.py`): the plan
§18 one-month integration test, run with REAL matches through the real engine,
on the **real 2026-27 calendar**. Four design problems it exposed:

- **The rules must be real, not invented.** An earlier version of the test world
  made up its own kickoff times, rest periods and matchday spacing, which proves
  the plumbing and nothing about the football. `world/realworld.py` now holds
  the published 2026-27 data: PL 38 matchdays / 20 clubs / 3 relegated with
  four international breaks including the three-week World Cup window; the CL's
  eight real league-phase matchday windows and its full knockout calendar to the
  5 June 2027 final at the Metropolitano; EFL Cup and FA Cup round dates; and
  the real rules (CL two-legged with extra time and penalties, NO away-goals
  rule, abolished 2021).
- **Rounds must land on real dates, not derived ones.** `SchedulePattern`
  gained `fixed_round_dates` and `round_date_windows`. A real competition
  announces a SET of days ("8-10 September", or a Tue/Wed/Thu batch); a
  day-count slip invents dates nobody announced, and an early version drifted
  the EFL round onto a Saturday the EFL never listed. Only published dates are
  candidates now, and a moved round is REPORTED via `Calendar.slipped()` rather
  than being a violation or a silent success.
- `LeagueCompetition`/`KnockoutCompetition` now validate `schedule_kind`, not
  the `type` label. The real Champions League league phase is
  `type="continental"` but is genuinely a league, and keying on the label made
  it unbuildable.
- `_ROUND_NAMES_BY_SIZE` had no entry for 32 or 64, so an 18-club field (padded
  to 32) raised `KeyError` — which is exactly the FA Cup third round's size.
  Both are now supported.

- The month window must be selected **per competition**. A twelve-club round
  robin is 66 fixtures over eleven matchdays; it saturates every midweek slot,
  so the cup tie can only be placed after the continental season ENDS — which
  pushed the "one-month" test three months apart. `build_test_world` now
  truncates the leagues so the world fits in ~4 weeks.
- **REAL BUG, found in the first real match:** eight unused substitutes were
  counted as appearances. The exporter emits a stat line for every NAMED player,
  including subs who never came on (minutes 0), and `ledger.apply_post_match`
  incremented `appearances` unconditionally. That would have inflated every
  appearance total in the warehouse for every season. Fixed: an appearance is a
  match PLAYED (`minutes > 0`), in both the continuous total and the scoped
  line, and `match_history` now records `played`.

The continuity checker is itself tested by breaking the ledger six ways (reset
across a boundary, an over-claiming competition line, a drifted continuous
total, a double-booked club, a forged appearance count) and asserting it catches
each. A checker that has never failed has not been tested.

The real calendar is genuinely congested — the EFL Round Three window (8-17
September) overlaps UEFA matchday 1 (8-10 September) — and the world layer
resolves it correctly: the cup yields to the higher-priority European night and
lands on a real EFL evening, with no club playing twice. That is a better test
than the invented calendar it replaced, because it is a problem real football
actually has.

**Phase 9 (continental group stage) is the only audit row still outstanding.**
The current continental competition is a single round robin, which is a
legitimate league phase and is what the phase-10 test uses. A true
group → knockout structure is still unbuilt.

Open items, none blocking:

1. ~~**Seed reproducibility** — the highest-value loose end in the whole
   project.~~ **CLOSED 2026-10-04**, with the stated cause retracted: it was
   `PYTHONHASHSEED`, not the module-level brain/mind caches
   (`brain_integration._brain_registry`, `_pos_brain_cache`,
   `cognition.mind._minds`). Three `hash()`-on-string sites are now crc32 and
   cross-process replay is byte-identical. The *production roster* path
   (`test_determinism.py` point 3 — same seed, same hash seed, real rosters
   from `PLOFA-2026-2027.xlsx`, 3057 / 3060 / 3095 events) is a SEPARATE open
   bug, allocation-dependent (`id()` or an unseeded generator), and was not
   re-tested after the crc32 fix. **DO NOT add cache clearing at the top of
   `_initialize_simulation()`** — that was the proposed fix and it is a TRAP:
   it would wipe the season-hydrated minds (`auto_run_match.py:1157` carries
   last matchday's temperament) and discard the 13 caller `register_brain`
   sites that are how every A/B harness selects a brain set.
2. **Phase 9** — continental group → knockout.
3. **`world_data/` JSON templates** — the test world is built from the live
   roster rather than from data files. Unblocked now the schema is complete.
4. **`world/production.py`** — `world/ingest.apply_competition_context` already
   does the hand-off; this is naming/location.

Standing instruction: do not start a phase before the previous phase's suite is
green and the 26/27 regression suites still pass. After any future
`migrate-competitions`-style rebuild, run `VACUUM` and re-check `integrity_check`.

### 26/27 regression status (2026-09-25)
`tests/tests.py`, `test_chronography.py`, `test_possession_causality.py`,
`test_checkpoint7_subsystems.py` → **49 passed, 1 failed (20m22s)**. The single
failure is `tests.py::test_pass_matrix_sums_match_real_events` (395 vs 394) —
the **known pre-existing failure** documented in README §6, with its exact
documented signature (matrix − manual = the number of successful crosses, a
test-vs-production definitional mismatch in `PassMatrix.build`, not a sim bug).
No new 26/27 regression.

## PERCY THE CHOSEN ONE — GOAT PLAYER WIRING (2026-10-16)

Percy Luka (Player ID 11, RW, Team 1, Tolandian, traditional_winger) is the first player in the universe (2018 → today). He already had a soul (ATTACKING_PROPHET, Ω=1.55) and the cognition layer existed, but he played with the same 24→32→32→10 brain (2,186 params) as everyone else. This section documents how he was made into the chosen one — bigger brain, dedicated training, never-forgetting mind — wired into both runners so he's active in Sunday's fixture and available for testing.

### Module: `percy_the_chosen.py` (new file)

Self-contained module. Does not modify any existing brain/evolution code. Registers Percy's brain + mind by exact player **name** in the brain/mind registries, so `NeuralDecisionBrain.decide` serves him before any position-file auto-load. No other player is affected.

### Brain architecture

| | Percy | Standard player |
|---|---|---|
| Architecture | 32 → 64 → 64 → 10 | 24 → 32 → 32 → 10 |
| Params | **6,922** | 2,186 |
| Input | 24 shared + 8 WING role features | 24 shared only |
| Role family | WING | per position |

The WING role block feeds Percy wing-specific geometry: `marker_dist, one_v_one, inside_channel, overlap_option, cross_zone, cross_target_open, box_arrival, cover_behind`. He learns winger decisions, not generic ones.

### Training pipeline (the project's real pipeline, not a custom shortcut)

`train_percy()` uses `brain_evolution.synthetic_fitness()` scored against the production surrogate `brains/surrogate_pos.json` — the exact same fitness signal that trained the brains in `brains/`.

T1 production hyperparams (per `brains/_manifest_all.json`):
- surrogate: `brains/surrogate_pos.json`
- 80 gen × 32 pop × 300 states, seed 123
- role_features ON, perception ON
- goal_bias: 0.0 (playmaker) / 0.25 (attack-oriented)

Two trained variants saved:

| File | Style | goal_bias | attack_oriented | Fitness | Best for |
|---|---|---|---|---|---|
| `brains/percy.json` | Attack-oriented | 0.25 | True | 0.5336 | Scoring, crosses, shots |
| `brains/percy_attack.json` | Attack-oriented | 0.25 | True | 0.5336 | Backup (identical to percy.json) |
| `brains/percy_playmaker.json` | Playmaker/creator | 0.0 | False | 0.4746 | Through-balls, switches, safe build-up |

`percy.json` is the active brain wired for Sunday. If attack doesn't click in the real season, copy `percy_playmaker.json` over `percy.json` and Percy switches to creator mode.

### Wiring — call once before kickoff

`wire_percy(player, train_if_missing=False)` registers:
1. **Big brain** under `"Perćy Luka"` (real roster) and `"Percy"` (scratch alias)
2. **Never-forgetting mind** under both names (decay OFF, forget threshold 0, view_radius 45m, confidence 1.5)
3. **Player DNA** (vision/composure/decisions → 99) and **soul** (ATTACKING_PROPHET, Ω=1.55) if the player object is passed

Insertion points:
- `auto_run_match.py` — after the cognition loop (line ~1244), before `MatchEngine` creation
- `run_match.py` — after soul attachment (line ~337), before match config

Example:
```python
from percy_the_chosen import wire_percy
percy_player = next((p for p in all_players_flat if p.name in ("Perćy Luka", "Percy")), None)
if percy_player is not None:
    wire_percy(player=percy_player, train_if_missing=False)
```

If Percy isn't in the squad → `percy_player` is None, wiring skipped, warning printed, match runs normal. No crash.

### A/B test results (7 matches per arm, same seeds)

| Stat | Chosen-One (attack) | Baseline RW | Delta |
|---|---|---|---|
| Goals | 1 | 1 | 0 |
| Shots on target | 2 | 1 | +1 |
| xG | 0.838 | 0.356 | +0.482 |
| Passes | 84 | 78 | +6 |
| Carries | 78 | 114 | −36 |
| Decision quality | 0.801 | 0.705 | +0.096 |
| Team points | 13 (4W-1D-2L) | 16 (5W-1D-1L) | −3 |

Intent mix (chosen-one): SAFE_PASS:42, SWITCH:37, PROGRESSIVE_PASS:41, CARRY:9, CROSS:3, PROTECT_POSSESSION:5
Intent mix (baseline): CARRY:39, PROGRESSIVE_PASS:39, SAFE_PASS:27, DRIBBLE:8, SWITCH:18, PROTECT_POSSESSION:11, RECYCLE:1, SHOOT:1, CROSS:1

Percy is definitively different — playmaker/creator (switches, through-balls, progressive passes) vs baseline carrier/dribbler. Higher decision quality, higher xG, more crosses/shots on target.

### Reusable template for future "goat" players

To make any player the chosen one, follow these steps:

1. **Create a module** like `percy_the_chosen.py`:
   - `build_<name>_brain(seed, input_size)` — oversized architecture, DNA-seeded weights
   - `train_<name>(generations, population, n_states, seed, goal_bias, attack_oriented)` — GA using `synthetic_fitness()` + `surrogate_pos.json`
   - `build_<name>_mind(view_radius)` — never-forgetting (decay OFF, threshold 0), supreme confidence
   - `wire_<name>(player, ...)` — register brain + mind by exact name

2. **Pick input width**: Use `V2_INPUT_D[role_family]` for role-features (e.g., WING=32, ST=32, CB=31) or `INPUT_SIZE=24` for plain

3. **Pick architecture**: Bigger hidden layers (64→64) give ~3× capacity. Don't go so big that training collapses.

4. **Train with the project's pipeline**: `synthetic_fitness()` + production surrogate, not a custom fitness function

5. **Wire into runners**: `auto_run_match.py` (after cognition loop) and `run_match.py` (after soul attachment)

6. **Register by name, not position**: `_brain_registry[player.name]` is checked before position-file auto-load, so the player keeps their special brain regardless of what position they're listed as

7. **Save variants**: Save attack-oriented and playmaker variants separately; switch by copying the desired one over `brains/<name>.json`

### Position flexibility — if Percy plays ST or CAM instead of RW

Percy's brain is registered by **name** (`"Perćy Luka"`), not position, so `NeuralDecisionBrain.decide` serves him his oversized brain **regardless of what position he's played at**. He's still Percy — soul (Ω=1.55), never-forgetting mind, maxed DNA, bigger network all carry over.

**However**, his brain was trained with WING role features (8 dims: marker_dist, one_v_one, cross_zone, box_arrival, cover_behind, etc.). If played at ST or CAM:
- The sensor builder still produces a 32-d vector (24 shared + 8 role block) because ST/CAM role blocks are also 8 dims
- But the role block features are ST/CAM semantics, not WING semantics — the dimensions match but the meanings differ
- He'd be interpreting ST/CAM geometry through WING-trained weights: still "Percy" (same network, same mind, same soul), but not optimally utilized since his weights were tuned for wing geometry

So yes, he's still Percy everywhere — but he's at his best as a RW (or wide attacking role) where his WING-trained role features match what he actually sees.

### Files modified (commit ea56071)

- `auto_run_match.py` — Percy wiring after cognition loop
- `run_match.py` — Percy wiring after soul attachment
- `tools/gates/validate_neural_xl.py` — fixed heuristic decide kwarg (pops `coach_instructions`)
- `percy_the_chosen.py` — new module (brain, mind, training, wiring)
- `percy_ab_test.py` — 7-match A/B test script
- `percy_match_test.py` — single-match attestation script
- `brains/percy.json` — active attack-oriented brain
- `brains/percy_attack.json` — attack backup
- `brains/percy_playmaker.json` — playmaker backup

### All-time DB reference (Percy's career before the brain upgrade)

Player ID 11, Perćy Luka, RW, Team 1, Tolandian, traditional_winger — all-time totals:
- Goals: 52, Assists: 25, xG: 21.54, xA: 13.93, Shots: 200, SOT: 111 (55.5%), Progressive passes: 546, Progressive carries: 390
- Scores at 2.4× his xG — elite finisher who also creates

`auto_run_match.TEAM_CATALOG` defines 20 clubs; `PLOFA-2026-2027.xlsx` carries
squads for 18. **Resolved by Trevor: Hartwell City and Thornfield United are test
teams with test players, entered by hand in the scratch runner `run_match.py`.
The 18 real clubs are the whole world.** No action needed.

Consequences, all verified and now expected rather than suspicious:
- `auto_run_match.py:936` hard-exits on a team absent from the Excel, so the
  production path correctly refuses the two test teams.
- `plofa_output/Hartwell_City_vs_Thornfield_United_MD01(good)/` came from the
  scratch runner, which is the correct place for a test fixture.
- The world identity index reports 18 clubs, built from the Excel. That is the
  right answer. `EXPECTED_CLUBS = 18` in `tests/test_world_ids.py` is the real
  football world, not a data gap.

## THE PER-POSITION REWARD SCALE — the v5 census's real cause (2026-10-21)

Plan item 1.1, DONE and gated. **This section also CORRECTS the plan's own
diagnosis**, which was right about the conclusion and wrong about the
number.

### THE PLAN'S "45x" WAS NOT A PROPERTY OF THE TABLES

`PLAN_2026-10-21.md` said *"the reward's spread varies **45x** between
positions (0.09 for the CB, 0.002 for the CM)"*. Measured on the v5 corpus,
those two numbers are **not** the per-position spread of the tables:

| quantity | measured |
|---|---|
| the per-position spread of the FITTED cells | **3.3x** (widest LW 0.0228 / flattest RW 0.0069) |
| the per-position spread WITHIN a bucket (the GA's gradient) | **6.8x** |
| the per-position sd of the RAW samples | **4.8x** |
| **45x** | the ratio of TWO DIFFERENT BRAINS' mean rewards (the v5 brain vs the shipped one), i.e. a number about two brains, not about the table |

So the scale is **not** correcting a 45x spread. It is correcting something
the table's spread does not show at all, and the number that shows it is the
one the GA actually reads.

### THE NUMBER THAT DECIDES THE GA: THE REWARD'S SWING vs THE GUARDS

`synthetic_fitness` scores a brain with

    base    = 0.5 + 0.5 * clip(mean_reward, -1, 1)      # the football
    fitness = base
              + CONFIDENCE_GUARD_W * (mean_conf - 0.5)   # 0.05
              + DIVERSITY_GUARD_W * bdiv                 # 0.05
              - indecision_penalty / n_states            # up to 0.10

Every guard is a FIXED-WEIGHT term. The reward's weight is whatever the
position's own table happens to produce. The reward's total swing — the
range `base` can take between the best available intent and the worst, over
the production 300-state corpus (seed 123), read through the real
`expected_success` — measured on the v5 corpus:

| | CAM | CB | CDM | CF | CM | GK | LB | LW | RB | RW | ST |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **before** the scale | .0119 | .0090 | .0048 | .0144 | .0068 | .0059 | .0184 | .0120 | .0131 | .0114 | .0147 |
| **after** the scale | .628 | .634 | .508 | 1.237 | .624 | .778 | .939 | .527 | .838 | 1.644 | .720 |
| after / largest guard (0.10) | 6.3x | 6.3x | 5.1x | 12.4x | 6.2x | 7.8x | 9.4x | 5.3x | 8.4x | 16.4x | 7.2x |

**The guards were 5x to 20x THE OBJECTIVE.** That is the measured cause of
the v5 census: with the reward contributing 5-18% of what the guards
contributed, the GA optimised peakedness, and every fitness number stayed
unremarkable (0.60-0.67). After the scale the reward contributes **5x to
16x** the largest single guard for every position.

### THE FIX

`FitnessSurrogate.scales[position]` = the count-weighted std of that
position's **fitted** cells (not the raw samples — the shrinkage is what
makes the fitted spread differ, and the raw sd varies only 4.8x). Floor
`_SCALE_FLOOR = 0.002`; 1.0 when the position has fewer than
`_SCALE_MIN_CELLS = 3` cells; the `""` global row is skipped (it pools every
position, so no single scale is meaningful for it). Applied at READ time in
`expected_success` only, so the stored table stays in raw units and the raw
corpus can be refitted freely. It is a positive per-position constant, so
**the ranking of intents within a position is unchanged**.

### ⛔ THE VETO MUST BE SCALED TOO — THE PLAN'S SPEC WAS WRONG

`PLAN_2026-10-21.md` said: *"`FORBIDDEN_VALUE = -1.0` stays in scaled O(1)
space, so `expected_success` must return it undivided."* **That is wrong,
and it was wrong in the dangerous direction.**

`FORBIDDEN_VALUE = -1.0` was chosen because the RAW threat return's range is
about [-0.6, +1.0] — and on the real corpus the fitted cells run
**[-0.124, +0.137]**, so in RAW units the veto really is below every
legitimate reward. After the scale the cells spread much wider: the most
negative SCALED cell is **-7.59** (RB) and **59 cells across the corpus
fall below a fixed -1.0**. A veto that sits *above* a legitimate action is
not a veto — **it is a reason to perform the forbidden one**, and the GA
would have been handed a reason to cross. Dividing the veto with the same
scale as everything else restores the invariant **structurally**: every raw
reward is above -1.0, so every scaled reward is above `-1.0/scale`, which
is exactly where the scaled veto sits. Measured by
`tools/diag/_diag_forbidden_scale.py`.

### ⚠️ THE SCALE CREATED A PROBE BUG — AND THE NEW TEST PASSED FOR THE WRONG REASON

Both the balance probe and the new test excluded the veto from the
"which intents can this role perform" set **by VALUE**
(`if v == FORBIDDEN_VALUE: continue`). That worked while the veto was the
fixed -1.0 and stopped matching the moment it was scaled — so the veto
(-52.9 for the CAM) entered the min and the measured swing was inflated
**26x-66x**. The test then PASSED, for the wrong reason: it was asserting
against an inflated number. **A test that passes for the wrong reason is
worse than no test**, and only the probe's absurd magnitudes exposed it.
Fixed by excluding by **ROLE VOCABULARY** (`role_features.ROLE_INTENTS`),
which is what the GA chooses among — and which cannot accidentally exclude
a legitimate action that happens to scale below -1.0.

### THE OBJECTIVE IS NOT "ONE ACTION WINS EVERYWHERE" — the table has headroom

`tools/diag/_diag_intent_value.py` reads the surrogate the way the GA reads
it (through `expected_success`, with the family and global fallbacks) over
the production corpus, and reports per intent the share of states in which
it is the best available action, plus the **HEADROOM**: the per-state
oracle minus the best single-intent constant.

| pos | best constant | state-dependent oracle | HEADROOM | top intent's argmax share |
|---|---|---|---|---|
| CAM | +0.138 | +0.408 | **+0.271** | 22.7% |
| CB | +0.239 | +0.506 | **+0.267** | 28.0% |
| CDM | +0.198 | +0.621 | **+0.423** | 22.3% |
| CF | +0.522 | +1.274 | **+0.751** | 22.0% |
| CM | +0.345 | +0.818 | **+0.474** | 21.7% |
| GK | +0.336 | +0.965 | **+0.630** | 31.3% |
| LB | +0.179 | +0.611 | **+0.432** | 45.7% |
| LW | +0.089 | +0.375 | **+0.286** | 22.0% |
| RB | +0.321 | +0.693 | **+0.372** | 27.0% |
| RW | +0.317 | +1.116 | **+0.799** | 25.0% |
| ST | +0.254 | +0.658 | **+0.403** | 21.0% |

The best action wins in only **10-45%** of states, and the headroom is
+0.27 to +0.80 in scaled reward units — several times the largest guard.
**So the objective CAN express a state-dependent choice**; whether the GA
finds it is a different question, and the census is what asks it.

### THE 20-GENERATION SMOKE — BETTER, NOT SOLVED

One position (ST), 20 generations x 32 pop x 300 states, seed 123, the
production flags (`--role-features --perception --perception-roles`),
identical in every other respect:

| surrogate | fitness | intent mix (2,000 states) |
|---|---|---|
| v5 (no scale — the old behaviour) | 0.5251 | **RECYCLE 100.0%** |
| **v6 (scaled)** | 0.6974 | CARRY 79.7% · PROTECT_POSSESSION 20.2% · PROGRESSIVE_PASS 0.1% |
| shipped T1 (reference) | — | 9 intents, max share 35.0% |

**The v5 smoke reproduces the v5 census failure exactly (100% RECYCLE), and
the scale changes the outcome** — but the v6 brain is still a collapse by
the census criteria (2 intents, one at 79.7%). Fitness is NOT the gate and
the two numbers agree on nothing here.

### THE GATES

- **`tests/test_reward_scale.py` (18, NEW file).** The scale is computed
  from the fitted cells and is not 1.0 for a real spread; the value the GA
  reads is the raw value divided by the scale; a wide and a narrow position
  end up with the same O(1) spread; the ranking inside a position is
  unchanged; the veto is expressed in the position's units and is below
  every legitimate action; a two-cell position is left alone; the scale
  survives save/load; a surrogate saved WITHOUT a scale loads and does not
  divide; and — on the real corpus — the reward's swing beats the largest
  guard for **every** position, the scaled mean is far from the clip, and
  the veto is below every legitimate action.
- **`tools/diag/_diag_role_pooling_guard.py` — 12/12 CAUGHT**, file restored
  byte-identically (sha256), baseline green before and after. M10 removes
  the scale division, M11 the save, M12 the load. Every mutant must still
  **compile** (a syntax error counted as a catch proves nothing).
- **Probes:** `tools/diag/_diag_scale_spread.py` (the spreads, per position),
  `tools/diag/_diag_scale_balance.py` (the swing vs the guards — the gate),
  `tools/diag/_diag_forbidden_scale.py` (the veto),
  `tools/diag/_diag_intent_value.py` (what the table tells the GA, plus the
  headroom), `tools/diag/_diag_raw_positions.py` (the corpus's positions and
  per-position reward stats).
- **Combined gate: 182 passed, 1 skipped** (reward_scale, role_pooling,
  threat_reward, fitness_balance, surrogate_pairing, football_brain,
  brain_schema_alignment).
- **Free refit done:** `brains/surrogate_pos_v6.json` (3,696 pairs from
  `brains/surrogate_pos_v5.raw.json`, with the scales). **Nothing in
  `brains/` (the production T1 set) has been touched.**

### THE NEXT STEP

Item 1.3 — re-evolve all eleven positions (25-30 min, **needs CPU
permission**) against `brains/surrogate_pos_v6.json` into a scratch
`brains_v6/`, with the production config (80 gen x 32 pop x 300 states,
seed 123, role features + perception). Then item 1.4, the census
(`_diag_intent_mix_all.py brains_v6`), which is the gate: <=3 intents, any
action over 40%, or a role-specific action out of role = REJECTED,
**regardless of fitness**. Then item 1.5, the real-match gate: pass maps
and the eye, never fitness.

## THE CEILING — WHY THE v6 RETRAIN COLLAPSES ANYWAY (2026-10-21)

**THE VERDICT FIRST: the v6 set is REJECTED by the census. Every one of the
eleven positions collapses.** The scale fixed the balance; it did not and
could not fix the retrain. The reason is now measured rather than argued,
and it is NOT the scale and NOT the GA.

| pos | intents | max share | top intent | SHOOT | CROSS | shipped T1 |
|---|---|---|---|---|---|---|
| GK | 3 | 59.3% | SWITCH | 0.0% | 0.0% | 10 intents, top 34% |
| CB | **1** | **100.0%** | PROGRESSIVE_PASS | 0.0% | 0.0% | (schema mismatch) |
| LB | 3 | 95.0% | PROGRESSIVE_PASS | 0.0% | 0.1% | (schema mismatch) |
| RB | 3 | 82.4% | RECYCLE | 0.0% | 1.1% | (schema mismatch) |
| CDM | 2 | 81.6% | SAFE_PASS | 0.0% | 0.0% | 7 intents, top 36% |
| CM | 2 | 92.5% | CARRY | 0.0% | 0.0% | 7 intents, top 41% |
| CAM | **1** | **100.0%** | SAFE_PASS | 0.0% | 0.0% | 7 intents, top 35% |
| LW | 2 | 77.8% | SWITCH | 0.0% | 0.0% | (schema mismatch) |
| RW | 2 | 99.7% | RECYCLE | 0.0% | 0.0% | (schema mismatch) |
| ST | 3 | 61.5% | PROTECT_POSSESSION | **0.0%** | 0.0% | 9 intents, top 35% |
| CF | 3 | 61.4% | CARRY | 0.0% | 0.0% | 9 intents, top 31% |

**THE v6 SET IS WORSE THAN THE v5 SET ON THE COLLAPSE AXIS** (v5: 8 of 11 at
<=3 intents; v6: 11 of 11). And the reason is instructive: **the guards
were accidentally supplying diversity pressure.** With the reward dominant,
the GA follows the reward's preference exactly — and the reward's preference
is a single intent. So fixing the balance made the collapse PURER, which is
a fact about the objective, not about the search.

**THE SHIPPED T1 SET STAYS.** `brains_v6/` is REFERENCE-ONLY; its
`_manifest_all.json` records the configuration and the census (the parallel
runner's own manifest was lost when the server restarted mid-run; the three
wave-3 workers finished as orphans and wrote their brains, so the set is
complete). The "schema mismatch" in the last column is the shipped brains
being 31/32-wide against the current 32/33-wide schema — the 2026-10-05
sensor growth — so seven of the eleven shipped brains cannot even be
evaluated on the current role-width corpus, and their extra sensors are
INERT (the documented warn-once).

**THE SCALE WAS NECESSARY AND IS NOT SUFFICIENT.** This is the most
important section in this file for anyone about to spend another hour of
CPU on this workstream.

### THE COLLAPSE IS NOT THE SCALE, AND IT IS NOT THE GA EITHER

One position (ST), the production configuration (80 gen x 32 pop x 300
states, seed 123, `--role-features --perception --perception-roles`):

| surrogate | generations | intent mix (2,000 states) | fitness |
|---|---|---|---|
| v5 (no scale) | 20 | **RECYCLE 100.0%** | 0.5251 |
| v6 (scaled) | 20 | CARRY 79.7% · PROTECT_POSSESSION 20.2% | 0.6974 |
| **v6 (scaled)** | **80** | **CARRY 100.0%** | 0.7231 |
| shipped T1 (reference) | — | 9 intents, max share 35.0% | — |

**More generations make it WORSE.** The GA is not failing to find a rich
solution; it is converging on the best CONSTANT, and the constant is what
the objective prefers.

### THE HEADROOM IS REAL AND THE GA LEAVES ALL OF IT ON THE TABLE

`tools/diag/_diag_fitness_parts2.py` decomposes the fitness of an evolved
brain on the GA's OWN corpus (role width, perception on, seed 123) and
verifies its arithmetic against the real `synthetic_fitness` (it matched to
4 decimals):

| | mean_reward | conf | bdiv | penalty | base | FITNESS |
|---|---|---|---|---|---|---|
| the evolved v6 ST | +0.3076 | 0.971 | ~0 | 0.000 | 0.6538 | **0.6773** |
| the best CONSTANT | — | — | — | — | — | **0.6773** (identical) |
| the state-dependent ORACLE | — | — | — | — | — | **0.8499** |

**The oracle is worth +0.17 of fitness and the brain captured none of it.**
The reward's headroom now beats the guard budget (0.19) — so the balance is
right and the search is what fails.

**AND THE SHIPPED FOOTBALL BRAIN IS WORSE THAN THE CONSTANT ON THIS
OBJECTIVE.** The shipped T1 ST brain (7 intents, max share 30.3% — a broad,
sensible football mix) scores **0.5090** against the constant's **0.6773**,
with the oracle at 0.8068. So the objective does not merely fail to reward
good football; **it actively prefers "always carry" to good football.**

### WHY: THE TABLE'S ORDERING IS ONLY WEAKLY PREDICTIVE, AND THE STRIKER'S
### CLEAREST CHANCE IS NOT IN THE CORPUS AT ALL

The striker's best bucket — `P1FT1OH0S1C1G1` (final third, goal-close,
central, WITH SPACE) — reads, in the ST's own cells:

    PROTECT_POSSESSION   +0.1068   n=3     <- the top value in the whole ST table
    THROUGH_BALL         +0.0285   n=1
    PROGRESSIVE_PASS     +0.0228   n=1
    SAFE_PASS            +0.0195   n=2
    CARRY                +0.0193   n=2
    SWITCH               +0.0101   n=1

**THERE IS NO SHOOT CELL.** The striker's table says: in your clearest
chance, protect possession. And the table is **right about the corpus**,
because the corpus contains ~10 striker decisions in that bucket across six
matches and **not one shot**. The ε-exploration draws uniformly from the
role's vocabulary, so the striker's shots land wherever he happens to be;
with ten decisions in the situation that matters, uniform exploration puts
approximately zero shots in it.

**THE ARITHMETIC BEHIND IT.** 3,696 pairs spread over 11 positions x 64
buckets = 704 buckets, i.e. **5.25 pairs per bucket and ~2.5 per
(bucket, intent) cell**. `fit` shrinks every cell with fewer than
`min_samples` samples toward its bucket mean, so **65% of the cells are 75%
bucket mean** and the ordering the GA is asked to learn is mostly the
bucket's average.

### THE HELD-OUT TEST, DONE PROPERLY (and two of my own instruments were wrong)

Fit on half the corpus, judge on the other half, and compare the table's
preferred intent with what was actually realised:

| | buckets judged | A's preference is B's best | chance | what following A cost |
|---|---|---|---|---|
| 6 features (current) | 43 | **37.2%** | 30.8% | -0.00299 |
| 5 features | 42 | 23.8% | 28.8% | -0.00363 |
| 4 features | 38 | 18.4% | 28.0% | -0.00529 |
| 3 features | 27 | 29.6% | 20.2% | -0.00684 |
| 2 features | 28 | 21.4% | 22.2% | -0.00628 |

**Two of my instruments were wrong and are recorded because both produced a
clean, confident, false answer:**

1. **The FIRST held-out test read the TABLE's own values on held-out
   states.** A table's value for a bucket does not change when the state
   changes, so "held out" measured the bucket composition and nothing else.
   **A held-out test that reads the model's own output is not a held-out
   test.** The version above reads the REALISED reward.
2. **The `paid` column's sign was backwards in its own docstring.** The
   comparison is against the best realised mean, so `paid` is always <= 0
   and **more negative means WORSE**, not better. The first write-up read it
   the other way round.

**AND THE BUCKET-RESOLUTION HYPOTHESIS WAS TESTED AND REJECTED.** The
arithmetic above says the buckets are too fine for the corpus, so coarser
schemes should learn more. They do not: the current 64-bucket scheme is the
best of the five. **The measurement killed the hypothesis, and the plan's
"do not change the shrinkage" stands** — the shrinkage is not the defect,
the corpus's COVERAGE is.

### SO: THE CEILING IS THE CORPUS'S COVERAGE OF THE DECISIONS THAT MATTER

The scale was necessary and is done. It is not sufficient, because the
objective can only teach what the corpus contains, and the corpus contains
almost no striker shots in the box.

**THE LEVER IS THE COLLECTOR, NOT THE FIT.** The ε-exploration already
replaces a decision with a random action the ROLE can perform; what it does
not do is look at the SITUATION. The sensors are already computed a few
lines below the exploration call in `_snapshot_decide`, so the bucket is
available. A situation-aware exploration — bias the explored action toward
the actions the bucket can be judged on (SHOOT for the striker in a
final-third goal-close bucket, CROSS for the wings, THROUGH_BALL for the
midfielders), and/or raise the exploration rate in the final third — would
put the missing shots INTO the buckets that matter. Cost: one collection
(12-16 min), a free refit from the raw corpus, and the evolve.

**NOT CHANGED, ON PURPOSE:** the shrinkage (the plan's instruction stands
and the resolution test supports it), the reward definition, the production
brains in `brains/`, and the collector's exploration policy (that is the
proposal, not a decision this session took).

### THE PROPOSAL IS BUILT, OFF BY DEFAULT, AND AWAITS A DECISION

`surrogate_collect.py` now has a **situation-aware** ε-exploration,
`_EXPLORE_SITUATIONAL` (**default False**) and the CLI flag
`--explore-situational`:

- the explored action is drawn with the PHASE's weights
  (`_SITUATION_ACTIONS`: a final-third situation samples a shot, a cross or
  a ball in behind; an open situation a switch or a carry; your own half a
  safe pass or a recycle) — and the weights are a table of what a situation
  AFFORDS, the same kind of object as `role_features.ROLE_INTENTS`, not a
  table of what happened to happen;
- the exploration RATE is raised in the final third
  (`_EXPLORE_EPS_FINAL_THIRD = 0.75`), because with ~10 decisions in the
  situation that matters, uniform exploration puts approximately zero shots
  in it even when the shot is in the vocabulary;
- **the ROLE vocabulary is untouched**: the striker is still never asked to
  cross and the keeper is still never asked to shoot;
- the sensors are now computed BEFORE the exploration (they are the same
  vector either way, and `perceive` draws from a SEEDED generator, so the
  reorder does not move the global RNG stream — pinned by a test).

`tests/test_situational_exploration.py` (11 tests, new file) pins the switch
being off by default, the unchanged uniform path, the role vocabulary, the
phase weights, the raised rate, the phase agreeing with the surrogate's own
bucket key, and the RNG-stream freedom of the reorder.

> **AND THE NEGATIVE CONTROL CAUGHT TWO OF ITS OWN TESTS — the fourth
> recorded instance of the same defect.** `tools/diag/_diag_situational_guard.py`
> (5 mutations, sha256-verified restore) reported **MISS, MISS, CAUGHT,
> CAUGHT, CAUGHT** on its first run. The two misses were M1 (the phase
> weights emptied) and M2 (the rate not raised), and both misses were **the
> tests, not the mutations**: they compared two COUNTS (the shots drawn in
> the final third vs the shots drawn in the own half) and a comparison
> between two counts at the same rate is **a coin flip** — 400 draws of a
> 1-in-9 share sit inside each other's noise, so the test passed with the
> mechanism mutated out. **A threshold sitting inside the measured noise
> band is not a threshold.** The fix: assert SHARES (the shot at >30% of
> the explored draws in the goal-close phase, and 0% in the own half) and a
> RATIO (the attacking phase explored more than 3x the own half), never an
> ordering. One more trap in the same family: the test's "original intent"
> was SAFE_PASS, and because `PlayerIntent` members are singletons an
> explored SAFE_PASS is indistinguishable from an un-explored one — so the
> original is DRIBBLE, which is in neither phase's option list.
>
> **5/5 CAUGHT** after the fix, file restored byte-identically, baseline
> green before and after.

**WHAT IT COSTS AND WHAT IT IS GATED ON:** one collection (12 matches, ~24-30
min — the v5 corpus was 6 matches and 12-16 min), then a free refit, then
the evolve (25-30 min). **The gate is the corpus census, not the fitness:**
the striker's SHOOT must appear in a final-third bucket with n >= 3, and the
midfielder's THROUGH_BALL in a final-third bucket too. **Do not skip to the
evolve** — a refit of a corpus that still has no striker shots in the box
produces the same collapse with a higher fitness number, which is exactly
the T2G160 failure mode.

### THE 2-MATCH PROBE — THE MECHANISM WORKS

`brains/surrogate_pos_v7_probe.raw.json` (2 matches, seed 123, the same
opponent styles as the v5 corpus, ε 0.25 + `--explore-situational`), 1,216
pairs, 561 explored:

| | v5 corpus (6 matches, uniform exploration) | **v7 probe (2 matches, situational)** |
|---|---|---|
| the striker's final-third decisions | 31, **0 shots** | 9, **6 shots** |
| the striker's SHOOT cell in a final third | **n = 0** | **n = 4, mean +0.00155** |
| the midfielder's THROUGH_BALL cell in a final third | n = 4, **mean exactly 0.0** | n = 8, mean +0.00502 |
| the holding midfielder's THROUGH_BALL cell | n = 2 | n = 6 |
| the full-backs' CROSS cells | LB n = 7, RB n = 3 | LB n = 10, RB n = 8 |

**The holes that remain in the probe are SAMPLE-SIZE holes, not mechanism
holes:** the CAM had 6 final-third decisions in two matches (so 1 shot), the
LW 5 (1 cross) and the RW 4 (0 crosses, 3 shots). Twelve matches is six
times the probe, which is what the census needs.

**AND THE "worth exactly 0.0" RED FLAG IS GONE** for the midfielder's
through ball (n = 8, mean +0.00502): the `goal_close` phase only samples the
action where the geometry can take it, so the refused-strike marker carries
— worth exactly 0.0 — no longer fill the cells that matter. **That was the
second reason the `goal_close` phase exists, and it is measured.**

**WHAT THE PROBE DOES NOT SAY:** the 12-match corpus is what the census
must clear. The probe's job was to prove the mechanism, and it does.

### THE CORPUS CENSUS — THE HOLE, COUNTED (and it is the gate on a new corpus)

`tools/diag/_diag_corpus_bucket_coverage.py` reads a RAW corpus and asks the
per-BUCKET question (the older `_diag_corpus_coverage.py` asks the
per-INTENT one, which was the right question when the hole was "one shot in
4,011 decisions" and is the wrong question now). On the v5 corpus, six
matches, ε-exploration 0.25:

| position | intent | final-third buckets | best n | mean reward | verdict |
|---|---|---|---|---|---|
| **ST** | SHOOT | 7 | **0** | — | **MISSING** |
| **CAM** | SHOOT | 5 | **0** | — | **MISSING** |
| CF | SHOOT | 5 | 1 | +0.0268 | MISSING |
| **CDM** | THROUGH_BALL | 7 | 2 | +0.0999 | MISSING |
| **CM** | THROUGH_BALL | 14 | 4 | **+0.00000** | **RED FLAG: worth exactly 0.0** |
| RW | CROSS | 9 | 2 | -0.0077 | MISSING |
| LB | CROSS | 16 | 7 | +0.0120 | OK |
| LW | CROSS | 8 | 5 | +0.0046 | OK |
| RB | CROSS | 15 | 3 | +0.0000 | OK |

**THE STRIKER HAD 31 FINAL-THIRD DECISIONS IN SIX MATCHES AND NOT ONE SHOT.**
His final third was: PROTECT_POSSESSION 6, SAFE_PASS 5, DRIBBLE 5,
PROGRESSIVE_PASS 4, RECYCLE 3, SWITCH 3, CARRY 3, THROUGH_BALL 2 — and no
shot. The CAM had 12 final-third decisions and no shot; the CF had 15 and
one.

**Why the ε-exploration did not fix it, in arithmetic:** 25% of the striker's
decisions were replaced, his final third held 31, so ~8 were explored, and
SHOOT is 1 of the 9 actions in the striker's vocabulary — an expected
**0.9 shots, observed zero.** The uniform exploration puts the rare actions
wherever the player happens to be, and the situations that can JUDGE them
are the rare ones.

**AND THE "worth exactly 0.0" RED FLAG is its own artefact:** the CM's
through-ball cell in a final-third bucket has four samples and their mean
reward is exactly zero. A through ball the geometry refused becomes a
zero-length marker carry, which `threat_return` scores at precisely 0.0. So
a corpus that samples the action where it cannot be performed teaches the
table that the action is worth nothing — the "ignorance is attractive"
defect wearing a different hat. **This is why the situational exploration
has a `goal_close` phase of its own:** a shot is only sampled where the
geometry can take it.

**THE PREDICTION THE NEW CORPUS MUST MEET (and the gate):** the striker's
SHOOT cell in a final-third bucket with n >= 3, the midfielders' THROUGH_BALL
in a final-third bucket with n >= 3, and a SHOOT cell whose mean reward is
**not** exactly 0.0. The arithmetic says it will work: at the new rate and
weights the striker's ~31 final-third decisions (62 at twelve matches)
become ~23 shots, against 0 today.

### THE v7 CORPUS, THE GATE PASSED, AND THE TABLE CAN NOW SAY "SHOOT"

`brains/surrogate_pos_v7.raw.json` — **12 matches**, seed 123, the same three
opponent styles as the v5 corpus, ε 0.25 **+ `--explore-situational`**,
**7,811 pairs** (3,737 explored, 48%).

**THE CORPUS CENSUS PASSES.** Merged with the v5 corpus (11,507 pairs — the
merge is FREE and it fills the one thin cell):

| position | intent | best final-third cell | mean reward | verdict |
|---|---|---|---|---|
| **ST** | SHOOT | **n = 16** | **+0.01479** | OK |
| CF | SHOOT | n = 3 | +0.00893 | OK |
| CAM | SHOOT | n = 4 | +0.01159 | OK |
| CM | THROUGH_BALL | n = 83 | -0.00190 | OK |
| CDM | THROUGH_BALL | n = 28 | +0.02055 | OK |
| LW | CROSS | n = 8 | +0.00221 | OK |
| RW | CROSS | n = 6 | -0.00117 | OK |
| LB | CROSS | n = 34 | -0.00294 | OK |
| RB | CROSS | n = 42 | +0.01986 | OK |

**THE STRIKER'S TABLE CAN NOW SAY "SHOOT".** The bucket that used to read
`PROTECT_POSSESSION +0.1068` with no shot in it now reads:

    P0FT1OH0S0C1G1  (final third, goal-close, central, no space)
        SHOOT              +0.0125   n=16     <- the top value by SEVEN TIMES
        SWITCH             -0.0018   n=2
        PROGRESSIVE_PASS   -0.0022   n=5
        CARRY              -0.0023   n=1
        RECYCLE            -0.0024   n=2
        THROUGH_BALL       -0.0909   n=3

**THE MERGE FIXED THE ONE THIN CELL, AND WHY IT MATTERED.** The CF's own
final-third shot cells sat at **n = 2 and worth exactly 0.0** in the v7
corpus alone: every one of those shots was **refused by the engine's shot
gate**, so each was recorded as the zero-length marker carry. The v5 corpus
holds ~15 more CF final-third decisions, and those shots FIRED — so the merged
cell is n = 3 at **+0.00893**. **A refused strike and a taken strike are
different data, and the refusal is an artefact of where the player stood, not
of the action.** (`tools/diag/_diag_merge_corpus.py` — free, and it prints the
same cells the census prints.)

**THE BALANCE HOLDS ON v7:** the reward's swing is **4.2x-19.6x the largest
guard** for every position (CAM 12.9x, CB 6.2x, CDM 4.2x, CF 13.8x, CM 6.7x,
GK 10.3x, LB 9.0x, LW 8.0x, RB 9.3x, RW 19.6x, ST 13.1x).

### ⛔ AND THE RETRAIN STILL COLLAPSES — BECAUSE THE SEARCH IS THE BLOCKER
Two more single-position runs on the v7merged surrogate, the production
configuration, 80 generations:

| run | the striker's intent mix | census |
|---|---|---|
| default (per-brain corpus, mutation strength 0.3) | **PROTECT_POSSESSION 100.0%** | FAIL |
| mutation strength 0.05 | PROTECT_POSSESSION 68.6% · CARRY 31.4% | FAIL (2 intents) |
| one corpus per generation (shared) | CARRY 55.9% · PROTECT_POSSESSION 44.1% | FAIL (2 intents) |

**Both GA fixes help and neither is enough.** The two changes are real
defects and both are kept: `mutation_strength` (0.3 perturbs 15% of the
weights by 30% of their own size — a RANDOM WALK, not a hill climb) and
`shared_corpus` (every brain used to be evaluated on its OWN corpus, so the
selection compared fitnesses across DIFFERENT samples, and the
corpus-to-corpus noise (~+-0.02 of fitness) is the same order as the
differences the search has to resolve).

**BUT THE DECISIVE NUMBER IS THE OBJECTIVE'S OWN OPTIMUM.** The surrogate
defines a policy — the per-bucket argmax — and that policy is EXACTLY what
the census wants:

| the striker's | CARRY | PROTECT | RECYCLE | DRIBBLE | **SHOOT** | SAFE | PROG | SWITCH | THROUGH |
|---|---|---|---|---|---|---|---|---|---|
| the table's argmax share | 24.0% | 19.7% | 18.3% | 11.3% | **11.0%** | 4.7% | 4.3% | 4.3% | 2.3% |

**NINE intents, a maximum share of 24%, and the striker SHOOTING 11% of the
time** — comfortably inside the census (which rejects <=3 intents, >40% share
and role-inappropriate actions). And the oracle is worth **+0.44 to +0.51**
(scaled) over the best constant, i.e. **+0.22 to +0.25 of fitness**, which
**beats the whole guard budget (0.19)**.

**SO: THE OBJECTIVE IS NOT THE CEILING. THE TABLE'S GLOBAL OPTIMUM IS A
BROAD, SHOOTING, NINE-INTENT MIX WORTH +0.44 OVER THE CONSTANT — AND THE GA
FINDS THE CONSTANT AND STOPS.** The constant is a LOCAL optimum that a bias
tweak reaches in one step; the useful state-dependence is a specific
function (a threshold rule over six bucket features) that a blind search from
a random initialisation does not reach. That is not a tuning problem: a
random net's state-dependence is USELESS (its argmax is random), so selection
kills the state-dependent members and keeps the constant.

**THE TWO FIXED HALVES AND THE REMAINING HALF:**
- the per-position reward SCALE — the balance, done and gated;
- the corpus COVERAGE — the table can now say "shoot", done and gated;
- **the SEARCH — the last half, and it is a DESIGN decision, not a
  calibration.**

### ✅ THE THIRD HALF — THE DISTILLATION, AND THE FIRST SET THAT PASSES THE CENSUS

**THE USER'S DECISION (2026-10-21): distillation, then GA refinement.**

`brain_distill.py` — the net is fitted to **the surrogate's own policy**
(its per-bucket argmax over the ROLE's vocabulary) by cross-entropy
(numpy Adam, `FootballBrain.forward_logits` was split out of `forward` so the
gradient can be written in terms of the logits). **It is NOT a new
objective**: the target comes from the same surrogate, on the same corpus,
with the same reward — it is a different way to reach the optimum the
objective already defines.

**THE COST IS THE HEADLINE: 2 SECONDS PER POSITION** (2,000 states, 120
epochs, 99% agreement) against the GA's **4 minutes** for a brain that
collapsed to one action. The whole eleven positions distill in **20
seconds**; distil + refine (80 generations, warm-started) took **2.6
minutes**.

**THE AGREEMENT WITH THE TABLE IS 97.4%-99.1%** for every position, and the
resulting intent mix IS the table's mix.

**AND THE SET PASSES THE CENSUS — THE FIRST IN THE PROJECT'S HISTORY:**

| pos | intents | max share | top intent | SHOOT | CROSS | shipped T1 |
|---|---|---|---|---|---|---|
| GK | 7 | 19.8% | THROUGH_BALL | 0.0% | 0.0% | 10 intents, 34% |
| CB | 7 | 30.9% | RECYCLE | 0.0% | 0.0% | (schema mismatch) |
| LB | 8 | 22.0% | THROUGH_BALL | 0.0% | 9.6% | (schema mismatch) |
| RB | 9 | 25.1% | SAFE_PASS | 0.0% | 12.7% | (schema mismatch) |
| CDM | 8 | 26.0% | RECYCLE | 0.0% | 0.0% | 7 intents, 36% |
| CM | 8 | 27.5% | RECYCLE | 0.0% | 0.0% | 7 intents, 41% |
| CAM | 8 | 17.9% | PROGRESSIVE_PASS | 14.9% | 0.0% | 7 intents, 35% |
| LW | 8 | 22.6% | SHOOT | 22.6% | 15.7% | (schema mismatch) |
| RW | 8 | 24.6% | SHOOT | 24.6% | 15.5% | (schema mismatch) |
| **ST** | **9** | **19.7%** | RECYCLE | **17.6%** | 0.0% | 9 intents, 35%, **SHOOT 0.0%** |
| CF | 8 | 25.4% | SAFE_PASS | 9.1% | 0.0% | 9 intents, 31% |

`brains_v7/` — **no collapse flagged: every position uses >4 intents, no
action over 40%, and no role-specific action out of role.** And it is
BROADER than the shipped T1 at every position, and **the striker SHOOTS
17.6% of his decisions where the shipped one never shoots at all.**

**THE REFINEMENT IS THE GA, WARM-STARTED.** `brain_evolution.evolve` grew an
`init_brain` parameter: the population becomes that brain plus small
mutations of it, so the search is a local refinement of a policy that is
already good. With `--mutation-strength 0.05` the striker's mix went from
SHOOT 16.6% to **17.9%** and the max share from 19.9% to 19.8% — a small
improvement. **Its value is NOT proven**: the fitness SATURATES at 1.0 for
four of the eleven positions (GK, RW, ST, CF), so the GA's selection is
blind above the clip. **The census, not the fitness, is the gate** ✓ and the
census passes ✓.

**A CENSUS RULE WAS WRONG, AND IT WAS FLAGGING GOOD FOOTBALL.** The
role-inappropriate rule was a hand-written list of positions that "should
not shoot" and it listed **the CAM** — but the CAM is the attacking
midfielder, `ROLE_INTENTS["AM"]` includes SHOOT and `ROLE_ONLY_INTENTS`
puts SHOOT with {ST, AM, WING}: the AM family performed SHOOT in the
corpus, so a CAM shooting is role-appropriate. The rule is now DERIVED from
the role vocabulary, and the CAM's 14.9% of shots is no longer flagged.

### THE INSTRUMENTS THIS SECTION USES (all read-only, all in `tools/diag/`)

- `_diag_scale_balance.py` — the reward's swing against the guards (the
  gate). `[surrogate]` = a fitted file or a raw corpus.
- `_diag_scale_spread.py` — the per-position spreads, fitted and raw.
- `_diag_forbidden_scale.py` — the veto's position in the scaled space.
- `_diag_intent_value.py` — what the table tells the GA, per intent, on the
  GA's OWN corpora (`--ga`), plus the headroom.
- `_diag_fitness_parts2.py` — the fitness decomposition (replaces the stale
  `_diag_fitness_parts.py`, which reproduces the OLD formula), with the
  best-constant and oracle columns and a replication check against the real
  `synthetic_fitness`.
- `_diag_headroom_predictive.py` — the held-out test on REALISED rewards.
- `_diag_bucket_predictive.py` — does the table's per-bucket preference
  predict the realised outcome?
- `_diag_bucket_resolution.py` — the bucket-resolution sweep (the rejected
  hypothesis).
- `_diag_corpus_bucket_coverage.py` — **the gate on a new corpus**: the
  per-BUCKET coverage of the actions that matter, with the "worth exactly
  0.0" red flag.

## THE FANS — THE HAND-TUNED FILL RATE, REPLACED BY A SOURCED MODEL (2026-10-21)

The user: *"the fill rate or fans are just hand tuned numbers, that's bad, i
think the fans should be scaled based on city populations or something
realistic... why don't we apply that [the level of detail in this project]
to fans, not render them visually, but something better than what am seeing
right now."*

**THE DIAGNOSIS WAS RIGHT, AND IT WAS MEASURABLE.** The crowd was

    attendance = stadium_capacity × fill_rate

with `fill_rate` read from a HAND-TUNNED table keyed on a "match tier":
derby 0.95, big6_vs_big6 0.97, big6_home 0.90, big6_away 0.85, standard
0.78, low_interest 0.62 — plus a squad-quality term, a kickoff-time term,
a weather term, a standings term, and a random residual of **±4-6% of the
CAPACITY**.

Measured over PLOFA's 26/27 season (63 matches, `_diag_attendance_now.py`):
**mean occupancy 85.0%**, range 57.7%-98.5%, and **seven clubs pinned at the
98.5% ceiling on every one of their home matches** — the signature of a
saturating table, not a crowd. Red Wolves drew 43,685 into a 70,000-seat
ground (62.4%); Claw drew 85,695 into 87,000 (98.5%) on all three of its
home matches.

### THE BENCHMARK — Transfermarkt, Premier League 24/25, accessed 2026-10-21

**15,380,351 spectators over 380 matches → 40,474 per match, 429 sell-outs,
96.8% mean occupancy.** The per-club table (capacity, average attendance,
occupancy) is IN THE REPO in `attendance_model.PL_24_25` — twenty clubs,
Old Trafford 74,158/73,815/99.5% down to Vitality 12,357/11,200/90.6% — and
the gate checks that every row's occupancy IS its own attendance over its
own capacity, that the stated league mean is the total over the matches, and
that 96.8% is what the twenty rows average to. **A benchmark in the code
that the code checks is not a number in a comment.**

### THE STRUCTURE THE BENCHMARK REVEALS

**In a top-flight league the crowd IS the ground.** Sixteen of the twenty
clubs sit between **95.3% and 99.9%**. The two that do not are the two real
cases in the table, and they are the reason the model has the shape it has:

- **Everton 74.3%** — a new 52,769-seat ground for a club whose market
  could not fill it. **A ground can outgrow its market.**
- **Manchester City 87.8%** — a deliberately oversized ground.

So the model is: **the ground sets the ceiling; the club's own standing
decides how close to it the club gets; the match decides where in that
range the crowd lands.**

### ⛔ AND THE CITIES DO NOT SPLIT THE CROWD — the answer to "scale by city population"

This is the natural question and the real data answers it. **Seven of the
twenty clubs are London clubs** — Arsenal, Tottenham, Chelsea, West Ham,
Fulham, Brentford, Crystal Palace — and between them they drew **256,000
spectators a matchday out of one city**, with **every one of them between
95.4% and 99.9% full**. Liverpool and Everton share Merseyside (98.7% and
74.3%); Manchester United and Manchester City share Manchester (99.5% and
87.8%).

**A shared city does not split a crowd**: each club has its own support
base, and in a saturated league the demand is far above the supply. So a
per-city population would add **a hand-tuned number without adding
realism** — and the project has no population data anyway (the 1997 club
register gives city, district, founding year, ground, capacity, kit, style,
sponsor and budget class; there is no population field, and the cities are
fictional, so any population would be invented).

**WHAT THE CITY DATA *IS* GOOD FOR, and it is still real:** the register's
city field makes a shared market explicit — **three clubs in "Pearls Lake
Town"** (Pearls, Triumpher, Club Chovers), two in "Oxland City" (Oxton, Tryox
City), two in "Uditon City", two in "Old Lige Road City" — which is exactly
real football's geography and exactly what the London refutation says does
NOT split the crowd. The world layer's `City.population` field stays
available for the metric it is genuinely good for (**fans per 1,000
inhabitants**), which this project cannot compute until the populations are
authored. **That is a DATA decision, not a modelling one** — and it is the
user's to make.

### THE MODEL — `attendance_model.py` (new, pure, no engine imports)

    market_score(squad_quality, position_score)   → 0..1
    base_occupancy(score)                         → the club's standing occupancy
    attendance_for(capacity, home_score, away_score, is_derby,
                   match_multiplier, residual)     → the crowd

- **the ground is the ceiling** (exactly as in the benchmark — no club in the
  Premier League draws more than its stadium holds);
- the occupancy is centred on the benchmark's **96.8%** and moves with the
  club's standing by ±3 points (the benchmark's own spread among the
  sixteen "typical" clubs is 4.6 points);
- **the opponent's drawing power is an ABSOLUTE term** (±1 point) — the
  honest replacement for the old hand-written `big6_home` 0.90 / `big6_away`
  0.85 rows, which were a table standing in for exactly this one effect.
  The first version of the model made it RELATIVE to the host, and the
  gate caught what that does: a bottom club hosting the same opponent drew
  MORE than a leader hosting it. The travelling support of a big club is a
  fact about that club, not about who it is visiting;
- the kickoff-time, weather and form multipliers are unchanged (they come
  from `weather_physics` and the standings);
- the residual is now **±1.2% of the CROWD**, not ±4-6% of the capacity;
- **the ceiling is deliberately out of reach** (0.968 + 0.030 = 0.998 <
  0.999), because the old model's ceiling was reachable and seven clubs sat
  on it.

### THE CENSUS — the model's distribution against the real one

`_diag_attendance_census.py`, per club, from the season's own inputs (the
recorded ground, the warehouse's final standings, the real squad's average
DNA rating), with the OLD numbers — the recorded attendance, i.e. the thing
the user was looking at — beside the new:

| | OLD (recorded) | NEW (model) | REAL PL 24/25 |
|---|---|---|---|
| mean occupancy | **85.0%** | **96.8%** | **96.8%** |
| median occupancy | 84.4% | 96.9% | 98.0% |
| min occupancy | 62.4% | 94.9% | 74.3%\* |
| max occupancy | 98.5% | 98.3% | 113.2%\* |
| clubs sitting on the clamp | 1 | **0** | 0 |
| mean attendance per club | 44,518 | 49,767 | 40,474 |

\* the real extremes are Everton's over-built new ground and Villa Park's
published capacity (which excludes standing areas; the club's real capacity
is ~42,918, putting them at ~97%). The benchmark's artefact is NAMED in
the code (`CAPACITY_ARTEFACT_CLUBS`) so anything asserting "attendance
never exceeds capacity" excludes it deliberately.

**The per-club change is the visible one** — Oxton 33,590 → 43,272 (74.6% →
96.2%), Red Wolves 43,685 → 68,374 (62.4% → 97.7%), Play City 60,440 →
76,929 (75.5% → 96.2%), Claw 85,695 → 85,014 (98.5% → 97.7%, i.e. it came
OFF the clamp). **The mean attendance per club is above the real PL's
40,474 because PLOFA's grounds are bigger (mean 51,300 vs the PL's ~38,000)
— that is the world's grounds, not the model.**

### A DEAD SIGNAL FOUND ON THE WAY: the "bigness" proxy was inert

The old model scaled the crowd by the squad's average rating:
`(avg_rating - 65) / 100`. Measured across the 26/27 squads, the average
DNA rating runs **67.7 to 74.2** (median 70.3) — a 6.5-point spread on a
100-point scale — so that term moved the fill rate by **0.077 to 0.092**:
**less than two percentage points, i.e. effectively nothing.** The rating
is a weak market signal and the league position carries more of it, which
is why `market_score` weights them equally and the gate pins that the
rating's contribution stays weak.

### THE GATES

- **`tests/test_attendance_model.py` (16, NEW file)** — the benchmark's
  internal consistency (three ways); the London refutation; the ground as
  the ceiling; the model centred on the benchmark; **non-vacuity**: a leader
  must outdraw a bottom club; the model's spread against the benchmark's
  band; **the ceiling must be unreachable**; the rating's weakness; and
  three STRUCTURAL guards that the old table is gone (no `fill_rate_map`,
  and no tier→fraction mapping under any name).
  - **Two of those guards were wrong on their first run and the corrections
    are the interesting part.** The first banned the old tier fill rates as
    float LITERALS and fired on `random.uniform(0.90, 1.10)` — the TICKET
    PRICE variation, which has nothing to do with the crowd. The second
    banned any mapping keyed by the tier names and fired on
    `TIER_TICKET_PRICES`, which is the legitimate PRICING table (the same
    tiers, but the values are currency). **A guard that fires on unrelated
    code gets switched off, and then it fires on nothing** — so the guard
    now checks BOTH the tier names AND that the values are occupancy
    fractions.
- `tests/test_attendance_performance.py` (7) — **unchanged and green**, which
  is the point: the zero-regression property (no standings → exactly
  neutral), the leader > bottom, hot > cold form, the away side's smaller
  nudge, the modifier's smallness and the bounds all still hold.

### WHAT IS STILL HAND-TUNNED, STATED RATHER THAN HIDDEN

**`TIER_TICKET_PRICES` is still a hand-written table** (derby 65.00 →
low_interest 28.00). It is a PRICING model, which is a different question
from the crowd, and it was not part of the complaint — but it is the same
kind of number and it is now the most obviously hand-tuned thing left in
the financials. The honest position: **the prices are a separate
calibration with a separate benchmark (real Premier League average ticket
prices, revenue per spectator) and it has not been done.** Do not let the
crowd's new realism imply the revenue is real.

**AND THE SEASON'S RECORDED NUMBERS ARE STILL THE OLD ONES.** The model
changes what a NEW export produces; the 63 matches already in the warehouse
keep the old attendance, because they are history. Re-running the season's
matches would rewrite them (and the revenue with them) — that is a
decision, not a side effect.

**THE FANS / ATTENDANCE (2026-10-21) — read before touching the crowd, the
fill rate, the attendance, or the revenue:**
- `attendance_model.py` (NEW) — the sourced model and the benchmark in the
  same file: `PL_24_25` (the real Premier League 24/25 per-club table),
  `PL_MEAN_OCCUPANCY_24_25 = 0.968`, `PL_MEAN_ATTENDANCE_24_25 = 40_474`,
  `CAPACITY_ARTEFACT_CLUBS` (Villa Park's stated capacity excludes standing
  areas), and the three functions `market_score` / `base_occupancy` /
  `attendance_for`. **Pure, no engine imports** — so it can be tested and
  calibrated on its own.
- `exporter.py` — `MatchFinancials.compute` now calls the model; the
  hand-tuned tier fill-rate table is GONE; `MIN_FILL_RATE`/`MAX_FILL_RATE`
  are the model's bounds; `_season_score`/`_form_score` are class helpers
  (they were closures inside `compute`). **The TICKET PRICE table
  (`TIER_TICKET_PRICES`) is still hand-written — a separate calibration
  that has not been done.**
- `tests/test_attendance_model.py` (16, NEW) — the benchmark's
  self-consistency, the London refutation, the ground as the ceiling, the
  non-vacuity, and the structural guards that the old table is gone.
- `tests/test_attendance_performance.py` (7) — the pre-existing suite,
  unchanged and still green (the zero-regression properties).
- Probes: `_diag_attendance_now.py` (what the season recorded),
  `_diag_attendance_census.py` (the model's distribution against the real
  benchmark, per club), `_diag_market_signals.py` (the squads' rating
  distribution — how the inert "bigness" proxy was found),
  `_diag_club_market_data.py` (what per-club data the 1997 register holds,
  and what it does not).

## THE TIMELINE'S DENSITY — THE FLOW IS REAL, THE PACE INSIDE A POSSESSION IS NOT (2026-10-21)

The user, reviewing the export fix: *"the engine's times are a model, not a
stopwatch... it means what? football wise? does it compromise the new flow?"*

**MEASURED (`_diag_timeline_football.py`, one real match).**

| | PLOFA | real football |
|---|---|---|
| pass-like events in a match | 797 | ~1,050-1,150 (both teams, PL) |
| MEAN interval between passes | **7.09 s** | **~5 s** ✓ close |
| MEDIAN interval between passes | **1.10 s** ✗ | 3-5 s ✗ |
| p90 interval between passes | **40.87 s** ✗ | ~15 s ✗ |
| the first 26 events of the match | **7.04 s** ✗ | ~60 s ✗ |
| a possession's internal spacing | 0.3-0.6 s per action ✗ | 2-4 s ✗ |
| the gap after a possession ends | 40-115 s ✗ | 10-20 s ✗ |

**SO: THE TOTAL IS RIGHT AND THE DISTRIBUTION IS WRONG.** 797 passes at 7.1 s
each is football-plausible; the same 797 passes bunched into 1.1-second
spacings with 40-115 second silences between possessions is not.

**AND THE CAUSE IS THE ENGINE'S CLOCK, NOT THE EXPORT.** A "chain" is the
engine's unit of work (a possession, a set piece, a shot sequence) and its
window is the clock advance during it — which is the **BALL's motion** ✓.
So the chain's off-ball actions (the press, the duel, the reception, the
decision) are resolved *inside the ball's flight time*: a 30 m pass's ~1.2 s
flight carries a PASS, a MISCONTROL, a PRESS and the next PASS ✗✗. And
`chronograph()` then spreads even a long chain evenly across its events, so
a chain of four events in 1.2 s puts them 0.3 s apart ✗.

**Real football's possession is flight + decision + execution** — a player
takes 0.5-1.5 s to decide what to do with the ball, and the off-ball events
happen over those seconds, not inside the flight ✗. **That is the missing
time, and it is the engine's clock semantics, not the exporter's
presentation.**

**DOES IT COMPROMISE THE NEW FLOW? NO — the flow is a different property.**
- The ORDER is real ✓ (the events are appended in chain order and the
  chronology has **0** out-of-order steps ✓).
- The times are monotonic and the inter-chain gaps are the clock's own ✓.
- The export no longer lies about WHEN an event happened — the timestamps
  are the engine's truth and they agree with the engine's own chronology ✓.

**What it compromises is the DENSITY** ✗ — and that is exactly what a
watcher's eye catches next: the opening of the export reads as a 7-second
scrum followed by a 53-second silence, where a real match's first 26
actions take a minute or two ✗. **The flow is fixed; the pace is the next
defect in this thread, and it is an engine change, not an export change.**

**THE FIX, and its cost.** The chain's window must cover the football's own
durations rather than the ball's motion alone: the ball's flight ✓ (already
there) **plus the receiver's decision time** ✓ (0.5-1.5 s, from the same
football literature this project cites for its other calibrated quantities)
**plus the off-ball actions' own durations** ✓ — and the events placed
**sequentially by type** instead of spread evenly ✓. Then the inter-chain
dead time shrinks to compensate, so the match's total clock stays at ~96
minutes ✓.

**THE COST, STATED PLAINLY:** it changes **every match's timeline** ✓ — the
chronology, the momentum model's xT series (which reads the clock ✓), and
every calibration number taken on the current clock ✗ — so it needs its own
gate and a multi-match sweep, and it is **not** a change to make inside a
thread that also changed the export. It is the next piece, with its own
commit.

## THE DECISION TIME — THE MISSING TIME, AND THE TIMELINE NOW READS LIKE FOOTBALL (2026-10-21)

The user, on the export's time axis: *"i dont to see 26 actions in 7 seconds
then later 50 seconds of nothing. that decision time is the missing time.
yes i think fix that with urgency and ensure it is well."*

### THE DIAGNOSIS, MEASURED BEFORE ANYTHING WAS TOUCHED

`_diag_clock_anatomy.py`, one real match:

| | before | real football |
|---|---|---|
| live (play) time in a 96-minute match | **1,410.7 s = 23.5 min (24.7%)** | ~55-60 min (~60%) |
| DEAD time | **4,289.5 s = 71.5 min (74.4%)** | ~36-40 min |
| seconds per event inside a chain | **0.449 s** (median) | 2-4 s |
| the gap between one possession and the next | **46.35 s** (median) | 10-20 s |
| the median interval between passes | **1.10 s** | 3-5 s |
| the first 26 events of the match | **7.04 s** | ~60 s |
| chains / events per chain | 418 / 6.92 | — |

**THE CAUSE IS THE ENGINE'S CLOCK, AND IT IS ONE SENTENCE:** a chain's
window is the clock advance during it, and that advance is the **BALL's
motion** ✓. So the off-ball actions inside a possession — the press, the
duel, the reception, the *decision* — were resolved inside the ball's
flight, and `chronograph()` then spread the chain's events evenly across
it. **A 30 m pass's ~1.2 s flight carried a PASS, a MISCONTROL, a PRESS and
the next PASS.**

**REAL FOOTBALL'S POSSESSION IS FLIGHT + CONTROL + DECISION + EXECUTION.**
A player takes the ball, controls it, looks up, decides, and only then plays
it. That decision time was missing from the CLOCK.

### THE FIX — `match_timeline.py` (new, pure, no engine imports)

- **the chain's window** = the ball's motion **+ the chain's decision
  time** ✓ — the time the ball's carrier spends between receiving the ball
  and playing it (the control 0.3 s + the decision 0.8 s) ✓;
- **the off-ball contests do NOT extend the window** ✓ — a press and a duel
  happen ALONGSIDE the on-ball action, which is what real football does and
  why they are not added ✓;
- **the events are placed by the football's own weights** ✓, not spread
  evenly ✓: a long pass's flight (distance / 22 m/s ✓) weighs more than a
  short one's, a carry's travel (distance / 5 m/s ✓) more than a press's
  contest ✓ — so the timeline reads as a possession instead of a burst ✓;
- **the clock fixes itself** ✓ — the engine's dead-time phase is
  `remaining = 60.0 - (the clock's elapsed time in this minute)` ✓, so the
  chains eating football-real time leaves proportionally less dead time ✓:
  **the match still totals 96 minutes ✓ and the silences shrink ✓.**

### MEASURED AFTER — the user's complaint, the same match

```
BEFORE:  00:00.000 PASS     00:00.280 MISCONTROL   00:00.580 PASS
         00:00.280 PRESS    00:00.890 BALL_RECEIPT  00:01.190 GROUND_DUEL
         00:00.580 PASS     00:01.490 PRESS        00:01.790 PASS
         ... 26 actions in 7.04 s ...   then 01:00.040 (a 53-second silence)

AFTER:   00:00.000 PASS     00:02.480 MISCONTROL   00:03.800 BALL_RECEIPT
         00:04.060 CARRY    00:05.730 PRESS        00:06.430 PASS
         00:07.740 BALL_RECEIPT  00:08.000 CARRY  00:09.290 PRESS
         ... 30 actions in 22.0 s ...
```

| | before | **after** | real football |
|---|---|---|---|
| live share of the match | 24.7% | **55.9%** | ~60% ✓ |
| median interval between passes | 1.10 s | **3.52 s** | 3-5 s ✓ |
| p90 interval between passes | 40.87 s | **13.99 s** | ~15 s ✓ |
| seconds per event inside a chain | 0.449 s | **1.309 s** | 2-4 s (closer) |
| the gap between possessions | 46.35 s | **28.70 s** | 10-20 s (closer) |
| the first 26 events | 7.04 s | **~19 s** | ~60 s |
| mean interval between passes | 7.09 s | **6.98 s** | ~5 s ✓ (preserved) |
| the match's total clock | 96.4 min | **96.4 min** | 96 min ✓ (unchanged) |

**THE MEAN IS PRESERVED AND THE DISTRIBUTION IS REPAIRED** ✓ — which is the
signature of a redistribution rather than an addition ✓.

### THE GATE — `tests/test_match_timeline.py` (15, NEW file)

**It pins the OUTCOME, not the constant** ✓ — a test that asserts
`DECISION_S == 0.8` asserts nothing about the football ✓, so the slow gate
runs a REAL match and pins what the number is FOR: the live share (50-70%),
the median interval between passes (2-6 s), the densest chain's span
against its event count, the gap between possessions (median < 40 s, max
< 70 s), the match's total clock (96 ± 3 min ✓), the timeline's
monotonicity ✓, and that the events' stamped clocks and the chronology
agree ✓. Plus the module's arithmetic (an on-ball event consumes decision
time; an off-ball contest does not; a long pass weighs more than a short
one; every event gets a positive weight ✓) and two STRUCTURAL guards (the
engine calls the module ✓, and the even spread `span * i / (n - 1)` is
gone ✓).

### THE SIDE EFFECT, MEASURED AND STATED: THE DISTANCES MOVED

**The live/dead split changed ✓ (24.7% → 55.9% live ✓) and the live
phase's off-ball movement is the CHASING (faster ✓) while the dead
phase's is the HOLDING (slower ✓) — so moving time from dead to live
raises the distance covered ✓.** Measured, one match
(`tools/diag/_diag_measured_activity_verify.py`):

| | before (2026-10-19/20) | **after** | CIES n°68 |
|---|---|---|---|
| CB | 8.41 km | **10.41 km** | 9.2 km |
| LB | 9.66 | **11.00** | — |
| CDM | 9.80 | **11.18** | — |
| CM | 11.10 | **12.28** | 10.6 km |
| LW | 10.54 | **11.91** | — |
| ST | 10.04 | **11.82** | — |
| standing share | 4.3-16.9% | **4.3-7.9%** | 20-30% (CB) |

**So the per-position distance calibration of 2026-10-20 is now STALE** ✗ —
the numbers are 1.3-2.4 km higher ✗ and the centre-backs are now above
the CIES figure ✗. **This is a CALIBRATION consequence, not a defect** ✓
(the live share is now real ✓, so the distance follows ✓) — but it must be
re-fitted against the benchmark, and that is a deliberate multi-match step
of its own ✓, exactly as the standing rule says about a fit ✓.

**THE STANDING SHARE IS UNCHANGED AND STILL GOOD** ✓ (4.3-7.9% ✓, the
2026-10-19 fix intact ✓) — the locomotion work is not undone by this ✓.

### ⛔ AND THE REMAINING GAP IS NOT THE POSSESSION STRUCTURE — I SAID IT WRONG, AND THE REAL NUMBERS SAY SO

**I wrote above: *"the engine plays 2-4 possession sequences per minute where
real football plays ~10-12"*, and that is WRONG.** The real Premier League's
ball-in-play share in 2026-27 is **57.7%** (Opta, via The Analyst: *"the
average time the ball has been in play in Premier League games in 2026-27 is
57.7% — the highest since 2020-21"*) ✓ — i.e. **42.3% of the match is the ball
OUT OF PLAY** ✓. And the ordinary restarts take (the same source, the 2026
World Cup): a **throw-in 13.2 s**, a **goal kick 23.7 s** ✓, while a
Premier League **corner averages 29.0 s** (Opta's October report ✓) and a
substitution stoppage **65.5 s** ✓.

**SO THE FIXED ENGINE'S NUMBERS, AGAINST THOSE:**

| | PLOFA now | real Premier League |
|---|---|---|
| ball in play | **55.9%** | **57.7%** ✓ (within 2 points) |
| ball out of play | 41.8 min of 96 | ~42 min ✓ |
| the out-of-play gap | 28.7 s | **13.2-29.0 s** for an ordinary restart ✓ (and 29.0 s for a corner ✓) |
| possessions per minute | 4.3 | ~3-4 ✓ (the ball changes hands every ~15-20 s ✓) |
| the length of one possession | 8 s | ~11-14 s ✗ (slightly short) |
| passes per match | 814 | ~900-1,000 ✗ (slightly short) |
| events per minute | 31 | 22-28 ✗ (slightly over — the engine's own docstring says so ✓) |

**THE TOTALS ARE RIGHT.** The ball-in-play share is within two points of
the real Premier League, the dead time is within a minute, and **a 28-second
pause is a real restart length** ✓ — it is a corner's average ✓. So the
28-second gap is NOT a defect to be shrunk: it is the right size, in the
right place, at the right total.

**WHAT IS STILL OFF IS THE RHYTHM, NOT THE AMOUNT.** Real football's dead
time arrives **two or three times a minute** and **varies enormously** — 13 s
for a throw-in, 24 s for a goal kick, 29 s for a corner, 45-50 s for a
corner taken slowly (Arsenal 45.2 s, Sunderland 49.5 s, Opta's October
figures ✓), 65 s for a substitution ✓. PLOFA's arrives **about once a
minute, at a near-constant ~28 s** ✓ — **metronomic where real football is
varied** ✓. That is the one remaining tell when you watch a match, and it
is a symptom of the ENGINE'S MINUTE STRUCTURE (the dead time is literally
"the rest of the minute" ✓) rather than of the clock ✓.

**SO THE NEXT PIECE IS NOT "MORE POSSESSIONS" ✓ — it is "POSSESSIONS OF
VARIED LENGTH" ✓** (some minutes one long spell ✓, some minutes four short
ones ✓), which would spread the same 42 minutes of dead time across the
match in real football's shape ✓ and leave the ball-in-play share exactly
where it is ✓. **And the pass volume and the events-per-minute are the
same finding from the other end** ✓: at 814 passes and 31 events a minute
against the real ~900-1,000 and 22-28 ✓, PLOFA's football is slightly
denser and slightly faster than a real match ✓.

**AND THE MATCH'S OUTCOMES BARELY MOVED, for a reason worth recording:**
the engine runs a FIXED number of possession sequences per minute
(`_simulate_minute`), NOT a clock-driven loop ✓✓ — so widening a chain's
window changes the TIMELINE, not who wins the match ✓. What it does change
is how long the off-ball integrator runs inside a live chain ✓, and the
live/dead split ✓ — which is the distance effect above ✓.

**THE TWO `test_small_game` FAILURES ARE BISECTED AS PRE-EXISTING ✓** (not
"documented pre-existing", which is not evidence ✓). The shape suite reports
`test_striker_gets_in_behind_not_offside` (an XPASS(strict) — the xfail
marker is stale ✓) and `test_fullback_can_tuck_inside` (the full-back's
width range 0.12 m against a 2.0 m assertion ✓). `_diag_smallgame_bisect.py`
puts the engine back into the PRE-FIX state (the decision time off ✓ — the
only change that could reach the shape, since the even spread is a
placement and not a position ✓) and runs the same two tests: **both arms
fail identically, down to the same 17-digit number
(`0.12435051023646793`) ✓** — so they are independent of this work ✓.
Everything else in the shape suite is green (51 passed ✓).

**THE DECISION TIME (2026-10-21) — read before touching the clock, the
chronology, or anything that spreads events over time:**
- `match_timeline.py` (NEW) — the football's own timings and the two
  functions the engine uses: `chain_decision_seconds(events)` (what a chain
  ADDS to the clock) and `placement_weights(events)` (how the events sit
  inside the chain's window). `DECISION_S = 0.8`, `CONTROL_S = 0.3`,
  `BALL_SPEED_MPS = 22`, `CARRY_SPEED_MPS = 5`, `CONTEST_S` (the off-ball
  contests). **Pure, no engine imports.**
- `match_engine.py` — `_absorb_motion` adds the decision time to the
  chain's window; `chronograph()` places the events by WEIGHT (the even
  spread `span * i / (n - 1)` is GONE). The event's stamped clock and the
  chronology's are still the same number ✓.
- `tests/test_match_timeline.py` (15, NEW) — **pins the OUTCOME, not the
  constant**: the live share, the interval between passes, the densest
  chain's span against its event count, the gap between possessions, the
  match's total clock, the monotonicity, and the chronology's agreement
  with the events. Plus the module's arithmetic and two structural guards.
- Probes: `_diag_clock_anatomy.py` (the clock's anatomy — the live/dead
  split, the chains, the gaps), `_diag_timeline_football.py` (the timeline
  against real football: the pass interval, the first 30 events),
  `_diag_timeline_note.md` / `_diag_timeline_note2.md` (the two write-ups,
  with the measurements).
- **THE REMAINING GAP IS NOT THE CLOCK AND NOT THE POSSESSION COUNT:** the
  ball-in-play share is now **55.9%** against the real Premier League's
  **57.7%** ✓ (Opta, via The Analyst ✓) — within two points ✓ — and a
  28-second out-of-play gap is a real restart length ✓ (a corner averages
  29.0 s ✓). What is left is the RHYTHM: real football's dead time arrives
  2-3 times a minute and varies from 13 s to 65 s ✓, PLOFA's arrives once
  a minute at a near-constant 28 s ✓. The next piece is possessions of
  VARIED LENGTH ✓, not more of them ✓ — and a claim that the engine plays
  "2-4 possessions a minute where real football plays 10-12" was WRONG ✓
  (the real figure is ~3-4 ✓) and is corrected above ✓.
