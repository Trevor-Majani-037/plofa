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
  - **AN A/B ACROSS TWO `simulate()` CALLS IS INVALID.** Both arms scored
    differently (0-0 vs 1-0) because the second inherits the first's module-level
    brain caches — the seed-reproducibility bug biting exactly where it hurts.
    Compare by replaying ONE recorded feed through both configurations instead.
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
- **BRAIN-ONLY vs DECIDERS, 3 matches per arm, SEPARATE PROCESSES** (an A/B
  across two `simulate()` calls in one process is invalid — the second
  inherits the first's module-level brain caches):

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

- **CORNER CONVERSION, 6 seeds in 6 SEPARATE PROCESSES** (mandatory — an
  in-process sweep inherits the brain caches): 3 goals from 58 corners =
  **5.2%**. Real Premier League is 3–4%, so this is in the band, and it is up
  from step 1's 2.8%.
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
- **STREAM PARITY SHIM — DO NOT DELETE.** Removing the fabricated origin also
  removed a `random.uniform(5, 20)` draw, and that is not cosmetic: it shifts
  every later number in the global stream. Proven, not assumed — restoring that
  one discarded line at `event_chain.py` turns
  `test_match_crosses_stamped_geometrically` from fail to pass with no
  behavioural change whatsoever, and every calibration figure in this file was
  taken on the stream it preserves. It is named `_STREAM_PARITY_DRAW` and
  commented as load-bearing so it is not "cleaned up". The real fix is the
  documented one: make a match reproducible from `random.seed` (module-level
  brain/mind caches survive `simulate()`), then the shim can go.
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
  per-(team, minute, zone) in `_striker_run_cache`.
- `tests/test_positional_play.py` — 46 tests: the offside-line regressions, the
  rest-defence invariant (and that it stays silent in a normal shape), the
  depth-only clamp, the no-global-RNG guarantee, and a full real match.
- `_diag_posplay_speed.py`, `_diag_cross_probe.py` — throughput and
  cross-detection probes (diagnostics, not tests).
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
suspect for the same reason, not just the world layer.** This is why the phase-5
gate counts *executions* of the new code instead of comparing simulated output —
a fingerprint gate would report "regression" on a green engine.

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

1. **Seed reproducibility** — the highest-value loose end in the whole project.
   The live match is not reproducible from `random.seed()`; the cause is
   module-level brain/mind caches (`brain_integration._brain_registry`,
   `_pos_brain_cache`, `cognition.mind._minds`) that survive a simulation and
   are only cleared by explicit API calls. This undermines every calibration
   number taken so far. The fix is three lines at the top of
   `_initialize_simulation()`, but it needs a judgement call: if that state is
   MEANT to persist across matches, clearing it changes behaviour.
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

## RESOLVED — 20 vs 18 clubs (Trevor, 2026-09-25)
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