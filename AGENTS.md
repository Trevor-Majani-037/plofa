## DEFENSIVE SITUATIONAL BRAIN PIPELINE (2026-10-24/25)

Built a complete pipeline for per-decision defensive action data with full situational context, culminating in a distilled FB brain and real-match gate.

### 1. CORPUS COLLECTOR (`defensive_situational_collect.py`)

Patches `_def_action_choice` and `DefensiveChain.generate` to capture per-decision defensive action data with full 51-dim situational context:

- **State (24-d)**: pressure, space, thirds, centrality, goal-closeness, danger
- **Role block (13-d)**: defensive action vocabulary per family
- **Memory block (8-d)**: episodic memory from `NeuralStateStore`
- **Plan block (6-d)**: team plan summary
- **Entity tokens (12-d)**: attention over nearby entities

Outputs sequences grouped by `(team, player)` for BPTT, with GRU hidden state carried touch-to-touch, reset on possession change/goal-line reset.

**Results (12 matches, seed 123)**: 23 sequences, 34 steps (FB only — Lewis Driutt 12 steps, Moffart Randier 11 steps). Saved to `brains_def/situational_sequences.json`.

### 2. SITUATIONAL SURROGATE (`build_defensive_situational_surrogate_v2.py`)

Merges situational corpus (34 entries) + legacy samples (forced 377, heuristic 191, natural 9 = 611 total):

- 4 families: CB, FB, DM, GK
- 4 actions: tackle, interception, clearance, block
- Bucket keys: `Z{zone}D{danger}A{attack}O{opp}E{engage}F{foul}M{momentum}`
- Per-family + global fallback tables with per-bucket action values
- Output: `brains_def/surrogate_situational.json` (FB: 49 buckets, 111 action entries)

**Key finding**: Surrogate is clearance-heavy (36/111 entries = clearance), reflecting the heuristic's bias.

### 3. DISTILLER (`tests/distill_defensive_brains.py`)

Pure numpy BPTT distillation (80 epochs, 64 hidden, cross-entropy vs surrogate's argmax):

- Architecture: 51 → 64 → 64 → 4 (tackle/interception/clearance/block)
- FB brain: 49 training samples, 80 epochs, ~45% accuracy on surrogate
- Output: `brains_def/situational/FB.npz` (51→64→64→4)

### 4. CENSUS GATE — FAIL

| Action | Count | % |
|---|---|---|
| clearance | 1380 | 69.0% |
| tackle | 411 | 20.5% |
| interception | 194 | 9.7% |
| block | 15 | 0.8% |

**FAIL**: clearance 69% > 40% limit; 4 active intents > 3 limit. Root cause: surrogate is clearance-heavy; distillation amplifies bias.

### 5. REAL-MATCH GATE — FAIL (3 matches per arm, seed 123)

| Metric | Incumbent | Challenger | Gate |
|---|---|---|---|
| Goals For | 1 | 2 | PASS |
| Goals Against | 10 | 12 | **FAIL** |
| Record | 0-0-3 | 0-0-3 | — |
| Home Poss | 43.8% | 46.1% | — |

**Why it failed**: Census showed clearance 69% — the brain just clears everything. The integration patch wasn't actually using the brain's decisions (just called original `_def_action_choice`). Challenger conceded 2 more goals in 3 matches.

### ROOT CAUSE & NEXT STEPS

1. **Surrogate bias**: Clearance 36/111 entries → distillation amplifies
2. **Integration incomplete**: Patch called original `_def_action_choice`, never queried brain
3. **Data coverage**: Only FB family has data (12 matches, 23 sequences)

**Next steps**:
1. Rebalance surrogate before distillation (reduce clearance, add diversity)
2. Fully wire brain decisions into `_def_action_choice` with 51-dim input
4. Collect ~12 more matches with diverse styles for CB/DM/GK families
5. Retry distillation + gate

### FILES ADDED/MODIFIED

**New files**:
- `defensive_situational_collect.py` — corpus collector
- `brains_def/situational_sequences.json` — corpus
- `brains_def/surrogate_situational.json` — situational surrogate
- `brains_def/situational/FB.npz` — distilled FB brain
- `tests/build_defensive_situational_surrogate.py` — v1 surrogate builder
- `tests/build_defensive_situational_surrogate_v2.py` — v2 surrogate builder
- `tests/distill_defensive_brains.py` — distiller
- `tests/gate_3matches.py` — 3-match gate runner
- `tests/quick_match.py` — quick match runner
- `tests/real_match_gate.py` — 7-match gate framework
- `tests/test_defensive_brain.py` — integration test
- `tests/test_zachery_match.py` — zachery test

**Modified**:
- `AGENTS.md` — this session documented
- `defensive_situational_collect.py` — final collector
- `brain_schema.py` — situational brain schema
- `surrogate_collect.py` — collector utilities
- `match_engine.py` — patch points
- `auto_run_match.py` — match runner
- Various manager brains/state files (season progression)

**Cleaned up**:
- Moved test files to `tests/`
- Deleted 100+ intermediate `_fix_*.py`, `_analyze_*.py`, `_find_*.py`, `_rewrite_*.py` debug scripts
- Root directory clean

## ARCHITECTURAL PRINCIPLE: THE BRAIN OWNS THE VERB, THE MATRIX OWNS THE NOUN (2026-10-25)

An explicit architectural consensus reached by user and developer on the division of responsibilities in match simulation:

### 1. Division of Responsibilities
* **The Attacking Matrix (AM) & Phase Regressions (PR) = The Coach's Tactical Plan (The Noun)**:
  - Governs structure, pitch geometry, positional balance, offensive spacing, and available passing receivers.
  - In real football, managers (Guardiola, Arteta, Ancelotti) drill where teammates run, who underlaps/overlaps, and where passing angles exist. The AM and PR perform exactly this role.
* **The Neural Decision Brain (v8 SituationalBrain / Soul Brains) = Player Agency (The Verb)**:
  - Evaluates the immediate situation (entity attention over 16 nearby players, recurrent GRU memory, pressure, pocket space, fatigue).
  - Selects the action intent (`SAFE_PASS`, `PROGRESSIVE_PASS`, `THROUGH_BALL`, `SWITCH`, `DRIBBLE`, `CARRY`, `RECYCLE`, `PROTECT_POSSESSION`).

### 2. Mandatory Directive for Future Agents
* **Do NOT criticize or attempt to bypass the Attacking Matrix (AM) or Phase Regressions (PR)** under the mistaken impression that pass maps or receiver selections "should be 100% generated by neural weights alone."
* Forcing a ~14k parameter neural net to invent whole-pitch 22-man coordinate geometry from scratch leads to structural collapse, tactical amnesia, and unrealistic passing chaos.
* The division between **Tactical Structure (AM/PR)** and **Cognitive Agency (Brain)** is fundamental, deliberate, and permanent. Respect this seam.

## THE GOAL IS A MOUTH, NOT A DISTANCE (2026-10-31)

### The gap
The on-ball 24-d vector (`brain_sensors.extract_sensors`) knew only three things about the goal: `[11] is_near_goal` (< 25 m), `[14] goal_distance_norm`, `[15] central_lane` (`|y - 34| < 18`). Two flags and a distance. It had no *angle*.

At (80, 20) and (80, 50) the old vector was nearly identical: goal distance 28.7 m vs 29.7 m, `is_near_goal = 0` both, `central_lane = 1` both. In truth one player sees the goal 29° to his RIGHT with the mouth 12.8° wide, the other sees it 33° to his LEFT with the mouth 11.9° wide. The brain could not tell a shooting angle from a crossing angle.

### The fix (no retrain, no width change)
Slots 2-3 were **exact duplicates of slots 0-1** for an on-ball carrier (the carrier stands at the ball: `s[2]-s[0] == 0.0`), so they carried literally zero information. They now carry the goal's visual geometry:

| slot | feature | meaning |
|---|---|---|
| `[2]` | `goal_angle_norm` | signed bearing to the goal CENTRE from the attacking direction, `/90°`, clamp ±1. 0 = dead ahead, +1 = goal line on the right |
| `[3]` | `goal_mouth_norm` | angle the two POSTS subtend, `/` the penalty-spot reference angle, clamp 0..1. 1.0 = a penalty-spot sight of goal, →0 on the byline |

Both come from `_goal_geometry()` in `brain_sensors.py`, using the SAME constants the tactical layer already uses (`attacking_matrix.POST_OFFSET_Y = 3.66`, `REFERENCE_ANGLE = 2·atan2(3.66, 11.0)` → `_REFERENCE_MOUTH_ANGLE`). The brain and the matrix now read the same goal.

Measured (attacking right, seed-independent, pure trigonometry):

| position | old `[2] [3]` | new `[2] [3]` | posts (deg) | width | goal dist |
|---|---|---|---|---|---|
| (80, 20) | 0.7619 0.2941 | **+0.3250 0.3469** | 35.24 / 22.47 | 12.77° | 28.65 m |
| (80, 50) | 0.7619 0.7353 | **−0.3624 0.3236** | −26.27 / −38.18 | 11.91° | 29.68 m |
| (85, 34) | 0.8095 0.5000 | 0.0000 0.5635 | 10.37 / −10.37 | 20.74° | 20.00 m |
| (94, 34) | 0.8952 0.5000 | 0.0000 1.0000 | 18.40 / −18.40 | 36.81° | 11.00 m (penalty spot reference) |
| (104, 20) | 0.9905 0.2941 | 0.9546 0.0620 | 86.76 / 84.48 | 2.28° | 14.04 m (byline: angle is gone) |

### Invariants that make the swap safe
* **Width unchanged** — still 24 inputs, so every 24-wide brain loads and runs; `slots 15 (central_lane), 11, 12, 13, 14` untouched, so `value_model._CENTRAL`, surrogate bucket keys and probe labels keep their meaning.
* **Off-ball vector is deliberately different** — `extract_offball_sensors` keeps runner_x/runner_y in slots 2-3, because a runner does not stand at the ball and that offset IS the off-ball decision. Documented in the function docstring.
* **Not routed through FOV/noise** — the goal is a fixed landmark, not another actor. Perception degrades what a player *decides* around the goal, not whether he knows where it is.
* **Deterministic and mirror-exact** — attacking right from (80,20) and attacking left from (25,48) produce bit-identical slots 2-3.

### Consumer migrated
`surrogate_collect.py` sequence-splitting read `sensors[2]` as the carrier's x to detect a goal-line reset. It now reads `sensors[0]` (slot 2 was only ever a copy of slot 0). Any future code that wants carrier x must use slot 0.

### Files
* `brain_sensors.py` — `_goal_geometry()`, `_GOAL_CENTER_Y`, `_GOAL_POST_HALF_WIDTH`, `_REFERENCE_MOUTH_ANGLE`; slots 2-3 rewritten; off-ball divergence documented.
* `surrogate_collect.py` — jump detector migrated slot 2 → slot 0.
* `value_model.py` — `_DROP_COLS` comment updated: 2-3 are no longer duplicates but are still dropped so the 37-wide phi stays valid for existing fits.
* `tests/test_goal_vision.py` — 9 new tests (real angles, peer example, byline collapse, equal-distance central vs wide, width invariant, mirror symmetry, determinism, brain still consumes).
* `tests/test_football_brain.py` — `_in_range()` helper; slot 2 is a legal signed slot (with slot 18).

## PRODUCTION BRAIN SET: v8 SITUATIONAL SHIPS, T1 IS RETIRED (verified 2026-10-31)

Earlier notes in this repo say the "shipped T1 set" is what plays matches. **That is stale.** T1 was replaced on 2026-10-23 by commit `c7b6918` (*PROMOTE v8 SITUATIONAL BRAINS TO PRODUCTION*). Verified from the working tree, not from the docs:

| check | result |
|---|---|
| `brain_integration.BRAIN_DIR` | `brains` (overridable via `PLOFA_BRAIN_DIR`) |
| `brains/_manifest_all.json` | `architecture: situational`, `sensor_schema: v5_situational`, `training_method: bptt_situational`, `input_size: 47`, `hidden_size: 32`, `params_per_position: 14000` |
| `_brain_for_position(<pos>)` for GK/CB/LB/CDM/CM/CAM/LW/ST/CF | **`SituationalBrain`** in all 9 cases (no `FootballBrain`) |
| `brains/<POS>.json` vs `brains_v8/<POS>.json` | **byte-identical**, SHA256 match 11/11 |
| file size | **~458 KB** each (11 files) |

### The size discriminator (this is the quick way to tell them apart)

| set | what it is | on-disk size | keys | meta |
|---|---|---|---|---|
| **v8 situational (SHIPPED)** | attention over 12-d entity tokens + GRU + memory + plan heads | **~458 KB** | `arch_version`, `meta`, `weights` | `arch_version 3`, `sensor_schema v5_situational`, `training_method bptt_situational` |
| T1 (`v2_role_features` MLP) | `32 → 32 → 32 → 10` dense, 24-d shared + role tail | **~62–70 KB** | `arch`, `w1..w3`, `b1..b3`, `meta` | `arch_version 1`, `sensor_schema v2_role_features`, `training_method ga_surrogate` |
| v1 24-d MLP | older 24 → 32 → 32 → 10 | ~62 KB | `arch, w1..w3, b1..b3` | `arch_version 1`, `sensor_schema v1_24d` |

So: a `brains/` file around **460 KB with a `weights` key is v8 situational**; anything around **62 KB with `w1/b1` keys is pre-v8**.

### Recovering T1
T1 is in git at `c7b6918^` (i.e. `a862acb`):

```
git checkout c7b6918^ -- brains/
```

Note the promotion commit message points at `brains_backup_T1_20261023/` — **that directory was never created.** The two `brains_backup_t1_*` directories on disk are the 2026-09-12/13 sets (`v1_24d` / pre-v2role), not the T1 set that shipped in October. Git history is the only reliable backup.

### Docs that still say "shipped T1" (all predate `c7b6918`)
* `_ag_section_final.md` — "SHIPPED SET REMAINS T1". This was the v8-vs-T1 gate draft; the user then promoted v8 anyway (fewer goals conceded, better xG, higher forward%). Kept for the numbers, corrected at the top.
* `PLAN_2026-10-21.md`, `PLOFA_ARCHITECTURE_AUDIT.md` — dated plans/audits, historical.

### Consequence of the 2026-10-31 sensor change for the shipped v8 brains
The v8 weights were BPTT-distilled while slots 2-3 were still duplicate positions, so the new goal geometry is information the network was **not fitted for**. Measured on the actual files (324 synthetic states per brain, role/memory/plan/`h_prev` zeroed so only slots 2-3 differ; old = position duplicated, new = goal geometry):

| brain | mean L1 Δprob | max Δprob | intent flips | mean ΔP(chosen) |
|---|---|---|---|---|
| GK | 0.0089 | 0.0177 | 0.0% | −0.0041 |
| CB | 0.0837 | 0.2207 | 7.4% | +0.0012 |
| LB | 0.0667 | 0.1279 | 13.9% | −0.0003 |
| CDM | 0.0915 | 0.1129 | 24.4% | −0.0084 |
| CM | 0.1065 | 0.1101 | 4.3% | −0.0266 |
| CAM | 0.1157 | 0.1879 | 18.8% | −0.0311 |
| LW | 0.2225 | 0.4199 | 19.4% | −0.0199 |
| ST / CF | 0.1716 | 0.3569 | 7.7% | +0.0085 |
| **mean** | **0.1154** | — | **11.5%** | — |

No information is lost — the carrier's position is still slots 0-1, which are untouched — but ~11.5% of intents flip on states where only slots 2-3 changed, and the chosen intent's confidence barely moves (|ΔP| ≤ 0.031). Read this as "the network is reacting to new inputs it was not trained on", not as a validated improvement. The next v8 retrain sees the geometry natively; until then, match-level validation of this change is the 7-match gate, not a single friendly.
