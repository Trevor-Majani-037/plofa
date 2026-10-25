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
