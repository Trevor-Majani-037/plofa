# PLOFA — deep research audit

**Date:** 2026-11-07
**Scope:** the whole working tree at `D:\PLOFA\plofa`, HEAD `45f6501` plus uncommitted changes.
**Method:** static reading of the live code path, an import-graph build over all 132 root
modules, a real 90-minute match simulated through `brain_observatory`, and SQL over
`alltime.db` (83 played fixtures) and 85 exported match timelines (227,332 events).

Read-only. Nothing in the repository was modified. Where a measurement required a
broken tool, the tool was copied to a temp directory and patched there, never in place.

Every claim below is marked **[verified]** (reproduced here), **[read]** (from source),
or **[inferred]** (reasoning from verified facts).

---

## 1. What this project is

A full-match football simulation engine in Python. Not a game: a *research instrument*
whose thesis is that player decisions should be produced by evolved/distilled neural
networks operating on a perceived world, while tactical structure stays hand-authored.

That thesis is stated explicitly in `AGENTS.md` as an architectural commitment:

> **The brain owns the verb, the matrix owns the noun.**

This is the most important thing to understand about the codebase. It is not an
accident of implementation. Nearly every odd-looking seam — why pass direction is a
lookup table, why the brain outputs labels instead of coordinates, why `DECIDER_BOUNDARY.md`
exists at all — follows from that commitment, and the code is internally consistent with it.

---

## 2. Scale

| Measure | Value |
|---|---|
| Root Python modules | 132 |
| Root LOC | 94,182 |
| Test files | 145 |
| Test functions | 1,884 |
| Test LOC | 45,037 |
| Sub-package LOC (`physics/`, `world/`, `cognition/`, `battlefield/`) | 11,750 |
| **Total Python** | **~151,000 lines** |
| Commits | 66 (first commit 2026-09-11 — eight weeks) |
| Brain directories | 31 (`brains/` plus 30 historical/smoke sets) |
| `.git` size | 354 MB |

151k lines in eight weeks, by a very small team, with this much prose documentation, is
an unusual shape. The documentation-to-code ratio is the highest I have seen on a
project of this size and it is genuinely load-bearing: `AGENTS.md`, `DECIDER_BOUNDARY.md`,
`PLOFA_ARCHITECTURE_AUDIT.md` and `BRAIN_COST_ANALYSIS.md` each record *why* a decision
was made and *what measurement* settled it.

---

## 3. Architecture as shipped

```
auto_run_match.py            season, XI, fatigue, availability      1,533 lines
  └─ match_engine.py         MatchEngine                            7,828 lines / 151 functions
       ├─ minute loop        _simulate_minute                       1,000 lines in one function
       ├─ event_chain.py     PossessionChain                        11,036 lines
       │    └─ generate()                                          2,586 lines in one function
       ├─ position_engine.py shape, space, drift                    5,748 lines
       ├─ attacking_matrix.py the coach's plan                      1,700+ lines
       └─ brain_integration.py  the decision seam                   1,000+ lines
  └─ exporter.py             Excel / CSV / JSON / PNG              4,776 lines
```

### The three monoliths

| File | Lines | Largest single function |
|---|---|---|
| `event_chain.py` | 11,036 | `PossessionChain.generate` — **2,586 lines** |
| `match_engine.py` | 7,828 | `_simulate_minute` — **1,000 lines** |
| `position_engine.py` | 5,748 | `drift_minute` — 435 lines |
| `exporter.py` | 4,776 | `_process_event` — 700 lines |

A 2,586-line function is the single largest maintenance risk in the repository. It is not
a style complaint: that function is the possession chain — every pass, carry, shot,
dribble and turnover in the match — and it has no internal seams that a test can grab.

---

## 4. The brain: lineage, and what actually runs

Four generations exist on disk. Only one plays matches.

| Set | Architecture | Size | Status |
|---|---|---|---|
| v1 | 24 → 32 → 32 → 10 MLP | ~62 KB | retired |
| T1 | `v2_role_features` MLP | ~62–70 KB | retired at `c7b6918` (2026-10-23) |
| **v8 situational** | **47-d input, GRU h=32, entity attention, memory + plan heads** | **~458 KB** | **SHIPPED** |
| defensive situational | 51 → 64 → 64 → 4 | 62 KB `.npz` | **built, gated, FAILED, never wired** |

The v8 set is real and live: `brains/_manifest_all.json` declares
`sensor_schema: v5_situational`, `training_method: bptt_situational`, `input_size: 47`,
`hidden_size: 32`, `params_per_position: 14000`, and `_brain_for_position()` returns a
`SituationalBrain` for all 11 positions. **[read]**

### The defensive brain is a decade behind the attacking one

This is the sharpest inconsistency in the codebase.

- `match_engine._ensure_def_action_loaded()` auto-loads `brains_def/ACTION.json`
- that file is dated **2026-09-10**, has keys `w1..w3, b1..b3` and **no meta block**
- it is the old `DefensiveActionBrain`, 24 → 32 → 32 → 4

Meanwhile `brains_def/situational/{CB,DM,FB,GK}.npz` (2026-10-24) — the 51-d situational
defensive brains with BPTT distillation — are **loaded by nothing**. `AGENTS.md` records
their gate as FAIL (clearance 69% of decisions), and the code never got the promised
"fully wire brain decisions into `_def_action_choice`" fix.

So: attackers think with a 14k-parameter recurrent attention network. Defenders think
with a 40 KB feedforward net from September. **[verified]**

---

## 5. The decision seam — what the brain actually owns

`DECIDER_BOUNDARY.md` is honest about this and it is worth restating because it bounds
everything else:

```
NeuralDecisionBrain.decide()
  ├─ probs                    10 numbers
  ├─ intent = _INTENT_BY_INDEX[argmax]     ← the ONLY learned choice
  └─ target = _find_target(intent)         ← hand-written lookup
        SAFE_PASS        → nearest teammate
        PROGRESSIVE_PASS → best forward
        THROUGH_BALL     → best forward
        SWITCH           → wide teammate
```

The network chooses **which of ten labels**; geometry chooses **where the ball goes**.
`POLICY_INTENT_AUTHORITY = False` by default, so the matrix and the forced-receiver layer
retain authority over the delivery class at several gates.

The document's own measurement is the damning one: with the deciders off, median pass
displacement stays at ~0 m in both arms, because direction was never the brain's to move.
Backward passes run 31–37% against a real-football 10–15%, and the team achieves **zero
net progress across ninety minutes**.

That is a *chosen* state (option 3 in the document: "accept it as a rule, and make it
deliberate"). It is coherent. But the rule was never written down as a rule — there is no
forward-bias constant with a name and a test pinning the resulting distribution to a real
band, which is what option 3 actually requires.

---

## 6. Measured health — is the football right?

### 6.1 One live match, seed 123, Oxton vs Natrican

Ran through `brain_observatory` (patched copy): **Oxton 1–6 Natrican**, with an own goal
after 1 minute, a straight red at 36', and two second-yellows at 72' and 74'.

Not a calibration statement from n=1, but note the shape: three sendings-off and an
own goal in one match. Discipline variance is worth a separate look.

### 6.2 Season 26/27 — 83 fixtures in `alltime.db`

| Metric | PLOFA 26/27 | Real-football reference | Verdict |
|---|---|---|---|
| Goals per match | 3.30 | 2.8–3.0 | high |
| Shots per team | 15.2 | 12–14 | high |
| Shots on target | 6.4 | 4.8 | high |
| Pass accuracy | 84.2% | 84–86% | good |
| Fouls per team | 12.9 | 11–14 | good |
| Corners per team | 6.8 | 4.6–5.5 | high |
| Offsides per team | 5.1 | ~2.0 | **2.5× too high** |
| xG per team | 2.09 | 1.4–1.5 | **~40% too high** |

From the 85 exported timelines (a slightly different sample, measured event-by-event):

| | |
|---|---|
| Shots | 2,158 |
| Goals | 230 |
| xG | 256.5 |
| Conversion | **10.7%** (real ≈ 10–11% — good) |
| xG per shot | **0.119** (real ≈ 0.10–0.11 — high) |

**Reading:** the *finishing* is calibrated correctly. The *chance quality model* is not —
it assigns ~12–14% more xG per shot than the shots are worth, which is why the season
table shows teams "underperforming xG" by 25% across the board. That is not variance, it
is a systematic bias in `XGEngine.calculate_geometric`.

### 6.3 There is no home advantage

| Season | Home goals/match | Away goals/match |
|---|---|---|
| 24/25 (legacy, real data) | 1.63 | 1.31 |
| 25/26 (legacy) | 1.57 | 1.31 |
| **26/27 (simulated)** | **1.54** | **1.76** |

**[verified]** `is_home` is threaded through `_resolve_team_profile()` → `auto_team_style()`
and then **used for nothing**. `auto_team_style` accepts it and returns a `TeamProfile`
that does not depend on it. There is no crowd, no travel, no referee bias, no style shift.

The 1.54 vs 1.76 split is noise (n=83, z ≈ −0.98), not an away advantage. The real finding
is the *absence* of the +0.3 goal home edge that real football has and that the legacy
seasons in the same database show.

### 6.4 The battlefield's "home-goal inflation" was a harness artifact

`README.md` §8 reports home 2.229 vs away 1.583 goals and names "home-goal inflation
(+24%)" as *the largest behavioural defect*, recommending an investigation of the
home-advantage model. **[inferred]** That number cannot come from the engine, because the
engine has no home term at all.

`battlefield/run_plofa.py:73` is the likely source:

```python
slot = idx % n if is_home else (idx + n // 2) % n
```

Home and away are assigned *style slots offset by five* in a palette ordered
attacking → defensive. Home always faces away across a fixed tactical gap, and clubs are
paired by list parity (`clubs[2i]` vs `clubs[2i+1]`). Whatever that measures, it is not
home advantage.

**Consequence:** a documented calibration target ("fix home-goal inflation") is chasing a
number produced by the test harness. The real defect is the opposite one — no home edge
exists.

### 6.5 The season is not homogeneous

xG per team by matchday:

| MD | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| shots | 15.3 | 17.8 | 15.4 | 17.7 | 14.7 | 13.4 | 14.3 | 13.7 | 14.5 |
| **xG** | **2.20** | **3.32** | **2.48** | **2.52** | 1.73 | 1.49 | 1.91 | 1.72 | 1.46 |
| goals | 1.32 | 1.67 | 1.61 | 1.50 | 1.50 | 1.17 | 2.11 | 1.61 | 1.56 |

Shot volume is flat. Goals are flat. **xG falls 37% after matchday 4** — a model change
landing mid-season, not a change in the football. Any season-long xG table currently mixes
two different xG engines. **[verified]**

---

## 7. The observability chain is severed — the most important finding

The project's central claim is that decisions are made by a network. The natural question
is *"why did it choose that?"* There are three independent answers, and all three are
currently broken.

### Break 1 — only two event types record the decision

`event_chain.py` stamps `metadata["active_brain"]` at exactly two sites: the shot
(line 2134) and the carry (line 2289). A **pass** — 61,199 of 227,332 events, by far the
most common action — carries no intent at all. **[read]**

### Break 2 — the exporter drops metadata

`_timeline_json()` serialises 15 fields. `metadata` is not one of them. Measured across
all 85 exported matches:

```
events with metadata: 0 / 227,332
events with active_brain: 0
```

Every intent, confidence, reason and decision-quality score the engine computes is
discarded at the last step. **[verified]**

### Break 3 — `brain_observatory.py` cannot run

Four independent faults in one 730-line file:

| Line | Fault |
|---|---|
| 254 | `*: int = 300` — parameter name lost, **SyntaxError** |
| 431 | `r.get('obeyed',?)` — `?` placeholder left in code, **SyntaxError** |
| 487 | imports `collect_decisions` — **function does not exist** |
| — | `auto_run_match.py:1415` imports `write_trace` — **function does not exist** |
| 660 | `--save-trace` writes `"decisions": []` with the comment *"empty – will be filled"*, and nothing fills it |

`auto_run_match.py` calls this inside `try/except Exception`, so in production it degrades
to `⚠️ Brain observatory skipped:` and the season continues. That is exactly why nobody
noticed: the failure is caught, printed once per match, and looks like a warning.

Corroborating evidence that this has been broken for a while: `temp_out.txt` at repo root
contains a shell traceback from someone trying `import brain_observatory as bo; bo.analyse_m...`,
and `tests/__pycache__/test_brain_observatory.cpython-314-pytest-9.1.1.pyc` exists while
`tests/test_brain_observatory.py` does not. **[verified]**

**Net effect: zero `*_brain_trace.json` files exist anywhere in the repository.** The
question "does the brain's choice track the situation?" is currently unanswerable with the
project's own tools.

---

## 8. The test suite has no working aggregate runner

```
pytest tests/  →  129 tests collected, 5 errors, 89 s
```

But there are **1,884 test functions across 145 files**. Individual files collect normally:

| File | Tests |
|---|---|
| `test_physics.py` | 58 |
| `test_brain_schema_alignment.py` | 39 |
| `test_attacking_matrix.py` | 24 |
| `test_body_physics.py` | 12 |

Cause: several modules execute on import, and `tests/test_brain_schema.py:225` ends with
`raise SystemExit(1 if fails else 0)`. pytest hits it during collection and aborts. **[verified]**

26 of the 145 `test_*.py` files contain no pytest-style test at all — 8 of them do not
even parse. These are scripts wearing a `test_` prefix.

Coverage of the most important file in the project:

| Module | Test files referencing it |
|---|---|
| `match_engine` | 81 |
| `event_chain` | 40 |
| **`situational_brain`** (the shipped brain) | **1** (25 tests) |
| `value_model` | **0** |
| `plofa_broadcast` | **0** |

The 25 tests in `test_situational_brain.py` pass. But a 14k-parameter recurrent network
that makes every on-ball decision in the engine is covered by one file, while the
`physics/` package — which is switched **off** by default — has 93 tests across two files.

**Where the tests do exist, they are excellent.** The three newest suites (goal vision,
flank selection, body physics) all pass, 32/32, in 11 seconds. `test_goal_vision.py` in
particular is a model of what a good test looks like: real trigonometry, mirror symmetry,
byline collapse, and a width invariant. The culture is right; the coverage is uneven.

---

## 9. Dead weight

### 9.1 Orphaned root modules — 32 of 132

Modules that nothing in the root imports:

```
analyze_excel            brain_evolution_old      brain_self_trainer
calibrate_gps            consequence_probe        consequence_surrogate
decision_journal         decision_proposals       engine_refinement_proof
evolve_all_parallel      evolve_brains            evolve_defensive_action
evolve_manager           evolve_offball           evolve_team_offball
fov_audit                gamestate_distill        gk_probe
outcome_discrimination_proof  patch_match_jsons   percy_v10
philosophy_proof         player_personality       plofa_export
production_roster_sanity season_stats             seed_confidence
situational_decision     small_game               sys_patch
wire_brain
```

Some are legitimate CLI entry points (`evolve_brains`, `season_stats`). Others are
abandoned experiments. `sys_patch.py` is a 3-byte BOM and nothing else.

### 9.2 `value_model.py` — the thesis, unimplemented

Its docstring is the most ambitious in the repository:

> *"This module learns V(s) — the expected chain-level consequence of being in vision
> state s — directly from real-match data … so credit assignment is temporal: an early
> progressive pass shares credit for the goal it set up."*

It is imported by exactly one module, `consequence_probe.py`, which is itself orphaned.
The consequence-learned value model has never touched the engine. **[verified]**

### 9.3 31 brain directories, 457 tracked files

`brains/` is live. The other 30 are snapshots, smoke tests and failed retrains
(`brains_smoke_v5st`, `brains_smoke_v6st_g80`, `brains_smoke_v7st_ms05`, …). All are
tracked in git.

### 9.4 Repository bloat

Largest blobs in git history:

| Size | Path |
|---|---|
| 73.8 MB | `output/gps_verify/gps_raw_m101.csv` |
| 73.5 MB | `output/gps_verify/gps_raw_m4242.csv` |
| 70.1 MB | `output/gps_verify/gps_raw_m202.csv` |
| 30.8 MB | `brains_offball/samples.json` |
| 13.9–17.7 MB | `season_stats.json` (committed repeatedly) |

`.gitignore` excludes `output/` and `*.json`, but these were committed before the rules
existed, so they remain tracked. `season_stats.json` is *explicitly whitelisted* back in
(`!/season_stats.json`) and churns 22 MB per matchday. The last uncommitted diff shows
`season_stats.json | 697,125 ++++----`.

`.git` is **354 MB** for a project whose source is 151k lines.

### 9.5 Stray files at root

`nul` (a Windows redirect accident, containing the text *"Could not find files for the
given pattern(s)"*), `~$PLOFA-2026-2027.xlsx` (an Excel lock file), `temp_out.txt` (a
debug traceback), `sys_patch.py` (BOM only), plus three deleted-but-unstaged diagnostic
directories.

---

## 10. Process health

### What is genuinely strong

1. **The documentation is a decision record, not a manual.** `DECIDER_BOUNDARY.md` states
   the problem, the measurement, three options, and *which one was chosen and why*. That
   is rare and it is the reason this audit could be written in a few hours instead of days.

2. **The culture measures before believing.** `BRAIN_COST_ANALYSIS.md` opens by refuting
   its own earlier projection: *"My earlier projection of 2.3–3.8× was wrong, and the
   reason is instructive."* That sentence is worth more than the benchmark.

3. **Determinism was hunted down properly.** `perception.py:220` documents that `hash()`
   of a string is `PYTHONHASHSEED`-salted, that three seeds produced three different
   matches, and that the fix was `crc32`. That is a subtle bug found and fixed correctly.

4. **The physics layer is additive by construction.** `MatchConfig.physics_enabled` defaults
   to `False`, and the doc's guarantee is that with it off, output is byte-identical. 11,750
   lines of work that cannot break the shipped engine — the right way to build an experiment.

5. **`AGENTS.md` corrects itself.** The "PRODUCTION BRAIN SET" section explicitly retires
   a claim that earlier sections still make, and tells the reader which docs are stale.

### What is weak

1. **Verification is not automated where it matters most.** The one question the project
   exists to answer — *is the brain's choice situationally driven?* — has no working tool
   and no test. Everything else is measured by hand.

2. **Silent degradation is the default.** 211 `except Exception` blocks. Most are
   deliberate resilience (the observatory must never kill a match), but the effect is that
   a completely broken subsystem prints one warning line and disappears. The observatory
   has probably been dead for days.

3. **The season is treated as a measurement instrument but is not controlled.** Mid-season
   xG model changes, no home advantage, one live match producing three red cards — the
   fixtures are played before the instruments are calibrated.

4. **Naming does not distinguish production from experiment.** `brains/`, `brains_v8/`,
   `brains_retrain2_g80/`, `brains_smoke_v7st_shared/` all look equally official. Only the
   manifest and the file size tell you which one plays.

---

## 11. What I would do, in order

### Tier 1 — restore the ability to see (half a day)

1. **Fix `brain_observatory.py`.** Three edits: `*, n: int = 300`; `'?'` instead of `?`;
   implement or remove `collect_decisions` and `write_trace`. Add `tests/test_brain_observatory.py`
   (the `.pyc` proves one existed).
2. **Make the exporter emit a decision ledger.** Do not dump the full metadata blob into
   every event — write a compact sidecar: `{minute, player, intent, confidence, zone,
   pressure, outcome}` per decision. ~40 lines in `exporter.py`, one file per match.
3. **Stamp `active_brain` on PASS.** It is the most common action and currently the only
   one with no record.

Until (1)–(3) are done, every claim about brain behaviour in the docs is unverifiable by
the project's own tooling.

### Tier 2 — close the two live football gaps (one day)

4. **Give `is_home` a job or delete it.** The simplest honest version: a small multiplier
   on shot quality and duel win probability for the home side, sized to produce the
   +0.3 goals/match the legacy seasons show. Then pin it with a test over 200 matches.
5. **Recalibrate xG per shot from 0.119 to ~0.105.** The conversion rate (10.7%) is already
   correct, so this is a pure shot-quality fix and it will stop every team looking like it
   underperforms xG by 25%.

### Tier 3 — stop the bleeding (one day)

6. **Fix collection:** guard the module-level `SystemExit` in `test_brain_schema.py`,
   move it under `if __name__ == "__main__"`. Then get `pytest tests/` to collect the real
   1,884 and record a baseline pass count.
7. **`git rm --cached` the three GPS CSVs and `brains_offball/samples.json`.** Stop
   committing `season_stats.json` — write a `season_summary.json` (a few KB) for the repo
   and keep the 22 MB file local.
8. **Archive the 30 non-live brain directories** into a single `brains_archive/` with a
   README table of what each one was and why it lost.

### Tier 4 — structural (a week, only if the project continues)

9. **Split `PossessionChain.generate`.** 2,586 lines is the ceiling on how fast this project
   can move. The natural seams are already visible in the code: `_attempt_pass`,
   `_attempt_carry`, `_attempt_shot`, `_resolve_duel`. Extract them one at a time, with the
   existing 40 `event_chain` test files as the safety net.
10. **Decide the defensive brain.** Either wire `brains_def/situational/*.npz` properly
    (the `AGENTS.md` next-steps list) or delete them. Leaving a gated-and-failed brain set
    in the tree while a September network plays is the worst of both.
11. **Decide `value_model`.** Wire it, or move it to `experiments/` with a note. Right now
    the most interesting idea in the repo is dead code.

---

## 12. The one-paragraph verdict

PLOFA is a serious piece of engineering with an unusual and admirable habit of writing
down what it learned, and it is further along than most hobby simulation projects ever get:
the football is recognisably football (84% pass accuracy, 10.7% conversion), the brain
architecture is real (recurrent attention over a perceived world), and the documentation is
better than most commercial codebases. Its weakness is not ambition or skill — it is that
the instrument has drifted out of calibration with the thing it is trying to measure. The
brain decides, and then the decision is thrown away before anyone can look at it; the
season is played, and then half of it is measured with a different xG model than the other
half; the home team is labelled home, and then treated exactly like the away team. Fix the
three observability breaks first. Everything else on this list becomes easier to argue
about once you can see what the network actually did.

---

## Appendix — reproduction

```bash
# import graph / orphans
python - <<'PY'
import os, ast, collections
mods = {f[:-3]: f for f in os.listdir('.') if f.endswith('.py')}
...
PY

# test collection
.venv/Scripts/python.exe -m pytest tests/ --collect-only -q
.venv/Scripts/python.exe -m pytest tests/test_physics.py --collect-only -q

# season metrics
sqlite3 alltime.db "select season, avg(goals), avg(xg) from team_match_stats t
                    join matches m using(match_id) group by season"

# decision census over exported timelines
python - <<'PY'
import json, glob
for f in glob.glob('plofa_output/*/*.json'):
    for e in json.load(open(f))['timeline']:
        assert 'metadata' not in e        # 227,332 events, 0 have it
PY

# observatory (currently fails)
.venv/Scripts/python.exe brain_observatory.py --home Oxton --away Natrican
# → SyntaxError: brain_observatory.py, line 254
```
