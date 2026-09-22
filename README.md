# PLOFA 26/27 — Football Match Simulation Engine

PLOFA is a full-match football (soccer) simulation engine written in Python.
It simulates complete 90-minute matches event-by-event — passes, carries,
shots, tackles, set pieces, cards, injuries, substitutions — driven by
player DNA attributes, personalities, manager profiles, tactics, referee
strictness, weather, and a chain-based possession model. Output is
StatsBomb-style: Excel workbooks, CSVs, JSON timelines, and tactical
visualisations (shot maps, pass networks, xG timelines, pressure maps).

Season state (form, fatigue, injuries, suspensions, minutes, league table)
persists across matchdays in `season_state.json`, and a between-matchday
**training system** develops players week to week.

---

## 1. Requirements & setup

- **Python 3.14** (project venv ships its own interpreter).
- All dependencies are installed in `.venv/` — use it for everything:

```powershell
cd C:\Users\Trevor Majani\Downloads\plofa_checkpoint6\plofa
& ".\.venv\Scripts\python.exe" -m pytest tests\tests.py -q
```

> Do **not** use the bare `python.exe` on PATH (it is a broken Windows Store
> stub). Always invoke `.\.venv\Scripts\python.exe` explicitly.

---

## 2. Running matches

### Automatic runner (default)

**`auto_run_match.py`** — edit only the `USER CONFIG` section (teams, date,
matchday, referee, weather, derby flag). Everything else is automatic:

- reads 300+ players from `PLOFA-2026-2027.xlsx`
- checks injuries / suspensions / fatigue from match history
- picks the best available XI per formation, builds the bench (max 7, always
  a backup GK), falls back to 2nd-team players if needed
- applies soul-player buffs, saves form/fatigue/injury state, updates the
  league table, exports all match files

> ⛔ **TREVOR ONLY — agents must NEVER run this file.** It overwrites the
> authoritative `season_state.json` on every run; re-running a played
> fixture corrupts the season ledger. For scratch/test simulations use
> `run_match.py` instead.

### Manual runner (scratch / experiments)

**`run_match.py`** — "the only file you edit week to week" for custom
matches: fill in match info, home/away squads, team styles, and soul
players, then run with the venv Python. Output goes to
`outputs/<HomeTeam_vs_AwayTeam_MD##>/`:

| File | Contents |
|---|---|
| `match.xlsx` | 8 sheets, full stats |
| `players.csv` | per-player match data |
| `match.json` | full event timeline |
| `shot_map.png` / `pass_network.png` / `xg_timeline.png` | tactical visuals |
| `match_summary.png` / `pressure_map.png` | tactical visuals |

### Season orchestration

**`season_manager.py`** — `SeasonState` (persisted to `season_state.json`)
plus `run_matchday(...)`, which plays a round of fixtures, applies pre-match
form/fatigue (`apply_pre_match`), records post-match effects
(`record_post_match`), runs the **training week** for every club (see §5),
advances the matchday, and saves. Pass `training=False` to skip the
training layer.

---

## 3. Architecture

### Match core

| Module | Role |
|---|---|
| `match_engine.py` | `MatchEngine`: owns the clock, timeline, phases, game state; absorbs event chains (`_absorb_chain`), executes substitutions (`_execute_substitution`), builds `MatchChronology`. `EventType` enum is the shared language of the sim. |
| `event_chain.py` | Possession chains (`ChainResult`): attack, penalties, restarts, VAR, offside. Chains report `goal_scored`, `restart_required`, `offside_detected`. |
| `decision_brain.py` | `DecisionBrain`: per-action AI — what a player tries (pass / carry / shoot) from DNA, personality, game state and tactical context. Covered by `test_decision_brain.py`. |
| `active_play_brain.py` | Live in-possession decision support. |
| `squad_manager.py` | Stamina/fatigue model, in-match injury rolls (`roll_injury`), availability, `SubstitutionController` (tactical / stamina / injury subs). |
| `position_engine.py` | Spatial state: player coordinates, halves reset at kickoff (`_reset_positions_to_halves`). |
| `threat_engine.py` / `threat_engine` | Attacking threat tracking, peaks on goals, resets at kickoff. |
| `tactical_ai.py`, `tactical_shapes.py`, `attack_patterns.py` | Formations, stances, attacking patterns. |

### Player & staff models

| Module | Role |
|---|---|
| `player_dna.py` | `DNAFactory`, `SquadBuilder` (`SquadBuilder.build(team, starters)["starters"]`, starters = `(name, pos, specialties, age)` tuples), attribute domains (physical / technical / mental / passing / defending / goalkeeper), `PlayerFormState` (confidence, fatigue, injuries). |
| `player_personality.py` | `PersonalityFactory`, `PersonalityTraits` — professionalism etc. drive training and behaviour. |
| `player_soul.py` | Soul players (season-defining stars, e.g. Percy) with greatness pillars. |
| `manager_profile.py` | `ManagerPool` / `ManagerProfile` (risk tolerance, possession preference, …) — drives tactics and training focus. |
| `referee_pool.py` | Referee strictness model (cards, fouls). |
| `squad_chemistry.py` | Chemistry effects on pass completion. |

### World & physics

`weather_physics.py` (clear/rain/wind/fog effects), `possession_physics.py`,
`possession_phases.py`, `pitch_control.py`, `geometry_engine.py`,
`virtual_gps.py` (player tracking data), `marking.py`, `pressing_profiles.py`,
`chance_creation.py`, `set_piece_routines.py`, `sequence_engine.py`,
role behaviour modules (`striker_behavior.py`, `winger_behavior.py`,
`midfielder_behavior.py`, `fullback_behavior.py`).

### Outputs & analysis

`exporter.py` (`PLOFAExporter` — xlsx/csv/json/png), `pass_network.py`
(`PassMatrix`), `opta_stats.py` / `opta_analytics.py`, `season_stats.py`,
`player_maps.py`, `cross_detector.py`, `long_pass_detector.py`,
`pass_classifier.py`, `advanced_valuation.py`. Historical/aggregate outputs
live in `plofa_output/`, `outputs/`, `output/`; `plofa_streamlit/` holds the
dashboard app; `build_app_data.py` prepares its data.

### Specs

Behavioural specs live in `.kiro/specs/` (`match-simulation-refinements`,
`attacking-matrix`, `defensive-awareness`, `weather-physics`).

---

## 4. Event model

Every match produces a `result.timeline` of `MatchEvent`s:

```python
MatchEvent(minute=..., second=...,
           event_type=EventType.PASS, team=..., player=...,
           phase=MatchPhase.PEAK_INTENSITY, game_state=GameState.LEVEL,
           metadata={...})
```

Key types: `PASS`, `PROGRESSIVE_PASS`, `SWITCH_OF_PLAY`, `THROUGH_BALL`,
`CROSS_SUCCESS`, `GOAL`, `OWN_GOAL`, `PENALTY_SCORED`,
`VAR_DISALLOWED_GOAL`, `SUBSTITUTION`, plus the checkpoint-6 additions below.
Clock accounting (`MatchChronology`) splits every match into measured play
vs `dead_time` via chain clock marks.

---

## 5. Checkpoint-6 subsystems (new)

### Goal celebrations — `EventType.GOAL_CELEBRATION`

After every goal (open play, own goal, penalty — all flow through the
`goal_scored` branch of `_absorb_chain`), the engine pauses 10–30 s,
advances the match clock, and emits a `GOAL_CELEBRATION` event with
`metadata={"duration": <seconds>}`. The chain clock mark is folded at the
pre-celebration clock, so celebration time lands in `dead_time`, never in
measured play. Durations are drawn from a dedicated cosmetic RNG
(`_COSMETIC_RNG`) so presentation randomness never perturbs the seeded
football sequence.

### Injury timeline events — `EventType.INJURY`

In-match injuries were already rolled by `squad_manager` and already forced
substitutions — but nothing recorded the injury itself. Now, whenever
`_execute_substitution` handles an injury-forced sub, it emits an `INJURY`
event **before** the `SUBSTITUTION` event (injury → change causality), with
`injury_type` / `injury_severity` / `injury_minute` metadata from the
stamina state.

### Training weeks — `training_system.py`

A between-matchday layer that runs in `season_manager.run_matchday()` after
the fixtures, before save. For every player:

- **Professionalism** (personality) sets effectiveness — pros sharpen,
  coasters can regress.
- **Manager emphasis** sets focus — high-risk → attack, risk-averse →
  defense, possession-first → tactics/timing; role defaults per position.
- **Young players (<24)** gain bounded attribute development in the focus
  domains; veterans hold level.
- **Confidence** drifts toward a training-sharpness band; **fatigue**
  recovers faster for professionals.
- Everything commits into `SeasonState`, so it persists and feeds next
  week's `apply_pre_match`.

```python
from training_system import TrainingSystem
system = TrainingSystem()
records = system.run_week(players, team_name="Hartwell City",
                          matchday=3, state=season_state, manager=mgr)
print(system.report_text(records, "Hartwell City"))
```

---

## 6. Tests

Run from the repo root with the venv Python:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests\tests.py -q
& ".\.venv\Scripts\python.exe" -m pytest tests\test_chronography.py tests\test_possession_causality.py -q
& ".\.venv\Scripts\python.exe" -m pytest tests\test_checkpoint7_subsystems.py -q
```

| Suite | Covers |
|---|---|
| `tests/tests.py` | Core regression: pass matrices, xG, positions, chemistry, realism |
| `tests/test_chronography.py` | Clock accounting: measured play vs dead time, monotonicity |
| `tests/test_possession_causality.py` | Possession chain cause/effect |
| `tests/test_checkpoint7_subsystems.py` | **New (P1–P7):** celebration emission + clock, injury-event ordering, training records/commitment, young-pro development, full-match smoke |
| `test_decision_brain.py`, `test_width_changes.py` (root) | DecisionBrain behaviour, pitch-width fixes |

### Known pre-existing failure (do not "fix" blindly)

`tests/tests.py::test_pass_matrix_sums_match_real_events` fails on pristine
code too. Root cause (verified): `PassMatrix.build` (`pass_network.py:99`)
counts `CROSS_SUCCESS` events into the matrix **without an outcome check**,
while the test's manual tally only counts completed passes
(`PASS`/`PROGRESSIVE_PASS`/`SWITCH_OF_PLAY`/`THROUGH_BALL` with
`outcome=True`). The gap (`matrix − manual`) always equals the number of
successful crosses — a test-vs-production definitional mismatch, not a sim
bug. Fixing it means deciding whether a successful cross is a completed
pass (production semantics) or not (test semantics), which changes exporter
numbers — out of scope for this checkpoint.

---

## 7. League data

`PLOFA-2026-2027.xlsx` is the roster source of truth (300+ players).
`roster_loader.py` reads it; `season_state.json` / `manager_state.json` /
`referee_state.json` / `season_stats.json` are the mutable season ledger —
back up before experimenting (several `.backup`/`.corrupt-*` snapshots exist
from past incidents).

---

## 8. Battlefield validation vs real football

**When:** calibration sessions `2026-09-17` → validation `2026-09-18`
(day-1 and day-2 calibration, final 48-match run 2026-09-18).

**What it is:** `battlefield/` is the quantitative test harness that plays
PLOFA in-memory, tallies per team-match features, and compares them against
a real corpus (StatsBomb open data, 102 La Liga matches) with
KS-tests and ±15% bands. Full results live in
`battlefield/BATTLEFIELD_REPORT.md` (regenerated by `battlefield/compare.py`).

**How to reproduce (mandatory — see the determinism note below):**

```powershell
& ".\.venv\Scripts\python.exe" -m battlefield.probe_zones --matches 16 --seed 2026
& ".\.venv\Scripts\python.exe" -m battlefield.run_plofa --matches 48 --seed 2026 --out battlefield\data\plofa_feats.json
& ".\.venv\Scripts\python.exe" -m battlefield.compare
```

### 2026-09-18 result (48 PLOFA matches vs 102 real): 10/16 metrics within ±15%, 9/16 KS tests pass

| Metric | Unit | Real | PLOFA | Ratio | Within 15% |
|--------|------|------|-------|-------|------------|
| Shots | per team-match | 12.147 | 12.302 | 1.013 | ✅ |
| xG | per team-match | 1.426 | 1.416 | 0.993 | ✅ |
| Shots on target | per team-match | 4.804 | 5.062 | 1.054 | ✅ |
| SOT share | % of shots | 39.752 | 41.093 | 1.034 | ✅ |
| Corners | per team-match | 4.623 | 5.292 | 1.145 | ✅ |
| Pass accuracy | % | 82.416 | 79.938 | 0.970 | ✅ |
| Possession | % | 50.000 | 50.000 | 1.000 | ✅ |
| Fouls | per team-match | 14.402 | 13.323 | 0.925 | ✅ |
| Yellow cards | per team-match | 1.701 | 1.844 | 1.084 | ✅ |
| Away goals | per match | 1.480 | 1.583 | 1.070 | ✅ |
| Home goals | per match | 1.794 | 2.229 | 1.242 | ❌ |
| Total goals | per match | 3.275 | 3.812 | 1.164 | ❌ |
| Inside-box shot % | % of shots | 66.359 | 41.915 | 0.632 | ❌ |
| Passes | per team-match | 538.113 | 425.917 | 0.792 | ❌ |
| Offsides | per team-match | 0.152 | 4.958 | 32.629 | ❌ |
| Ball recoveries | per team-match | 42.108 | 17.719 | 0.421 | ❌ |

**Known residual gaps (as of 2026-09-18):**
- **Home-goal inflation** (+24%) drives total goals (+16%) — the largest
  *behavioural* defect; investigate the home-advantage model.
- **Inside-box shot %** (41.9 vs 66.4) — too many edge-of-box attempts; the
  attacking-shape/zone distribution needs work, not just the volume taps.
- **Passes** (0.79×) is definitional: PLOFA emits CARRY/BALL_RECEIPT as
  separate events while StatsBomb counts only pass-family events (see
  report's fair-accounting notes). Offsides and ball recoveries are also
  counting-definition gaps, not sim bugs.

### Two calibration-era discoveries (2026-09-17) — read before touching shot volume

1. **Determinism is NOT on by default.** Python's per-process hash
   randomization (`PYTHONHASHSEED`) changed dict/set iteration order between
   runs, so identical code produced ±15% swing in outcomes (verified: goals
   1.33 vs 2.17 for the same seed). **Always set `$env:PYTHONHASHSEED="0"`**
   before any battlefield (or seed-reproducibility) run;
   `battlefield/run_plofa.py` guards it with `ensure_deterministic_process()`.
   Every calibration measurement taken before this fix is untrustworthy.
2. **The per-minute shot funnel in `match_engine.py` is dead code.**
   Instrumentation proved it executes ~once per match and never passes its
   roll — shot volume comes from the chain matrix take-gates in
   `event_chain.py` (`(shot_score - floor)/0.30`, both sites) and the
   geometry distance gate in `_select_action_from_position`. Calibration of
   shot volume happens via those gates and the XGEngine distance sharpening;
   the funnel's `shots_per_sequence` knob is inert.

**Calibration knobs applied 2026-09-17/18 (engine layer only; brains untouched):**
- take-gate floors `0.60 → 0.63` (`event_chain.py`, two sites)
- XGEngine distance sharpening `×0.25/0.45/0.62 → ×0.22/0.41/0.57`
- keeper spill `0.42 → 0.47` (lifts goals/xG toward real ~1.15)
- corner sites trimmed: parry base 0.34→0.30, woodwork 0.40→0.36,
  blocked 0.42→0.38, deflection 0.40→0.37

---

## 9. All-time stats database (`alltime.db`)

**When:** built 2026-09-18, after the battlefield validation above.

**What it is:** a permanent SQLite warehouse (schema in `alltime_db.py`)
holding every PLOFA match ever played in one place, Opta-style — match
results, per-team per-match stats, per-player per-match stats, season
standings, goals, and league-shape records. Rows are tagged with a
**fidelity** flag so old data is never confused with current data:

| Fidelity | Seasons | Source | Notes |
|----------|---------|--------|-------|
| `legacy` | 24/25, 25/26 | `D:\TOLAND FOOTBALL FEDERATION\PLOFA-2025-2026.COM\PLOFA-ALL-TIME.xlsx` | 612 matches, 16.5k player-match rows, 68 standings rows, 2479 raw goal-log rows. Older StatsBoss-era numbers are **not as true** as today's engine; treat as best-effort history. |
| `v26` | 26/27 (live) | exporter JSON in `plofa_output/` | 37 matches synced so far (MD1–4 fixture set); every future match appends automatically. |

**CLI (run inside the repo venv; keep `PYTHONHASHSEED=0` for reproducibility):**

```powershell
& ".\.venv\Scripts\python.exe" alltime_db.py init            # create schema (idempotent)
& ".\.venv\Scripts\python.exe" alltime_db.py import-legacy   # read the xlsx above
& ".\.venv\Scripts\python.exe" alltime_db.py sync            # ingest everything under plofa_output\ (idempotent)
& ".\.venv\Scripts\python.exe" alltime_db.py report          # per-season summary + all-time leaderboards
```

**After every matchday you play:** just re-run `sync` — it scans
`plofa_output\` and appends any new exporter JSON it hasn't seen yet
(`sync` reports `scanned`/`ingested`). Run `report` to see updated totals.

`sync` de-duplicates on `(season, matchday, home, away)` and goals on
`(season, matchday, team, minute, scorer)`, so re-running is safe. The DB
file is gitignored; the tables are:

- `matches` · `team_match_stats` · `player_match_stats` (≈110 stat columns)
- `player_season_stats` · `season_standings` · `players` · `teams` · `seasons`
- `goals` (26/27+) · `legacy_detailed_goals_raw` (raw goal-log archive)
- `meta` (schema version + notes)

Legacy seasons currently live at player/aggregate fidelity (no `team_match_stats`
rows); team-level legacy queries are one `GROUP BY` over `player_match_stats`
away. Tests: `tests/test_alltime_db.py` (synthetic 26/27 package + idempotency).

### Player renames / name variants — one identity per person

PLOFA renames players between seasons (e.g. `Victor James` → `Rayan Victor James`)
and the legacy sheets squeezed punctuation/accents (`D John` vs `D. John`,
`Roy Steupy` vs `Roy Steupŷ`). Without handling, the same human becomes two
rows forever. The `alias` commands fix that retroactively AND forward:

```powershell
# register a name as another name of the same person, then rewire stored rows:
& ".\.venv\Scripts\python.exe" alltime_db.py alias add "Victor James" "Rayan Victor James" --apply

# auto-propose same-person pairs (same club across seasons + same position):
& ".\.venv\Scripts\python.exe" alltime_db.py alias scan

# show/govern registered aliases:
& ".\.venv\Scripts\python.exe" alltime_db.py alias list
& ".\.venv\Scripts\python.exe" alltime_db.py alias apply --all
```

Once registered, every future `sync`/ingest maps the old spelling to the
canonical identity automatically, and `report` aggregates across the variants.
`alias scan` only proposes — it never merges without `alias add ... --apply`.

Applied 2026-09-18 (8 confirmed renames): `D John→D. John`,
`Franća→Diederik Franća`, `Hee Jo→Wang Hee Jo`, `Onike-Lisim→Onike Lisim`,
`Roy Steupy→Roy Steupŷ`, `Tony-Belé→Tony Belé`, `Vuwo Urida→Benard Vuwo Urida`,
`Omar-Seet→Omar Seet` (Pearls 24/25 → Seafcea 25/26 → Justice 26/27).
Canonical = the spelling you use today; old spellings auto-map forever.
`Rayan Victor Jam¢s` was already a single identity (pre-fixed; Justice → Pearls).

---

## 10. 2026-09-22 — Match event-count collapse & restoration (Checkpoint 32)

**Symptom:** post-Checkpoint-6 seed runs dropped the match timeline from
~3000+ events to ~1900–2000 (Hartwell vs Thornfield MD1, seed 1031). Chains
shrunk to ~4 events and possessions died early.

Two independent regressions caused it. Each was isolated with controlled
swaps (same-brains A/B, consequence critic disabled via `PLOFA_CONSEQUENCE=0`,
HEAD-module overlay runs) before any change was made.

### 10.1 Brain over-switching (fixed)

`brains/surrogate_pos.json` carried sparse SWITCH cells with lucky,
goal-weighted payoffs that out-priced the safe options (e.g. CM
midfield-pressure SWITCH=2.31 vs SAFE_PASS=1.78). Evolution optimises the
ARGMAX of `expected_success`, so the converged per-position nets learned
"SWITCH is the best midfield play" and emitted it ~1 in 4 touches — and the
engine's forced-switch delivery truncated possessions.

`brains_trainer/rebalance_switch_surrogate.py` produced
`brains_trainer/surrogate_switchfix.json`, which caps SWITCH expected-success
in every own-half/middle bucket strictly below the best safe option
(SAFE_PASS / RECYCLE / PROTECT_POSSESSION / PROGRESSIVE_PASS), leaving
final-third switches untouched. All 11 position brains were retrained
against it with the identical pipeline (same arch/features/generations/seed —
verified structure-identical, only weights differ) and promoted into
`brains/` (pre-switch snapshot kept in
`brains_backup_pre_switchfix_20260922_101516/`). Result: `SWITCH` intents
164→~50, `SWITCH_OF_PLAY` events 163→~60.

### 10.2 The forced-receiver layer was disabled (the dominant loss)

The real head-versus-current difference was a behavioural flag:
`PossessionChain.POLICY_INTENT_AUTHORITY` (event_chain.py). It had been
flipped to `True`, making the neural brain the **only** receiver selector
and downgrading every deterministic layer to advice:

- the phase engine's `RELEASE_TO_GK` / `EMERGENCY_DROP_TO_GK` directives
  (`regression_mode`) were zeroed before they could pick the keeper
  (`event_chain.py` `if cls.POLICY_INTENT_AUTHORITY: regression_mode = None`),
- the AttackingMatrix pass target and the Checkpoint-24 wide-combo override
  were gated off.

The pre-flip code (your 2026-09-20 GitHub upload and the pre-Checkpoint-21
head) forces a real receiver on pass-like touches — matrix target or phase
directive or wide-combo — and delivers to it. That "forced receiver"
guarantee is what produced the 3000–3700-event matches and the modern
12–30-touch keeper lines. Flipping authority on starved the keeper
(~4–11 GK receptions/match, your report) and collapsed match events to
~1900–2000 in one stroke.

### 10.3 Delivery-range vs spacing decoupling (secondary symptom)

With authority on, every pass went through `_pass_destination_to_receiver`,
whose cap (`min(d, max(pass_dist, 3))`) was calibrated when receivers sat
~15m apart; the Checkpoint-21-onwards shape/stretch layer sits them ~26m
out, so ~48% of passes died `underhit`. Checkpoint 32 re-couples the
delivery reach to actual spacing (`0.93 × separation`, 30m ceiling; long
passes untouched). Constants: `REACH_SPACING_TARGET`, `REACH_SPACING_CEIL_M`.
Plus the GK is now a `SAFE_PASS` outlet again in the defensive third
(`_find_target._nearest_teammate`, brain_integration.py).

### 10.4 The fix: authority re-enabled + neural brain retained

`POLICY_INTENT_AUTHORITY` defaults to `False` (Checkpoint 32b). The neural
policy still samples the intent; the forced-receiver layers guarantee
deliveries and keeper involvement. The switchfix brains (§10.1) remain the
loaded policy, so the earlier SWITCH overpricing stays fixed.

### Verified end state (seed 1031)

timeline **~3100–3400** (restored into the old 3000–3700 band; two runs
3172 / 3394) · PASS ~770–800 (≈690–730 completed) · `SWITCH_OF_PLAY`
80–86 · GK receptions via pass **~51–58/match** (~30–39 home, ~12–19 away;
was ~18 total) · GK primary actions ~57 receipts / ~55 passes — real
build-up keeper territory · `TURNOVER` ~140–165.

> Runs still carry per-process RNG variance; set `$env:PYTHONHASHSEED="0"`
> for reproducible comparisons (see §8). Run with the prev-generation brains
> in `brains_switchfix/` via `PLOFA_BRAIN_DIR=<path>` if you ever need the A/B
> baseline.
