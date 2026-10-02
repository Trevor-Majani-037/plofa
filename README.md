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

**`physics/` — the continuous physical-time layer.** A standalone package with
zero module-level imports from the live pipeline, so it can be built and
calibrated without touching it. Off by default (`MatchConfig.physics_enabled`).
When enabled it is consulted at four points in a live match:

| Point | Where | What it adds |
|---|---|---|
| Pass block | `event_chain.py` — between `_pass_block_probability` and its dice roll | The engine asks *is he near the lane?*; this asks *can he get there before the ball does?* |
| Keeper save | `event_chain.py` — between `_is_shot_savable` and the xG division | `effective_reach` has no flight time in it, so it cannot tell a shot from 30 m (1.20 s) from one at 8 m (0.27 s) |
| Player movement | `match_engine.py` — after the off-ball step | Momentum, an acceleration limit on turning, and fatigue in the legs. The engine's shape logic still decides the target; the physics only decides how the body gets there |
| Ball flight time | `event_chain.py` — the receiving event's timestamp | A 40 m ball in behind used to be logged arriving at the instant it was struck |

The first two can only *remove* credit the physics cannot justify, never add
any. The last two refine what the engine already computed. Full report,
including what is deliberately **not** done: [`docs/PHYSICS-LAYER.md`](docs/PHYSICS-LAYER.md).

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
| `tests/test_world_ids.py` | World identity: ID minting, name↔ID reversibility, collision loudness, live roster coverage |
| `tests/test_world_model.py` | World schemas: JSON round trip, ID-only reference graph, per-season stadium capacity |
| `tests/test_world_competition.py` | World competition: rules-as-data, additive context fragment, league + knockout progression |
| `tests/test_world_calendar.py` | World calendar: **fixture-collision guarantee**, determinism, priority, rest, postponement |
| `tests/test_world_ledger.py` | World ledger: **plan §18 one-continuous-state**, scoped vs carry-across, availability, 26/27 write guard |
| `tests/test_world_ingest.py` | Engine→world crossing: verbatim pass-through, identity, and a **real simulated match** |
| `tests/test_world_qualification.py` | Promotion/relegation + continental entry, all rules as data; a tie that was not played is never decided |
| `tests/test_world_month.py` | Real 2026-27 dates and rules; the calendar against the real football schedule; the §18 continuity checker |
| `tests/test_alltime_db_competitions.py` | Warehouse competition keying: additive, idempotent, checksum-verified, refuses unknown schemas |
| `tests/test_extra_time.py` | **Extra time + penalty shootouts** (the engine-touching phase): the 26/27 gate, ET, the shootout, two-legged aggregates |

### World-layer suites (audit §16, phases 1–8)

```powershell
# fast layer — no match simulation
& ".\.venv\Scripts\python.exe" -m pytest tests\test_world_ids.py tests\test_world_model.py tests\test_world_competition.py tests\test_world_calendar.py tests\test_world_ledger.py tests\test_world_qualification.py tests\test_world_month.py tests\test_alltime_db_competitions.py tests\test_alltime_db.py -q

# crossing layer — runs a real match, ~90s
& ".\.venv\Scripts\python.exe" -m pytest tests\test_world_ingest.py -q

# engine-touching phase — runs many real matches, ~10-15 min
& ".\.venv\Scripts\python.exe" -m pytest tests\test_extra_time.py -q

# the audit §20 proof harness
& ".\.venv\Scripts\python.exe" -m world.proof
```

The world layer is imported by **nothing** in the live 26/27 pipeline — only by
its own tests. See §11.

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

---

## 11. World layer (`world/`) — Phases 1–4

The **world** is everything around the match: countries, cities, stadiums, clubs,
people, competitions, and the calendar that decides *when* anyone plays. It is
being built against the World Football plan and `PLOFA_WORLD_LAYER_AUDIT.md`,
which audits the gap between today's single-league PLOFA and that plan.

**The one invariant that never changes: no competition runs its own engine.**
The world layer selects *which* match, *when*, and *under which rules*; every
match still rounds-trips through the same `MatchEngine` (plan §10).

### The isolation guarantee

`world/` is imported by **nothing** in the live 26/27 pipeline — only by its own
tests. It never writes to `season_state.json`, `season_stats.json`,
`manager_state.json`, `referee_state.json` or `plofa_output/`, and it reaches the
live types (`LeagueTable`, `FixtureList`, `MatchConfig`) only through lazy or
duck-typed imports at call time. Both properties are enforced by tests, not by
convention.

### Modules

| Module | Role |
|---|---|
| `world/ids.py` | Canonical ID minting + the reversible name↔ID adapter and alias layer |
| `world/model.py` | Typed world schemas (Country, City, Stadium, Club, Player, Manager, Referee) + JSON (de)serialisation |
| `world/competition.py` | `CompetitionRules` as data, the additive `MatchContextFragment`, `LeagueCompetition`, `KnockoutCompetition` |
| `world/calendar.py` | Multi-competition fixture scheduler with a hard no-collision guarantee |
| `world/ledger.py` | World-facing state store: one continuous player state across every competition |
| `world/ingest.py` | Engine → world translator: real `MatchResult` → ledger, reusing the exporter's own lines |
| `world/realworld.py` | **Real 2026-27 rules and published fixture dates** as data (PL, UCL, EFL Cup, FA Cup) |
| `world/testworld.py` | The controlled test world built on those real rules and dates (plan §17) |
| `world/month.py` / `world/runmonth.py` | The one-month integration test (plan §18) and its runner |
| `world/proof.py` | The audit §20 proof harness (`python -m world.proof`) |

### Identity

IDs are **derived, never ordinal**: `mint_id(kind, key)` takes a crc32 of the
canonical key and renders it base36, so `"Hartwell City"`, `"hartwell  city"`
and `"  HARTWELL CITY "` are one identity, and inserting a new entity never
renumbers the existing ones. Player identity is club-qualified
(`player_key(club, name)`) so two players with the same name in different clubs
never merge. Diacritics are preserved — a real identity never loses
distinguishing marks.

Three operations, deliberately separated so a merge can never be silent:

- `register(...)` — attach a name to an ID. Idempotent for the same key;
  **raises** if a *different* canonical key would land on an owned ID, including
  when `id_override` supplies the ID explicitly.
- `recannonicalise(...)` — the only way an existing identity's canonical name
  changes (a real rename, e.g. `Victor James` → `Rayan Victor James`). The old
  spelling keeps resolving by default, matching the `alltime_db.alias`
  retro-compatibility rule.
- `validate()` — the integrity gate: every registered name must reverse to its
  own ID.

The live 26/27 roster currently resolves to **18 clubs and 433 players**
(`EXPECTED_CLUBS` / `EXPECTED_PLAYERS` in `tests/test_world_ids.py`), all names
unique roster-wide. A sample of club digests is **frozen** in that suite — if the
canonical-key rule or the digest ever changes, those flip and any world ledger
already written stops resolving, so the change must be deliberate.

### The calendar and its one hard invariant

Plan §12: *"The system must not accidentally schedule the same team to play two
matches at the same time."*

`Calendar.build()` places every planned fixture from every competition and
guarantees:

1. **No club plays twice on one calendar day** — the hard invariant.
2. **Minimum rest** (default 3 days) between any two of a club's fixtures,
   overridable per competition.
3. **One venue, one match, one day** — a stadium is a resource like a squad.
4. **Contested dates go to the higher-priority competition**, so a cup tie is
   never squeezed out by the league just because the league was generated first.
   `continental_midweek` (300) > `cup_midweek` (200) > `league_saturday` (100).
5. **Fully deterministic** — placement order is a total order over stable keys,
   so a calendar is reproducible with no RNG and no dependence on dict/set
   iteration order. `test_build_is_deterministic_across_processes` runs the
   build in two fresh interpreters under *different* `PYTHONHASHSEED` values and
   requires byte-identical output.

A fixture that cannot be placed inside the horizon is reported as an
`unscheduled` violation and left out — never silently dropped, and never allowed
to break an invariant. `validate()` re-derives every invariant from the placed
fixtures alone, independently of the placement code, because an invariant
checked only by the code that enforces it is not checked at all.

`postpone()` searches forward for the first date that satisfies every invariant
rather than merely rolling to the next matching weekday — in a league where every
club plays every week, no candidate weekday is ever free, so a naive
postponement could never succeed. `reschedule()` refuses outright rather than
applying a move that would create a collision.

### The ledger — one continuous player state

Plan §7 is the reason `world/ledger.py` exists:

> A player's state must not reset simply because he moves from the league to a
> cup or Champions League. […] The player entering the Champions League must
> remember what happened on Saturday.

The ledger is built around one distinction, and conflating these two is the bug
the design exists to prevent:

| | Lives where | Survives a competition boundary? |
|---|---|---|
| **Carry-across** — fatigue, fitness, injuries, suspensions, cards, confidence, form, workload, development | once per player, top level | **yes** — that is the point |
| **Scoped** — appearances, minutes, goals, assists, shots, xG, cards | `per_season[season]["per_competition"][competition]` | no — league and continental rows *coexist* (plan §8) |

Career totals are a **query** over the scoped lines, never a second counter that
can drift; `SCOPED_TO_CARRY_ACROSS` declares which continuous field each scoped
stat accumulates into, and the suite asserts the two lists cannot diverge.

The ledger deliberately does **not** model fatigue, injury probability or
development. Those belong to `squad_manager` / `training_system`, and plan §13
is explicit that existing logic is reused rather than duplicated. Callers push
engine-computed values in via `apply_post_match`; the ledger owns *scoping and
continuity*, not physics.

Suspension is resolved against the **competition's own** rule, because a
five-yellows-in-six ban in the league need not apply in a cup with a different
accumulation window. The window is counted in matches, not days, and every
yellow is stamped with the match index at which it was earned — a card that is
not recorded is a card the accumulation rule can never see.

Injuries and suspensions expire on matchday boundaries via `settle()` /
`advance_matchday()`. Without that step a player is unavailable **forever**,
which silently ends a season; `record_matchday()` bundles record-then-advance as
the safe default.

**The 26/27 dual-write rule (audit §14).** The ledger refuses, in code, to open
`season_state.json`, `season_stats.json`, `manager_state.json`,
`referee_state.json` or `alltime.db` as its own store — a stray write there
corrupts a season that cannot be replayed. `read_plofa_season_state()` reads the
live ledger **read-only** and hands back copies; `import_plofa_season_state()`
seeds only season-level facts, never duplicating per-player state that the live
`SeasonState` remains authoritative for.

### The crossing: real engine output → world ledger

`world/ingest.py` is where the two halves of the world layer meet. Its one
governing rule is that **the ledger never re-derives anything the engine already
decided** — plan §13 requires existing squad/fitness logic to be reused, not
duplicated, so this module is a translator and nothing more:

| Value | Source | Treatment |
|---|---|---|
| per-player line (minutes, goals, assists, shots, xG, cards) | `exporter.PLOFAExporter.accumulator.stats` — the same dict the xlsx/csv/JSON use | verbatim, so the world and the published reports can never disagree |
| fatigue / fitness | `squad_manager.SubstitutionController.stamina` | verbatim, with a lossless 0-100 → 0-1 unit mapping |
| injuries | `PlayerStaminaState.is_injured` / `injury_type` / `injury_minute` | verbatim |
| confidence / form | `SeasonState` (not on `MatchResult`) | caller-supplied; defaults to `None` = "engine did not say" |

Two consequences worth knowing:

- **`shots` is a definitional sum.** The exporter splits them into
  `shots_on_target` / `shots_off_target` / `shots_blocked_att` and has no single
  key, so the ledger's one field is their sum — pinned by name in
  `world.ingest.SHOT_PARTS` so the choice is visible rather than buried.
- **`MatchResult` does not carry the substitution controller.** Post-match
  stamina lives on the `SubstitutionController` the caller created and passed to
  `set_stamina_controller`, so `apply_result(..., sub_controller=...)` is
  *required* for fatigue and injuries to cross. Omit it and those fields stay
  absent — the ledger reads `None` as "keep what you had" rather than inventing
  a plausible number. Found by the real-match test, not assumed.

`apply_competition_context()` is the reverse direction and the only hand-off to
the engine: it folds a competition's rules into a real `MatchConfig` and returns
a **new** config, never mutating the input.

### Competition keying in the warehouse (audit §16 phase 7)

`alltime.db` assumed one competition per season. Three constraints encoded that,
and each broke the moment the same two clubs could meet in a league *and* a cup:

```
matches             UNIQUE(season, matchday, home_team_id, away_team_id)
player_match_stats  PRIMARY KEY (season, match_date, team_id, player_id)
goals               UNIQUE(season, matchday, team_id, minute, scorer)
```

`player_season_stats` and `season_standings` had no competition dimension at
all. SQLite cannot add a column to a PRIMARY KEY or UNIQUE constraint, so this is
a controlled table rebuild in `alltime_db_competitions.py`:

- **refuses to guess** — each rebuild declares the exact constraint text it
  expects; an unrecognised schema aborts with nothing changed;
- **preserves data** — row counts *and* per-table content checksums are compared
  before/after, and a mismatch rolls back;
- **respects foreign keys** — off during the rebuild, `foreign_key_check`
  afterwards;
- **is idempotent** — a second run reports "already migrated";
- **backs up first** — `alltime.db.pre-competition-keying`, plus WAL sidecars.

```powershell
& ".\.venv\Scripts\python.exe" alltime_db.py migrate-competitions --dry-run
& ".\.venv\Scripts\python.exe" alltime_db.py migrate-competitions
& ".\.venv\Scripts\python.exe" alltime_db.py competition-status
```

**Applied 2026-09-25** to the live warehouse: 3 tables rebuilt, 2 columns added,
19,467 rows backfilled to `competition_id='CMP-PLOFA'`, row counts and content
checksums preserved, 0 FK violations, `integrity_check: ok`. A replay *within* a
competition is still correctly refused — only cross-competition meetings are now
legal. The rebuild left 138 MB of free pages, reclaimed with `VACUUM`.

### Extra time & penalty shootouts (audit §16 phase 5)

The only phase that touches the match engine, and therefore the one with the
strictest gate. Everything is behind two `MatchConfig` flags that both default
to `False`:

```python
MatchConfig(..., extra_time=True, penalties=True)   # a knockout tie
```

- **Extra time** runs through the *same* `_run_minute` closure as the first 90 —
  there is no simplified "cup mode". Same brains, chains, physics and
  substitutions, on tired legs. It ends the moment a side leads by two, or when
  the 30 minutes are up, whichever comes first. The interval before the second
  period buys 6% recovery against half-time's 18%.
- **The shootout draws from `_COSMETIC_RNG`**, the dedicated cosmetic stream —
  *not* the football RNG. A shootout is the resolution of a tie, not part of the
  football; if it consumed the seeded sequence it would change every subsequent
  event. This is the same discipline already used for goal-celebration durations,
  and there is a test that asserts the football stream is untouched.
- Kick order is outfielders → keeper → substitutes, and the shootout emits a
  `PENALTY_SCORED` / `PENALTY_MISSED` event per kick. **A shootout is never a
  match goal** — it appears in `MatchResult.shootout`, not in the scoreline.
- `winner_team` is deliberately `""` for an ordinary league match: a league
  match has no winner concept, and reporting the leader would leak knockout
  semantics into every warehouse row.
- Two-legged support: `MatchConfig.aggregate` carries the score *before* the leg
  and the result reports the running total; `away_goals_rule` breaks a level leg
  from the standing total. The leg result is always checked first — you cannot
  lose a leg and advance.

### The phase-5 gate, and why it counts executions

The claim is "a 26/27 league match never executes a line of the new code". That
is proved **runtime-verbatim**: the guard `if self.config.extra_time or
self.config.penalties:` is asserted to exist and to sit after the 90 minutes, and
a default match is asserted to leave every new state field at its neutral value
with no shootout event on the timeline.

It is deliberately **not** proved by comparing simulated output, because the live
match is not reproducible from `random.seed()` — two fresh processes with the
same seed and `PYTHONHASHSEED=0` produce different event timelines (verified
2026-09-25, pre-existing, unrelated to this phase; the live path draws on an
entropy-seeded source that `random.seed` and `np.random.seed` do not control).
A fingerprint comparison would be a *false* gate: it would report "regression" on
a green engine. Counting executions cannot lie.

> **Open finding, pre-existing:** the live match is not seed-reproducible. Every
> calibration number measured before this is suspect for the same reason. Worth
> a dedicated investigation — it is not a phase-5 defect.

### Real 2026-27 rules and fixture dates (`world/realworld.py`)

An earlier version of the test world invented its own kickoff times, rest
periods and matchday spacing. That proves the plumbing works and nothing else —
it says nothing about whether the rules are right, and matching real football is
the entire purpose of the simulator. `world/realworld.py` holds the **published**
2026-27 calendar as data, with sources:

- **Premier League** — 20 clubs, 38 matchdays, 3 relegated; 21 Aug 2026 – 30 May
  2027; fixtures released 19 June; 33 weekend + 5 midweek rounds; **four**
  international breaks including a three-week World Cup window.
- **UEFA Champions League** — 36 clubs, 8 league-phase matchdays on the real
  dates (8–10 Sep, 13/14 Oct, 20/21 Oct, 3/4 Nov, 24/25 Nov, 8/9 Dec, 19/20 Jan,
  27 Jan 2027), top 8 direct to the R16; play-offs from 16/17 Feb through the
  final on **5 June 2027 at the Metropolitano**; two-legged knockouts with extra
  time and penalties; **no away-goals rule** (abolished 2021).
- **EFL Cup** — real round dates, final Sunday 21 March 2027.
- **FA Cup** — all 11 rounds on their published dates, fifth round 6 March 2027.

**Rounds are placed on real dates, not derived ones.** `SchedulePattern` gained
`fixed_round_dates` and `round_date_windows`, because a real competition
announces a *set of days* ("8-10 September", or a Tue/Wed/Thu batch) and no tie
is ever played on any other. A day-count slip invents dates nobody announced —
which the suite caught: an early version drifted the EFL round onto a Saturday
the EFL never listed.

`SchedulePattern.round_date_windows` is the fix, and a moved round is now
reported (`Calendar.slipped()`) rather than treated as either a violation or a
silent success.

### The controlled one-month integration test (audit §16 phase 10, plan §18)

Plan §18 is the test the plan calls "extremely important":

> Saturday domestic league → Wednesday Champions League → Saturday domestic
> league → Wednesday domestic cup. […] **The same player must have one
> continuous state throughout.**

`world/runmonth.py` runs it for real — the calendar places the fixtures, the
same `MatchEngine` simulates them, `world/ingest.py` translates the results,
and `world/ledger.py` keeps the state. **24 real matches, five competitions,
two federations, four weeks.**

```powershell
& ".\.venv\Scripts\python.exe" -m world.runmonth     # ~10-20 min
```

`world/testworld.py` builds the test world per plan §17, on the **real** rules
and dates above: the 18 real squads from the 26/27 roster in a Premier League
shape, entering a real Champions League league phase and a real EFL Cup round.
The one compromise is scale — 18 clubs against a real Premier League's 20 and
the Champions League's 36 — and it is stated rather than padded with invented
clubs. The squads are real because a test world with invented players could not
run a real match, and running a real match is the point.

The month is anchored on the **real** Champions League opening matchday, so what
gets played is what UEFA and the Premier League actually scheduled. The test
this produces is genuinely worth asking: **can the scheduler honour the real
football calendar?** Real 2026-27 is congested — the EFL round and the European
matchday overlap in September — and the world layer resolves it with the cup
yielding to the higher-priority European night, exactly as it should.

Two things the driver had to get right, both found by running it:

- The month window is selected **per competition** and the leagues truncated to a
  few real rounds. A full 18-club double round robin is 34 matchdays, which
  saturates every midweek slot and leaves the cup tie nowhere legal to go.
- The continuity checker is verified by **deliberately breaking the ledger six
  different ways** and asserting the checker notices each one. A checker that has
  never failed has not been tested.

The checker verifies after *every matchday*: minutes/goals/appearances never go
backwards, the continuous total equals the sum of the per-competition lines, no
line claims more than the player actually played, appearances equal matches with
minutes, and no club plays twice in a day.

**Bug it found in the very first real match:** eight unused substitutes were being
counted as appearances. The exporter emits a stat line for every *named* player,
including subs who never came on, and the ledger counted all of them — which
would have inflated every appearance total in the warehouse.

### Current state and the next gate

Phases 1–8 are complete, including phase 5. The whole world layer is now in
place: identity, schemas, competitions, calendar, ledger, ingest, qualification
and knockout resolution.

`python -m world.proof` builds the plan §17 test world — 2 countries, 16 clubs,
a Saturday league + Tuesday cup + Wednesday continental — and prints the
collision report, the plan §12 week for one club, both tables, the cup bracket,
and the plan §18 continuity check. Current result: **180 fixtures, 0 collisions,
0 violations, one continuous state holds.**

`tests/test_world_ingest.py` closes the loop with a **real simulated match**,
asserting the ledger's totals equal the exporter's exactly.

Open items, neither blocking:
- the live match is not seed-reproducible (above) — this affects every
  calibration measurement, not just the world layer;
- `world_data/` JSON templates for the test world (audit §17) are still inline in
  `world/proof.py` rather than as data files. Plan §21 puts data population
  after the schema, and the schema is now complete, so this is unblocked.




