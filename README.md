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
