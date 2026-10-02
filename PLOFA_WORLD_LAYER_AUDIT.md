# PLOFA WORLD-LAYER ARCHITECTURE AUDIT
### Phase 1 deliverable for the World Football plan (see `Documents/# PLOFA FOOTBALL WORLD.txt`)

Read-only audit of the **world layer** (clubs, players, fixtures, competitions, standings,
statistics persistence) of the current PLOFA 26/27 repository. The intelligence engine
(brains, sensors, evolution, rewards) is already audited in `PLOFA_ARCHITECTURE_AUDIT.md`;
this document audits the *world*, not the brain.

**Scope reviewed:** `roster_loader.py`, `season_manager.py`, `auto_run_match.py`,
`match_engine.py` (MatchConfig only), `exporter.py`, `season_stats.py`, `alltime_db.py`,
`manager_profile.py`, `referee_pool.py`, `squad_manager.py`, `player_dna.py`,
`training_system.py`, `build_app_data.py`, `manager_perception.py`, and the live
`season_state.json`.

Companion working state: this file added at repo root. No production file modified.

---

## 1. CURRENT WORLD ARCHITECTURE

```
PLOFA-2026-2027.xlsx  (single "PLAYERS" sheet + "FIXTURES" sheet — source of truth)
   │
   ▼
roster_loader.get_loader()            PlayerRecord · POS_MAP · FORMATION_SLOTS · SUB_TIMING
   │                                   (roster_loader.py:40–90; records :97–178; cols :345–432)
   ▼
auto_run_match.py  (PRODUCTION: writes passenger ledger season_state.json; Trevor-only, --play)
   ├─ TEAM_CATALOG  20 clubs inline       (auto_run_match.py:229–373)  ← identity = club NAME
   ├─ SOUL_PLAYERS 13 entries inline      (auto_run_match.py:150–218)
   ├─ USER CONFIG: date/matchday/season = "26/27"/COMPETITION = "PLOFA" (auto_run_match.py:84–140)
   ├─ availability (SeasonState.is_available) → squad pick → SquadBuilder.build(team, starters)
   ├─ MatchConfig (match_engine.py:869–887; competition="PLOFA" at :877)  ← the one competition seam
   ▼
MatchEngine.simulate()  →  MatchResult  (timeline, xG, cards, subs, stamina, injuries)
   ▼
PLOFAExporter.export_all()              exporter.py:2222
   ├─ output folder  plofa_output/<Home>_vs_<Away>_MD<##>/   (auto_run_match.py:1243–1246)
   ├─ {name}.xlsx (8 sheets) · {name}_players.csv · {name}.json (+ 8 PNGs)
   └─ match.json already carries  "competition": config.competition   (exporter.py:3379)
   ▼
season_state.json   {season, players, chemistry, standings, cognition, philosophies, fixture_ledger}
   └─ standings flat dict keyed by club NAME; one league, no competition/enum dimension
season_stats.json   aggregates from every plofa_output/<dir>/match.json (no competition keying)
manager_state.json / referee_state.json  (per-club-name manager pools, referee rotation)
   ▼
alltime_db.py (SQLite warehouse)        alltime_db.py:280–414
   ├─ matches.competition TEXT              (:308)  ← dead metadata today
   ├─ UNIQUE(season, matchday, home_team_id, away_team_id)   (:327)  ← one competition per season only
   ├─ player_match_stats PK (season, match_date, team_id, player_id)  (:361)
   ├─ goals UNIQUE (season, matchday, team_id, minute, scorer)        (:403)
   └─ NAME-based teams/players with alias/identity mapping (README §9) — retroactive renames
```

**World shape today:** one fictional country (toland; default nationality `"Tolandian"`,
`player_dna.py:1504`), **18 clubs**, one top division, 34 matchdays, one competition.
There is **no** cup, knockout, second division, promotion, relegation, continental
competition, playoff, or qualification code anywhere. Grep for
`cup|knockout|relegat|promot|division|qualif|champions` returns only comments, the
trainer's unrelated "brain promotion", and one cosmetic token check in
`manager_perception.py:424–425`.

---

## 2. DEFINITIONS USED BY THIS AUDIT

| Term | Meaning |
|---|---|
| Authentication identity | today's runtime key — a **string name** (club/player/team). |
| Canonical identity | a stable `*_id` (e.g. `club_id="TOL-CLU-0007"`, `player_id="TOL-PLY-0117"`). |
| World Layer | countries, cities, clubs, stadiums, leagues, cups, continental comps, calendars, registrations, history. |
| Match Engine | everything from `MatchConfig` down: brains, possession, chains, physics, xG, timeline. |
| Ledger | the machine-managed state files (`season_state.json`, `season_stats.json`, `alltime.db`). |
| Source data | human-edited inputs (Excel/CSV/JSON country and competition definitions). |

---

## 3. WHAT IS ALREADY REUSABLE (do not rebuild)

1. **`MatchConfig` is a ready-made `MatchContext`.** `match_engine.py:869–887` already has
   `home_team`, `away_team`, `match_date`, `matchday`, `season`, `competition`, `venue`,
   `stadium_capacity`, `referee`, `referee_strictness`, `is_derby`, `home_advantage`,
   `weather`, `start_time`, `weather_enabled`. The whole engine below it is competition-
   agnostic — it never inspects league/cup semantics. Generalisation = populate more
   fields, not re-engineer.
2. **The match engine itself** (`event_chain.py`, `possession_physics.py`,
   `geometry_engine.py`, brains, `squad_manager`, `tactical_*`, referee/manager models,
   exporter) is competition-agnostic end-to-end. It solves "WHAT happens in this match".
3. **`SeasonState` player store is already the cross-match player ledger.** One
   `players` dict holds form/fatigue/injuries/confidence per player and carries across
   matchdays. This is the substrate for plan §7 (global player state) — it just needs
   a competition dimension and stable IDs, not a rewrite.
4. **Referee rotation + manager pools** are per-club models with real behaviour; they are
   world data (per-competition instances), not engine logic.
5. **`battlefield/`** is a reusable in-memory N-match harness with a `PYTHONHASHSEED=0`
   determinism guard and KS/±15% comparison discipline — the correct pattern for the plan's
   §17 test world.
6. **`alltime_db` alias/identity layer** (README §9) already solves retroactive name→same-
   person merging. It is the embryo of §23 canonical identity, currently applied only to
   legacy/name strings.
7. **`training_system.py`** already commits cross-matchday development into `SeasonState` —
   player development follows the player, not the fixture (plan §7 foundation).

---

## 4. WHAT IS HARD-CODED / PLOFA-SPECIFIC (must become data or config)

| # | Hard-coded | Location | Plan relationship |
|---|---|---|---|
| 1 | `COMPETITION = "PLOFA"`, `SEASON="26/27"` | `auto_run_match.py:91–92` | §4 competition-agnostic |
| 2 | `competition="PLOFA"` default | `match_engine.py:877` | §4 (metadata today, unused) |
| 3 | `competition="PLOFA"` literal | `season_manager.py:1099` | §4 |
| 4 | Output folder `<Home>_vs_<Away>_MD<##>/` | `season_manager.py:1175`, `auto_run_match.py:1243` | §4 · collision once >1 competition |
| 5 | `TEAM_CATALOG` — 20 clubs inline | `auto_run_match.py:229–373` | §6 club schema |
| 6 | `SOUL_PLAYERS` — 13 entries inline | `auto_run_match.py:150–218` | §6 player schema |
| 7 | Big-6 fallback set inline | `auto_run_match.py:~1239` | §6 club schema |
| 8 | Single flat standings keyed by club name | `season_manager.py:170–232`, `season_state.json["standings"]` | §6 competition schema · §14 |
| 9 | Circle-method round-robin only | `season_manager.py:62–163` | §12 calendar (§12 unsupported) |
| 10 | One-league-per-season assumption in DB PKs | `alltime_db.py:327,361,403` | §8 global stats · §23 IDs |
| 11 | Name-as-identity in every layer | ledger, folders, season_stats, DB | §23 stable IDs |
| 12 | Default nationality `"Tolandian"` | `player_dna.py:1504` | §5 country data |
| 13 | `MAX_SUBS = 3` "PLOFA standard" | `squad_manager.py:402–412` | → competition rule |
| 14 | 5-yellows-in-6 league rule | `squad_manager.py:914` | → competition rule |
| 15 | Venue = `<Club> Stadium` fallback | `auto_run_match.py:98,777–780` | §6 stadium schema |
| 16 | Attendance from `standings` league position | `exporter.py:285–322` | → competition rule |

Everything in this table is **world/competition data**, not engine logic. The engine does
not need to change for any of it — the loader/ledger configuration does.

---

## 5. WHAT NEEDS ABSTRACTION (the seams)

1. **Identity keying.** Replace name-as-key with canonical IDs minted through an adapter
   (see §9), preserving name-based I/O for the live season.
2. **Fixture generation.** Generalise the circle-method `FixtureList` (season_manager.py:62)
   into a calendar that composes fixtures from competitions with dates, kickoffs, priority,
   and a no-overlap constraint (§12).
3. **Standings → competition standings.** `LeagueTable` is a flat 3-1-0 table. A competition
   needs its own rules: points system, tie-breakers, ET/penalties flag, aggregate rules,
   promotion/relegation positions, qualification slots (§9, §14, §15).
4. **Competition progression.** Cup draws, knockout rounds, group→knockout — a generic
   `Competition` model that owns its own stage/round state (§9).
5. **Statistics accumulation.** `season_stats.py` currently sums every match.json into one
   bucket. Needs `(country_id, competition_id, season, stage)` scoping so a player's league
   and continental rows coexist (§8).
6. **Output naming.** Folder/file names and `alltime_db` PKs must include competition so the
   same two clubs can meet in league + cup without collision (§8).

---

## 6. WHAT SHOULD BECOME THE SHARED CORE

Keep as shared core (already is): `MatchConfig` + `MatchEngine` + chains + physics + brains +
exporter. **Add** to shared core:

- `world/model.py` — canonical schemas (Country, City, Stadium, Club, Player, Manager,
  Competition).
- `world/ids.py` — canonical ID minting + the name↔ID adapter/aliasing layer.
- `world/competition.py` — generic competition framework (rules, stages, progression).
- `world/calendar.py` — multi-competition fixture scheduler with collision guarantee.
- `world/ledger.py` — the world-facing state store (superset of SeasonState) that carries
  competition scope + stable IDs while remaining backward-readable as v1.

## 7. WHAT SHOULD BE COMPETITION CONFIGURATION (data, not code)

Per competition: type (league/cup/continental), participating club_ids, rules (points,
tie-breakers, ET, penalties, aggregate, away goals, sub limits, yellow-card rules),
schedule pattern, stages, qualification in/out, and its `country_id`, `season_id`.

## 8. WHAT SHOULD BE WORLD DATA

Per country: cities, stadiums (with per-season capacity history — plan §6 "historical
capacity must remain historically correct"), clubs (identity, colours, style, rivalries,
honours), players, managers, referees. Per competition-history: winners, records, previous
entrants.

## 9. WHAT MUST REMAIN UNTOUCHED DURING 26/27

- `auto_run_match.py` production path and its writes to `season_state.json` (Plan §19).
  All world work sits behind an **adapter**: the live season continues keyed on names;
  canonical IDs are derived (never written into the authoritative ledger during 26/27).
- The live `season_state.json` ledger, `season_stats.json`, `manager_state.json`,
  `referee_state.json`.
- The intelligence stack (brains, evolution, sensors) — already protected by the V2 audit.

---

## 10. PROPOSED WORLD FOOTBALL ARCHITECTURE (design)

```
WORLD DATA (source)                     LEDGER (runtime)
  countries/*.json                        world_state.json  (world-facing)
  tol and/*.json                          ├─ players (global, canonical IDs)
  clubs/*.json  stadiums/*.json           ├─ clubs    │
  cities/*.json  referees/*.json          ├─ competitions (own standings/stages)
                                          ├─ fixtures/calendar   ─┐
  COMPETITIONS/*.json                     └─ history ─────────────┤
  league.json  cup.json  champions.json                          │
        │                                                         │
        ▼                                                         ▼
  world/competition.py ──► resolve(match_context) ──► MatchConfig.competition_id
                                       │                       │
                                       ▼                       ▼
                                  fixture generator   MatchEngine (UNCHANGED)
                                       │                       │
                                       ▼                       ▼
  world/calendar.py              MatchResult ──► exporter ──► folders scoped by competition_id
                                       │
                                       ▼
                  season_stats / alltime.db — rows keyed (season, competition_id, …)
```

The one true invariant is unchanged: **no competition runs its own engine.** Every match
rounds-trips through the same `MatchEngine`. The world layer only selects *which* match,
*when*, and *under which rules*.

### Component responsibilities

| Module | Purpose |
|---|---|
| `world/model.py` | Typed dataclasses + JSON (de)serialization for every entity in §5 of the plan. |
| `world/ids.py` | Canonical ID generation; **adapter** mapping name↔ID in both directions. |
| `world/competition.py` | Generic `Competition` + `CompetitionStage` + progression (league table, cup knockout, group→knockout). Rules are data. |
| `world/calendar.py` | Assembles a season calendar from competitions; enforces no same-team collisions; applies priority/rest-days. |
| `world/ledger.py` | The world state store; per-competition standings, global player state, history. Reads v1 SeasonState for 26/27 (read-only adapter). |
| `world/qualification.py` | Last season's standings/cup winner → next season's entrants (plan §15). |
| `world/production.py` | THE ONLY thing allowed to load world data into `MatchConfig`. |

---

## 11. PROPOSED DATA SCHEMAS (v0 — enough to build the test world, plan §6/§21)

```jsonc
// country.json
{
  "country_id": "TOL",            // global unique (plan §23)
  "name": "Toland",
  "continent": "Europa",
  "association": "Toland Football Federation",
  "league_structure": ["division_1"],          // list of division ids + promotion rules
  "cup_structure": ["toland_cup"],
  "coefficient": 12.5,
  "city_ids": ["TOL-CITY-01", ...],
  "stadium_ids": [...], "club_ids": [...],
  "referee_ids": [...], "manager_ids": [...], "player_ids": [...]
}

// stadium.json  — capacity is PER-SEASON (plan §6 historical rule)
{
  "stadium_id": "TOL-STA-03", "name": "Hartwell Park",
  "city_id": "TOL-CITY-01", "surface": "grass",
  "pitch": {"length_m": 105, "width_m": 68},
  "capacity_by_season": {"25/26": 52000, "26/27": 52000},   // non-decreasing, editable
  "atmosphere": {"base": 0.72, "home_advantage_bonus": 0.08}
}

// club.json
{
  "club_id": "TOL-CLU-0003", "name": "Hartwell City", "country_id": "TOL",
  "city_id": "TOL-CITY-01", "stadium_id": "TOL-STA-03",
  "division_id": "TOL-D1", "colors": {"home": "#003087", "away": "#C8102E"},
  "style": {"attack": 0.72, "press": 0.81, "intensity": 0.65},
  "rivalry_ids": ["TOL-CLU-0011"], "honours": [],
  "historical_records": {}
}

// player.json  (superset of current Excel/PPlayerRecord; identity-aware)
{
  "player_id": "TOL-PLY-0117", "canonical_name": "Percy Osei",
  "aliases": ["Percy", "P. Osei"],                // name↔ID adapter seeds
  "club_id": "TOL-CLU-0003", "country_id": "TOL",
  "position": "ST", "dob": "1997-04-12", "nationality": "Tolandian",
  "dna": {...}, "personality": {...}, "soul_profile": null,
  "career": {"appearances": 0, "goals": 0, "clubs": []}   // plan §24
}

// competition.json  (league / cup / continental share this shape)
{
  "competition_id": "TOL-D1",
  "type": "league",                                // league | cup | continental(group/ko)
  "country_id": "TOL",
  "points": {"win": 3, "draw": 1},
  "tie_breakers": ["pts", "gd", "gf", "head_to_head"],
  "extra_time": false, "penalties": false, "aggregate": false,
  "substitutions": {"max": 3},
  "suspension_rule": {"type": "yellow_accum", "threshold": 5, "window": 6},
  "promotion": {"from": "TOL-D1", "to": null},
  "relegation_slots": 0,
  "qualification_out": [{"competition_id": "EUR-CL", "slots": [[1, 4]]}]
}
```

**Identity adapter rule (critical):** canonical IDs are **derived**, never invented.
`world/ids.py` mints `club_id`/`player_id` from the Excel roster and the live ledger via a
deterministic mapping, and aliases every name already present (`alltime_db.alias` as the
precedent). Nothing in the 26/27 authoritative ledger is rewritten with an ID in place of a
name. New world data (future countries) enters with IDs from day one.

---

## 12. PROPOSED COMPETITION ABSTRACTION

One generic `Competition` object owns: `competition_id`, `type`, `participants` (club_ids),
`stage` (LEAGUE_PHASE | GROUP | KNOCKOUT), current round, `rules`, and a provenance hook
`context_for(round) -> MatchContextFragment`. The fragment is merged into `MatchConfig`,
which is the **only** hand-off to the engine:

```python
ctx = competition.context_for(stage="r16", leg="first")   # {extra_time, penalties, aggregate, importance}
config = MatchConfig(..., competition_id=ctx.competition_id, extra_time=ctx.extra_time,
                     penalties=ctx.penalties, aggregate_score=ctx.aggregate, ...)
result = MatchEngine(config, home_profile, away_profile).simulate()
competition.apply_result(result)                          # table / progression / qualification
```

`extra_time`, `penalties`, `aggregate`, `substitution rules`, `suspension rules`,
`importance` are **MatchConfig expansions** (additive fields, defaulting to current
behaviour) — the engine consumes them only when non-default, so no 26/27 regression (plan
§10–11, §19).

---

## 13. PROPOSED FIXTURE / CALENDAR ARCHITECTURE

Replace the single circle-method generator with a `Calendar` that:

1. Collects all competitions for a country/season and their scheduling patterns
   (weekend league, midweek cup/continental, rest windows).
2. Assigns dates/kickoffs with **one hard invariant:** a club never has two fixtures on
   the same calendar day (plan §12), and enforces minimum rest (default 3 days) configurable
   per competition.
3. Emits `fixture_id`s; each fixture knows `competition_id`, `round`, `stage`, `leg`, `venue`.
4. `availability` is resolved against *global* player state at the time the fixture is
   played (injuries/suspensions/register eligibility per competition — plan §13).

The scheduler is deterministic (seeded) so a season calendar is reproducible.

---

## 14. PROPOSED GLOBAL STATE ARCHITECTURE

`world_state.json` (world-facing) structured as:

```
world_state
├── seasons { "26/27": { competitions: { "TOL-D1": {...}, "EUR-CL": {...} } } }
├── players { player_id: { state, per_season: { "26/27": { per_competition: {...}} } } }
├── clubs   { club_id: { honours, per_season: {...} } }
├── managers/ referees / calendars / history
```

Dual-write rule during 26/27: the **live** `season_state.json` remains authoritative for the
PLOFA league and is untouched. The world ledger reads it (via adapter) when a request spans
competitions; the *second* competition (test cup/champions league) is run in its own
competition-scoped files that only the world layer writes. No cross-contamination.

---

## 15. PROPOSED GLOBAL STATISTICS ARCHITECTURE

- Extend the `alltime_db` UNIQUE keys to include `competition_id` (migration:
  backfill `competition_id="TOL-D1-PLOFA"` on existing rows, then relax the PKs — a pure
  additive migration, idempotent, verified by `tests/test_alltime_db.py`).
- `season_stats.py` gains an optional competition filter; the default bucket reproduces
  today's aggregates exactly.
- Career totals (plan §8, §24) are queries over rows keyed by `player_id` +
  `competition_id` — every cross-competition leaderboard answer comes from actual simulated
  data, never fabricated.

---

## 16. PROPOSED IMPLEMENTATION PHASES (world-aware revision of plan §31)

| # | Phase | Deliverable | Touches 26/27? |
|---|---|---|---|
| 1 | **World schemas + IDs (v0)** | `world/model.py`, `world/ids.py` adapter; JSON templates for the plan §6 entities | No |
| 2 | **MatchConfig expansion** | Additive fields (`competition_id`, `extra_time`, `penalties`, `aggregate`, …) defaulting to current behaviour | No (behind defaults) |
| 3 | **Competition framework** | `world/competition.py`: league model first (reuses LeagueTable), then cup knockout | No |
| 4 | **Calendar** | `world/calendar.py` multi-competition scheduler + collision test | No |
| 5 | **Extra time + penalties (engine)** | Real continuation of match state (plan §10) + shootout phase (§11) behind flags | Behind flags |
| 6 | **Global state ledger** | `world/ledger.py`; adapter over 26/27 league state | Read-only for 26/27 |
| 7 | **Global statistics** | alltime_db competition key migration + season_stats filter | Additive migration |
| 8 | **Qualification + promotion/relegation** | `world/qualification.py`; rules as data | No |
| 9 | **Continental prototype** | Champions League (group → knockout) on the test world | No |
| 10 | **Test world (plan §17/§18)** | 2 countries, 1–2 divisions, 8–12 clubs, league + cup + CL; controlled one-month integration test | No |

**Every phase ships with the determinism discipline already in README §8**
(`PYTHONHASHSEED=0`, seeded runs) and regression asserts that 26/27 match output is
byte-identical when new features are disabled.

---

## 17. MODULE/CONFIG — FILE TOUCH MAP

**New modules (world layer):**
`world/__init__.py` · `world/model.py` · `world/ids.py` · `world/competition.py` ·
`world/calendar.py` · `world/ledger.py` · `world/qualification.py` ·
`world/production.py` · `tests/test_world_ids.py` · `tests/test_world_calendar.py` ·
`tests/test_world_competition.py` · `tests/test_world_ledger.py`

**New config/data (test world):**
`world_data/TOLAND/` (country, cities, clubs, stadiums, players, managers, referees) ·
`world_data/COMPETITIONS/` (league, cup, champions templates)

**Modified (additive only, defaults preserve 26/27):**
`match_engine.py` (MatchConfig additive fields) · `exporter.py` (competition scoping of
folder names, additive) · `season_stats.py` (optional competition bucket) ·
`alltime_db.py` (PK migration, backfill) · `season_manager.py` (LeagueRunner reads
competition config) · `roster_loader.py` (optionally emit canonical IDs via adapter, don't
require them) · `squad_manager.py` (competition-rule hooks for MAX_SUBS/suspensions)

**NOT modified:**
`event_chain.py` · `possession_physics.py` · `geometry_engine.py` · `football_brain.py` ·
`brain_integration.py` · `brain_evolution.py` · all `brains/` · the live
`season_state.json`/`season_stats.json`/`manager_state.json`/`referee_state.json` during
26/27. `auto_run_match.py` is only extended at the adapter boundary (config → MatchConfig),
never disturbed in production path.

---

## 18. RISKS

1. **Identity migration (highest).** Name→ID is the keystone. If the adapter is not
   reversible and deterministic, the live season's reproducibility breaks. Mitigation:
   derived-only IDs, alias precedence from `alltime_db`, out-of-repo test world.
2. **MatchConfig creep.** Adding fields for every competition is fine; letting the engine
   *branch* on them is not. Rule: the engine only consumes a field when set; all new
   fields default to current behaviour and are covered by an A/B regression.
3. **Fixture collision bugs.** The calendar's invariant must be enforced by test, not
   convention (plan §29).
4. **Statistics double-count.** The first real risk after IDs; the alltime_db UNIQUE-key
   migration and a per-competition ingestion test are mandatory before any second
   competition goes live.
5. **Scope creep.** The plan's §30 warning applies: no ORM, no web frame, no async. The
   world layer is five small modules + data.
6. **Determinism drift.** New world code must set the same `PYTHONHASHSEED=0` discipline
   used by `battlefield/`.

---

## 19. WHAT DATA/TEMPLATES ARE NEEDED FROM THE PROJECT OWNER (plan §21, §34)

Before the world layer is populated beyond the test world, provide **templates** (not
hundreds of entities):

1. **Country definition** — country list, continents, association names.
2. **League structures** — how many divisions per country; promotion/relegation counts;
   playoff rules; points/tie-breakers.
3. **Club lists** — for the real world: name, city, stadium, colours, style, rivalries,
   honours history.
4. **Stadium data** — capacities per season (historical correctness), pitch dims, surface.
5. **Player databases** — the PLOFA roster already exists; the new concern is *canonical
   names + aliases* and whether legacy `alltime.db` merges are to continue.
6. **Manager + referee databases** — who belongs to which country/competition.
7. **Competition rules** — domestic cup format (rounds, replays, ET, penalties),
   Champions League format (league phase size, knockout legs, away goals or not),
   qualification slots per country.
8. **Historical data** — old league tables/winners to seed §24 history.
9. **Continent / coefficient model** — if coefficients are wanted at all in v1.

**Default (if not provided):** the world layer ships with the 2-country/8–12-club test
world from plan §17 only, and PLOFA/Toland stays exactly as it is today.

---

## 20. PROOF-OF-CONCEPT PLAN (smallest something that proves the architecture)

**Goal:** prove plan §18's integration test on a toy without touching 26/27.

1. `world/model.py` + `world/ids.py` (schemas + adapter) with unit tests for
   determinism and reversibility.
2. `MatchConfig` additive expansion + a test that a fixture with `competition_id != PLOFA`
   still produces byte-identical output when all new fields are "off".
3. `world/competition.py` league + single-round knockout; unit tests on progression.
4. `world/calendar.py` on 8 clubs with league + cup; a **collision test** (no club two
   matches same day).
5. `world/ledger.py` writing a *separate* `world_test_state.json`; a player plays league
   (Sat), cup (Wed) and the ledger shows one continuous state (plan §18's checklist, first
   10 rows only).
6. alltime_db competition-key migration + idempotency test.
7. A 3-match dry-run script (`scripts/` or `world/proof.py`) printing the resulting
   standings, cup bracket, and per-player cross-competition stat line.

*Acceptance:* the toy month's player state is continuous across three competitions, the
table and bracket advance correctly, no fixture collision, and the live 26/27 ledger is
byte-identical before/after every proof run.

---

*Audit produced read-only against the working tree on 2026-09-23. Line numbers reference
the current working copy; re-verify against a frozen baseline before merging. The
intelligence-layer sibling audit is `PLOFA_ARCHITECTURE_AUDIT.md`.*