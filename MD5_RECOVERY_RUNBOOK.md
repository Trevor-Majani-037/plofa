# Runbook: Red Wolves vs Avada Zenith MD5 recovery (2026-09-20/21)

## 1. What happened

- The original MD5 (Red Wolves 2–1 Avada Zenith, played 2026-09-19) was
  accidentally re-run on 2026-09-20 (~15:04). The re-run folder
  `plofa_output/Red_Wolves_vs_Avada_Zenith_MD05__ACCIDENTAL_RERUN_20260920_1504/`
  contains only xlsx/png/csv — **no raw JSON**.
- Resulting damage:
  1. Raw `plofa_output/Red_Wolves_vs_Avada_Zenith_MD05/*.json` gone.
  2. `alltime.db` missing the RW/AZ MD5 row (42 season rows instead of 43).
  3. `build_app_data.py` skips the fixture when collecting app data.
  4. `season_state.json` `fixture_ledger` empty, so any future MD5 re-run would
     push both clubs to 6 played (double-record).

## 2. What was still correct (do NOT "fix" these)

- `season_stats.json`: canonical 42 matches incl. the TRUE MD5 (RW 2–1 AZ);
  per-player `per_matchday['5']` holds full payloads for all 28 participants.
- Preserved app export
  `.../adventurous-mendel/data/matches/Red_Wolves_Avada_Zenith_MD5.json`
  (built 2026-09-19, before the accident): score, goals, team stats, all 36
  players with ratings.
- `season_state.json` post-original-MD5: RW W3 D0 L2 PTS 9 GF 11 GA 8;
  AZ W0 D1 L4 PTS 1 GF 3 GA 8; injuries Willi Kićh (knock 09-19),
  Souze Maod (muscular 09-19), Hans Misźich (ligament 09-05).

Decision taken by user: **restore the original, seed the ledger**
(no re-simulation).

## 3. Recovery procedure (in order)

### Step 0 — backups
```
Copy-Item alltime.db alltime.db.pre-md5-resync-<date> -Force
# season_state.json backups already exist:
#   .pre-md4-rebase-20260920_225829 / _225948
#   .pre-ledger-seed-20260920_231517 / _231600 / _231650
```

### Step 1 — seed the ledger
```
python scripts/seed_rw_az_md5_ledger.py
```
Replays DB MD1–4 in a scratch `SeasonState` to build the MD4-end snapshot
(RW W2 D0 L2 PTS 6, AZ W0 D1 L3 PTS 1), carries pre-MD5 injuries only
(MD5_DATE = 2026-09-19), seeds
`fixture_ledger["5|Red Wolves|Avada Zenith"]`, and self-verifies that a
re-run unwinds to MD4 then re-records MD5 (expect INVARIANT OK).
Gotcha (already fixed in script): baseline template must be
`scratch.get_player_state(player)`, NOT live state — otherwise
`season_matches` double-counts (Nathan Opaz showed 9 instead of 4).

### Step 2 — reconstruct the raw JSON
```
python scripts/reconstruct_rw_az_md5.py
```
Writes `plofa_output/Red_Wolves_vs_Avada_Zenith_MD05/Red_Wolves_vs_Avada_Zenith_MD5.json`.
Deterministic, no engine run. Details in section 5.

### Step 3 — DB sync
```
python alltime_db.py sync --dry-run   # expect scanned: 2 (history file + new file)
python alltime_db.py sync             # expect ingested: 1; the history-file error is benign/pre-existing
```
Verify: 43 season rows, 6 MD5 rows, RW 2–1 AZ present with 36 player rows,
3 goals. Re-running sync must ingest 0 (idempotent per season/matchday/home/away).

### Step 4 — app data regen
```
python build_app_data.py
```
Expect `Found 43 match files` incl. `[OK] Red Wolves vs Avada Zenith MD5 (2–1)`.
Writes to `.../adventurous-mendel/data`.

### Step 5 — final verify
- `season_state.json`: RW 3-0-2 PTS 9, AZ 0-1-4 PTS 1, ledger key present,
  Nathan Opaz `season_matches` 5.
- Regenerated `matches/Red_Wolves_Avada_Zenith_MD5.json`: score 2–1,
  possession 51.8/48.2, 3 goals, 36 players.

## 4. Backups inventory (repo root)

- `alltime.db.pre-md5-resync-20260921` (149 MB, pre-sync)
- `season_state.json.pre-md4-rebase-20260920_225948` (last known-good pre-rebase)
- `season_state.json.pre-ledger-seed-20260920_231650` (pre-seed)

## 5. Reconstruction design notes (why the script does what it does)

- **Participants (28):** `per_matchday['5']` copied 1:1, minus 6 accumulator-only
  keys (`fantasy_assists`, `goal_assists`, `open_play_shot_assists`,
  `second_assists`, `setpiece_shot_assists`, `shots_faced_by_creation`), plus
  `save_pct` (None outfield). Key order forced to the Ganester MD5 template
  (228 keys) — verified 0 mismatches across all 36 players.
- **Bench (8, 0 min):** not in `per_matchday`; synthesized from app-export
  identity+rating with zeroed stats. Names: Phillip Nuran, Pals Kackŷ,
  Paul Emeka, Seane Bellind (RW); William Diuda, Ernest Michael, Joni Anders,
  Tony Durwen (AZ). Only Seane Bellind / Ernest Michael / Joni Anders exist in
  `season_stats` info (age/nationality/DNA filled); the other 5 get nulls.
- **Assist override (engine wins):** the engine credited the MD5 corner assist
  to Julian Lingĺ (`setpiece_assists` 1); the accumulator stored it on
  Benardo Zico. The goals list agrees with the engine. So `assists`,
  `open_play_assists`, `setpiece_assists`, `open_play_cc`, `setpiece_cc` are
  taken from the app export for every player (only those 2 differed).
- **Possession:** export pct 51.8/48.2 kept; `possession_s` set to 518.0/482.0
  to reproduce it exactly (true clock seconds unrecoverable).
- **Left empty (unrecoverable, tolerated by both consumers):** `financials`
  → DB NULLs; `timeline` → DB corners 0 for this row. `build_app_data.py`
  reads only `match`/`goals`/`players` (verified by grep).
- **Known pre-existing drift (not ours):** current `build_app_data` computes
  `key_passes` = 0 for every match (Ganester regen 0/0 vs preserved 4/13).
  Regen output matches current code behavior everywhere.

## 6b. Rerun-folder quarantine (2026-09-21)

The accidental-rerun folder
`plofa_output/Red_Wolves_vs_Avada_Zenith_MD05__ACCIDENTAL_RERUN_20260920_1504/`
was MOVED (not deleted) to repo-root `.quarantine_accidental_rerun_20260920_1504/`
(all 11 files preserved). Reasons:
- Its `Red_Wolves_vs_Avada_Zenith_MD5.xlsx` basename-matches the reconstructed
  JSON, so `build_player_analytics.py` would merge WRONG-GAME coordinates
  (shots/passes x/y) into RW/AZ players (xlsx is fallback when timeline empty).
- `app/plofa_assets` is a junction to repo `plofa_output` — one move fixed both.
- Nothing references the rerun PNGs (regen `assets` = {}); they must never be
  presented as the original match.
Delete the quarantine only when certain no forensic need remains.

## 6. Never do this

- Never run `auto_run_match.py --play` for this fixture "to fix it" — that is
  what caused the incident. Re-runs are now safe (ledger replace), but the
  reconstructed JSON + DB row already represent the truth.
- Never re-simulate MD5 to "fill" financials/timeline — invented numbers
  would corrupt the warehouse. NULLs are honest.
