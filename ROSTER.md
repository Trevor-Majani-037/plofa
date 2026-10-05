# The roster workbook is not in this repo

`PLOFA-2026-2027.xlsx` is **excluded** from version control. It is the real
2026/27 squad workbook: 18 clubs and roughly 440 players.

## Why

It is your data, not source code. It has also been publicly visible on GitHub
since 2026-09-20, so it was decided to stop tracking it going forward rather
than continue to publish it on every commit.

## What breaks without it

This is a hard runtime dependency, not an optional extra. A fresh clone will
fail until you supply the workbook:

| depends on it | where |
|---|---|
| `roster_loader.EXCEL_FILE` | points at this exact filename; no override in the engine path |
| `tests/test_weather_physics.py` | `load_fixture_info("PLOFA-2026-2027.xlsx", ...)` |
| `tests/test_world_ids.py` | ground truth for the 26/27 identity index |
| `tests/test_world_model.py` | stadium capacity / season provenance |
| every probe in `tools/diag/` | `build_pair()` → `get_loader(XLSX)` |

`roster_loader.py` will tell you so directly if it is missing.

## Supplying it

Drop the file in the repository root, next to `match_engine.py`:

```
PLOFA/
  PLOFA-2026-2027.xlsx   <- here
  match_engine.py
  ...
```

Nothing else needs configuring — the path is hard-coded, so the filename and
location must be exactly as above.

## Note on the two "test" clubs

`auto_run_match.TEAM_CATALOG` lists 20 clubs, the workbook carries 18.
**Hartwell City** and **Thornfield United** are test teams entered by hand in
the scratch runner (`run_match.py`); the 18 real clubs are the whole world.
The production path hard-exits on a team absent from the workbook, which is
correct behaviour, not a bug.