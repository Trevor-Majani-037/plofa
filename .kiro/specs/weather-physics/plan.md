## Weather + Fixture-Time Physics — New-Session Plan

### Current state (as of 2026-09-04)
- `MatchConfig.weather` is a cosmetic string field (`clear|rail|wind|fog`) set once per match in `auto_run_match.py` — **never read anywhere** in the sim (zero effect on pass/possession/stamina/shots/attendance).
- Match kickoff times exist in the Excel `FIXTURES` sheet (`Start Time` column, e.g. `12:30`) but are never read by code. `MatchConfig` has no `start_time` field.
- `exporter.py` attendance fill rate uses a random factor; no real-time or real-climate dependency.

### Build plan (new session)
1. **Refactor weather out of dead string** → `WeatherCondition` dataclass with fields: `rain_intensity 0-1`, `wind_speed m/s`, `wind_direction deg`, `visibility`, `pitch_wetness 0-1`, `temperature_c`.
2. **`weather_physics.py`** — module that maps `WeatherCondition` → numerical multipliers for:
   - Pass accuracy / pass spread (lateral error)
   - Shot power / shot accuracy
   - Stamina drain rate
   - Dribble control / carry distance
   - Pressing willingness (managers call off press in heavy rain)
   - Pitch grip (slipping: tackles + carries)
   - GK positioning accuracy (fog / rain reduces diving reach)
3. **Fixture-time layer** — pull `Start Time` from the Excel FIXTURES sheet into `auto_run_match.py` → feed `MatchConfig.start_time`. Then build a `FixtureTimeEffect`:
   - Time-of-day attendance multiplier (floodlight night games = slightly lower gate)
   - Temperature proxy from month/climate (affects stamina drain)
4. **Real-climate assignment** — per-venue, month → baseline `WeatherCondition` from real-world climate data (UK clubs: rain in Oct-Feb, cold/dry in summer, etc.). Add randomness on top.
5. **Toggle**: all weather effects default `OFF` behind a `WeatherPhysics.enabled` flag so the official runner (`auto_run_match.py`) is unaffected unless explicitly opted in.
6. **Tests**: per-field physics tests (e.g. "rain reduces pass lateral accuracy by X-Y%") + integration test confirming the full chain reads weather and adjusts pass probability.
7. **Verification throwaway**: run a single match with clear vs heavy rain and confirm stats diverge.

### Files to touch
- `match_engine.py` — add `start_time` to `MatchConfig`; optionally add `WeatherCondition`
- `weather_physics.py` — **new file**
- `event_chain.py` — read weather multipliers for pass accuracy / shot quality
- `pass_classifier.py` — lateral error from wind
- `squad_manager.py` — stamina drain multiplier from weather
- `exporter.py` — optional floodlight/attendance display from start time
- `auto_run_match.py` — read `Start Time` from Excel; feed into config; add `WEATHER_ENABLED` toggle
- `tests/test_weather_physics.py` — **new file**

### Files NOT touched (safe for MD1-MD3 runs this week)
Everything is safe. No weather or time fields are threaded into the sim during this session; the deferral is complete.
