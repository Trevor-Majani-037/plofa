# PLOFA — project memory

## Machine is SHARED — hard boundary (2026-11-16, user instruction)
- Another agent runs on this machine in a different app. It owns its own
  processes and files.
- **Never kill, pause, or interfere with a process I did not start myself**,
  and never touch files outside the work I was explicitly asked to do —
  even if a process looks stale, orphaned, or redundant. Ask first.
- On 2026-11-16 I killed two `tests/gate_token_source.py --matches 7` PIDs
  (4500, 5944) believing they were my own leftovers. They were the other
  agent's. Do not repeat.
- Corollary: before any long run, check what is already consuming CPU and
  *report* it rather than acting on it.

## CPU is scarce — the user pays for it literally (overheating laptop)
- No multi-hour collection without explicit go-ahead. Chunk it instead.
- Every long run must be *resumable* and must write its corpus after every
  match, never only at the end.

## Interpreter
- `.venv/Scripts/python.exe` (3.14.6, numpy 2.5.1). Bare `python` has no numpy.

## Pitch / geometry facts that have caused bugs
- Pitch 105 x 68. `attacks_right` **is** team identity (`set_squad`:
  `team_name == home_team`); `MatchState` never swaps ends.
- Sensor slot `[0]` is NORMALISED ball x (`x/105`). Multiply by 105 to get
  metres — never twice.
- Slot `[18]` is `score_diff/5`, in the PLAYER's frame (sign flipped for the
  away side as of 2026-11-16). The consequence critic reads slot 18.

## Measurement discipline
- A uniform multiplier can never change a preference (`k·a > k·b` iff
  `a > b`). Any "situational weighting" must be a gain/loss SPLIT.
- Held-out, mirror-symmetry, and within-cell contrast before any claim.
- If a diagnostic's headline looks alarming, check the baseline before
  shipping it as a finding (the near-tie 60% was a normalisation artefact).
