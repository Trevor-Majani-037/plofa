# PLOFA — Defensive block work: state of play

Written at the end of a session that ran out of context mid-task. Everything
here is needed to resume; nothing here needs re-deriving.

## TL;DR

`physics/` is finished and verified. The **defensive block** work is half-done:
the shape chain is correct, three of the user's four claims are already true in
PLOFA, and one contained fix remains. **The 26/27 regression against the
generalised block was still running when this session ended — that verdict is
unknown and gates everything.**

## 1. Where the code is

| file | what |
|---|---|
| `physics/movement.py` | Momentum, stamina, reaction gate. **Done, tested.** |
| `physics/adapter.py` | `offball_step`, `pass_flight_time`, the two gates. **Done.** |
| `position_engine.py` | `BLOCK_PRESETS` + `defensive_block_target()` — the block shape |
| `match_engine.py` | `_defensive_block` (team → `low`/`mid`/`high`), anchor substitution, held slot, recovery pace |
| `scripts/physics/probe_low_block.py` | Measures the block. **The measuring tool — use it, don't eyeball.** |
| `docs/PHYSICS-LAYER.md` §16–17 | Physics layer, fully documented |

## 2. What was asked for

A post on low blocks (three rules: stay compact, wide trigger, stay connected)
and the request that **mid block and high block should behave as low blocks when
out of possession, with different areas of the pitch, and that "if you don't
have the ball, don't allow spaces" is a rule.**

Implemented as one discipline with three presets:

```python
BLOCK_PRESETS = {
    "low":  (back 19, mid 31, front 37 | hw 20/16/11 | shift .42 squeeze .30),
    "mid":  (back 30, mid 42, front 48 | hw 22/18/13 | shift .45 squeeze .24),
    "high": (back 42, mid 54, front 60 | hw 24/20/15 | shift .50 squeeze .18),
}
```

Every team gets one; `pressing_profiles.profile_for_style` picks the height, so
there is one source of truth. `park_the_bus → low`, `gegenpressing → high`,
`attacking`/`tiki_taka → mid`.

## 3. THE OPEN QUESTION — run this first

**The 26/27 regression against the generalised block was never seen.**

Previous baselines were all `29 passed, 1 failed` — the one failure being the
known pre-existing `test_pass_matrix_sums_match_real_events` (395 vs 394).

```powershell
$env:PYTHONHASHSEED="0"
Set-Location "D:\PLOFA\plofa"
& ".\.venv\Scripts\python.exe" -m pytest tests\tests.py -q
```

**Why it matters more than before:** the low-block-only version was gated on a
boolean true only for `LOW_BLOCK_CONTAIN`. The generalised version applies to
**every team out of possession**. The blast radius changed completely. If it
moves, the block needs a config flag rather than being on by default.

## 4. The remaining fix: CB zones

Measured, defending, the two CBs hold separate lanes (15.7m mean) but
**4 of 49 defending frames put them 0.1m apart**. The cause is in
`defensive_block_target`:

```python
sign = 1.0 if state.current_y >= centre else -1.0
```

Each CB keeps whichever side of the block centre he is *already* on. That is not
zone ownership — it is "do not cross the middle" — so when both drift to one
side they get the same target and collapse.

**Fix:** assign zones from `home_x`/`home_y` rather than inferring from current
`y`. Left CB owns the left lane, right CB the right. Then stacking is impossible
by construction, and "may leave his zone but must return" becomes a property of
the target rather than an average. Roughly ten lines in
`defensive_block_target`.

## 5. Claims already proven true in PLOFA (don't re-litigate these)

| claim | measurement |
|---|---|
| CBs spread in possession to offer passing options | 17.4m mean y gap, never below 3.8m, 0/48 frames stacked |
| CBs never share a y in possession | 0/48 breach |
| CDM is free in y, not pinned | y range 10.4–59.2, spread 48.8m, sd 10.2 |
| CDM is a genuine passing option | 30 passes, 22 receipts as a named target |
| No teleporting to block positions | 1,253,572 integrator calls, largest single call **0.6869m** (6.87 m/s) vs a 1.20m ceiling |

The no-teleporting proof is the one to re-run after any movement change.

## 6. Known-wrong measurements — do not repeat these errors

Three separate times a measurement was confounded and produced confident
nonsense. Each was caught only by instrumenting the right population.

1. **Averaging position over all frames** while asking a question about
   *defending*. A team's back four push up when it has the ball, so a naive
   mean said a low block sat 42m from its own goal — which reads as "they are in
   the opponent's half" and means nothing. **Split by possession.**
2. **Matching a club name against a side key** (`"Hartwell City" in "home"`) to
   decide the depth axis. Never true, so the axis was mirrored and every number
   was wrong while looking plausible. **Use the side key directly.**
3. **Sampling between integrator calls** and calling it movement. Reported a
   112m "teleport" that was a kickoff repositioning. **Measure before/after a
   single call.**

Two hypotheses were also wrong, both caught by measuring:
- Suppressing CK37/CK38 made the block **3m shallower**, not deeper, despite
  "back-line spread" sounding like a re-widening operation. Now documented in
  `match_engine.py` so it is not retried.
- Giving the block more integrator ticks would have changed **nothing**: players
  already tracked the final target to within 1.2m. The target was wrong, not
  the movement.

**The lesson: instrument the layer you suspect, and let it name its own
culprit. Do not reason from function names.**

## 7. Current measured block, and the known gap

| defending, ball central | value | target |
|---|---|---|
| back four, from own goal | 58.6m | 27.4m |
| midfield | 52.2m | — |
| front two | 49.2m | 43.3m |
| back four → midfield gap | 6.4m | separated ✓ |
| midfield → front two gap | 3.1m | outlet exists ✓ |
| block width | 46.4m | narrows ✓ |

**The structure is right and the depth is wrong.** A tracer bisection of every
shaping layer proved the target survives the chain **completely untouched**
(27.4m at the anchor substitution, 27.4m at the end) — so there is no downstream
overwrite and the earlier belief that one existed was wrong. The players simply
do not spend enough of their 1.25M integrator calls in the block branch: only
~2.5% of them were captured as "low-block CB, out of possession".

So the open question is a **frequency** question, not a geometry one: why is the
block consulted on so few ticks? That is the next thing to establish after the
regression is green.

## 8. Rules that still apply

- `$env:PYTHONHASHSEED="0"` **before** Python starts. Setting it afterwards does
  nothing — string hashing is fixed at interpreter startup.
- Always `.\.venv\Scripts\python.exe`, not bare `python`.
- `run_match.py` lives in `D:\PLOFA\plofa\`, not `D:\PLOFA\`.
- Never run `auto_run_match.py` (Trevor-only; overwrites `season_state.json`).
  Its pure helpers are fine to import.
- The open determinism bug is **cross-process** divergence on the production
  squad path (`id()`-dependent, positions not ordering). The engine *does* replay
  given identical inputs; the earlier "minute 28" diagnosis was wrong and is
  corrected in `docs/PHYSICS-LAYER.md` §16.2.
