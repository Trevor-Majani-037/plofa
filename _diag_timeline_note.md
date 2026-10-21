
## THE TIMELINE'S DENSITY — THE FLOW IS REAL, THE PACE INSIDE A POSSESSION IS NOT (2026-10-21)

The user, reviewing the export fix: *"the engine's times are a model, not a
stopwatch... it means what? football wise? does it compromise the new flow?"*

**MEASURED (`_diag_timeline_football.py`, one real match).**

| | PLOFA | real football |
|---|---|---|
| pass-like events in a match | 797 | ~1,050-1,150 (both teams, PL) |
| MEAN interval between passes | **7.09 s** | **~5 s** ✓ close |
| MEDIAN interval between passes | **1.10 s** ✗ | 3-5 s ✗ |
| p90 interval between passes | **40.87 s** ✗ | ~15 s ✗ |
| the first 26 events of the match | **7.04 s** ✗ | ~60 s ✗ |
| a possession's internal spacing | 0.3-0.6 s per action ✗ | 2-4 s ✗ |
| the gap after a possession ends | 40-115 s ✗ | 10-20 s ✗ |

**SO: THE TOTAL IS RIGHT AND THE DISTRIBUTION IS WRONG.** 797 passes at 7.1 s
each is football-plausible; the same 797 passes bunched into 1.1-second
spacings with 40-115 second silences between possessions is not.

**AND THE CAUSE IS THE ENGINE'S CLOCK, NOT THE EXPORT.** A "chain" is the
engine's unit of work (a possession, a set piece, a shot sequence) and its
window is the clock advance during it — which is the **BALL's motion** ✓.
So the chain's off-ball actions (the press, the duel, the reception, the
decision) are resolved *inside the ball's flight time*: a 30 m pass's ~1.2 s
flight carries a PASS, a MISCONTROL, a PRESS and the next PASS ✗✗. And
`chronograph()` then spreads even a long chain evenly across its events, so
a chain of four events in 1.2 s puts them 0.3 s apart ✗.

**Real football's possession is flight + decision + execution** — a player
takes 0.5-1.5 s to decide what to do with the ball, and the off-ball events
happen over those seconds, not inside the flight ✗. **That is the missing
time, and it is the engine's clock semantics, not the exporter's
presentation.**

**DOES IT COMPROMISE THE NEW FLOW? NO — the flow is a different property.**
- The ORDER is real ✓ (the events are appended in chain order and the
  chronology has **0** out-of-order steps ✓).
- The times are monotonic and the inter-chain gaps are the clock's own ✓.
- The export no longer lies about WHEN an event happened — the timestamps
  are the engine's truth and they agree with the engine's own chronology ✓.

**What it compromises is the DENSITY** ✗ — and that is exactly what a
watcher's eye catches next: the opening of the export reads as a 7-second
scrum followed by a 53-second silence, where a real match's first 26
actions take a minute or two ✗. **The flow is fixed; the pace is the next
defect in this thread, and it is an engine change, not an export change.**

**THE FIX, and its cost.** The chain's window must cover the football's own
durations rather than the ball's motion alone: the ball's flight ✓ (already
there) **plus the receiver's decision time** ✓ (0.5-1.5 s, from the same
football literature this project cites for its other calibrated quantities)
**plus the off-ball actions' own durations** ✓ — and the events placed
**sequentially by type** instead of spread evenly ✓. Then the inter-chain
dead time shrinks to compensate, so the match's total clock stays at ~96
minutes ✓.

**THE COST, STATED PLAINLY:** it changes **every match's timeline** ✓ — the
chronology, the momentum model's xT series (which reads the clock ✓), and
every calibration number taken on the current clock ✗ — so it needs its own
gate and a multi-match sweep, and it is **not** a change to make inside a
thread that also changed the export. It is the next piece, with its own
commit.
