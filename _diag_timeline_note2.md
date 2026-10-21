
## THE DECISION TIME — THE MISSING TIME, AND THE TIMELINE NOW READS LIKE FOOTBALL (2026-10-21)

The user, on the export's time axis: *"i dont to see 26 actions in 7 seconds
then later 50 seconds of nothing. that decision time is the missing time.
yes i think fix that with urgency and ensure it is well."*

### THE DIAGNOSIS, MEASURED BEFORE ANYTHING WAS TOUCHED

`_diag_clock_anatomy.py`, one real match:

| | before | real football |
|---|---|---|
| live (play) time in a 96-minute match | **1,410.7 s = 23.5 min (24.7%)** | ~55-60 min (~60%) |
| DEAD time | **4,289.5 s = 71.5 min (74.4%)** | ~36-40 min |
| seconds per event inside a chain | **0.449 s** (median) | 2-4 s |
| the gap between one possession and the next | **46.35 s** (median) | 10-20 s |
| the median interval between passes | **1.10 s** | 3-5 s |
| the first 26 events of the match | **7.04 s** | ~60 s |
| chains / events per chain | 418 / 6.92 | — |

**THE CAUSE IS THE ENGINE'S CLOCK, AND IT IS ONE SENTENCE:** a chain's
window is the clock advance during it, and that advance is the **BALL's
motion** ✓. So the off-ball actions inside a possession — the press, the
duel, the reception, the *decision* — were resolved inside the ball's
flight, and `chronograph()` then spread the chain's events evenly across
it. **A 30 m pass's ~1.2 s flight carried a PASS, a MISCONTROL, a PRESS and
the next PASS.**

**REAL FOOTBALL'S POSSESSION IS FLIGHT + CONTROL + DECISION + EXECUTION.**
A player takes the ball, controls it, looks up, decides, and only then plays
it. That decision time was missing from the CLOCK.

### THE FIX — `match_timeline.py` (new, pure, no engine imports)

- **the chain's window** = the ball's motion **+ the chain's decision
  time** ✓ — the time the ball's carrier spends between receiving the ball
  and playing it (the control 0.3 s + the decision 0.8 s) ✓;
- **the off-ball contests do NOT extend the window** ✓ — a press and a duel
  happen ALONGSIDE the on-ball action, which is what real football does and
  why they are not added ✓;
- **the events are placed by the football's own weights** ✓, not spread
  evenly ✓: a long pass's flight (distance / 22 m/s ✓) weighs more than a
  short one's, a carry's travel (distance / 5 m/s ✓) more than a press's
  contest ✓ — so the timeline reads as a possession instead of a burst ✓;
- **the clock fixes itself** ✓ — the engine's dead-time phase is
  `remaining = 60.0 - (the clock's elapsed time in this minute)` ✓, so the
  chains eating football-real time leaves proportionally less dead time ✓:
  **the match still totals 96 minutes ✓ and the silences shrink ✓.**

### MEASURED AFTER — the user's complaint, the same match

```
BEFORE:  00:00.000 PASS     00:00.280 MISCONTROL   00:00.580 PASS
         00:00.280 PRESS    00:00.890 BALL_RECEIPT  00:01.190 GROUND_DUEL
         00:00.580 PASS     00:01.490 PRESS        00:01.790 PASS
         ... 26 actions in 7.04 s ...   then 01:00.040 (a 53-second silence)

AFTER:   00:00.000 PASS     00:02.480 MISCONTROL   00:03.800 BALL_RECEIPT
         00:04.060 CARRY    00:05.730 PRESS        00:06.430 PASS
         00:07.740 BALL_RECEIPT  00:08.000 CARRY  00:09.290 PRESS
         ... 30 actions in 22.0 s ...
```

| | before | **after** | real football |
|---|---|---|---|
| live share of the match | 24.7% | **55.9%** | ~60% ✓ |
| median interval between passes | 1.10 s | **3.52 s** | 3-5 s ✓ |
| p90 interval between passes | 40.87 s | **13.99 s** | ~15 s ✓ |
| seconds per event inside a chain | 0.449 s | **1.309 s** | 2-4 s (closer) |
| the gap between possessions | 46.35 s | **28.70 s** | 10-20 s (closer) |
| the first 26 events | 7.04 s | **~19 s** | ~60 s |
| mean interval between passes | 7.09 s | **6.98 s** | ~5 s ✓ (preserved) |
| the match's total clock | 96.4 min | **96.4 min** | 96 min ✓ (unchanged) |

**THE MEAN IS PRESERVED AND THE DISTRIBUTION IS REPAIRED** ✓ — which is the
signature of a redistribution rather than an addition ✓.

### THE GATE — `tests/test_match_timeline.py` (15, NEW file)

**It pins the OUTCOME, not the constant** ✓ — a test that asserts
`DECISION_S == 0.8` asserts nothing about the football ✓, so the slow gate
runs a REAL match and pins what the number is FOR: the live share (50-70%),
the median interval between passes (2-6 s), the densest chain's span
against its event count, the gap between possessions (median < 40 s, max
< 70 s), the match's total clock (96 ± 3 min ✓), the timeline's
monotonicity ✓, and that the events' stamped clocks and the chronology
agree ✓. Plus the module's arithmetic (an on-ball event consumes decision
time; an off-ball contest does not; a long pass weighs more than a short
one; every event gets a positive weight ✓) and two STRUCTURAL guards (the
engine calls the module ✓, and the even spread `span * i / (n - 1)` is
gone ✓).

### THE SIDE EFFECT, MEASURED AND STATED: THE DISTANCES MOVED

**The live/dead split changed ✓ (24.7% → 55.9% live ✓) and the live
phase's off-ball movement is the CHASING (faster ✓) while the dead
phase's is the HOLDING (slower ✓) — so moving time from dead to live
raises the distance covered ✓.** Measured, one match
(`tools/diag/_diag_measured_activity_verify.py`):

| | before (2026-10-19/20) | **after** | CIES n°68 |
|---|---|---|---|
| CB | 8.41 km | **10.41 km** | 9.2 km |
| LB | 9.66 | **11.00** | — |
| CDM | 9.80 | **11.18** | — |
| CM | 11.10 | **12.28** | 10.6 km |
| LW | 10.54 | **11.91** | — |
| ST | 10.04 | **11.82** | — |
| standing share | 4.3-16.9% | **4.3-7.9%** | 20-30% (CB) |

**So the per-position distance calibration of 2026-10-20 is now STALE** ✗ —
the numbers are 1.3-2.4 km higher ✗ and the centre-backs are now above
the CIES figure ✗. **This is a CALIBRATION consequence, not a defect** ✓
(the live share is now real ✓, so the distance follows ✓) — but it must be
re-fitted against the benchmark, and that is a deliberate multi-match step
of its own ✓, exactly as the standing rule says about a fit ✓.

**THE STANDING SHARE IS UNCHANGED AND STILL GOOD** ✓ (4.3-7.9% ✓, the
2026-10-19 fix intact ✓) — the locomotion work is not undone by this ✓.

### WHAT IS STILL OPEN, AND IT IS NOT THE TIMELINE

**THE REMAINING 28-SECOND GAPS ARE THE POSSESSION STRUCTURE, NOT THE
CLOCK.** The engine runs **2-4 possession sequences per minute** and then
pads the rest of the minute ✓; real football plays **~10-12** ✓ (the ball
changes hands every ~15-20 s ✓). The gap IS the minute's leftover ✓ — and
it is the same fact as the other remaining gap: **PLOFA plays 814 passes
per match where the real Premier League plays ~1,050-1,150** ✓. More,
shorter possessions would fix both ✓ — but that is a change to the
SIMULATION (it moves every match's outcome ✗✗), it invalidates every
calibration number taken on the current possession structure ✗, and a
parallel session is mid-flight on the neural corpus ✗✗. **It is a separate
piece with its own gate, and it is not folded in here.**

**AND THE MATCH'S OUTCOMES BARELY MOVED, for a reason worth recording:**
the engine runs a FIXED number of possession sequences per minute
(`_simulate_minute`), NOT a clock-driven loop ✓✓ — so widening a chain's
window changes the TIMELINE, not who wins the match ✓. What it does change
is how long the off-ball integrator runs inside a live chain ✓, and the
live/dead split ✓ — which is the distance effect above ✓.
