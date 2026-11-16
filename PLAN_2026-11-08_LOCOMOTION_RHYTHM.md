# PLAN — LOCOMOTION RHYTHM: stand, jog, burst, stand

**Written:** 2026-11-08
**Status:** not started. Deliberately not started before matchday 10 — see §6.
**Scope:** how bodies move off the ball. Not the ball, not the brains, not the
shapes. The attacking matrix and the phase regressions are untouched.

---

## 0. Why this exists

The broadcast reads wrong, and it is not the viewer. Bodies are continuous now
(3568 → 4 big steps, see commit `147ff39`), the ball belongs to someone
(108 → 0 landings), the sprint counter is honest (commit `34a1368`). What is
left is the **rhythm of off-ball movement**: nobody stands still, nobody jogs,
and the measured distance is wrong by 29–39%.

Measured on Hartwell 2–2 Thornfield, seed 42, 101 min (`_raw_player_path.json`
at 5 Hz, all 28 players):

| band (Di Salvo edges) | engine BOOKS | body ACTUALLY did |
|---|---|---|
| standing (< 0.5 m/s) | 5.2% | **31.0%** (24.6% completely frozen) |
| walking (0.5–2.0) | 64.3% | 47.8% |
| jogging (2.0–4.2) | 14.7% | 12.1% |
| running (4.2–7.0) | 13.3% | 7.9% |
| sprinting (≥ 7.0) | 2.5% | 1.1% |

Real Premier League outfielder, roughly: **20 / 40 / 20 / 12 / 3**.

Distance, same match, engine booking vs the path the viewer actually draws:

| player | booked | drawn (5 Hz) | ratio |
|---|---|---|---|
| Luca Ferrini | 12.74 km | 9.85 km | 1.29 |
| Mateo Sanz | 11.02 km | 7.31 km | 1.39 |
| Tavish Crane | 12.26 km | 8.66 km | 1.42 |
| **squad total** | **187.0 km** | **134.6 km** | **1.39** |

So there are two independent defects and they hide each other:

* **the ladder has no jog** — its base pace *is* a walk (§1), and
* **the meter over-reads** — it records the step the engine *ordered*, not the
  distance the body *travelled* (§3).

Fixing the ladder first would mean tuning the new constants against an inflated
number, and the result would be a second generation of the same error. **Order
matters: 3 → 2 → 1.**

---

## 1. Defect A — the ladder has no jog, and no rest

`match_engine._JOG_SPEED` (line ~2873) gives every outfield role **1.75–2.0
m/s**. Di Salvo's walking band ends at 2.0, so the constant the engine calls
"jog" is **walking by the band edges the same file uses**. Plain off-ball
travel cannot land anywhere but the walk band.

Three multipliers then push it down further. Measured share of all walking
time, from the recorded path (`tools/diag/_diag_walk_band_shape.py`):

| branch | code | pace | share of walking |
|---|---|---|---|
| far-side involvement gate | `jog * 0.30` | 0.53–0.60 m/s | **31.8%** |
| block shuffle | `jog * 0.55` | 0.96–1.10 | 18.4% |
| settled patrol orbit | `jog * 0.80` | 1.40–1.60 | 7.6% |
| plain shape travel | `jog` | 1.75–2.00 | 21.0% |

Nothing in that table produces **2.5–3.5 m/s**. The only way past walking is a
special branch — touch assignment `0.85 × top` ≈ 6 m/s, block/cover recovery
`0.75 × top`, striker press — so the distribution is **bimodal: walk or
burst**. That is exactly what the broadcast looks like: men creeping, then
lunging. Real football is trimodal with a fat jog in the middle.

The rest state is missing for a documented reason: the patrol orbit and block
shuffle were added because the *opposite* bug came first — 98% of dead-phase
block ticks (234,157 of 239,409) were booked as standing when bodies were
genuinely frozen at their slots, and distance collapsed. The model is stuck
oscillating between two failures.

---

## 2. Root cause — the constants were tuned against the inflated meter

The comment on `_JOG_SPEED` records that the CB pace was raised until CB
totals reached ~9.2 km, "inside the CIES figure", measured by
`_diag_cb_pace.py`. That tuning target was read off the **booked** distance,
which runs 1.39× hot. The target was met by inflating the meter, not by moving
the bodies: booked 12.7 km looked correct, drawn was 8.7 km of creep.

**Consequence for this plan:** no constant below can be trusted until §3 is
done, and any future pace tuning must quote the drawn distance, not the booked
one.

---

## 3. Fix 1 (do this first) — measure the body, not the order

**File:** `match_engine.py`, `_record_offball_distance` (line ~3007) and its
call sites at `_offball_move_player` (line ~3795), `_offball_run` (~3143) and
the set-piece path (~7299).

**Now:** the caller passes `moved`, which is the *intended* step,
`min(speed * DT, dist)`. The physics layer then moves the body less than that
(acceleration limits, body separation, target flips). Everything downstream —
`physics_distance_m`, the five band buckets, `physics_top_speed_mps`, the
sprint bout input, stamina — is fed the intent.

**Change:**

1. In `_offball_move_player`, capture the body position at tick entry
   (`before = (cx, cy)`).
2. After the physics body has been placed for this tick, read the body's new
   position (`after`) from the same store the viewer reads — the physics
   position, **not** `_phys_target`.
3. Pass `moved = hypot(after - before)` and band on
   `moved / duration_s`, into `_record_offball_distance`.
4. Keep the `moved == 0` early-return branch: a frozen tick must book
   `standing` time, not disappear (that was the 2026-10-17 fix and it must
   not regress).

**Do not** re-try banding `moved / duration_s` while `moved` is still the
intent — that was measured and is inert, because `moved == speed * DT` exactly
whenever the body is more than one step from its target. The fix is only a fix
when `moved` is the *actual* displacement. (Comment left in the code at the
`NOTE ON speed` block records this.)

**Expected:** booked distance drops ~28%, so a squad total near 135 km instead
of 187 km, and the band table finally matches the picture.

**Gate:** after this change alone,
`sum(physics_distance_m) / sum(drawn path)` must be **0.95–1.10**. Any other
value means the position being read is not the one the path records.

---

## 4. Fix 2 — give the ladder a jog

**File:** `match_engine.py`, `_JOG_SPEED` (~2873), the far-side gate
(~3564), `_BLOCK_SHUFFLE_FRAC` (~2933), the patrol constant in
`_offball_move_player` (`jog * 0.8`, ~3726).

**Change:**

* **Base pace → a real jog.** 1.75–2.0 m/s becomes **2.8–3.2**, role-ordered
  the same way (GK lowest, wide forward highest). This is the single change
  that moves the mass out of the walk band and into 2.0–4.2.
* **Far-side gate 0.30 → 0.45.** A player on the far side of the pitch still
  jogs; 0.53–0.60 m/s is a shuffle no footballer does for a whole match. This
  is the largest single contributor to the walk band (31.8%).
* **Block shuffle stays slow** (0.55 × the new jog ≈ 1.6 m/s) — a defender
  adjusting his line genuinely walks, and the band wants those minutes in the
  walk bucket.
* **Patrol orbit** 0.80 × the new jog ≈ 2.4 m/s, in the jog band.

**Interaction with distance:** base pace roughly ×1.5 and the gate ×1.5 will
push drawn distance up hard. With §3 in place the number is honest, so tune it
against the target below, and expect the first attempt to overshoot (the
current drawn figure of 8.7–9.9 km for outfielders is *below* the real
10–12 km, so there is headroom, but not 1.5×).

**Gate:** drawn distance per outfielder **10–12 km per 90**; keeper **4–5 km**
(see §7 — keepers are a separate defect and will not reach this by
tuning the outfield ladder).

---

## 5. Fix 3 — a rest state with a duty cycle

**File:** `match_engine.py`, `_offball_move_player`, the two `else` branches
that run once the player is inside the 2.0 m arrival band: the block shuffle
(~3698) and the patrol orbit (~3719).

**Now:** both branches move every tick, forever. That is why nobody stands.

**Change:** give each player a deterministic 6 s cycle — **~2 s adjusting,
~4 s holding** — phased by a stable per-player seed (the existing
`sum(ord(c) for c in pname) % 628` pattern; no global RNG draw, per the
project's reproducibility rule).

* During the *adjust* window: existing behaviour (shuffle / orbit).
* During the *hold* window: **position frozen**, `moved = 0`, booked as
  standing.
* The cycle resets immediately on a trigger — ball within the chase radius, a
  new touch assignment, a defensive emergency, a set-piece slot. A rest state
  must never delay a player who is actually needed.

This is the watchdog risk of the whole plan: the 98%-standing bug was the
first attempt at "players hold their position". The difference is that a
**hold** is now bounded to 4 s and interruptible, rather than an unbounded
state, and it runs *after* the recovery/approach branches that actually move
players into position.

**Gate:** standing 18–22%, and **total distance must not fall** below the §4
target. If distance collapses, the hold window is eating the adjust window.

---

## 6. Sequencing, commits, and rollback

| # | change | commit | why this order |
|---|---|---|---|
| 1 | §3 measure the body | `locomotion: record the travelled step` | every later number depends on it |
| 2 | §4 pace ladder | `locomotion: a real jog, and a softer far-side gate` | only tunable once the meter is honest |
| 3 | §5 duty cycle | `locomotion: a rest state with a 2 s / 4 s duty cycle` | changes distance again; last |

Each is independently revertable. Do **not** squash them: if the duty cycle
regresses distance, the pace ladder underneath it must be recoverable on its
own.

Do not run this before a matchday. It changes distance, sprint counts, stamina
drain and the whole visual read, and it needs the gate in §8 — which is a real
run, not a unit test.

---

## 7. Adjacent defect, explicitly out of scope

**Keepers cover 0.78–2.19 km** against a real 4–5 km, and their band time is
booked for only ~2.5% of the match — they are skipped on most ticks. That is a
keeper-participation defect, not a pace-ladder defect, and tuning `_JOG_SPEED`
will not reach it. Track it separately; do not let it block this plan.

---

## 8. The gate

### Tools (already written, use these exact commands)

```powershell
# 1. The engine's own booking (what the stats page publishes)
& ".\.venv\Scripts\python.exe" tools\diag\_diag_locomotion_bands.py

# 2. Where inside the walk band the mass sits, with branch fingerprints
& ".\.venv\Scripts\python.exe" tools\diag\_diag_walk_band_shape.py

# 3. Do they actually stand? 0.2 s tick, no window
& ".\.venv\Scripts\python.exe" tools\diag\_diag_frozen_ticks.py

# 4. Booked distance vs drawn distance -- the inflation ratio
& ".\.venv\Scripts\python.exe" tools\diag\_diag_booked_vs_drawn.py

# 5. Sprint bouts: count vs time, and how long each high-speed effort lasts
& ".\.venv\Scripts\python.exe" tools\diag\_diag_sprint_fragments.py
& ".\.venv\Scripts\python.exe" tools\diag\_diag_effort_durations.py
```

Tools 2–5 need the raw path. Produce it with a traced match:

```powershell
$env:PLOFA_TRACE_JUMPS="1"
& ".\.venv\Scripts\python.exe" plofa_broadcast\broadcast.py --seed 42 --no-server --no-standalone
```

### Acceptance (all must hold, same match, seed 42)

| metric | target | how |
|---|---|---|
| standing | 18–22% | tool 3 |
| walking | 35–42% | tool 3 |
| jogging | 18–25% | tool 3 |
| running | 10–16% | tool 3 |
| sprinting | 2–5% | tool 3 |
| booked / drawn distance | 0.95–1.10 | tool 4 |
| outfielder drawn distance | 10–12 km / 90 | tool 4 |
| sprint bouts per outfielder | 25–60 | tool 5 |
| mean high-speed effort | 2–4 s | `_diag_effort_durations.py` |
| squad `physics_top_speed_mps` | max ≥ 8.0, not pinned to 7.78 | replay frames |
| frames with two keepers | 0 (unchanged) | `tests/test_sub_goalkeeper_guard.py` |
| players landing on the ball | 0 (unchanged) | `tests/test_body_physics.py` |

### Regression gate (must not move)

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests\test_sprint_bouts.py `
  tests\test_body_physics.py tests\test_physics_integration.py `
  tests\test_positional_realism.py tests\test_positional_play.py `
  tests\test_sub_goalkeeper_guard.py -q
```

Baseline: **80 pass** on the first four, and the movement suite at
**192 pass / 1 known failure**
(`test_out_of_possession_touchline_press_gated_by_press_intensity`, which also
fails at HEAD `45f6501` — do not "fix" it as part of this work).

### Football gate (not a unit test)

A 7-match run through `tests/real_match_gate.py` on the same seed, comparing
goals for / goals against / possession / xG against the incumbent. A movement
model that improves the bands but concedes more is not a win. Nothing is
promoted on the band table alone.

---

## 9. New tests to write (one per defect, written before the fix)

1. `test_recorded_distance_matches_drawn_path` — build a tick sequence, drive
   `_record_offball_distance` through the real `_offball_move_player`, assert
   booked distance is within 10% of the summed path. **This is the test that
   would have caught the 39% inflation years ago.**
2. `test_frozen_tick_books_standing` — a tick with zero displacement books
   standing time and does not vanish (pins the 2026-10-17 fix).
3. `test_jog_constant_is_a_jog` — every outfield entry in `_JOG_SPEED` is
   ≥ 2.0 m/s, i.e. outside the walk band by the module's own edges. A one-line
   guard against the exact regression that produced this plan.
4. `test_duty_cycle_rests_and_wakes` — a settled player is frozen for part of
   the 6 s cycle and moving for the rest; a ball entering the chase radius
   ends the hold within one tick.
5. `test_duty_cycle_is_deterministic` — same player, same match, same phase,
   across processes. No global RNG draw.

---

## 10. What success looks like on the broadcast

Not "more distance". The visible difference:

* players **stand on their toes** between engagements, then go — instead of
  drifting in a permanent shuffle,
* a winger **jogs across** to cover rather than creeping, so the block slides
  instead of crawling,
* sprints read as **2–4 s bursts** with clear recovery, not 0.16 s twitches,
* and the stats page stops disagreeing with the screen.
