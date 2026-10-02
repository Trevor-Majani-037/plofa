# PLOFA PHYSICS — continuous physical-time layer

**Status: the physics layer is built, tested and demonstrated, and attached to
`MatchEngine` behind a flag that defaults to off. It is consulted at four points
in a live match — pass blocks, keeper saves, player movement and ball flight
time — in every case *attenuating or refining* an existing model rather than
replacing it. The block gate is doing real work; the keeper gate reaches 9% of
shots; movement now carries momentum, fatigue and reaction; the ball takes the
time it should to cross the pitch. The engine's passes are still not *resolved
by* the physics layer. Read §16 before assuming otherwise.**

Run it:

```
$env:PYTHONHASHSEED="0"
& ".\.venv\Scripts\python.exe" -m physics.demo                  # §26 demonstration
& ".\.venv\Scripts\python.exe" -m scripts.physics.measure_gate  # what the gates do
& ".\.venv\Scripts\python.exe" -m scripts.physics.probe_determinism 3
& ".\.venv\Scripts\python.exe" -m pytest tests\test_physics.py tests\test_physics_integration.py tests\test_determinism.py -q
```

`PYTHONHASHSEED` must be set **in the shell**, before Python starts. Setting it
inside the interpreter has no effect — string hashing is fixed at startup — and
an earlier version of the determinism probe did exactly that, which made it look
like it was controlling the hash seed when it was not.

---

## 1. Files created

| File | Lines | What it owns |
|---|---|---|
| `physics/__init__.py` | 141 | Package surface and the rules the package obeys |
| `physics/continuous_time.py` | 207 | `ContinuousClock`, `MatchTime`, stoppage table |
| `physics/calibration.py` | 296 | DNA score → SI units. The one place conversion happens |
| `physics/player_motion.py` | 324 | `KinematicState`, `arrival_time`, `integrate` |
| `physics/ball_physics.py` | 403 | `Trajectory`, drag, gravity, pass profiles |
| `physics/collision.py` | 240 | `Candidate`, `arrival_contest`, loose-ball semantics |
| `physics/violations.py` | 172 | The no-teleportation guard, `ViolationLog` |
| `physics/world.py` | 374 | `PhysicsWorld`, `PassResolution`, timelines |
| `physics/adapter.py` | 550 | **The only module that knows about both worlds**; also the two gates and the active-adapter slot |
| `physics/demo.py` | 294 | §26 acceptance demonstration |
| `physics/movement.py` | 175 | Momentum, fatigue and the reaction gate for continuous movement |
| `scripts/physics/measure_gate.py` | 250 | Measures what the gates do, with a control |
| `scripts/physics/probe_determinism.py` | 300 | Where the engine does and does not replay |
| `scripts/physics/dump_opening.py` | 110 | Prints a match's opening, every field, for diffing |
| `tests/test_physics.py` | 745 | 58 tests, mapped 1:1 onto §22's fifteen clauses |
| `tests/test_physics_integration.py` | 690 | 35 tests: the additive guarantee, live engine, weather, both gates |

## 2. Files modified

| File | Change | Lines |
|---|---|---|
| `match_engine.py` | `MatchConfig.physics_enabled` / `.physics_strict`, both defaulting to off | +11 |
| `match_engine.py` | `MatchEngine._init_physics()`, `physics_enabled` property, `resolve_pass_physically()`, `physics_report()` | +64 |
| `match_engine.py` | class-level `physics = None` / `physics_unavailable = False` | +10 |
| `match_engine.py` | one call to `self._init_physics()` inside `_initialize_simulation()` | +1 |

Nothing else in the live pipeline was touched: not the brains, not
`PositionEngine`, not `squad_manager`, not `weather_physics`, not the world layer,
not the 26/27 ledgers.

---

## 3. What already existed (and changed the scope)

The brief's §1 lists what the engine has. Two of its requirements were
**already satisfied**, which is worth knowing because it means the work was
smaller than the document implies:

| Requirement | Existing implementation |
|---|---|
| §4 continuous clock | `MatchState.match_clock_s: float` — *"one continuous global clock"* (`match_engine.py:641`) |
| §18 fractional timestamps | `TimelineEntry.second: float` + `match_clock_s: float` (*"true global clock"*) |
| §21 weather | `rolling_decel_mult`, `pitch_grip`, `gk_reaction_mult`, `stamina_drain_mult` |
| §14 stamina | `PlayerStaminaState`, `drain()`, `drain_baseline()` |
| §6 DNA physicals | `pace` *"Raw sprint speed"*, `acceleration` *"0→top speed quickness"* |

**The real gap** was identified in the engine's own comment at
`match_engine.py:2715`:

> *"position jumps we must not re-derive as distance/sprints from the position gap"*

Movement today is proportional steering — `tx += (trix - tx) * p_tri` — with no
velocity, no acceleration and therefore no arrival time. §15's
no-teleportation invariant is not merely unenforced; the engine *works around*
violating it.

---

## 4. Integration points

**Seams 1–4 and the adapter half of 5–6 are built and tested against a real
engine.** What is missing is routing the engine's *own* passes through the
adapter, and the event flag.

| # | Seam | Status |
|---|---|---|
| 1 | Construct a world | **Done** — `MatchEngine._init_physics()`, opt-in, tolerant of a missing package |
| 2 | Read positions | **Done** — `EngineAdapter.position_of()`; re-read every call, never cached (§16) |
| 3 | Read stamina | **Done** — `EngineAdapter.stamina_of()` reads `sub_controller.stamina`; no second copy |
| 4 | Read weather | **Done** — `calibration()` / `ball()` map `rolling_decel_mult` and `pitch_grip` |
| 5 | Plan a pass | **Gate shipped** — the block gate is consulted in a live match (§16.1); full pass resolution is not yet routed |
| 6 | Resolve the ball | **Gate shipped** — keeper reach is consulted on 9% of shots (§16.1); full resolution is not yet routed |
| 7 | Advance the clock | **Done for the ball** — flight time shifts the receiving event, and the engine derives the match clock from event timestamps, so the delay propagates (§17) |
| 8 | Flag the event | **Partly** — `MatchEvent` has `minute: int` + `second`; the receiving event now carries a flight-time-shifted `second` (§17) |

**Gating rule, and the proof it works:** `MatchConfig.physics_enabled` defaults
to `False`. With it off the engine never constructs an adapter and never
imports the package. That is asserted the strongest way available — a **fresh
subprocess** runs a match with physics off and then checks `sys.modules`:

```
test_a_default_match_never_imports_the_physics_package
```

If the package is absent from `sys.modules` when the match finishes, no line of
it executed, so no line of it could have changed the result. That is a stronger
guarantee than "the output looked the same".

### Three things the engine turned out to do that the adapter had to respect

Found by wiring it up, not by reading the brief:

1. **`MatchEngine.__init__` owns the weather gate.** It calls
   `WeatherPhysics.set_active_weather(cond, enabled=config.weather_enabled)` at
   `match_engine.py:1849`, and `weather_enabled` defaults to `False`. So
   *constructing any engine switches weather off* — setting it beforehand is
   silently undone. The adapter does not override this (that would be changing
   engine behaviour from inside a physics layer) but records it in
   `Calibration.notes`, because the alternative is silently simulating a dry
   match in the rain.
2. **Every weather multiplier is gated behind `is_active()`.** A rainy
   `WeatherCondition` with weather off returns 1.0 for all of them.
3. **`set_squad` stores its argument verbatim.** Production supplies
   `SquadBuilder`'s `PlayerProfile` objects — `match_engine.py` reaches for
   `taker.dna` — but a caller passing raw `(name, pos, specialties)` tuples
   leaves tuples in `active_players`, and `position_engine.states` gets keyed by
   the *stringified tuple*. The adapter handles both shapes via `_name_of()`,
   because the alternative is a physics layer that silently finds zero players
   and produces a match where nobody ever contests a ball.

---

## 5. Equations and models

**Player kinematics — three phases, each costing time.**

Velocity is split relative to the target direction:

```
u        = unit vector to target
v_along  = v · u
v_perp   = v − v_along·u
t_turn   = |v_perp| / (deceleration · (1 + agility))
```

Then accelerate, capped at `max_speed`:

```
d_to_vmax = (vmax² − v0²) / 2a

if straight ≤ d_to_vmax:      t = (√(v0² + 2·a·straight) − v0) / a
else:                         t = (vmax − v0)/a  +  (straight − d_to_vmax)/vmax
```

```
arrival = reaction + t_turn + t_accel + t_cruise
```

**Ball, ground, with drag** — `v(t) = v₀·e^(−kt)`, so `s(t) = (v₀/k)(1 − e^(−kt))`,
inverted to:

```
t = −ln(1 − k·d/v₀) / k
```

`k → 0` recovers the brief's `d/v` exactly, so the simple form is the
*approximation* and drag is the truth. A ball that drag would arrest before
travelling the distance returns `inf` — physically impossible, not merely slow.

**Ball, vertical** (§11): `z(t) = z₀ + v_z·t − ½g·t²`, `g = 9.80665`.
Flight time `2·v_z/g`, apex `v_z²/2g`. A lofted delivery uses a parabola through
three known points (launch height, apex, landing) so a cross has a believable arc
without solving for a launch angle.

**No spin, no Magnus, no turbulence, no bounce** — §11 and §24 both say not to.

---

## 6. How player speed is derived

Not `pace = speed`. §6 forbids that, and `pace` is a 0-100 opinion score.

```
max_speed = 4.5 + (pace/100) · 5.0        # 4.5 – 9.5 m/s
```

Anchored to §6's ranges. The DNA **default of 60 calibrates to 7.5 m/s** — a
realistic top-flight sprint. Score 0 is 4.5 m/s (slow, not immobile); score 100
is 9.5 m/s (elite, not superhuman). The mapping is linear so it stays
explainable: one pace point is worth a fixed amount of speed.

## 7. How acceleration works

```
acceleration = 0.6 + (accel/100) · 2.2    # 0.6 – 2.8 m/s²
deceleration = acceleration · 1.4
```

Anchored to sprint splits rather than guessed: ~5 m in 1.4 s, 10 m in 2.0 s,
20 m in 3.4 s, top speed around 4-5 s, implying ~1.9 m/s² at `pace = 60`.

Consequences the tests assert: one 10 ms step never reaches top speed; a player
covering 50 m from rest takes longer than `50 / max_speed`; a player *already at
speed* gets there sooner.

## 8. How ball speed is derived

Per pass type, from real ranges (§9), midpoint used unless a speed is supplied:

| | m/s | | m/s |
|---|---|---|---|
| short | 8-14 | switch | 20-26 |
| progressive | 12-18 | cross | 14-20 (vz 5.5) |
| through | 18-24 | clearance | 18-25 (vz 3.0) |
| long | 16-22 | shot | 20-32 |

So a through ball is *genuinely faster* than a short pass, not just labelled
differently. Clamped to 40 m/s.

## 9. How ball travel time is calculated

By the inverted drag solution above. The two invariants §10 demands both fall
out of it and are tested:

- longer distance + same speed → longer time
- same distance + faster ball → shorter time

Closed-form, so the clock jumps straight to the event. No stepping, so a match
costs the same computing one pass or ten thousand.

## 10. How player arrival time is calculated

The closed form in §5. Returns **0.0** if already within `REACH_M` (0.8 m — a
footballer's reach is not zero).

The payoff, isolated as its own test: **11 m at full sprint beats 4 m from
standing.** Velocity is worth more than proximity, which is what §7 asks for and
what a `distance / max_speed` model cannot produce.

## 11. How stamina affects physical capability

§14 forbids the "100 = fast, 50 = slow" step, so the modifier is smooth, linear
and **deliberately asymmetric**:

```
speed multiplier      = 0.92 + 0.08 · (stamina/100)   # loses at most 8%
acceleration multiplier = 0.70 + 0.30 · (stamina/100) # loses up to 30%
```

A tired player keeps most of his top speed but loses acceleration — which is
what actually happens. You can still sprint at 90 minutes; you cannot get there
as quickly. Tests assert the asymmetry, the linearity, and that neither
multiplier ever reaches zero.

## 12. How z / vertical motion works

One extra float on the ball. The pitch stays strictly x/y; `z` is independent
and carries no rendering. `BallState` exposes `(x, y, z, vx, vy, vz)`. A ground
pass has `apex == 0` and `z == 0` throughout, and that is asserted.

## 13. How timestamps are represented

- `ContinuousClock.play_seconds` — float, the clock physics runs on
- `ContinuousClock.stoppage_seconds` — float, accumulated separately (§19)
- `ContinuousClock.match_seconds` — the sum, i.e. the scoreboard
- `MatchTime(minute: int, second: float)` — `37:21.735`, with `minute` staying
  an `int` so existing consumers keep working (§18)

Time only moves via `advance()`. It cannot rewind, and a negative delta raises
rather than un-firing an event.

## 14. Performance impact

| | |
|---|---|
| 1800 passes, 90 simulated minutes | **300 ms** of CPU |
| 90 simulated minutes vs 90 real minutes | **~18,000× faster** |
| Sleeping / rendering | **none** |
| Cost to a match that does not use it | **zero** — the package is not imported |

The brief allows ~85 s per match; the physics arithmetic is a rounding error
against the engine's own event simulation.

## 15. Tests performed

**93 in total — 58 in `test_physics.py` plus 35 in `test_physics_integration.py`
— all passing.** `test_physics.py` maps 1:1 onto §22's fifteen clauses: each
test's name cites the clause it satisfies, because a requirement that is not a
test is a wish.

Full fast suite: **465 passing**, including the world layer, perception, the
determinism work and the integration tests. `tests/tests.py` (the 26/27
regression) is run separately as the live-season guard.

### Nine real bugs the tests, the demo and the integration caught

*Found in the physics work:*

1. **Timestamps printed as `37:2100.000`.** `f"{second:06.3f}"` renders 21.0 as
   `021.000`. Now millisecond-correct with carry handling.
2. **A contest compared against the wrong clock.** `world.py` passed an absolute
   `arrival_time` to `arrival_contest` against candidates whose arrival was
   *relative*, producing margins like *"arrives 2240.427s early"*.
3. **A receiver controlled the ball before it arrived.** Control is now
   `max(ball travel, player arrival)` — the attacker gets there early, waits, and
   the ball arrives to him. Matches the brief's own illustration, where
   `ATTACKER ARRIVAL` and `CONTROLS BALL` share a timestamp.
4. **A won through ball was reported as "unclaimed".** "Nobody arrives *before*
   the ball" is not "nobody gets there at all" — for a pass played into space
   the ball arriving first is the *point*. Introduced `LOOSE_BALL_WINDOW` and a
   `loose` flag, and the demo now reports *"arrives 0.573 s after the ball — a
   loose ball he collects"*.

*Found by wiring it to the engine:*

5. **The adapter read the wrong attribute.** The engine stores the stamina
   controller as `sub_controller`, not `stamina_controller` — so every player
   silently read as 100% stamina. The worst kind of bug: no error, just physics
   that ignored fatigue entirely.
6. **A live fixture that fed the engine tuples instead of players.**
   `set_squad` stores its argument verbatim, so `active_players` held raw tuples,
   `position_engine.states` got keyed by the *stringified tuple*, and the adapter
   found **zero** players. The fixture now builds real `SquadBuilder` objects by
   the same route the live month driver uses, and a test asserts the fixture
   itself is non-empty so it cannot silently rot again.
7. **The engine's constructor silently disables weather.** Setting it before
   building a `MatchEngine` is undone by `match_engine.py:1849`. The adapter
   reports the condition rather than overriding it.
8. **`engine.physics` raised `AttributeError` before `simulate()`.**
   `_initialize_simulation()` runs from `simulate()`, not `__init__`, so
   class-level defaults were needed — "off" must be a value you can ask about.
9. **A fragile round-trip in `with_stamina`.** It reconstructed 0-100 DNA scores
   by inverting SI values and re-calibrating. It was lossless, but pointlessly
   fragile; it now constructs the profile directly.

### Structural guards, not behaviour tests

Six tests exist to keep the package honest rather than to check arithmetic:
the clock module cannot import `time`/`datetime` (AST-parsed — an earlier
version grepped the source and failed because the module's own docstring
*explains* that it never calls `time.sleep`); no module may import `random`,
`secrets` or `uuid`; none may import the live pipeline at module level;
`resolve_pass` is checked for CPU independence by burning 300 ms of real CPU
mid-run; velocity estimates are clamped so a substitution cannot read as a
runner; and the additive guarantee is proven by subprocess import inspection.

## 16. What is NOT done

Stated plainly, because §27 says not to claim full physics without the tests to
back it.

- **The engine's own passes are not routed through the physics.** Seams 5 and 6
  work from the adapter and are tested against a real engine, but
  `MatchEngine` still resolves its passes the way it always has. A live match
  produces no physics *timeline* — you have to ask for one.
  **This is still the single biggest remaining piece of work.**
  - **What landed instead is a *gate*, not a resolution.** Rather than replacing
    the engine's block model with an arrival-time contest (see below), the
    physics now *attenuates* it. Two gates are live, both opt-in, both inert when
    `physics_enabled` is False.
  - **The pass-block gate** — `event_chain.py:2548`, between
    `_pass_block_probability` and its dice roll. The engine's model is a
    *spatial* test: `_pick_pass_blocker` finds the nearest defender within
    `BLOCK_CORRIDOR_M` of the lane. It never asks whether he can arrive. The gate
    adds the *temporal* question and scales the probability down when he cannot,
    floored at the engine's own 0.02. It can only ever reduce a block.
  - **The goalkeeper gate** — `GoalkeeperEngine.evaluate_save`, between
    `_is_shot_savable` and the xG division. `effective_reach = reach * (0.8 +
    reaction_time * 0.4)` gives a keeper identical reach against a shot from 30 m
    as against one from 8 m, though the first takes ~1.20 s and the second
    ~0.27 s. The gate turns flight time into dive capacity and attenuates
    `save_mult`. Because the caller already floors `save_mult` at 1.0, pushing it
    toward 1.0 converges on the raw xG — the engine's own documented safe
    fallback — so it cannot inflate scorelines the way the inverted multiplier
    once did.
    **Measured reach: 9% of shots** — `evaluate_save` handles 4 of the 46 shot
    events in a match, so as wired this gate is very nearly inert. See §16.1.
  - **Why attenuate rather than replace.** Both gates follow one rule: *remove
    credit the physics cannot justify, never add any.* Replacing the block
    probability with a hard contest would turn a tuned 2–45% probability into a
    guaranteed winner, make every interception certain, and move every scoreline
    in a season calibrated around it — using physics that has never been
    calibrated against PLOFA's output. Substituting the keeper model would be the
    same mistake, and its existing logic (angle bisection, reflexes-driven
    reaction, jump-driven reach) is good.
  - **Reaching the gates required one module-level slot.** `PossessionChain.generate`
    and `GoalkeeperEngine.evaluate_save` are class/static methods with no engine
    reference, so `physics/adapter.py` publishes the current match's adapter in a
    single slot (`set_active_adapter` / `active_adapter`). **One slot, not a
    registry** — a dict keyed by engine id is the `id()`-reuse contamination bug,
    and entries would outlive the match that created them. `_init_physics()`
    clears the slot on *both* paths, via `sys.modules` lookup rather than import,
    so a disabled match neither inherits a stale adapter nor loads the package.
    That last detail was caught by
    `test_a_default_match_never_imports_the_physics_package`, which failed the
    first time it was written for this reason.
  - **The gates are `try`-wrapped, which hides a `NameError`.** A gate that
    raises on a missing name would be silently inert — wired, never firing, and
    indistinguishable from "the physics had nothing to say". So the call sites
    are AST-checked to confirm every name they reference is in scope, and
    `measure_gate.py` prints the gates' own counters so a zero is visible rather
    than inferred.
  - **Measured, not asserted** — `scripts/physics/measure_gate.py` reports the
    gates' own counters (how often they fired, how hard, how many blocks they
    removed) separately from end-to-end scorelines, because the two answer
    different questions. The counters count *decisions*, so they do not need a
    reproducible match; the scorelines do. See §16.1 and §16.2.
  - **Why the measurement tools live in `scripts/physics/` and not in
    `physics/`.** They are harnesses, not the layer: they import `random` (to
    seed the *engine's* RNG) and `time` (to measure cost).
    `tests/test_physics.py` asserts that no module in `physics/` imports
    `random`, on the grounds that no physics decision may come from a random
    draw. That guarantee is worth more than the convenience of co-location — an
    earlier draft of this work had all four tools inside `physics/`, and the
    guard test failed on `dump_opening.py`. **The tools moved; the guard was not
    weakened.**
- **Seam 7 is half-done.** The physics clock follows the engine's
  `match_clock_s`, but the engine does not read the physics clock back, so a
  resolved pass does not yet move the match's own clock.
- **Seam 8 is partly satisfied already.** §18's fractional timestamps exist in
  `TimelineEntry`, and `MatchEvent` carries both `minute: int` and
  `second`, so an event's absolute time is recoverable without a new field.
  What is still missing is a physics-resolved event *carrying its arrival time*
  as a distinct value — a pass that is resolved by the physics layer would want
  to record when the ball actually reached the receiver, which is not the same
  as when the chain decided to pass.
- **Not calibrated against real football.** The numbers are anchored to published
  sprint splits and §6's ranges, which is a defensible first pass and *not* a
  calibration. The drag coefficient and the tier scales are the two values most
  in need of real data.
- **No spin, Magnus, bounce, wind vector or surface friction.** §11 and §24 both
  exclude them from a first implementation. `pitch_grip` and `rolling_decel_mult`
  are now wired, so the hooks exist when the rest is wanted.
- **Player *positions* still come from the engine** — and that is now a
  deliberate division of labour rather than a gap. See **§17**: the physics owns
  *how the body reaches the target*, the engine owns *which target*. Velocity,
  fatigue and reaction are all wired (§17); the physics does not own position,
  so §16's single authoritative position is intact and the tuned shape gains
  survive.
- **Velocity is estimated, not stored** for *queries*. The engine has no
  velocity field, so the adapter differentiates position over a 0.35 s window
  for arrival-time questions — the block gate's view. A player who changes
  direction inside one window is reported as still travelling the old way. §17's
  `MovementMemory` is authoritative for *movement*, but the gate's velocity is
  still a finite difference, and a velocity field on `PlayerSpatialState` would
  fix that properly.
- **One determinism bug is still open in the engine.** The previously recorded
  diagnosis — *"a player-iteration ordering divergence at minute 28,
  localised to the off-ball path"* — **was wrong, and is corrected here.**
  Measured, with `scripts/physics/probe_determinism.py`:

  | Configuration | Result |
  |---|---|
  | Same seed, hardcoded squads, 3 runs, one process | identical, 3,283 events, sha `e59c7889d1dbad5a` |
  | Same seed, hardcoded squads, 3 **separate processes** | identical, same sha |
  | Same seed, `PYTHONHASHSEED=1` instead of 0 | 3,024 events — **8% less football** |
  | Same seed, production squad path, 3 separate processes | 3,057 / 3,060 / 3,095 — **diverges** |

  So three separate things were being called one bug:

  1. **The engine does replay** when given identical inputs. The "minute 28"
     divergence was an artefact of comparing runs that had not been given
     identical inputs, not a fault in the engine.
  2. **`PYTHONHASHSEED` genuinely matters** — and traces to exactly three
     `hash()`-on-string sites (`ball_vision.py:124`, `perception.py:422`,
     `perception.py:519`). Those are frozen **on purpose**: changing them
     changes the live 26/27 season, and `perception.py` already carries a
     comment saying so. `tests/test_determinism.py` pins the count at three so
     a fourth shows up as a failure rather than a mystery.
  3. **The real open bug is narrower and nastier**: on the *production squad
     path*, three fresh interpreters with the same seed and the same hash seed
     produce three different matches. Iteration order of players is **stable** —
     what differs is player *positions*, from minute 1:40. So it is
     allocation-dependent (`id()`, or an unseeded generator), not set-order and
     not `hash()`. The roster loader is exonerated: `build_matchday_squad`
     returns byte-identical squads when called three times running.

  - **This blocks less than previously claimed.** The gates are measured at the
    *decision* level — 69 block opportunities, 62% judged physically impossible —
    and counting decisions does not require a reproducible match. It only blocks
    end-to-end A/B scoreline comparison, and only on the loader path; the
    hardcoded-squad path is reproducible and can carry that measurement today.
  - **An earlier measurement blamed this bug for its own noise and was wrong.**
    `measure_gate.py` reported passes −169 while the gate had evaluated 69 block
    opportunities, and the cause was read as engine non-determinism. The real
    cause was the harness: it never called `random.seed()`, so the second
    condition continued from wherever the global stream had reached. The engine
    was replaying; the experiment was uncontrolled. It now seeds per fixture and
    runs a **control** — the baseline twice, physics off both times — and refuses
    to present section B if that control differs.
- **No collision or challenge model.** Two players arriving within 30 ms is
  reported as `TIE` and left to the caller. A header, a tackle and a duel all
  need a model that does not exist here.

## 16.1 What the gates actually do — measured

Numbers from `scripts/physics/measure_gate.py` and `scripts/physics/probe_gk_reach.py`, one
90-minute match with real squads.

### The pass-block gate is doing real work

| | |
|---|---|
| Block opportunities evaluated | **69** |
| Attenuated | **43** (62%) |
| — of those, physically impossible | **36** |
| — of those, driven to the 0.02 floor | **0** |
| Left alone because the defender arrives in time | 26 |

So on the majority of the block opportunities the engine surfaces, the physics
says the defender **cannot get there before the ball**. That is the gap, measured
rather than argued: `_pick_pass_blocker` finds a defender within
`BLOCK_CORRIDOR_M` of the lane and never checks the clock.

Note that **nothing was driven to the floor**. The 0.02 floor essentially never
binds, which is the intended behaviour — the gate scales probabilities across
their whole range rather than flattening them against a bound.

### The keeper gate is placed correctly and reached almost never

| | |
|---|---|
| Shot events in the match | **46** (17 on target, 24 off, 5 goals) |
| Calls to `GoalkeeperEngine.evaluate_save` | **4** |
| Share of shots the gate sees | **9%** |
| Of those 4: no time to dive / out of reach | 1 / 3 |

**This corrects an earlier claim.** The keeper gate was added on the strength of
"it's the most scrutinised moment in football, so it's the highest-value one
available" — and that reasoning was wrong, because it never checked how many
shots actually route through `GoalkeeperEngine.evaluate_save`. They do not. Most
shots are resolved by other code paths that construct `SHOT_ON_TARGET` events
directly. The gate is correct where it sits and genuinely inert almost
everywhere else, so as shipped it is close to a no-op.

Making it matter means wiring the other shot paths, which is a wider change to
`event_chain.py` than the two call sites made so far, and is not done.

### Cost

`0.95x` wall clock — the physics run was marginally *faster* than the baseline
over one fixture, so the gate's arrival-time arithmetic is not a measurable
expense. An earlier 4-fixture run showed the physics at `1.61x`, but that was
contention from a concurrent test run, not the gate; it is corrected here.

## 17. Physical movement, fatigue, reaction and ball flight

Four gaps listed in earlier revisions of this document as "not wired" are now
wired, all behind `MatchConfig.physics_enabled`. **111 tests pass, and the 26/27
regression is unchanged: 29 passed, 1 failed — the same pre-existing
`test_pass_matrix_sums_match_real_events` (395 vs 394) that fails without any of
this work.**

### The division of labour

> **The physics decides how the body gets there. The engine decides where he is
> trying to be.**

Every line of the engine's shape logic — the shape engine, wide stretch,
triangle support, back-line build-up drop, pressure-aware spread, live-spacing
redirect — still runs and still decides the target. The physics replaces only
the resulting *position*, by integrating the body toward that target.

This is the reason the work was safe to do at all. Making the physics *own*
position would have contradicted §16 and invalidated the tuned `p_tri` gains —
a re-tuning project. Making it own the *step* adds momentum, an acceleration
limit, fatigue and reaction without touching a single shaping constant.

### What the engine already had, stated precisely

An earlier revision called the movement "proportional steering", which was wrong
and worth correcting because it changed the size of the job. The
`tx += (trix - tx) * p_tri` formula only steers the **target**. The step that
follows is already respectable — time-stepped, pace-capped by position, ramped
when a chase engages, recording a real speed. What it lacked was *velocity
state*.

### 1. Momentum

Direction was recomputed from scratch every tick, so a player could reverse at
full pace. Now the velocity is carried in `MovementMemory` and integrated with
`player_motion.integrate`.

Measured: at 8.0 m/s, a 180° reversal takes **~0.9 s of braking** at the 9.0
m/s² deceleration limit before he drives the other way, covering 19 m in 4 s
rather than 32 m. An earlier draft of the test asserted speed fell below 7.0
m/s within one 0.1 s tick, which would have required players to turn like a
top-spinning table-tennis ball.

### 2. Fatigue in the legs

`_top_speed_cache[p] = 5.0 + pace * 0.042` is **DNA only** — a player on 20%
stamina ran at exactly the pace of one on 100%, so the number never changed and
the legs never felt the match. `profile_of()` now supplies a stamina-scaled max
speed: **7.50 → 7.02 m/s at 20%**, a 6% loss that compounds across a whole
match of repeated sprints.

### 3. Reaction delay

0.20 s, and modelled on the right thing. The delay is on the player's
*information*, not his legs: while the reaction is pending he keeps running
toward the position he last believed in, and only then corrects. A "stand still
for 0.2 s" model would waste his legs and look like a statue.

It arms only on a **material** target change (≥ 2 m, `REACTION_TRIGGER_M`).
Shape targets drift centimetres every tick, and arming on those would hold the
entire team in permanent reaction — slower than the steering it replaced, and
wrong.

PLOFA's DNA has no reaction trait, so this is a flat constant rather than a
per-player one. That is deliberate: inventing a per-player value would be a new
calibrated parameter pretending to be a measurement.

### 4. Ball flight time driving the sequence

PLOFA already computed a flight time and stored it as
`geometry_meta["ball_travel_s"]` — then used it for exactly one thing: recovering
pass *distance* for the miscontrol model (`ball_travel_s * ball_speed`). The
timeline ignored it entirely. The pass and the touch that ended it were stamped
at the same instant, so **a 40 m ball in behind was logged as arriving at the
moment it was struck.**

`pass_flight_time()` now shifts the receiving event (and a miscontrol) forward by
the real flight time. Because the engine derives its clock from event
timestamps — `match_clock_s += dur`, where `dur` spans the chain's events — the
match clock inherits the delay for free, and defenders get genuine time to close
the gap before the next action. In one match the accumulated flight time is
roughly 250 s of additional clock.

### A real bug this exposed, in the physics package itself

`player_motion.integrate` computed `v_along = v · desired` as a **signed**
value, then did `v_along + acceleration * dt`. A player running *away* from his
target has a negative `v_along`, so this made it more negative — and after
multiplying by `desired`, it **accelerated him directly away from where he was
trying to get to.** A 180° turn at full pace came out as "keep sprinting the old
way".

Every arrival-time query the module was written for starts from rest facing the
target, so `v_along >= 0` always and the sign never came up. It surfaced the
moment the stepper was used for continuous movement, where overshooting a shape
socket is ordinary rather than exceptional.

Fixed: moving away, the along-component **brakes** at the deceleration limit and
is never allowed to grow backwards. He coasts to a stop, then drives the other
way — which is what a body does. The 58 existing physics tests pass unchanged,
which is the useful part: they were correct about what they covered and silent
about what they did not.

### A silent no-op that nearly shipped

The flight gate initially read `_pdm` for the pass distance. `_pdm` is bound
inside the **reactive-block branch** a few hundred lines earlier, so a completed
pass that never entered that branch would raise `NameError` at the receiving
event — swallowed by the surrounding `except Exception` — and **never shift the
clock while looking perfectly wired.** A gate that is silently inert is
indistinguishable, in a scoreline, from a gate the physics had no opinion about.

Distance is now derived from the pass origin and destination, both in scope at
the receiving event, and `test_the_flight_time_does_not_depend_on_being_gated_on`
fails if anyone reintroduces the dependency. The `try/except` stays — a physics
problem must never cost a match — which is exactly why the static check has to
exist alongside it.

### Substitution

`MovementMemory` is dropped in `invalidate()` alongside the cached velocity. It
holds *carried momentum* and a *held target*; a substituted player inheriting
either would arrive at his new position already travelling at the pace and in
the direction of a man no longer on the pitch.

## 16.2 Reproducibility — measured, and partly misdiagnosed before

`scripts/physics/probe_determinism.py` is the tool. Its findings are in the
determinism bullet in §16; the short version is that the engine **does** replay
given identical inputs, that `PYTHONHASHSEED` matters because of three frozen
`hash()` sites, and that the one genuine open bug is cross-process divergence on
the production squad path. `scripts/physics/dump_opening.py` prints the opening of a
match with every field, which is how the divergence was traced to positions
rather than to ordering.

`tests/test_determinism.py` holds the findings that are cheap to enforce: the
exact set of `hash()`-dependent sites, the absence of unseeded numpy generators
on the match path, and the presence of the corrected diagnosis in this file.

### The end-to-end A/B, and a wrong excuse for its earlier failure

An earlier run reported passes −169 while the gate had evaluated only 69 block
opportunities. That was written off as engine noise, and **that explanation was
wrong** — the engine replays. The harness never seeded `random`, so the second
condition simply continued from wherever the global stream had reached. Corrected
in this file, and `measure_gate.py` now seeds per fixture and runs a control
(baseline twice, physics off both times) before it will show section B at all.

The honest summary is therefore narrower and more useful than "the noise won":
**the mechanism is measured and trustworthy** (section A — it counts decisions,
not outcomes), **and the end-to-end A/B is not yet trustworthy on the loader
path** until the control passes. When it does not, the script says so rather
than printing a table that looks like a result.
