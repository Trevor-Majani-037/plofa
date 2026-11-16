# Brain Cost Analysis — "all brains at 10 Hz" vs the hybrid

**Date:** 2026-10-27
**Status:** measurement only. No project file was modified. No season state was touched.
**Runners used:** a purpose-built harness that mirrors `run_match.py` and writes nothing. `auto_run_match.py` was never called.

---

## 1. Headline numbers

| Measurement | Value |
|---|---|
| Full 90-minute match, **current hybrid** | **148.4 s** |
| **Brain calls in that whole match** | **3,048** |
| **Total time spent in those calls** | **0.40 s — 0.3 % of the match** |
| One `offball_press_prob` call, end to end | **293–391 µs** |
| …of which the **neural network itself** | **28.6 µs (7.4 %)** |
| …of which **perception** (the inputs) | **284 µs (73 %)** |

The first and most important result: **the brain is not the expensive part. Building
what the brain looks at is.**

The second: **the hybrid already calls the brains only ~0.25 times per second.**

### Live call census (one full match, instrumented)

| Brain hook | Calls | Time |
|---|---:|---:|
| `_offball_press_prob` (per-player press gate) | 1,347 | 0.25 s |
| `_team_press_g` (shared unit controller) | 576 | 0.09 s |
| `OffBallBrain.forward` | 1,035 | 0.05 s |
| `_def_action_choice` (defensive action) | 30 | 0.01 s |
| `DefensiveActionBrain.forward` / `.predict` | 30 / 30 | <0.01 s |
| **total** | **3,048** | **0.40 s (0.3 %)** |

1,347 calls over 5,400 seconds is **0.25 Hz**. That is the number "10 Hz for
everyone" would have to be compared against — a **~390× increase in calls**.

---

## 2. Where the 387 µs actually goes

Profiled stage by stage on a live engine, CM (a position that really owns an
evolved brain — GK/CB/ST return `None` in 0.45 µs and are excluded):

| # | Stage | µs | Share |
|---|---|---:|---:|
| 1 | `_build_views` (weakref-cached) | 0.4 | 0.1 % |
| 2 | roster list comprehensions | 2.8 | 0.7 % |
| 3 | **`perceive_ball`** | **87.5** | **22.6 %** |
| 4 | **`perceive_offball_actors`** | **196.5** | **50.7 %** |
| 5 | `extract_offball_sensors` | 26.1 | 6.7 % |
| 6 | **`brain.forward` — the actual NN** | **28.6** | **7.4 %** |
| | **total** | **387.2** | 100 % |

Inside stage 4 alone, ~84 µs is random-number machinery that is rebuilt from
scratch on **every single call**:

| | µs |
|---|---:|
| `np.random.default_rng(seed)` — constructed per call | 26.7 |
| 44 × `rng.normal` (2 draws for each of 22 actors) | 57.0 |

---

## 3. What "10 Hz for everyone" actually costs (measured, not projected)

I forced a fresh brain decision on **every tick for every chasing player**, defeating
both the memo caches and the press-window gate. Measured over a full match:

| | Baseline | All-10 Hz | Change |
|---|---:|---:|---:|
| Wall time | 148.4 s | **164.9 s** | **+16.5 s (+11 %)** |
| `_offball_press_prob` calls | 1,347 | **84,427** | 63× |
| `_team_press_g` calls | 576 | **84,427** | 147× |
| `OffBallBrain.forward` calls | 1,035 | **112,427** | 109× |
| Time inside brains | 0.40 s | **24.14 s** | 60× |
| Share of wall time | 0.3 % | **14.6 %** | — |

**My earlier projection of 2.3–3.8× was wrong, and the reason is instructive.** I
assumed all 22 players would be consulted every tick. They are not: the press
decision lives *inside* `if chasing:`, so only players actually near the ball are
ever asked. 84,427 calls over 5,400 s is ~15.6 Hz across the whole pitch, or about
**0.7 Hz per player** — not 10 Hz per player.

So the honest headline is: **forcing the brains to 10 Hz costs +11 % match time,
not +130 %.** It is affordable. That makes the argument against it sharper, not
weaker — the cost is no longer the obstacle, so the *behavioural* case has to
carry the whole decision.

---

## 4. Does it improve anything? — No, and here is the precise reason

`perceive_offball_actors` computes its noise from a seed derived as:

```
draw_seed = crc32(f"{player}:{x:.3f}:{y:.3f}:{n_def}:{n_tm}")  ^  cfg.seed
```

That makes perception a **pure, deterministic function of (seed, player, position)**.

Consequences:

1. **More frequent calls add decisions, not information.** The seed is a hash of the
   player's *position*, so as he moves the noise field is sampled afresh — the brain
   is never eating stale perception. But each call produces one independent sensor
   vector, and the brain consumes **one vector per decision**. Sampling the same
   underlying truth ten times a second does not make any single decision
   better-informed; it just produces nine additional decisions that then have to be
   acted on. The current design already re-samples the field every time a press
   window opens, so the brain is not being starved of information at 0.25 Hz.

2. **The existing throttle is already at decision level, not tick level.** In
   `_offball_move_player` the brain is consulted only when a press window *opens*:

   ```python
   if cst["p"] <= 0.0 and cst["allow"] == 0.0:
       ... g = _team_press_g(...)  /  prob = _offball_press_prob(...) ...
       cst["allow"] = 1.0 if random.random() < prob else -1.0
   ```

   `cst["allow"]` is a **commitment** — it stays set for the life of the press. A
   press commitment is a discrete act with momentum. Re-deciding it every 100 ms
   would not make the player smarter; it would make him twitch, committing and
   un-committing a press ten times a second while the surrounding shape integrator
   has no time to carry the body anywhere. That is a realism regression dressed up
   as fidelity.

3. **It would perturb the seeded football stream.** Perception draws from an RNG
   built per call. Calling it more often draws at different moments, so a seeded
   match would no longer reproduce. That is a correctness regression, not an
   improvement.

---

## 5. So is the current cadence right?

Measured, not assumed: the hybrid consults the per-player press brain
**1,347 times in 5,400 seconds — 0.25 Hz.**

I then tried to *raise* that cadence with a per-player 3 Hz throttle. It changed
almost nothing:

| Arm | Wall | `offball_press_prob` calls | Brain time |
|---|---:|---:|---:|
| baseline | 148.4 s | 1,347 | 0.40 s |
| **realistic @ 3 Hz** | 137.8 s | **1,475** (+9.5 %) | 0.35 s |
| all-10 Hz (forced) | 164.9 s | 84,427 (63×) | 24.14 s |

**The 3 Hz throttle was never the binding constraint, so it did almost nothing.**

That is the real finding of this section. The cadence is not set by a deliberate
"think every N ms" throttle that could be turned up — it is set by **how often a
press window opens**, which is a fact about the football (how often a defender finds
himself within 12.5 m of a live ball with the press undecided). Adding a 3 Hz timer
on top of that changes 1,347 calls into 1,475.

So there are only two ways to make off-ball decisions more frequent:

- force the gate open every tick → 63× the calls, +24 s of CPU, and a player
  re-deciding his press commitment ten times a second (twitchy, not smarter);
- **change the football** so press windows open more often — i.e. better press
  *triggers*. Same speed, more decisions, because there is more to decide.

The second is where the fidelity actually lives, and it costs nothing to add.

Biology for context, already encoded in `possession_physics.reaction_time()`:
0.42 s (slow reader) → 0.15 s (elite reflexes) is **motor reaction**. Perceptual
decision-making — recognising a press trigger and choosing to commit — is slower
than that, not faster.

---

## 6. Where the real optimisation is

If match time ever needs to come down, the 73 % spent on perception is the target,
and it can be cut without touching football behaviour:

1. **Reuse the RNG.** `default_rng(seed)` costs 26.7 µs per call and is rebuilt
   every time. A seeded `RandomState` pool, or drawing from a single generator
   keyed by the same `crc32`, removes that per-call cost outright.
2. **Cache perception per (player, rounded position).** Stage 4 is a pure function
   of its inputs, so memoising on `(player, x, y)` is safe and exact — the same
   guarantee the existing `_offball_press_cache` already relies on.
3. **Hoist the `_truth()` lookups.** `perceive_offball_actors` calls
   `position_engine.get_position()` **44 times** per invocation (22 for visibility,
   another 22 inside the distance sort). One pass over the roster removes the
   second sweep.

Those three are pure wins: same behaviour, less work. Running everything at 10 Hz
is the opposite — more work, same behaviour.

---

## 7. Verdict

**Do not switch to all-brains-at-10 Hz.** Measured, it costs +11 % match wall time
and 60× the brain CPU to re-take a decision that is already being taken as often as
the football asks for it. It buys nothing, and re-rolling a press commitment ten
times a second would make players twitch rather than read the game.

The deeper result is that **cadence was never the lever.** A 3 Hz throttle moved the
call count by 9.5 %; the frequency is dictated by how often press *windows* open,
not by any timer you can wind up.

If off-ball play is to get more realistic, the lever is the **press trigger**: make a
press fire on a cue (receiver back to goal, heavy touch, lane closed, unit already
jumping) instead of a Bernoulli over a situational vector. That adds decisions
without adding per-decision cost, and it is the part of real off-ball play your
engine currently does not have at all — alongside scanning, which it also lacks.

The hybrid design is correct. The expensive thing was never the brain, and the
brain is only 7 % of the work.
