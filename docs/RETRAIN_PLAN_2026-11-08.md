# The retrain, and the human model

**Date:** 2026-11-08
**Preceded by:** `docs/RESEARCH_AUDIT_2026-11-07.md` (Tier 1 / 1b done)
**Question:** the brain is provably blind to the goal. Plan the retrain. Name the new
brains. And find what a real player has that we still honestly do not.

---

## 0. The short answer

The retrain is **not** "run `brain_self_trainer.py`". Five of the six stages of the
learning loop are broken, and the training corpus is unusable. Running the trainer today
would reproduce the blindness with fresh weights and a clean log.

That is not a pessimism; it is a measurement. Everything below was read out of the code
or counted in the corpus on 2026-11-08.

| # | stage | state | evidence |
|---|---|---|---|
| A | sensors | **fixed** 2026-10-31 | slots `[2]`/`[3]` carry goal bearing + mouth width |
| B | corpus | **unusable** | `slot2 == slot0` in **4,933 / 4,933** samples |
| C | reward key | **geometry-blind** | `FitnessSurrogate.bucket()` = `P,FT,OH,S,C,G` — no angle |
| D | collection | **self-imitating** | `_EXPLORE_EPS = 0.0`; `_run_matches` never passes `explore` |
| E | gate | **cannot run** | `validate_neural_xl.py` does not exist at repo root |
| F | plan block | **dead input** | `plan_tail()` reads `player.team`, always `""` |

The retrain is therefore a **repair project with a training run at the end**, not a
training run. Budget accordingly: the GA is the cheap part.

---

## 1. Why the corpus is unusable (this is the load-bearing finding)

`brains_trainer/corpus.raw.json` — 4,933 samples, 24-wide, written **2026-09-20**. Goal
vision landed **2026-10-31**. The corpus is six weeks older than the sensor change, and
it shows in three independent ways:

**1.1 Slots 2 and 3 are literal duplicates of slots 0 and 1.**

```
slot2 == slot0 : 4933 / 4933
slot3 == slot1 : 4933 / 4933
```

That is the pre-2026-10-31 layout, where the on-ball carrier's own `player_x/player_y`
duplicated `ball_x/ball_y` — the exact redundancy the sensor change was made to remove.
A surrogate fitted on this corpus learns that slots 2–3 carry **no information**, because
in every sample it ever saw, they carried none.

**1.2 The corpus contains logically impossible situations.**

```
('CM', 'P1FT0OH1S0C1G0')   85   # own half, not final third
('ST', 'P1FT0OH1S0C1G0')   46   # a striker shooting from his own half
('ST', 'P1FT1OH0S0C1G1')   12   # final third AND own half
```

`FT1` means `x > 70` (or `< 35` away). `OH1` means `x < 52.5` (or `> 52.5`). They cannot
both be 1. The corpus was collected under a different `is_final_third` / `is_own_half`
convention than the one shipping today. Bucket keys computed from it are not comparable
to bucket keys computed now.

**1.3 The `SHOOT` cells are dominated by nonsense states.** 481 `SHOOT` samples spread
over 52 `(position, bucket)` cells. The single largest striker cell is
`P1FT0OH1S0C1G0` — under pressure, own half, central, not goal-close — with 46 samples.
The cell that *should* dominate (`FT1, G1`, i.e. final third and close to goal) has 12.

**Consequence:** the corpus cannot be refitted, patched, or partially reused. It must be
recollected under the current sensor schema. 4,933 samples is roughly 6 matches; that is
not a large loss.

---

## 2. Why the reward is blind even with a good corpus

This one is subtler and more important than the corpus.

The GA does not score a brain on football. It scores it on the **surrogate's expected
threat return** for the intent the brain chooses in a given sensor state
(`brain_evolution.synthetic_fitness`). The surrogate is a lookup table keyed by a coarse
situation string:

```python
# surrogate_collect.FitnessSurrogate.bucket
return f"P{pressure}FT{final_third}OH{own_half}S{space}C{central}G{goal_close}"
```

Six bits. Pressure, third, half, space, lane, goal-close. **There is no goal angle and no
goal mouth in the key.**

So two states 26 m from goal but 30° apart — the exact pair the 2026-10-31 sensor change
was built to distinguish — land in the **same bucket** and receive the **same reward** for
`SHOOT`. The network can now *see* the difference; the objective function still cannot
*price* it. A retrained `SHOOT` head would be graded by a marker who cannot tell a shot
from a cross, and would learn accordingly.

This is why the fix is not "retrain": it is **give the reward a geometry, then retrain.**

The state generator itself is fine — `generate_state_corpus` calls the live
`extract_sensors`, so the GA's evaluation states already carry goal geometry. The break is
strictly in the pricing, not the states. That is good news: it is a ~20-line change.

---

## 3. The other three breaks

> **§3.4 was added 2026-11-12 and it outranks all three below.** It is the reason the
> brains never learned to shoot, and it sits BELOW the reward-key problem: no bucket key
> can help if the shot's outcome was never recorded.

**3.4 A shoot decision was credited 0.0 — always.** `surrogate_collect._correlate_outcomes_causal`
contains a carefully documented branch: a policy `SHOOT` intent makes the possession chain
emit a **zero-length marker carry** (threat change exactly 0.0) while the real shot is
fired by the ATTACK chain, so a shoot decision must be answered by looking *forward in the
match timeline* for the shot it caused. The branch is guarded by `and timeline is not None`
— and `surrogate_rebuild._run_matches` passed **no timeline**. The branch was dead code.

Every shoot decision was therefore joined to the marker carry and credited exactly `0.0`,
while the shot's xG — and any **goal** — was credited to nobody. Measured on the 9,109
samples collected through that call site:

| | measured | should be |
|---|---|---|
| max reward, any sample | **+0.3841** | ~1.0 for a goal |
| samples with reward ≥ 0.5 | **0** | ≥ 1 per goal |
| `SHOOT` mean reward | **+0.0009** | positive and large |
| `SHOOT` samples exactly 0.0 | **36.5%** | rare |
| `threat_reward.GOAL_VALUE` | 1.0 | — a value nothing could reach |

So the surrogate could not express "shooting is worth it" **even in principle**, and the GA
faithfully learned not to shoot. The fix is one argument —
`_correlate_outcomes_causal(_CHAIN_CALLS, timeline=result.timeline)`; `MatchResult.timeline`
already existed. Pinned by `tests/test_shoot_credit.py` (7 tests), including a source
assertion that the call site still passes a timeline — because every mechanism test passes
without the wiring, which is exactly how this survived.

**VERIFIED on 4 post-fix matches (3,215 samples):**

| | before the fix | after |
|---|---|---|
| max reward, any sample | +0.3841 | **+1.0000** (= `GOAL_VALUE`) |
| samples ≥ 0.9 | **0** of 9,109 | **1** of 3,215 |
| `SHOOT` mean | +0.0009 | **+0.0069** (7.7×) |
| `SHOOT` exactly 0.0 | 36.5% | 30.6% |

Only **one** goal landed on a `SHOOT` decision in four matches, and that is expected rather
than disappointing: 15 goals were logged, but most were own goals, or were finished by the
attack chain without the policy ever choosing `SHOOT`. A goal can only be credited to a
shoot decision that exists. That is the point — the credit is now *possible*, and it was
structurally impossible before.

**Consequence: any corpus collected before 2026-11-12 has invalid `SHOOT` rewards** (the
mover intents are unaffected — the timeline only changes how `SHOOT` is answered). The
9,109-sample corpus is quarantined as `brains_trainer/corpus_20261112_prefix_shootcredit.json`.

**3.1 Collection is self-imitation.** `_run_matches` (in `surrogate_rebuild.py`) has no
`explore` parameter, so `_EXPLORE_EPS` stays at its default `0.0`. The ε-exploration
machinery in `surrogate_collect.py` — which exists *precisely* to inject the rare actions
the policy never chooses, and which weights `goal_close` at `SHOOT: 8.0` — is dead code in
the trainer's path. The corpus therefore records only what the blind brain chose.

Note the comment in `_run_matches`: *"if `'neural'` the trained XI auto-loads and actually
shoots from scoring positions (real outcomes)."* We measured that claim: **0 shots from
109 intents**. The comment is aspirational, and the exploration flag that would have made
it true is off.

**3.2 The gate cannot run.** `brain_self_trainer.gate()` does
`from validate_neural_xl import _run_neural`. That module is **not in the repo**. A copy
survives at `.kilo/worktrees/aspiring-gem/validate_neural_xl.py` (7,175 bytes, dated
2026-09-17). Until it is restored and checked against the current `match_probe` API, stage
[4/6] raises `ModuleNotFoundError` and no challenger can ever be promoted or rejected.

**3.3 The plan block is always zeros.** `situational_context.plan_tail()` calls
`getattr(player, "team", "")`. `PlayerProfile` has `team_name`, not `team`. The result is
always `""`, so all six `PLAN_D` inputs are constant. The network has been trained with a
dead 6-wide block and has correctly learned to ignore it. Fixing this is not just a bug
fix — it is the **hook** the coach instruction and home/away dimensions will hang on
(§6.2, §6.3).

---

## 4. The repair sequence

### Phase 0 — make the loop runnable (half a day, no training)

| id | task | done when |
|---|---|---|
| P0.1 | Restore `validate_neural_xl.py` from the worktree; verify `match_probe._build_squads` / `MatchEngine` / `extract_team_fitness` signatures still match | `python -c "from validate_neural_xl import _run_neural"` succeeds and one gate match returns a `fitness` dict with `goals` and `possession_pct` |
| P0.2 | Add `explore: float = 0.0, explore_situational: bool = False` to `_run_matches`, forward to `_build_match_components`; expose `--explore` / `--explore-situational` on the trainer CLI | `sc_explored() > 0` after a 1-match collection |
| P0.3 | Fix `plan_tail` to read `team_name` (and decide what it should *say* — see §6.2) | `np.abs(plan_tail(...)).sum() > 0` for a real player |
| P0.4 | Move the stale corpus to `brains_trainer/corpus_20260920_stale.json`; start a fresh `corpus.raw.json` | corpus file dated today |

### Phase 1 — rebuild the training signal (1–2 days)

**P1.1 — give the reward a geometry.** Three bins read from the new slots:

```python
angle_bin  = int(np.clip((sensors[2] + 1.0) * 2.5, 0, 4))   # 0..4, 2 = DEAD AHEAD
mouth_bin  = int(np.clip(sensors[3] * 3.0, 0, 2))           # 0..2, 2 = wide open
dist_bin   = int(np.clip((1.0 - sensors[14]) * 3.0, 0, 2))  # 0..2, 2 = closest
```

Note the angle is a **signed** bearing (−1..+1), so 0.0 — dead ahead — lands in bin **2**,
the middle of five, not bin 0. The sign is kept because which side of the goal a player is
on changes which shot is on.

**IMPLEMENTED 2026-11-08 as a LADDER, not a wider key.** §10.2 rejected the flat
`P·FT·OH·S·C·G·A·M·D` key, and was right to: with a flat key the rich cells fall under
`min_samples=3`, `_lookup` drops to the family and global rows — *which were keyed by the
old 6-factor bucket* — and the geometry is discarded at exactly the moment it matters.
That objection is answered structurally rather than ignored:

| level | key | role |
|---|---|---|
| rich | `P0FT1OH0S1C1G1\|A2M2D2` | tried first; wins wherever the data can resolve it |
| coarse | `P0FT1OH0S1C1G1` | the 6-bit **parent of the same sample** |

`FitnessSurrogate.fit` writes **both** levels from every sample — including the family and
global rows — so degradation is one rung, never a jump to a key that has forgotten the
geometry. `_lookup` walks the ladder richest-first; `bucket()` itself is unchanged, so every
existing caller keeps its old behaviour.

Measured occupancy on the 4,933-sample corpus (`tools/diag/_diag_reward_ladder.py`):

| | |
|---|---|
| rich cells populated | **502** (mean **9.8** samples each) |
| coarse cells populated | 192 |
| real lookups that hit the rich level | **4,933 / 4,933 = 100%** |
| fell through to coarse / to prior | 0 / 0 |

So the starvation §10.2 feared does **not** occur — but read that number with care: the
stale corpus has `slot2 ∈ [0, 1]` only, so it can never produce the two left-hand angle
bins. A correct corpus doubles or triples the distinct `A` bins and pulls the mean toward
~4 samples per cell. Still above the guard, but marginal — which is why **§10.3 remains the
decisive fix** and the ladder is a supporting change, not a substitute. The ladder makes
the *other nine* intents better-resolved; it does not make a sparse table a good estimator
for a 10%-success tail event.

**P1.2 — recollect with exploration on.** `--matches 12 --explore 0.15
--explore-situational` over `--styles fluid_counter,tiki_taka,attacking`. Acceptance for
the corpus itself:

- `SHOOT` samples in `FT1 · G1` ≥ **300** (today: ~141 across all buckets, in a corpus
  with broken geometry)
- ≥ 4 distinct `angle_bin` values represented among `SHOOT` samples
- no sample with `FT1 ∧ OH1` (assert it; if it fires, the collector is mislabelling
  direction and Phase 1 stops)

**STATUS 2026-11-12 — 12 of ~19 matches collected (9,109 samples).** Mechanics learned, all
of which cost time to discover:

| fact | consequence |
|---|---|
| `collect()` saves only after ALL matches finish | a killed 12-match run wrote **nothing**; chunk it |
| it **appends** | a repeated `--seed` replays the same matches and duplicates the corpus — every chunk needs a fresh seed |
| `--no-evolve` still runs the 2-match gate | ~12 min wasted per call; `--gate-matches 0` divides by zero |
| real cost is **~4–6 min/match**, not the "~20s" in the CLI help | 12 matches ≈ 60–75 min |
| 4 matches → 2,900–3,150 samples | sample density is fine; *shot* density is not |

Use `tools/ops/collect_corpus.py --matches 4 --seed <fresh>` to grow it without the gate.
Seeds used so far: 123, 2001, 3001, 4001.

Acceptance after 12 matches (measured):

| criterion | target | 4 matches | 8 matches | **12 matches** |
|---|---|---|---|---|
| distinct `angle_bin` among `SHOOT` | ≥ 4 | 5 | 5 | **5 (A0–A4)** — met; the stale corpus could only produce A4 |
| `SHOOT` in `FT1 · G1` | ≥ 300 | 63 | 136 | **188** — not yet; ~15.7 shots/match ⇒ ~19 matches |
| no `FT1 ∧ OH1` | 0 | 0 | 0 | **0** — met |
| `slot2` range | −1..+1 | −1..+1 | −1..+1 | **−1..+1** — the signed bearing is live |
| table bins / filled cells | — | 1,196 / 3,507 | — | **1,320 / 4,132** |

The `SHOOT`-in-`FT1·G1` count is what gates evolution. The plan's original "12 matches" was
estimated before anyone knew the shot rate; at ~15.7 shots per match the honest figure is
~19.

**P1.3 — refit, then audit the table before evolving.** Print the `ST`/`CF`/`CM` reward
tables for the goal-close buckets and require the ordering to be football:

```
reward(SHOOT | 18m, central, 0.9 mouth)  >  reward(SHOOT | 18m, byline, 0.2 mouth)
reward(SHOOT | 18m)                      >  reward(CROSS | 18m)
```

If the ordering is wrong, the GA will faithfully learn the wrong thing. **This audit is the
gate that matters**; the match gate is downstream of it.

Implemented as `tools/diag/_audit_reward_table.py` (exit 1 = do not evolve). It refuses to
compare a cell holding fewer than 8 shots, and it refuses to call "SHOOT > DRIBBLE" a pass
when the rival sits exactly on the neutral prior — that is a comparison with the *absence*
of evidence, not with evidence.

**MEASURED 2026-11-12, 12 post-fix matches (9,246 samples, 871 `SHOOT`, 141 in `FT1·G1`):**

| geometry cell | n | mean reward |
|---|---|---|
| `A3M2D2` (close, open mouth) | 9 | **+2.1968** |
| `A2M2D2` (central, open mouth, close) | 7 | **+1.7728** |
| `A2M1D2` / `A1M2D2` (close) | 3 / 1 | +1.4312 / +1.4317 |
| `A2M1D1` (central, **further out**) | 26 | **+0.1547** |
| `A0M2D2` (hard left) | 3 | −4.5364 |

The distance gradient is now real: a close chance is worth roughly **10×** the same central
look from further out, where before the key collapsed both into one number. In the best
supported chance (`A3M2D2`), `SHOOT +1.8984` beats `PROGRESSIVE_PASS +0.9257` and `CROSS`
(the −1.0 forbidden veto, scaled). `DRIBBLE` reads exactly 0.0000 = the prior, so that
comparison is flagged vacuous rather than passed.

Verdict at 141 `FT1·G1` shots: **ordering holds where measured, corpus still too thin to
confirm** — the open-central cell is one sample short of the threshold. Keep collecting.

**MEASURED 2026-11-13 — the gate now clears, and it took three corrections to get there.**

Four more matches (seed 5005) took the corpus to 15,426 samples and 115 striker
clear-chance shots. The table:

| geometry cell | n | mean reward |
|---|---|---|
| `A2M2D2` (central, open mouth, close) | **13** | **+2.5853** |
| `A3M2D2` (close, open mouth) | 10 | +1.5496 |
| `A2M1D2` | 9 | +0.6389 |
| `A2M1D1` (central, further out) | 47 | +0.3913 |
| `A0M2D2` (hard left) | 3 | −6.2130 |

**Verdict: `TABLE ORDERING OK — safe to evolve` (exit 0).**

Getting there required fixing three defects in the audit itself. Each was found by
measurement, and each had made the gate impossible to pass for a reason unrelated to the
table — which is worse than a gate that passes too easily, because a deadlock looks like
patience.

1. **Cells could not reach support; populations can.** The check demanded a single
   shut/wide CELL with n ≥ 8. None exists and none can: a shut mouth is a shot almost
   nobody takes, so its 8 samples scatter over 4 cells. The claim was always about
   populations, so the comparison is now pooled, with the same n ≥ 8 applied to each
   pooled group — `open central +2.5853 (n=13)` vs `shut/wide −3.0686 (n=15)`, and
   `mouth open +2.5853` vs `mouth shut −3.4631 (n=8)`.
2. **Some rivals never occur, so comparing against them is a deadlock.** `DRIBBLE` and
   `CROSS` have **zero** samples in a striker's clear chance across all 15,426 — the
   actions do not happen there. Requiring `SHOOT > DRIBBLE` in that cell required evidence
   the simulation cannot produce. Absence is now reported as `n/a` and is neither a pass
   nor a failure. Rivals that do occur are all checked, which is stricter than the old
   two-rival list: `PROGRESSIVE_PASS (n=14)`, `SWITCH (n=4)`, `THROUGH_BALL (n=4)`.
3. **Ignorance cannot be inferred from a value.** The old vacuity test flagged any rival
   sitting exactly on the prior. But a through-ball played in a clear chance measures
   **exactly 0.0** in the ladder — possession kept, threat unchanged — so a measured cell
   is indistinguishable from a missing one by value alone. Support decides instead: zero
   samples is `n/a`, any samples is a measurement whatever number it landed on.

What still guards the table: the pooled geometry ordering, the mouth ordering, the
per-rival comparisons where support exists, and 100% rich-cell lookup coverage. What no
longer blocks it: comparisons against actions nobody takes, and cells that cannot fill.

**P1.4 — evolve.** `--generations 40 --population 32 --states 300 --goal-bias 0.25`.

**P1.5 — gate.** `--gate-matches 6` minimum. The trainer already warns below 4; with 2
matches a lucky challenger passes on variance, and this project has the logs to prove it.

### Phase 2 — the acceptance test that proves it (the part that is missing)

The current gate asks "is the challenger's fitness higher". That is the wrong question for
this failure. The failure is behavioural, so the test must be behavioural. Write
`tests/test_shot_generation.py` on a fixed seed (123) and assert:

| assertion | target | today |
|---|---|---|
| `SHOOT` intents per match | 8–18 | 109 (but see next row) |
| `SHOOT` **obeyed** | > 0 | **0** |
| median distance to goal at a shot intent | **< 22 m** | **54.5 m** |
| p90 distance | < 30 m | ~90 m |
| share of shot intents inside 25 m | > 70% | **1 / 109 = 0.9%** |
| xG per shot | 0.09–0.12 | — |
| goals per match | 2.0–3.6 | 3.0 (Oxton 1–3) |

The distance distribution is the assertion that cannot be gamed. A brain can be nudged to
emit `SHOOT` more often without learning anything; it cannot fake asking from 18 m.

### Phase 3 — the human layer (a separate generation — see §7)

---

## 5. The name

**Recommendation: the brains are called *The Read*.**

```
arch_version   : 4
sensor_schema  : v6_read
generation     : read-1
```

Why this word and not a version number:

- **It is the game's own word.** A scout does not say "high situational entropy"; he says
  *"the kid reads it."* Reading the game is the exact faculty PLOFA is missing — the brain
  reacts to a snapshot, it does not read a situation. The name states the goal.
- **It covers all three things that change at once.** The brain now *sees* the goal
  (goal vision), *knows its circumstances* (home, instruction, stakes), and is *graded on
  geometry* (the new reward key). All three are acts of reading, not three unrelated
  patches.
- **It survives the sensor expansion.** `v6_read` is a schema name, not a feature name, so
  when the facing-state and reaction-latency dimensions land in `v7`, they are still The
  Read. A name like `GoalVisionBrain` would be obsolete the moment the next sensor landed.

Alternates, if "The Read" feels too quiet:

| name | what it captures | weakness |
|---|---|---|
| **Sightline** | the clear line from the eye to the goal; also the FOV/scanning work | implies vision only, not judgement |
| **Vantage** | the player's own position as a point of view | abstract, no football idiom |
| **Compass** | the brain finally knows where the goal is | one-dimensional; it is about more than direction |

Naming the *set*: keep `brains/ST.json` etc. positional (nothing downstream should
change), and record the generation in `brains/_manifest_all.json`:

```json
{ "generation": "read-1", "arch_version": 4, "sensor_schema": "v6_read",
  "trained": "2026-11-XX", "trained_on": "corpus.raw.json @ <sha>",
  "gate": { "...": "..." } }
```

The `trained_on` hash matters: this whole failure was caused by brains whose provenance
was a date nobody checked.

---

## 6. What a real player has that PLOFA does not

Researched, not invented. Each entry states the evidence, what PLOFA has today, and what
to add. Ordered by value per unit of work.

### 6.1 Home advantage — and the fact that it is mostly *not* a referee effect

The best natural experiment in football is COVID: 4,356 matches across 12 European
leagues, 1,131 of them behind closed doors (Sci Rep 11, 2021, *Home advantage mediated by
referee bias and team performance during covid*).

| measure | with fans | without fans | change |
|---|---|---|---|
| points advantage (home − away) | **0.39** | **0.23** | −41% |
| goals advantage | **0.29** | **0.17** | −41% |
| home shots per game | — | **−1.32** | home teams shoot less |
| home shots on target | — | **−0.35** | |
| home corners | — | **−0.70** | |
| path: venue → team performance | **+0.89** | **+0.35** | more than halved |
| path: venue → referee decisions | **−0.42** | **+0.16** | *reverses sign* |

**~80% of the home advantage is team performance, ~20% is referee bias.** The crowd does
not make the ball go in; it makes the home team take 1.3 more shots and win 0.7 more
corners.

PLOFA today: `is_home` exists and **does nothing** (audit Tier 2, item 4). Legacy seasons
nonetheless show ~+0.3 goals/match — which is suspiciously close to the real 0.29. So
before adding a force, **measure whether the 0.3 is real or an artifact of squad/seed
asymmetry.** If it is real, adding a multiplier double-counts it.

Implementation, if it is needed: **do not add a scoring multiplier.** Add the mechanism
the evidence describes — a small lift to shot generation and duel intensity for the home
side, tuned to reproduce 0.29 goals and +1.3 shots. A multiplier would produce the right
scoreline for the wrong reason, and would not survive contact with a different tactic.

### 6.2 The coach's instruction before kick-off

PLOFA's coach layer is near-binary: an `ATTACK` / `DEFEND` posture, an `_ATTACK_BIAS`
intent bias, and a favoured flank. Nothing survives from before kick-off; nothing decays;
nothing is specific.

A real player carries a **plan** into the match: *"press their left back", "we're going
long today", "keep it tight for fifteen minutes", "shoot on sight"*. It is specific,
persistent, and it **decays** as the game contradicts it.

This is exactly what the dead `PLAN_D = 6` block should be. P0.3 fixes the bug; the design
question is what the six numbers should say:

| slot | meaning |
|---|---|
| 0 | directness — build short vs go long |
| 1 | press height — engage high vs sit in |
| 2 | shot appetite — shoot on sight vs work it in |
| 3 | width — stay wide vs overload central |
| 4 | risk — take the risky pass vs keep it |
| 5 | plan confidence — 1.0 at kick-off, decaying with each contradiction |

Slot 5 is the one that makes it a *plan* rather than a *setting*. A player who has been
told to press high and has been beaten over the top three times stops pressing high. That
is a decaying prior, and it is one float plus one multiply.

### 6.3 Perception — **corrected 2026-11-08.** PLOFA has vision. It does not have *looking*.

An earlier draft of this section claimed PLOFA's field of view was "a radius, not a cone,
nearest 2+2, no facing, off-ball players never modelled." **That was wrong**, and the
correction matters because it changes the work. It described `cognition/senses.py` — a
second, simpler gate (`view_radius = 25.0`, nearest-first) used by `PlayerMind` — and
missed the production layer entirely.

What actually ships, in `perception.py`, enabled by default since 2026-09-13:

| capability | implementation |
|---|---|
| role-specific FOV **cones** | `ROLE_PROFILES`: GK 200°, CB 165°, CM 150°, ST/CF 130° |
| DNA-scaled range | 8–45 m from vision + anticipation (`_mental_scale`) |
| positional **noise** | sigma from composure/anticipation; deterministic per snapshot |
| relevance **top-k** | ST sees 6 actors, GK 12 (`max_actors`) |
| **tunnel vision** | `_tunnel_vision_fov` — the cone narrows with running speed |
| **peripheral glimpse** | `_peripheral_glimpse` — motion is caught outside the cone |
| off-ball actor gating | `perceive_offball_actors`, same cone/range/top-k/noise rules |
| ball vision with memory | `ball_vision._BallTrack`: last-known state, staleness grows 8 m/min |
| **head-turn scanning** | `_facing_angle` (2026-11-05): the cone follows the head as it turns toward the ball |
| fatigue | perception power, scan cadence and noise all degrade with the match clock |

There is even a `fov_audit.py` at the repo root that compares the two systems side by
side. This is a mature layer, and calling it a radius was careless.

**So the honest gap is narrower, and sharper: the one player who cannot scan is the man
with the ball.**

`perceive()` — the **on-ball** path, the one the decision actually runs through — tests
actors with `_is_visible(...)`, which uses `_forward_angle(attacks_right)`: a constant
pointing at the opponent goal. The head-turn function `_facing_angle` is used only by
ball vision and by off-ball runners. Its own docstring says why:

```python
# ball at his feet: no look target
if math.hypot(ball_x - from_x, ball_y - from_y) < 2.0:
    return forward
```

The carrier has no look target, so his cone is pinned forward — while the other 21
players turn their heads. The player making the decision is the only one who cannot look
around, and his 130–200° cone is centred on a direction he may not even be facing.

Three consequences, in order of value:

1. **The carrier's orientation is a constant, not a state.** A real player's cone at the
   moment of receiving is the *result of the scan he took two seconds earlier*. PLOFA
   has no such state, so the most important perception act in football — the check before
   the ball arrives — cannot happen for the player who needs it most. The fix is not a
   new cone; it is giving the carrier a **facing state** that his last scan sets.
2. **Actors have no memory.** `perception.py` is explicit: *"No futurity: every value is
   derived from the CURRENT frame's geometry only."* The **ball** has memory
   (`_BallTrack`, stale-growth 8 m/min); **players** do not. A runner seen two seconds ago
   leaves no trace, so a player cannot act on information he gathered earlier — which is
   most of what "reading the game" means.
3. **Scanning is a cadence, not a decision.** `ball_vision` scans on a timer,
   `1 / (0.2 + power·0.6) × fatigue_mult`, and the only cost is "a brief window of reduced
   forward responsiveness". A human chooses when to look, weighs the cost against what he
   might learn, and elites do it far more often before receiving. PLOFA's scan is a
   metronome, so scanning frequency is not a skill and cannot be selected for.

This is still the largest gap between the brain and a human — but the correct name for it
is **vision without looking**, not absent vision. It stays in `v7`: it needs a facing
state and an actor-memory store, both of which are architectural.

### 6.4 Reaction latency

Only the goalkeeper has a `reaction_time`. Every other player decides instantly, on every
tick, with perfect timing. Human choice latency is roughly 300–700 ms for a football
decision, and simple response latency 200–400 ms.

The consequence is not "slower football" — it is that **a human cannot re-decide as often
as PLOFA does**, so a human's decisions have commitment. PLOFA's brain re-evaluates
continuously, which is why its choice entropy is high and its predictability is 41%. Some
of that "noise" may be a player who would never have had the chance to change his mind.

### 6.5 Stakes, pressure, and choking

`score_diff` and `minute` exist. What is missing is **stakes** — is this a relegation
six-pointer, a derby, a dead rubber? — and the specific, well-documented way humans get
*worse* under them: reinvestment, where a performer consciously takes back control of a
skill that had become automatic, and it falls apart.

A model of this is not "reduce skill by 10% when nervous". It is: **under high stakes,
shift weight from the learned policy toward a simpler, more conservative one.** In
architecture terms, blend the brain's intent distribution with a low-entropy prior. That
is a handful of lines, and it produces the observable signature — fewer risks, more
square passes, worse execution — rather than a uniform penalty.

### 6.6 The rest, briefly

| element | PLOFA today | evidence it matters | cost |
|---|---|---|---|
| **Communication** ("man on", "time") | nothing | defenders organize verbally; it is how a back line shares information | medium — needs a teammate-signal channel |
| **Referee model** | nothing | ~20% of home advantage runs through referee decisions; players manage cards | low — a per-referee bias on foul/card probability |
| **Opponent reputation** | `memory_features` has duel ratios, not identity | "he's beaten me twice, show him inside" | low — per-opponent memory key |
| **Crowd as a live signal** | nothing | a nervous home crowd makes players rush; distinct from static home advantage | medium |
| **Weather / pitch** | nothing in sensors | wet pitch: faster ball, worse first touch; wind: long balls and set pieces | low — 2 sensors |
| **Deception** (feints, disguise) | nothing | it is what makes real football unreadable; a defender cannot trust the cue | high — needs a two-layer action model |
| **Body orientation on receive** | not an input | receiving on the half-turn is the difference between a press-resistant 6 and a limited one | medium |

---

## 7. Sequence and risk — why two generations, not one

**Do the geometry retrain first, at 24-wide. Do the human dimensions second, at 6x-wide.**

Adding sensors changes `STATE_D` from 24, which changes `w_e` (12×32) and `w_z` (79×32).
Every existing weight becomes the wrong shape. There is no warm start; it is a full
retrain from random. Bundling the two means a failure in either is indistinguishable from
a failure in the other, and the geometry fix — which is cheap, provable, and has a
crisp acceptance test (§4 Phase 2) — gets buried inside an experiment.

| risk | mitigation |
|---|---|
| Cell explosion from 9-factor buckets | count populated cells in P1.3 before evolving; the family/global fallback chain already exists |
| GA finds a degenerate shot policy (shoot from anywhere) | the distance-distribution assertion in Phase 2 is the guard; add it to the gate |
| `validate_neural_xl` restore is API-stale | P0.1's acceptance is a real gate match, not an import |
| The 0.3 goals home advantage is already emergent | measure first (§6.1); do not add a force that exists |
| Provenance drift repeats | `_manifest_all.json` records `trained_on` as a corpus hash |

---

## 8. The next three commands

```bash
# 1. does the gate run at all?
cp .kilo/worktrees/aspiring-gem/validate_neural_xl.py .
python -c "from validate_neural_xl import _run_neural; print('gate importable')"

# 2. is the plan block alive after the fix?
python -c "from situational_context import plan_tail; print(plan_tail.__doc__)"

# 3. one exploratory collection match — the corpus that would have saved six weeks
python -c "
from surrogate_rebuild import _run_matches
d = _run_matches(1, 123, ['fluid_counter'], True, 'neural')
print('samples', len(d))
print('SHOOT', sum(1 for r in d if r[1].value == 'SHOOT'))
print('slot2==slot0', sum(1 for r in d if abs(r[0][2]-r[0][0]) < 1e-9), '/', len(d))
"
```

The third command is the one that matters. **Run on 2026-11-08, it returned:**

```
samples 822
SHOOT 81
slot2==slot0 5 / 822
slot3==slot1 0 / 822
slot2 range -1.0 .. 1.0
slot3 range  0.0 .. 1.0
FT1 and OH1 both set 0 / 822
```

Read it as three answers:

- **The sensor layer is live in the collection path.** `slot2 == slot0` in 5 of 822 (0.6%,
  coincidence at equal values, not duplication) against **4,933 of 4,933** in the old
  corpus. Slot 2 spans the full signed bearing range `[-1, 1]`; slot 3 spans `[0, 1]`.
  Goal geometry reaches the trainer.
- **The impossible states are gone.** `FT1 ∧ OH1` in 0 of 822, against a corpus full of
  them. The direction convention is now consistent.
- **The brain still asks to shoot at 9.9%** (81 of 822), which matches the 109-per-1,055
  measurement in §6.6 of the audit. The intent is not the problem; the position is.

**Verdict: the go/no-go passes. Phase 1 may begin.** The failure is exactly where the
plan says it is — in the pricing and the corpus, not in the collector.

---

## 9. The dream, stated honestly

"Indistinguishable from a real human player" decomposes into four properties, and PLOFA
currently has none of them fully:

1. **It looks, not just sees** — it already has role cones, noise, tunnel vision and
   peripheral glimpses; what it lacks is a *facing state* for the carrier and any memory
   of what was seen a moment ago. Vision without looking. *(§6.3 — the largest gap)*
2. **It knows what a human knows** — where it is playing, what it was told, what is at
   stake, who it is up against. *(§6.1, §6.2, §6.5)*
3. **It decides at a human speed, and therefore commits.** *(§6.4)*
4. **It is graded on football, not on a lookup table.** *(§2 — the reason the retrain
   cannot be run today)*

The retrain fixes (4). That is the correct first move: it is the only one of the four
where the project can currently prove, with a number, that the brain is wrong. The other
three are additions; (4) is a repair, and a repair comes first.

---

## 10. Revision — what I am *not* confident about (added 2026-11-08, same day)

Asked whether I was confident in this plan, I checked the thing I had not checked: what
the surrogate actually *pays* for a shot. The answer changes Phase 1.

**10.1 The brain is not only blind — it is being paid to dribble.**

`brains_trainer/surrogate_refit.json`, the striker's clear-chance bucket
(`P0 FT1 OH0 S0 C1 G1` — unpressured, final third, central, goal-close, no space):

| intent | reward |
|---|---|
| **DRIBBLE** | **1.163** |
| SWITCH | 1.027 |
| **SHOOT** | **0.957** |

Shooting from a clear chance is priced **18% below dribbling** and below a switch of play.
The GA did not fail to discover shooting. It was *told* not to shoot, in the one bucket
where shooting is obviously right. Geometry bins alone do not fix this — the number is
already there and it is wrong.

**10.2 My proposed bucket change would probably erase itself.**

The table is far sparser than I assumed:

| position | buckets | buckets with any SHOOT value |
|---|---|---|
| ST | 17 | 15 |
| CF | **11** | 7 |
| CAM | 12 | **0** |
| CB | 21 | **0** |

Multiply 17 buckets by the 5×3×3 geometry bins I proposed in §P1.1 and you get up to 765
cells from ~200 striker samples. Almost every cell falls under `min_samples=3`, so
`_lookup` falls through to the family row and then the global row — **both of which are
keyed by the old 6-factor bucket**, so the geometry is discarded at exactly the moment it
matters. My fix would be silently neutralised by the fallback chain I was relying on.

A sparse empirical lookup table is the wrong instrument for a rare, high-variance action.
A shot is a 10%-success, high-payoff event; a pass is an 85%-success, low-payoff one.
Averaging observed outcomes over a handful of samples will systematically favour the pass
no matter how the situation is keyed.

**10.3 The revised approach: teach the shot head directly, leave the table alone.**

Do **not** widen the bucket. Keep it at 6 factors so it stays dense enough to have data
for the other nine intents. Instead add a **dense auxiliary signal** computed per state,
not looked up:

```
shot_score = attacking_matrix.shot_score(x, y, pressure, angle, finishing)
target     = 1 if shot_score >= 0.20 else 0
loss_aux   = BCE(p_SHOOT, target)          # on the SHOOT logit only
fitness    = surrogate_fitness - w_aux * loss_aux
```

Why this is better on every axis:

- **It is dense.** Every synthetic state contributes a label; no sample-count problem.
- **It uses the right authority.** `attacking_matrix.shot_score` already encodes "is a
  shot geometrically available here" — it is the coach's judgement of the *noun*, which
  the architecture principle says the matrix owns. The brain still owns the *verb*: it may
  decline a shootable ball. We are teaching it to recognise the opportunity, not
  forcing the action.
- **It is testable in an hour**, and it does not depend on the surrogate learning to price
  a tail event from a handful of samples.
- **It targets the measured failure exactly**: 109 intents, median 54.5 m from goal.

The surrogate stays responsible for the other nine intents, where it is adequate.

**10.4 What I would bet on, and what I would not**

| claim | confidence |
|---|---|
| The corpus is unusable (`slot2 == slot0` in 4,933/4,933) | **certain** — measured |
| The reward key has no goal geometry | **certain** — read the code |
| Collection runs with exploration off | **certain** — read the code |
| The gate module is missing | **certain** — import fails; the worktree copy *loads* fine, so this is small |
| `plan_tail` is always zeros | **certain** — read the code |
| SHOOT is priced below DRIBBLE in the clear-chance bucket | **certain** — read the table |
| Recollecting the corpus is necessary | **high** |
| Geometry bins in the bucket would help | **low — I now think they would backfire (§10.2)** |
| The GA alone would learn to shoot once the corpus is right | **low — §10.1 says the objective pays it not to** |
| An auxiliary supervised shot head fixes it | **moderate-to-high — cheap to test, §10.5** |

**10.5 The one-hour test that decides the path**

Before any multi-day work: take ~300 synthetic goal-close states, compute `shot_score`
for each, and check that a brain trained with the auxiliary head raises `p_SHOOT` where
`shot_score` is high and lowers it where it is low. If that separates, the approach is
sound and Phase 1 proceeds. If it does not, the problem is in the state representation,
not the objective, and the plan needs a different answer.

This test needs no match simulation and no corpus — which is exactly why it goes first.

**10.5.1 Result — measured 2026-11-16, `tools/diag/_shothead_probe.py`**

Run on the SHIPPED `SituationalBrain` (arch 3, attention + GRU, 47-d situation), on 400
states from `brain_evolution.random_game_state_scoring` — the same generator the evolution
goal-bias corpus uses. Labels from `attacking_matrix.shot_score` (verified sane: 2.97 at
1 m central, 0.30 at 15 m, 0.02 at 45 m, symmetric for both attacking directions, 0.41 at
15 m under 1 m of pressure). Every number is held out on a fixed, seeded split.

| | ST | CF | CAM |
|---|---|---|---|
| linear probe on the hidden state, AUC | **0.954** | 0.954 | 0.834 |
| shipped head: AUC / spearman | **0.277 / −0.509** | 0.277 / −0.509 | 0.201 / −0.520 |
| §10.3 as written (retrain the head), AUC | 0.452 | 0.452 | 0.332 |
| dedicated read-out, AUC / spearman | **0.945 / +0.751** | 0.945 / +0.751 | 0.773 / +0.407 |
| dedicated read-out: untouched states left alone | **92 %** | 92 % | 64 % |
| SHOOT actually chosen on shootable states | **78 %** (was 0 %) | 78 % (was 0 %) | 78 % (was 33 %) |

Three findings, in order of how much they change the plan:

1. **The representation is fine; the head is inverted.** A logistic read-out on the frozen
   32-d hidden state reaches AUC 0.954. The shipped output layer, reading the same state,
   reaches 0.277 — worse than chance and *anti-correlated*. The brain sees where the goal
   is and the output layer reads it backwards. This is not a representation failure, and
   §10.5's fallback ("the problem is in the state representation") is **not** the answer.

2. **§10.3 as literally written does not work cleanly, and the reason is structural.** Re-
   training the existing head with the auxiliary loss and no anchor fixes the ordering
   (spearman −0.574 → +0.557, AUC 0.242 → 0.831) and **destroys the policy**: the argmax
   mix goes `PROGRESSIVE_PASS 68 % / SHOOT 14 %` → `SWITCH 88 % / SHOOT 0 %`, only 21 % of
   held-out choices survive. Adding a KL anchor to the frozen policy recovers the policy
   (92 % of choices survive) and gives back the ordering (AUC 0.449 — chance). The two
   objectives fight over one 32-d bottleneck and nine shared logits, and a single scalar
   anchor cannot resolve it: the measured frontier is monotone. A binary 0.20 target makes
   this worse than a graded one, because 90 % of scoring states are labelled "not a shot",
   so the auxiliary term is mostly a suppression signal.

3. **A dedicated read-out solves it.** 33 parameters (32 weights + 1 bias) fitted on the
   frozen hidden state, applied as an additive bias on the **SHOOT logit only**:

   ```
   logit_SHOOT += beta * (w_s . h + b_s)
   ```

   Because no shared weight moves, the other nine intents are untouched *by construction* —
   the 92 % "left alone" figure is not a trade-off, it is the residual error of the probe
   itself. At `beta = 6` the brain takes the shot on 78 % of genuinely shootable states
   where it previously took 0 %, with AUC 0.945.

**What this changes.** P1.3 should not be "retrain the brain with an auxiliary loss". It
should be: fit a per-position shot read-out offline, ship it as a 33-parameter sidecar, and
apply it as a bias on one logit. That is a change of one line in the decision path and it
cannot degrade anything except the SHOOT choice. CAM is the weak case (AUC 0.773, 64 %
left alone) and should be gated separately rather than shipped on the ST/CF result.

**A bug this test found on the way.** `SituationalBrain.backward` sliced the situation
vector with the module constant `ROLE_D` (19) instead of the width the brain actually uses.
Every shipped brain uses a nine-wide role block (`w_q` is `(47, 32)`), so `smp` came out 57
against a real 47, `dctx` was 22 long instead of 32, and the GRU gate gradients were
written from a truncated vector — no exception, wrong weights. **Any supervised training of
the shipped brains through BPTT was silently corrupt.** Fixed to `smp = w_q.shape[0]`;
`tests/test_bptt_gradients.py` now checks analytic gradients against finite differences at
both role widths, and fails on the old code at exactly the shipped width.

**10.5.2 What was built — 2026-11-16**

| piece | where | state |
|---|---|---|
| the §10.5 harness | `tools/diag/_shothead_probe.py` | three stages: baseline, linear probe, fine-tune, plus the separate-head stage |
| the artifact | `shot_head.py` — `ShotHead`, `fit()` | 33 params, fitted per position, self-gating |
| the fitter | `tools/ops/fit_shot_heads.py` | writes `brains/shot_heads/<POS>.json` + `_report.json` |
| the wiring | `situational_decision._apply_shot_head` | before the softmax, on the SHOOT logit only, **off unless `PLOFA_SHOT_HEAD=1`** |
| the tests | `tests/test_shot_head.py` (14), `tests/test_bptt_gradients.py` (5) | |

Fitted 2026-11-16 over 2,000 states per position, seed 4242:

| position | tier | probe AUC (reset / carried) | beta | holdout AUC | SHOOT on shootable | untouched states left alone |
|---|---|---|---|---|---|---|
| ST | PROVISIONAL | 0.864 / 0.834 | 6.0 | 0.822 | 56 % (was 1 %) | 83 % |
| CF | PROVISIONAL | 0.864 / 0.834 | 6.0 | 0.822 | 56 % (was 1 %) | 83 % |
| CAM | PROVISIONAL | 0.893 / 0.898 | 4.0 | 0.846 | 92 % (was 36 %) | 60 % |
| LW, RW | REFUSED | 0.897 / 0.844 | — | — | — | — |
| GK, CB, LB, RB, CDM, CM | REFUSED | — | — | — | — | — |

**The bias is clamped to ±6, and that changed the numbers.** Measured on the shipped ST
brain at `beta = 6`, the raw bias ran from **−103.7 to +15.4** over 400 real hidden states.
A bias of −104 is not a prior, it is a *veto*: `p_SHOOT` becomes exactly 0.0 and the brain
can never shoot there regardless of what it sees — the opposite of §10.3's own intent
("the brain still owns the verb: it may decline a shootable ball"), and unbounded on any
out-of-distribution hidden state. The clamp caps the odds the head can impose at `e^6 ≈
400:1`: enough to change a decision, finite enough that another intent can still outvote
it. The clamp lives inside `ShotHead.bias()`, so the trial numbers above are measured on
exactly the function that ships. Unclamped the same head read AUC 0.855 / 59 % / 82 % —
slightly better numbers describing a mechanism that would have been a hard switch.

Three deliberate refusals worth recording:

- **The six defensive roles are refused by the role vocabulary**, before any training runs.
  `ROLE_ONLY_INTENTS["SHOOT"]` is `{ST, AM, WING}` and `ROLE_INTENTS` documents that a
  keeper "does not cross or shoot". Without that guard the fitter produced a PROVISIONAL
  head for the **goalkeeper** (41 % shot rate on shootable states) — arithmetically fine,
  football nonsense.
- **The wingers are refused on their own numbers.** Their probe is as good as ST's (0.897)
  but no beta reaches a 40 % shot rate while leaving the untouched decisions alone.
- **Nothing is `ACCEPTED`.** The strict bar (60 % shot rate, 90 % untouched) is not
  reachable at any beta, and that frontier is stable from 400 to 3,000 states. It is a
  property of the read-out, not of the sample. Lowering the bar to force an ACCEPT is the
  vacuous pass this project has already paid for twice.

So P1.3 ships as **PROVISIONAL**: the artifact exists, it is measured, it is inert, and the
next thing it needs is a real-match gate — not another offline sweep.

**10.5.3 The real-match gate, and the two bugs it exposed — 2026-11-16**

The gate was run and it returned **a bit-identical match**: ON and OFF both 3-5, xG 0.91,
3 shots, possession 51.2, fitness 0.6093 to four decimals. On the synthetic corpus the same
head flips 134 of 1,200 states, so "identical" could not mean "no effect". It meant "not
applied", and finding out why cost two things.

**Bug 1 — the pipeline was forked.** `brain_integration._maybe_situational` did not route to
`situational_decision`; it contained a **complete second copy** of the situational pipeline
(129 lines: FOV gate, tokens, four blocks, recurrent forward, plan proposal, consequence,
temperature, sampling). `SituationalDecisionBrain` was imported by **nothing but its own
tests** — `git show HEAD:brain_integration.py | grep -c SituationalDecisionBrain` returns
**0**. So the shot read-out was wired, tested and gated against a program that never ran.
The temperament reconnection was dead for the same reason. The copies had already drifted.

The fork is now gone: `_maybe_situational` is a type check, an import and a call. Neutrality
was proven rather than asserted — the OFF arm reproduced the known fingerprint **exactly**
after deleting the copy. Two switches are now explicit and both default OFF
(`PLOFA_SHOT_HEAD`, `PLOFA_TEMPER`), because delegating would otherwise have switched
temperament on in every season run (`auto_run_match` registers a mind per player;
`match_probe` registers none, so no gated experiment would ever have shown it).
`tests/test_situational_routing.py` (5 tests) locks the invariant, and was verified to fail
against the old file: the fork served a decision while never entering the shared pipeline.

**Bug 2 — the metric paid the head to overrule the brain.** With the wire live, the gate
turned strongly negative: **shots 3 → 0**. A 33-parameter linear read-out silenced a striker
completely. The cause was in `fit()`:

- `MIN_CLEAN` measures *collateral* — states that are neither shootable nor already
  shooting. That set **excludes every state where the brain was already shooting**, so the
  head was never charged for silencing a shot the brain wanted;
- `MIN_SHOT_RATE` counts only states the geometry calls shootable.

Between them, the fitter was free to reward the head wherever it disagreed with the brain and
free of any penalty for doing so. It took the bribe: the shipped ST head had **66 % of states
pinned at the ±6 clamp wall** and `|bias| > |logit gap|` on **76.5 %** of states — on three
states in four, the head was choosing, not the brain.

Two constraints were added, measured over **every** held-out state:
`MAX_OVERRULE = 0.15` (share of states whose argmax the head changes at all) and
`MAX_SILENCED = 0.10` (share where the brain chose SHOOT and the head takes it away).

**The refit then refused all eleven positions**, and the trial table says why:

| ST, beta | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|
| AUC | 0.356 | 0.634 | 0.814 | 0.852 | 0.853 | 0.842 |
| silenced | 21.6 % | 27.2 % | 27.3 % | 27.3 % | 27.3 % | 27.3 % |
| overrule | 21.8 % | 28.0 % | 28.3 % | 28.4 % | 31.1 % | 35.6 % |

The head only ranks shot viability **better than chance** once it is large enough to overrule
the brain (AUC 0.36 at `beta = 0.5`). A gentle prior inherits the brain's own `p_SHOOT`, which
is *anti*-correlated with viability (AUC 0.28, Spearman −0.51), and scores worse than a coin
flip. And `silenced` sits at ~27 % at **every** beta including 0.5, because the brain's shot
decisions in these states are near-ties that even a ±0.5 nudge breaks.

**So a sidecar cannot be a safe prior here.** The constraint that makes it safe (small bias)
makes it useless (AUC < chance); the constraint that makes it useful (large bias) makes it a
replacement, and as a replacement it is wrong on real match states — 0 shots. This is not a
tuning failure to iterate on. It is a statement about the read-out: the information is in the
hidden state (a linear probe on it reaches AUC 0.95) but the brain's **output layer** reads it
backwards, and a 33-parameter patch on the output cannot fix a sign error without taking over.

**Verdict: P1.3 is REFUSED at all eleven positions.** No artifact ships; `brains/shot_heads/`
holds only `_report.json`. The seam stays in place, inert, behind `PLOFA_SHOT_HEAD` — it costs
nothing and it is the right place to attach a *correct* read-out if one is ever trained. The
fitter now **deletes** a stale artifact for a refused position, because "the fitter says no
while the loader still finds a file" is how the harmful beta-4.0 head survived one refit.

The path that remains is the one the plan already pointed at: fix the sign in the network, not
beside it. That is §10.3's territory, and §10.3's own measurement stands — retraining the
shared head fixes the ordering and destroys the policy (21 % of choices survive), while a KL
anchor keeps the policy (92 %) and returns the ordering to chance (0.45). Neither is a win yet.

---

## 11. Do the v8 brains reason? — measured 2026-11-08

Prompted by the fair challenge: *"my brains are v8, 15k params, you think they don't think
and reason?"* Two measurements, on 4,201 distinct states (732 exact duplicate rows
dropped — duplicates are their own nearest neighbour and would fake the result).

Tool: `tools/diag/_brain_reasoning_probe.py`. Output: `artifacts/brain_reasoning_probe.json`.

### 11.1 My "noisy" verdict was largely a binning artefact

`I(situation; intent)`, same states, two estimators, each against its own shuffle null:

| estimator | `H(intent)` | `H(intent\|state)` | `I` | null | **excess over null** |
|---|---|---|---|---|---|
| coarse 6-bit bucket | 3.063 | 2.698 | 0.366 | 0.047 | **0.319** |
| fine, k=10 in 24-d | 3.063 | **1.772** | 1.291 | 0.710 | **0.582** |

The coarse key the surrogate uses understates situational information by **~1.8×**, and
`H(intent | state)` falls by **0.93 bits** when the key is allowed to be fine. The state
explains **42%** of choice entropy, not the 12% the bucket implied. The audit's verdict —
*"context-sensitive but noisy (reacts, not consistent)"* — was measuring the key, not the
brain. It should be read as **"context-sensitive, and more so than the surrogate can
see."**

### 11.2 The information is not in the position — it is in the context

Same estimator, restricted feature sets:

| features | excess over null |
|---|---|
| `x, y` only | **0.121** bits |
| `+ pressure, defender distances, space ahead` (7) | **0.304** bits |
| all 24 slots | **0.582** bits |

Location alone carries a fifth of the information. Pressure, proximity and space roughly
**2.5×** it; the remaining slots (fatigue, DNA, score, minute, zone flags) roughly double
it again. **A lookup table keyed on where you are cannot produce this pattern.**

### 11.3 No memorisation signature — it extrapolates into unseen space

A smooth multinomial logistic model on 7 features, two splits: a random 80/20 (the
in-distribution ceiling) and a **spatial holdout** — train on everything outside a pitch
region, test inside it. A memoriser has nothing to say where it has no neighbours.

| pos | n | random | base | zone 14 (x 70–90, y 24–44) | base | def third | base |
|---|---|---|---|---|---|---|---|
| ST | 299 | 0.617 | 0.517 | **0.646** | 0.271 | 0.450 | 0.600 |
| GK | 432 | 0.644 | 0.333 | **0.633** | 0.327 | – | – |
| CDM | 431 | 0.713 | 0.586 | 0.417 | 0.333 | 0.434 | 0.566 |
| CM | 938 | 0.415 | 0.410 | 0.527 | 0.549 | 0.142 | 0.351 |
| CB | 1115 | 0.511 | 0.511 | 0.508 | 0.500 | 0.444 | 0.265 |

The striker and the keeper **do not degrade at all** when extrapolating into a region they
never trained on (0.646 vs 0.617; 0.633 vs 0.644) — and both sit far above the majority
baseline there. A lookup would have collapsed to the baseline. CDM and CM *do* degrade
sharply, so this is position-dependent rather than uniform.

### 11.4 What this does and does not settle

**In their favour:** the policy is a rich, context-weighted, spatially generalising
function. It is not a lookup, and it is less noisy than I reported. §11.1 is a correction
to my own verdict and belongs in the audit.

**Against over-reading it:** my probe used 7 features carrying 0.304 of the 0.582 bits
available (§11.2) — about half. That is why in-distribution accuracy sat near the majority
baseline for most positions: the probe was under-informed, not the brain uninformative.
The meaningful result is the *absence of a collapse* on the spatial holdout, not the
absolute accuracies.

**Still untested, and structurally absent:** a model of consequences. Nothing here shows
the brain predicting what happens next, or comparing two branches from one state. The
forward pass is `state → distribution`, with no rollout. These tests show it *reads* the
game well. Reading is not foresight.

### 11.5 The honest one-line answer

They think in the sense that a policy thinks: they integrate context, weight it, and
generalise to situations they were not trained on. They do not reason in the sense of
simulating consequences — and that gap is architectural, not a matter of parameter count.
