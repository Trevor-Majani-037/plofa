# PRE-FLIGHT: what must be true before recollecting the corpus

**2026-11-16.** Written because the honest answer to "is the 80-minute run a surebet?" was
**no**, and because this project has already paid twice for discovering a defect *after*
the expensive step.

The rule this document exists to enforce:

> A long run may only start when every gate below is **PASS**. A gate that is PENDING is
> not a pass. A gate that cannot be checked without the long run is not a gate — it must be
> moved earlier or the run must be made cheap enough to falsify first.

## Why this matters now

The recollection is the precondition for making the objective state-dependent, which is the
precondition for the foresight plan. If the corpus is wrong, everything downstream of it is
wrong, and the failure is silent — a brain that is subtly worse for half of every match is
not something you notice, it is something you discover months later or never.

## The gates

| # | gate | what it proves | status |
|---|---|---|---|
| 1 | **Argument decoding** | the collector records the real `game_state` and `attacks_right`, not its neighbour | **PASS** — `_decode_situational_args`, 14 tests; verified on a real match (scoreline `+0.20` then `+0.40`) |
| 2 | **Score mapping** | every engine state maps to the right goal difference | **PASS** — `_score_diff_of` table, all 7 states tested; CRUISE was 0, CHASE was inverted |
| 3 | **Mirror symmetry** | an away-team row is the same kind of object as a home-team row | **PASS** — sensors symmetric to 1e-9 except ball_x/ball_y; role block 1.6e-15; tokens 1.1e-15 as a set. 22 tests |
| 4 | **Reward knows the scoreline** | the labels are not scoreline-blind | **PASS** — `_stakes_pair` wired into both reward sites, recorded as `stakes_mode`; see §"The multiplier had to become asymmetric" |
| 5 | **Corpus carries varied scorelines** | there is data for the table to learn from | **FAIL before the gate-8 fix, PASS after** — pre-fix control: range `+0.000..+0.400`, **0.0 % of steps behind for either team** |
| 6 | **The table can express urgency** | a scoreline effect is estimated at all | **REDESIGNED** — the partition was measured and rejected; a pooled layer replaced it |
| 7 | **Urgency actually changes a preference** | step 2 works at all | **SPLIT** into design (must pass on a pilot) and effect (needs the full corpus) |
| 8 | **The scoreline is in the player's own frame** | "I am behind" means behind for HIM | **PASS** — `_player_score_diff`, 28 tests, `attacks_right` is the team identity |

### The multiplier had to become asymmetric

`stakes()` scaled the WHOLE return by one number. That cannot work, and the reason
is arithmetic rather than tuning:

> `k·a > k·b` exactly when `a > b`, for any `k > 0`.

A uniform multiplier makes a decision more or less *confident*. It can never make
it *different*. So a corpus labelled with it — however large — distils to the same
indifferent policy, and `expected_success` keeps the same argmax in every cell.
This is `tests/test_urgency_and_stakes.py::test_a_uniform_multiplier_cannot_change_a_preference`,
pinned as a property so the scalar is never re-introduced with expectations it
cannot meet.

The fix is `stakes_pair(score_diff, minute) -> (gain, loss)`, because win
probability is **convex when you are behind and concave when you are ahead**:

| band | gain | loss | football reading |
|---|---|---|---|
| 2+ down | 1 + 0.25·late | 1 − 0.55·late | desperate: maximum risk |
| 1 down | 1 + 0.60·late | 1 − 0.35·late | chasing: a goal rescues it |
| level | 1 + 0.35·late | 1 + 0.20·late | one goal decides either way |
| 1 up | 1 − 0.25·late | 1 + 0.45·late | protecting: turnovers kill |
| 2+ up | 1 − 0.35·late | 1 + 0.20·late | cushion: the game is won |

At 88' a team two down is told a turnover is worth **0.49** of normal and a goal
**1.23**; a team one up is told the opposite (**1.42** and **0.77**). That
asymmetry is what lets a long shot lose to a square ball when you are protecting
a lead and win when you are chasing one —
`test_a_differential_multiplier_can_change_a_preference` walks through those
numbers. `stakes()` is kept, unchanged, as the published scalar for reporting.

### Why the urgency partition was rejected

The plan was to append `K{behind,level,ahead}T{early,mid,late}` to the bucket
key. It was measured first:

| table | rich samples | rich cells | per cell | cells with ONE sample |
|---|---|---|---|---|
| shipped `surrogate_refit.json` | 15,426 | 2,685 | 5.75 | **40.5 %** |
| 3-match pilot | 1,658 | 746 | 2.22 | 59.9 % |

A further 9-way split takes the shipped table to **0.64 samples per cell**. The
fit's `min_samples=3` shrink cannot rescue a cell with one observation, and a
reward table fitted on one sample per cell is noise — which is *worse* than
indifference, because it is invisible: the player looks erratic rather than
uninterested.

So the effect is estimated as a **pooled additive shift** instead of a finer
partition: one number per (position, scoreline band, intent), taken as the mean
residual of that intent's samples against the **coarse parent cell** they fell
in, shrunk toward zero with 5 pseudo-observations. The parent is used rather than
the rich cell on purpose — the rich cell aggregates over bands by construction,
so residualising against it makes the band contrast depend on how samples happen
to split between geometry sub-cells, and a cell with a single observation has a
residual of identically zero.

This is the same hierarchical instinct as `A·M·D`, one level up: **do not spend
resolution the data cannot pay for.** Absent evidence the shift is 0.0, so an old
table loads and behaves exactly as before (verified: 2,000 comparisons, 0
differences).

### Gate 8: where team identity enters the decision path

Three candidates were on the table; the one taken is the smallest that is
correct. `attacks_right` is already in the decision path and **is** team
identity — `set_squad` sets it to `team_name == home_team`, and `MatchState`
never swaps ends. So the frame is a sign flip in one place,
`brain_integration._player_score_diff`, used by the live path and by the
collector so the sensors, the record and the stakes multiplier cannot disagree.

It refuses a non-`bool` rather than coercing: a truthy `GameState` enum is
precisely the shape of the defect that made the shipped corpus 100 %
attacking-right, and a lenient check would silently skip the flip.

**Known blast radius, stated rather than discovered later:** the consequence
critic reads the same 24-d block, so its slot-18 input was trained on the
home-frame value. The critic is already slated for a retrain onto the v5 schema;
until then it is reading a slot whose away-half has changed sign.


## What the pre-flight found (and would otherwise have shipped)

Six defects, all live, five of them invisible from the home half alone:

1. **`_goal_geometry` branch cut.** `abs(a_top - a_bottom)` without wrapping. For the away
   team both post angles sit near ±π, so the mouth computed as ~5.92 rad instead of 0.36 and
   clamped to 1.0. **An away striker 20 m out dead central read the goal as penalty-spot wide
   (1.000) instead of 0.563** — and the surrogate's `A·M·D` bucket consumed that value
   directly, saturating `mouth_bin` at 2 for every central away-team position.
2. **`situation_tokens` lateral offset.** `dx` was flipped for the away team, `dy` was not,
   so the two teams saw opposite sign conventions for the same relative geometry.
3. **`situation_tokens` SPACE feature.** The entity list holds tuples, and it was passed
   straight into `_nearest_opponent_gap`, which calls `_pos_of` — a tuple has no `.name`, so
   it returned the `(0, 0)` fallback. "Gap to the nearest marker" was really **distance to
   the pitch's corner flag**. A 2 m marker read as 22 m of space. It survived months because
   the origin is a fixed point: on the home half it correlates with attacking progress, and
   the corpus was 100 % attacking-right, so nothing ever asked it to be symmetric.
4. **The scoreline was home-frame for every player** (gate 8). An away side 2-0 up read
   `−2`. Measured on the pre-fix pilot: **range `+0.000 .. +0.400`, and not one step in
   1,250 where either team was behind** — even in a match that finished 3-0.
5. **A uniform stakes multiplier cannot change a preference** (see above). The reward work
   of 2026-11-16 was arithmetically incapable of producing the behaviour it was for.
6. **A vacuous test.** `test_the_reward_scales_with_the_multiplier` set `type = "PASS"` on
   its fixture, but the reward reads `ev.event_type`, so both sides of its assertion were
   `0.0` and `0.0 * 2 == 0.0` passed. It was blind to exactly the failure it existed to
   catch. Fixed, and it now asserts the base is non-zero first.

## What the pre-flight has NOT proved

- **The effect size.** A 3-match pilot shrinks every urgency shift to ~20 % of its measured
  value, so gate 7's *effect* number is expected to be small there and meaningful only on
  the full corpus. Only the *design* half of gate 7 is binding on a pilot.
- **The distillation.** The chain is verified link by link — differential reward → labels →
  fitted table → urgency layer → `expected_success` → `intent_targets` argmax → policy, and
  `state[18]` is an input — but a distilled net has not yet been shown to vary its choice
  with the scoreline. That is the one remaining step, and it is cheap to test on one
  position before the full run.
- **Whether 27 matches is enough.** The pilot's density numbers say whether the answer
  scales.

## The two probe errors worth remembering

A symmetry probe fails first on its own harness. Both of these were caught before reporting:

- mirrored at 100 instead of **105** — invented an `is_final_third` asymmetry that does not
  exist (`x > 70` and `x < 35`, and 105 − 70 = 35 exactly);
- built each scene separately with the same `rng`, so the halves got different **role labels**
  (`_role_coords_to_players` draws them) — reported 13 of 24 slots as asymmetric, mostly
  noise. Rows must not be sorted and paired either; compare each column sorted.

## What cannot be known before the run

Being explicit, because this is the part that turns into an apology:

- **Whether the effect is large enough to change behaviour at full corpus size.** A pilot
  shrinks every shift to ~20 % of its measured value, so the *size* of the effect is only
  knowable after the run. The *existence* of a differentiated effect is not.
- **Whether the distilled policy carries it.** Every link is verified, but the last one — a
  net whose argmax moves when slot 18 moves — has to be shown, not assumed. Gate 9.
- **Whether 27 matches is enough.** The pilot's density numbers say whether the answer scales:
  at 5.75 samples per rich cell in the shipped table and 2.22 in the pilot, a 9× corpus
  reproduces the shipped density and does not improve on it.

## Decision rule

1. ~~Add the scoreline dimension to the bucket key, as a ladder~~ — **measured and rejected**;
   a pooled layer replaced it (see above). The ladder idiom is still the right one for
   *geometry*, which the data does pay for.
2. Run a **3-match pilot** (~16 min) and check gates 5, 6 and 7-design on it.
3. Run a **one-position distillation** on the pilot surrogate and check that the resulting
   net's argmax moves with slot 18 (gate 9). This is the last link and the cheapest one to
   break.
4. Only if all of those pass, run the full collection. A failure at this stage is a failure
   of the design, not of the sample size, and more matches will not fix it.

Gate 7 is reported in two halves for this reason, and the distinction is the whole point of
this document:

- **design** — are the shifts different across intents, and are any decisions close enough
  to a tie for them to matter? This must pass on the pilot. A layer whose shift is the same
  for every intent can never flip anything, at any sample size.
- **effect** — how many decisions actually flip. On a pilot this is expected to be small and
  is *reported*, not gated.

