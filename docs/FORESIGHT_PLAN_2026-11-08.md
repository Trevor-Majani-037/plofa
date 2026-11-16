# Seeing ahead — foresight for the v8 brains

**Date:** 2026-11-08
**Context:** the brains recur, attend, plan and generalise to unseen situations
(`RETRAIN_PLAN_2026-11-08.md` §11). They cannot picture a future. This is how to give
them one.

---

## 0. The short answer

**You already have instinct. You do not have imagination.**

`brains_trainer/critic_v5.json` is a Q-function: `Q(state, intent) → expected payoff`,
TD-trained at γ = 0.97, and it is **on by default** — `consequence_decision.py` blends it
into every decision at strength 0.3:

```python
q, advantage = critic.classify(sensors, position, probs=p)   # advantage = q − p·q
logits = log(p) + 0.3 * advantage
```

That is a real form of looking ahead: *"from states like this, this action has paid off
before."* It is what a player means by instinct.

What it is **not** is imagination: *"if I do this, the world looks like that, and that is
good or bad."* There is no transition model anywhere in the project — nothing computes
`s' = f(s, a)`. The critic scores actions against an average of past outcomes, not against
a simulated future. That is the whole gap.

So the work is not "add foresight to a brain that has none". It is **upgrade instinct into
imagination**, on top of a critic that already exists.

---

## 1. What exists, measured

| piece | state | evidence |
|---|---|---|
| Q-critic | **exists, on by default** | `value_critic`, arch `[35,32,32,10]`, γ=0.97, blend 0.3 |
| critic input | **24 sensors + position one-hot** | arch input 35 |
| critic schema | **`v1_24d` — blind to goal geometry** | `meta.sensor_schema` |
| critic date | **2026-09-20** — six weeks before goal vision | file mtime |
| critic sees tokens / GRU state | **no** | input is 35 floats, nothing else |
| transition model | **absent** | no `f(s,a)` anywhere |
| rollout / search | **absent** | forward pass is `state → distribution` |
| plan heads | 4 phases, team-level, hysteresis | `PLAN_OUT_D = 4` |

Two things follow immediately, and both are cheap:

1. **The instinct is stale and half-blind.** A `v1_24d` critic trained on 2026-09-20 cannot
   see the goal. Every decision today is nudged by a critic that is guessing.
2. **The instinct is starved.** It sees 24 floats and a position. It cannot see the entity
   tokens, the attention weights, or the GRU state — the three things that make the brain
   good at reading a situation (§11). The critic is judging with less information than the
   player it is advising.

### 1.1 MEASURED 2026-11-13 — it is not half-blind, it is dead

The table above understates the problem, and the measurement changes what Level 0 has to
prove. Over the 12,297-decision corpus collected after the shot-credit fix:

| measurement | value | what it means |
|---|---|---|
| first-layer units that never activate | **30 of 32** | the layer is dark |
| second-layer units that never activate | **31 of 32** | so is the next one |
| Q range across every real state | **0.3115 → 0.4046** | 0.0123 inside a 1.30 band |
| share of Q variance explained by intent alone | **71.5%** | mostly a lookup table |
| holdout rank correlation, critic | **+0.0595** | |
| holdout rank correlation, 10-number intent prior | **+0.0905** | **the critic loses to a table** |

So the "consequence correction" blended at 0.3 into every decision since 2026-09-20 was a
near-constant vector: an intent prior wearing a neural network. It nudged every decision
the same way — toward `SHOOT`, away from `THROUGH_BALL` and `CROSS` — wherever on the
pitch it happened. That is not judgement, and it is not state-conditioned.

**Root cause, reproduced.** The TD two-stage recipe at `lr=1e-2` kills the ReLU stack: the
Adam steps are large enough to push every unit's pre-activation negative, after which the
gradient through it is exactly zero forever. Re-running the exact v5 recipe reproduces it
(28/32 and 30/32 dead). The same recipe at `lr=5e-3` keeps 32/32 alive and roughly doubles
holdout rank correlation. Default changed to 5e-3 in `critic_surrogate.py`.

**The Level 0 gate in §3 was vacuous, and is now replaced.** "Q(SHOOT) above Q(SAFE_PASS)
in an unpressured central clear chance" *already passes* with the dead v5 critic — on 100%
of 36 real clear chances. It passes because `SHOOT` has the highest intent prior in the
corpus, not because the critic sees the chance. A gate that a constant vector satisfies
tests nothing. The functional gate is now `critic_health()` in `critic_surrogate.py`:

| check | fails when | why this and not loss |
|---|---|---|
| **Q spread** | range < a quarter of the reward band | a critic that whispers cannot express goal (1.35) against turnover (0.20); its advantage term cannot move a decision |
| **beats a table** | holdout rank correlation ≤ intent prior fitted on train | if the state conditioning is decorative, a table is cheaper and auditable |

Loss cannot do this job: a dead network has an ordinary weighted MSE, because predicting
the mean *is* the loss-minimising answer once every gradient is zero.

Dead-unit count is reported but deliberately **not** part of the verdict. The guard's first
draft failed any critic with more than a quarter of a layer dark; a synthetic run then
trained to holdout rank correlation **0.986** with 23 of 32 second-layer units dark, because
the surviving units carried the function. Anatomy is a hint, performance is the verdict.

**Still open in Level 0:** feeding the critic the brain's latent (tokens, attention, GRU
state) is untouched. That changes the critic's input width and therefore its interface with
`consequence_decision.apply_reasoning`, so it is a separate, versioned step. What is fixed
here is the goal vision and the health gate.


---

## 2. What "seeing ahead" decomposes into

Three capabilities. A system needs all three, in this order.

| # | capability | question it answers | PLOFA today |
|---|---|---|---|
| 1 | **Predict** | if I do this, what happens? | **nothing** |
| 2 | **Evaluate** | is that outcome good? | Q-critic, stale, 1-step |
| 3 | **Choose** | which action's future scores best? | argmax over the blend |

The brain does (2) and (3) on a *remembered* future. It has never done (1). Everything
below is about building (1) without breaking (2).

---

## 3. The ladder — four levels, each independently useful

### Level 0 — fix the instinct (days, not weeks)

Retrain the critic on the **v5 situational schema** so it can see goal bearing and mouth
width, and feed it the brain's own latent instead of 24 raw floats.

```
now:   Q(24 sensors + position)      -> 10 values
after: V(tokens, state, role, memory, plan, h) -> 10 values
```

This is a straightforward supervised job on traces we already record, and it is a
**prerequisite**: imagination scored by a blind critic is a blind imagination. It is also
the cheapest real improvement available, and it does not touch the brains.

**Gate:** the new critic must rank intents the way football does on held-out states —
`Q(SHOOT)` above `Q(SAFE_PASS)` in an unpressured central clear chance. (Today the reward
table does the opposite: §10.1 of the retrain plan.)

### Level 1 — one step of imagination (the core idea)

Learn a transition model on the latent the brain already carries:

```
h' = f(h, intent, context)          # a 32-d latent, small MLP
score(intent) = r_immediate + γ · V(h')
choose argmax
```

At decision time: for each of the 10 intents, imagine the next latent, score it, pick the
best. That is *"what happens if I do this?"* — one step, ten extra matmuls, negligible cost
against 1,055 decisions a match.

This is where foresight actually appears. Everything before it is preparation.

**Gate:** the transition model must beat the trivial baseline — predicting `h' = h`
(nothing changes). If it cannot, the latent is not predictive and Level 2 is required
before Level 1 can work.

### Level 2 — make the latent predictive (the real work, if Level 1 fails)

**This is the risk, and it is a known one.** The GRU state was trained to *choose actions*,
not to *predict the world*. A representation optimised for action selection is often a poor
substrate for prediction — it keeps what discriminates between actions and throws away what
would forecast the next state.

If Level 1's transition model is no better than `h' = h`, the fix is not a bigger model. It
is to **train the latent to be predictive**, by adding an auxiliary loss during brain
training:

```
loss = action_loss + λ · || g(h_{t+1}) − target_{t+1} ||²
```

where the target is a compact description of what actually happened next — ball position,
who won it, pressure, whether a chance was created. λ around 0.1–0.3.

This is a trainer change, not a brain-architecture change, and it makes every later
capability cheaper. **Budget for it.** If Level 1 works without it, skip it.

### Level 3 — two plies, and choosing who to pass to (only if 1–3 pay off)

The real action space is not 10 intents; it is 10 intents × up to 8 recipients. Two plies
over that is ~6,400 rollouts per decision — too many. But the attention layer already ranks
the entities, so search only the top 3 attended teammates: 10 × 3 = 30 actions per ply,
900 for two plies. That is affordable, and it is the version that starts to look like a
midfielder checking two passes ahead.

**Gate:** must beat Level 1 on chances created per match, not on fitness.

---

## 4. The sequencing constraint that matters most

**Foresight multiplies your values. If the values are wrong, it makes the wrongness more
efficient.**

We know the current values are wrong in at least one place: in the striker's clear-chance
bucket, the reward for shooting is 0.957 and for dribbling 1.163 (`RETRAIN_PLAN` §10.1).
A brain that can see ahead, graded by that value function, will *deliberately* choose the
dribble — and it will be harder to argue with, because now it has a reason.

So the order is not negotiable:

1. fix the shot head and the reward geometry (retrain plan, Phases 0–2)
2. fix the critic on the v5 schema (Level 0)
3. only then add the transition model and rollout (Level 1)

Doing 3 before 1–2 produces a confident, articulate, wrong player.

### 4.1 MEASURED 2026-11-16 — the player is indifferent, and the reward is why

The phrase that started this was "indifference from the real world": these players do not
behave like footballers who care. That is measurable, and it is worse than a style
problem — it is a property of the objective.

`tools/diag/_indifference_probe.py` holds the geometry **completely fixed**, varies ONE
context axis, and reports the total-variation distance between the resulting intent
distribution and the baseline. The control is the carrier moved ±10 m, which *must* move
the decision; without it a small number would be indistinguishable from a broken harness.

| axis | mean TV | argmax flips | share of the control's effect |
|---|---|---|---|
| score_diff −3 … +3 | 0.025 | 2.1 % | **5 % — IGNORED** |
| minute 5 … 89 | 0.020 | 2.2 % | **4 % — IGNORED** |
| pressure on/off | 0.378 | 42.5 % | 81 % — responds |
| *control: x ±10 m* | *0.466* | *50.4 %* | *100 %* |

Universal across roles — score_diff is 3–9 % of the control for ST, CAM, CM, CB and GK;
the clock is 4–12 %; pressure is the dominant driver everywhere (17–61 % of states flip).

**The information is present and the policy discards it.** Sensor `[18] score_diff` moves
exactly as commanded (−0.600 / 0.000 / +0.600 for −3 / 0 / +3) and `[19] minute_norm` goes
0.056 → 0.989 from minute 5 to 89, while the argmax does not change once. This is the same
shape as the shot finding (§`RETRAIN_PLAN` 10.5.1): information present, policy blind to it.

**And the reason is upstream of the brain.** `threat_reward.py` contains no reference to
`game_state`, `score_diff` or `minute` — not one. The reward is a pure pitch-geometry
potential (an xT grid), so a goal is worth exactly `GOAL_VALUE = 1.0` whether it is 0–0 at
minute 1 or 0–1 down at minute 89. The brain ignores the scoreline because **it was trained
against an objective that has no scoreline**. Wiring the context in would not help: there
is nothing on the other end for it to be worth.

**What this does to the plan's ordering.** §4 says foresight multiplies your values, so
wrong values make the wrongness more efficient. This is the extreme case: the values are
*constant* in the two dimensions a footballer's urgency lives in. A player who can imagine
the future perfectly, and values a goal identically in every minute of every scoreline, will
plan flawlessly toward a goal that never becomes more urgent. Foresight would be invisible.

So the sequence gains one step, and it goes **before Level 1**:

0. fix the shot head and the reward geometry (retrain plan, Phases 0–2)
1. fix the critic on the v5 schema (Level 0)
2. **make the objective state-dependent — give the player something to want**
3. only then add the transition model and rollout (Level 1)

Step 2 is small and testable, and the test already exists: re-run
`_indifference_probe.py` and require `score_diff` and `minute` to reach a real share of the
control. Until they do, Levels 1–3 have nothing to multiply.

**The mechanism for step 2, built 2026-11-16.** `threat_reward.stakes(score_diff, minute)`
returns a multiplier in 0.65 … 1.60 — chasing late is worth most (1.60 at −1 in the 89th),
level and late next (1.50), cruising late least (0.65). `threat_return(..., stakes_mult=)`
applies it to the **whole** return, not just to goals: a chasing team's turnover has to cost
more than a cruising team's, or the model is incoherent. The default is exactly 1.0, so every
existing corpus, report and comparison is byte-identical until a caller opts in — the same
discipline as `PLOFA_SHOT_HEAD`. Nine tests in `tests/test_stakes.py`, including one that
pins the default to the old numbers event by event.

What it does NOT do yet: change a single brain. The multiplier has to reach the corpus
before it can reach a policy, and the corpus has not been recollected against it. That is
the cost of step 2 and it is the reason the ordering puts it after the reward geometry is
settled rather than beside it.

### 4.2 MEASURED 2026-11-16 — the scoreline never reached the training data

§4.1 says the brain ignores the scoreline because the objective has no scoreline. That is
true, and it is not the whole story. Two further defects were found, and they are upstream
of the reward: **the scoreline was never in the data the brains were trained on.**

**(a) The collector decoded its arguments one slot to the right.**
`surrogate_collect._snapshot_decide` receives `decide`'s arguments and reads the situational
keywords out of them. `event_chain.py` calls:

```
decide(player, x, y, teammates, defenders, engine, profile,
       under_pressure, attacks_right, game_state, minute=..., ...)
```

Nine positional arguments after the player, `minute` by keyword. The wrapper read `args[7]`,
`args[8]`, `args[9]`. Measured against the engine's own call shape:

| wrapper believed | actually received |
|---|---|
| `under_pressure` | `attacks_right` |
| `attacks_right` | `game_state` — an enum, hence **always truthy** |
| `game_state` | **nothing** (`args[9]` does not exist) → `None` |

So `_score_diff_of(None)` returned 0 for every row, and `attacks_right` was always True.
Measured on `brains_trainer/corpus.raw.json`: **`score_diff` is exactly 0.0 in 100 % of the
15,426 rows** (`minute` does vary, 0.011 → 1.000, because it arrives as a keyword and the
out-of-range fallback landed on it by luck).

The second consequence is easy to miss and just as serious: every recorded decision,
**the away team's included, was perceived as attacking to the right.** Half the corpus is
geometrically mirrored.

**(b) `_score_diff_of` mapped four of the seven engine states wrongly.**

| state | old | correct | note |
|---|---|---|---|
| `HOME_CRUISE` (2-goal lead) | 0 | +2 | **live** — a 2-0 lead read as 0-0 |
| `AWAY_CRUISE` (2-goal deficit) | 0 | −2 | **live** |
| `HOME_CHASE` (chasing) | +1 | −2 | sign inverted; latent, the engine never emits CHASE |
| `AWAY_CHASE` | −1 | +2 | sign inverted; latent |

**(c) The table has no dimension for urgency anyway.** `FitnessSurrogate.bucket` is
`P·FT·OH·S·C·G | A·M·D` — pressure, third, half, space, centrality, goal-distance, then the
goal-geometry ladder. **Not one bit is the scoreline or the clock.** So even a corpus with
varying scorelines would be averaged back to indifference within each cell.

**What this changes.** Step 2 was scoped as "apply `stakes` to the corpus". It is actually
four things, in order: fix the collector (done — `_decode_situational_args`, 14 tests in
`tests/test_scoreline_reaches_the_brain.py`), fix the score mapping (done), **extend the
bucket key** so urgency has somewhere to live, and then **recollect**. Only then does
`stakes` have anything to multiply, because until now the objective was averaging over a
constant.

The bucket extension is the one open design decision, and the file already contains its own
template: `goal_geometry` added `A·M·D` to a 6-bit key by making the key **hierarchical** —
rich cell first, coarse parent as fallback — precisely so that sparse data degrades to the
old behaviour instead of collapsing to the prior. A stakes suffix (`K{behind,level,ahead}`
× `T{early,mid,late}`, nine cells) should be added the same way, not as a flat widening of
the key.

The most uncomfortable implication: **the shipped brains were distilled from a corpus in
which half the decisions were geometrically mirrored and no decision knew the score.** That
is a sufficient explanation for the indifference, and it is also a reason to distrust any
fine-grained conclusion drawn from those brains — including, possibly, the shot head's
anti-correlated `p_SHOOT` (§`RETRAIN_PLAN` 10.5.3). A recollection is not optional
housekeeping; it is the precondition for both step 2 and any credible retrain.

---

## 5. How we would know it works

Three tests, in increasing strength. All are cheap; none requires trusting a fitness number.

**5.1 Transition accuracy.** Predict `h_{t+1}` on held-out sequences; compare squared error
against the `h' = h` baseline. Needs to beat it clearly, not marginally.

**5.2 Counterfactual divergence.** Find decisions where the greedy choice and the 1-ply
choice *differ* — the cases where looking ahead changes the answer. Measure how often the
1-ply agent takes the 1-ply action, and whether those matches produce more chances. If the
two never disagree, the rollout is decorative.

**5.3 The specific football test.** Construct the situation where instinct and foresight
must disagree: a safe square pass now that leads to a turnover under pressure, versus a
risker ball that creates a chance. Instinct (Q on the current state) prefers the pass.
Foresight should prefer the through ball. If the agent still passes, it is not imagining.

---

## 6. Cost, honestly

| level | work | risk |
|---|---|---|
| 0 — critic on v5 + latent | a day; supervised on existing traces | low |
| 1 — transition model, 1 ply | 2–3 days; needs replayed latents | **medium — may fail §3 Level 1's gate** |
| 2 — predictive auxiliary loss | 3–5 days; trainer change, full retrain | medium; makes everything else cheaper |
| 3 — 2-ply with attention pruning | 3–4 days | low once 1 works |

Compute is not the constraint. 1-ply is ~10 extra 32-d matmuls per decision. The constraint
is **data and a correct value function** — which is why Level 0 comes first and why the shot
fix comes before that.

---

## 7. What this still will not give you

Worth stating, because "sees ahead" is easy to over-read:

- **No deception.** A feint requires modelling what the *opponent* believes, then acting to
  corrupt it. A forward model of the world is not a model of another mind.
- **No stakes.** It will not play differently in a relegation six-pointer, because nothing
  in the system knows what is at stake (`RETRAIN_PLAN` §6.5).
- **No learning inside the match** — *addressed in §9*, which is mostly a reconnection of
  machinery that already exists, plus a per-possession review.
- **No commitment** — *addressed in §8*, where the decision moves to the moment the ball is
  released and is then locked. That is the mechanism that makes a choice a choice.

Foresight makes it a player who can see the next move. It does not make it a player with
nerves, a crowd, or a reason to be afraid of losing — §6.5 of the retrain plan covers what
that would take.

---

## 8. Commitment — decide before the ball arrives

### 8.1 The problem, stated precisely

The brain re-decides on every tick, at zero cost. A choice you can revise for free is not
a choice; it is a preference. Meaning comes from irreversibility — from having decided on
information you can no longer get back.

### 8.2 The mechanism: move the decision to the release

A real player commits **while the ball is travelling to him**, not when it arrives. He
decides "first time, wide" as the pass leaves his teammate's foot. If the picture changes
in that half-second, he has already committed.

So: **the decision is made at ball-release, on the predicted arrival state, and locked.**

```
at release of a pass toward P:
    t_arrive   = distance / ball_speed
    s_arrive   = transition_model(s_now, t_arrive)      # predicted, not observed
    committed  = P.brain(s_arrive)                       # one decision, locked
    P.commit(committed, valid_until=t_arrive)

at arrival:
    execute(committed)          # no re-decision
```

Three properties fall out of this, and they are why it is the right mechanism rather than
an artificial lock timer:

1. **Commitment is emergent, not imposed.** There is no "you may not change your mind for
   0.4 s" rule. There is simply no second decision, because the moment for deciding passed.
2. **It makes scanning valuable.** Your predicted arrival state is only as good as what you
   knew at release. This is the same gap as `RETRAIN_PLAN` §6.3 — vision without looking —
   and commitment gives looking a *payoff*.
3. **It makes foresight load-bearing.** `s_arrive` comes from the transition model of §3
   Level 1. Commitment and imagination are the same machinery viewed from opposite ends:
   one predicts forward to decide, the other locks the decision in place.

### 8.3 The football failure it creates — and why that is good

A committed intent that the world contradicts is **caught in possession**. That is not a
bug; it is the most common failure in football, and it is currently impossible in PLOFA
because the carrier can always re-route. The ledger (§9) records it as *surprise*: the gap
between the state you predicted and the state you got.

A player who is never caught in possession is not good. He is not deciding.

### 8.4 Two smaller commitment layers

- **Physical commit.** SHOOT, CROSS and long passes require weight transfer. Once chosen,
  a 0.15–0.3 s lock prevents a switch. Cheap, and it stops the absurd case of a player
  starting a shooting motion and turning it into a square ball.
- **The shout (optional).** When a player commits, his teammates' plan and attention shift
  toward supporting him. The team-level `TeamPlan.propose` hysteresis already exists; this
  would extend it from phases to individuals. Higher cost, real coherence benefit.

### 8.5 How we would know it worked

| measure | today | expected |
|---|---|---|
| intent changes per possession | high (free re-deciding) | **sharp drop** |
| caught-in-possession rate | **0 — impossible** | 5–15% of receptions |
| prediction hit rate (committed intent still right at arrival) | n/a | the *skill* number: should climb as scanning improves |
| contradiction rate between committed and executed | n/a | ~0 by construction |

The prediction hit rate is the interesting one: it is a direct measure of how well the
player reads the future, and it is the number that should rise when §3 Level 1 lands.

---

## 9. Learning that never stops — the per-possession ledger

### 9.1 What already exists (more than expected)

The in-match learning loop is **built**, and it persists across matches:

| piece | where | state |
|---|---|---|
| every event observed mid-match | `match_engine.py:6950` → `mind.observe(event)` | **live** |
| episodic memory store + decay | `cognition/mind.py::_remember` | **live** |
| memory → temperament | `experience.apply_memory` — *"a scar changes behaviour now"* | **live** |
| temperament biases | `risk_in_box`, `tackle_aggression`, `confidence` | **live** |
| memory read by the shipped brain | `memory_tail` → 8 input floats | **live** |
| persistence across matches | `season_state.set_player_cognition` / `from_state` | **live** |
| temperament **applied** to the shipped brain | `mind.temper()` | **CONNECTED 2026-11-08** |

**The disconnect is the finding.** `mind.temper()` — the function that turns those biases
into a distorted probability distribution — is called at exactly one place,
`cognition_brain.py:97`. That is the cognition path, which is bypassed for every shipped
v8 brain, because `_maybe_situational` returns first.

So today: a player's temperament is computed from his scars, saved to the season state,
carried into the next match — and then **never applied to the player who is actually
playing.** The shipped brains read the memory as 8 input floats and are never tempered by
it. Half a system, wired to the wrong path.

**Step one of this section is therefore not a new feature. It is a one-line reconnection**,
in `situational_decision.py`, before sampling:

```python
from cognition.mind import get_mind
mind = get_mind(player.name)
if mind is not None and mind.engaged:
    probs = mind.temper(probs, labels, under_pressure, fatigue)
```

That single change makes a player who has been beaten in the box measurably more
conservative *within the same match*. It is the cheapest real behaviour change on the list.

#### DONE — 2026-11-08, and what it actually took

The one-liner was not quite enough, and the reason is worth keeping: `temper` is only safe
if the vector it returns is the same shape as the one it was handed. `chosen_idx` indexes
`INTENT_LABELS` **positionally** (`brain_integration._INTENT_BY_INDEX`), so a short vector
would not crash — it would silently relabel every downstream intent. So the seam rejects a
malformed result rather than trusting it, and returns the untouched distribution instead.

Implemented as `situational_decision._apply_temper`, called from `decide()` between the
softmax and the coach/consequence seams. Order is deliberate: **temper BEFORE the coach
bias**, because an order should bias a scar, not erase it. The consequence critic works in
log space afterwards, so the distortion survives it exactly.

Measured end-to-end through the real shipped brain (`brains/CAM.json`, carrier at (40, 30),
seed 99, `tests/test_temperament_live.py`):

| temperament | chosen intent | p | shift (total variation) |
|---|---|---|---|
| neutral (risk 1.0, conf 1.0) | SHOOT | 0.8360 | **0.0000** |
| burned in the box (risk 0.4) | SHOOT | 0.8250 | 0.0462 |
| firm (conf 2.0) | SHOOT | 0.9650 | 0.1296 |
| indecisive (conf 0.3) | SHOOT | 0.4350 | 0.3990 |

The neutral row is the important one: **shift 0.0**, i.e. the shipped game's behaviour is
byte-identical until a player actually has a scar. A live match with 11 registered minds ran
657 decisions without error.

Two tests guard this, and they guard different things: `test_temperament_seam.py` (12 tests)
covers the arithmetic and the guards, and one of its assertions reads the SOURCE of
`decide()` to prove the call site still exists — because the original bug was a correct
function that nobody called, and a unit test alone would have passed on the broken system.
`test_temperament_live.py` (4 tests) covers the wiring through the production brain.

### 9.2 What is genuinely missing: the review

Memory reacts to events. It does not **review** them. A human does something different and
better: after a move breaks down, he replays it — *"I should have gone wide there"* — and
that review changes what he does next time.

That is what the user asked for: *"a file that treats itself like a post-match evaluator per
player, takes the post match back to the match to analyse with them."* The right compression
is not per match. It is **per possession** — the post-mortem, run continuously, during play.

### 9.3 The design: `player_ledger.py`

One object per player, alive for the whole season, updated at every possession end.

```
on possession end (for each player involved):
    1. REPLAY   the possession's decisions through the critic (§3 Level 0)
                -> for each decision: what I chose, what it returned,
                   and the best alternative from the same state
    2. REGRET   regret = V(best alternative) - V(chosen)     per decision
    3. UPDATE   fold regret + surprise (from §8.3) into beliefs, EMA,
                with a learning rate that decays across the match
    4. PERSIST  write beliefs to disk, per player, across fixtures
```

**The belief vector** — 10 floats, distinct from the existing 8 memory features because
these are about *the opponent and today*, not about *me and my career*:

| # | belief | learned from |
|---|---|---|
| 0 | `opp_press_speed` | how fast they close me down |
| 1 | `opp_weak_flank` | which side keeps yielding |
| 2 | `opp_line_height` | how high their back line sits |
| 3 | `my_pass_works` | my completion rate today |
| 4 | `my_shot_works` | my shooting return today |
| 5 | `teammate_hot[0..2]` | who is actually delivering |
| 6 | `referee_leniency` | contact allowed, from fouls called |
| 7 | `form` | recent personal success, decaying |
| 8 | `surprise` | mean gap between predicted and actual (§8.3) |
| 9 | `belief_confidence` | how much evidence these rest on |

**The learning rate decays within the match** — fast in the first twenty minutes, slow
after. That is how humans behave: you work the opponent out early, then you stop revising.

**Why beliefs and not weights.** Backpropagating during a match would be non-deterministic,
would cause catastrophic forgetting, and would make every gate unreadable. Humans do not
retrain their nervous system at half-time; they update *beliefs*. So the two timescales are:

| timescale | what changes | mechanism |
|---|---|---|
| seconds–minutes | beliefs, temperament | EMA + `apply_memory` — **no weights touched** |
| matches–seasons | the network | the trainer |

The brain already conditions on inputs, so a belief block changes the choice without a
single gradient step. That is the elegant part: **the brain does not learn mid-match. Its
beliefs do — and the brain reads them.**

### 9.4 The free gift: counterfactuals

Step 1 of the review — replay and score the alternative — is **exactly the counterfactual
reasoning identified as absent** in `RETRAIN_PLAN` §11.4. The brain cannot represent *"if
I had passed instead"*. The ledger computes it every possession, offline, and feeds the
answer back as a belief.

So the review does not just add learning. It adds the one form of reasoning the network
structurally lacks, without changing the network.

### 9.5 How we would know it worked

| measure | expected |
|---|---|
| regret per possession | falls over a match, and over a season |
| belief calibration | `opp_press_speed` correlates with the opponent's actual press rate |
| behavioural: same fixture, second half vs first | measurable shift in risk and pass selection |
| cross-match carryover | a player's beliefs after facing a team predict his choices against them next time |

The last one is the real test: if beliefs persist and matter, a player should start the
return fixture already knowing how they play.

### 9.6 Risks

- **Belief drift.** A player can learn something wrong and get worse. Mitigate by shrinking
  beliefs toward the population mean, and by capping the per-possession update.
- **Determinism.** The ledger must be seeded and replayable, or the gate becomes noise. Same
  discipline as `perception.py`'s per-snapshot seed.
- **Input width.** The belief block changes the situation width — so it is a `v7` change and
  must not be bundled with the `v6_read` retrain (§7 of the retrain plan).

---

## 10. Revised sequence, with these two included

| step | work | depends on | cost |
|---|---|---|---|
| 0 | shot fix + reward geometry | — | retrain plan Phases 0–2 |
| 1 | **reconnect `mind.temper()`** to the shipped path (§9.1) | nothing | **hours** |
| 2 | critic on the v5 schema + latent (§3 L0) | 0 | days |
| 3 | `player_ledger.py` — replay, regret, beliefs, persist (§9.3) | 2 | days |
| 4 | transition model + 1-ply rollout (§3 L1) | 2 | days |
| 5 | **pre-commit** (§8.2) — needs `s_arrive` | 4 | days |
| 6 | 2-ply with attention pruning (§3 L3) | 4 | days |

Step 1 is available today and costs nothing. Step 5 is the one that gives choices meaning,
and it is only possible once step 4 can predict an arrival state — which is why commitment
and foresight are one project, not two.
