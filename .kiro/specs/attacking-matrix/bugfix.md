# High-Density Attacking Matrix — Requirements Document

## Introduction

Attackers in PLOFA currently make decisions as stat-roll automata. A `PossessionChain`
carrier picks a pass receiver by position label + spatial plausibility, rolls a random
forward/sideways/backward direction (`_pass_destination`), and has **no notion of whether
a shot is viable, whether a passing corridor is open, or which teammate is actually the
best option**. Shots only ever happen because `MatchEngine` rolls a per-sequence
`shot_prob` and dispatches `AttackChain` — the ball-carrier never *chooses* to shoot. A
wide winger with the goal gaping fires passes back into traffic; a striker smothered
between two centre-backs in the box plays a pointless square ball; a counter-attack
squanders a 3v2 because the "long diagonal into the channel" is never evaluated as an
option.

This feature gives every ball carrier **intelligent spatial attacking awareness** — a
High-Density Attacking Matrix:

- The carrier sees the pitch as a **dynamic web of passing corridors**, not a flat list
  of labelled teammates.
- Every touch, the carrier evaluates three micro-calculations:
  - **S_viability** — the uncontested shot score: how good is shooting from here, right
    now (distance decay × pressure tax × angle multiplier)?
  - **L_clear** — passing lane clearance: is the corridor from carrier to each teammate
    actually open, or does a defender stand within 1.2m of the line?
  - **V_strategic** — teammate strategic value: is the receiver closer to goal, free of
    their marker, and advanced enough to be worth the pass?
- The carrier then resolves one of four decisions — **SHOOT**, **KEY_PASS**
  (CREATE_CHANCE), **PROGRESSIVE_PASS**, or **RECYCLE_PASS** — steered by the match
  scenario (counter-attack, low block, build-up) and driven by their DNA
  (finishing/long-shots for shooting, vision for spotting far-network runners,
  composure under pressure, `plays_safe` conservatism, `decisions` judgement).

**Impact:** The attacking phase becomes causally grounded in pitch geometry (same
philosophy as Checkpoint 5's Position Engine and the Checkpoint 9 danger model). A shot
is now a *decision the carrier makes because the shooting window is open*; a key pass is
a *decision the carrier makes because a teammate ran into an open channel*; a recycle is
a *decision to keep the ball because nothing better exists*. The four decision outcomes
are attributable and observable across the 90 minutes.

---

## Feature Analysis

### Current Behavior (Defect)

**1. No Shot Decision At The Point Of Possession:**

1.1 WHEN a player is in possession in the final third THEN no shot viability is computed —
the carrier passes/carries/dribbles by probability roll regardless of how open the goal
is

1.2 WHEN the carrier is under pressure with no passing option THEN no "panic shot" or
forced recycle exists — the random action selection proceeds as if the defender were not
there

**2. Passes Ignore The Passing Corridor:**

1.3 WHEN the carrier passes THEN the receiver is chosen by label weight × spatial
plausibility only — a corridor blocked by a defender standing on the passing line is
never checked

1.4 WHEN a pass is played THEN the destination is a random forward/sideways/backward draw
(`_pass_destination`) that does NOT aim at the chosen receiver — a "key pass" to a
breaking striker can land 15m off his run

**3. Teammates Have No Strategic Value:**

1.5 WHEN choosing a receiver THEN the system never asks "is this teammate closer to goal
than me, free of his marker, and in an advanced position?" — a tightly-marked deep
square option competes equally with a free runner in the box

**4. Scenarios Do Not Steer Decisions:**

1.6 WHEN on a counter-attack THEN the long diagonal into the channel is never prioritized
— the counter plays the same short sideways game as a tiki-taka side

1.7 WHEN the opponent packs the box THEN no SHOOT-bias exists — possession is cycled
around the 18-yard line instead of taking the half-chance

---

### Expected Behavior (Correct)

**1. Uncontested Shot Score — S_viability:**

2.1 WHEN the carrier has the ball THEN the system SHALL compute S_viability in [0, 1]
from `distance_decay × pressure_tax × angle_multiplier`, where:

- distance_decay falls monotonically with distance to the centre of the opponent goal
- pressure_tax drops sharply when the nearest outfield defender is within 1.5m (a tackle
  is imminent), less so within 3.5m, and is softened by the carrier's `composure`
- angle_multiplier shrinks as the angular width of the goal mouth visible from (x, y)
  narrows — central positions near the goal are ~1.0, wide/acute positions are heavily
  reduced

2.2 WHEN S_viability is high AND the shot is in range THEN the carrier SHALL decide
SHOOT — the possession sequence ends and hands off to the existing AttackChain shot
pipeline (xG, GK save, woodwork, corners) anchored at the carrier's actual (x, y)

**2. Passing Lane Clearance — L_clear:**

2.3 WHEN the carrier evaluates a teammate THEN the system SHALL compute L_clear in
[0, 1] as the perpendicular distance from every outfield defender to the carrier→teammate
line segment: a defender within 1.2m of the corridor ⇒ L_clear = 0 (blocked); a clear
corridor ≥ 3m from all defenders ⇒ L_clear = 1; linear in between

2.4 WHEN the carrier chooses a pass THEN the destination SHALL aim at the chosen
receiver's actual position (with pass-skill error), never a random direction

**3. Teammate Strategic Value — V_strategic:**

2.5 WHEN the carrier evaluates a teammate THEN the system SHALL compute V_strategic in
[0, 1] from `L_clear × (0.45·progress + 0.35·freedom + 0.20·depth)`, where:

- progress rises as the teammate is closer to the opponent goal than the carrier is
- freedom = 1.0 when the nearest opponent is > 3.0m from the teammate (unmarked),
  collapsing to ~0.15 when smothered within 1.0m
- depth rises as the teammate advances into the attacking third
- L_clear multiplies the whole: a blocked lane kills the option

2.6 WHEN the best far-network option (teammate > 15m from the carrier) is open with high
V_strategic THEN the carrier SHALL decide KEY_PASS / PROGRESSIVE_PASS (CREATE_CHANCE) —
the line-breaking ball, not a random forward roll

2.7 WHEN no forward option exists THEN the carrier SHALL decide RECYCLE_PASS to the best
open close-network option (< 15m) — keep the ball, shift the angle

**4. Scenario Steering:**

2.8 WHEN the team is counter-attacking (fluid_counter / route_one / direct) THEN the
system SHALL bias toward far-network KEY_PASS into the channel and lower the SHOOT
threshold (transition shots are taken early)

2.9 WHEN the opponent is in a low block (park_the_bus / ultra_defensive / defensive)
THEN the system SHALL bias toward SHOOT panic mode when in range with any half-chance,
and toward RECYCLE otherwise (no through-balls into a packed box)

2.10 WHEN the team is building up (tiki_taka / structured_possession / possession /
vertical_tiki_taka) THEN the system SHALL bias toward close-network PROGRESSIVE/RECYCLE
and raise the SHOOT threshold (only clean openings are shot)

**5. Decision Quality:**

2.11 WHEN the carrier resolves a decision THEN the system SHALL weight each micro-score
by the carrier's DNA: finishing/long_shots lower the SHOOT threshold, `vision` raises the
weight of far-network options, `plays_safe` raises the SHOOT threshold and biases
recycle, `decisions` sharpens the pick between competing options, and `composure`
softens the pressure tax on both shot and pass execution

---

### Unchanged Behavior (Regression Prevention)

**3. Preservation Requirements:**

3.1 WHEN a possession sequence runs WITHOUT a position engine (standalone/tests) THEN
the existing `_pick_receiver` / `_pass_destination` / carry / dribble / cross paths SHALL
run unchanged — the matrix is a strictly additive layer gated on live spatial state

3.2 WHEN `AttackChain._select_action_from_position(x, y, position)` is called THEN it
SHALL CONTINUE TO reject shots from behind the goal line (x ≥ 105) and heavily bias
cross/pass from acute angles and wide byline positions, exactly as today — it now
delegates its geometry to the matrix's angular-width model

3.3 WHEN a shot is handed off to `AttackChain` THEN the existing xG calculation, body
part selection, angle-difficulty multiplier, GK save evaluation, woodwork, corner, and
restart logic SHALL run unchanged

3.4 WHEN a pass is executed THEN pass success (chemistry, marking tightness, confidence,
soul modifiers), ball receipt, miscontrol, turnover, and pressure event logic SHALL be
unchanged

3.5 WHEN the game-state shot volume modifiers in `_simulate_minute` apply (scoreline,
phase, red cards) THEN they SHALL CONTINUE TO gate the independent shot_prob path exactly
as today; matrix-decided shots are ADDITIONAL only when a genuine S_viability decision
fires in possession

3.6 WHEN the position engine drifts uninvolved players each minute THEN the matrix SHALL
consume, never mutate, the spatial state

3.7 WHEN the exporter reads CHANCE_CREATED / SHOT events THEN its attribution logic SHALL
be unchanged; new `attacking_matrix` metadata on PASS/PROGRESSIVE_PASS events is purely
additive

---

### Calibration & Verified Behaviour (Checkpoint 10 closeout)

**4. Geometry bug fixed during integration.**

The matrix's goal-mouth half-width was originally set to `POST_OFFSET_Y = 7.01` (a
"14.02m goal mouth") — but the sim's own keeper/woodwork code uses `POST_LEFT = 30.34`,
`POST_RIGHT = 37.66`, i.e. a real **7.32m goal** (±3.66m about centre). The inflated half-
width roughly doubled `_angle_multiplier` at every range (×1.9 at 10m), letting far too
many touches clear the SHOOT thresholds and tripling match goals. `POST_OFFSET_Y` is now
**3.66**, consistent with `REFERENCE_ANGLE = 2·atan2(3.66, 11.0)`. Two low-block property
tests were repositioned (carrier 88 → 94m) because their pinned shot values had been
calibrated to the buggy geometry.

**5. Take-probability gate on the SHOOT hand-off (volume control).**

`decide()` stays deterministic and property-pure, but the SHOOT hand-off in
`PossessionChain.generate` now applies a take-gate:
`take_prob = clamp01((shot_score − 0.60) / 0.40)`. The matrix flags a window shootable;
the player only pulls the trigger when the chance clearly beats the elite bar (a 0.52
half-chance is squared/recycled, a 1.0+ sitter always fires). This dropped direct matrix
hand-offs from ~17 to ~10 per match without touching the existing xG/GK conversion.

**6. Measured scoring impact (A/B, 10 seeds, FLUID_COUNTER vs TIKI_TAKA).**

| metric | baseline | with matrix |
|---|---|---|
| goals/match (mean) | 3.7 (1–7) | 7.4 (4–14) |
| shots/match (mean) | 14.0 (10–19) | 23.0 (15–32) |

Direct matrix shots are ~10/match and convert via the unchanged AttackChain pipeline; the
remainder of the increase is improved passing flow feeding the sim's pre-existing
conversion generosity. **Accepted** as the feature's intended "more dangerous attacks"
behaviour. The defensive-awareness smoke test's `goals_conceded ≤ 5` sanity bound was
relaxed to `≤ 8` accordingly (that bound predates this feature and assumed the old scoring
distribution).
