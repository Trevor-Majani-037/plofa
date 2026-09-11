# Defensive Awareness — Requirements Document

## Introduction

Defenders in PLOFA currently act as stat-roll automata. A `DefensiveChain` picks a
defender by position label, rolls a fixed success table, and emits an event — the
defender never "sees" the ball, their own goal, or the danger between them. Clearances
are a single undifferentiated `CLEARANCE` event, with no distinction between the two
real-world categories (Opta / StatsBomb): **foot clearances** (kicking the ball away
from a dangerous area without intending possession for a teammate) and **headed
clearances** (redirecting an aerial ball out of danger without aiming at a teammate).

This feature gives defenders **intelligent spatial awareness**:

- Every defender understands their own current `(x, y)`, the ball's `(x, y)`, and the
  goalpost `(x, y)` of the goal they defend.
- They coordinate through a shared, live **danger / threat level**: danger rises as the
  ball approaches the defended goalpost xy, and defenders' only job for 90 minutes is to
  keep the ball away from that goal — tackling, intercepting, pressing, blocking, and
  clearing crosses to drive the danger level back down.
- Danger reduction is never perfect: football is not on your side every time. When the
  opponent does force the ball into the box, the defence can still fail — the danger
  level peaks and a goal can be conceded.

**Impact:** Defensive play becomes causally grounded in pitch geometry (same philosophy
as Checkpoint 5's Position Engine and Checkpoint 6/7's shot geometry + restart logic).
Clearances split into headed/foot with attribute-correct success models, defenders pull
into a compact goal-side block when the ball gets close to their goal, and the danger
level is a live, observable quantity across all 90 minutes.

---

## Feature Analysis

### Current Behavior (Defect)

**1. No Spatial Threat Model:**

1.1 WHEN the opponent moves the ball close to a team's goal THEN no danger/threat level
is computed, tracked, or acted upon — the defence reacts identically whether the ball is
at the halfway line or the penalty spot

1.2 WHEN a defensive action is needed THEN the defender is chosen by position label
weight only, ignoring how far each defender actually is from the ball's current
coordinates

**2. Undifferentiated Clearances:**

1.3 WHEN a defender clears the ball THEN a single `CLEARANCE` event is emitted with no
`body_part` and no distinction between a headed clearance and a foot clearance

1.4 WHEN a defender heads a cross clear THEN the success probability uses no aerial
attributes — a 0-jump no-nonsense CB heads clearances with the same rate as an aerial
dominant target man

1.5 WHEN a defender hoofs a loose ball clear THEN the success probability uses no
defensive clearances/composure/anticipation attributes

**3. No Coordination:**

1.6 WHEN the ball enters the defending third THEN defenders do not compact toward a
goal-side, ball-facing block — each player drifts independently toward their own home
coordinate with no shared threat signal

1.7 WHEN a team is under heavy pressure THEN the choice of defensive action (tackle vs.
interception vs. clearance vs. block) is drawn from a fixed weight table regardless of
how much danger the team is actually in

---

### Expected Behavior (Correct)

**1. Live Danger / Threat Level:**

2.1 WHEN the opponent moves the ball anywhere on the pitch THEN the system SHALL compute
a per-team danger level from the ball's position relative to the defended goalpost xy
(ball closer to goal ⇒ higher danger), modulated by centrality (central shots are more
dangerous than wide), danger zone (six-yard box > penalty area > edge of box > deep),
approach momentum (a ball advancing toward the goal is more dangerous than one going
backwards), and attacker density near the ball

2.2 WHEN the ball is in a team's defensive half THEN that team's danger level SHALL be a
monotonically non-decreasing function of proximity to the defended goal (for fixed y) —
closer ball ⇒ danger never falls

2.3 WHEN a defender performs a successful defensive win (tackle, interception, block) or
an effective clearance THEN the team's danger level SHALL fall by an amount proportional
to how far the ball is moved away from the defended goal

2.4 WHEN a goal is conceded THEN the conceding team's danger level SHALL peak (the threat
was realised), then reset to the low kickoff baseline once play restarts

2.5 WHEN a match ends THEN the danger level history (minute-by-minute) SHALL be
observable for export and analysis

**2. Headed vs. Foot Clearances:**

2.6 WHEN a defender clears an aerial ball (cross, corner, high ball, headed pass) in a
dangerous area THEN the system SHALL emit a headed clearance: `body_part = "head"`,
success driven by `aerial_dominance` (jumping + heading + bravery), heading technique,
and composure

2.7 WHEN a defender clears a low/loose ball THEN the system SHALL emit a foot clearance:
`body_part = "foot"` (`right_foot`/`left_foot`), success driven by defending `clearances`,
`marking`, `composure`, and `anticipation`

2.8 WHEN a clearance is ineffective (failure) THEN the danger level SHALL NOT fall — the
ball stays in the danger zone, typically dropping to an opponent or out for a corner

**3. Spatial Defender Selection & Coordination:**

2.9 WHEN a defensive action is needed at coordinates (x, y) THEN the defender SHALL be
selected using positional plausibility at (x, y) — the nearest, most relevant defenders
react first, never a defender stranded on the opposite side of the pitch

2.10 WHEN a team is out of possession and danger exceeds a threshold THEN its defensive
line (GK/CB/LB/RB/CDM) SHALL pull toward a compact, goal-side block that sits between the
ball and the defended goal, with the block deepening (closer to goal) as danger rises and
shifting to the ball side laterally

2.11 WHEN danger is high in the box THEN the system SHALL bias the defensive action
choice toward clearances and blocks (get it away) and away from attempting to play out —
when danger is low it SHALL bias toward tackles and interceptions (win it back)

---

### Unchanged Behavior (Regression Prevention)

**3. Preservation Requirements:**

3.1 WHEN a corner is won (existing causal corner system) THEN the system SHALL CONTINUE
TO queue `pending_corner_for` and generate a `CORNER_TAKEN` event

3.2 WHEN a shot has xG calculated THEN the system SHALL CONTINUE TO apply body_part
multipliers, pressure penalties, and situation adjustments unchanged

3.3 WHEN a goal is scored THEN the system SHALL CONTINUE TO update the score, emit the
GOAL event, shift momentum, and trigger celebration + kickoff flow unchanged

3.4 WHEN a tackle fails THEN the system SHALL CONTINUE TO apply the existing foul / card
probability logic unchanged

3.5 WHEN the position engine drifts uninvolved players home each minute THEN the system
SHALL CONTINUE TO apply phase/game-state modifiers and line cohesion unchanged

3.6 WHEN a defensive event is emitted THEN the system SHALL CONTINUE TO append it to the
timeline, drain stamina, and update `last_ball_x/y` via the existing event absorb path

3.7 WHEN no danger is present (ball far from a team's goal) THEN defensive behaviour
SHALL be statistically indistinguishable from the current system (danger-scaled
modifiers must converge to the existing baseline at danger ≈ 0)
