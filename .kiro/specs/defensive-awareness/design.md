# Defensive Awareness — Design

## Overview

Adds an **intelligent defensive awareness layer** to the PLOFA 26/27 match engine. It
gives every defender a spatial understanding of the ball, their own position, and the
goalpost xy of the goal they defend, and ties all defensive behaviour — positioning,
action choice, and clearance type — to a shared, live **danger / threat level** that
rises as the ball approaches the defended goal and falls as the defence pushes the ball
away.

The work is spread across five pieces that mirror the existing Checkpoint 5-8 module
conventions:

- **`threat_engine.py` (NEW)** — the Danger/Threat Intelligence module: pure geometry
  → per-team live danger level, danger history, and defensive intensity scaling.
- **`event_chain.py`** — `DefensiveChain` becomes danger-aware: headed vs foot
  clearances, danger-scaled action selection, and spatially-plausible defender pick.
- **`position_engine.py`** — `defensive_block()` compact goal-side coordination.
- **`match_engine.py`** — wiring: threat engine lifecycle, live danger tracking in
  `_absorb_chain`, danger-scaled contests in `_simulate_minute`, block pull in
  `_run_minute`.
- **`exporter.py`** — defensive-awareness JSON section (per-team danger summary +
  headed/foot clearance split).

**Testing Approach:** property-based exploration of the danger function and clearance
selection (asserting monotonicity, zone ordering, central>wide, attribute sensitivity),
plus preservation checks that defensive flow at zero danger matches the old baseline.

---

## Glossary

- **Danger Level (D)**: 0–100 per-team live quantity. How threatened the goal a team
  defends is right now. 0 = no threat (ball in the opponent's half); 100 = the opponent
  is about to score (ball in the six-yard box, central, under no pressure).
- **DangerZone**: `six_yard_box` / `inside_box` / `edge_of_box` / `outside_box` /
  `deep` — derived from the ball's x relative to the defended goal (mirrors
  `PitchZone.xg_zone`).
- **Approach Momentum**: whether the last ball movement advanced toward the defended
  goal (threat rising) or away from it (relief). 1.0 = advancing, 0.0 = retreating.
- **Headed Clearance**: Opta/StatsBomb "clearance" where the defender uses their head to
  redirect an aerial ball out of danger without aiming at a teammate. `body_part="head"`.
- **Foot Clearance**: a defensive intervention where the defender kicks the ball away
  from a dangerous area with no intended possession for a teammate. `body_part="foot"`.
- **Defensive Block**: the coordinated goal-side, ball-facing shape the defensive line
  (GK/CB/LB/RB/CDM) pulls into when out of possession in danger.
- **Threat Event**: any timeline event that updates the ball position; the threat engine
  consumes these to keep danger live.

---

## Danger Model (Formal)

### Defended Goal

For team T, the goalpost xy of the goal T defends:

```
HOME defends goal at (0, 34); AWAY defends goal at (105, 34)
goal_x(T) = 0 if T == home else 105
```

All danger is measured against `goal_x(T)`. Equivalently, when the ATTACKING team attacks
`attacks_right`, the DEFENDING team defends `goal_x = 105 if attacks_right else 0`
(identical to `BaseChain.goal_x`).

### DangerAssessment

```
FUNCTION assess(ball_x, ball_y, own_goal_x,
                attackers_near=0, defenders_near=0,
                ball_aerial=False, approach=1.0) -> DangerAssessment

  dist       := HYPOT(ball_x - own_goal_x, ball_y - 34.0)      // m to goal centre
  proximity  := CLAMP(1.0, 0.0, 1.0 - dist / 70.0)             // 1 at goal mouth, 0 beyond 70m
  centrality := 1.0 - MIN(1.0, ABS(ball_y - 34.0) / 28.0)      // 1 central, 0 on touchline
  zone       := xg_zone(ball_x, ball_y, attacks_right=(own_goal_x == 0))  // mirrored
  zone_mult  := { six_yard_box: 1.40, inside_box: 1.25, edge_of_box: 1.10,
                  outside_box: 1.00, deep: 0.55 }
  pressure   := CLAMP(1.0, 0.5, 0.5 + 0.15*(attackers_near - defenders_near))
  momentum   := 0.65 + 0.35 * approach
  shot_pos   := (dist < 30) AND (angle_to_goal < 60°)            // in a shooting position
  shot_mult  := 1.20 if shot_pos else 1.00

  level      := 100 * MIN(1.0,
                  proximity ** 1.5
                  * zone_mult
                  * (0.55 + 0.45 * centrality)
                  * pressure
                  * momentum
                  * shot_mult)

  RETURN DangerAssessment(level, proximity, centrality, zone, zone_mult,
                          pressure, momentum, dist, ball_x, ball_y,
                          attackers_near, defenders_near)
```

**Guarantees (properties tested):**
- Monotonic in proximity: for fixed `ball_y`, `assess(x1).level <= assess(x2).level`
  whenever `x1` is strictly farther from `own_goal_x` than `x2` (in the defended half).
- Central > wide: `assess(x, 34).level > assess(x, 55).level` at equal x.
- Zone ordering: `six_yard_box > inside_box > edge_of_box > outside_box`.
- Zero threat at distance: `dist >= 70` ⇒ `level == 0` (ball in opponent's half).

### Danger Reduction (defensive success)

```
FUNCTION danger_after_clearance(old_level, own_goal_x,
                                from_x, from_y, to_x, to_y,
                                success) -> float
  IF NOT success: RETURN old_level * 0.95      // scuffed — barely moved, stays dangerous
  moved_away := (HYPOT(to_x - own_goal_x, to_y - 34)
                 - HYPOT(from_x - own_goal_x, from_y - 34))
  relief     := CLAMP(1.0, 0.15, moved_away / 45.0)
  RETURN old_level * (1.0 - 0.75 * relief)     // effective clear reduces up to 75%
```

Tackle/interception/block wins apply a smaller flat relief: `old_level * 0.80` (the ball
is won but often still in the defensive half).

### Danger Classification

```
LOW      = D < 30     — ball comfortably away, defence reorganising
MODERATE = 30 ≤ D < 60 — ball in defensive third, pressure building
HIGH     = 60 ≤ D < 85 — ball in/near the box, clear the lines
CRITICAL = 85 ≤ D      — six-yard scramble, bodies on the line
```

---

## Clearance Model (Headed vs Foot)

Inside `DefensiveChain.generate(action_type="clearance")`:

```
FUNCTION clearance_type(ball_aerial) -> ("head" | "foot")
  RETURN "head" if ball_aerial else "foot"

FUNCTION clearance_success(defender, type, danger) -> float
  IF type == "head":
      base := aerial_dominance/100 * 0.55
             + technical.heading/100   * 0.25
             + mental.composure/100    * 0.10
             + defending.clearances/100 * 0.10
  ELSE:
      base := defending.clearances/100 * 0.40
             + defending.marking/100   * 0.15
             + mental.composure/100    * 0.25
             + mental.anticipation/100 * 0.20
  panic := 1.0 - 0.12 * (danger / 100)     // critical danger → sloppier
  RETURN CLAMP(0.15, 0.90, base * panic)
```

Destination (always AWAY from the defended goal):

- **Headed**: short and angled — `end_x` roughly 45–70m from the defended goal, `end_y`
  toward the nearest touchline. ~30% of effective headers go directly out for a throw-in
  (safe); failures drop to a second ball.
- **Foot**: longer and decisive — `end_x` roughly 50–78m away; under CRITICAL danger it
  becomes a big "hoof", more likely out of play (safe) but also more likely scuffed.
- Failures keep the ball in the danger zone (`end_x` within ~15m of the start) and may
  win the opponent a corner.

`CLEARANCE` events now carry `body_part` (`head`/`right_foot`/`left_foot`) and
`metadata = {clearance_type: "headed"|"foot", danger_before, danger_after, effective}`.

---

## Defender Coordination (`position_engine.defensive_block`)

```
FUNCTION defensive_block(team_name, ball_x, ball_y, own_goal_x, danger, minute,
                         pull_strength=0.5) -> None
  IF danger < 25 OR ball not in defended half: RETURN      // baseline behaviour
  line_depth := 8 + 14 * (1 - danger/100)    // deep block near goal at high danger
  behind     := 4.0 + line_depth             // metres goal-side of the ball
  block_x    := ball_x - behind * SIGN(own_goal_x - ball_x)
  block_y    := ball_y                       // ball-side lateral anchor

  FOR each defensive player (GK, CB, LB, RB, CDM):
      target_x := CLAMP(block_x within own half, 2..(own_goal_x±2))
      lateral  := ball_side factor: nearer players pull harder to block_y,
                  far-side players half as hard
      intensity := pull_strength * (0.30 + 0.70 * danger/100)
      current_x += (target_x - current_x) * intensity
      current_y += (target_y - current_y) * intensity * lateral
  Apply light line cohesion afterwards (existing _apply_line_cohesion).
```

This is additive and only fires when a team is **out of possession** with danger ≥ 25, so
in-possession drift and non-danger play are untouched (preservation).

---

## Danger-Scaled Action Selection (`_simulate_minute`)

```
base_weights   := {tackle: 0.35, interception: 0.30, clearance: 0.20, block: 0.15}
IF danger >= 85: weights := {tackle: 0.18, interception: 0.08, clearance: 0.44, block: 0.30}
ELIF danger >= 60: weights := {tackle: 0.22, interception: 0.14, clearance: 0.40, block: 0.24}
ELIF danger >= 30: weights := {tackle: 0.30, interception: 0.28, clearance: 0.24, block: 0.18}
ELSE: base_weights                                       // danger == 0 → unchanged baseline
```

`ball_aerial` is inferred from the last possession event being a cross / corner /
aerial duel / headed touch.

---

## Threat Engine Lifecycle (`match_engine.py`)

- **Init** (`__init__`): `self.threat = ThreatEngine()`, registered per team.
- **`_absorb_chain`**: for every event, after `last_ball_x/y` updates, call
  `self.threat.observe_event(event, home_team, away_team, minute)` → recomputes danger
  for the threatened team; applies relief on defensive wins; peaks + resets on goals.
- **`_simulate_minute`**: before the defensive contest, read
  `danger = self.threat.danger_at(defending_team, ctx_x, ctx_y)`, compute `ball_aerial`,
  pick danger-scaled weights, and pass `danger_level` / `ball_aerial` / `own_goal_x` /
  `position_engine` into `ChainDispatcher.defensive_action`.
- **`_run_minute`**: after `drift_minute`, for each team out of possession with
  `danger ≥ 25`, call `position_engine.defensive_block(...)`.
- **`MatchResult`**: carries `threat` so the exporter can render the danger history.

---

## Correctness Properties

Property 1: **Monotonic Danger** — _For any_ ball positions p1, p2 in the defended half
with p1 strictly farther from the defended goalpost xy than p2 at the same y, the danger
level at p2 SHALL be ≥ the danger level at p1.

Property 2: **Central > Wide** — _For any_ equal-distance ball positions, the central
position SHALL yield strictly higher danger than a wide position.

Property 3: **Zone Ordering** — _For any_ two ball positions in the same half, danger in
`six_yard_box` SHALL exceed danger in `inside_box`, which SHALL exceed `edge_of_box`,
which SHALL exceed `outside_box`/`deep`.

Property 4: **Zero-Threat Baseline** — _For any_ ball position ≥70m from the defended
goal, danger SHALL be 0 and defensive behaviour SHALL match the pre-feature baseline.

Property 5: **Headed vs Foot Selection** — _For any_ clearance where the ball is aerial
in the danger zone, `body_part="head"`; otherwise `body_part` is a foot.

Property 6: **Attribute Sensitivity** — Headed-clearance success SHALL rise with the
defender's `aerial_dominance`; foot-clearance success SHALL rise with defending
`clearances`.

Property 7: **Danger Relief** — _For any_ successful defensive win, danger SHALL fall;
_for any_ failed clearance, danger SHALL NOT fall materially (≤5%).

Property 8: **Defender Coordination** — _For any_ out-of-possession team with danger ≥ 25
and the ball in its defensive half, the defensive line's mean distance to the defended
goal SHALL decrease after `defensive_block` vs. before.

Property 9: **Preservation** — At danger == 0, `DefensiveChain` tackle/interception/block
probabilities and events SHALL be statistically unchanged from the pre-feature baseline.
