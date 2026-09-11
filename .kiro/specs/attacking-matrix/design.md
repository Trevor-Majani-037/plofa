# High-Density Attacking Matrix — Design

## Overview

Adds an **intelligent attacking decision layer** to the PLOFA 26/27 match engine. Every
ball carrier with live spatial state evaluates the pitch as a web of passing corridors,
a shooting window, and a set of strategically-valued teammates — then resolves one of
four decisions: **SHOOT**, **KEY_PASS** (create chance), **PROGRESSIVE_PASS**, or
**RECYCLE_PASS** — steered by match scenario and player DNA.

The work is spread across four pieces that mirror the existing Checkpoint 5-9 module
conventions:

- **`attacking_matrix.py` (NEW)** — the Attacking Decision Intelligence module: pure
  geometry → S_viability, L_clear, V_strategic, and the scenario-aware decision
  resolution.
- **`event_chain.py`** — `PossessionChain` evaluates the matrix on every touch (per-touch
  SHOOT decision + matrix-selected pass targets aiming at real receiver positions);
  `AttackChain._select_action_from_position` delegates its geometry to the matrix.
- **`match_engine.py`** — wiring: after absorbing a possession chain, if it decided
  SHOOT, dispatch `AttackChain` anchored at the carrier's (x, y) instead of rolling the
  independent shot_prob.
- **`chain_result`** — new hand-off fields on `ChainResult` (`shoot_decision`,
  `shoot_player`, `shoot_x/y`, `shoot_under_pressure`).

**Testing Approach:** property-based exploration of the three micro-calculations
(monotonicity in proximity, pressure tax, central > wide, lane blocking, strategic value
ordering), decision-resolution properties per scenario, integration properties that a
matrix SHOOT decision hands off to AttackChain, and preservation checks that possession
chains without a position engine are unchanged.

---

## Glossary

- **Ball Carrier (A₀)**: the player in possession at (x, y).
- **Target Goal Destination G**: the centre of the opponent goal mouth (105, 34) when
  attacking right, (0, 34) when attacking left.
- **Recycle Zone**: around the carrier — the region where possession is kept safe.
- **Close Network**: teammates within 15m of the carrier.
- **Far Network**: teammates beyond 15m of the carrier.
- **Pressure Boundary**: the radius of the closest closing outfield defender around the
  carrier.
- **Passing Corridor**: the line segment from the carrier to a teammate; blocked when an
  outfield defender sits within 1.2m of the segment.
- **S_viability**: the uncontested shot score ∈ [0, 1].
- **L_clear**: passing lane clearance ∈ [0, 1] for a single corridor.
- **V_strategic**: teammate strategic value ∈ [0, 1].
- **Scenario**: counter / low_block / build_up / balanced — derived from team style.

---

## Geometry Model (Formal)

### Constants

```
GOAL_CENTER_Y       = 34.0
GOAL_TOP_POST_Y     = 30.34
GOAL_BOTTOM_POST_Y  = 37.66
GOAL_HALF_WIDTH_M   = 3.66                       # 7.32m mouth
REFERENCE_ANGLE     = 2 * ATAN2(3.66, 11.0)      # ≈ 0.643 rad — angular width of the
                                                 # goal mouth seen from the penalty spot

CLOSE_NETWORK_M     = 15.0
FAR_NETWORK_M       = 30.0
PRESSURE_TAX_RANGE_M = 1.5
LANE_BLOCK_RADIUS_M = 1.2
FREE_RADIUS_M       = 3.0
```

`goal_x(attacks_right) = 105.0 if attacks_right else 0.0` (identical to
`BaseChain.goal_x`).

### S_viability — Uncontested Shot Score

```
FUNCTION nearest_defender_dist(x, y, def_players, position_engine) -> float | None
  RETURN MIN over outfield def_players of HYPOT(def_pos, (x, y))     // None if none

FUNCTION shot_score(carrier, x, y, def_players, position_engine, attacks_right) -> float
  dist         := HYPOT(goal_x - x, 34.0 - y)                     // m to goal centre
  distance_decay := 1 / (1 + (dist / 12.0) ** 1.8)                // 1 at 0m → ~0.16 at 30m

  nearest      := nearest_defender_dist(x, y, def_players, position_engine)
  IF nearest is None:        raw_tax := 1.0
  ELIF nearest < 1.5:        raw_tax := 0.30    // tackle imminent — shot rushed/blocked
  ELIF nearest < 3.5:        raw_tax := 0.60
  ELIF nearest < 8.0:        raw_tax := 0.85
  ELSE:                      raw_tax := 1.0
  composure    := carrier.dna.mental.composure / 100
  pressure_tax := raw_tax + (1 - raw_tax) * composure * 0.5   // composed shooters cope

  dx           := MAX(1.0, ABS(goal_x - x))
  ang_top      := ATAN2(y - GOAL_TOP_POST_Y, dx)
  ang_bot      := ATAN2(y - GOAL_BOTTOM_POST_Y, dx)
  angular_width := ABS(ang_top - ang_bot)                       // rad visible goal mouth
  angle_mult    := CLAMP(0.05, 1.0, angular_width / REFERENCE_ANGLE)

  RETURN distance_decay * pressure_tax * angle_mult
```

**Guarantees (properties tested):**
- Monotonic in proximity: for fixed y, pressure, and same half, `shot_score` never falls
  as the carrier moves closer to `goal_x`.
- Pressure tax: with a defender at 0.5m, `shot_score` < `shot_score` with the nearest
  defender at 10m (same geometry).
- Attribute sensitivity: at equal geometry under pressure, a high-composure carrier scores
  ≥ a low-composure carrier.
- Central > wide: at equal distance to goal, `shot_score(x, 34)` > `shot_score(x, 55)`.

### L_clear — Passing Lane Clearance

```
FUNCTION point_segment_dist(p, a, b) -> float
  // standard perpendicular distance from point p to segment [a, b]

FUNCTION lane_clearance(ax, ay, tx, ty, def_players, position_engine) -> float
  d_lane := MIN over outfield def_players of
                point_segment_dist(def_pos, (ax,ay), (tx,ty))    // ∞ if no defenders
  IF d_lane >= 3.0:          RETURN 1.0
  ELIF d_lane <= 1.2:        RETURN 0.0       // blocked corridor
  ELSE:                      RETURN (d_lane - 1.2) / 1.8
```

**Guarantees (properties tested):**
- A defender on the carrier→teammate line within 1.2m ⇒ L_clear = 0.
- A corridor clear of all defenders by ≥ 3m ⇒ L_clear = 1.
- Monotonic: L_clear is non-decreasing in d_lane.

### V_strategic — Teammate Strategic Value

```
FUNCTION strategic_value(carrier, teammate, ax, ay, def_players, position_engine,
                         attacks_right) -> float
  tx, ty     := position of teammate
  lane       := lane_clearance(ax, ay, tx, ty, def_players, position_engine)
  d_AG       := HYPOT(goal_x - ax, 34 - ay)      // carrier's distance to goal
  d_TG       := HYPOT(goal_x - tx, 34 - ty)      // teammate's distance to goal
  progress   := CLAMP(0, 1, 0.5 + (d_AG - d_TG) / 20.0)   // 0.5 neutral, 1.0 if 10m closer

  nearest_opp := MIN over outfield def_players of HYPOT(def_pos, teammate_pos)
  freedom     := 1.0 if nearest_opp > 3.0
                 else CLAMP(0.15, 1.0, (nearest_opp - 1.0) / 2.0)   // 1.0 at 3m, 0.15 at 1m
  depth       := CLAMP(0, 1, (tx - 35)/70) if attacks_right
                 else       CLAMP(0, 1, (35 - tx)/70)               // 0 halfway, 1 at goal

  RETURN lane * (0.45 * progress + 0.35 * freedom + 0.20 * depth)
```

**Guarantees (properties tested):**
- A free (unmarked) advanced teammate with an open lane has higher V_strategic than a
  marked deep teammate.
- V_strategic is non-decreasing in lane clearance and in freedom.

### Zones

```
FUNCTION network_zone(dist) -> "close" | "far"
  RETURN "close" if dist < 15.0 else "far"
```

---

## Decision Resolution

### Scenario derivation

```
FUNCTION scenario_for(profile) -> str
  style := profile.style.value if hasattr(profile, 'style') else "balanced"
  "fluid_counter" | "route_one" | "direct"                       -> "counter"
  "park_the_bus" | "ultra_defensive" | "defensive"               -> "low_block"
  "tiki_taka" | "structured_possession" | "possession"
  | "vertical_tiki_taka"                                          -> "build_up"
  otherwise                                                       -> "balanced"
```

### Scores

```
shot        := shot_score(carrier, x, y, def_players, position_engine, attacks_right)
dist_to_goal := HYPOT(goal_x - x, 34 - y)
under_pressure := nearest_defender_dist(x, y, ...) is not None AND < 1.5

FOR each teammate T (outfield, ≠ carrier):
    lane(T) := lane_clearance(x, y, tx, ty, ...)
    value(T) := strategic_value(carrier, T, x, y, def_players, position_engine, attacks_right)
    zone(T) := network_zone(HYPOT(tx - x, ty - y))
    track BEST close option (max value) and BEST far option (max value, requiring lane ≥ 0.6)
```

### Thresholds (attribute-driven)

```
shooting_quality := (finishing + long_shots) / 2 / 100            # 0..1
vision           := dna.mental.vision / 100
decisions        := dna.mental.decisions / 100
plays_safe       := dna.tendencies.plays_safe                     # 0..1
confidence       := (dna.form.confidence / 100) if form else 0.5

base_threshold   := { counter: 0.42, low_block: 0.45, build_up: 0.60, balanced: 0.50 }
shot_threshold   := base_threshold
                    - 0.12 * shooting_quality          # elite finishers shoot earlier
                    + 0.15 * plays_safe                # conservative carriers hold the ball
                    - 0.06 * (confidence - 0.5)        # confident players commit
                    + 0.06 * (0.5 - decisions)         # poor judgement = worse timing
```

### Priority cascade (deterministic given the scores)

```
1.  ELITE SHOT      IF dist_to_goal < 20 AND shot >= 0.50
                    AND NOT (best_far.lane >= 0.6 AND best_far.value >= shot)
                    THEN -> SHOOT

2.  COUNTER CHANCE  IF scenario == "counter" AND best_far.lane >= 0.6
                    AND best_far.value >= 0.35 AND (shot < 0.40 OR best_far.value >= shot * 0.9)
                    THEN -> KEY_PASS  (long diagonal into the channel)

3.  LOW-BLOCK SHOT  IF scenario == "low_block"
                    AND shot >= 0.42 AND dist_to_goal < 35
                    THEN -> SHOOT  (panic mode — packed box, take the half-chance)

4.  CLEAN BUILD-UP  IF scenario == "build_up" AND best_far.lane >= 0.6
                    AND best_far.value >= 0.50
                    AND (vision >= 0.65 OR best_far.value >= 0.65)
                    THEN -> KEY_PASS   (vision finds the line-breaker)

5.  PROGRESSIVE     IF best_close.value >= 0.40 AND best_close.progress >= 0.60
                    THEN -> PROGRESSIVE_PASS

6.  PANIC SHOT      IF under_pressure AND shot >= 0.30 AND dist_to_goal < 35
                    AND best_close.value < 0.35
                    THEN -> SHOOT   (defender closing, no short outlet)

7.  RECYCLE         THEN -> RECYCLE_PASS  (best close option, else nearest open teammate)
```

`decisions` sharpens the pick: when a borderline tier (2/3/4) is within 5% of the winning
score, a high-`decisions` carrier still takes the right tier; a low-`decisions` carrier
may drop to recycle (modelled deterministically as a small threshold nudge so it is
testable).

### AttackingDecision (return type)

```
@dataclass
class AttackingDecision:
    action: str                # SHOOT | KEY_PASS | PROGRESSIVE_PASS | RECYCLE_PASS
    shot_score: float
    target: str | None         # teammate name for passes
    target_lane_clearance: float
    target_strategic_value: float
    zone: str                  # recycle | close_network | far_network
    under_pressure: bool
    best_lane_clearance: float
    best_strategic_value: float
```

---

## Wiring

### ChainResult hand-off fields (event_chain.py)

```
shoot_decision: bool          = False
shoot_player: str             = ""
shoot_x: float                = 0.0
shoot_y: float                = 0.0
shoot_under_pressure: bool    = False
```

### PossessionChain per-touch evaluation

At the top of the MAIN ACTION step (before the carry/pass/dribble roll), when a position
engine is wired in:

```
dec := AttackingMatrix.decide(carrier, players, def_players, x, y,
                              attacks_right, scenario=scenario_for(team_profile),
                              position_engine=position_engine)

IF dec.action == "SHOOT":
    result.shoot_decision = True
    result.shoot_player   = carrier.name
    result.shoot_x, result.shoot_y = x, y
    result.shoot_under_pressure = dec.under_pressure
    BREAK                                   # sequence ends; MatchEngine hands off

ELIF dec.action is a pass action AND dec.target:
    receiver := player with name == dec.target          # matrix-selected target
    # pass aimed AT the receiver's live position (pass-skill error), NOT a random roll
    end_px, end_py := pass_destination_to_target(...)
    is_long/is_prog := from dec.action + receiver distance
    metadata += {"attacking_matrix": dec.action.lower(), "lane_clearance": ...}
```

The pass SUCCESS path (chemistry, marking, confidence, miscontrol, receipt, turnover) is
unchanged; only the receiver pick and destination aim change when the matrix fires.

When NO position engine is wired in, `decide` returns a fallback decision whose action is
never SHOOT and whose target is None → the existing `_pick_receiver` / `_pass_destination`
path runs unchanged (preservation 3.1).

### MatchEngine hand-off

In `_simulate_minute`, immediately after `_absorb_chain(poss_result)`:

```
IF poss_result.shoot_decision AND NOT poss_result.possession_lost:
    att_result := ChainDispatcher.attack(
        minute, attacking_team, defending_team,
        att_players, def_players, att_profile, def_profile, self.state,
        SituationType.OPEN_PLAY,
        position_engine=self.position_engine,
        context_x=poss_result.shoot_x,
        context_y=poss_result.shoot_y,
        attacks_right=attacks_right)
    IF self._absorb_chain(att_result, minute): BREAK
    # mark this sequence as "shot already taken" so the independent shot_prob
    # roll below does not double-fire
ELSE:
    < existing shot_prob / set-piece / attack dispatch unchanged >
```

`AttackChain.generate` already anchors the shot at `context_x/context_y` when the context
is in the attacking half (its `in_attacking_half` branch), so the matrix's SHOOT decision
becomes the shot's actual origin.

### `_select_action_from_position` delegation

`AttackChain._select_action_from_position(x, y, player_position)` keeps its signature and
return contract (`"shoot" | "pass" | "cross" | "dribble"`) but derives its geometric
rejections from the matrix's `angle_multiplier` / distance model:

- x ≥ 105 → "cross" if wide else "pass" (unchanged)
- angular width narrower than ~25% of reference (angle > ~70°) → "cross"/"pass"
- wide positions near the byline → weighted cross/pass/dribble (unchanged)
- otherwise → "shoot"

---

## Correctness Properties

Property 1: **Shot Score Monotonicity** — _For any_ two carrier positions p1, p2 in the
same attacking half with p2 strictly closer to `goal_x` at the same y and the same
defensive pressure, `shot_score(p2)` SHALL be ≥ `shot_score(p1)`.

Property 2: **Pressure Tax** — _For any_ identical shooting geometry, moving the nearest
outfield defender inside 1.5m SHALL strictly reduce S_viability vs. a defender ≥ 8m away.

Property 3: **Central > Wide** — _For any_ equal-distance shot, the central position SHALL
yield strictly higher S_viability than a wide position.

Property 4: **Attribute Sensitivity (Shooting)** — _For any_ fixed geometry under
pressure, a carrier with higher `composure` SHALL have ≥ S_viability; and the SHOOT
threshold SHALL fall as `finishing`/`long_shots` rise (elite finishers shoot earlier).

Property 5: **Lane Blocking** — _For any_ carrier→teammate corridor, an outfield defender
within 1.2m of the segment SHALL yield L_clear = 0; a corridor ≥ 3m from all defenders
SHALL yield L_clear = 1; L_clear SHALL be non-decreasing in distance-to-lane.

Property 6: **Strategic Value Ordering** — _For any_ two teammates T1 (free, advanced,
open lane) and T2 (marked, deep, blocked lane), `V_strategic(T1)` SHALL be > `V_strategic(T2)`.

Property 7: **Decision — Elite Shot** — _For any_ carrier inside 20m of the goal with
`shot ≥ 0.50` and no better far-network option, the decision SHALL be SHOOT.

Property 8: **Decision — Counter Chance** — _For any_ counter scenario with an open
far-network runner (`lane ≥ 0.6`, `value ≥ 0.35`) and a modest shot, the decision SHALL be
KEY_PASS (long diagonal into the channel) with strictly higher frequency than in build_up.

Property 9: **Decision — Low Block Panic** — _For any_ low_block scenario with the carrier
in range (`dist < 35`) and `shot ≥ 0.42`, the decision SHALL be SHOOT.

Property 10: **Decision — Recycle** — _For any_ carrier with no open far option, no
valuable close option, and no in-range shot, the decision SHALL be RECYCLE_PASS.

Property 11: **Integration — Shot Hand-off** — _For any_ PossessionChain run with a wired
position engine where the carrier reaches a shoot-eligible position, the ChainResult SHALL
carry `shoot_decision = True` with the carrier's (x, y), and a full match run SHALL
complete with matrix decisions appearing in the timeline and no double-fire with the
independent shot_prob path.

Property 12: **Preservation** — _For any_ PossessionChain run WITHOUT a position engine,
behavior SHALL be unchanged from the pre-feature baseline (no `shoot_decision`, same
`_pick_receiver`/`_pass_destination` path, same event distribution).
