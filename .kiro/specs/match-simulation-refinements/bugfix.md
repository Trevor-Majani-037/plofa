# Bugfix Requirements Document

## Introduction

This bugfix addresses critical missing match flow mechanics in the PLOFA 26/27 football match simulation engine. The system currently simulates possession, shots, fouls, and cards, but lacks fundamental match restart mechanisms (throw-ins, goal kicks, offsides) and post-goal flow (celebrations, center restarts). Additionally, the shot selection logic allows unrealistic shooting decisions from impossible angles.

**Impact:** Without these mechanics, matches lack realism - balls go out of play with no restart, offside attacks count as valid, and goals lead directly to the next sequence with no celebration or proper kickoff. Players also shoot from positions where a real footballer would pass or cross instead.

---

## Bug Analysis

### Current Behavior (Defect)

**1. Unrealistic Shot Selection:**

1.1 WHEN a player is at position (105, 55) — 2 meters behind the goal line and 21 meters wide of center — THEN the system allows them to shoot directly at goal and potentially score

1.2 WHEN a player is at an acute angle (e.g., x > 100, y < 20 or y > 48) with narrow goal visibility THEN the system allows shooting with the same probability as from a central position

1.3 WHEN a wide player (LW/RW) is near the byline with a poor shooting angle THEN the system doesn't intelligently bias toward crossing/passing instead of shooting

**2. Missing Throw-In Mechanic:**

1.4 WHEN the ball goes out of bounds behind the goal line between the corner flags (not resulting in a goal) THEN the system does not detect this as a throw-in situation

1.5 WHEN a throw-in should occur THEN no THROW_IN event is generated and play continues unrealistically

**3. Missing Goal Kick Mechanic:**

1.6 WHEN the defending team sends the ball behind their own goal line (not resulting in a goal) THEN the system does not detect this as a goal kick situation

1.7 WHEN a goal kick should occur THEN no GOAL_KICK event is generated and the defending goalkeeper does not restart play

**4. Missing Offside Detection:**

1.8 WHEN an attacking player is ahead of the second-to-last defender in the attacking half at the moment the ball is played forward THEN the system does not detect this as offside

1.9 WHEN an offside offense occurs THEN play continues without stoppage and no OFFSIDE event is generated

1.10 WHEN an offside is called THEN no free kick restart is awarded to the defending team

**5. Missing Goal Celebration Sequence:**

1.11 WHEN a goal is scored THEN the match immediately proceeds to the next possession sequence without any celebration pause

1.12 WHEN a goal is scored THEN no time is added to the match clock for celebration duration

1.13 WHEN a goal is scored THEN no specific GOAL_CELEBRATION event appears in the timeline

**6. Incomplete Center Restart After Goals:**

1.14 WHEN a goal is scored and celebration completes THEN the conceding team receives kickoff possession but players are not properly reset to their own halves

1.15 WHEN the center restart occurs THEN the ball position (state.last_ball_x, state.last_ball_y) resets to center circle but player positions may remain displaced from pre-goal states

---

### Expected Behavior (Correct)

**1. Intelligent Shot Selection:**

2.1 WHEN a player is at position (105, 55) or any position behind the goal line (x ≥ 105) THEN the system SHALL prevent shooting and instead force a pass/cross decision or treat it as an impossible action

2.2 WHEN a player is at an acute angle (distance from goal line < 5m, y-offset from center > 15m) THEN the system SHALL apply heavy xG penalty through angle geometry AND reduce the probability of attempting a shot in favor of crossing/passing

2.3 WHEN a wide player (LW/RW/LB/RB) is near the byline (x > 95, y < 20 or y > 48) THEN the system SHALL bias action selection: 65% cross, 20% pass back, 10% dribble, 5% shot

**2. Throw-In Mechanic:**

2.4 WHEN a possession sequence results in the ball going out behind the goal line (y < 2 or y > 66, 0 < x < 105) via a defending team touch THEN the system SHALL generate a THROW_IN event and award possession to the attacking team

2.5 WHEN a throw-in is awarded THEN the system SHALL emit a THROW_IN event to the timeline with correct location coordinates (x at where ball went out, y at 0 or 68 touchline)

2.6 WHEN a throw-in is taken THEN play SHALL resume with a possession sequence starting from the throw-in location

**3. Goal Kick Mechanic:**

2.7 WHEN an attacking team sends the ball behind the defending team's goal line (not between the goal posts) THEN the system SHALL generate a GOAL_KICK event

2.8 WHEN a goal kick is awarded THEN the system SHALL emit a GOAL_KICK event with the defending team's goalkeeper as the primary actor

2.9 WHEN a goal kick is taken THEN play SHALL resume with a possession sequence starting from the goalkeeper's position (x ≈ 8-18, y ≈ 34)

**4. Offside Detection:**

2.10 WHEN an attacking player receives a forward pass while positioned ahead of the second-to-last defender in the attacking half (x > 52.5 for their team) THEN the system SHALL detect offside

2.11 WHEN an offside offense is detected THEN the system SHALL immediately stop play, emit an OFFSIDE event, and prevent any subsequent shot/chance from that attack chain

2.12 WHEN an offside is called THEN the system SHALL award a free kick restart to the defending team at the location where the offside occurred

**5. Goal Celebration Sequence:**

2.13 WHEN a goal is scored THEN the system SHALL pause match simulation for a random duration between 10-30 seconds

2.14 WHEN the celebration pause occurs THEN the system SHALL add the celebration duration to the match clock (e.g., goal at 67:23 + 18-second celebration = resume at 67:41)

2.15 WHEN the celebration occurs THEN the system SHALL emit a GOAL_CELEBRATION event to the timeline with duration metadata

**6. Proper Center Restart After Goals:**

2.16 WHEN a goal celebration completes THEN the system SHALL reset all player positions to their starting formation positions (calling PositionEngine to reset current_x/current_y to home_x/home_y for all players)

2.17 WHEN the center restart kickoff happens THEN the system SHALL ensure the conceding team starts with possession at the center circle (52.5, 34) and emit a KICKOFF event

2.18 WHEN the kickoff possession sequence begins THEN the system SHALL ensure all players are in their own halves (home team players with current_x < 52.5, away team with current_x > 52.5) before play resumes

---

### Unchanged Behavior (Regression Prevention)

**3. Existing Match Flow:**

3.1 WHEN a corner is won (existing causal corner system) THEN the system SHALL CONTINUE TO queue pending_corner_for and generate a CORNER_TAKEN event

3.2 WHEN a goal is scored from a non-offside attack THEN the system SHALL CONTINUE TO update score, emit GOAL event, shift momentum, and trigger the existing pending_kickoff_for mechanism

3.3 WHEN a foul is committed THEN the system SHALL CONTINUE TO generate FOUL_COMMITTED events and handle card probability without change

3.4 WHEN a red card is issued THEN the system SHALL CONTINUE TO reduce team to 10 players, apply possession/press penalties, and shift momentum

**4. Existing Shot Outcome Logic:**

3.5 WHEN a valid shot (from realistic position/angle) has xG calculated THEN the system SHALL CONTINUE TO apply body_part multipliers, pressure penalties, and situation adjustments

3.6 WHEN a shot is on target THEN the system SHALL CONTINUE TO call GoalkeeperEngine.evaluate_save() with the same parameters and logic

3.7 WHEN a shot hits the woodwork THEN the system SHALL CONTINUE TO handle rebound_in logic and corner awards

**5. Position Engine Spatial State:**

3.8 WHEN a player touches the ball THEN the system SHALL CONTINUE TO update their current_x/current_y via position_engine.record_touch()

3.9 WHEN a minute elapses THEN the system SHALL CONTINUE TO drift uninvolved players back toward home positions with phase/game-state modifiers

3.10 WHEN a substitution occurs THEN the system SHALL CONTINUE TO register the incoming player with PositionEngine.register_substitute()

**6. Event Timeline & Stamina:**

3.11 WHEN any event is generated THEN the system SHALL CONTINUE TO append it to self.timeline and drain stamina via SubstitutionController

3.12 WHEN a substitution is executed THEN the system SHALL CONTINUE TO emit SUBSTITUTION event, update active_players, and register with position_engine
