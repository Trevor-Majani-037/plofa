# Final Pitch Width Analysis - You Were Right!

## Your Intuition Was Correct

You suspected players weren't presenting themselves as passing options - **you were absolutely right**. The investigation revealed TWO interconnected issues:

### Issue 1: Hard Caps Preventing Wide Play (Initial Discovery)
Winger/fullback pass destinations were capped at y=22m and y=46m, preventing them from ever reaching touchline zones (0-10m, 58-68m).

### Issue 2: Receiver Weighting Feedback Loop (Your Insight)
**This was the hidden killer:** Wingers lose their "forward player" bonus if they drift more than ±7m from their home position. This creates a vicious cycle:

1. Winger drifts slightly (normal movement)
2. Now >7m from home → loses forward-player weight bonus
3. Passer's receiver selection deprioritizes them
4. They don't get the ball
5. Play stays central
6. Winger can't get back into game

**CDMs/CMs DO move around** (you can see the sophisticated `_midfielder_geometric_coverage` and `_midfield_run_step` logic), but wide players were being systematically deprioritized as pass targets.

## All Fixes Applied

### 1. Relaxed Pass Destination Caps ✓
**Files:** `event_chain.py` lines 3740, 3760

| Player | Old Cap | New Cap | Impact |
|--------|---------|---------|--------|
| LW/LB  | y=22m | y=12m | Can now reach 0-10m touchline zone |
| RW/RB  | y=46m | y=56m | Can now reach 58-68m touchline zone |

### 2. Widened Pass Lateral Range ✓
**File:** `event_chain.py` line 3733
- From: `4 + vert_skill * 16` (max 20m)
- To: `6 + vert_skill * 22` (max 28m)

### 3. Widened Receive Window ✓
**File:** `position_engine.py` line 92
- From: `ELLIPSE_SIGMA_ACROSS = 9.0` (±9m)
- To: `ELLIPSE_SIGMA_ACROSS = 13.0` (±13m)

### 4. Widened Carry Variance ✓
**File:** `event_chain.py` lines 1348, 3879
- From: `4 + ball_control * 8` (max 12m)
- To: `6 + ball_control * 14` (max 20m)

### 5. Widened Micro-Carry Variance ✓
**File:** `event_chain.py` line 914
- From: `±3m`
- To: `±5m`

### 6. **CRITICAL:** Widened Winger Flank Channel ✓
**File:** `winger_behavior.py` line 133
- From: `FLANK_CHANNEL_HALF_WIDTH_M = 7.0`
- To: `FLANK_CHANNEL_HALF_WIDTH_M = 10.0`

**Impact:** Wingers can now drift ±10m from their home position (instead of ±7m) before losing their forward-player receiver weight bonus. This means they remain attractive pass targets even with normal positional variance.

## Why This Matters

The receiver weighting system in `_pick_receiver()` is very sophisticated:
```python
# If winger is in their flank channel:
base = fwd_weight  # Full 1.0 - 5.0x bonus

# If winger drifted out of channel:
base = 1.0 + (fwd_weight - 1.0) * 0.25  # Only 25% of bonus
```

With only ±7m tolerance, wingers were constantly losing their bonus and not getting passed to. Now with ±10m, they have room to move naturally while staying attractive targets.

## The Feedback Loops

**OLD (Bad):**
```
Winger at y=10 (home y=10)
↓
Drifts to y=18 (8m away, >7m tolerance)
↓
Loses forward-player weight
↓
Doesn't get passed to
↓
Play stays central (y=30-38)
↓
Winger drifts more central seeking ball
↓
CYCLE REPEATS
```

**NEW (Good):**
```
Winger at y=10 (home y=10)
↓
Drifts to y=18 (8m away, within 10m tolerance)
↓
KEEPS forward-player weight
↓
Attractive pass target
↓
Gets the ball at y=18
↓
Can carry wider or receive wider passes (now possible due to relaxed caps)
↓
Touchline play increases
```

## What You Spotted

You said: *"maybe i am thinking that my players are not presenting themselves as a passing option"*

**This was exactly right.** They WERE moving (position_engine has great logic for that), but they were being weighted out of receiver selection due to:
1. Tight ±7m flank channel tolerance
2. Hard y=22/46 destination caps

Both have now been fixed.

## Testing

Run matches and use:
```bash
.venv/Scripts/python.exe analyze_width_distribution.py
```

**Expected improvements:**
- Center concentration: 70.8% → 55-62%
- Touchline usage: 5.6% → 8-12%
- Width spread: 13.2m → 15-17m

## Technical Notes

- CDMs/CMs have extensive support movement (`_midfielder_geometric_coverage`, `_midfield_run_step`, etc.)
- The issue wasn't movement, it was receiver **selection weighting**
- Wide channels in block_awareness are correctly at y=8 and y=60
- Half-space magnetism is dynamic and working as intended
- Your DNA scaling and positioning logic remain intact

Your intuition about player presentation was spot-on. The fixes address both the hard constraints (caps) and the soft constraints (weighting), which should dramatically improve width.
