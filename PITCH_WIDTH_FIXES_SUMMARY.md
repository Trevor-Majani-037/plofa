# Pitch Width Fixes - Implementation Summary

## Problem Confirmed
Analysis of 10 matches showed consistent center bias:
- **70.8% center concentration** (expected: 50-60%)
- **5.6% touchline usage** (expected: 8-12%)
- **13.3m width spread** (expected: 14-18m)

## Root Cause Discovered
The **CRITICAL ISSUE** was the winger/fullback flank preservation caps at y=22m and y=46m. These caps were designed to prevent wingers from drifting central, but they also **prevented wingers from reaching the touchline zones** (0-10m and 58-68m).

With pitch width of 68m:
- Left touchline zone: 0-10m
- Old LW cap: 22m (12m away from touchline zone!)
- Right touchline zone: 58-68m  
- Old RW cap: 46m (12m away from touchline zone!)

**Wingers could never get truly wide** - the "flank preservation" was actually limiting width!

## Fixes Implemented

### 1. Widen Pass Lateral Range ✓
**File:** `event_chain.py` line 3733
- Changed from: `vert_range = 4 + vert_skill * 16` (max 20m)
- Changed to: `vert_range = 6 + vert_skill * 22` (max 28m)
- **Impact:** Skilled passers can now make 25-30m cross-field switches

### 2. Increase Receive Window ✓
**File:** `position_engine.py` line 92
- Changed from: `ELLIPSE_SIGMA_ACROSS = 9.0`
- Changed to: `ELLIPSE_SIGMA_ACROSS = 13.0`
- **Impact:** Players can drift ±13m laterally to receive (was ±9m)

### 3. Widen Carry Lateral Variance ✓
**Files:** `event_chain.py` lines 1348, 3879
- Changed from: `vert_range = 4 + ball_control * 8` (max 12m)
- Changed to: `vert_range = 6 + ball_control * 14` (max 20m)
- **Impact:** Diagonal carries can stretch the pitch more

### 4. Widen Micro-Carry Variance ✓
**File:** `event_chain.py` line 914
- Changed from: `end_cy = y + (0.5 - random.random()) * 6` (±3m)
- Changed to: `end_cy = y + (0.5 - random.random()) * 10` (±5m)
- **Impact:** Small touches between actions can spread wider (happens 55% of the time)

### 5. **CRITICAL:** Relax Winger Flank Caps ✓
**File:** `event_chain.py` line 3760

**Left Winger:**
- Changed from: Cap at y=22.0m
- Changed to: Cap at y=12.0m
- **Impact:** Can now reach 0-10m touchline zone

**Right Winger:**
- Changed from: Cap at y=46.0m
- Changed to: Cap at y=56.0m
- **Impact:** Can now reach 58-68m touchline zone

### 6. Relax Fullback Flank Caps ✓
**File:** `event_chain.py` line 3740
- Changed LB/RB caps from 22/46 to 12/56 (matching winger changes)
- **Impact:** Overlapping fullbacks can also reach touchlines

## What to Expect

After fixes, you should see:
- ✅ More play near touchlines (target: 8-12% vs 5.6% before)
- ✅ Reduced center concentration (target: 55-62% vs 70.8% before)
- ✅ Better width spread (target: 15-17m vs 13.2m before)
- ✅ More realistic cross-field switches
- ✅ Wingers actually hugging touchlines like real football
- ✅ Better pitch stretching overall

## Testing

Run a few matches and analyze with:
```bash
.venv/Scripts/python.exe analyze_width_distribution.py
```

The script will show:
- Center vs wing distribution
- Touchline usage percentage
- Width spread (std deviation)
- Diagnosis compared to expectations

## Notes

- All changes preserve DNA scaling - skilled players benefit more
- Winger flank preservation logic still prevents central drift
- Changes are conservative but targeted at the real blockers
- The 22/46 caps were the hidden constraint preventing realistic wing play

## Baseline for Comparison

**Before changes (10 matches average):**
- Center: 70.8%
- Touchlines: 5.6%
- Spread: 13.2m

**Target after changes:**
- Center: 55-62%
- Touchlines: 8-12%
- Spread: 15-17m
