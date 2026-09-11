# PLOFA Pitch Width Investigation
**Issue:** Play is overly concentrated in the center of the pitch (69.5% avg) with poor touchline usage (5.7% avg)

## Quantitative Analysis

Analyzed 5 matches from `plofa_output/`:

| Metric | Actual | Expected | Status |
|--------|--------|----------|--------|
| Center concentration (18-50m) | 69.5% | 50-60% | ⚠️ HIGH |
| Touchline usage (0-10m, 58-68m) | 5.7% | 8-12% | ⚠️ LOW |
| Width spread (std deviation) | 13.3m | 14-18m | ⚠️ NARROW |

### Worst Case Example
**Hartwell City vs Thornfield United MD1:**
- 78.2% center concentration
- 4.9% touchline usage  
- 12.0m width spread

## Root Causes Identified

### 1. **Pass Lateral Range Too Narrow** (`event_chain.py` line ~3728)
```python
vert_range = 4 + vert_skill * 16  # max 20m lateral spread
end_py = y + (0.5 - random.random()) * vert_range
```
**Problem:** Even skilled passers (vert_skill=1.0) only vary passes by ±10m laterally. Real football has 25-30m switches.

### 2. **Receive Window Too Narrow** (`position_engine.py` line 92)
```python
ELLIPSE_SIGMA_ACROSS: float = 9.0
```
**Problem:** Players only drift ±9m laterally to receive. Wide players can't exploit touchlines.

### 3. **Carry Lateral Variance Too Small** (`event_chain.py` line 3871 & 1348)
```python
# In _generate_carry:
vert_range = 4 + (player.dna.technical.ball_control / 100) * 8  # max 12m

# In main sequence:
vert_range = 4 + (last_player.dna.technical.ball_control / 100) * 8
new_y = y + (0.5 - random.random()) * vert_range
```
**Problem:** Max lateral carry variance is ±6m, preventing pitch-stretching runs.

### 4. **Winger Flank Caps May Be Too Restrictive** (`event_chain.py` line ~3752)
```python
if player.position in ("LW", "RW"):
    if y < 22.0:
        end_py = min(end_py, 22.0)  # Prevents going < 22m
    elif y > 46.0:
        end_py = max(end_py, 44.0)  # Prevents going > 46m
```
**Problem:** These preserve width but may prevent wingers from getting truly wide (touchlines are at y=0 and y=68).

## Proposed Fixes

### Priority 1: Widen Pass Lateral Range
**File:** `event_chain.py` line ~3728
```python
# OLD:
vert_range = 4 + vert_skill * 16  # max 20m

# NEW:
vert_range = 6 + vert_skill * 22  # max 28m for skilled passers
```
**Impact:** Allows 25-30m cross-field switches for skilled playmakers.

### Priority 2: Increase Receive Window
**File:** `position_engine.py` line 92
```python
# OLD:
ELLIPSE_SIGMA_ACROSS: float = 9.0

# NEW:
ELLIPSE_SIGMA_ACROSS: float = 13.0
```
**Impact:** Players can drift ±13m to receive, allowing wider positioning.

### Priority 3: Widen Carry Lateral Variance
**File:** `event_chain.py`

**Location 1** (line ~3871 - `_generate_carry`):
```python
# OLD:
vert_range = 4 + (player.dna.technical.ball_control / 100) * 8  # max 12m

# NEW:
vert_range = 6 + (player.dna.technical.ball_control / 100) * 14  # max 20m
```

**Location 2** (line ~1348 - main possession sequence):
```python
# OLD:
vert_range = 4 + (last_player.dna.technical.ball_control / 100) * 8

# NEW:
vert_range = 6 + (last_player.dna.technical.ball_control / 100) * 14
```

**Impact:** Diagonal runs can stretch the pitch more realistically.

### Priority 4: Relax Winger Flank Caps (**CRITICAL - THIS WAS THE MAIN BLOCKER**)
**File:** `event_chain.py` line ~3752
```python
# OLD:
if y < 22.0:
    end_py = min(end_py, 22.0)  # Capped at 22m - can't reach touchlines!
elif y > 46.0:
    end_py = max(end_py, 46.0)  # Capped at 46m - can't reach touchlines!

# NEW:
if y < 26.0:
    end_py = min(end_py, 12.0)  # Can now reach 0-10m touchline zone
elif y > 42.0:
    end_py = max(end_py, 56.0)  # Can now reach 58-68m touchline zone
```
**Impact:** THIS WAS THE KEY ISSUE - old caps at 22/46 prevented wingers from ever reaching touchline zones (0-10m, 58-68m). Now they can stretch truly wide.

## Implementation Plan

1. **Phase 1:** Apply fixes 1-3 (pass range, receive window, carry variance)
2. **Test:** Run 3-5 matches and analyze with `analyze_width_distribution.py`
3. **Phase 2:** Only if touchline usage still < 7%, consider relaxing winger caps
4. **Verify:** Ensure changes don't break existing functionality (winger positioning, flank preservation)

## Expected Outcomes

After fixes:
- Center concentration: 55-62% (down from 69.5%)
- Touchline usage: 8-11% (up from 5.7%)
- Width spread: 15-17m (up from 13.3m)
- More realistic wing play and cross-field switches
- Better pitch stretching in open play

## Testing Command
```bash
.venv/Scripts/python.exe analyze_width_distribution.py
```

## Notes

- Changes preserve existing DNA scaling (skilled players get more variance)
- Winger flank preservation logic (CHECKPOINT 18, 21, 31) remains intact
- No changes to home positions or formation structure needed
- Conservative increases to avoid over-correction
