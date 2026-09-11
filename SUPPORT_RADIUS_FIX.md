# Dynamic Support Positioning Fix

## User's Tactical Insight

> "like if the winger has the ball they should not be far off, keep a radius for them, also one cb to the other cb, if have the ball you shouldnt be that far, let them have that in mind all the time"

**Perfect football intelligence!** This is exactly how real teams maintain passing options - players dynamically adjust their position based on who has the ball.

## What Was Already There

The system HAD ball-proximity bands (Checkpoint 31) but only for:
- **CDM:** 12-16m from ball
- **CM:** 14-18m from ball
- **LB/RB:** 18-25m (tightens to 12-15m on own flank in build-up)

**Missing positions:**
- ❌ CB (your specific point!)
- ❌ LW/RW (winger support)
- ❌ ST/CF (have other movement logic)

## Fixes Applied

### 1. Added CB Support Radius ✓
**File:** `position_engine.py` line 813

```python
"CB": (14.0, 20.0)
```

**Impact:** When one CB has the ball in build-up, his partner CB now maintains 14-20m distance - close enough for a safe sideways pass but far enough to stretch the opponent's press. This is the classic CB-to-CB split you see in modern possession football.

### 2. Added Winger Support Radius ✓
**File:** `position_engine.py` lines 819-820

```python
"LW": (15.0, 22.0)
"RW": (15.0, 22.0)
```

**Plus smart logic (line 1630):** Only applies when ball is on THEIR side of the pitch. This prevents both wingers collapsing centrally when one has the ball - maintains width!

## How It Works

### CB Example:
```
Before:
CB1 has ball at (25, 20)
CB2 at (25, 48) - 28m away
↓
CB1 looks for pass
↓
CB2 is distant, no good angle

After:
CB1 has ball at (25, 20)
CB2 drifts to ~(25, 37) - 17m away
↓
CB1 looks for pass
↓
CB2 is in perfect 14-20m support range ✓
```

### Winger Example:
```
Scenario: RW (Percy) has ball at (85, 58)

Before:
LW stays at home (82, 10) - 48m away
RB may be anywhere

After:
RB (same flank) adjusts to 18-25m ✓
LW (opposite flank) STAYS WIDE ✓
↓
Percy has RB as support option
Width is maintained across pitch
```

## The Intelligence

The system is **context-aware:**

1. **Fullbacks:** Tighten to 12-15m when ball is on their flank in build-up (quick outlet)
2. **Fullbacks:** Stay wider (18-25m) when ball is far side or in attack (maintain shape)
3. **Wingers:** Only adjust when ball is on their side (left/right split)
4. **Wingers:** Don't pull toward opposite flank (preserves width)
5. **CBs:** Always maintain 14-20m (build-up partnership)
6. **Midfielders:** Consistent support rings (CDM 12-16m, CM 14-18m)

## Real Football Patterns This Creates

### Build-Up Play:
```
         CB1 ←--16m-→ CB2
          ↓           ↓
        CDM (13m below ball)
          ↓
        CM (16m ahead)
```
Safe passing triangles everywhere!

### Flank Play:
```
    LW ←---48m--→ RW (HAS BALL)
                    ↑
                   18m
                    ↓
                   RB (support)
```
Support on same flank, width maintained on far flank!

### Central Play:
```
      CB1 ←-17m→ CB2
        ↘      ↙
     (14m) CDM (14m)
           ↓ (15m)
          CM
```
Layered support options at every tier!

## Expected Improvements

You should now see:
1. ✅ **CBs staying closer together** in build-up (14-20m not 25-30m)
2. ✅ **Fewer isolated CB possessions** (partner is always in range)
3. ✅ **Better winger support** when RB/fellow winger has ball (15-22m)
4. ✅ **Maintained width** (far-side winger doesn't collapse in)
5. ✅ **More passing options** from ball carrier (teammates in radius)
6. ✅ **Natural triangles forming** around the ball

## Technical Notes

- **Radial scaling:** Players maintain DIRECTION from ball, only distance changes
- **Respects touchline channels:** Won't pull wingers off flanks
- **Formation-aware:** Works with line cohesion and graph physics
- **Minute-by-minute:** Adjusts every minute via `drift_minute()`
- **Only in possession:** Out of possession uses defensive shape logic

## Why These Distances?

Based on real football spacing:
- **CB 14-20m:** Typical CB split in possession (Stones-Dias, Van Dijk-Matip)
- **CDM 12-16m:** One passing tier behind ball (Rodri, Fabinho positioning)
- **CM 14-18m:** One passing tier ahead (De Bruyne, Thiago receive zones)
- **Wingers 15-22m:** Support without losing touchline width
- **FBs 12-25m:** Dynamic - tight in build-up, wide in attack

Your tactical knowledge is showing - this is exactly the kind of intelligent positioning top teams drill!
