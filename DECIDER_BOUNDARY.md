# The Decider Boundary — why pass direction is hand-coded, and what to do about it

**Status: DELIBERATELY ACCEPTED (2026-09-30). Not a bug report. A decision record.**

The author has decided that the current hand-coded directional design is what
the coach wants to see. This document exists so that the finding is not lost,
so that nobody "discovers" it again in six months and mistakes it for a new
bug, and so that the revisit — if there ever is one — starts from measurements
rather than from scratch.

Read this before touching `_pass_destination` or `_best_forward`.

---

## The finding in one paragraph

The evolved on-ball brain (`NeuralDecisionBrain`, 24→32→32→10) does **not**
decide which direction a pass goes. It cannot. Its entire output is one of ten
intent *labels*. Which teammate receives the ball, and therefore which way the
ball travels, is chosen by a hand-written function dispatched on that label.
Pass direction in PLOFA is 100% hand-coded and 0% learned, and the fitness has
been paying for forward progress the whole time while the policy had no output
capable of claiming it.

---

## The code path, verified end to end

```
brain_integration.NeuralDecisionBrain.decide(...)
  ├─ probs                                  # the network's output: 10 values
  ├─ intent = _INTENT_BY_INDEX[chosen_idx]  # ← line 550, the ONLY learned choice
  └─ target = _find_target(...)             # ← line 562, hand-written
         ├─ SAFE_PASS        → _nearest_teammate
         ├─ RECYCLE          → _deepest_teammate
         ├─ PROGRESSIVE_PASS → _best_forward          # line 688
         ├─ THROUGH_BALL     → _best_forward          # line 692
         └─ SWITCH           → _wide_teammate
```

`brain_integration.py:676`, on `_find_target` itself:

> *"This is a lightweight geometric lookup, not a scoring system."*

Ten outputs. All ten are labels. None of them means *further forward*, *wider*,
*deeper*, or *more lateral*. The target coordinates are then produced by
`event_chain.PossessionChain._pass_destination`, whose directional terms
(`pos_fwd_bias` 0.55–0.80 by position, `style_dir` 0.82–1.18 by team style) are
likewise hand-written coefficients.

## The reward was correct the whole time

`surrogate_collect.py:154` scores a completed pass as

```python
score += 2.0 + min(pass_advance / 25.0, 1.5)
```

and `pass_advance` is signed and direction-normalised at `event_chain.py:2476`:

```python
pass_advance = end_px - x
if not attacks_right:
    pass_advance = -pass_advance
```

So a 25 m forward pass scores 3.5 and a 10 m backward pass scores 1.6. The
surrogate has explicitly paid for forward progress since 2026-09-11. The GA
maximised it faithfully. **The policy simply had no output that could claim
any of it.**

This is the same lesson as the v3 retrain ("the objective is the ceiling")
landing one level deeper. The objective was never the problem.

---

## The measurements that established it

Real roster (Oxton v Natrican, both from `PLOFA-2026-2027.xlsx`), 3 matches per
arm, each arm in a **separate process** — an A/B across two `simulate()` calls
in one process is invalid because the second inherits the first's module-level
brain caches.

| | deciders (today) | brain-only | real PL |
|---|---|---|---|
| events | 2866 | 2382 | — |
| passes | 808 | 641 | — |
| GK receptions | 37.7 | 25.0 | — |
| **median pass length** | **9.7 m** | **23.9 m** | 15–18 m |
| forward | 27.9% | 35.8% | 35–40% |
| square | 41.0% | 27.0% | 45–50% |
| backward | 31.1% | 37.2% | 10–15% |
| **median displacement** | **−0.2 m** | **−0.07 m** | — |
| progressive | 26.5% | 41.7% | 25–35% |
| ends in final third | 15.1% | 15.6% | 15–20% |

Four things to take from that table:

1. **Median displacement stays at ~0 in BOTH arms.** Switching the deciders off
   could not move it, because direction was never the brain's to move.
2. **"Progressive" became *long*, not *forward*** (9.7 m → 23.9 m, Δx still 0).
   The only lever the brain owns is `is_prog`, which feeds `_pass_distance`.
   Length is reachable through the output space; direction is not.
3. **The distribution got wider, not more biased** (fwd 27.9→35.8 *and*
   back 31.1→37.2). Removing a damper adds variance to a distribution whose
   mean is set by a lookup table.
4. **The deciders are a damper, not a source of direction.** They exist to hold
   pass length near real; without them the median overshoots to 23.9 m, well
   past the 15–18 m band. Checkpoint 32b's failure also reproduced: GK
   receptions −34%, events −17%.

---

## `POLICY_INTENT_AUTHORITY` — half-wired, now fixed

The switch already existed and was named exactly what the experiment needs. It
was wired to **five** sites (`event_chain.py` 1556, 1559, 1665, 1709, 1737),
all of which sit *before* the brain is called — and **not** to the two that
force the delivery class. So the previous "brain decides" test only ever
switched off the brain's choice of **receiver**, never its choice of **what
kind of pass to play**.

The whole fix is two conditions:

```python
if regression_mode is not None and not cls.POLICY_INTENT_AUTHORITY:
elif wide_combo_mode and not cls.POLICY_INTENT_AUTHORITY:
```

Default is `False`, so the shipped engine is byte-identical to before.

**Consequence for the record:** Checkpoint 32b's documented result in
`AGENTS.md` is a *partial* result. Anyone reading it would otherwise believe
this experiment had been run in full. It had not.

### Reproducing it

```
python _diag_brain_only.py --arm deciders --seed 31 --matches 3 --out d.json
python _diag_brain_only.py --arm brain    --seed 31 --matches 3 --out b.json
```

Two processes, always. Never both arms in one process.

---

## The three ways to revisit this, if you ever do

The choice is a design decision about the project's thesis, so it belongs to
the author. Recorded here so it does not have to be re-derived.

### Option 1 — a signed forwardness head (cheap, falsifiable)

Add two outputs to the network, −1..+1, feeding a weight in `_best_forward`
that trades forward gain against distance and risk. Roughly 20 lines and ~40
parameters. The existing surrogate already rewards it correctly, so the GA can
learn it immediately with no fitness change.

This is the recommended starting point **if** a revisit happens, because it is
falsifiable cheaply: *if the GA cannot discover a forward preference even when
the reward pays for it and it has an output to spend on it, that tells us
something important about the fitness or the surrogate* — and we would have
learned it for the price of a small experiment rather than a rewrite.

### Option 2 — a pointer head over teammates (the honest fix)

The network outputs a weighting per candidate receiver and picks the target
itself. This matches the project's thesis properly: the evolved policy would
then own *what to do* **and** *where to go*. Cost: a real architecture change,
a larger net, retraining from scratch, and every existing brain file in
`brains/` becomes invalid. Not a small change.

### Option 3 — accept it as a rule, and make it deliberate

Direction is a design decision, like the offside line or the rest-defence
invariant. Write a forward bias down on purpose, give it a name, a comment
explaining the football, and a test pinning the resulting distribution to a
real-football band. The team then has a *chosen* verticality instead of an
accidental symmetry.

This is a legitimate answer and was effectively chosen on 2026-09-30. The
unacceptable state is the current one: a rule nobody wrote on purpose that
produces a team with **zero net progress across ninety minutes**
(median displacement −0.2 m / −0.07 m, backward passes 31–37% against a real
10–15%).

---

## Related known gaps, recorded while in the area

Not part of this decision, but found while measuring and not yet fixed:

- **`pass_direction` metadata is uncorrelated with geometry.** The engine's own
  label cross-tabulates against raw coordinates as near-uniform
  (label "forward" → 125 forward / 123 square / 146 backward; label "backward"
  → 140 / 116 / 103), while `sideways` is correct (73 of 83). Marginal
  distributions match (365/371 labels vs 268/262 geometry), which suggests a
  stale or shuffled value rather than a wrong formula. **Untested hypothesis:
  it clusters within a possession or per player.** `event_chain.py:2482`
  already makes this exact argument for crosses and long passes — *"Data
  providers do not classify a delivery by intent… run the pure geometric
  detector over every pass"* — and `detect_cross` / `detect_long_pass` already
  exist. `pass_direction` was simply never converted. Whatever the cause, the
  exporter is stamping a field that is wrong roughly two-thirds of the time.
- **Synthetic training states have no net-forward asymmetry to learn from.**
  `brain_evolution.random_game_state` / `random_game_state_scoring` place
  teammates ahead in the attack direction correctly and the *sensors* are
  direction-normalised (`is_final_third`, `is_own_half`, `goal_dist_norm` all
  take `attacks_right`), so the net **can** represent a directional policy in
  principle. It is option 1 and 2 above, not the state generator, that is
  missing.
- **`POLICY_INTENT_AUTHORITY` is not reachable from the production runner.**
  It is a class attribute only; no runner sets it.

---

## The lesson worth keeping

Four separate times in one session, a plausible mechanism for a football
problem dissolved under direct measurement: a "speed ceiling" that 18 of 20
players clear, a "burst cancelled by the resustain rule" that was rare, a
"shape oscillating end to end" that was an un-normalised half-time flip, and a
"run layer re-tasking players six times a minute" that was twice.

The one that held was the one verified by reading the code path end to end.

Read the units from the source, not the field name. Normalise the frame before
comparing positions. And when a mechanism explains a symptom neatly, that is
reason to measure it, not reason to believe it.
