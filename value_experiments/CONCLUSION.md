# Consequence-Learning Value Prototype — Decisive Experiment

Date: 2026-09-12. 6 real matches, live neural XI (T1 brains), seed 21,
opponents = Probe FC (balanced) vs Rival FC (fluid_counter), templates
rotated base/CAM/CF. Read-only; zero engine edits.

## What was built
- `value_model.py` — `ValueModel`: linear value function V(s) via numpy
  ridge regression over an interpretable feature map (21 surviving sensors
  + 5 physics interactions + 11-position one-hot = 37 features). Closed
  form, no ML frameworks, inspectable coefficients.
- `consequence_probe.py` — offline collector: pairs every
  `NeuralDecisionBrain.decide` snapshot (the sensor state the brain
  actually saw) with the possession EPISODE outcome that followed it
  (goal / engine xG / shots / turnovers), then
  `run_experiment(...)` compares predictors:
  - learned V(s)          (out-of-sample CV)
  - production v1 surrogate `brains/surrogate_pos.json` expected_success
  - intent-mean baseline
  on the same decisions against the same real consequence R.

## Credit assignment (the point)
The v1 surrogate labels a decision by its IMMEDIATE execution event
("did THIS touch connect?" — hand-authored weights, no temporality). The
prototype instead attributes each decision the whole possession's outcome
(3×goal + engine xG + 0.2×shots − 0.6×turnovers, all engine-measured), so
an early progressive pass shares credit for the goal it built.

## Results (3217 decisions, 2586 episodes, 4.4 touches/episode)
Episode anatomy — why prediction is hard:
- payoff episodes (shot or goal): **5.0%**
- whose episodes  2...
- goal episodes: **0.5%**, turnover episodes: **63.2%**
- mean episode R = **−0.446** (turnover-mass-dominated target)

Predictors of real possession outcome (same decisions):
- value model V(s):  OOS rho **0.071** (R² −1.83, Pearson 0.033)
- surrogate v1:      rho **0.048** (Pearson 0.026)
- intent baseline:   rho **0.049** (Pearson 0.080)

Everything is essentially at noise level → **neither the production
surrogate nor intent alone carries any possession-level signal**. This
confirms the audit's core claim: the current learning objective optimises
immediate event success, not consequences.

But the learned value IS structurally real, not random:
- value rises toward the opponent goal: rho **0.37** (forward-progress and
  goal-proximity)
- top standardized weights are football-sane: closer to goal (s14) ↑,
  nearer defender (s4) ↓, final-third crowding (FT*NEARDEF) ↑,
  CAM positive
- payoff view: model's top-decile decisions catch payoff at **2.8% vs
  1.5% base** (~1.9× lift); OOS rank-correlation on the payoff indicator
  0.077
- V-delta direction demo: within episodes, when model V rose the payoff
  share was 2.5% (315 eps) vs 1.5% when V fell (341 eps) — the RIGHT
  direction, tiny signal

Per-position in-sample rho (sanity, NOT evidence of generalisation):
CF 0.54, LW 0.39, CAM 0.36, RW 0.36, LB 0.25, GK 0.21, CDM 0.17, CB
0.14, CM 0.08, ST 0.02.

## Verdict
Consequences are PAYABLE in principle (model learns football structure),
but a single-state linear V cannot beat sampling noise when 63% of
episodes are turnovers and only 0.5% are goals. The objective shape
matters more than the model: any V2 value objective must
(a) situation-sample/episode-weight scoring situations (goal_bias-like),
(b) reduce the turnover-mass dominance (or normalise per-episode),
(c) use episode/chain returns, not immediate-event labels — and only then
is a value model (or a surrogate trained on consequences) worth fitting.

## Artifacts
- `value_experiments/consequence_dataset.json` — 3217 records
  (sensors, intent, position, player, minute, R, goals, xg, shots, turnovers)
- `value_experiments/value_experiment_report.json` — full report incl.
  per-position table, physics coefficients, payoff view, V-delta demo

## Reproduce
```
PYTHONPATH=. .venv\Scripts\python.exe consequence_probe.py --matches 6 --seed 21 --out value_experiments
```