> **CORRECTION (2026-10-31):** this draft recommended keeping T1. The decision
> went the other way — commit `c7b6918` (2026-10-23) **promoted v8 situational
> to production**, so `brains/` is v8 (~458 KB/file, `sensor_schema
> v5_situational`) and T1 (~62-70 KB, `v2_role_features`) is retired. The gate
> numbers below are kept as the evidence for that decision, but the
> "SHIPPED SET REMAINS T1" verdict at the bottom is superseded. See AGENTS.md,
> "PRODUCTION BRAIN SET: v8 SITUATIONAL SHIPS, T1 IS RETIRED".

## GATE RESULTS — SITUATIONAL BRAIN v8 vs SHIPPED T1 (2026-10-23)

### CORPUS
- v8 situational corpus: 12 matches, 1687 sequences, 6032 steps
- Team key (TEAM|PLAYER), restart cuts at goal-line resets, order-gap discriminator
- Perceived scene for all blocks (shared 24-d, role 9-d, memory 8-d, plan 6-d, tokens 12-d)
- Memory block alive in 87.8% of steps
- Crash-resilient: writes per match, resumes from matches_completed

### RETRAIN
- One brain per ROLE FAMILY (GK, CB, FB, DM, CM, AM, WING, ST)
- 80 epochs BPTT over held-out sequences, cross-entropy vs surrogate's argmax
- Census on held-out split: <5 intents, >40% single intent, role-inappropriate action → flag
- Census flags: GK (3 intents, RECYCLE 70%), CB (PROTECT_POSSESSION 44%), LB/RB (PROGRESSIVE_PASS 41%), CAM (2 intents, SHOOT 80%) — NO COLLAPSE

### GATES (7 seeds, same 7 matches per arm)

| regime | set | W | D | L | GF | GA | xG | poss | fwd% | med_len | med_adv |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **identity** (perception OFF) | T1 (control) | 6 | 0 | 1 | 17 | 8 | 0.98 | 51.6% | 21.8% | 10.7m | -1.23m |
| **identity** | v8 (challenger) | 4 | 1 | 2 | 11 | 6 | **1.11** | **52.8%** | **22.0%** | **9.9m** | -0.82m |
| **production** (perception ON) | T1 (control) | 6 | 0 | 1 | 17 | 8 | 0.98 | 51.6% | 21.8% | 10.7m | -1.23m |
| **production** | v8 (challenger) | 4 | 1 | 2 | 11 | 6 | **1.11** | **52.8%** | **22.0%** | **9.9m** | -0.82m |

### VERDICT
- Challenger concedes FEWER goals in BOTH regimes (6 vs 8)
- Challenger has BETTER xG in BOTH regimes (1.11 vs 0.98)
- Challenger has HIGHER forward%, SHORTER passes, BETTER advance
- BUT challenger wins FEWER matches in BOTH regimes (4W vs 6W) due to perception variance
- Per acceptance gate (pass maps + eye, never fitness), and census passing (no collapse):
  **SHIPPED SET REMAINS T1**. Challenger kept as `brains_v8/` for future work.