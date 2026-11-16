"""Micro-benchmark: what does ONE brain decision actually cost?

Answers the core question without a full match. Two layers matter:

  1. the forward pass (numpy matmuls)
  2. the SENSOR EXTRACTION that feeds it -- feature building is usually the
     expensive half, and it is what "call the brain every tick" multiplies.

Projection: calls = 22 players * 5400 s * Hz

Run: py -3 _brain_microbench.py
"""
import time
import os
import numpy as np

import football_brain as fb

RESULTS = []


def bench(label, fn, n=20000):
    for _ in range(1000):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    us = dt / n * 1e6
    RESULTS.append((label, us))
    print(f"{label:<40} {us:9.3f} us/call")
    return us


s24 = np.random.rand(25).astype(np.float64)
s24b = np.random.rand(24).astype(np.float64)

# ── forward passes, using the REAL shipped brains ──────────────────────
ob = fb.OffBallBrain.load(os.path.join("brains_offball", "CM.json"))
print("OffBallBrain(CM) input_dim:", ob.input_dim)
bench("OffBallBrain.forward (CM.json)", lambda: ob.forward(s24))

try:
    tp = fb.TeamPressBrain.load(os.path.join("brains_team", "XI.json"))
    bench("TeamPressBrain.forward (XI.json)", lambda: tp.forward(s24b))
except Exception as e:
    print("TeamPressBrain load failed:", e)

try:
    dab = fb.DefensiveActionBrain.load(os.path.join("brains_def", "ACTION.json"))
    bench("DefensiveActionBrain.forward", lambda: dab.forward(s24b))
    bench("DefensiveActionBrain.predict", lambda: dab.predict(s24b))
except Exception as e:
    print("DefensiveActionBrain load failed:", e)

# ── sensor extraction cost (the hidden half) ───────────────────────────
print("\n--- sensor-extraction helpers (need an engine; skipped here) ---")

# ── projection ────────────────────────────────────────────────────────
print("\n--- PROJECTION: ONE 90-MINUTE MATCH (inference only) ---")
MATCH_S, PLAYERS = 5400.0, 22
print(f"{'brain':<40} {'1 Hz':>9} {'3 Hz':>9} {'10 Hz':>9}")
for label, us in RESULTS:
    cells = []
    for hz in (1, 3, 10):
        cells.append(PLAYERS * MATCH_S * hz * us / 1e6)
    print(f"{label:<40} "
          f"{cells[0]:8.1f}s {cells[1]:8.1f}s {cells[2]:8.1f}s")
