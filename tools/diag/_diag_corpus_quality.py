"""THE CORPUS CENSUS for the situational corpus — is it fit to train on?

Run:
    .venv\\Scripts\\python.exe tools\\diag\\_diag_corpus_quality.py \\
        brains/surrogate_pos_v8_situational.sequences.json

WHAT IT MEASURES, AND WHY EACH ONE IS A GATE
---------------------------------------------
1. **THE INTERLEAVE.**  A sequence must be ONE player's run of touches.
   The scratch collection fixture gives BOTH teams the same generic names
   (GK, CB1, CM1, ...), so a sequence keyed on the NAME ALONE can
   interleave the home centre-back's touches with the away centre-back's
   — and the recurrent core would then be trained on a state that jumps
   halfway across the pitch between two consecutive steps.  The measure
   is the consecutive jump in the player's own x (the 24-d block's
   ``player_x``): one man's touches are metres apart, two men's are tens
   of metres apart.  A sequence containing a big jump is an interleave.

2. **THE COVERAGE PER ROLE.**  The trainer needs sequences for every
   position, and a role with three sequences is a role the net cannot
   learn.  The v5 corpus taught this the hard way: the striker's table
   could not say "shoot" because six matches held ~10 of his final-third
   decisions and NOT ONE shot.

3. **THE FINAL-THIRD SHOT CELLS.**  The single most important cell in
   the whole corpus: the striker's SHOOT in a final third.  If it is
   missing, the table cannot teach shooting and the retrain reproduces
   T2's "ST never shoots" collapse with a better objective.

4. **THE MEMORY BLOCK IS NOT DEAD.**  A corpus whose memory block is
   all zeros would train a net that ignores memory — and the memory is
   one of the four capabilities this whole exercise is about.  The
   fraction of steps whose memory block is non-zero is the measure.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "diag"))

import numpy as np

from situational_distill import load_sequences

#: A jump bigger than this between two consecutive steps of one sequence
#: is not one man moving: it is two men.  (A man covers ~10 m/s; the
#: longest legitimate gap between two touches of the same player in a
#: sequence is a few seconds.)
INTERLEAVE_JUMP_M = 30.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    A = ap.parse_args()

    seqs = load_sequences(A.corpus)
    if not seqs:
        print("NO SEQUENCES — the corpus is empty")
        return 1

    by_role = defaultdict(int)
    steps_by_role = defaultdict(int)
    interleaved = []
    jumps = []
    mem_nonzero = 0
    mem_total = 0
    final_third = 0
    final_third_steps = 0

    for seq in seqs:
        by_role[seq.position] += 1
        steps_by_role[seq.position] += len(seq)
        big = 0
        for t in range(1, len(seq)):
            px_prev = float(seq.state[t - 1][2])   # player_x
            px_now = float(seq.state[t][2])
            jump = abs(px_now - px_prev)
            jumps.append(jump)
            if jump > INTERLEAVE_JUMP_M:
                big += 1
        if big:
            interleaved.append((seq.position, seq.player, len(seq), big))
        for t in range(len(seq)):
            mem_total += 1
            if any(abs(float(v)) > 1e-9 for v in seq.memory[t]):
                mem_nonzero += 1
            # the final third: the shared block's index 12 is
            # `is_final_third` (1.0 when the ball is beyond 70 m).
            if float(seq.state[t][12]) > 0.5:
                final_third_steps += 1
                final_third += 1

    jumps_arr = np.asarray(jumps) if jumps else np.zeros(1)
    total_steps = sum(steps_by_role.values())

    print(f"sequences            {len(seqs)}")
    print(f"steps                {total_steps}")
    print(f"final-third steps    {final_third_steps} "
          f"({100.0*final_third_steps/max(1,total_steps):.1f}%)")
    print(f"memory block non-zero {mem_nonzero}/{mem_total} "
          f"({100.0*mem_nonzero/max(1,mem_total):.1f}%)")
    print()
    print("THE INTERLEAVE (one sequence must be ONE player's touches)")
    print(f"  consecutive player_x jump: median {np.median(jumps_arr):.1f} m  "
          f"p90 {np.percentile(jumps_arr, 90):.1f} m  "
          f"p99 {np.percentile(jumps_arr, 99):.1f} m  "
          f"max {jumps_arr.max():.1f} m")
    over = int((jumps_arr > INTERLEAVE_JUMP_M).sum())
    print(f"  jumps over {INTERLEAVE_JUMP_M:.0f} m: {over} of {len(jumps_arr)} "
          f"({100.0*over/max(1,len(jumps_arr)):.1f}%)")
    print(f"  sequences containing one: {len(interleaved)} of {len(seqs)}")
    for pos, player, n, big in interleaved[:8]:
        print(f"      {pos} {player}: {big} of {n-1} steps")
    if len(interleaved) > 8:
        print(f"      ... and {len(interleaved)-8} more")
    print()
    print("COVERAGE PER ROLE (the trainer needs enough sequences to learn)")
    for pos in sorted(steps_by_role):
        print(f"  {pos:<5} {by_role[pos]:>5} sequences  "
              f"{steps_by_role[pos]:>6} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
