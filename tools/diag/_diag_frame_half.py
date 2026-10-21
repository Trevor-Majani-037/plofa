"""IS THE RAW FRAME STABLE ACROSS HALVES?  (the shape numbers depend on it)

THE QUESTION
------------
The export's frame is documented as "home attacks right, always", and the
raw coordinates are the engine's own (`team_attacks_right` is set once per
team at `initialize_team` and never flipped at half time).  But the
export's frame note records an OPEN question: on a live match the away
side's passes sit mostly at high x, which should not happen.

The shape measurement reads raw x, so it needs the answer: if the frame
FLIPS at half time, the "median x" is a mixture of two frames and every
depth number in it is meaningless.

This runs one match and prints the median x per half, for both teams, plus
the share of the events beyond the halfway line.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_frame_half.py
"""
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "gates"))

from validate_neural_xl import _run_neural  # noqa: E402


def main() -> int:
    brains = sys.argv[1] if len(sys.argv) > 1 else "brains_v7"
    seed = 21
    random.seed(seed)
    result, _f = _run_neural(brains, seed, "balanced", "fluid_counter")

    per = {}
    for e in result.timeline:
        if e.location_x is None:
            continue
        half = 1 if e.minute <= 45 else 2
        per.setdefault((e.team, half), []).append(float(e.location_x))
    print(f"one match: {result.score_str}   (home = {result.config.home_team})")
    print()
    print(f"{'team':<12}{'half':<7}{'events':>8}{'median x':>11}"
          f"{'beyond 52.5':>14}")
    for (team, half), xs in sorted(per.items()):
        xs.sort()
        beyond = 100.0 * sum(1 for x in xs if x > 52.5) / len(xs)
        print(f"{team:<12}{half:<7}{len(xs):>8}{statistics.median(xs):>11.1f}"
              f"{beyond:>13.1f}%")
    print()
    print("A frame that FLIPS at half time would show the two halves of the")
    print("SAME team mirrored about 52.5 (i.e. medians summing to ~105).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
