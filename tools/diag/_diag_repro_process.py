"""Cross-process reproducibility: ONE match per process, digest out. (2026-10-04)

`_diag_seed_repro.py` answered the IN-PROCESS question and falsified its own
premise: two `simulate()` calls in one process with the same seed produce
byte-identical timelines, so module-level caches were never the in-process
culprit and `np.random.default_rng` (called 531,424 times per match) does not
reach the event timeline.

But the claim AGENTS.md actually carries EVIDENCE for is CROSS-process:

  "Two fresh processes with the same random.seed(4242) and PYTHONHASHSEED=0
   produce different event timelines (verified 2026-09-25: 2873 events / 1-4
   vs 3170 events / 1-2)."

That is a different question and it is not settled by the in-process result.
This probe runs exactly ONE match and prints a digest, so the caller can run
it several times in separate processes and compare. It never runs two matches
in one process -- that is the question the other probe answered.

FLAGS, because the variables must be separated rather than assumed:
  --seed N          football seed (default 777, the value that reproduced
                    perfectly in-process)
  --no-python-seed  skip random.seed() entirely
  --no-np-seed      skip np.random.seed()
  --pythonhashseed  echo the interpreter's PYTHONHASHSEED back in the output,
                    so a run that cannot be attributed to the seed can at
                    least be attributed to the hash seed

Prints a single DIGEST line last, so a caller can grep it and compare without
parsing anything else.
"""
import argparse
import hashlib
import os
import random
import sys

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--no-python-seed", action="store_true")
ap.add_argument("--no-np-seed", action="store_true")
ap.add_argument("--pythonhashseed", action="store_true")
A = ap.parse_args()

if not A.no_python_seed:
    random.seed(A.seed)
if not A.no_np_seed:
    try:
        import numpy as np
        np.random.seed(A.seed)
    except Exception:
        pass

if A.pythonhashseed:
    print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', '<unset>')}")

from _diag_chance_coords import build_pair  # noqa: E402

res = build_pair("Oxton", "Natrican").simulate()

rows = ["|".join([
    e.event_type.name,
    getattr(e, "player", "") or "",
    getattr(e, "secondary_player", "") or "",
    f"{getattr(e, 'location_x', 0.0):.1f}",
    f"{getattr(e, 'location_y', 0.0):.1f}",
    f"{getattr(e, 'minute', 0.0):.2f}",
]) for e in res.timeline]

digest = hashlib.sha1("\n".join(rows).encode()).hexdigest()[:12]
goals = len([e for e in res.timeline if e.event_type.name == "GOAL"])
print(f"events={len(rows)} goals={goals} "
      f"pyhash={os.environ.get('PYTHONHASHSEED', '<unset>')} "
      f"pyseed={'skipped' if A.no_python_seed else A.seed} "
      f"npseed={'skipped' if A.no_np_seed else A.seed}")
print(f"DIGEST {digest}")