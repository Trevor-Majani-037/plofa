"""NEGATIVE CONTROL: are these 2 preservation failures pre-existing, or mine?

`tests/test_preservation_properties.py` reports:
  FAILED test_realistic_shot_woodwork_and_rebound_preserved
  FAILED test_position_engine_drift_toward_home

AGENTS.md lists BOTH as documented pre-existing, but that bisect was run at the
`geometric_awareness` pass. "Documented" is not evidence that a LATER change
did not cause them -- the project's own rule.

So: reconstruct the PRE-REFACTOR `attacking_matrix.py` by reverse-substituting
the nine `W["..."]` lookups back to their literals, run the two failing nodes
3x per arm, then restore the file BYTE-EXACTLY from an in-memory backup.

`git stash` is unsafe here (a parallel session shares these files), so the
reverse substitution is done in memory and the restore is asserted by length and
hash, not assumed.

Run:  .venv\\Scripts\\python.exe _diag_weights_bisect.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys

TARGET = "attacking_matrix.py"
NODES = [
    "tests/test_preservation_properties.py::test_realistic_shot_woodwork_and_rebound_preserved",
    "tests/test_preservation_properties.py::test_position_engine_drift_toward_home",
]

# the pre-refactor literals, exactly as they were
REVERSE = {
    'W["w_progress"]': "0.30",
    'W["w_freedom"]': "0.45",
    'W["w_depth"]': "0.20",
    'W["w_same_flank"]': "0.06",
    'W["w_winger_flank"]': "0.12",
    'W["w_mid_halfspace"]': "0.06",
    'W["w_mid_ahead"]': "0.04",
    'W["w_mid_width"]': "0.04",
    'W["w_false_nine"]': "0.12",
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def run_nodes(tag):
    for node in NODES:
        for trial in range(3):
            r = subprocess.run(
                [sys.executable, "-m", "pytest", node, "-q", "--no-header", "-x"],
                capture_output=True, text=True,
            )
            status = "PASS" if r.returncode == 0 else "FAIL"
            print(f"  {tag:<8} {node.split('::')[1][:52]:<54} trial{trial+1} {status}",
                  flush=True)


def main():
    original = open(TARGET, "rb").read()
    src = original.decode("utf-8")

    # ── ARM B: current code (weights table live) ──────────────────────────
    print("  ARM B — current (weights table)")
    run_nodes("current")

    # ── ARM A: pre-refactor, reconstructed ───────────────────────────────
    reverted = src
    for k, v in REVERSE.items():
        reverted = reverted.replace(k, v)
    n = sum(src.count(k) for k in REVERSE)
    print(f"\n  reverse-substituted {n} lookups")
    if n == 0:
        print("  *** nothing to substitute -- the refactor is not present. ***")
        return
    open(TARGET, "wb").write(reverted.encode("utf-8"))
    try:
        print("  ARM A — pre-refactor (literals)")
        run_nodes("prerefact")
    finally:
        open(TARGET, "wb").write(original)
        back = open(TARGET, "rb").read()
        print(f"\n  restored: {sha(original)} == {sha(back)} -> "
              f"{'BYTE-EXACT' if back == original else '*** MISMATCH ***'}")

    print("\n  VERDICT: if both arms fail all 3 trials, pre-existing.")
    print("  If only ARM B fails, this refactor caused it.")


if __name__ == "__main__":
    main()