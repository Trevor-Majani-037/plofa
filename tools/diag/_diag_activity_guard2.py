"""NEGATIVE CONTROL for the 2026-10-19 locomotion/accounting changes.

A guard that has never been seen to fail is not a guard. Each mutation below
reintroduces one of the defects the new tests claim to pin, runs ONLY the test
that should catch it, and requires it to go RED. The files are restored from a
byte-exact in-memory snapshot captured immediately before the mutation (never a
stale backup on disk) and the restore is asserted by hash.

  M1  the block arrived branch hard-zeroes `moved` again (the freeze)
  M2  the dead phase stops opting in to `include_on_ball` (the on-ball freeze)
  M3  the per-minute overlap cap is removed (published time > clock)
  M4  the residue redistribution is removed (telemetry creates time)

Usage:  python tools/diag/_diag_activity_guard2.py
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

MUTATIONS = [
    (
        "M1 block freeze",
        "match_engine.py",
        """                nx, ny, moved = self._block_shuffle_target(
                    cx, cy, tx, ty, ang,
                    self._BLOCK_SHUFFLE_RATE, self._BLOCK_SHUFFLE_R_M, budget)""",
        """                nx, ny, moved = self._block_shuffle_target(
                    cx, cy, tx, ty, ang,
                    self._BLOCK_SHUFFLE_RATE, self._BLOCK_SHUFFLE_R_M, budget)
                moved = 0.0""",
        "tests/test_measured_activity.py::BlockLineIsHeldNotFrozenTests"
        "::test_the_live_block_branch_does_not_hard_zero_the_step",
    ),
    (
        "M2 dead-phase opt-out",
        "match_engine.py",
        "include_on_ball=self._DEAD_TIME_ON_BALL",
        "include_on_ball=False",
        "tests/test_measured_activity.py::DeadPhaseStillStepsTheOnBallManTests"
        "::test_the_dead_phase_opts_in_and_the_live_phase_does_not",
    ),
    (
        "M3 no overlap cap",
        "opta_analytics.py",
        "                    overlap_s = max(0.0, measured_total_s - 60.0)",
        "                    overlap_s = 0.0",
        "tests/test_measured_activity.py::TimeIsAUnionTests"
        "::test_overlap_is_scaled_back_into_the_minute",
    ),
    (
        "M4 no residue redistribution",
        "opta_analytics.py",
        "                        if abs(_k - 1.0) > 1e-12:",
        "                        if False:",
        "tests/test_measured_activity.py::TimeIsAUnionTests"
        "::test_the_residue_cannot_create_time",
    ),
]


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    failures = 0
    for label, rel, old, new, node in MUTATIONS:
        p = ROOT / rel
        before = p.read_bytes()
        h0 = digest(p)
        text = before.decode("utf-8")
        nl = "\r\n" if b"\r\n" in before else "\n"
        old = old.replace("\n", nl)
        new = new.replace("\n", nl)
        if text.count(old) != 1:
            print(f"[SKIP] {label}: anchor matched {text.count(old)} times")
            failures += 1
            continue
        p.write_bytes(text.replace(old, new, 1).encode("utf-8"))
        try:
            # require the mutant to still IMPORT: a syntax/collection error
            # counted as a caught mutation proves nothing about the assertion.
            imp = subprocess.run(
                [PY, "-c", "import match_engine, opta_analytics"],
                cwd=ROOT, capture_output=True, text=True)
            if imp.returncode != 0:
                print(f"[SKIP] {label}: mutant does not import\n{imp.stderr[-400:]}")
                failures += 1
                continue
            r = subprocess.run([PY, "-m", "pytest", "-q", node],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"[MISS] {label}: test stayed GREEN — not pinned")
                failures += 1
            else:
                print(f"[CAUGHT] {label}: {node.rsplit('::', 1)[-1]}")
        finally:
            p.write_bytes(before)
        restored = digest(p) == h0
        if not restored:
            print(f"[FATAL] {rel} was not restored byte-identically")
            return 2
    # baseline must be green afterwards
    r = subprocess.run([PY, "-m", "pytest", "-q",
                        "tests/test_measured_activity.py"],
                       cwd=ROOT, capture_output=True, text=True)
    print("-" * 70)
    print(r.stdout.strip().splitlines()[-1])
    if r.returncode != 0:
        print("[FATAL] baseline is not green after restore")
        return 2
    print("ALL CONTROLS PASSED" if failures == 0 else f"{failures} CONTROL(S) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
