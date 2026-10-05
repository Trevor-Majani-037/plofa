r"""
_diag_dna_guard.py — negative control for tests/test_dna_awareness_shots.py.

A guard that has never been seen to fail is not a guard. This restores each
defect in turn from a BYTE-EXACT backup, runs the real test file, and asserts
that the specific test which claims to cover that defect goes RED. The file is
restored byte-identically afterwards and the restore is itself asserted.

Four mutations, one per claim:

  M1  remove `geometric_awareness` from `_build_mental` -> the attribute returns
      to the 50.0 dataclass default. Expect the VARIATION tests to fail.
  M2  set the default band to (55, 74), the value measurement rejected, and
      which made the >= 55 gate vacuous. Expect the gate/straddle test to fail.
  M3  restore `dna.mental.composure` / `dna.technical.finishing` in
      `_shot_on_target_prob`. Expect both behavioural tests AND the AST pin to
      fail.
  M4  make `effective_composure` a copy of `effective_dribbling`'s formula
      (drop fatigue). Expect the composure-formula test to fail.

Run: .venv\Scripts\python.exe _diag_dna_guard.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
TEST = os.path.join("tests", "test_dna_awareness_shots.py")

DNA = os.path.join(HERE, "player_dna.py")
CHAIN = os.path.join(HERE, "event_chain.py")

# (label, file, exact original text, replacement text, tests expected to fail)
MUTATIONS = [
    (
        "M1  geometric_awareness not generated",
        DNA,
        "            geometric_awareness = cls._attr(arch, "
        "\"mental.geometric_awareness\", (50, 70), mental_age),",
        "            geometric_awareness = 50.0,  # MUTANT",
        ["test_it_is_not_the_dataclass_default_any_more",
         "test_every_outfielder_position_gets_a_value",
         "test_awareness_scales_the_steering_factors_continuously",
         "test_cam_pocket_bonus_is_now_non_zero_for_some_players"],
    ),
    (
        "M1b geometric_awareness given a constant of its own (e.g. 60.0)",
        DNA,
        "            geometric_awareness = cls._attr(arch, "
        "\"mental.geometric_awareness\", (50, 70), mental_age),",
        "            geometric_awareness = 60.0,  # MUTANT",
        ["test_the_default_band_is_the_house_mental_default"],
    ),
    (
        "M2  default band (55, 74) — the measured-wrong value",
        DNA,
        "            geometric_awareness = cls._attr(arch, "
        "\"mental.geometric_awareness\", (50, 70), mental_age),",
        "            geometric_awareness = cls._attr(arch, "
        "\"mental.geometric_awareness\", (55, 74), mental_age),",
        ["test_outfielders_still_clear_the_gates_but_not_by_construction",
         "test_the_default_band_is_the_house_mental_default"],
    ),
    (
        "M3  shot path reads raw attributes again",
        CHAIN,
        "        comp = shooter.dna.effective_composure / 100.0\n"
        "        fin  = shooter.dna.effective_finishing / 100.0",
        "        comp = shooter.dna.mental.composure / 100.0\n"
        "        fin  = shooter.dna.technical.finishing / 100.0  # MUTANT",
        ["test_the_shot_probability_reacts_to_form",
         "test_the_shot_probability_reacts_to_fatigue",
         "test_the_shot_path_does_not_read_the_raw_attributes"],
    ),
    (
        "M4  effective_composure drops fatigue",
        DNA,
        "        return self.mental.composure * self.form.form_multiplier * "
        "self.form.fatigue_multiplier * self.live_performance_mult",
        "        return self.mental.composure * self.form.form_multiplier * "
        "self.live_performance_mult  # MUTANT",
        ["test_effective_composure_exists_and_matches_the_documented_formula"],
    ),
]


def run_tests() -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", TEST, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=HERE, capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def failed_test_names(output: str) -> set[str]:
    names = set()
    for line in output.splitlines():
        if line.startswith("FAILED ") or "::" in line and line.startswith("FAILED"):
            body = line.split(" ", 1)[1] if line.startswith("FAILED ") else line
            names.add(body.split("::")[-1].split(" ")[0].strip())
    return names


def main() -> int:
    baseline_rc, baseline_out = run_tests()
    print("=" * 78)
    print("BASELINE — the unmutated tree must be GREEN for any of this to mean anything")
    print("=" * 78)
    print(f"exit {baseline_rc}")
    tail = [ln for ln in baseline_out.splitlines() if "passed" in ln or "failed" in ln]
    for ln in tail[-3:]:
        print("  " + ln)
    if baseline_rc != 0:
        print("\nBASELINE IS RED — stop. A negative control against a failing")
        print("baseline cannot distinguish 'the guard works' from 'everything fails'.")
        return 2
    print()

    backup_dir = tempfile.mkdtemp(prefix="dna_guard_")
    backups = {}
    for label, path, _, _, _ in MUTATIONS:
        key = os.path.basename(path)
        if key not in backups:
            dst = os.path.join(backup_dir, key)
            shutil.copy2(path, dst)
            backups[key] = (path, dst)
    originals = {k: open(v[1], "rb").read() for k, v in backups.items()}

    all_ok = True
    try:
        for label, path, find, repl, expected in MUTATIONS:
            key = os.path.basename(path)
            src_path, backup_path = backups[key]
            current = open(backup_path, "r", encoding="utf-8").read()
            if find not in current:
                print(f"!! {label}: ANCHOR TEXT NOT FOUND — the file moved on.")
                print("   Not running this mutation; an anchor that does not match")
                print("   would silently mutate nothing and report PASS.")
                all_ok = False
                print()
                continue
            open(path, "w", encoding="utf-8").write(current.replace(find, repl, 1))

            rc, out = run_tests()
            failed = failed_test_names(out)
            caught = [t for t in expected if t in failed]
            missed = [t for t in expected if t not in failed]

            print("=" * 78)
            print(label)
            print("=" * 78)
            print(f"  exit {rc}   failing tests: {len(failed)}")
            for t in caught:
                print(f"    RED  (correct)  {t}")
            for t in missed:
                print(f"    !!   MISSED      {t}   <- this test does not cover it")
                all_ok = False
            if not caught:
                print("    !!   NOTHING WENT RED — the tests do not detect this defect")
                all_ok = False
            # restore before the next mutation
            open(path, "wb").write(originals[key])
            print()

        # Final assertion: the tree is byte-identical to where we started.
        print("=" * 78)
        print("RESTORE")
        print("=" * 78)
        for key, (path, _backup) in backups.items():
            after = open(path, "rb").read()
            same = after == originals[key]
            print(f"  {key:<18} byte-identical: {same}  ({len(after)} bytes)")
            if not same:
                all_ok = False
        rc, out = run_tests()
        print(f"  post-restore test run exit {rc}")
        if rc != 0:
            all_ok = False
    finally:
        for key, (path, _backup) in backups.items():
            open(path, "wb").write(originals[key])
        shutil.rmtree(backup_dir, ignore_errors=True)

    print()
    print("=" * 78)
    if all_ok:
        print("NEGATIVE CONTROL PASSED — every mutation was caught, tree restored.")
    else:
        print("NEGATIVE CONTROL FAILED — see the MISSED lines above.")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
