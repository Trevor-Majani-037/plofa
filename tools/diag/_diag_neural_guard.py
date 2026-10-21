"""NEGATIVE CONTROL for tests/test_threat_reward.py and
tests/test_brain_schema_alignment.py.

A guard that has never been seen to fail is not a guard.  This reintroduces the
defects those files exist for, one at a time, and requires the RIGHT test class
to go red -- not merely "something failed".

Eight mutations, across four files:

  M1  threat_reward.py   reverse the potential difference (PHI(start)-PHI(end))
                         -> the verticality ordering tests
  M2  surrogate_collect  FitnessSurrogate.prior back to 0.5
                         -> the neutral-prior tests
  M3  threat_reward.py   EXECUTION_TYPES narrowed to {"PASS"} (the live bug:
                         progressive passes, switches, through balls, crosses
                         and dribbles were never paired)
                         -> the collector-pairing tests
  M4  brain_evolution    remove the signed->non-negative re-base
                         -> the "a negative mean does not clamp to zero" test
  M5  brain_evolution    collapse_penalties defaulted True (the recorded cause
                         of T2's second collapsed optimum)
                         -> the off-by-default test
  M6  role_features      drop the "a name this build no longer emits" raise
                         -> the schema-raise test
  M7  role_features      drop the declared-count-vs-architecture check
                         -> the count-mismatch tests
  M8  role_features      silence the legacy truncation warning
                         -> the warn-once tests

Every mutant must still COMPILE.  A syntax error counted as a caught mutation
proves nothing about which assertion pins the defect -- `_diag_distance_guard.py`
M2 and `_diag_deference_guard.py` both made that mistake, the latter by writing
LF anchors against a CRLF file so every mutation silently became a no-op.  So
anchors here are escaped literals joined with an explicit ``\\r?\\n``.

Each file is restored from an in-memory snapshot taken immediately before its
mutation and verified by sha256.  A backup is only valid for the state it was
taken from.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_neural_guard.py
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TESTS = [
    "tests/test_threat_reward.py",
    "tests/test_brain_schema_alignment.py",
    "tests/test_fitness_balance.py",
]


def _pat(*lines: str) -> str:
    """An exact multi-line anchor, tolerant of the file's own line endings."""
    return r"\r?\n".join(re.escape(l) for l in lines)


# (file, label, pattern, replacement, expected failing test CLASS)
MUTATIONS = [
    ("threat_reward.py",
     "M1 potential difference reversed (start - end)",
     _pat("            return _phi(grid, end, attacks_right) - "
          "_phi(grid, start, attacks_right)"),
     "            return _phi(grid, start, attacks_right) - "
     "_phi(grid, end, attacks_right)",
     "VerticalityIsTheOptimumTests"),

    ("surrogate_collect.py",
     "M2 FitnessSurrogate.prior back to 0.5",
     _pat("    prior: float = 0.0"),
     "    prior: float = 0.5",
     "SurrogateCarriesTheRewardDefinitionTests"),

    ("threat_reward.py",
     "M3 EXECUTION_TYPES narrowed to {'PASS'} (the live collector bug)",
     r"_MOVER_TYPES \| _LOSS_TYPES \| _SHOT_ON_TYPES \| _SHOT_OFF_TYPES"
     r"\s*\| _GOAL_TYPES \| _MISS_TYPES",
     '{"PASS"}',
     "CollectorPairingTests"),

    ("brain_evolution.py",
     "M4 signed->non-negative re-base removed",
     _pat("        base = 0.5 + 0.5 * max(-1.0, min(1.0, mean_reward))"),
     "        base = mean_reward",
     "TheFitnessFunctionSurvivesASignedRewardTests"),

    ("brain_evolution.py",
     "M5 collapse_penalties defaulted True",
     _pat("    collapse_penalties: bool = False,"),
     "    collapse_penalties: bool = True,",
     "TheFitnessFunctionSurvivesASignedRewardTests"),

    ("role_features.py",
     "M6 'a name this build no longer emits' raise removed",
     _pat("        missing = [n for n in want if n not in cur]",
          "        if missing:",
          "            raise RoleSchemaError(",
          "                f\"{brain_label or 'brain'} was trained on role "
          "feature(s) \"",
          "                f\"{missing} that {family} no longer emits. "
          "Known: {cur}. \"",
          "                \"Removing or renaming a role feature is a "
          "breaking schema \"",
          "                \"change; add the feature back, or retrain the "
          "brain.\"",
          "            )",
          "        idx = [cur.index(n) for n in want]"),
     "        idx = [cur.index(n) for n in want if n in cur]",
     "ByNameSelectionTests"),

    ("role_features.py",
     "M7 declared-count-vs-architecture check removed",
     _pat("        if len(want) != tail_width:"),
     "        if False:",
     "ByNameSelectionTests"),

    ("role_features.py",
     "M8 legacy truncation warning silenced",
     _pat("    if key not in _ALIGN_WARNED:"),
     "    if False and key not in _ALIGN_WARNED:",
     "LegacyPathWarnsTests"),

    # ── the forward-option input (added 2026-10-20) ──────────────────────
    ("role_features.py",
     "M9 FB's forward option becomes the DISTANCE sensor it replaced "
     "(overlap_option)",
     _pat("    short_cut = _clamp(near_tm / 40.0) if near_tm is not None "
          "else 0.0",
          "",
          "    _, pass_lane_open = _forward_options(x, y, tms_n, defs, pe, ar)"),
     "    short_cut = _clamp(near_tm / 40.0) if near_tm is not None else 0.0"
     "\n\n    pass_lane_open = overlap_option",
     "TheWideRolesCanSeeAForwardOptionTests"),

    ("role_features.py",
     "M10 the midfield menu recomputes the lane instead of sharing "
     "_forward_options",
     _pat("    fwd_tms, pass_lane_open = _forward_options(x, y, tms_n, defs, "
          "pe, ar)"),
     "    _fwd = [t for t in tms_n\n"
     "            if _prog(_pos(pe, t.name, (x, y))[0], x, ar) > 4.0]\n"
     "    _opens = [_openness(*_pos(pe, t.name, (x, y)), defs, pe)\n"
     "              for t in _fwd]\n"
     "    fwd_tms = _fwd\n"
     "    pass_lane_open = max(_opens) if _opens else 0.0",
     "TheForwardOptionHasOneDefinitionTests"),

    ("role_features.py",
     "M11 the menu declares the feature but the FB block never appends it",
     _pat("    return [wide_marker_dist, one_v_one_space, inside_channel, "
          "overlap_option,",
          "            cross_zone, own_line_height, short_cut, pass_lane_open]"),
     "    return [wide_marker_dist, one_v_one_space, inside_channel, "
     "overlap_option,\n"
     "            cross_zone, own_line_height, short_cut]",
     "TheWideRolesCanSeeAForwardOptionTests"),

    # ── the FITNESS BALANCE (2026-10-20) ────────────────────────────────
    ("brain_evolution.py",
     "M12 the confidence term is multiplicative again, so it outranks the "
     "reward by ~50x and the GA optimises PEAKEDNESS instead of football",
     _pat("        fitness = (base",
          "                   + CONFIDENCE_GUARD_W * (mean_conf - 0.5)",
          "                   + DIVERSITY_GUARD_W * bdiv)"),
     "        fitness = base * (0.5 + 0.5 * mean_conf)",
     "FitnessBalanceTests"),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_tests() -> tuple:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-p",
         "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, text=True,
        env={"PYTHONHASHSEED": "0", "PYTHONPATH": str(REPO),
             "PATH": os.environ.get("PATH", ""),
             "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    originals = {}
    for rel in {m[0] for m in MUTATIONS}:
        originals[rel] = (REPO / rel).read_bytes()
    hashes = {k: sha(v) for k, v in originals.items()}

    rc, out = run_tests()
    if rc != 0:
        print("BASELINE IS NOT GREEN -- aborting, nothing measured.")
        print(out[-2500:])
        return 1
    print("baseline: green")
    print()

    failures = 0
    try:
        for rel, label, pattern, repl, expected in MUTATIONS:
            target = REPO / rel
            snapshot = target.read_bytes()
            body = snapshot.decode("utf-8")
            hits = len(re.findall(pattern, body))
            if hits != 1:
                print(f"  [{rel}] {label}\n    ANCHOR MATCHED {hits} TIMES -- "
                      f"the mutation is a no-op, aborting this arm.")
                failures += 1
                target.write_bytes(snapshot)
                continue

            mutant = re.sub(pattern, repl, body, count=1)
            try:
                compile(mutant, str(target), "exec")
            except SyntaxError as exc:
                print(f"  [{rel}] {label}\n    MUTANT DOES NOT COMPILE "
                      f"({exc}) -- counting as a MISS, not a catch.")
                failures += 1
                target.write_bytes(snapshot)
                continue

            target.write_bytes(mutant.encode("utf-8"))
            rc, out = run_tests()
            caught = rc != 0 and expected in out
            if caught:
                names = re.findall(r"FAILED (\S+?)(?:\s+-|$)", out, re.M)
                short = [n.split("::", 1)[1] if "::" in n else n for n in names]
                print(f"  [{rel}] {label}\n    CAUGHT: "
                      f"{', '.join(dict.fromkeys(short)) or '?'}")
            else:
                print(f"  [{rel}] {label}\n    MISS (rc={rc}); expected "
                      f"{expected} to fail.")
                print("    " + (out.strip().splitlines() or [""])[-1])
                failures += 1

            target.write_bytes(snapshot)
            assert sha(target.read_bytes()) == sha(snapshot), "RESTORE FAILED"
    finally:
        for rel, blob in originals.items():
            (REPO / rel).write_bytes(blob)

    for rel, blob in originals.items():
        assert sha((REPO / rel).read_bytes()) == hashes[rel], \
            f"FINAL RESTORE FAILED for {rel}"
    rc, out = run_tests()
    print()
    for rel, h in hashes.items():
        print(f"restored byte-identically: {rel} (sha256 {h[:16]})")
    print(f"baseline after restore: {'green' if rc == 0 else 'RED'}")
    print()
    print("ALL CONTROLS PASSED" if not failures else f"{failures} MISS(ES)")
    return 0 if (failures == 0 and rc == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
