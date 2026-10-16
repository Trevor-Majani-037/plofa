"""
_diag_momentum_guard.py — NEGATIVE CONTROL for the two AST guards in
tests/test_momentum_model.py.

A green guard that has never been seen to fail is not a guard. This
reintroduces each defect the guards exist to catch, confirms the guard goes
RED for the right reason, and restores the files byte-identically.

Run:  .venv\\Scripts\\python.exe tools\\diag\\_diag_momentum_guard.py
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPORTER = os.path.join(ROOT, "exporter.py")
ENGINE = os.path.join(ROOT, "match_engine.py")
TEST = "tests/test_momentum_model.py"
FAILS = []

ANCHOR = "series = opta.momentum_series if opta is not None else []"


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
          f"{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


def run_tests(k=None):
    """Run the suite (optionally only the guards matching `k`).

    A single `-k` selector per mutation is what makes "it went RED for the
    RIGHT reason" checkable: with the whole file run, M1 ALSO trips M2's
    closure guard (importing momentum_model into match_engine pulls the
    mutated exporter into the dependency walk) and the summary cannot say
    which defect was caught. It also means a guard that cannot fail at all
    still shows up as a red run.
    """
    env = dict(os.environ, PYTHONPATH=".")
    cmd = [os.path.join(ROOT, ".venv", "Scripts", "python.exe"), "-m", "pytest",
           TEST, "-q", "--no-header", "--tb=no"]
    if k:
        cmd += ["-k", k]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout


def mutate(path, old, new):
    """Snapshot the file, apply the mutation, RETURN the snapshot.

    The snapshot is taken HERE, immediately before the mutation. The first
    version captured all backups once at module level and reused them; after
    an aborted run left a stale one behind, the next run "restored" the WRONG
    bytes and silently reverted `plot_momentum` to the old bar chart. A
    backup is only valid for the state it was taken from.
    """
    with open(path, "rb") as fh:
        backup = fh.read()
    src = backup.decode("utf-8")
    assert old in src, \
        f"anchor not found in {os.path.basename(path)} — the mutation " \
        f"would be a no-op, and a no-op that 'passes' proves nothing"
    with open(path, "wb") as fh:
        fh.write(src.replace(old, new, 1).encode("utf-8"))
    return backup


def restore(path, backup):
    """Write the snapshot back, then PROVE the round trip."""
    with open(path, "wb") as fh:
        fh.write(backup)
    with open(path, "rb") as fh:
        after = fh.read()
    if after != backup:
        raise SystemExit(f"ABORT: could not restore {path} byte-identically")
    print(f"  restored {os.path.basename(path)} byte-identically "
          f"({len(backup)} bytes)")


# ── baseline ────────────────────────────────────────────────────────────
print("\n0. BASELINE (must be green, or the controls prove nothing)")
rc, out = run_tests()
check("suite green before any mutation", rc == 0,
      out.strip().splitlines()[-1] if out.strip() else "")

# ── M1: the chart goes back to the goal-driven log ──────────────────────
print("\n1. M1 — exporter reads result.momentum_log again")
try:
    bak = mutate(EXPORTER, ANCHOR,
                 "series = [{'minute': r['minute'], 'momentum': r['momentum']}\n"
                 "                   for r in self.result.momentum_log]")
    rc, out = run_tests("test_the_exporter_no_longer_plots")
    check("guard goes RED", rc != 0)
    check("...and the OTHER guard stays green (only the right one broke)",
          "1 failed" in out, out.strip().splitlines()[-1] if out.strip() else "")
finally:
    restore(EXPORTER, bak)

# ── M2: the engine imports the reporting model ─────────────────────────
print("\n2. M2 — match_engine imports momentum_model (breaks the one-way rule)")
try:
    bak = mutate(ENGINE, "from __future__ import annotations",
                 "from __future__ import annotations\nimport momentum_model")
    rc, out = run_tests("test_no_engine_module_imports")
    check("guard goes RED", rc != 0)
    check("...and the OTHER guard stays green (only the right one broke)",
          "1 failed" in out, out.strip().splitlines()[-1] if out.strip() else "")
finally:
    restore(ENGINE, bak)

# ── M3: prose must NOT trip the guard ───────────────────────────────────
print("\n3. M3 — prose mentioning momentum_log must NOT trip the guard")
try:
    bak = mutate(EXPORTER, ANCHOR,
                 "# this replaced a series built from result.momentum_log\n        "
                 + ANCHOR)
    rc, out = run_tests()
    check("guard stays GREEN on a comment (a name in prose is not a call site)",
          rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
finally:
    restore(EXPORTER, bak)

rc, out = run_tests()
check("suite green again after all mutations and restores", rc == 0,
      out.strip().splitlines()[-1] if out.strip() else "")

print("\n" + "=" * 62)
print("RESULT:", "ALL CONTROLS PASSED" if not FAILS else f"{len(FAILS)} FAILED")
for f in FAILS:
    print("   FAILED:", f)
print("=" * 62)
sys.exit(1 if FAILS else 0)