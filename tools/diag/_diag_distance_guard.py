"""NEGATIVE CONTROL for tests/test_distance_accounting.py.

A green guard that has never been seen to fail is not a guard. Each mutation
restores ONE of the two defects and asserts a real assertion goes red -- then
restores the file byte-identically.

M1  remove the physical reach ceiling
M2  move the RAW-geometry accumulation back OUTSIDE the consecutive-tick guard
    -- the exact pre-fix shape -- with the ceiling left in place, so it
    isolates the "a position gap books distance" defect from the cap defect
M3  strip the ceiling, so the structural non-vacuity check can be shown to fire
    on its own

TWO GATES BEFORE A MUTATION COUNTS AS EVIDENCE
  G1 the anchor must exist and be UNIQUE (a non-matching anchor mutates
     nothing and then reports a clean PASS -- the worst possible outcome)
  G2 the mutant must still IMPORT. An earlier version of this control "caught"
     M2 as an IndentationError during collection: rc != 0, printed CAUGHT, and
     proved nothing about which assertion pins the defect. A mutation caught
     by a syntax error is a broken experiment, not a red assertion.

CONDUCT NOTE (AGENTS.md, 2026-10-05): the snapshot is taken INSIDE `mutate()`,
immediately before the write, and RETURNED. An earlier probe compared against a
backup captured outside the mutation and "restored" a stale file, silently
reverting real work while reporting success. A backup is only valid for the
state it was taken from.
"""
import subprocess
import sys

TARGET = "possession_physics.py"
TESTS = "tests/test_distance_accounting.py"

MUTATIONS = [
    ("M1 remove the physical reach ceiling",
     [("                    seg_dist = min(seg_dist, dt * speed)\n", "")]),
    ("M2 accumulate the RAW geometry outside the tick guard (the pre-fix "
     "shape), ceiling left in place so it isolates the gap defect",
     [("                seg_dist = math.hypot(dx, dy)\n"
       "                dt = t1 - t0\n",
       "                seg_dist = math.hypot(dx, dy)\n"
       "                total_dist += seg_dist\n"
       "                dt = t1 - t0\n"),
      ("                    seg_dist = min(seg_dist, dt * speed)\n"
       "                    total_dist += seg_dist\n",
       "                    seg_dist = min(seg_dist, dt * speed)\n")]),
    ("M3 strip the ceiling entirely -> the structural non-vacuity check must "
     "fire on its own",
     [("                    seg_dist = min(seg_dist, dt * speed)\n",
       "                    pass\n")]),
]


def run_tests():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "--tb=no"],
        capture_output=True, text=True, cwd=".",
    )
    tail = [ln for ln in r.stdout.splitlines()
            if "passed" in ln or "failed" in ln or "error" in ln]
    return r.returncode, (tail[-1] if tail else r.stdout.strip()[-200:])


def mutate(edits):
    src = open(TARGET, encoding="utf-8").read()
    backup = src                     # snapshot INSIDE, immediately before
    for old, new in edits:
        if old not in src:
            raise SystemExit(f"ANCHOR MISSING -- mutation would be a no-op:\n{old!r}")
        if src.count(old) != 1:
            raise SystemExit(f"anchor is not unique ({src.count(old)}x) -- refusing")
        src = src.replace(old, new)
    open(TARGET, "w", encoding="utf-8").write(src)
    # G2: the mutant must still import, or the control is a syntax check.
    chk = subprocess.run([sys.executable, "-c", "import possession_physics"],
                         capture_output=True, text=True, cwd=".")
    if chk.returncode != 0:
        open(TARGET, "w", encoding="utf-8").write(backup)
        raise SystemExit(f"MUTANT DOES NOT IMPORT -- invalid control:\n{chk.stderr[-400:]}")
    return backup


def main():
    rc0, base = run_tests()
    print(f"BASELINE (unmutated): rc={rc0}  {base}")
    if rc0 != 0:
        raise SystemExit("baseline is not green -- a red guard proves nothing")
    original = open(TARGET, encoding="utf-8").read()

    failures = 0
    try:
        for name, edits in MUTATIONS:
            backup = mutate(edits)
            rc, out = run_tests()
            ok = rc != 0 and "error" not in out
            print(f"  {'CAUGHT ' if ok else 'MISSED '} {name}")
            print(f"            rc={rc}  {out}")
            if not ok:
                failures += 1
            open(TARGET, "w", encoding="utf-8").write(backup)
            assert open(TARGET, encoding="utf-8").read() == original, \
                "restore did not reproduce the file byte-exactly"
    finally:
        open(TARGET, "w", encoding="utf-8").write(original)

    assert open(TARGET, encoding="utf-8").read() == original
    rc, out = run_tests()
    print(f"RESTORED: rc={rc}  {out}   file byte-identical")
    print(f"\n{len(MUTATIONS) - failures}/{len(MUTATIONS)} controls caught")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())