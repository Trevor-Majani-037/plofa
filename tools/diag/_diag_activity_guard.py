"""NEGATIVE CONTROL for tests/test_measured_activity.py.

A green guard never seen to fail is not a guard. Each mutation restores ONE
defect and asserts a real assertion goes red, then restores byte-identically.

  M1 discard `moved == 0` again (the headline defect) -> standing unmeasurable
  M2 publish the per-position TABLE for every minute   -> seconds not measured
  M3 go back to `template_distance + fraction_of_trace` -> distance not measured
  M4 drop the standing band entirely                  -> largest band gone

GATES BEFORE A MUTATION COUNTS AS EVIDENCE
  G1 anchor exists and is UNIQUE
  G2 the mutant must still IMPORT; an IndentationError is not a red assertion
  G3 the baseline must be green first
"""
import subprocess
import sys

TARGETS = ("match_engine.py", "opta_analytics.py", "position_engine.py")
TESTS = "tests/test_measured_activity.py"

MUTATIONS = [
    ("match_engine.py",
     "M1 discard `moved == 0` again — the headline defect",
     [("""        if moved <= 0:
            self.position_engine.record_physics_distance(
                name, distance_m=0.0, duration_s=duration_s,
                speed_mps=speed)
            return""",
       """        if moved <= 0:
            return""")]),
    ("opta_analytics.py",
     "M2 publish the per-position TABLE for every minute (ignore the measurement)",
     [("                    tpl_s = {b: prof[b] * uncovered_s for b in ACTIVITY_BUCKETS}",
       "                    tpl_s = {b: prof[b] * 60.0 for b in ACTIVITY_BUCKETS}")]),
    ("opta_analytics.py",
     "M3 revert to `template_distance + fraction_of_trace`",
     [("                    baseline_dist = physics_dist + template_dist",
       "                    baseline_dist = template_dist + physics_dist * 0.45")]),
    ("position_engine.py",
     "M4 drop the standing band from the band table",
     [("ACTIVITY_BAND_EDGES: Tuple[float, ...] = (0.5, 2.0, 4.2, 7.0)",
       "ACTIVITY_BAND_EDGES: Tuple[float, ...] = (2.0, 4.2, 7.0)")]),
]


def run_tests():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "--tb=no"],
        capture_output=True, text=True, cwd=".")
    tail = [ln for ln in r.stdout.splitlines()
            if "passed" in ln or "failed" in ln or "error" in ln]
    return r.returncode, (tail[-1] if tail else r.stdout.strip()[-200:])


def mutate(target, edits, originals):
    src = originals[target]
    for old, new in edits:
        if old not in src:
            raise SystemExit(f"ANCHOR MISSING in {target} -- no-op mutation:\n{old!r}")
        if src.count(old) != 1:
            raise SystemExit(f"anchor not unique in {target} "
                             f"({src.count(old)}x) -- refusing")
        src = src.replace(old, new)
    open(target, "w", encoding="utf-8").write(src)
    chk = subprocess.run([sys.executable, "-c", f"import {target[:-3]}"],
                         capture_output=True, text=True, cwd=".")
    if chk.returncode != 0:
        open(target, "w", encoding="utf-8").write(originals[target])
        raise SystemExit(f"MUTANT DOES NOT IMPORT -- invalid control:\n{chk.stderr[-500:]}")


def main():
    rc0, base = run_tests()
    print(f"BASELINE (unmutated): rc={rc0}  {base}")
    if rc0 != 0:
        raise SystemExit("baseline is not green -- a red guard proves nothing")

    originals = {t: open(t, encoding="utf-8").read() for t in TARGETS}
    failures = 0
    try:
        for target, name, edits in MUTATIONS:
            mutate(target, edits, originals)
            rc, out = run_tests()
            ok = rc != 0 and "error" not in out
            print(f"  {'CAUGHT ' if ok else 'MISSED '} {name}")
            print(f"            rc={rc}  {out}")
            if not ok:
                failures += 1
            open(target, "w", encoding="utf-8").write(originals[target])
            for t in originals:
                assert open(t, encoding="utf-8").read() == originals[t], \
                    f"restore of {t} was not byte-exact"
    finally:
        for t, s in originals.items():
            open(t, "w", encoding="utf-8").write(s)

    rc, out = run_tests()
    print(f"RESTORED: rc={rc}  {out}   files byte-identical")
    print(f"\n{len(MUTATIONS) - failures}/{len(MUTATIONS)} controls caught")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())