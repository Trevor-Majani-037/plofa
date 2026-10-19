"""NEGATIVE CONTROL for tests/test_keeper_attribution.py.

A green guard never seen to fail is not a guard. Each mutation restores ONE
defect and asserts a real assertion goes red (not a collection error), then
restores the file byte-identically.

  M1 stamp the TEAM xGOT back onto every GK        -> keeper scope goes red
  M2 mirror `e.team` for OWN_GOAL like a GOAL      -> own-goal test goes red
  M3 restore the old stationary GK template        -> GK work-rate tests red
  M4 restore the flat `physics_dist * 0.45`        -> additive tests red

GATES BEFORE A MUTATION COUNTS AS EVIDENCE
  G1 anchor exists and is UNIQUE (a non-matching anchor mutates nothing and
     then reports a clean PASS -- the worst possible outcome)
  G2 the mutant must still IMPORT; an IndentationError is not a red assertion
  G3 the baseline must be green first
"""
import subprocess
import sys

EXPORTER = "exporter.py"
OPTA = "opta_analytics.py"
TESTS = "tests/test_keeper_attribution.py"

MUTATIONS = [
    (EXPORTER, "M1 stamp the TEAM xGOT back onto every goalkeeper (the "
     "originally reported bug)",
     [('        xgot = round(self.keeper_xgot.get(name, 0.0), 3)',
       '        xgot = round(self.team_xgot.get(s.get("team", ""), 0.0), 3)')]),
    (EXPORTER, "M2 mirror `e.team` for OWN_GOAL the way a GOAL is mirrored",
     [("                defending = (\n"
       "                    e.team if e.event_type == EventType.OWN_GOAL\n"
       "                    else (config.away_team if e.team == config.home_team\n"
       "                          else config.home_team))",
       "                defending = (config.away_team if e.team == config.home_team\n"
       "                             else config.home_team)")]),
    (OPTA, "M3 restore the old stationary GK template (0.80, 0.18, 0.02, ...)",
     [('    "GK":  (0.66, 0.17, 0.15, 0.02, 0.00),',
       '    "GK":  (0.80, 0.18, 0.02, 0.00, 0.00),')]),
    (OPTA, "M4 drop the measured metres entirely (publish the template only "
     "-- the original defect shape)",
     [("                    baseline_dist = physics_dist + template_dist",
       "                    baseline_dist = template_dist")]),
]


def run_tests():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "--tb=no"],
        capture_output=True, text=True, cwd=".")
    tail = [ln for ln in r.stdout.splitlines()
            if "passed" in ln or "failed" in ln or "error" in ln]
    return r.returncode, (tail[-1] if tail else r.stdout.strip()[-200:])


def mutate(target, edits, originals):
    src = originals[target]                    # snapshot taken by main()
    for old, new in edits:
        if old not in src:
            raise SystemExit(f"ANCHOR MISSING in {target} -- no-op mutation:\n{old!r}")
        if src.count(old) != 1:
            raise SystemExit(f"anchor not unique in {target} "
                             f"({src.count(old)}x) -- refusing")
        src = src.replace(old, new)
    open(target, "w", encoding="utf-8").write(src)
    chk = subprocess.run([sys.executable, "-c",
                          f"import {target[:-3]}"],
                         capture_output=True, text=True, cwd=".")
    if chk.returncode != 0:
        open(target, "w", encoding="utf-8").write(originals[target])
        raise SystemExit(f"MUTANT DOES NOT IMPORT -- invalid control:\n{chk.stderr[-400:]}")


def main():
    rc0, base = run_tests()
    print(f"BASELINE (unmutated): rc={rc0}  {base}")
    if rc0 != 0:
        raise SystemExit("baseline is not green -- a red guard proves nothing")

    originals = {t: open(t, encoding="utf-8").read() for t in (EXPORTER, OPTA)}
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