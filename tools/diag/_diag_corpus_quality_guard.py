"""NEGATIVE CONTROL — the corpus census, seen to FAIL.

Run:  .venv\\Scripts\\python.exe tools\\diag\\_diag_corpus_quality_guard.py

The corpus census is a GATE on the corpus, so it has to be able to fail.
Four mutations, each a real defect in a corpus, and each must make the
census say so:

  M1  an INTERLEAVE is spliced in (a step from another player's
      sequence) — the census must flag the sequence as impossible.
  M2  a RESTART is injected (the carrier's x is reset to a goal line
      with a small order gap) — the census must flag it.
  M3  the MEMORY block is zeroed — the census must report that the
      memory is dead, because a net trained on that corpus would ignore
      one of the four capabilities.
  M4  the TEAM is stripped from the sequences — the census must report
      that the team key is no longer separating the two sides, which is
      the interleave coming back.

Every mutant must still be a READABLE corpus (the census must not
crash — a gate that crashes is not a gate, it is a stop sign).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "diag"))

PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable

PROBE = os.path.join(ROOT, "tools", "diag", "_diag_corpus_quality.py")
CORPUS = os.path.join(ROOT, "artifacts", "_smoke_sequences.json")


def run_probe(path):
    proc = subprocess.run([PY, PROBE, path], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def main() -> int:
    if not os.path.exists(CORPUS):
        print(f"no smoke corpus at {CORPUS} — run the one-match collection "
              f"first")
        return 2
    base = load(CORPUS)

    # ── the baseline must be CLEAN ──────────────────────────────
    rc, out = run_probe(CORPUS)
    if "IMPOSSIBLE ones" not in out:
        print("the baseline probe output does not have the expected shape — "
              "the guard is pointing at the wrong thing")
        print(out[-1500:])
        return 1
    if "0 sequences" not in out:
        print("the baseline corpus is not clean — fix the corpus before "
              "testing the gate on it")
        print(out[-1500:])
        return 1
    print("baseline: clean corpus, 0 impossible sequences\n")

    tmp = tempfile.mkdtemp()
    results = []

    def mutate(name, fn, must_contain):
        data = json.loads(json.dumps(base))
        fn(data)
        path = os.path.join(tmp, f"{name}.json")
        save(path, data)
        rc2, out2 = run_probe(path)
        ok = must_contain in out2 and "Traceback" not in out2
        results.append((name, ok))
        print(f"{name:28s} {'CAUGHT' if ok else 'MISS'}  "
              f"(looking for {must_contain!r})")
        if not ok:
            print(out2[-1200:])

    # M1 — an interleave: splice a step from another sequence in.
    def m1(data):
        seqs = data["sequences"]
        src = seqs[0]["steps"][0]
        for s in seqs[1:]:
            if len(s["steps"]) >= 2:
                s["steps"].insert(1, json.loads(json.dumps(src)))
                break

    # M2 — a restart: reset the carrier's x to a goal line, small gap.
    def m2(data):
        for s in data["sequences"]:
            if len(s["steps"]) >= 2:
                s["steps"][1]["state"][2] = 0.0
                s["steps"][1]["gap"] = 2
                break

    # M3 — the memory block is dead.
    def m3(data):
        for s in data["sequences"]:
            for st in s["steps"]:
                st["memory"] = [0.0] * len(st["memory"])

    # M4 — the team is stripped: the interleave comes back.
    def m4(data):
        for s in data["sequences"]:
            s["team"] = ""

    mutate("M1 interleaved step", m1, "IMPOSSIBLE ones")
    mutate("M2 a restart", m2, "IMPOSSIBLE ones")
    mutate("M3 the memory is dead", m3, "memory block non-zero 0/")
    mutate("M4 the team is stripped", m4, "carried by more than one team: 0")

    shutil.rmtree(tmp, ignore_errors=True)

    caught = sum(1 for _n, ok in results if ok)
    print(f"\n{caught}/{len(results)} mutations caught")
    if caught != len(results):
        return 1
    print("ALL MUTATIONS CAUGHT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
