"""NEGATIVE CONTROL — the situational brain's guards, seen to FAIL.

Run:  .venv\\Scripts\\python.exe tools\\diag\\_diag_situational_brain_guard.py

WHAT THIS IS
------------
Every mutation below puts a REAL defect back into the code and asserts
that the guard suite goes RED.  A guard that has never been seen to fail
is not a guard — it is a comment.

The mutations are the four bugs this work already had, plus the four
failure modes the guards exist to prevent:

  M1  the GRU's update-gate sign (``dh'/dz``) flipped
  M2  the SHORT softmax Jacobian in the attention
  M3  the weight-gradient orientation transposed (``dv.T @ e``)
  M4  the attention MASK ignored (padded tokens take mass)
  M5  the plan's HYSTERESIS removed (one touch adopts a plan)
  M6  the memory block always zero
  M7  the provisioned mind ENGAGED (which would change every decision)
  M8  the router sends an ORDINARY brain down the situational path

THE RULES THIS FILE FOLLOWS
---------------------------
* every mutant must still COMPILE — a syntax error counted as a "caught"
  mutation proves nothing about which assertion pins the defect;
* the anchors are matched against the file's OWN line ending (this
  project has CRLF files, and an LF anchor against a CRLF file matches
  NOTHING, so every mutation silently becomes a no-op — which is how a
  control reports "0/5 caught" while the code is broken);
* the file is restored BYTE-IDENTICALLY, and the restore is verified.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable

TEST = os.path.join(ROOT, "tests", "test_situational_brain.py")

BRAIN = os.path.join(ROOT, "situational_brain.py")
STATE = os.path.join(ROOT, "player_state.py")
MEM = os.path.join(ROOT, "memory_features.py")
INTEG = os.path.join(ROOT, "brain_integration.py")


# ── the mutations ─────────────────────────────────────────────────────
# (name, file, old, new, the test that must go RED)
MUTATIONS = [
    ("M1 gru update-gate sign", BRAIN,
     "dz = dh_prime * (h_prev - n)",
     "dz = dh_prime * (n - h_prev)",
     "test_the_backward_pass_matches_finite_differences"),

    ("M2 short softmax jacobian", BRAIN,
     "dscores = alpha * (dalpha - dot)",
     "dscores = dalpha * alpha * (1.0 - alpha)",
     "test_the_backward_pass_matches_finite_differences"),

    ("M3 transposed weight gradient", BRAIN,
     'grads["w_v"] += c["e"].T @ dv',
     'grads["w_v"] += dv.T @ c["e"]',
     "test_the_backward_pass_matches_finite_differences"),

    ("M4 attention mask ignored", BRAIN,
     "scores = np.where(mask > 0.0, scores, -np.inf)",
     "scores = scores + 0.0",
     "test_a_masked_token_takes_no_attention_mass"),

    ("M5 plan hysteresis removed", STATE,
     "if self._candidate_hits >= int(min_hits) and \\",
     "if self._candidate_hits >= 1 and \\",
     "test_one_confident_touch_is_not_a_plan"),

    ("M6 memory block always zero", MEM,
     "out[5] = _clamp01(len(events) / COUNT_SCALE)",
     "out[5] = 0.0",
     "test_remembering_something_changes_the_block"),

    ("M7 provisioned mind engaged", MEM,
     "    mind.engaged = False\n    register_mind(player_name, mind)",
     "    mind.engaged = True\n    register_mind(player_name, mind)",
     "test_the_provisioned_mind_is_UNENGAGED"),

    ("M8 ordinary brain routed situationally", INTEG,
     "        brain = _resolve_brain(player)\n"
     "        if not isinstance(brain, SituationalBrain):\n"
     "            return None",
     "        brain = _resolve_brain(player)\n"
     "        if False:\n"
     "            return None",
     "test_an_ordinary_brain_does_not_route_into_the_situational_path"),
]


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _eol(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return b"\r\n" if b"\r\n" in raw else b"\n"


def _apply(path, old, new):
    """Replace `old` with `new`, honouring the file's own line ending."""
    eol = _eol(path)
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    o = old.replace("\n", eol.decode("latin-1"))
    n = new.replace("\n", eol.decode("latin-1"))
    if o not in text:
        return False, "ANCHOR NOT FOUND"
    if text.count(o) != 1:
        return False, f"ANCHOR MATCHED {text.count(o)} TIMES"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace(o, n))
    return True, "applied"


def _run_test(node):
    proc = subprocess.run(
        [PY, "-m", "pytest", f"{TEST}::{node}", "-q"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        timeout=900)
    return proc.returncode != 0, proc.stdout[-1500:]


def main() -> int:
    baseline_fail, out = _run_test(
        "TheBackwardPassIsCorrectTests::"
        "test_the_backward_pass_matches_finite_differences")
    if baseline_fail:
        print("BASELINE IS RED — the guard suite does not pass on the "
              "unmutated code, so it cannot prove anything.")
        print(out)
        return 1
    print("baseline GREEN\n")

    results = []
    restore_failed = False
    for name, path, old, new, node in MUTATIONS:
        backup = tempfile.mktemp(suffix=".bak")
        shutil.copy2(path, backup)
        before = _sha(path)
        try:
            applied, why = _apply(path, old, new)
            if not applied:
                results.append((name, "MISS", why))
                print(f"{name:38s} MISS  ({why})")
                continue
            # The mutant must still COMPILE.
            comp = subprocess.run([PY, "-c", "import ast,sys;"
                                   f"ast.parse(open(r'{path}',encoding='utf-8')"
                                   ".read())"],
                                  cwd=ROOT, capture_output=True, text=True)
            if comp.returncode != 0:
                results.append((name, "MISS", "mutant does not compile"))
                print(f"{name:38s} MISS  (mutant does not compile — a syntax "
                      f"error is not a caught mutation)")
                continue
            caught, tail = _run_test(node)
            if caught:
                results.append((name, "CAUGHT", node))
                print(f"{name:38s} CAUGHT  ({node})")
            else:
                results.append((name, "MISS", node))
                print(f"{name:38s} MISS  ({node} stayed green)")
                print(tail)
        finally:
            shutil.copy2(backup, path)
            os.remove(backup)
            after = _sha(path)
            if after != before:
                print(f"  !! RESTORE FAILED for {path}")
                restore_failed = True

    # The baseline must be green again after all that.
    baseline_fail, out = _run_test(
        "TheBackwardPassIsCorrectTests::"
        "test_the_backward_pass_matches_finite_differences")
    print(f"\nbaseline after restore: "
          f"{'RED (bad)' if baseline_fail else 'GREEN'}")
    caught = sum(1 for _n, s, _w in results if s == "CAUGHT")
    print(f"{caught}/{len(results)} mutations caught")
    if caught != len(results) or baseline_fail or restore_failed:
        return 1
    print("ALL MUTATIONS CAUGHT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
