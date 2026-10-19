"""BISECT: is `test_gps_matches_engine_physics` failing because of the physical
reach ceiling, or was it already failing?

The test compares the VirtualGPS recorder's per-player distance against the
engine's own `physics_distance_m` sum and demands a mean drift < 25 m. The
ceiling I added only ever REDUCES the engine's number, so on its face it can
only have made this agreement worse -- unless the GPS was the inflated one.
Do not assume either way. Apply the exact pre-fix state, run the real pytest
node once per arm, restore byte-exactly.

The pre-fix shape is reconstructed with TWO coordinated edits (the same pair the
negative control uses), because a one-shot edit that only breaks indentation
produces a syntax error instead of the old behaviour.
"""
import hashlib
import subprocess
import sys

TARGET = "possession_physics.py"
NODE = "tests/test_virtual_gps.py::test_gps_matches_engine_physics"

PRE_EDITS = [
    # put the raw-geometry accumulation back at method-body level
    ("                seg_dist = math.hypot(dx, dy)\n"
     "                dt = t1 - t0\n",
     "                seg_dist = math.hypot(dx, dy)\n"
     "                total_dist += seg_dist\n"
     "                dt = t1 - t0\n"),
    # and remove the ceiling from inside the tick guard
    ("                    seg_dist = min(seg_dist, dt * speed)\n"
     "                    total_dist += seg_dist\n",
     "                    total_dist += seg_dist\n"),
]


def digest(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def run():
    r = subprocess.run([sys.executable, "-m", "pytest", NODE, "-q", "--no-header",
                        "--tb=line"], capture_output=True, text=True, cwd=".")
    line = ""
    for ln in r.stdout.splitlines():
        if "drift" in ln:
            line = ln.strip()
    passed = r.returncode == 0
    nums = []
    if line and "drift [" in line:
        body = line.split("drift [", 1)[1].rstrip("]").strip()
        try:
            nums = [float(x) for x in body.split(",")]
        except ValueError:
            nums = []
    tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln]
    return passed, (tail[-1] if tail else ""), nums


def main():
    original = open(TARGET, encoding="utf-8").read()
    d0 = digest(TARGET)
    print(f"file digest before any arm: {d0}")

    ok_post, out_post, nums_post = run()
    mean_post = sum(nums_post) / len(nums_post) if nums_post else float("nan")
    print(f"\nARM  post-fix  (ceiling ON)   : {'PASS' if ok_post else 'FAIL'}  "
          f"mean drift {mean_post:.1f} m   {out_post}")

    src = original
    for old, new in PRE_EDITS:
        assert src.count(old) == 1, f"anchor count {src.count(old)}"
        src = src.replace(old, new)
    open(TARGET, "w", encoding="utf-8").write(src)
    chk = subprocess.run([sys.executable, "-c", "import possession_physics"],
                         capture_output=True, text=True, cwd=".")
    if chk.returncode != 0:
        open(TARGET, "w", encoding="utf-8").write(original)
        raise SystemExit(f"pre-fix arm does not import:\n{chk.stderr[-400:]}")

    try:
        ok_pre, out_pre, nums_pre = run()
        mean_pre = sum(nums_pre) / len(nums_pre) if nums_pre else float("nan")
        print(f"ARM  pre-fix   (ceiling OFF)  : {'PASS' if ok_pre else 'FAIL'}  "
              f"mean drift {mean_pre:.1f} m   {out_pre}")
    finally:
        open(TARGET, "w", encoding="utf-8").write(original)

    d1 = digest(TARGET)
    print(f"\nfile digest after restore    : {d1}  "
          f"{'IDENTICAL' if d0 == d1 else '*** NOT IDENTICAL ***'}")
    print(f"\nVERDICT: pre-fix mean drift {mean_pre:.1f} m vs post-fix "
          f"{mean_post:.1f} m   (test threshold 25.0 m)")
    if ok_pre == ok_post:
        print("=> the ceiling did NOT change this test's verdict: PRE-EXISTING.")
    elif ok_pre and not ok_post:
        print("=> REGRESSION caused by the ceiling.")
    else:
        print("=> the ceiling FIXED it (it was failing before).")


if __name__ == "__main__":
    main()