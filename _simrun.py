"""Background: measure home-draw-away split for BALANCED equal teams and for
normal teams, many matches, dumping to a log file."""
import sys, random
import _repro as R

strict = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
mode = sys.argv[3] if len(sys.argv) > 3 else "balanced"
out = open("_simlog.txt", "a")
for i in range(n):
    if mode == "balanced":
        hg, ag, hrc, arc = R.run_one(strict, 5000 + i, balanced=True)
    elif mode == "swap":
        hg, ag, hrc, arc = R.run_one(strict, 5000 + i, swap=True)
    else:
        hg, ag, hrc, arc = R.run_one(strict, 5000 + i)
    res = f"{mode} s={strict} m{i}: {hg}-{ag} red h={hrc} a={arc}"
    print(res); out.write(res + "\n"); out.flush()
out.close()
