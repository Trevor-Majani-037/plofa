import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "plofa_broadcast/replay.json"
d = json.load(open(path, encoding="utf-8"))
watch = {"Joel Trent", "Luca Ferrini", "Pavel Renko", "Keano Walsh", "Jasper Vance"}
for m in (78, 82, 86, 90):
    f = d["frames"][m - 1]
    print("minute", m)
    for side in ("home", "away"):
        for r in f[side]:
            if r["position"] == "GK" or r["player"] in watch:
                print("   %-5s %-16s %-4s x=%6.1f y=%5.1f" % (
                    side, r["player"], r["position"], r["x"], r["y"]))
