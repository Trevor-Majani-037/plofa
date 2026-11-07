"""Which code path performs the big coordinate writes? Reads the stack dumps
that `PositionState.__setattr__` prints under PLOFA_TRACE_JUMPS=1.

Blocks are delimited by the `[TRACE current_*]` header, not by a fixed
window, because a stack dump is longer than a screenful."""
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "tools/diag/_fix11d_run.log"
raw = open(path, "rb").read()
if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
    txt = raw.decode("utf-16", errors="replace")
else:
    txt = raw.decode("utf-8", errors="replace")
lines = txt.splitlines()

starts = [i for i, l in enumerate(lines) if "TRACE current" in l]
blocks = []
for k, i in enumerate(starts):
    end = starts[k + 1] if k + 1 < len(starts) else len(lines)
    blocks.append(lines[i:end])

print("big-write trace blocks:", len(blocks))
c = Counter()
sizes = Counter()
for b in blocks:
    frames = [l.strip() for l in b if 'File "' in l]
    keep = [f for f in frames if "__setattr__" not in f]
    sig = []
    for f in keep[-2:]:
        try:
            fn = f.split('File "')[1].split('"')[0].rsplit("\\", 1)[-1]
            ln = f.split("line ")[1].split(",")[0]
            sig.append(f"{fn}:{ln}")
        except IndexError:
            pass
    c[" <- ".join(reversed(sig))] += 1
    try:
        sizes[b[0].split("(")[1].split(" m)")[0]] += 1
    except IndexError:
        pass
for k, v in c.most_common(12):
    print(f"{v:4d}  {k}")
