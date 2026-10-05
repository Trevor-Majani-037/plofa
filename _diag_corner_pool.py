"""How often should SHORT_CORNER be picked? Read the pools, do not infer it.

`SHORT_CORNER` is in every style pool, so it is selectable everywhere -- but
"in the pool" is not "often". Its share is the count of entries against the
pool size, and real football works a corner short on roughly 10-20%.

The measured match gave 3 short corners out of 8 actual corners = 37%, which is
above that band. This prints what the pools actually imply, so the measurement
can be compared against the intention rather than guessed at.
"""
import re
import sys
from collections import Counter

sys.path.insert(0, ".")
import set_piece_routines as spr  # noqa: E402

pools = getattr(spr, "_STYLE_CORNER_POOLS", {})
if not pools:
    # the pools may be assembled rather than literal; fall back to the source
    import pathlib
    src = pathlib.Path("set_piece_routines.py").read_text(encoding="utf-8")
    i = src.find("_STYLE_CORNER_POOLS")
    j = src.find("_STYLE_FK_POOLS")
    seg = src[i:j]
    pools = {}
    for m in re.finditer(r'"(\w+)": \(([^)]*)\)', seg, re.S):
        items = [x.strip().replace("SetPieceRoutine.", "").split(".")[-1]
                 for x in m.group(2).split(",") if x.strip()]
        if items:
            pools[m.group(1)] = items

print(f"{'style':<24} {'SHORT':>6} {'pool':>5} {'share':>7}")
tot_short = tot_all = 0
for name, items in pools.items():
    sc = sum(1 for x in items
             if str(getattr(x, "name", x)).endswith("SHORT_CORNER"))
    tot_short += sc
    tot_all += len(items)
    print(f"{name:<24} {sc:>6} {len(items):>5} "
          f"{100 * sc / max(1, len(items)):>6.0f}%")
if tot_all:
    print(f"{'POOLED':<24} {tot_short:>6} {tot_all:>5} "
          f"{100 * tot_short / tot_all:>6.0f}%")
print()
print("real football: a corner worked short runs roughly 10-20% of the time,")
print("more when chasing a game late. Measured one match: 3 of 8 = 37%.")
print()
print("OVERALL COMPOSITION (which routine is picked at all):")
c = Counter()
for items in pools.values():
    c.update(items)
for k, v in c.most_common():
    print(f"   {k:<24} {v}")