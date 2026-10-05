"""AUDIT: is POLICY_INTENT_AUTHORITY now FULLY wired?

I claimed the switch was half-wired and patched the two sites I found. That is
a claim about the sites I HAPPENED to find, which is weaker than a claim about
all of them. This walks every write to the variables that can override a
sampled on-ball intent and reports, for each, whether an
`POLICY_INTENT_AUTHORITY` guard appears within the preceding few lines.

Read-only. It asserts nothing about behaviour; it is an inventory so the answer
to "is it fully wired" is a list rather than an assurance.
"""
import pathlib
import re

SRC = pathlib.Path("event_chain.py")
lines = SRC.read_text(encoding="utf-8").splitlines()
VARS = ("forced_receiver", "forced_end", "is_prog", "is_switch",
        "long_intent", "matrix_decision", "active_decision")
WINDOW = 8

print("=== writes to decision variables, with nearest preceding guard ===")
rows = []
for i, l in enumerate(lines):
    m = re.match(r"\s*(" + "|".join(VARS) + r")\s*=\s*[^=]", l)
    if not m:
        continue
    var = m.group(1)
    guard = None
    for j in range(max(0, i - WINDOW), i):
        if "POLICY_INTENT_AUTHORITY" in lines[j]:
            guard = j + 1
    rows.append((i + 1, var, guard, l.strip()[:70]))

for ln, var, guard, txt in rows:
    tag = f"guard@{guard}" if guard else "UNGUARDED"
    print(f"{ln:5d}  {var:<16} {tag:<12} {txt}")

unguarded = [r for r in rows if r[2] is None]
print(f"\ntotal writes: {len(rows)}   unguarded: {len(unguarded)}")
byvar = {}
for ln, var, g, txt in rows:
    byvar.setdefault(var, [0, 0])
    byvar[var][0] += 1
    if g is None:
        byvar[var][1] += 1
print(f"{'variable':<18}{'writes':>8}{'unguarded':>12}")
for var, (tot, un) in sorted(byvar.items()):
    print(f"{var:<18}{tot:>8}{un:>12}")

print("\n=== the unguarded ones, in context, for a human decision ===")
for ln, var, g, txt in unguarded:
    lo, hi = max(0, ln - 4), min(len(lines), ln + 2)
    print(f"\n--- {var} at line {ln} ---")
    for j in range(lo, hi):
        mark = ">>" if j + 1 == ln else "  "
        print(f"{mark} {j+1:5d} {lines[j].strip()[:96]}")
