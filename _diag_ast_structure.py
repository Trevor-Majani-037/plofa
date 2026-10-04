"""Structural check after editing event_chain.py / match_engine.py.

Follows the rule recorded in AGENTS.md: use `ast`, never a regex over
`^    def `, and assert class structure rather than trusting a successful
import (a re-parented class body is still valid Python).
"""
import ast, sys

for path in ("event_chain.py", "match_engine.py"):
    with open(path, "rb") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    top = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    print(f"{path}: {len(tree.body)} top-level nodes, "
          f"{len(top)} column-0 classes")
    for c in top:
        methods = [n.name for n in c.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        dupes = {m for m in methods if methods.count(m) > 1}
        print(f"   class {c.name}: {len(methods)} methods"
              + (f"  DUPLICATES {dupes}" if dupes else ""))
        if dupes:
            sys.exit(f"FAIL: duplicate methods in {c.name}")

# The specific things today's edit touched.
with open("event_chain.py", "rb") as fh:
    ec = ast.parse(fh.read())
attack = next(n for n in ec.body
              if isinstance(n, ast.ClassDef) and n.name == "AttackChain")
methods = {n.name for n in attack.body if isinstance(n, ast.FunctionDef)}
for required in ("generate", "_named_outfielder", "_named_shooter",
                 "_resolve_assister", "_pick_gk", "_pick_shooter"):
    assert required in methods, f"MISSING {required}"
for banned in ("_pick_creator",):
    assert banned not in methods, f"{banned} still present"
print("\nOK: AttackChain has _named_outfielder, no _pick_creator; no duplicate methods.")
