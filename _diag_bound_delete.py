"""Bound a deletion by AST, never by a regex over `^    def `.

The recorded lesson (AGENTS.md, 2026-10-02): `match_engine.py` has a
column-0 `class MatchResult` after `MatchEngine` ends. A scan for the next line
matching `^    def ` sailed 85 lines past the method, through the module banner
and the dataclass header, and silently re-parented everything after it onto
`MatchEngine` — still valid Python, so `import` succeeded and only a NameError
from deep inside `simulate()` revealed it. A duplicate class then shadowed the
repair.

So: parse, take `node.end_lineno`, verify the range, print it, and assert the
class structure afterwards.
"""
import ast
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "event_chain.py"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "_generate_pass_event"

with open(PATH, "rb") as fh:
    src_bytes = fh.read()
tree = ast.parse(src_bytes, filename=PATH)

owner = None
node = None
for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
    for f in cls.body:
        if isinstance(f, ast.FunctionDef) and f.name == TARGET:
            if node is not None:
                sys.exit(f"FAIL: {TARGET} defined more than once")
            owner, node = cls.name, f

if node is None:
    print(f"{TARGET}: not found")
    sys.exit(0)

print(f"{PATH}: {TARGET} is in class {owner}")
print(f"  lines {node.lineno} .. {node.end_lineno}")

# Everything in the range must belong to THIS function.
inner = [n for n in ast.walk(node)
         if isinstance(n, (ast.FunctionDef, ast.ClassDef))
         and n is not node]
if inner:
    sys.exit(f"FAIL: range contains nested defs {[n.name for n in inner]}")

# No other reference anywhere in the repo (comments do not count).
refs = 0
import os
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".venv",
                                            "brains", "plofa_output", "battlefield")]
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        if os.path.abspath(p) == os.path.abspath(PATH):
            continue
        with open(p, "rb") as fh:
            t = fh.read()
        try:
            tt = ast.parse(t, filename=p)
        except SyntaxError:
            continue
        for n in ast.walk(tt):
            if isinstance(n, ast.Attribute) and n.attr == TARGET:
                refs += 1
            if isinstance(n, ast.Name) and n.id == TARGET:
                refs += 1
print(f"  external references (AST, comments excluded): {refs}")

lines = src_bytes.decode("utf-8").split("\n")
block = "\n".join(lines[node.lineno - 1:node.end_lineno])
print(f"  block is {len(block)} chars, {block.count(chr(10)) + 1} lines")
print("-" * 70)
print(block)
