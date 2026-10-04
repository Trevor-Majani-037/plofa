"""Delete `PossessionChain._generate_pass_event` and PROVE the file survived.

The rules this follows are the ones `match_engine.py` taught the hard way
(silently re-parenting a column-0 `class MatchResult` onto `MatchEngine`,
which still imports, then being shadowed by a duplicate further down):

  1. Never bound a deletion by a regex over `^    def `. That regex only sees
     METHODS; the class-structure scan sailed 85 lines past the end of the
     method and through a module-level dataclass.
  2. Take the bounds from `ast` (`lineno`/`end_lineno`), never from a search.
  3. Snapshot the structure BEFORE, and assert against the snapshot AFTER.
  4. A successful `import` proves nothing about class structure in this file,
     so the structural check is the real check.
"""
import ast
import sys

PATH = "event_chain.py"

with open(PATH, encoding="utf-8") as fh:
    src = fh.read()

lines = src.splitlines(keepends=True)


def structure(text):
    """Column-0 classes and top-level nodes — the things a bad delete breaks."""
    tree = ast.parse(text)
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    return len(tree.body), tuple(classes)


before_nodes, before_classes = structure(src)
print(f"BEFORE  top-level nodes={before_nodes}  "
      f"column-0 classes={len(before_classes)}")

# ── locate by AST ─────────────────────────────────────────────────────────
target = None
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "PossessionChain":
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and sub.name == "_generate_pass_event":
                target = sub
if target is None:
    sys.exit("NOT FOUND: PossessionChain._generate_pass_event — aborting")

start, end = target.lineno, target.end_lineno
print(f"TARGET  PossessionChain._generate_pass_event  lines {start}..{end}  "
      f"({end - start + 1} lines)")

# ── refuse if anything nested, so the bound is provably exact ─────────────
if ast.walk(target) and any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n is not target for n in ast.walk(target)):
    sys.exit("REFUSING: the block contains a nested def/class, so the bound "
             "may clip a neighbour.")

# ── confirm zero external references (comments excluded) ──────────────────
refs = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Attribute) and node.attr == "_generate_pass_event":
        # the definition itself is a FunctionDef, not an Attribute
        refs += 1
    if isinstance(node, ast.Name) and node.id == "_generate_pass_event":
        refs += 1
print(f"REFS    attribute/name references outside the def: {refs}")
if refs:
    sys.exit(f"REFUSING: {refs} live references — this method is reachable.")

# ── delete by EXACT source text, then rewrite ─────────────────────────────
removed = lines[start - 1:end]
# Swallow the trailing blank lines that separated it from the next member.
while removed and removed[-1].strip() == "":
    removed.pop()
    end -= 1

new_lines = lines[:start - 1] + lines[end:]
new_src = "".join(new_lines)

# ── assert structure AFTER ────────────────────────────────────────────────
try:
    after_nodes, after_classes = structure(new_src)
except SyntaxError as e:
    sys.exit(f"ABORT: result does not parse ({e}). File NOT written.")

assert after_nodes == before_nodes, (
    f"top-level node count changed {before_nodes} -> {after_nodes}; a "
    f"column-0 statement was swallowed")
assert after_classes == before_classes, (
    f"column-0 classes changed:\n  before {before_classes}\n  after  "
    f"{after_classes}")
assert "_generate_pass_event" not in new_src, "name still present"
# The sibling that followed it must still be inside PossessionChain.
after_tree = ast.parse(new_src)
poss = [n for n in after_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PossessionChain"]
assert len(poss) == 1, f"PossessionChain count = {len(poss)}"
assert not any(
    isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
    and s.name == "_generate_pass_event" for s in poss[0].body)
print(f"AFTER   top-level nodes={after_nodes}  "
      f"column-0 classes={len(after_classes)}  "
      f"PossessionChain methods={len(poss[0].body)}")

with open(PATH, "w", encoding="utf-8", newline="") as fh:
    fh.write(new_src)
print(f"\nWROTE {PATH}: removed {end - start + 1} lines.")