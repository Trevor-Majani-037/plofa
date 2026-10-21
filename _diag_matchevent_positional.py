"""Is any `MatchEvent(...)` constructed POSITIONALLY?

The `match_clock_s` field was INSERTED into the dataclass BEFORE `metadata`.
A keyword construction is unaffected; a POSITIONAL one would silently bind
`metadata` (a dict) to `match_clock_s` and the real metadata to `metadata`'s
old slot — i.e. a silent, catastrophic mis-binding that no test would
notice except as "the metadata is gone".

This walks the AST of every module and reports any `MatchEvent(...)` call
with a positional argument.  Keyword-only is the requirement.

Run:  .venv\\Scripts\\python.exe -u _diag_matchevent_positional.py
"""
from __future__ import annotations

import ast
import os
import sys


def check(path):
    with open(path, encoding="utf-8", newline="") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"  SKIP {path}: {e}")
        return 0
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = None
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            name = fn.attr
        if name != "MatchEvent":
            continue
        if node.args:
            bad.append((node.lineno, len(node.args)))
    for lineno, n in bad:
        print(f"  POSITIONAL  {path}:{lineno}  ({n} positional arg(s))")
    return len(bad)


def main():
    roots = sys.argv[1:] or ["."]
    total = 0
    scanned = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", "plofa_output",
                                        ".quarantine_26", ".corrupt", "brains",
                                        "venv", ".venv")]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                scanned += 1
                total += check(p)
    print(f"\nscanned {scanned} files; positional MatchEvent(...) calls: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
