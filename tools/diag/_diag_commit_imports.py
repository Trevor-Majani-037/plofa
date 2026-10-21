"""Does the COMMITTED tree import anything it does not contain?

A commit that stages the code but not the module it imports is a commit
that cannot be cloned.  This walks the COMMITTED python files, collects
their local imports, and checks each one is tracked in git.

    .venv\\Scripts\\python.exe tools\\diag\\_diag_commit_imports.py
"""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _local_modules():
    """Every local module name in the repo (a .py file or a package)."""
    mods = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", "plofa_output",
                                    ".quarantine_x", "node_modules")]
        rel = os.path.relpath(dirpath, ROOT)
        # The ROOT ITSELF is "." — and `rel.startswith(".")` is true for it,
        # so skipping on that prefix SKIPS THE WHOLE REPOSITORY ROOT, which
        # is where the most-imported modules live (threat_reward.py).  The
        # first version of this probe therefore reported "everything is
        # committed" while `threat_reward.py` was visibly untracked.
        if rel != "." and rel.startswith("."):
            continue
        for f in filenames:
            if f.endswith(".py"):
                name = f[:-3]
                if rel != ".":
                    name = os.path.join(rel, name).replace("\\", ".")
                mods.add(name)
                mods.add(name.split(".")[-1])
    return mods


def main() -> int:
    tracked = set(subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split())
    tracked_stems = {t[:-3] for t in tracked if t.endswith(".py")}
    tracked_stems |= {t.split("/")[-1][:-3] for t in tracked
                      if t.endswith(".py")}
    local = _local_modules()

    committed_py = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()

    missing = {}
    for rel in committed_py:
        path = os.path.join(ROOT, rel)
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods = [node.module or ""]
            for m in mods:
                if not m:
                    continue
                top = m.split(".")[0]
                if top not in local:
                    continue          # a standard-library or installed module
                if m in tracked_stems or top in tracked_stems:
                    continue
                missing.setdefault(top, set()).add(rel)
    if not missing:
        print("every local import in the committed tree is committed too")
        return 0
    print("MISSING FROM THE COMMIT:")
    for mod, users in sorted(missing.items()):
        print(f"  {mod}.py  (imported by {len(users)} committed file(s))")
        for u in sorted(users)[:6]:
            print(f"      {u}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
