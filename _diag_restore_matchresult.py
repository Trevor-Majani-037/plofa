"""Restore `class MatchResult`, which is absent from the working tree.

HOW THIS WAS FOUND
A real-match diagnostic crashed at the very last line of `simulate`:

    match_engine.py:4005  result = MatchResult(...)
    NameError: name 'MatchResult' is not defined

`git diff -- match_engine.py` shows `-class MatchResult:`, so the class is
deleted relative to HEAD. Nothing else is missing: every other class in the
current file is either present in HEAD or newer (`MomentumEngine`), so this is
one isolated deletion, not a wholesale overwrite by some other tool.

Only the ONE BLOCK is restored, from HEAD, verbatim. Restoring the whole file
would throw away every legitimate local change (3074 insertions across the
four files under work).

In HEAD the class is the LAST one in the file (lines 5451-5673, to EOF), and
the current file ends inside a `MatchEngine` method, so appending it restores
the original position.

This is worth flagging to the user rather than quietly repairing: the file
lost a 223-line class between two successful runs of `_diag_corner_conv.py`,
and nothing in this session's edits to `match_engine.py` could have done it.
"""
import pathlib
import subprocess

head = subprocess.run(["git", "show", "HEAD:match_engine.py"],
                      capture_output=True, text=True,
                      encoding="utf-8").stdout.splitlines()

start = next(i for i, l in enumerate(head) if l.startswith("class MatchResult"))
end = next((i for i in range(start + 1, len(head)) if head[i].startswith("class ")),
           len(head))
block = head[start:end]
print(f"restoring {len(block)} lines from HEAD ({start + 1}..{end})")

p = pathlib.Path("match_engine.py")
cur = p.read_text(encoding="utf-8").splitlines()
assert not any(l.startswith("class MatchResult") for l in cur), \
    "already present -- refusing to duplicate"
while cur and not cur[-1].strip():
    cur.pop()
cur += ["", ""] + block
p.write_text("\n".join(cur) + "\n", encoding="utf-8")

after = p.read_text(encoding="utf-8").splitlines()
names = [l.split()[1].split(":")[0].split("(")[0]
         for l in after if l.startswith("class ")]
print("class MatchResult present:", "MatchResult" in names)
print("total classes now:", len(names), "| file lines:", len(after))