"""Normalise the appended AGENTS.md block to the file's CRLF endings.

The section was appended with PowerShell's `Out-File -Append`, which wrote
LF-only lines into a CRLF file.  The blank line before the heading was also
eaten.  This rewrites ONLY the appended tail, and asserts that the head is
untouched apart from the one blank line.

Run:  .venv\\Scripts\\python.exe -u _diag_fix_agents_eol.py
"""
import re

P = "AGENTS.md"
MARKER = b"## THE FANS"

with open(P, "rb") as f:
    raw = f.read()

i = raw.find(MARKER)
if i < 0:
    raise SystemExit("marker not found")
head, tail = raw[:i], raw[i:]

before = (tail.count(b"\r\n"), tail.count(b"\n") - tail.count(b"\r\n"))

# 1. the missing blank line before the heading
head2 = head.replace(b'0.0" red flag.\r\n', b'0.0" red flag.\r\n\r\n')
if head2 == head:
    print("the blank line was already there (or the anchor moved)")

# 2. the tail to CRLF
tail2 = tail.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")

after = (tail2.count(b"\r\n"), tail2.count(b"\n") - tail2.count(b"\r\n"))
with open(P, "wb") as f:
    f.write(head2 + tail2)
print(f"tail endings before CRLF={before[0]} LF={before[1]} "
      f"-> after CRLF={after[0]} LF={after[1]}")
print(f"head {len(head)} -> {len(head2)} bytes")
