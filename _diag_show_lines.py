"""Show the exact bytes of a line range (trailing whitespace, tabs, CRLF)."""
import sys

path = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
with open(path, encoding="utf-8", newline="") as f:
    lines = f.readlines()
for i in range(start - 1, min(end, len(lines))):
    raw = lines[i].rstrip("\n").rstrip("\r")
    print(f"{i+1}: {ascii(raw)}")
