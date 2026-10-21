"""Report the line endings of a range, line by line."""
import sys

path = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
with open(path, "rb") as f:
    raw = f.read()
lines = raw.split(b"\n")
mixed = 0
for i in range(start - 1, min(end, len(lines))):
    ends = "CRLF" if lines[i].endswith(b"\r") else "LF  "
    if not lines[i].endswith(b"\r"):
        mixed += 1
    print(f"{i+1}: {ends}  {lines[i][:60].decode('utf-8', 'replace')!r}")
print(f"\nlines in range without CR: {mixed}")
print(f"file: CRLF={raw.count(b'\\r\\n')}  bare LF={raw.count(b'\\n') - raw.count(b'\\r\\n')}")
