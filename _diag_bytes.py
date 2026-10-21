"""Byte-level view of one line (to settle an encoding question)."""
import sys

path = sys.argv[1]
lineno = int(sys.argv[2])
with open(path, "rb") as f:
    raw = f.read()
lines = raw.split(b"\n")
line = lines[lineno - 1]
print("bytes:", line)
for i, b in enumerate(line):
    if b > 127:
        print(f"  byte {i}: 0x{b:02x}")
try:
    print("utf-8:", line.decode("utf-8"))
except UnicodeDecodeError as e:
    print("utf-8 DECODE FAILED:", e)
    print("cp1252:", line.decode("cp1252"))
