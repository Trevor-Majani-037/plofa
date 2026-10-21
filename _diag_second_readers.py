"""Who reads an event's `second`?  (The grep keeps matching `secondary_player`.)"""
import re
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.readlines()
    print(f"=== {path}")
    pat = re.compile(r"(?<![a-zA-Z_])(second|Second)(?![a-zA-Z_])")
    for i, line in enumerate(lines, 1):
        if pat.search(line):
            print(f"  {i}: {line.rstrip()[:110]}")
