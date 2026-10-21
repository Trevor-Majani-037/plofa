"""Byte-diff the one line that would not match."""
PATH = "surrogate_collect.py"

want = "        # bucket mean across *other* intents as if it was this intent's"

with open(PATH, "rb") as f:
    raw = f.read()
lines = raw.split(b"\n")
have = lines[813].rstrip(b"\r")
wb = want.encode("utf-8")

print("want:", ascii(want))
print("have:", ascii(have.decode("utf-8", "replace")))
print("len want", len(wb), "len have", len(have))
for i in range(max(len(wb), len(have))):
    a = wb[i] if i < len(wb) else None
    b = have[i] if i < len(have) else None
    if a != b:
        print(f"  first difference at {i}: want {a!r} have {b!r}")
        break
else:
    print("  identical")
