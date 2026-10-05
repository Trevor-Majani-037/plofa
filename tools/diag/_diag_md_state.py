import json
d = json.load(open("season_state.json"))
fl = d.get("fixture_ledger", {})
print("fixture_ledger type:", type(fl).__name__)
if isinstance(fl, dict):
    for k in list(fl)[:6]:
        print(" key:", k, "->", str(fl[k])[:300])
elif isinstance(fl, list):
    print(" len", len(fl))
    for row in fl[:8]:
        print(" ", str(row)[:300])
print()
st = d.get("standings", {})
print("standings keys:", list(st)[:6])