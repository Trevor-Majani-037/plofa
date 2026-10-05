"""Read the StatsBomb reference file and report its SCHEMA, not its contents.

Pointing this at a real StatsBomb open-data file is the whole point: the
schema has to be read, not remembered. Guessing a provider's taxonomy is how
you ship an export that looks right and is wrong in thirty places.

Reports:
  * top-level blocks and their sizes
  * every event type with its id and count
  * the union of keys used by each event type
  * the nested sub-objects and their keys
  * the match / lineup block shapes
  * the coordinate frame, taken from the data rather than assumed
"""
import collections
import json
import pathlib
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else r"D:\Downloads ⬇️\Statsbomb Date\15956.json"


def main():
    p = pathlib.Path(SRC)
    if not p.exists():
        cands = list(pathlib.Path(r"D:\Downloads").rglob("15956.json"))
        if not cands:
            print("NOT FOUND")
            return
        p = cands[0]
    print(f"FILE: {p}\nSIZE: {p.stat().st_size/1024/1024:.1f} MB")

    d = json.loads(p.read_text(encoding="utf-8"))
    # StatsBomb open data normally ships as a 4-block object
    # (match / lineup / events / live_match_periods). This reference is a BARE
    # EVENTS ARRAY, which is the more useful reference for the part we actually
    # have to synthesise. Handle both and say which one we got.
    if isinstance(d, list):
        print("SHAPE: bare events array (no match/lineup blocks)")
        evs = d
        blocks = {}
    else:
        print("SHAPE: full StatsBomb object")
        print(f"TOP-LEVEL: {', '.join(d.keys())}")
        for k, v in d.items():
            print(f"   {k:<22} {type(v).__name__:<6} "
                  f"len={len(v) if hasattr(v, '__len__') else '-'}")
        evs = d["events"]
        blocks = d

    print(f"\nEVENT COUNT: {len(evs)}")
    types = collections.Counter((e["type"]["id"], e["type"]["name"]) for e in evs)
    print(f"EVENT TYPES ({len(types)} distinct)")
    for (tid, name), n in sorted(types.items()):
        print(f"   id={tid:<4} {name:<26} {n:>5}")

    print("\nTOP-LEVEL EVENT KEYS (union, and how many events carry each)")
    keys = collections.Counter(k for e in evs for k in e)
    for k, n in sorted(keys.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<24} {n:>5}")

    print("\nNESTED SUB-OBJECTS: keys per sub-object name")
    sub = collections.defaultdict(collections.Counter)
    for e in evs:
        for k, v in e.items():
            if isinstance(v, dict) and v:
                sub[k].update(v.keys())
    for name in sorted(sub):
        ks = ", ".join(f"{a}({b})" for a, b in sub[name].most_common())
        print(f"   {name}: {ks}")

    print("\nSAMPLE: one of each event type, keys only")
    seen = set()
    for e in evs:
        nm = e["type"]["name"]
        if nm in seen:
            continue
        seen.add(nm)
        top = [k for k in e if k not in ("type",)]
        print(f"   {nm:<24} {', '.join(top)}")

    print("\nCOORDINATE FRAME (from the data, not assumed)")
    xs, ys = [], []
    for e in evs:
        loc = e.get("location")
        if loc and len(loc) >= 2:
            xs.append(loc[0]); ys.append(loc[1])
    if xs:
        print(f"   x: min={min(xs):.2f} max={max(xs):.2f}")
        print(f"   y: min={min(ys):.2f} max={max(ys):.2f}")
        print(f"   n_with_location={len(xs)} of {len(evs)}")
    ex = [e for e in evs if e["type"]["name"] == "Pass" and e.get("location")]
    if ex:
        e = ex[0]
        print(f"\n   example Pass: from={e['location']} to={e['pass'].get('end_location')}")
        print(f"     pass keys: {', '.join(e['pass'].keys())}")
        print(f"     pass_reception keys: {', '.join((e['pass'].get('pass_reception') or {}).keys())}")

    m = (blocks.get("match") or [{}])[0]
    if m:
        print("\nMATCH BLOCK")
        print("   " + ", ".join(m.keys()))
        print(f"   home={m.get('home_team')} away={m.get('away_team')} "
              f"score={m.get('home_score')}-{m.get('away_score')}")
    lu = (blocks.get("lineup") or [{}])[0]
    if lu:
        print("\nLINEUP BLOCK (first team)")
        print("   " + ", ".join(lu.keys()))
        print(f"   team: {lu.get('team',{}).get('name')} "
              f"n_players={len(lu.get('players',[]))}")
        if lu.get("players"):
            p0 = lu["players"][0]
            print(f"   player keys: {', '.join(p0.keys())}")
            print(f"   sample: {p0.get('player_name')} pos={p0.get('position')} "
                  f"jersey={p0.get('jersey_number')}")

    # The taxonomy, as a directly reusable table.
    print("\nTYPE TABLE (copy this verbatim into the exporter)")
    for tid, name in sorted(types):
        print(f'    ("SB_{name.upper().replace(" ", "_")}", {tid}, "{name}"),')


if __name__ == "__main__":
    main()
