"""Generate one PLOFA-format export from a real match, and validate it.

Not a synthetic fixture: the real 26/27 roster, the real brains, the real
engine. An export format proven only on a hand-built document is a format
whose gaps are unknown.
"""
import json
import random
import sys

import plofa_export as px


def roster_from(eng):
    """Club -> {player name -> attributes we can honestly export}."""
    out = {}
    for team in (eng.config.home_team, eng.config.away_team):
        d = {}
        for p in eng.active_players[team]:
            d[p.name] = {
                "name": p.name,
                "position": getattr(p, "position", None),
                "soul_archetype": getattr(getattr(p, "soul", None),
                                          "archetype", None),
                "jersey_number": getattr(p, "jersey_number", None),
            }
        out[team] = d
    return out


def main():
    sys.path.insert(0, ".")
    from _diag_watch import build

    eng, hr, ar = build()
    eng.enable_virtual_gps(0.1)
    random.seed(31)
    import io
    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real

    doc = px.export_match(res, engine=eng, roster=roster_from(eng))
    problems = px.validate(doc)

    print(f"SCORE {res.score_str}")
    print(f"events exported : {doc['index']['n_events']}")
    print(f"file size       : "
          f"{len(json.dumps(doc))/1024/1024:.2f} MB")
    print(f"validation      : "
          f"{'CLEAN' if not problems else str(len(problems)) + ' PROBLEMS'}")
    for p in problems[:10]:
        print("   !", p)

    print("\nframe:", doc["frame"]["name"], "-", doc["frame"]["origin"])
    print("\nby type (top 18):")
    for k, v in list(doc["index"]["by_type"].items())[:18]:
        print(f"   {k:<22}{v:>6}")
    print("\ndecision authority:")
    for k, v in doc["index"]["decision_authority"].items():
        print(f"   {k:<22}{v:>6}")
    print("\non-ball intent:")
    for k, v in list(doc["index"]["on_ball_intent"].items())[:12]:
        print(f"   {k:<22}{v:>6}")
    if "profiles" in doc:
        for key in doc["profiles"]:
            print(f"\n{key} totals:")
            for k, v in list(doc["profiles"][key]["totals"].items())[:10]:
                print(f"   {k:<22}{v:>6}")
    print("\ngaps (stated, not hidden):")
    for g in doc["gaps"]:
        print("   -", g)

    p = px.write_json(doc, "plofa_output/_format/Oxton_vs_Natrican.json")
    print(f"\nwrote {p}  ({p.stat().st_size/1024/1024:.2f} MB)")

    # show one pass in full — this is what a consumer actually reads
    for e in doc["events"]:
        if e["type"] == "PASS" and e.get("decision", {}).get("intent") == \
                "PROGRESSIVE_PASS":
            print("\n--- one PROGRESSIVE_PASS, verbatim ---")
            print(json.dumps(e, indent=1, ensure_ascii=False))
            break


if __name__ == "__main__":
    main()
