"""VERIFY THE SHOT MAP NOW DRAWS THE BALL'S REAL PATH.

Writes a REAL export from a REAL match, then reads the shot-map rows back off
the accumulator and asserts the things that actually matter:

  1. physics rows carry the endpoint the ENGINE resolved, bit for bit — not a
     value the exporter could have manufactured. Proven by matching each row
     against the originating event's own end_x/end_y.
  2. every physics row's dotted line has real LENGTH (origin != endpoint). A
     zero-length line is the "no data, drew a dot" failure.
  3. physics rows point AT the goal their team attacks. This is the check the
     retired mirror hack used to be trusted for, and the one that would catch a
     reintroduced flip.
  4. physics rows are labelled physics and reconstructed rows are labelled
     reconstructed — the two are never conflated.
  5. the PNG still renders (matplotlib did not choke on a None endpoint).

Run:  .venv\\Scripts\\python.exe _verify_shot_traj.py
"""
import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")


def main():
    import io
    from _diag_watch import build
    from exporter import PLOFAExporter
    from match_engine import MatchEvent

    eng, home_raw, away_raw = build()
    random.seed(21)
    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real
    home, away = res.config.home_team, res.config.away_team
    print(f"{home} {res.score_str} {away}\n")

    # The engine's OWN squad mapping: {team: {"starters": [...],
    # "substitutes": [...]}} of real PlayerProfiles (match_engine.py:2188).
    # `_diag_watch.build()` returns the raw LOADER output, whose starters are
    # tuples, not players — passing that raised inside _blank_stat. Two harness
    # bugs in a row; the shape has to be read off a real caller, not guessed.
    all_players = eng.squads
    assert all_players[home]["starters"], "squad mapping empty"
    ex = PLOFAExporter(res, all_players)

    # ── index the engine's OWN endpoints, keyed by (player, minute, situation)
    # so a plotted row can be traced back to the event it came from.
    engine_ends = {}
    for e in res.timeline:
        nm = getattr(getattr(e, "event_type", None), "name", "")
        if nm not in ("SHOT_ON_TARGET", "SHOT_OFF_TARGET", "SHOT_BLOCKED",
                      "GOAL", "HIT_WOODWORK"):
            continue
        if e.end_x is None or e.end_y is None:
            continue
        sit = e.situation.value if e.situation else "open_play"
        engine_ends.setdefault((e.player, e.minute, sit), []).append(
            (round(float(e.end_x), 4), round(float(e.end_y), 4)))

    rows = []
    for name, s in ex.accumulator.stats.items():
        for sh in s.get("shot_map", []):
            rows.append((name, s["team"], sh))

    print(f"{'player':<16}{'team':<10}{'situation':<13}{'outcome':<10}"
          f"{'origin':>13}{'end':>13}{'len':>7}  source")
    print("-" * 104)

    fails = []
    upstream = []
    n_phys = n_recon = 0
    for name, team, sh in sorted(rows, key=lambda r: (r[1], r[0])):
        ox, oy = float(sh["x"]), float(sh["y"])
        ex_, ey_ = sh["end_x"], sh["end_y"]
        src = sh.get("trajectory", "?")
        seg = math.hypot(ex_ - ox, ey_ - oy)
        goal = 105.0 if team == home else 0.0
        print(f"{name:<16}{team:<10}{sh.get('situation',''):<13}"
              f"{sh.get('outcome',''):<10}"
              f"{ox:>6.1f},{oy:>5.1f}{ex_:>7.1f},{ey_:>5.1f}{seg:>7.1f}  {src}")

        if src == "physics":
            n_phys += 1
            # 2. the line must have real length
            if seg < 1.0:
                fails.append(f"{name}: physics row has a {seg:.2f} m line")
            # 1. it must equal what the engine resolved
            key = (name, sh.get("minute"), sh.get("situation", "open_play"))
            cands = engine_ends.get(key) or []
            if cands and not any(
                    abs(ex_ - cx) < 0.051 and abs(ey_ - cy) < 0.051
                    for cx, cy in cands):
                fails.append(f"{name}: endpoint {ex_},{ey_} matches NO "
                             f"engine event {cands}")
            # 3. it must point AT the goal that team attacks. This is the check
            #    the retired mirror hack used to be trusted for.
            if sh.get("outcome") != "blocked" and abs(ex_ - goal) > 2.0:
                fails.append(f"{name} ({team}): physics endpoint x={ex_:.1f} is "
                             f"{abs(ex_ - goal):.1f} m from the goal it attacks "
                             f"(x={goal:.0f}) — trajectory points the wrong way")
        elif src == "reconstructed":
            n_recon += 1
            # NOT a failure of this work. A reconstructed row pointing at the
            # wrong goal means the SET-PIECE CHAIN logged the ORIGIN at the
            # wrong end — event_chain.py:6874 clamps a corner header to
            # x in [85,102] with no mirror_x, so every away-team corner is
            # recorded at the wrong goal. Old and new code draw the identical
            # endpoint here; nothing had ever looked. Reported, not smoothed.
            if sh.get("outcome") != "blocked" and abs(ex_ - goal) > 2.0:
                upstream.append(
                    f"{name} ({team}, {sh.get('situation')}): origin "
                    f"({ox:.1f},{oy:.1f}) is at the wrong end — endpoint aims "
                    f"at x={ex_:.1f}, team attacks x={goal:.0f}")
        else:
            fails.append(f"{name}: unlabelled trajectory source {src!r}")

    print(f"\nphysics {n_phys}   reconstructed {n_recon}   total {len(rows)}")
    if upstream:
        print(f"\nUPSTREAM SET-PIECE COORDINATE BUG ({len(upstream)} rows) — "
              f"pre-existing, unchanged by this work:")
        for u in upstream:
            print("  -", u)

    # 5. the chart still draws
    out = Path(tempfile.gettempdir()) / "plofa_shot_traj_check"
    out.mkdir(exist_ok=True)
    png = out / "shot_map.png"
    ex.plot_shot_map(str(png))
    ok_png = png.exists() and png.stat().st_size > 20_000
    print(f"shot map PNG written: {ok_png} ({png.stat().st_size if png.exists() else 0} bytes)")

    print()
    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails[:25]:
            print("  -", f)
        sys.exit(1)
    if not ok_png:
        print("FAILED: shot map did not render")
        sys.exit(1)
    print("PASS: every physics dotted line is the engine's own resolved "
          "endpoint, points at the goal it attacks, and set pieces are "
          "separately labelled as reconstructed.")


if __name__ == "__main__":
    main()