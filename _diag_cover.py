"""THE COVER — what the back line actually does when a CB is pressed.

The user watched matches and reported: when a CB has the ball, the other CB
does not come across to cover him, and the CDM's position is unclear.  The
football says the cover should be ~6 m away (the second-defender distance),
roughly level in x, and that the CB-to-midfield line distance should be
~16 m.

This probe measures the CURRENT behaviour, so the gap has a number and the
fix can be gated on it.  It samples the WHOLE shape at ONE instant per tick
(one dict per `_offball_run` call — both flanks same-instant by construction),
and it separates the two situations the engine must tell apart:

  * CARRIER-IS-A-CB   our CB has the ball and an opponent is on him
  * BUILD-UP          a midfielder has the ball and our back line is spread
                      (this one is CORRECT today and must not regress)

The carrier is identified as the team-mate nearest the ball (the engine
tracks the ball but not who holds it), and the presser as the opponent
nearest the carrier.

Usage:
    python _diag_cover.py [--seed 777]
"""
import argparse
import collections
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import match_engine as ME  # noqa: E402

SAMPLES = []
REJECT = collections.Counter()
COVER_CALLS = []


def _wrap_cover():
    """Instrument the cover constraint: does it fire, and how far is the
    cover from the socket it was given?"""
    import position_engine as PE
    orig = PE.PositionEngine.cover_target

    def patched(self, team_name, ball_x, ball_y, has_ball, attacks_right=True):
        out = orig(self, team_name, ball_x, ball_y, has_ball, attacks_right)
        if out:
            for name, (tx, ty) in out.items():
                st = self.states.get(name)
                if st is None or st.current_x is None:
                    continue
                COVER_CALLS.append({
                    "lag": ((st.current_x - tx) ** 2
                            + (st.current_y - ty) ** 2) ** 0.5,
                    "socket_x": tx, "socket_y": ty,
                    "pos_x": st.current_x, "pos_y": st.current_y,
                })
        return out

    PE.PositionEngine.cover_target = patched
    return orig


def _sample(self, home_has_ball):
    """Called after every off-ball tick: one whole-shape snapshot."""
    try:
        REJECT["calls"] += 1
        pe = self.position_engine
        ball_x = self.state.last_ball_x
        ball_y = self.state.last_ball_y
        team = self.config.home_team if home_has_ball else self.config.away_team
        other = self.config.away_team if home_has_ball else self.config.home_team

        ours = {}
        for n in pe.team_rosters.get(team, []):
            stt = pe.states.get(n)
            if stt is None:
                continue
            p = getattr(stt, "position", "")
            if p == "GK":
                continue
            ours[n] = (p, stt.current_x, stt.current_y)
        theirs = {}
        for n in pe.team_rosters.get(other, []):
            stt = pe.states.get(n)
            if stt is None:
                continue
            p = getattr(stt, "position", "")
            if p == "GK":
                continue
            theirs[n] = (p, stt.current_x, stt.current_y)
        if not ours or not theirs:
            REJECT["no_teams"] += 1
            return

        cbs = [n for n, v in ours.items() if v[0] == "CB"]
        cdms = [n for n, v in ours.items() if v[0] == "CDM"]
        if len(cbs) < 2:
            REJECT["fewer_than_2_cbs"] += 1
            return

        # the carrier: our nearest team-mate to the ball
        carrier = min(ours, key=lambda n: (ours[n][1] - ball_x) ** 2
                      + (ours[n][2] - ball_y) ** 2)
        c_d = ((ours[carrier][1] - ball_x) ** 2
               + (ours[carrier][2] - ball_y) ** 2) ** 0.5
        if c_d > 2.5:
            REJECT["ball_not_at_a_teammates_feet"] += 1
            return   # the ball is not at a team-mate's feet: not our moment

        # the presser: the opponent nearest the carrier
        presser = min(theirs, key=lambda n: (theirs[n][1] - ours[carrier][1]) ** 2
                      + (theirs[n][2] - ours[carrier][2]) ** 2)
        p_d = ((theirs[presser][1] - ours[carrier][1]) ** 2
               + (theirs[presser][2] - ours[carrier][2]) ** 2) ** 0.5

        cover = [n for n in cbs if n != carrier]
        if not cover:
            return
        cover = cover[0]
        cx, cy = ours[cover][1], ours[cover][2]
        kx, ky = ours[carrier][1], ours[carrier][2]

        # is the cover on the ball side of the carrier?  (the cover comes
        # ACROSS: toward the centre from a wide carrier, and toward the
        # carrier's side from a central one)
        to_centre = (34.0 - ky)
        cover_rel = (cy - ky) * (1 if to_centre >= 0 else -1)

        rec = {
            "case": "CB_CARRIER" if ours[carrier][0] == "CB" else "MID_CARRIER",
            "pressed": p_d <= 8.0,
            "press_d": p_d,
            "x_gap": abs(cx - kx),
            "y_gap": abs(cy - ky),
            "dist": ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5,
            "across": cover_rel,     # >0 = the cover came toward the centre
        }
        if cdms:
            dx, dy = ours[cdms[0]][1], ours[cdms[0]][2]
            rec["cdm_to_carrier"] = ((dx - kx) ** 2 + (dy - ky) ** 2) ** 0.5
            # the CB line is the mean of the two CBs' depth; the midfield
            # line is the CDM's depth.  The gap between the LINES is what the
            # football's 16 m rule is about.
            rec["line_gap"] = abs(((kx + cx) / 2.0) - dx)
        SAMPLES.append(rec)
    except Exception:
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds; the probe runs one match per "
                         "seed and aggregates")
    ap.add_argument("--off", action="store_true",
                    help="measure with the whole cover/screen feature OFF — the "
                         "'before' arm, with the identical instrument")
    args = ap.parse_args()
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else [args.seed])

    orig = ME.MatchEngine._offball_run
    orig_cont = ME.MatchEngine._continuous_offball_phase
    orig_cover = _wrap_cover()

    def patched(self, duration_s, home_has_ball):
        orig(self, duration_s, home_has_ball)
        _sample(self, home_has_ball)

    def patched_cont(self, minute, home_has_ball, *a, **kw):
        orig_cont(self, minute, home_has_ball, *a, **kw)
        _sample(self, home_has_ball)

    ME.MatchEngine._offball_run = patched
    ME.MatchEngine._continuous_offball_phase = patched_cont
    try:
        import random as _r
        import numpy as _np
        import run_match
        for _s in seeds:
            _r.seed(_s)
            _np.random.seed(_s)
            if args.off:
                import position_engine as _PE
                _PE.PositionEngine.COVER_ENABLED = False
                _PE.PositionEngine.SCREEN_ENABLED = False
                _PE.PositionEngine.BACKLINE_SPLIT_SKIPS_CB_CARRIER = False
            run_match.run()
            print(f"[seed {_s}] samples so far: {len(SAMPLES)}")
    except Exception as exc:
        print(f"match failed: {exc}")
    finally:
        ME.MatchEngine._offball_run = orig
        ME.MatchEngine._continuous_offball_phase = orig_cont
        import position_engine as PE
        PE.PositionEngine.cover_target = orig_cover

    out = []
    def w(line=""):
        out.append(line)
        print(line)

    w(f"samples: {len(SAMPLES)}")
    w(f"rejections: {dict(REJECT)}")
    w("")
    if COVER_CALLS:
        lags = [c["lag"] for c in COVER_CALLS]
        w(f"=== the cover CONSTRAINT: fired {len(COVER_CALLS)} times ===")
        w(f"  the cover's distance from the socket he was given: "
          f"median {st.median(lags):.1f} m, max {max(lags):.1f} m")
        w("  (a small lag = he is walking to the socket; a large one = "
          "something else is moving him)")
    else:
        w("=== the cover CONSTRAINT NEVER FIRED ===")
    for case in ("CB_CARRIER", "MID_CARRIER"):
        for pressed in (True, False):
            rows = [s for s in SAMPLES
                    if s["case"] == case and s["pressed"] == pressed]
            if not rows:
                continue
            w("")
            w(f"=== {case}, pressed={pressed}  (n={len(rows)}) ===")
            w(f"  CB pair x-gap     : median {st.median(r['x_gap'] for r in rows):5.1f} m   "
              f"(the football: the cover is roughly LEVEL -> a few metres)")
            w(f"  CB pair y-gap     : median {st.median(r['y_gap'] for r in rows):5.1f} m")
            w(f"  cover distance    : median {st.median(r['dist'] for r in rows):5.1f} m   "
              f"(the football: ~6 m)")
            within6 = 100.0 * sum(1 for r in rows if r["dist"] <= 6.0) / len(rows)
            w(f"  cover within 6 m  : {within6:5.1f}%")
            level = 100.0 * sum(1 for r in rows if r["x_gap"] <= 3.0) / len(rows)
            w(f"  cover level (x<=3): {level:5.1f}%   "
              f"(the user's observation: within +/-3 m)")
            across = 100.0 * sum(1 for r in rows if r["across"] > 0) / len(rows)
            w(f"  cover came ACROSS : {across:5.1f}%")
            if "cdm_to_carrier" in rows[0]:
                w(f"  CDM to carrier    : median {st.median(r['cdm_to_carrier'] for r in rows):5.1f} m")
                w(f"  CB line -> CDM    : median {st.median(r['line_gap'] for r in rows):5.1f} m   "
                  f"(the football: ~16 m)")

    with open("_diag_cover.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n[written to _diag_cover.txt]")


if __name__ == "__main__":
    main()
