"""DISTANCE-COVERED GUARD -- a trace segment is never booked faster than a body
can run, and a position gap is never booked as movement at all.

THE DEFECT THIS FILE EXISTS FOR (measured, 2026-10-17)
--------------------------------------------------------
`distance_covered` in the exported stats read 14.4-20.9 km a match against a real
10.5-13.0 km, because two writers into `state.physics_distance_m` fed it:

  writer (a)  match_engine._record_offball_distance  74 km  -- the real 10 Hz
  writer (b)  event_chain._accumulate_physics_stats  405 km -- PossessionEpisode
  writer (c)  position_engine.advance_to_slots       1.6 km -- set-piece run-in

writer (b)'s `distance_m` was a GEOMETRY PATH LENGTH over the trace, and the
three race resolvers place a player by geometric interpolation toward a MOVING
target:

    pos = start + (ball_now - start) * dr,      dr = ts / arrival

so his per-tick displacement scales with the BALL's speed, not his own.
Measured on a real match (`tools/diag/_diag_trace_overcount.py`):

    mean implied speed                    33.2 m/s (120 km/h)
    metres above 8.5 m/s (30.6 km/h)        70.1%
    metres above 12 m/s (43.2 km/h)         64.8%
    share from the race notes               98.2%  (intercept_race / pursuit /
                                                     contact)

The source KNEW: the comment above the band classification said "the coarse
per-tick interpolation makes receivers/defenders jump 1-2+ m per 0.1 s tick
(an implied 10-20 m/s), which a real GPS would never see" -- and then fixed it
for SPRINT COUNT, which reads the speed label, while leaving DISTANCE reading
the geometry. The label was always the honest quantity.

WHY A NEW FILE
--------------
`tests/test_opta_analytics_runs_sprints.py` is green about this and silent on
it: it asserts that per-minute values ACCUMULATE across a match, never that the
magnitude is physical. And `calibrate_gps.py` carries the correct real-world
band (10.5-13.0 km) but is a print-only script reading a CSV last written
2026-09-02, fed by `tools/diag/_gps_verify.py`, which imports `_repro` as a
top-level module and so cannot run from the repo root. Nothing in the suite
ever compared `distance_covered` to a real benchmark.

A suite that is green and silent about a feature is not evidence about it.
"""
import math
import unittest

from possession_physics import PossessionEpisode


class TraceDistanceBookTests(unittest.TestCase):
    """Books distance from a hand-built trace, so every number is exact."""

    def _episode_with(self, rows):
        """rows: list of (tick, x, y, speed_mps) for one player.

        NOTE: `_snap` timestamps from `self.elapsed`, NOT from a `tick`
        argument. A helper that appears to pass the tick through silently
        writes every row at t=0.0, which the consecutive-tick guard then
        discards -- and every distance assertion below returns 0.0 and looks
        like a code defect rather than a broken fixture.
        """
        ep = PossessionEpisode()
        for tick, x, y, speed in rows:
            ep.elapsed = tick
            ep._snap("P", x, y, speed, note="test")
        return ep

    # ── the physical-reach ceiling ────────────────────────────────────
    def test_an_unphysical_race_interpolation_is_capped_at_human_reach(self):
        # One 0.1 s tick in which the player "moves" 12 m -> an implied
        # 120 m/s. Even at a 9 m/s label he can only have covered 0.9 m.
        ep = self._episode_with([
            (0.0, 50.0, 34.0, 9.0),
            (0.1, 62.0, 34.0, 9.0),
        ])
        s = ep.calculate_distance_stats()["P"]
        self.assertAlmostEqual(s["distance_m"], 0.9, places=6)

    def test_the_cap_uses_the_segments_own_speed_label(self):
        # Same 12 m hop, but the engine labelled it a walk.
        ep = self._episode_with([
            (0.0, 50.0, 34.0, 1.5),
            (0.1, 62.0, 34.0, 1.5),
        ])
        s = ep.calculate_distance_stats()["P"]
        self.assertAlmostEqual(s["distance_m"], 0.15, places=6)

    def test_a_standing_still_segment_books_nothing(self):
        ep = self._episode_with([
            (0.0, 50.0, 34.0, 0.0),
            (0.1, 50.0, 34.0, 0.0),
        ])
        self.assertEqual(ep.calculate_distance_stats()["P"]["distance_m"], 0.0)

    # ── NON-VACUITY: the cap must not simply zero the trace out ───────
    def test_physical_geometry_is_still_booked_in_full(self):
        # 0.5 m in 0.1 s is 5 m/s -- reachable. A 9 m/s label allows 0.9 m,
        # so the 0.5 m geometry must survive the cap untouched.
        ep = self._episode_with([
            (0.0, 50.0, 34.0, 9.0),
            (0.1, 50.5, 34.0, 9.0),
        ])
        s = ep.calculate_distance_stats()["P"]
        self.assertAlmostEqual(s["distance_m"], 0.5, places=6)
        self.assertGreater(s["distance_m"], 0.0)

    def test_a_real_carry_run_totals_its_true_length(self):
        # 40 m of dribbling at 5.6 m/s over 8 s is physically honest and must
        # come out at 40 m, not capped and not dropped.
        rows = [(0.0, 30.0, 34.0, 5.6)]
        t = 0.0
        while t < 8.0 - 1e-9:
            t = min(8.0, t + 0.1)
            rows.append((round(t, 3), 30.0 + 5.6 * t, 34.0, 5.6))
        s = self._episode_with(rows).calculate_distance_stats()["P"]
        self.assertAlmostEqual(s["distance_m"], 5.6 * 8.0, places=1)

    # ── a position gap is not movement ───────────────────────────────
    def test_a_position_gap_books_no_distance(self):
        # 30 m of elapsed time, 30 m of coordinate change: a sparse-row /
        # cross-action hop. No elapsed time means no distance to book -- the
        # same rule (and the same precedent) as `run_tracking.RunTracker`'s
        # jump filter.
        ep = self._episode_with([
            (0.0, 30.0, 34.0, 8.0),
            (30.0, 60.0, 34.0, 8.0),
        ])
        self.assertEqual(ep.calculate_distance_stats()["P"]["distance_m"], 0.0)

    def test_a_gap_still_closes_an_open_sprint_run(self):
        """Non-vacuity on the OTHER branch: the gap must not skip the
        segment-close logic, or sprint runs would never terminate.

        MIN_SEGMENT_DIST is 3.5 m and a 0.1 s tick at an 8 m/s label books at
        most 0.8 m, so a qualifying sprint needs >= 5 consecutive ticks.
        """
        rows = [(0.0, 50.0, 34.0, 8.0)]
        for i in range(1, 7):                       # 6 x 0.8 m = 4.8 m > 3.5
            rows.append((round(i * 0.1, 3), 50.0 + 0.8 * i, 34.0, 8.0))
        rows.append((5.0, 50.0 + 0.8 * 6, 34.0, 8.0))   # gap: closes the run
        s = self._episode_with(rows).calculate_distance_stats()["P"]
        self.assertGreaterEqual(s["sprint_count"], 1.0)

    # ── the real football benchmark, on the number that is exported ──
    def test_no_segment_is_booked_above_the_human_ceiling(self):
        """A whole simulated duel, swept across effort labels: no booked
        segment may imply a speed above top human range (8.5 m/s)."""
        ep = PossessionEpisode()
        # The real race shape: the resolver walks the defender along a lerp
        # toward a MOVING ball, so his per-tick step tracks the ball's speed.
        ball_speed = 26.0
        for i in range(40):
            t = i * 0.1
            ep.elapsed = t
            ep._snap("D", 80.0 - ball_speed * 0.55 * t, 34.0,
                     8.0, note="intercept_race")
        rows = defaultdict_list(ep)
        total = 0.0
        for i in range(1, len(rows)):
            x0, y0, _, t0 = rows[i - 1]
            x1, y1, s1, t1 = rows[i]
            dt = t1 - t0
            if not (0.05 <= dt <= 0.15):
                continue
            total += math.hypot(x1 - x0, y1 - y0)
        # The RAW geometry is absurd -- this is the defect, asserted so a
        # future refactor cannot quietly restore it.
        span = rows[-1][3] - rows[0][3]
        self.assertGreater(total / span, 8.5,
                           "fixture no longer reproduces the unphysical race")
        # ...and the BOOKED distance is the physically reachable one.
        s = ep.calculate_distance_stats()["D"]
        self.assertAlmostEqual(s["distance_m"], 39 * 0.8, places=1)
        self.assertLessEqual(s["distance_m"] / 3.9, 8.5)


def defaultdict_list(ep):
    from collections import defaultdict
    out = defaultdict(list)
    for row in ep.trace:
        if row.player in ("ball", "gk_track", "drift"):
            continue
        out[row.player].append((row.x, row.y, row.speed_mps, row.tick))
    return out["D"]


class DistanceBookStructureTests(unittest.TestCase):
    """STRUCTURAL guards, because the defect lived one line away from a rule
    that was already correct -- a behavioural test on the class cannot see a
    `total_dist += seg_dist` that was moved back outside the guard."""

    def _source(self):
        import inspect
        return inspect.getsource(PossessionEpisode.calculate_distance_stats)

    def test_distance_is_only_ever_accumulated_under_the_tick_guard(self):
        """Every `total_dist +=` must sit inside `if 0.05 <= dt <= 0.15:`
        (indented deeper than it). A file-wide `total_dist +=` at method-body
        indentation is the pre-fix shape."""
        import re
        src = self._source()
        adds = [ln for ln in src.splitlines() if "total_dist +=" in ln]
        self.assertTrue(adds, "no accumulation found -- anchor is stale")
        # Exactly one accumulation site, and it is indented past the `for`.
        self.assertEqual(len(adds), 1, f"expected 1 accumulation, got {adds}")
        guard = [ln for ln in src.splitlines()
                 if re.match(r"\s*if 0\.05 <= dt <= 0\.15:", ln)]
        self.assertTrue(guard, "the consecutive-tick guard is gone")
        g_indent = len(guard[0]) - len(guard[0].lstrip())
        a_indent = len(adds[0]) - len(adds[0].lstrip())
        self.assertGreater(
            a_indent, g_indent,
            "total_dist is accumulated OUTSIDE the consecutive-tick guard, so a "
            "position gap books distance again (the pre-fix shape)")

    def test_the_physical_reach_ceiling_is_present_and_not_dead(self):
        """Non-vacuity FIRST: assert a cap of this exact shape exists at all,
        so a rename cannot turn every other assertion here into a no-op."""
        import re
        src = self._source()
        cap = re.search(r"seg_dist\s*=\s*min\(\s*seg_dist\s*,\s*dt\s*\*\s*speed\s*\)", src)
        self.assertIsNotNone(
            cap, "no reach ceiling in calculate_distance_stats -- the physical "
                 "ceiling has been removed")
        # And the geometry is still computed, so the cap clamps rather than
        # replaces: a `min` needs something on the left to clamp.
        self.assertIn("seg_dist = math.hypot(", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)