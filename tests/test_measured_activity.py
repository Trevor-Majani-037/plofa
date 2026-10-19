"""MEASURED ACTIVITY — the published seconds and distance must be the MEASUREMENT,
not the per-position table (2026-10-17).

WHAT WAS WRONG
--------------
`standing_seconds` .. `sprinting_seconds` were `opta_analytics.BASE_MINUTE_ACTIVITY`
— a hand-authored per-position table — scaled by `work_rate`, perturbed by
telemetry, then renormalised to exactly 60 s. `distance_covered` was that
table's distance plus a fraction of the physics trace.

The table had to INVENT its standing share (16-83% by position) because
**standing time was not measurable at all**: `MatchEngine._record_offball_distance`
opened with

    if moved <= 0:
        return

`_offball_move_player` has an `arrive` dead band, so a player settled at his
shape target moves 0 m on the tick — and his time was discarded before any
measurement. Measured on a real match (`_diag_standing_gap.py`): **306,504 of
666,476 stepped ticks dropped, 46%, ~1,533 s per player, 26.6% of a match.**

A second hypothesis was refuted on the way and is worth not re-deriving: it is
NOT that the engine simulates less than 90'. `_diag_match_clock_budget.py`
measured 5,782 s simulated against 5,760 s nominal (100.4%) and integrator
budgets of 5,570 s = 96.3% of the clock. The clock is fine; the WRITER was not.

WHY A NEW FILE. `test_opta_analytics_runs_sprints.py` is green about
`distance_covered` and silent on what it is made of — it asserts per-minute
values ACCUMULATE, never that the magnitude is a measurement. A suite that is
green and silent about a feature is not evidence about it.
"""
import unittest
from types import SimpleNamespace

from match_engine import MatchEngine
from opta_analytics import OptaAnalytics, SPEEDS
import position_engine as PE

BANDS = ("standing", "walking", "jogging", "running", "sprinting")


def _player(name, position="CM", work_rate=50.0, pace=60.0):
    return SimpleNamespace(name=name, position=position, dna=SimpleNamespace(
        preferred_foot="right", physical=SimpleNamespace(pace=pace),
        mental=SimpleNamespace(work_rate=work_rate)))


def _row(name, position="CM", minute=1, dist=0.0, band_s=(0.0, 0.0, 0.0, 0.0, 0.0),
         sprint=0.0, high=0.0, top=0.0, touches=0, touch_d=0.0, drift_d=0.0):
    return {
        "player": name, "position": position, "x": 52.5, "y": 34.0,
        "distance_touch": touch_d, "distance_drift": drift_d,
        "distance_total": touch_d + drift_d, "touches": touches,
        "peak_touch_jump": 0.0,
        "physics_distance_m": dist,
        "physics_standing_time_s": band_s[0],
        "physics_walk_time_s": band_s[1],
        "physics_jog_time_s": band_s[2],
        "physics_running_time_s": band_s[3],
        "physics_sprint_time_s": band_s[4],
        "physics_sprint_count": sprint,
        "physics_high_speed_sprint_count": high,
        "physics_top_speed_mps": top,
    }


def _result(frames):
    return SimpleNamespace(
        config=SimpleNamespace(home_team="Home", away_team="Away"),
        timeline=[], position_log=frames,
        momentum_log=[{"minute": m, "momentum": 0.0, "home_goals": 0,
                       "away_goals": 0} for m in range(1, 91)],
    )


def _analytics(frames, player=None):
    p = player or _player("P")
    a = OptaAnalytics(_result(frames),
                      {"Home": {"starters": [p], "substitutes": []}})
    a.compute()
    return a


class ActivityBandTests(unittest.TestCase):
    """The band edges are Di Salvo's published GPS thresholds, so the buckets
    mean something outside this repo rather than being tuned to make a total
    come out right."""

    def test_non_vacuity_the_five_bands_exist_and_are_ordered(self):
        self.assertEqual(len(PE.ACTIVITY_BAND_NAMES), 5)
        self.assertEqual(list(PE.ACTIVITY_BAND_NAMES),
                         ["standing", "walk", "jog", "running", "sprint"])
        edges = list(PE.ACTIVITY_BAND_EDGES)
        self.assertEqual(edges, sorted(edges))
        self.assertEqual(len(edges), 4)

    def test_the_edges_are_di_salvos_published_thresholds(self):
        # low <2.0 m/s, sprint >7.0 m/s (>25.2 km/h) -- the boundaries
        # calibrate_gps.py already cites for this project.
        self.assertEqual(PE.ACTIVITY_BAND_EDGES, (0.5, 2.0, 4.2, 7.0))
        self.assertEqual(PE.activity_band_index(1.99), 1)   # walking
        self.assertEqual(PE.activity_band_index(7.0), 4)    # sprinting

    def test_time_is_banded_into_exactly_one_band_and_not_lost(self):
        """`duration_s` must land in ONE band and the five must sum to it."""
        st = PE.PlayerSpatialState(player_name="X", position="CM", team="T",
                               home_x=50.0, home_y=34.0)
        eng = PE.PositionEngine.__new__(PE.PositionEngine)
        eng.states = {"X": st}
        for speed in (0.1, 1.0, 3.0, 5.0, 9.0):
            PE.PositionEngine.record_physics_distance(
                eng, "X", distance_m=0.0, duration_s=0.5, speed_mps=speed)
        total = (st.physics_standing_time_s + st.physics_walk_time_s
                 + st.physics_jog_time_s + st.physics_running_time_s
                 + st.physics_sprint_time_s)
        self.assertAlmostEqual(total, 2.5, places=6)
        self.assertAlmostEqual(st.physics_standing_time_s, 0.5, places=6)
        self.assertAlmostEqual(st.physics_sprint_time_s, 0.5, places=6)

    def test_an_explicit_band_is_not_double_counted(self):
        """A caller that already knows the band passes duration_s=0.0; that is
        how the episode trace avoids booking the same second twice."""
        st = PE.PlayerSpatialState(player_name="X", position="CM", team="T",
                               home_x=50.0, home_y=34.0)
        eng = PE.PositionEngine.__new__(PE.PositionEngine)
        eng.states = {"X": st}
        PE.PositionEngine.record_physics_distance(
            eng, "X", distance_m=0.0, duration_s=0.0, speed_mps=0.0,
            sprint_time_s=7.0)
        self.assertAlmostEqual(st.physics_sprint_time_s, 7.0, places=6)
        self.assertEqual(st.physics_standing_time_s, 0.0)


class StandingTimeIsRecordedTests(unittest.TestCase):
    """THE HEADLINE FIX: `moved == 0` is standing, not absence."""

    def test_a_stationary_tick_is_booked_rather_than_discarded(self):
        st = PE.PlayerSpatialState(player_name="X", position="CM", team="T",
                               home_x=50.0, home_y=34.0)
        eng = MatchEngine.__new__(MatchEngine)
        eng.position_engine = PE.PositionEngine.__new__(PE.PositionEngine)
        eng.position_engine.states = {"X": st}
        eng._sprint_state = {}
        # a settled player: 0 m moved at 0.2 m/s
        MatchEngine._record_offball_distance(eng, "X", 0.0, 0.2, 8.0, 0.1)
        self.assertAlmostEqual(st.physics_standing_time_s, 0.1, places=6)
        self.assertAlmostEqual(st.physics_distance_m, 0.0, places=6)

    def test_a_moving_tick_still_records_distance_and_sprints(self):
        st = PE.PlayerSpatialState(player_name="X", position="CM", team="T",
                               home_x=50.0, home_y=34.0)
        eng = MatchEngine.__new__(MatchEngine)
        eng.position_engine = PE.PositionEngine.__new__(PE.PositionEngine)
        eng.position_engine.states = {"X": st}
        eng._sprint_state = {}
        for _ in range(3):
            MatchEngine._record_offball_distance(eng, "X", 0.8, 8.0, 8.0, 0.1)
        self.assertAlmostEqual(st.physics_distance_m, 2.4, places=6)
        self.assertAlmostEqual(st.physics_sprint_time_s, 0.3, places=6)
        self.assertEqual(st.physics_sprint_count, 1.0)

    def test_the_writer_source_has_no_bare_early_return_on_moved(self):
        """Structural: `if moved <= 0:` must be followed by a RECORD, not a bare
        `return`. A behavioural test cannot see a re-introduced discard if the
        discard is moved after the record.

        The docstring QUOTES `if moved <= 0: return` in order to explain the
        defect, so the DOCSTRING is stripped before the scan -- a match inside
        prose is not a call site (the same reason
        `test_momentum_model` uses AST rather than a substring scan).
        """
        import inspect
        import re
        src = inspect.getsource(MatchEngine._record_offball_distance)
        # strip the docstring
        m = re.search(r'"""(.*?)"""', src, re.S)
        self.assertIsNotNone(m, "no docstring found -- anchor is stale")
        code = src[:m.start()] + src[m.end():]
        self.assertIn("if moved <= 0:", code)
        head = code.split("if moved <= 0:", 1)[1].split("return", 1)[0]
        self.assertIn("record_physics_distance", head,
                      "the moved<=0 branch returns WITHOUT recording — standing "
                      "time is unmeasurable again")


class PublishedSecondsAreMeasuredTests(unittest.TestCase):
    """With full coverage the published seconds ARE the measurement. There is
    no template left to hide behind, so this is exact."""

    def _one_minute_full_coverage(self, band_s, dist):
        return [{"minute": 1, "home": [_row("P", band_s=band_s, dist=dist)],
                 "away": [], "home_goals": 0, "away_goals": 0,
                 "possession_team": "Home", "phase": "open_play"}]

    def test_full_coverage_publishes_the_measured_seconds_exactly(self):
        band_s = (10.0, 25.0, 15.0, 8.0, 2.0)     # sums to 60
        a = _analytics(self._one_minute_full_coverage(band_s, 90.0))
        act = a._activity["P"]
        for i, band in enumerate(BANDS):
            self.assertAlmostEqual(act[band], band_s[i], places=6,
                                   msg=f"{band} is not the measured value")
        self.assertAlmostEqual(sum(act.values()), 60.0, places=6)

    def test_full_coverage_publishes_the_measured_distance_exactly(self):
        a = _analytics(self._one_minute_full_coverage(
            (10.0, 25.0, 15.0, 8.0, 2.0), 90.0))
        # 90.0 m measured over a fully covered minute, and NO template is added
        # because there is no uncovered second left to describe.
        self.assertAlmostEqual(a._distances["P"], 90.0, places=6)

    def test_partial_coverage_adds_only_the_unwatched_residue(self):
        # 30 s measured, 30 s unwatched
        band_s = (5.0, 12.0, 7.0, 4.0, 2.0)      # 30 s
        a = _analytics(self._one_minute_full_coverage(band_s, 45.0))
        act = a._activity["P"]
        self.assertAlmostEqual(sum(act.values()), 60.0, places=6)
        for band in BANDS:
            self.assertGreaterEqual(act[band], 0.0)
            # the measured bands are never overwritten downward
            i = BANDS.index(band)
            self.assertGreaterEqual(act[band], band_s[i] - 1e-9)
        self.assertAlmostEqual(a.player_data["P"]["activity_measured_pct"], 50.0,
                               places=1)

    def test_the_distance_is_measured_plus_residue_never_template_plus_trace(self):
        """The old shape was `template_distance + fraction_of_trace`. With full
        coverage the template term must be exactly zero, so a frame that is
        fully measured publishes exactly the measured metres."""
        a = _analytics(self._one_minute_full_coverage(
            (10.0, 25.0, 15.0, 8.0, 2.0), 137.0))
        self.assertAlmostEqual(a._distances["P"], 137.0, places=6)
        # and a 2-minute version, to prove it is not a single-frame artefact
        frames = [{"minute": m, "home": [_row("P", band_s=(10.0, 25.0, 15.0, 8.0, 2.0),
                                            dist=137.0)],
                   "away": [], "home_goals": 0, "away_goals": 0,
                   "possession_team": "Home", "phase": "open_play"}
                  for m in (1, 2)]
        a2 = _analytics(frames)
        self.assertAlmostEqual(a2._distances["P"], 274.0, places=6)

    def test_no_coverage_falls_back_to_the_table_and_says_so(self):
        a = _analytics(self._one_minute_full_coverage((0.0, 0.0, 0.0, 0.0, 0.0), 0.0))
        pd = a.player_data["P"]
        self.assertEqual(pd["activity_measured_pct"], 0.0)
        self.assertIn("template", pd["activity_source"])
        self.assertAlmostEqual(sum(a._activity["P"].values()), 60.0, places=6)

    def test_the_measured_share_is_published_so_nothing_is_mislabelled(self):
        a = _analytics(self._one_minute_full_coverage(
            (10.0, 25.0, 15.0, 8.0, 2.0), 90.0))
        pd = a.player_data["P"]
        self.assertEqual(pd["activity_measured_pct"], 100.0)
        self.assertEqual(pd["activity_source"], "measured")


if __name__ == "__main__":
    unittest.main(verbosity=2)