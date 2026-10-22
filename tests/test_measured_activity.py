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
import math
import random
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


class BlockLineIsHeldNotFrozenTests(unittest.TestCase):
    """A defender holding a block SHUFFLES along his line; he does not freeze.

    Before 2026-10-19 the arrived branch set `moved = 0.0` (after computing a
    step it then discarded), so 98% of dead-phase block ticks were booked as
    standing time: 234,157 of 239,409. These tests pin the replacement, and
    the one property the first replacement got wrong: chasing a circling
    point at a pace above the cap made the man CRAWL whenever the point swung
    back, which measures as standing even though he is moving.
    """

    DT = 0.1
    BUDGET = 1.5 * 0.55 * 0.1       # a CM's per-tick cap, 0.0825 m

    def _sweep(self, n=400, cy=34.0, ty=34.0, tx=50.0, cx=50.0,
               radius=1.0, rate=1.0, budget=None):
        b = self.BUDGET if budget is None else budget
        rows = []
        for i in range(n):
            nx, ny, moved = MatchEngine._block_shuffle_target(
                cx, cy, tx, ty, i * rate * self.DT, rate, radius, b)
            rows.append((nx, ny, moved))
            cx, cy = nx, ny
        return rows

    def test_the_shuffle_holds_a_constant_pace_instead_of_crawling(self):
        wave = 4.0 * 1.0 * 1.0 / (2.0 * math.pi) * self.DT      # 0.0637 m
        moves = [m for _, _, m in self._sweep()]
        self.assertGreater(sum(moves), 10.0, "the shuffle never moved at all")
        # The circle pursuit this replaced fell to 0.1-0.5 m/s on about a
        # QUARTER of ticks. A triangle wave is slow only on the reversal tick
        # itself, once per half period (pi/rate = 3.1 s = ~31 ticks).
        slow = [m for m in moves if m < wave - 1e-9]
        self.assertLess(len(slow) / len(moves), 0.05,
                        f"{len(slow)}/{len(moves)} ticks crawl — the wave is "
                        f"degenerating into a pursuit")
        # A slow tick is a REVERSAL, not a freeze: it is isolated (never two
        # in a row) and it is a negligible share of the ground covered.
        idx = [i for i, m in enumerate(moves) if m < wave - 1e-9]
        for a, b in zip(idx, idx[1:]):
            self.assertGreater(b - a, 1, "two consecutive slow ticks = a stall")
        self.assertLess(sum(slow) / sum(moves), 0.02)
        # Once on the wave, the pace is EXACTLY the wave's pace (the early
        # ticks run at the cap while catching up to it). Reversals excluded:
        # they are the isolated ticks just accounted for.
        steady = [m for m in moves[120:] if m >= wave - 1e-9]
        self.assertGreater(len(steady), 200)
        for m in steady:
            self.assertAlmostEqual(m, wave, places=9)
        self.assertGreater(wave / self.DT, PE.ACTIVITY_BAND_EDGES[0],
                           "the pace must be above the standing edge")

    def test_the_shuffle_stays_inside_the_radius_of_the_slot(self):
        for _, ny, _ in self._sweep():
            self.assertLessEqual(abs(ny - 34.0), 1.0 + 1e-9)

    def test_depth_is_only_closed_and_the_step_is_capped(self):
        # 10 m away in x, settled in y: every step is within the cap and the
        # depth error shrinks monotonically.
        rows = self._sweep(n=400, cx=40.0, tx=50.0)
        for _, _, m in rows:
            self.assertLessEqual(m, self.BUDGET + 1e-9)
        xs = [x for x, _, _ in rows]
        self.assertEqual(xs, sorted(xs), "the depth closure went backwards")
        # the y shuffle always takes priority; the remainder closes the gap
        self.assertLess(50.0 - xs[-1], 5.0, "depth error is not being closed")
        self.assertLess(50.0 - xs[-1], 50.0 - xs[0])

    def test_a_tight_cap_is_respected(self):
        for _, _, m in self._sweep(budget=0.01):
            self.assertLessEqual(m, 0.01 + 1e-9)

    def test_the_shuffle_consumes_no_global_randomness(self):
        state = random.getstate()
        self._sweep(n=50)
        self.assertEqual(random.getstate(), state)

    def test_the_live_block_branch_does_not_hard_zero_the_step(self):
        """Structural: the arrived branch must BOOK movement. A behavioural
        test of the helper cannot see a re-introduced `moved = 0.0` at the
        call site."""
        import inspect
        src = inspect.getsource(MatchEngine._offball_move_player)
        self.assertIn("_block_shuffle_target", src,
                      "the block branch no longer calls the shuffle helper")
        # the LAST `if _in_block:` is the arrived/shuffle branch (an earlier
        # one gates the move toward the slot)
        block = src.rsplit("if _in_block:", 1)[1]
        block = block.split("\n            else:", 1)[0]
        self.assertIn("_block_shuffle_target", block,
                      "anchor is stale — the shuffle call moved out of the "
                      "block branch")
        self.assertNotIn("moved = 0.0", block,
                         "the block branch freezes the defender again")


class DeadPhaseStillStepsTheOnBallManTests(unittest.TestCase):
    """The sticky per-minute on-ball set is right for the LIVE window (the
    possession trace covers it) and wrong for the DEAD window (nothing does).
    Excluding an on-ball player there froze him for the rest of the minute:
    measured, 27,773 of a CM's 36,561 excluded ticks were dead-phase."""

    def _calls(self):
        import ast
        import inspect
        import textwrap
        tree = ast.parse(textwrap.dedent(inspect.getsource(MatchEngine)))
        out = {}
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_offball_move_player"):
                    inc = next((k for k in node.keywords
                                if k.arg == "include_on_ball"), None)
                    if inc is None:
                        val = None
                    elif isinstance(inc.value, ast.Constant):
                        val = inc.value.value
                    elif (isinstance(inc.value, ast.Attribute)
                          and inc.value.attr == "_DEAD_TIME_ON_BALL"):
                        val = "flag"
                    else:
                        val = "expr"
                    out.setdefault(fn.name, []).append(val)
        return out

    def test_the_flag_exists_and_defaults_off(self):
        import inspect
        sig = inspect.signature(MatchEngine._offball_move_player)
        self.assertIn("include_on_ball", sig.parameters)
        self.assertIs(sig.parameters["include_on_ball"].default, False)

    def test_the_dead_phase_opts_in_and_the_live_phase_does_not(self):
        calls = self._calls()
        self.assertIn("_continuous_offball_phase", calls,
                      "anchor is stale — the dead-phase call site moved")
        self.assertIn("_offball_run", calls,
                      "anchor is stale — the live call site moved")
        self.assertTrue(all(v in (True, "flag")
                            for v in calls["_continuous_offball_phase"]),
                        "the dead phase does not step on-ball players")
        self.assertTrue(MatchEngine._DEAD_TIME_ON_BALL,
                        "the dead-phase kill switch is off")
        self.assertTrue(all(v is None for v in calls["_offball_run"]),
                        "the live phase must keep excluding on-ball players")


class TimeIsAUnionTests(unittest.TestCase):
    """A minute has sixty seconds. Two instruments write the same minute (the
    shape integrator and the possession trace) and their time OVERLAPS, so the
    row's sum is not time. Published, whole-match totals reached 6,195 s
    against 5,820 s of clock and `activity_measured_pct` could exceed 100."""

    def _frames(self, band_s, dist, n=1, **row_kw):
        return [{"minute": m, "home": [_row("P", band_s=band_s, dist=dist,
                                            **row_kw)],
                 "away": [], "home_goals": 0, "away_goals": 0,
                 "possession_team": "Home", "phase": "open_play"}
                for m in range(1, n + 1)]

    def test_overlap_is_scaled_back_into_the_minute(self):
        # 70 instrument-seconds in a 60 s minute
        a = _analytics(self._frames((20.0, 20.0, 15.0, 10.0, 5.0), 70.0))
        act = a._activity["P"]
        self.assertAlmostEqual(sum(act.values()), 60.0, places=6)
        self.assertAlmostEqual(a.player_data["P"]["activity_overlap_s"], 10.0,
                               places=6)
        # ratios preserved (standing and walking were equal)
        self.assertAlmostEqual(act["standing"], act["walking"], places=6)
        self.assertLessEqual(a.player_data["P"]["activity_measured_pct"], 100.0)

    def test_the_residue_cannot_create_time(self):
        """Telemetry adds to individual residue bands; without redistribution
        the split could sum past the gap it describes."""
        a = _analytics(self._frames((5.0, 12.0, 7.0, 4.0, 2.0), 45.0,
                                    touch_d=45.0, touches=5, top=8.0,
                                    sprint=2.0, high=2.0))
        self.assertAlmostEqual(sum(a._activity["P"].values()), 60.0, places=6)
        rows = self._frames((5.0, 12.0, 7.0, 4.0, 2.0), 45.0, n=3,
                            touch_d=45.0, touches=5, top=8.0, sprint=2.0,
                            high=2.0)
        for r in rows:
            r["home"][0]["peak_touch_jump"] = 30.0
        b = _analytics(rows)
        self.assertAlmostEqual(sum(b._activity["P"].values()), 180.0, places=6)

    def test_no_minute_publishes_more_than_sixty_seconds(self):
        # the pre-fix shape: 75 s booked by one instrument alone
        a = _analytics(self._frames((30.0, 20.0, 15.0, 7.0, 3.0), 75.0, n=2))
        self.assertAlmostEqual(sum(a._activity["P"].values()), 120.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)