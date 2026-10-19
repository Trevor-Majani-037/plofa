"""KEEPER ATTRIBUTION + GK WORK-RATE GUARD (2026-10-17).

THREE DEFECTS THIS FILE EXISTS FOR, all found the same way -- by asking
whether a published number could belong to someone who did not earn it.

1. GOALS PREVENTED WAS A TEAM TOTAL STAMPED ON EVERY `position == "GK"`.
   `exporter.StatAccumulator._build` wrote, for every goalkeeper:
       s["xgot_faced"]      = self.team_xgot[team]
       s["goals_prevented"] = team_xgot - goals_conceded
   so a BENCH KEEPER WHO NEVER CAME ON owned his team's entire xGOT and a
   non-zero Goals Prevented -- while `saves` was correctly 0, so his `save_pct`
   read 0.0 and his rating collected `min(gp*0.25, 0.75)` off a number he never
   influenced. A substitute who came on at 70' also inherited the 12' goal.

2. GK DISTANCE WAS 2.82 km A MATCH -- ABOUT HALF OF REAL. Two independent GPS
   studies, both already cited by this project's own `calibrate_gps.py`:
       Di Salvo et al. 2008, elite GKs:  5,611 +/- 613 m
       Kubayi et al. 2020, Olympic GKs:  4,819 m (range 4,036-6,640)
   `BASE_MINUTE_ACTIVITY["GK"]` was (0.80, 0.18, 0.02, 0.00, 0.00) = 29.2 m/min.

3. THE PHYSICS ADDITIVE WAS A SECOND FULL MATCH. `physics_dist * 0.45` added a
   flat fraction of a full re-measurement on top of a template that already
   represented the whole match -- a flat +2.53 km to every player. The
   `real_dist > baseline_dist * 1.15` excess logic written for this had been
   DEAD, sitting in the `else` of `if physics_dist > 0`.

WHY A NEW FILE. `tests/test_distance_accounting.py` (same session) covers the
trace-side physical-reach ceiling. These are the OTHER two distance paths and
the goalkeeping attribution, none of which any existing suite asserted a
magnitude for. A suite that is green and silent about a feature is not
evidence about it.
"""
import unittest
import textwrap
from types import SimpleNamespace as NS

from exporter import StatAccumulator
from match_engine import EventType
import opta_analytics


def _player(name, position):
    return NS(name=name, position=position)


def _accum(starters, timeline):
    """A StatAccumulator wired only as far as the keeper-attribution path."""
    ex = StatAccumulator.__new__(StatAccumulator)
    ex.result = NS(config=NS(home_team="HOME", away_team="AWAY"),
                   timeline=timeline)
    ex.players = {team: {"starters": [_player(n, p) for n, p in xi],
                         "substitutes": []}
                  for team, xi in starters.items()}
    ex._accumulate_keeper_facing()
    return ex


def _ev(etype, team, minute, xg=0.0, player="", secondary=""):
    return NS(event_type=etype, team=team, minute=minute, xg=xg,
              player=player, secondary_player=secondary)


XI = {"HOME": [("HomeKeeper", "GK"), ("HomeBenchKeeper", "GK"), ("HomeDef", "CB")],
      "AWAY": [("AwayKeeper", "GK"), ("AwayBenchKeeper", "GK"), ("AwayDef", "CB")]}
KEEPERS = {"HOME": ("HomeKeeper", "HomeBenchKeeper"),
           "AWAY": ("AwayKeeper", "AwayBenchKeeper")}


def _in(ex, team):
    return sum(ex.keeper_xgot[k] for k in KEEPERS[team])


class GoalsPreventedIsPerKeeperTests(unittest.TestCase):
    """The defect: a team total inherited by a man who never touched the ball.

    Fixture convention: `_ev(..., team="AWAY", SHOT_ON_TARGET)` is a shot BY
    Away AT Home's goal, so it is charged to HomeKeeper. Keep this straight —
    writing it the other way round produces a test that passes for the wrong
    reason, which is the failure mode this repo documents most often."""

    def test_a_bench_keeper_prevents_nothing(self):
        ex = _accum(XI, [
            _ev(EventType.SHOT_ON_TARGET, "AWAY", 10, xg=0.42),
            _ev(EventType.SHOT_ON_TARGET, "AWAY", 30, xg=0.31),
            _ev(EventType.GOAL, "AWAY", 44),
        ])
        # The keeper who was actually in goal owns all three...
        self.assertAlmostEqual(ex.keeper_xgot["HomeKeeper"], 0.73, places=6)
        self.assertEqual(ex.keeper_conceded["HomeKeeper"], 1)
        # ...and BOTH bench keepers, who never came on, own nothing. Pre-fix,
        # `goals_prevented` for either was team_xgot[their team] - conceded.
        for bench in ("HomeBenchKeeper", "AwayBenchKeeper"):
            self.assertEqual(ex.keeper_xgot.get(bench, 0.0), 0.0, bench)
            self.assertEqual(ex.keeper_conceded.get(bench, 0), 0, bench)

    def test_keeper_scope_reconciles_to_team_scope(self):
        """INVARIANT, not a comment. Sum of keepers == team. This is the check
        that caught an own goal charged to the wrong side."""
        ex = _accum(XI, [
            _ev(EventType.SHOT_ON_TARGET, "AWAY", 10, xg=0.42),
            _ev(EventType.HIT_WOODWORK, "AWAY", 22, xg=0.08),
            _ev(EventType.PENALTY_MISSED, "AWAY", 33, xg=0.79),
            _ev(EventType.GOAL, "AWAY", 44),
            _ev(EventType.SHOT_ON_TARGET, "HOME", 55, xg=0.19),
            _ev(EventType.GOAL, "HOME", 61),
        ])
        self.assertAlmostEqual(_in(ex, "HOME"), ex.team_xgot["HOME"], places=6,
                               msg="HOME xGOT lost or invented")
        self.assertAlmostEqual(_in(ex, "AWAY"), ex.team_xgot["AWAY"], places=6,
                               msg="AWAY xGOT lost or invented")
        self.assertEqual(sum(ex.keeper_conceded.values()), 2)

    def test_an_own_goal_is_charged_to_the_CONCEDING_side(self):
        """`e.team` means OPPOSITE things on GOAL and OWN_GOAL. A GOAL is built
        on the scoring side; an OWN_GOAL is built on the CONCEDING side, because
        `goal_team` is the beneficiary while the scorer put it in his own net.
        Getting this wrong is silent -- it just under-charges one keeper."""
        ex = _accum(XI, [_ev(EventType.OWN_GOAL, "HOME", 40)])
        # HOME put it in its own net, so HomeKeeper is charged; AwayKeeper is not.
        self.assertEqual(ex.keeper_conceded["HomeKeeper"], 1)
        self.assertEqual(ex.keeper_conceded.get("AwayKeeper", 0), 0)

    def test_a_substitute_keeper_owns_only_his_own_minutes(self):
        ex = _accum(XI, [
            _ev(EventType.SUBSTITUTION, "HOME", 70,
                player="HomeKeeper", secondary="HomeBenchKeeper"),
            _ev(EventType.SHOT_ON_TARGET, "AWAY", 10, xg=0.50),
            _ev(EventType.SHOT_ON_TARGET, "AWAY", 80, xg=0.25),
            _ev(EventType.GOAL, "AWAY", 85),
        ])
        self.assertAlmostEqual(ex.keeper_xgot["HomeKeeper"], 0.50, places=6)
        self.assertAlmostEqual(ex.keeper_xgot["HomeBenchKeeper"], 0.25, places=6)
        self.assertEqual(ex.keeper_conceded["HomeBenchKeeper"], 1)
        self.assertEqual(ex.keeper_conceded.get("HomeKeeper", 0), 0)
        self.assertAlmostEqual(ex.team_xgot["HOME"], 0.75, places=6)


class KeeperStatRowTests(unittest.TestCase):
    """THE CONSUMER, not the accumulator.

    The original defect lived in the per-player loop that WRITES the stat row,
    not in the code that accumulates xGOT. A guard written only against the
    accumulator therefore cannot see it -- and did not: mutation M1 of
    `_diag_keeper_guard.py` shipped MISSED on the first run, because every
    other test here inspects `keeper_xgot`/`keeper_conceded` directly and never
    ran the consumer. Naming a test after the symptom it does not exercise is
    the trap `_diag_cross_guard.py` documents; this class is the fix, and
    `_apply_keeper_stats` exists so it can be cheap enough to exist."""

    def _row(self, name, saves):
        return {"position": "GK", "saves": saves, "goals_conceded": 0,
                "xgot_faced": 0.0, "goals_prevented": 0.0, "save_pct": 0.0}

    def _ex(self, timeline):
        ex = _accum(XI, timeline)
        return ex

    def test_a_bench_keeper_row_reads_zero_not_the_team_total(self):
        """THE REPORTED BUG. AwayKeeper never came on; HomeKeeper faced
        0.73 xGOT. Before the fix BOTH rows showed 0.73."""
        ex = self._ex([
            _ev(EventType.SHOT_ON_TARGET, "AWAY", 10, xg=0.42),
            _ev(EventType.SHOT_ON_TARGET, "AWAY", 30, xg=0.31),
            _ev(EventType.GOAL, "AWAY", 44),
        ])
        played = self._row("HomeKeeper", 2)
        bench = self._row("HomeBenchKeeper", 0)
        ex._apply_keeper_stats("HomeKeeper", played)
        ex._apply_keeper_stats("HomeBenchKeeper", bench)
        self.assertAlmostEqual(played["xgot_faced"], 0.73, places=6)
        self.assertAlmostEqual(played["goals_prevented"], -0.27, places=6)
        self.assertEqual(played["goals_conceded"], 1)
        # the bench man owns nothing, on every axis
        self.assertEqual(bench["xgot_faced"], 0.0)
        self.assertEqual(bench["goals_prevented"], 0.0)
        self.assertEqual(bench["goals_conceded"], 0)
        self.assertEqual(bench["save_pct"], 0.0)

    def test_the_row_does_not_read_the_team_total(self):
        """NON-VACUITY: assert the team figure really is available and really
        is different, so the test cannot pass because team_xgot happens to be
        zero. Then assert the row ignores it."""
        ex = self._ex([_ev(EventType.SHOT_ON_TARGET, "AWAY", 10, xg=0.42)])
        self.assertAlmostEqual(ex.team_xgot["HOME"], 0.42, places=6)
        row = self._row("HomeBenchKeeper", 0)
        ex._apply_keeper_stats("HomeBenchKeeper", row)
        self.assertNotAlmostEqual(row["xgot_faced"], 0.42, places=6)

    def test_save_pct_uses_the_keeper_s_own_shots_faced(self):
        ex = self._ex([
            _ev(EventType.SHOT_ON_TARGET, "AWAY", 10, xg=0.42),
            _ev(EventType.GOAL, "AWAY", 44),
        ])
        row = self._row("HomeKeeper", 2)
        ex._apply_keeper_stats("HomeKeeper", row)
        self.assertEqual(row["goals_conceded"], 1)
        self.assertAlmostEqual(row["save_pct"], 66.7, places=1)


class InGoalResolverTests(unittest.TestCase):
    def test_resolver_follows_the_substitution_timeline(self):
        ex = _accum(XI, [
            _ev(EventType.SUBSTITUTION, "HOME", 70,
                player="HomeKeeper", secondary="BenchKeeper"),
        ])
        r = ex._in_goal_at()
        self.assertEqual(r("HOME", 10), "HomeKeeper")
        self.assertEqual(r("HOME", 70), "BenchKeeper")   # applied AT the minute
        self.assertEqual(r("HOME", 89), "BenchKeeper")
        self.assertEqual(r("AWAY", 89), "AwayKeeper")

    def test_a_team_with_no_keeper_returns_empty_rather_than_inventing(self):
        ex = _accum({"HOME": [("NoGoal", "CB")], "AWAY": XI["AWAY"]}, [])
        self.assertEqual(ex._in_goal_at()("HOME", 10), "")


class GoalkeeperWorkRateTests(unittest.TestCase):
    """Real benchmarks, stated in the test so a rename cannot silently void
    the assertion:
        Di Salvo et al. 2008 (elite GKs, GPS):  5,611 +/- 613 m  -> 5.00-6.22 km
        Kubayi et al. 2020 (Olympic GKs):          4,819 m        -> 4.04-6.64 km
    """

    def _per90(self, pos, work_rate=50.0):
        base = opta_analytics.BASE_MINUTE_ACTIVITY[pos]
        effort = 0.80 + 0.40 * (work_rate / 100.0)
        stand, walk, jog, run, sprint = base
        jog *= effort
        run = min(0.35, run * effort)
        sprint = min(0.10, sprint * effort)
        s = opta_analytics.SPEEDS
        per_min = (stand * 60 * s["standing"] + walk * 60 * s["walking"]
                   + jog * 60 * s["jogging"] + run * 60 * s["running"]
                   + sprint * 60 * s["sprinting"])
        return per_min * 90

    def test_a_goalkeeper_covers_a_realistic_distance(self):
        for wr in (0.0, 50.0, 100.0):
            km = self._per90("GK", wr) / 1000
            self.assertTrue(4.0 <= km <= 6.65,
                            f"GK at work_rate {wr}: {km:.2f} km is outside the "
                            f"real range (Di Salvo 5.00-6.22, Kubayi 4.04-6.64)")

    def test_the_goalkeeper_template_is_not_the_old_stationary_one(self):
        """NON-VACUITY first: assert the old band is genuinely out of range, so
        this test cannot pass by accident if the table is ever reverted."""
        old = (0.80, 0.18, 0.02, 0.00, 0.00)
        old_km = (old[0] * 60 * 0.2 + old[1] * 60 * 1.5
                  + old[2] * 60 * 2.8) * 90 / 1000
        self.assertLess(old_km, 3.2, "the old template is no longer out of range")
        self.assertNotEqual(tuple(opta_analytics.BASE_MINUTE_ACTIVITY["GK"]),
                            tuple(old))

    def test_the_goalkeeper_still_dominates_standing_time(self):
        """A keeper is the least mobile player on the pitch. If a fix ever made
        him a midfielder's profile this is the assertion that catches it."""
        gk = opta_analytics.BASE_MINUTE_ACTIVITY["GK"]
        cm = opta_analytics.BASE_MINUTE_ACTIVITY["CM"]
        self.assertGreater(gk[0], cm[0], "a keeper must stand more than a CM")


class PhysicsAdditiveIsAnExcessTests(unittest.TestCase):
    """⚠️ SUPERSEDED 2026-10-17 by the measured-activity blend, and this guard
    had to be repointed rather than deleted.

    This class originally asserted the *excess* form
    (`physics_dist - baseline_dist * covered_frac`), which replaced the flat
    `physics_dist * 0.45`. That mechanism is now gone too: `_compute_activity`
    publishes MEASURED metres for every second the integrator watched plus a
    template budget for the seconds it did not, so there is no additive left to
    be flat or excess-shaped.

    Left un-repointed, this guard would have gone RED on a strictly better
    implementation — which is how a guard trains people to delete it instead of
    reading it. What is asserted now is the invariant that supersedes both:

        distance == measured metres + template budget for UNWATCHED seconds

    i.e. the template term must be proportional to what was NOT observed, and
    must vanish entirely at full coverage."""

    def _src(self):
        import inspect
        return inspect.getsource(opta_analytics.OptaAnalytics._compute_activity)

    def test_the_distance_is_measured_plus_the_unwatched_residue(self):
        import re
        src = self._src()
        self.assertIsNotNone(
            re.search(r"baseline_dist\s*=\s*physics_dist\s*\+\s*template_dist", src),
            "the distance is no longer measured-metres-plus-residue")

    def test_neither_the_flat_nor_the_excess_additive_has_returned(self):
        """AST, NOT A SUBSTRING SCAN. The comment in this method explains that
        `physics_dist_additive` was retired, so a raw-source scan matches the
        PROSE. That is this repo's standing rule — a name inside a comment is
        not a call site — and the scan has to be structural or it fails on the
        fix's own documentation."""
        import ast
        import inspect
        import re
        src = inspect.getsource(opta_analytics.OptaAnalytics._compute_activity)
        tree = ast.parse(textwrap.dedent(src))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertNotIn("physics_dist_additive", names,
                         "the additive variable is live in the code again")
        # and no `physics_dist * <fraction>` anywhere in an expression
        for node in ast.walk(tree):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
                    and isinstance(node.left, ast.Name)
                    and node.left.id == "physics_dist"):
                lit = node.right
                if isinstance(lit, ast.Constant) and isinstance(lit.value, float):
                    self.fail(f"a flat multiplicative physics_dist term is back: "
                              f"physics_dist * {lit.value}")

    def test_the_template_only_covers_unobserved_seconds(self):
        import re
        src = self._src()
        self.assertIsNotNone(
            re.search(r"uncovered_s\s*=\s*max\(0\.0,\s*60\.0\s*-\s*measured_total_s\)", src),
            "the template residue is no longer scoped to the unwatched seconds")
        self.assertIn("tpl_s = {b: prof[b] * uncovered_s", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)