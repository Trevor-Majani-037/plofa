"""ROLE PACE TABLE GUARD -- no outfield role may sit in a class of its own, and
the pace table must actually be read by the integrator.

THE DEFECT THIS FILE EXISTS FOR (measured, 2026-10-20)
------------------------------------------------------
`MatchEngine._JOG_SPEED` is the sustained shape-holding pace that drives almost
every metre of off-ball movement. Every outfield role sat at 1.75-2.0 m/s
except the centre-back row, which sat at **1.5** -- inside the WALKING band
(Di Salvo: walk 0.5-2.0, jog 2.0-4.2) and below every other outfielder.

Measured on a real match (`tools/diag/_diag_cb_pace.py`, seed 777), that made
the centre-back the lowest outfield distance in the engine at 8.41 km. The
branch SHAPE was identical to a full-back's -- same four branches, same shares
(to_target ~30%, block_recover ~6%, block_shuffle ~44%, orbit ~18%) -- with
every branch slower (to_target 1.13 m/s vs LB 1.47, orbit 1.04 vs 1.32). So it
was a pace outlier, not a missing mechanism.

THE BENCHMARK WAS WRONG, WHICH IS THE MORE IMPORTANT HALF
---------------------------------------------------------
`AGENTS.md` carried "real centre back 10.5-11.0 km" from memory and the gap was
briefly read as ~2.3 km. The published figures are lower. CIES Football
Observatory Monthly Report n°68 (Oct 2021), 7,855 matches across 31 leagues,
SkillCorner tracking, five position modalities:

    centre backs    9.2 km / match   (the LEAST of any position)
    midfielders    10.6 km / match   (the most)
    wingers cover the most high-intensity (932 m) and sprinting (211 m);
    centre backs the least in both.

So the real deficit was ~0.8 km, not ~2.3 km, and the correction is a role-table
edit rather than a new mechanism. Same lesson as *SHAPE ANCHOR*'s full-back
band: check the benchmark's source and its aggregation before fitting anything
to it.

MEASURED AFTER (3 seeds x 2 arms, identical seeds, runtime-only arm switch,
`tools/diag/_diag_cb_pace_ab.py --seeds 777,57,31`):

    seed   off     on
    777    8.41    8.69
    57     8.26    9.25
    31     8.94    9.22
    mean   8.54    9.05    ->  -0.15 km from CIES 9.2

All three seeds move up. The control roles swing +/-1.5-2.0 km between arms in
BOTH directions with no mechanism (they are different matches after the first
pace-driven position change), which is the divergence signature, not an effect.
The activity accounting invariant held in all six matches: the maximum
published activity total equalled the match clock exactly.

WHAT THIS FILE PINS, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------
It pins the MECHANISM -- the table's shape and that the engine reads it -- in
milliseconds. It does NOT assert a distance band, because a distance band needs
a real match, one seed is a direction rather than a calibration, and a
single-draw match fixture is the class of flaky test this project has already
been burned by (`test_chance_provenance`). The sweep above is the measurement;
this file is the guard against the outlier coming back.
"""
import ast
import unittest
from pathlib import Path

from match_engine import MatchEngine

REPO = Path(__file__).resolve().parent.parent

# Every outfield role must be at least this pace. The centre-back row was the
# only outfield value below it (1.5), which is what made it an outlier rather
# than a role calibration. 1.75 is the winger value, i.e. the slowest role that
# was never flagged.
OUTFIELD_PACE_FLOOR = 1.75

OUTFIELD = ("CB", "DC", "LCB", "RCB", "DMC", "FB", "LB", "RB",
            "CM", "CDM", "CAM", "LM", "RM", "LW", "RW", "WF",
            "ST", "CF", "SS")

# Centre-back labels that a roster loader may emit for the same player. The
# position string arrives from `roster_loader.POS_MAP` -> `PlayerRecord.
# engine_pos` -> `SquadBuilder`, so an edit that fixes "CB" and forgets "LCB"
# silently leaves part of the population on the old pace. That failure is
# invisible at every behavioural layer, which is why it is pinned here.
CB_ALIASES = ("CB", "DC", "LCB", "RCB")


class RolePaceOutlierTests(unittest.TestCase):
    def test_no_outfield_role_sits_below_the_floor(self):
        table = MatchEngine._JOG_SPEED
        below = {k: v for k, v in table.items()
                 if k in OUTFIELD and v < OUTFIELD_PACE_FLOOR}
        self.assertEqual(
            below, {},
            f"outfield role(s) below the {OUTFIELD_PACE_FLOOR} floor: {below}. "
            "A single role in a class of its own is a pace outlier, not a "
            "calibration -- see this module's docstring for the CB case.")

    def test_the_centre_back_is_no_longer_the_single_slowest_outfielder(self):
        table = MatchEngine._JOG_SPEED
        outfield = {k: v for k, v in table.items() if k in OUTFIELD}
        slowest = min(outfield.values())
        cb = table["CB"]
        strictly_below_all = all(cb < v for k, v in outfield.items() if k != "CB")
        self.assertFalse(
            strictly_below_all,
            f"the centre-back pace ({cb}) is still strictly below every other "
            f"outfield role ({sorted(outfield.items(), key=lambda kv: kv[1])}). "
            f"Slowest outfield value is {slowest}.")

    def test_the_centre_back_aliases_share_its_pace(self):
        table = MatchEngine._JOG_SPEED
        for alias in CB_ALIASES:
            self.assertIn(alias, table, f"missing centre-back alias {alias!r}")
            self.assertEqual(
                table[alias], table["CB"],
                f"{alias!r} ({table[alias]}) disagrees with 'CB' ({table['CB']}). "
                "A roster emitting that label would silently get a different "
                "pace, and nothing behavioural would ever reveal it.")

    def test_the_old_value_would_have_failed_these_assertions(self):
        """NON-VACUITY. A guard that cannot fail is not a guard.

        The pre-fix value is 1.5. It must be below the floor AND strictly below
        every other outfield role, or the assertions above are testing nothing.
        """
        legacy = 1.5
        self.assertLess(legacy, OUTFIELD_PACE_FLOOR)
        current = {k: v for k, v in MatchEngine._JOG_SPEED.items()
                   if k in OUTFIELD and k not in CB_ALIASES}
        self.assertTrue(
            all(legacy < v for v in current.values()),
            f"1.5 is not strictly below every non-CB outfield value "
            f"({sorted(current.items(), key=lambda kv: kv[1])}); the outlier "
            "assertion above would not have caught the historical value.")

    def test_the_centre_back_is_still_a_defender_and_not_a_midfielder(self):
        """The correction must not have promoted CBs past the engine's own
        midfield pace. Real football has them below midfielders (9.2 vs 10.6)
        and the engine should agree in its ordering, not just its totals."""
        table = MatchEngine._JOG_SPEED
        self.assertLessEqual(table["CB"], table["CM"])
        self.assertLessEqual(table["CB"], table["CDM"])
        self.assertGreater(table["CB"], table["GK"])


class PaceTableIsReadTests(unittest.TestCase):
    """The table must not become decorative.

    Every historical case of "the mechanism exists and never runs" in this
    project reached a test that asserted the mechanism's own shape. This is an
    AST read-site count, so a table that is defined and never subscripted fails
    here rather than silently shipping a constant that looks calibrated.
    """

    @staticmethod
    def _tree():
        src = (REPO / "match_engine.py").read_text(encoding="utf-8",
                                                   errors="replace")
        return ast.parse(src)

    def test_the_table_is_defined_at_class_scope(self):
        found = []
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "_JOG_SPEED":
                        found.append(node.lineno)
        self.assertTrue(found, "MatchEngine._JOG_SPEED is not defined by an "
                               "assignment; this guard has become a no-op.")

    def test_the_integrator_reads_the_table(self):
        reads = 0
        for node in ast.walk(self._tree()):
            if (isinstance(node, ast.Attribute)
                    and node.attr == "_JOG_SPEED"
                    and isinstance(node.ctx, ast.Load)):
                reads += 1
        self.assertGreaterEqual(
            reads, 2,
            f"only {reads} load site(s) of _JOG_SPEED found in match_engine.py; "
            "the pace table is no longer driving both the step speed and the "
            "block-shuffle cap, so editing it would change nothing.")


if __name__ == "__main__":
    unittest.main()
