"""SCHEMA-BY-NAME ALIGNMENT -- a trained brain must receive the exact role
features it was trained on, or the loader must refuse it.

THE DEFECT THIS FILE EXISTS FOR (measured, 2026-10-05 -> found 2026-10-19)
-----------------------------------------------------------------------
`role_features.MENU_NAMES["CB"]` grew from 7 names to 9 when two build-up
sensors (`same_side_lane_open`, `far_side_gap`) were added, making a CB net
24 + 9 = 33 inputs wide.  `brains/CB.json` was trained at 24 + 7 = 31.

The loader compared WIDTHS, not names: 31 inputs were available, so it took
"the first 31 numbers" and dropped the tail.  Consequence, measured
(`_diag_receiver_signal.py`): the two trained sensors were **inert** -- the
padded vector was bit-identical pre- and post-change, and the argmax differed
on **0 of 4000** random inputs.  Two inputs had been paid for in training and
were unreachable in play.

Worse, the failure was SILENT.  A schema change that invalidates every trained
brain of a position logs nothing, crashes nothing, and simply degrades that
position to the previous behaviour.  That is the project's standing pathology
-- *a mechanism that exists, is wired to something, and never runs* -- with the
added insult that the numbers keep looking fine.

WHAT SHIPPED
------------
1. A brain may DECLARE its role features by name (`meta.role_features`), and
   `align_role_tail` returns exactly those, in the declared order.
2. A name the current build can no longer produce RAISES `RoleSchemaError`
   rather than being dropped.  A declared list whose length disagrees with the
   network's own input width also raises -- a shorter declaration is NOT
   zero-padded, because padding is indistinguishable from the silent
   truncation this exists to stop.
3. Legacy brains (no declaration) keep the historical width-only behaviour
   BYTE-FOR-BYTE, so no shipped brain changes behaviour -- but they warn ONCE
   per (family, width), so a silent drift cannot happen unnoticed again.
4. `brain_evolution.evolve()` stamps `role_features` on every newly evolved
   brain, so the declaration is the default rather than an optional extra.

WHY BY-NAME AND NOT BY-WIDTH
----------------------------
`test_inserting_a_feature_in_the_middle_shifts_a_legacy_brain_but_not_a_declared_one`
is the whole argument in one case: given a schema `["a", "NEW", "b", "c", "d"]`
and a brain trained on `["a", "b", "c", "d"]`, width-only selection hands that
brain `["a", "NEW", "b", "c"]` -- every value after `a` is the WRONG FEATURE,
with the right shape, so nothing complains.  By-name selection hands it the
four it asked for.  The CB case is exactly this, except the new names were
appended so the damage was confined to a missing tail.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import unittest
import warnings
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import role_features as RF
from role_features import (
    MENU_NAMES, V2_INPUT_D, V2_SHARED_D, RoleSchemaError, align_role_tail,
    reset_alignment_warnings,
)
from tactics_context import TACTICS_CONTEXT_D, V3_INPUT_D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The two sensors added 2026-10-05 that were inert on the shipped CB brain.
THE_INERT_TWO = ("same_side_lane_open", "far_side_gap")


def _fake_brain(tail_width: int, meta=None, family="CB"):
    """Minimal stand-in: `_aligned_tail` reads `.meta` and `.w1.shape[0]`."""
    return SimpleNamespace(
        meta=(meta if meta is not None else {}),
        w1=np.zeros((V2_SHARED_D + tail_width, 1)),
    )


def _player(position="CB1"):
    """`_aligned_tail` resolves the family from `player.position`."""
    return SimpleNamespace(position=position, name="Test CB")


# ─────────────────────────────────────────────────────────────
# 1. THE SCHEMA ITSELF
# ─────────────────────────────────────────────────────────────

class SchemaWidthsAreDerivedNotTypedTests(unittest.TestCase):

    def test_every_family_width_is_shared_plus_its_menu(self):
        """`V2_INPUT_D` must be DERIVED from `MENU_NAMES`.

        If the two are maintained by hand they drift, and the moment they drift
        a `tail_width` computed from the architecture is a statement about a
        menu that no longer exists.
        """
        for fam, names in MENU_NAMES.items():
            self.assertEqual(
                V2_INPUT_D[fam], V2_SHARED_D + len(names),
                f"{fam}: V2_INPUT_D says {V2_INPUT_D[fam]}, "
                f"shared {V2_SHARED_D} + {len(names)} names")

    def test_every_v3_width_is_v2_plus_the_tactics_block(self):
        for fam in MENU_NAMES:
            self.assertEqual(V3_INPUT_D[fam],
                             V2_INPUT_D[fam] + TACTICS_CONTEXT_D, fam)

    def test_the_cb_schema_really_did_grow(self):
        """Non-vacuity for this whole file: CB is 9, not 7."""
        self.assertEqual(len(MENU_NAMES["CB"]), 9)
        self.assertEqual(tuple(MENU_NAMES["CB"][-2:]), THE_INERT_TWO)
        self.assertEqual(V2_INPUT_D["CB"], 33)

    def test_feature_names_are_unique_within_a_family(self):
        """A duplicate name makes by-name selection ambiguous -- `index()`
        silently returns the first, which is the wrong-feature-value bug in
        its quietest form."""
        for fam, names in MENU_NAMES.items():
            self.assertEqual(len(set(names)), len(names), fam)


# ─────────────────────────────────────────────────────────────
# 2. BY-NAME SELECTION
# ─────────────────────────────────────────────────────────────

class ByNameSelectionTests(unittest.TestCase):

    def test_legacy_and_by_name_agree_for_a_brain_trained_before_the_growth(self):
        """The shipped CB brain's numbers must not move.

        CB's two new sensors were APPENDED, so "the first 7 by width" and
        "the 7 it declares by name" are the same seven values.  That is why
        the fix is provably byte-identical for every brain in `brains/`.
        """
        role = [float(i) for i in range(len(MENU_NAMES["CB"]))]
        legacy = align_role_tail("CB", role, 7)
        named = align_role_tail("CB", role, 7,
                                want_names=list(MENU_NAMES["CB"][:7]))
        np.testing.assert_array_equal(legacy, named)
        np.testing.assert_array_equal(named, np.arange(7.0))

    def test_the_declared_order_is_the_output_order_not_the_schema_order(self):
        """A brain's weights are indexed positionally, so if it declares a
        different order it must RECEIVE that order."""
        want = ["far_side_gap", "own_line_height", "box_threat"]
        role = [float(i) for i in range(len(MENU_NAMES["CB"]))]
        out = align_role_tail("CB", role, 3, want_names=want)
        idx = [MENU_NAMES["CB"].index(n) for n in want]
        np.testing.assert_array_equal(out, [role[i] for i in idx])
        self.assertEqual(out[0], role[8], "far_side_gap is the 9th CB feature")

    def test_inserting_a_feature_in_the_middle_shifts_a_legacy_brain_but_not_a_declared_one(self):
        """THE ARGUMENT, in one case.

        A width-only loader cannot tell an added feature from a shifted one:
        it hands back the first N values and every one after the insertion
        point is the WRONG FEATURE, at the correct width, with no error.
        """
        fam = "TESTFAM_INSERT"
        schema = ["a", "NEW", "b", "c", "d"]
        RF.MENU_NAMES[fam] = schema
        try:
            values = [10.0, 20.0, 30.0, 40.0, 50.0]     # a NEW b c d
            legacy = align_role_tail(fam, values, 4)
            named = align_role_tail(fam, values, 4,
                                    want_names=["a", "b", "c", "d"])
            # Legacy silently mislabels: it got NEW and lost d.
            np.testing.assert_array_equal(legacy, [10.0, 20.0, 30.0, 40.0])
            # By name: the four the brain actually trained on.
            np.testing.assert_array_equal(named, [10.0, 30.0, 40.0, 50.0])
            self.assertNotEqual(legacy.tolist(), named.tolist(),
                                "the two paths must not be equivalent here")
        finally:
            del RF.MENU_NAMES[fam]

    def test_a_declaration_without_a_role_family_raises(self):
        with self.assertRaises(RoleSchemaError):
            align_role_tail(None, [1.0, 2.0], 2, want_names=["a", "b"])

    def test_a_declaration_for_an_unknown_family_raises(self):
        with self.assertRaises(RoleSchemaError):
            align_role_tail("NOPE", [1.0], 1, want_names=["a"])

    def test_a_name_this_build_no_longer_emits_raises(self):
        """Renaming or removing a feature is a BREAKING schema change, and the
        loader must say so instead of returning a shorter vector."""
        with self.assertRaises(RoleSchemaError) as ctx:
            align_role_tail("CB", [0.0] * 9, 2,
                            want_names=["own_line_height", "a_feature_we_deleted"],
                            brain_label="brain[CB in=31]")
        msg = str(ctx.exception)
        self.assertIn("a_feature_we_deleted", msg)
        self.assertIn("brain[CB in=31]", msg,
                      "the message must name the brain; a bare schema error is "
                      "not actionable")

    def test_a_declaration_that_disagrees_with_the_architecture_raises(self):
        """A SHORTER declaration must NOT be zero-padded: padding produces a
        correct-width vector carrying the wrong facts, which is exactly the
        silent-truncation failure in the other direction."""
        with self.assertRaises(RoleSchemaError) as ctx:
            align_role_tail("CB", [0.0] * 9, 7,
                            want_names=list(MENU_NAMES["CB"][:3]))
        self.assertIn("declares 3", str(ctx.exception))
        self.assertIn("reserves 7", str(ctx.exception))

    def test_a_longer_declaration_than_the_architecture_also_raises(self):
        with self.assertRaises(RoleSchemaError):
            align_role_tail("CB", [0.0] * 9, 2,
                            want_names=list(MENU_NAMES["CB"][:5]))

    def test_a_declaration_with_no_values_yields_zeros_of_the_right_width(self):
        """Unknown role -> deterministic zeros, but the DECLARATION still has
        to be valid.  Zeroing is a decision about this player, not a licence
        to skip the schema check."""
        out = align_role_tail("CB", None, 3,
                              want_names=list(MENU_NAMES["CB"][:3]))
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_a_zero_width_tail_is_empty_for_every_path(self):
        self.assertEqual(align_role_tail("CB", None, 0).shape, (0,))
        self.assertEqual(
            align_role_tail("CB", None, 0, want_names=[]).shape, (0,))


# ─────────────────────────────────────────────────────────────
# 3. THE LEGACY PATH KEEPS ITS NUMBERS AND LOSES ITS SILENCE
# ─────────────────────────────────────────────────────────────

class LegacyPathWarnsTests(unittest.TestCase):

    def setUp(self):
        reset_alignment_warnings()

    def tearDown(self):
        reset_alignment_warnings()

    def test_an_exact_width_brain_needs_no_warning(self):
        role = [float(i) for i in range(len(MENU_NAMES["CB"]))]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = align_role_tail("CB", role, 9)
        np.testing.assert_array_equal(out, role)
        self.assertEqual([w for w in caught
                          if issubclass(w.category, RuntimeWarning)], [])

    def test_a_truncated_legacy_brain_warns_that_the_extra_features_are_inert(self):
        role = [float(i) for i in range(len(MENU_NAMES["CB"]))]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = align_role_tail("CB", role, 7)
        self.assertEqual(out.shape, (7,))
        notes = [str(w.message) for w in caught
                 if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("INERT", notes[0])
        self.assertIn("role_features", notes[0],
                      "the warning must name the fix, not just the problem")

    def test_a_zero_padded_legacy_brain_warns_that_inputs_are_missing(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = align_role_tail("CB", [1.0, 2.0, 3.0], 5)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 0.0, 0.0])
        notes = [str(w.message) for w in caught
                 if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(notes), 1)
        self.assertIn("zero-padded", notes[0])

    def test_the_warning_fires_once_per_family_and_width_not_once_per_tick(self):
        """It is called on every on-ball decision, so a per-call warning would
        be a log flood and would be switched off within a minute."""
        role = [0.0] * 9
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(50):
                align_role_tail("CB", role, 7)
        self.assertEqual(len([w for w in caught
                              if issubclass(w.category, RuntimeWarning)]), 1)

    def test_a_different_width_warns_separately(self):
        role = [0.0] * 9
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            align_role_tail("CB", role, 7)
            align_role_tail("CB", role, 8)
        self.assertEqual(len([w for w in caught
                              if issubclass(w.category, RuntimeWarning)]), 2)

    def test_a_declared_brain_never_warns(self):
        """Declaring is the fix, so declaring must be free."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            align_role_tail("CB", [0.0] * 9, 2,
                            want_names=list(MENU_NAMES["CB"][:2]))
        self.assertEqual([w for w in caught
                          if issubclass(w.category, RuntimeWarning)], [])


# ─────────────────────────────────────────────────────────────
# 4. THE LOADER -- no brain in `brains/` can be silently truncated
# ─────────────────────────────────────────────────────────────

class NoShippedBrainIsSilentlyTruncatedTests(unittest.TestCase):

    def setUp(self):
        reset_alignment_warnings()

    def tearDown(self):
        reset_alignment_warnings()

    def _shipped_brains(self):
        d = os.path.join(REPO, "brains")
        if not os.path.isdir(d):
            self.skipTest("no brains/ directory")
        out = []
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json") or fn.startswith("_") \
                    or "surrogate" in fn:
                continue
            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict) or "w1" not in raw:
                continue
            out.append((fn, raw))
        return out

    def test_no_shipped_brain_is_silently_truncated(self):
        """For every serialized brain: either it declares the features it
        wants and the declaration is valid, or the width-only path would warn.

        This is the property that makes the fix permanent rather than a
        one-off repair of `brains/CB.json`: a future schema change cannot
        quietly invalidate the shipped set without a warning, and a brain that
        DOES declare cannot load at all if its features are gone.
        """
        brains = self._shipped_brains()
        self.assertTrue(brains, "no trained brains found -- this test proved nothing")
        seen_tails = 0
        checked = 0
        for fn, raw in brains:
            w1 = np.asarray(raw["w1"])
            n_in = int(w1.shape[0]) if w1.ndim >= 1 else 0
            if n_in <= V2_SHARED_D:
                continue                    # v1 24-d brain: no tail at all
            meta = raw.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}
            fam = meta.get("role_family")
            if not fam:
                continue                    # off-ball / v1 brain, out of scope
            tail = n_in - V2_SHARED_D
            if meta.get("sensor_schema") == "v3_tactics_context":
                tail -= TACTICS_CONTEXT_D
            want = meta.get("role_features")
            if want is not None:
                # Must load without raising -- otherwise the shipped set is
                # already broken and this test says so.
                out = align_role_tail(fam, [0.0] * len(MENU_NAMES.get(fam, ())),
                                      tail, want_names=want, brain_label=fn)
                self.assertEqual(out.shape, (tail,), fn)
            seen_tails += 1
            cur = len(MENU_NAMES.get(fam, ()))
            if tail >= cur:
                continue                    # nothing to truncate
            # The warn-once dedupe is per (family, width), so without this reset
            # the SECOND truncated brain of the same family silently records no
            # warning and this test fails on the code being right.  (It did.)
            reset_alignment_warnings()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                align_role_tail(fam, [0.0] * cur, tail, brain_label=fn)
            self.assertTrue(
                [w for w in caught if issubclass(w.category, RuntimeWarning)],
                f"{fn} is {tail} wide against a {cur}-name {fam} schema and "
                "did NOT warn -- it would be truncated silently")
            checked += 1
        # Non-vacuity: the scan must actually have reached role-feature brains,
        # or the loop above proved nothing about the shipped set.
        self.assertGreater(
            seen_tails, 0,
            "no shipped brain carried a role-feature tail -- the scan is vacuous")
        # `checked` counts the truncated ones.  It is not a fixed number: a
        # future retrain that declares every feature legitimately drops it to 0,
        # and that is the set becoming CORRECT, not the test becoming blind.
        print(f"[shipped-brain scan] {seen_tails} role-feature brains, "
              f"{checked} truncated-but-warned")


# ─────────────────────────────────────────────────────────────
# 5. THE LOADER AND THE EVOLVER ARE WIRED TO IT (AST)
# ─────────────────────────────────────────────────────────────

def _src(path):
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read()


def _func_node(src, path, name):
    tree = ast.parse(src, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node, ast.get_source_segment(src, node) or ""
    return None, ""


class TheLoaderUsesItTests(unittest.TestCase):
    """A guard on the CALL SITE, because a correct helper nobody calls is
    exactly this project's recurring failure."""

    def test_build_sensors_routes_both_padded_branches_through_the_helper(self):
        src = _src("brain_integration.py")
        node, body = _func_node(src, "brain_integration.py", "build_sensors")
        self.assertIsNotNone(node, "build_sensors not found")
        calls = [n for n in ast.walk(node)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", ""))
                 == "_aligned_tail"]
        self.assertEqual(len(calls), 2,
                         "expected the v3 and v2 branches to both align by "
                         f"name; found {len(calls)} call(s)")
        self.assertNotIn("role[:", body,
                         "a width slice on the role block is the silent "
                         "truncation this exists to remove")

    def test_the_helper_reads_the_declaration_off_the_brain(self):
        src = _src("brain_integration.py")
        node, body = _func_node(src, "brain_integration.py", "_aligned_tail")
        self.assertIsNotNone(node, "_aligned_tail not found")
        self.assertIn("role_features", body,
                      "the loader must read meta.role_features, not a width")
        self.assertIn("want_names", body)

    def test_the_v3_branch_refuses_a_wrong_width_vector(self):
        """v3 pads a tactics block after the role tail, so a mis-sized build
        is possible there in a way it is not in v2. It must raise."""
        src = _src("brain_integration.py")
        _node, body = _func_node(src, "brain_integration.py", "build_sensors")
        self.assertIn("RoleSchemaError", body,
                      "the v3 branch built 31-wide from a 33-wide net")

    def test_role_schema_error_is_the_same_class_everywhere(self):
        """Callers catching it from `brain_integration` and from
        `role_features` must be catching the same thing."""
        import brain_integration as BI
        self.assertIs(BI.RoleSchemaError, RoleSchemaError)


class TheEvolverStampsItTests(unittest.TestCase):

    def test_evolve_declares_the_features_it_trained_on(self):
        src = _src("brain_evolution.py")
        node, _body = _func_node(src, "brain_evolution.py", "evolve")
        self.assertIsNotNone(node, "evolve not found")
        kw = set()
        calls = 0
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and \
                    getattr(n.func, "id", getattr(n.func, "attr", "")) \
                    == "brain_meta_dict":
                calls += 1
                kw |= {k.arg for k in n.keywords if k.arg}
        self.assertGreaterEqual(calls, 1, "no brain_meta_dict call in evolve()")
        self.assertIn("role_features", kw,
                      "evolve() does not declare role_features, so every new "
                      "brain inherits the silent-truncation hazard")

    def test_the_declaration_is_clamped_to_the_architecture(self):
        """A v3 brain's tail is the role block MINUS the tactics block, so the
        declared list must be sliced to the architecture's own width."""
        src = _src("brain_evolution.py")
        _node, body = _func_node(src, "brain_evolution.py", "evolve")
        self.assertIn("tail_width", body)
        self.assertIn("fam_names[:", body,
                      "the declaration is not clamped to tail_width; a wider "
                      "schema would declare more features than the net has")


# ─────────────────────────────────────────────────────────────
# 6. THE META FIELD SURVIVES SERIALISATION
# ─────────────────────────────────────────────────────────────

class MetaCarriesTheDeclarationTests(unittest.TestCase):

    def test_brain_meta_dict_accepts_and_emits_role_features(self):
        from brain_schema import brain_meta_dict
        d = brain_meta_dict(training_method="ga_surrogate",
                            sensor_schema="v2_role_features",
                            role_family="CB",
                            role_features=list(MENU_NAMES["CB"]))
        self.assertEqual(d["role_features"], list(MENU_NAMES["CB"]))

    def test_the_field_defaults_to_none_for_every_existing_brain(self):
        """Absence must mean "legacy", not "empty declaration" -- an empty
        list would be a declaration of zero features."""
        from brain_schema import BrainMeta
        self.assertIsNone(BrainMeta().role_features)


# ─────────────────────────────────────────────────────────────
# 7. THE FORWARD-OPTION INPUT (added 2026-10-20)
# ─────────────────────────────────────────────────────────────

class _FakePE:
    """Just enough position engine for `_pos`: a name -> (x, y) table."""

    def __init__(self, table):
        self._t = dict(table)

    def get_position(self, name):
        return self._t[name]


def _p(name, position):
    return SimpleNamespace(name=name, position=position)


class TheWideRolesCanSeeAForwardOptionTests(unittest.TestCase):
    """`pass_lane_open` for FB and WING.

    THE GAP: the on-ball brain's inputs contain nowhere the fact "a forward
    pass option is open".  DM/CM/AM have had `pass_lane_open` all along and CB
    has `same_side_lane_open`, so FB and WING were the only families blind to
    it -- and those are exactly the two roles that live on the touchline where
    the forward option is the whole point.
    """

    FAMILIES_THAT_ALREADY_HAD_IT = ("DM", "CM", "AM")

    def test_the_wide_families_now_have_it(self):
        for fam in ("FB", "WING"):
            self.assertIn("pass_lane_open", MENU_NAMES[fam], fam)

    def test_it_is_the_LAST_feature_so_no_existing_index_moves(self):
        """Appended, not inserted.

        The legacy loader takes "the first N" by width.  Appending means a
        shipped brain's first 7 numbers still mean exactly what they meant, so
        this change cannot mislabel a single trained input; inserting anywhere
        else silently relabels every feature after the insertion point.
        """
        for fam in ("FB", "WING"):
            self.assertEqual(MENU_NAMES[fam][-1], "pass_lane_open", fam)
        # And the pre-growth prefixes are exactly the old menus.
        self.assertEqual(MENU_NAMES["FB"][:7], [
            "wide_marker_dist", "one_v_one_space", "inside_channel",
            "overlap_option", "cross_zone", "own_line_height", "short_cut"])
        self.assertEqual(MENU_NAMES["WING"][:8], [
            "marker_dist", "one_v_one", "inside_channel", "overlap_option",
            "cross_zone", "cross_target_open", "box_arrival", "cover_behind"])

    def test_it_is_not_added_where_it_cannot_carry_information(self):
        """ST and GK are deliberately excluded.

        A striker's best forward option is usually himself, and a keeper's is
        normally a defender behind him, so the sensor would be ~constant --
        training budget spent on a feature with no variance.
        """
        for fam in ("ST", "GK"):
            self.assertNotIn("pass_lane_open", MENU_NAMES[fam], fam)
        # Non-vacuity: the families that SHOULD have it, do.
        for fam in self.FAMILIES_THAT_ALREADY_HAD_IT + ("FB", "WING"):
            self.assertIn("pass_lane_open", MENU_NAMES[fam], fam)

    def test_the_now_sensor_responds_to_a_marker_where_overlap_option_cannot(self):
        """The discriminating case -- and the reason a new sensor was needed.

        Two frames, IDENTICAL except that the opposition marker moves from 1 m
        to 20 m off our forward teammate.  `overlap_option` cannot move: it
        measures the distance to our own teammate.  `pass_lane_open` must.

        A sensor that correlates with something already in the vector is not a
        new input; this asserts the new one carries information the existing
        seven did not.
        """
        fb = _p("FB1", "LB")
        lw = _p("LW1", "LW")
        marker = _p("D1", "LW")
        x, y = 60.0, 10.0                       # full-back, attacking right
        tms = [fb, lw]
        marker_at = {}

        def block():
            pe = _FakePE({"FB1": (x, y), "LW1": (75.0, 8.0),
                          "D1": (marker_at["p"])})
            return RF._fb_block(fb, x, y, tms, [marker], pe, True)

        marker_at["p"] = (76.0, 8.0)            # marking him tightly
        tight = block()
        marker_at["p"] = (95.0, 8.0)            # marker gone, lane open
        loose = block()

        i_overlap = MENU_NAMES["FB"].index("overlap_option")
        i_lane = MENU_NAMES["FB"].index("pass_lane_open")

        self.assertAlmostEqual(tight[i_overlap], loose[i_overlap], places=9,
                              msg="overlap_option moved with the DEFENDER; it "
                                  "is supposed to be a function of our own "
                                  "teammate's position only")
        self.assertLess(tight[i_lane], 0.2,
                        f"a tightly marked forward option read as open "
                        f"({tight[i_lane]:.3f})")
        self.assertGreater(loose[i_lane], 0.9,
                           f"an unmarked forward option read as closed "
                           f"({loose[i_lane]:.3f})")

    def test_no_forward_option_at_all_reads_zero_not_a_neutral_half(self):
        """0.0 is a FACT ("nobody ahead"), not a default.

        A 0.5 neutral would tell the network "there is a half-open option"
        about a frame in which somebody is standing behind the ball.
        """
        fb = _p("FB1", "LB")
        behind = _p("LW1", "LW")
        x, y = 60.0, 10.0
        pe = _FakePE({"FB1": (x, y), "LW1": (40.0, 10.0)})   # behind the carrier
        out = RF._fb_block(fb, x, y, [fb, behind], [], pe, True)
        self.assertEqual(out[MENU_NAMES["FB"].index("pass_lane_open")], 0.0)

    def test_the_wing_block_produces_it_too(self):
        lw = _p("LW1", "LW")
        st = _p("ST1", "ST")
        x, y = 80.0, 8.0
        pe = _FakePE({"LW1": (x, y), "ST1": (95.0, 20.0)})
        out = RF._wing_block(lw, x, y, [lw, st], [], pe, True)
        self.assertEqual(len(out), len(MENU_NAMES["WING"]))
        self.assertEqual(out[-1], 1.0)      # nobody within 10 m of the striker

    def test_every_block_returns_exactly_its_declared_width(self):
        """The behavioural twin of `len(block) == V2_ROLE_D[fam]`.

        Driven through the SHIPPING entry point `role_block`, with real
        position labels, so it also exercises `_role_family`'s digit stripping
        (`CM1` -> `CM`) -- the trap that made `_cb_same_side_corridor` inert on
        arrival.  `role_block` carries its own cardinality assert, so a
        mismatch would raise rather than return a wrong-length vector.
        """
        x, y = 52.0, 34.0
        mates = [_p("ME", "CM1"), _p("A", "CB1"), _p("B", "LW"), _p("C", "ST")]
        foes = [_p("D1", "CB"), _p("D2", "CB"), _p("D3", "LW")]
        pe = _FakePE({"ME": (x, y), "A": (40.0, 30.0), "B": (70.0, 8.0),
                      "C": (80.0, 34.0), "D1": (55.0, 30.0),
                      "D2": (57.0, 38.0), "D3": (66.0, 8.0)})
        for label in ("GK", "CB1", "LB", "RB", "CDM", "CM1", "CM2",
                      "CAM", "LW", "RW", "ST", "CF"):
            me = _p("ME", label)
            pe._t["ME"] = (12.0, 34.0) if label == "GK" else (x, y)
            block = RF.role_block(me, pe._t["ME"][0], pe._t["ME"][1],
                                  mates, foes, pe, True)
            fam = RF._role_family(label)
            self.assertIsNotNone(fam, f"{label} has no role family")
            self.assertEqual(len(block), len(MENU_NAMES[fam]),
                             f"{label} ({fam}) block length != its menu")


class TheForwardOptionHasOneDefinitionTests(unittest.TestCase):
    """One concept, one implementation.

    `pass_lane_open` is now a feature of five families.  If `_mid_block` kept
    its own copy of the computation, the two would drift and the same feature
    NAME would mean two different things depending on the position -- the exact
    failure the by-name schema alignment exists to prevent, one level down.
    """

    def test_the_midfield_menu_uses_the_shared_helper(self):
        src = open(os.path.join(REPO, "role_features.py"), encoding="utf-8").read()
        mid = src[src.index("def _mid_block"):]
        mid = mid[:mid.index("\ndef ", 1)]
        self.assertIn("_forward_options(", mid,
                      "_mid_block computes the forward option itself instead "
                      "of calling the shared helper")
        self.assertNotIn(
            "pass_lane_open = max(", mid,
            "a second copy of the forward-lane computation has appeared")

    def test_every_family_declaring_it_gets_the_same_number(self):
        """Same geometry, same value, whatever the role family.

        The menus differ in their OTHER features; this one must not.
        """
        out = {}
        for fam, block in (("FB", RF._fb_block), ("WING", RF._wing_block)):
            me = _p("ME", "LM")
            st = _p("ST1", "ST")
            x, y = 70.0, 10.0
            pe = _FakePE({"ME": (x, y), "ST1": (88.0, 20.0)})
            vec = block(me, x, y, [me, st], [], pe, True)
            out[fam] = vec[MENU_NAMES[fam].index("pass_lane_open")]
        self.assertEqual(len(set(out.values())), 1,
                         f"the same geometry gave different forward-option "
                         f"values per family: {out}")
        self.assertEqual(out["FB"], 1.0)

    def test_the_midfield_families_are_byte_identical_to_the_old_formula(self):
        """The refactor must not have changed a single midfield number.

        Tested by re-implementing the PRE-refactor expression inline and
        requiring exact equality -- a refactor that quietly altered the
        midfield vector would invalidate every DM/CM/AM brain in the same
        commit that was supposed to leave them alone.
        """
        me = _p("ME", "CM1")
        a = _p("A", "ST")
        b = _p("B", "LW")
        x, y = 40.0, 30.0
        pe = _FakePE({"ME": (x, y), "A": (60.0, 30.0), "B": (70.0, 12.0)})
        tms = [me, a, b]
        defs = [_p("D1", "CB")]
        pe._t["D1"] = (66.0, 30.0)

        # The retired expression, verbatim.
        tms_n = RF._teammates(tms)
        fwd_tms = [t for t in tms_n
                   if RF._prog(RF._pos(pe, t.name, (x, y))[0], x, True) > 4.0]
        opens = [RF._openness(*RF._pos(pe, t.name, (x, y)), defs, pe)
                 for t in fwd_tms]
        old = max(opens) if opens else 0.0

        new = RF._forward_options(x, y, tms_n, defs, pe, True)[1]
        self.assertEqual(new, old)
        self.assertGreater(len(fwd_tms), 1,
                           "the fixture must give more than one forward "
                           "option or the max() is untested")


if __name__ == "__main__":
    unittest.main(verbosity=2)
