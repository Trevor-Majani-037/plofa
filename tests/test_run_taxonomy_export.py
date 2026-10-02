"""
PLOFA 26/27 - Run taxonomy export: observed vs intended
=========================================================
tests/test_run_taxonomy_export.py

The complaint this answers: "run types are calculated but not in any export
file". Which was correct, and the cause was worse than an oversight:

  * `RunTracker` classified six real run types on every match for months
    (advance / overlap / underlap / far_side / forward / support), with
    per-type cooldowns and gain thresholds, and was exposed as
    `MatchEngine.get_run_profile()`.
  * The exporter never called it. Its only run-ish counter,
    `_count_off_ball_runs`, walked `position_log` and incremented ONE flat
    counter for any frame-to-frame jump >= 15 m. A keeper shuffle, a
    half-time formation reset and a genuine diagonal run counted identically.
  * The live run layers then discarded the run MODE at the position-layer
    boundary, so even the decision that CAUSED a run was unrecoverable and
    the exporter could only ever guess from geometry.

So the suite pins two different facts, deliberately not merged:
  OBSERVED  - what the movement did (RunTracker, geometric, post hoc)
  INTENDED  - what the behaviour engine decided (behind / cut / orbit / ...)

and it pins the SHAPE fact that made this invisible: both live on the engine,
and neither was reachable from anything holding only the MatchResult - the
same trap world.ingest documented for `sub_controller`.
"""
from __future__ import annotations

import random
from datetime import date

import pytest

from exporter import StatAccumulator
from match_engine import (
    Intensity, MatchConfig, MatchEngine, PlayingStyle, TeamProfile, TeamStyle,
)
from player_dna import SquadBuilder
from run_tracking import INTENDED_RUN_MODES, RUN_TYPES, IntendedRunRecorder, RunTracker

ROLES = [
    ("GK", ["sweeper_keeper"]), ("CB", ["stopper_defender"]),
    ("CB", ["ball_playing_cb"]), ("LB", ["aggressive_fullback"]),
    ("RB", ["overlapping_fullback"]), ("CDM", ["anchor_man"]),
    ("CM", ["engine"]), ("CM", ["box_box"]), ("CAM", ["creator"]),
    ("LW", ["winger"]), ("ST", ["fox_in_box"]),
]


def _squad(team):
    return SquadBuilder.build(
        team, [(f"{team[:3]} {p} {i}", p, s, 26) for i, (p, s) in enumerate(ROLES)]
    )["starters"]


def _profile(name, style=TeamStyle.BALANCED, play=PlayingStyle.POSSESSION):
    return TeamProfile(name=name, style=style, playing_style=play,
                       intensity=Intensity.MEDIUM)


# =============================================================================
# 1. The two vocabularies are genuinely different things
# =============================================================================


def test_export_keys_are_unambiguous_despite_shared_words():
    """`overlap` and `underlap` legitimately name BOTH a geometric observation
    and an engine decision. That is a real collision in the raw vocabularies -
    the fullback engine genuinely chooses an "overlap run", and the tracker
    genuinely observes a player "overlapping" on its own.

    What must be unambiguous is the EXPORT, so this asserts the key namespaces
    rather than the words. If these two namespaces ever merged, a column called
    "overlap" would be silently counting one fact while being read as the
    other.
    """
    assert not set(RUN_TYPES).isdisjoint(INTENDED_RUN_MODES), (
        "expected the deliberate word overlap (overlap/underlap) - if this now "
        "passes, one vocabulary changed and the key namespaces need rechecking")
    for kind in RUN_TYPES:
        assert f"run_{kind}" not in {f"run_intended_{m}" for m in INTENDED_RUN_MODES}


def test_observed_taxonomy_is_the_six_gradient_types():
    assert set(RUN_TYPES) == {"advance", "overlap", "underlap", "far_side",
                              "forward", "support"}


def test_intended_vocabulary_covers_every_live_run_mode():
    """Every mode the wired engines can actually emit must be representable."""
    for mode in ("behind", "hold", "box",        # striker
                 "byline", "cut",                # winger
                 "overlap", "underlap", "tuck",  # fullback
                 "drop", "carry", "late", "orbit",  # CM
                 "roam"):                        # CAM
        assert mode in INTENDED_RUN_MODES, mode


# =============================================================================
# 2. IntendedRunRecorder
# =============================================================================


def test_recorder_counts_per_player_per_mode():
    r = IntendedRunRecorder()
    r.record("A", "ST", "behind")
    r.record("A", "ST", "behind")
    r.record("A", "ST", "hold")
    r.record("B", "CM", "orbit")
    p = r.profile()
    assert p["A"]["behind"] == 2
    assert p["A"]["hold"] == 1
    assert p["B"]["orbit"] == 1
    assert r.total("A") == 3
    assert r.position_of("A") == "ST"


def test_recorder_ignores_an_empty_mode():
    r = IntendedRunRecorder()
    r.record("A", "CM", "")
    assert r.profile() == {}
    assert r.total("A") == 0


def test_recorder_survives_a_mode_added_after_this_file_was_written():
    """A new engine mode must not KeyError the export."""
    r = IntendedRunRecorder()
    r.record("A", "ST", "some_future_run")
    assert r.profile()["A"]["some_future_run"] == 1


def test_recorder_profile_is_a_copy():
    """Mutating the returned dict must not corrupt the recorder's state."""
    r = IntendedRunRecorder()
    r.record("A", "ST", "behind")
    snap = r.profile()
    snap["A"]["behind"] = 999
    assert r.profile()["A"]["behind"] == 1


# =============================================================================
# 3. RunTracker still behaves (the six types were always right)
# =============================================================================


def test_run_tracker_classifies_a_diagonal_forward_run():
    """Player starts behind the ball and finishes decisively ahead of it.

    The geometry has to actually satisfy the predicate: RunTracker fires
    "forward" only when the player has been behind (ever_behind) AND is now
    ahead by AHEAD_M AND has >= FWD_CROSS of forward gain. Ending level with the
    ball satisfies none of it, which is why an earlier version of this test
    moved the player to exactly the ball's x and counted nothing.
    """
    t = RunTracker()
    t.sample(0.0, "A", 40.0, 34.0, 55.0, 34.0, True, "H", True)
    for i in range(1, 25):
        t.sample(i * 0.1, "A", 40.0 + i * 1.5, 34.0, 55.0, 34.0, True, "H", True)
    prof = t.profile().get("A", {})
    assert sum(prof.values()) >= 1, prof
    assert prof["forward"] >= 1, prof


def test_run_tracker_ignores_movement_while_out_of_possession():
    """Out of possession there is no run to describe, so nothing may count.

    A clear on a non-possession tick must discard accumulated gain, or a
    turnover mid-run would bank a run that never happened.
    """
    t = RunTracker()
    t.sample(0.0, "A", 40.0, 34.0, 55.0, 34.0, True, "H", True)
    for i in range(1, 8):
        t.sample(i * 0.1, "A", 40.0 + i * 1.5, 34.0, 55.0, 34.0, True, "H", True)
    t.sample(0.8, "A", 52.0, 34.0, 55.0, 34.0, True, "H", False)  # turnover
    counted = sum(t.profile().get("A", {}).values())
    t.sample(0.9, "A", 53.5, 34.0, 55.0, 34.0, True, "H", True)
    t.sample(1.0, "A", 55.0, 34.0, 55.0, 34.0, True, "H", True)
    assert sum(t.profile().get("A", {}).values()) == counted, (
        "a run was banked across a possession change")


# =============================================================================
# 7. The taxonomy must not collapse onto one label, and must not count a jump
#
# Before these fixes the six observed types resolved to ONE: support was 95% of
# 1497 runs and overlap fired once. Cause: the predicates were an elif chain
# ordered most-specific first, and the loosest one (`support`) sat last, so it
# absorbed everything. Two things fixed that, and both are pinned here because
# both failed the first time they were written.
# =============================================================================


def _feed(tracker, rows, speeds=None):
    if speeds is not None:
        tracker.set_top_speeds(speeds)
    for r in rows:
        tracker.sample(*r)


def test_a_net_forward_segment_is_not_labelled_support():
    """The exact defect: `back` accumulated independently of `gain`.

    The player bursts forward past the ball and then gives a little ground. Net
    movement is strongly forward, but he satisfied `back >= BACK_MIN` and was
    filed as a support drop, because support was last in the chain.
    """
    tr = RunTracker()
    rows, t, x = [(0.0, "A", 40.0, 34.0, 55.0, 34.0, True, "H", True)], 0.1, 40.0
    for _ in range(30):                      # 18 m forward: crosses the ball
        x += 0.6
        rows.append((t, "A", x, 34.0, 55.0, 34.0, True, "H", True))
        t += 0.1
    for _ in range(5):                       # 3 m of give-back, still ahead
        x -= 0.6
        rows.append((t, "A", x, 34.0, 55.0, 34.0, True, "H", True))
        t += 0.1
    _feed(tr, rows)
    p = tr.profile().get("A", {})
    assert p.get("support", 0) == 0, (
        f"a net-forward segment was labelled a support drop: {p}")
    assert p.get("forward", 0) + p.get("advance", 0) >= 1, p


def test_support_requires_having_been_up_front():
    """A support run is a TRANSITION: advanced -> behind.

    Without the `ever_ahead` gate, ordinary backward drift qualified and support
    was 95% of all runs. You cannot drop into support if you were never in front
    of the ball to leave.
    """
    tr = RunTracker()
    rows, t, x = [(0.0, "A", 30.0, 34.0, 55.0, 34.0, True, "H", True)], 0.1, 30.0
    for _ in range(20):                      # drifts further behind the ball
        x -= 0.5
        rows.append((t, "A", x, 34.0, 55.0, 34.0, True, "H", True))
        t += 0.1
    _feed(tr, rows)
    p = tr.profile().get("A", {})
    assert p.get("support", 0) == 0, (
        f"pure backward drift counted as a support run: {p}")


def test_a_player_who_was_ahead_and_drops_can_still_record_support():
    """The gate must not simply disable support. He must cross back behind."""
    tr = RunTracker()
    rows, t, x = [(0.0, "A", 75.0, 34.0, 60.0, 34.0, True, "H", True)], 0.1, 75.0
    for _ in range(40):                      # 24 m back: crosses behind the ball
        x -= 0.6
        rows.append((t, "A", x, 34.0, 60.0, 34.0, True, "H", True))
        t += 0.1
    _feed(tr, rows)
    p = tr.profile().get("A", {})
    assert p.get("support", 0) >= 1, f"support became uncountable: {p}"


def test_jump_filter_fires_on_a_genuine_teleport():
    """The guard must actually see a non-zero interval.

    It was originally written after `self._last_t[name] = t`, so its elapsed
    time was always exactly 0 and `0 < dt` never held: the guard was inert and
    a controlled replay showed jumps_filtered == 0 with 200+ non-physical
    samples present. A missing `math` import was hiding underneath it. Both are
    covered by asserting the counter MOVES.
    """
    tr = RunTracker()
    speeds = {"A": 7.0}
    rows = [
        (0.0, "A", 40.0, 34.0, 55.0, 34.0, True, "H", True),
        (0.1, "A", 40.2, 34.0, 55.0, 34.0, True, "H", True),
        (0.2, "A", 90.0, 34.0, 55.0, 34.0, True, "H", True),   # 50 m in 0.1 s
        (0.3, "A", 90.2, 34.0, 55.0, 34.0, True, "H", True),
    ]
    _feed(tr, rows, speeds)
    assert tr.jumps_filtered == 1, (
        f"jump guard did not fire ({tr.jumps_filtered}); it is inert")


def test_jump_filter_leaves_ordinary_movement_alone():
    tr = RunTracker()
    rows = [(i * 0.1, "A", 40.0 + i * 0.1, 34.0, 55.0, 34.0, True, "H", True)
            for i in range(20)]
    _feed(tr, rows, {"A": 7.0})
    assert tr.jumps_filtered == 0, "a 0.1 m step was mistaken for a teleport"


def test_jump_filter_is_inert_without_injected_speeds_and_that_is_checkable():
    """A silently no-op guard is worse than no guard.

    Without top speeds the tracker cannot know a player's limit, so it declines
    to filter rather than guessing — and `jumps_filtered == 0` makes that
    visible instead of hiding it.
    """
    tr = RunTracker()
    _feed(tr, [(0.0, "A", 40.0, 34.0, 55.0, 34.0, True, "H", True),
               (0.1, "A", 95.0, 34.0, 55.0, 34.0, True, "H", True)])
    assert tr.jumps_filtered == 0, "guessed a limit with no speeds injected"


def _ep_forward(n=30, dx=0.6):
    """Behind the ball, crosses into front of it."""
    rows, t = [(0.0, "A", 45.0, 34.0, 60.0, 34.0, True, "H", True)], 0.1
    x = 45.0
    for _ in range(n):
        x += dx
        rows.append((t, "A", x, 34.0, 60.0, 34.0, True, "H", True))
        t += 0.1
    return rows


def _ep_advance(n=30, dx=0.6):
    """Already ahead, keeps advancing — the general forward case."""
    rows, t = [(0.0, "A", 66.0, 34.0, 60.0, 34.0, True, "H", True)], 0.1
    x = 66.0
    for _ in range(n):
        x += dx
        rows.append((t, "A", x, 34.0, 60.0, 34.0, True, "H", True))
        t += 0.1
    return rows


def _ep_far_side(n=30, dx=0.6):
    """Opposite side of a wide ball, advancing."""
    rows, t = [(0.0, "A", 45.0, 20.0, 60.0, 50.0, True, "H", True)], 0.1
    x = 45.0
    for _ in range(n):
        x += dx
        rows.append((t, "A", x, 20.0, 60.0, 50.0, True, "H", True))
        t += 0.1
    return rows


def _ep_overlap(n=30):
    """Ball wide right, player on the right and getting WIDER."""
    rows, t = [(0.0, "A", 45.0, 53.0, 60.0, 50.0, True, "H", True)], 0.1
    x, y = 45.0, 53.0
    for _ in range(n):
        x += 0.6
        y += 0.12
        rows.append((t, "A", x, y, 60.0, 50.0, True, "H", True))
        t += 0.1
    return rows


def _ep_underlap(n=30):
    """Same side, was wider, now cutting inside."""
    rows, t = [(0.0, "A", 45.0, 56.0, 60.0, 50.0, True, "H", True)], 0.1
    x, y = 45.0, 56.0
    for _ in range(n):
        x += 0.6
        y -= 0.16
        rows.append((t, "A", x, y, 60.0, 50.0, True, "H", True))
        t += 0.1
    return rows


def _ep_support(n=40, dx=0.6):
    """Ahead of the ball, drops back across behind it."""
    rows, t = [(0.0, "A", 75.0, 34.0, 60.0, 34.0, True, "H", True)], 0.1
    x = 75.0
    for _ in range(n):
        x -= dx
        rows.append((t, "A", x, 34.0, 60.0, 34.0, True, "H", True))
        t += 0.1
    return rows


def test_the_taxonomy_actually_distinguishes():
    """Each of the six types must be reachable from a movement that means it.

    This is the property the export exists to make visible. Before the fix the
    chain collapsed: support was 95% of a live match's runs and `overlap` fired
    once, because the loosest predicate sat last and absorbed everything.

    Each episode is shaped to satisfy exactly one predicate, so a type that
    cannot fire at all fails here rather than hiding inside an aggregate share.
    """
    cases = {
        "overlap": _ep_overlap(),
        "underlap": _ep_underlap(),
        "far_side": _ep_far_side(),
        "forward": _ep_forward(),
        "advance": _ep_advance(),
        "support": _ep_support(),
    }
    for want, rows in cases.items():
        tr = RunTracker()
        _feed(tr, rows, {"A": 7.0})
        p = tr.profile().get("A", {})
        assert p.get(want, 0) >= 1, (
            f"the {want} predicate is unreachable: episode produced "
            f"{ {k: v for k, v in p.items() if v} }")


def test_no_single_observed_type_absorbs_a_mixed_feed():
    """The aggregate property, on a feed that provably contains every type."""
    tr = RunTracker()
    rows = []
    for ep in (_ep_overlap(), _ep_underlap(), _ep_far_side(),
               _ep_forward(), _ep_advance(), _ep_support()):
        if rows:
            # shift the clock forward so the tracker sees one continuous match
            last = rows[-1][0]
            ep = [(t + last + 0.1, *rest) for (t, *rest) in ep]
        rows.extend(ep)
    _feed(tr, rows, {"A": 7.0})
    p = tr.profile().get("A", {})
    total = sum(p.values())
    assert total >= 6, f"expected at least one run per episode, got {p}"
    worst = max(p.values()) / total
    assert worst < 0.60, f"one type absorbed {100*worst:.0f}% of runs: {p}"





def test_match_result_carries_both_run_profiles():
    """The fact that hid this for months.

    Both live on MatchEngine. An adapter holding only the MatchResult - which
    is what every exporter and world.ingest is - cannot see them. This is the
    same shape trap world.ingest documented for `sub_controller`.
    """
    random.seed(31)
    cfg = MatchConfig(home_team="Home FC", away_team="Away FC",
                      match_date=date(2026, 9, 29))
    eng = MatchEngine(cfg, _profile("Home FC"),
                      _profile("Away FC", style=TeamStyle.PARK_THE_BUS,
                               play=PlayingStyle.COUNTER))
    eng.set_squad("Home FC", _squad("Home FC"))
    eng.set_squad("Away FC", _squad("Away FC"))
    res = eng.simulate()

    assert hasattr(res, "run_profile"), "MatchResult lost run_profile"
    assert hasattr(res, "intended_run_profile"), "MatchResult lost intended_run_profile"
    assert isinstance(res.run_profile, dict)
    assert isinstance(res.intended_run_profile, dict)


def test_result_profiles_match_the_engine_ones():
    random.seed(31)
    cfg = MatchConfig(home_team="Home FC", away_team="Away FC",
                      match_date=date(2026, 9, 29))
    eng = MatchEngine(cfg, _profile("Home FC"), _profile("Away FC"))
    eng.set_squad("Home FC", _squad("Home FC"))
    eng.set_squad("Away FC", _squad("Away FC"))
    res = eng.simulate()
    assert res.run_profile == eng.get_run_profile()
    assert res.intended_run_profile == eng.intended_runs.profile()


def test_missing_profiles_degrade_instead_of_raising():
    """A hand-built result (tests, scratch callers) has neither attribute."""
    acc = StatAccumulator.__new__(StatAccumulator)
    class _R:
        pass
    r = _R()
    r.run_profile = None
    r.intended_run_profile = None
    acc.result = r
    acc.stats = {"A": {}}
    # _apply_run_taxonomies reads both off .result and returns early.
    acc._apply_run_taxonomies()
    assert acc.stats["A"] == {}


# =============================================================================
# 5. The exporter surfaces them
# =============================================================================


def test_exporter_populates_observed_and_intended_columns():
    random.seed(31)
    cfg = MatchConfig(home_team="Home FC", away_team="Away FC",
                      match_date=date(2026, 9, 29))
    eng = MatchEngine(cfg, _profile("Home FC"),
                      _profile("Away FC", style=TeamStyle.PARK_THE_BUS,
                               play=PlayingStyle.COUNTER))
    eng.set_squad("Home FC", _squad("Home FC"))
    eng.set_squad("Away FC", _squad("Away FC"))
    res = eng.simulate()

    acc = StatAccumulator(res, {
        "Home FC": {"starters": _squad("Home FC"), "substitutes": []},
        "Away FC": {"starters": _squad("Away FC"), "substitutes": []},
    })

    st = acc.stats["Hom ST 10"]
    for kind in RUN_TYPES:
        assert f"run_{kind}" in st, kind
        assert isinstance(st[f"run_{kind}"], int)
    assert "runs_observed_total" in st
    assert "runs_intended_total" in st
    for mode in INTENDED_RUN_MODES:
        assert f"run_intended_{mode}" in st, mode

    # The headline: the engine decided SOMETHING for the striker, and it is
    # now visible. Before this, the mode was dropped and the only evidence was
    # a position delta.
    assert st["runs_intended_total"] >= 1, (
        "no intended runs recorded for the striker - is the mode still dropped?")
    assert st["run_intended_behind"] + st["run_intended_hold"] \
        + st["run_intended_box"] >= 1, st


def test_intended_and_observed_are_independent_facts():
    """They must NOT be derived from each other.

    If they were, one would be a relabelling of the other and the whole point
    of recording intent is lost.
    """
    random.seed(31)
    cfg = MatchConfig(home_team="Home FC", away_team="Away FC",
                      match_date=date(2026, 9, 29))
    eng = MatchEngine(cfg, _profile("Home FC"), _profile("Away FC"))
    eng.set_squad("Home FC", _squad("Home FC"))
    eng.set_squad("Away FC", _squad("Away FC"))
    res = eng.simulate()
    assert res.intended_run_profile != res.run_profile, (
        "intended and observed run profiles are the same object - "
        "one is being derived from the other")
