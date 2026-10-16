"""DEFENSIVE STYLE TEMPERAMENT (2026-10-16)

`formation_stance_for` chose the defensive stance from the scoreline and the
clock and never read `profile.style` -- while its docstring stated it did.
Consequence: the ATTACKING half of the shape was style-driven (`pattern_for`
picks from `_STYLE_PATTERN_POOLS`) and the DEFENSING half was not, so a
park-the-bus side and a gegenpressing side held an identical back line until
the scoreline happened to move.

`tactical_shapes.style_defense_deltas` is the missing half, threaded through
`apply_formation_stance(style=...)` and stacked UNDER the game-state stance.

Why a NEW file rather than more tests in `test_positional_play`: that suite
is green about the stance system and says nothing about the style's influence
on it. A suite that is green and silent about a feature is not evidence about
that feature.

Run:  PYTHONPATH=. .venv\\Scripts\\python.exe -m pytest tests/test_style_defensive_temperament.py -q
"""
import ast
import os

import pytest

from match_engine import TeamStyle
from position_engine import PositionEngine
from player_dna import SquadBuilder
from tactical_shapes import (FormationStance, stance_delta_roles,
                             style_defense_deltas,
                             _STYLE_DEFENSE_ARCHETYPE)
import run_match as RM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 1. The table ────────────────────────────────────────────────────────

def test_every_teamstyle_maps_to_an_archetype():
    missing = [s.value for s in TeamStyle
               if s.value not in _STYLE_DEFENSE_ARCHETYPE]
    assert not missing, f"TeamStyle values with no defensive archetype: {missing}"


def test_every_archetype_used_has_a_delta_table():
    from tactical_shapes import _STYLE_DEFENSE_ROLE_DELTAS
    dangling = [a for a in set(_STYLE_DEFENSE_ARCHETYPE.values())
                if a not in _STYLE_DEFENSE_ROLE_DELTAS]
    assert not dangling, f"archetypes with no delta table: {dangling}"


def test_unknown_and_absent_style_produce_no_deltas():
    """A style the engine does not recognise must get the PRE-CHANGE
    behaviour, not an invented average. A probe reaching this with a made-up
    default would be measuring its own assumption."""
    assert style_defense_deltas(None) == {}
    assert style_defense_deltas("not_a_real_style") == {}
    assert style_defense_deltas("") == {}


def test_balanced_is_the_neutral_reference():
    assert style_defense_deltas("balanced") == {}
    assert style_defense_deltas("wing_play") == {}


# ── 2. It is a TEMPERAMENT, not a game-state override ───────────────────

def test_the_two_extremes_move_the_pitch_in_opposite_directions():
    sunk = style_defense_deltas("park_the_bus")
    high = style_defense_deltas("attacking")
    for role in ("CB", "LB", "RB", "CDM", "CM", "CAM"):
        assert sunk[role][0] < 0 < high[role][0], (
            f"{role}: park_the_bus {sunk[role]} and attacking {high[role]} must "
            f"push the anchor in opposite directions")


def test_a_sunk_style_holds_the_striker_up_more_than_the_wingers():
    """A block whose front pair drop with the defenders has no outlet. The
    existing SEE_IT_OUT table already encodes this (ST -8 vs LW -20); the
    style table must not repeat the mistake it was copied from."""
    sunk = style_defense_deltas("park_the_bus")
    assert abs(sunk["ST"][0]) < abs(sunk["LW"][0]), (
        f"ST {sunk['ST']} should hold up better than LW {sunk['LW']}")


def test_the_game_state_still_outranks_the_style():
    """The property that stops a style bias from overriding a manager. A team
    two goals up must execute SEE_IT_OUT regardless of its club."""
    for stance in (FormationStance.SEE_IT_OUT, FormationStance.ALL_OUT_CHASE):
        bus = stance_delta_roles(stance, style="park_the_bus").get("LB", (0,))[0]
        att = stance_delta_roles(stance, style="attacking").get("LB", (0,))[0]
        assert bus < att, (
            f"{stance.value}: park_the_bus ({bus}) must still sit deeper "
            f"than attacking ({att})")


def test_the_style_does_not_outrank_a_stance_on_its_own_team():
    bus_baseline = stance_delta_roles(
        FormationStance.BASELINE, style="park_the_bus").get("LB", (0,))[0]
    bus_see_it = stance_delta_roles(
        FormationStance.SEE_IT_OUT, style="park_the_bus").get("LB", (0,))[0]
    # SEE_IT_OUT alone moves an LB -7; the style alone -8. Together -18+.
    assert bus_see_it < bus_baseline - 7, (
        f"SEE_IT_OUT must still push well past the temperament: "
        f"{bus_baseline} -> {bus_see_it}")


def test_no_style_reproduces_the_pre_change_result_exactly():
    for stance in FormationStance:
        with_style = stance_delta_roles(stance, style="park_the_bus")
        without = stance_delta_roles(stance)
        assert with_style != without or style_defense_deltas(
            "park_the_bus") == {}


# ── 3. It reaches the anchors (the call path, which is where the bug was) ─

class _Prof:
    value = "balanced"
    attack_intensity = 55.0
    defensive_intensity = 55.0
    width = 0.5
    formation = "4_3_3"
    pressing_intensity = 55.0
    tempo = 55.0
    risk_appetite = 55.0


@pytest.fixture(scope="module")
def pe():
    p = PositionEngine()
    hs = SquadBuilder.build(team_name="T1", starters=RM.HOME_STARTERS,
                            substitutes=RM.HOME_SUBS,
                            soul_assignments=RM.SOUL_PLAYERS)
    p.initialize_team("T1", hs["starters"], _Prof())
    return p


def _anchors(p):
    return {n: s.home_x for n, s in p.states.items()
            if s.position in ("CB", "LB", "CDM", "CM")}


def test_the_style_reaches_home_x_on_a_real_engine(pe):
    pe.apply_formation_stance("T1", FormationStance.BASELINE, style="balanced")
    flat = _anchors(pe)
    assert pe.apply_formation_stance(
        "T1", FormationStance.BASELINE, style="park_the_bus") is True, \
        "a style CHANGE must re-apply -- the style is in the idempotency key"
    sunk = _anchors(pe)
    assert all(sunk[k] < flat[k] for k in flat), \
        f"every back-line anchor must move up the pitch: {flat} -> {sunk}"


def test_a_pressing_style_moves_the_anchors_the_other_way(pe):
    pe.apply_formation_stance("T1", FormationStance.BASELINE, style="balanced")
    flat = _anchors(pe)
    pe.apply_formation_stance("T1", FormationStance.BASELINE,
                              style="attacking")
    att = _anchors(pe)
    assert all(att[k] > flat[k] for k in flat), \
        f"every back-line anchor must move down the pitch: {flat} -> {att}"


def test_style_none_reproduces_the_pre_change_anchors(pe):
    pe.apply_formation_stance("T1", FormationStance.BASELINE, style="balanced")
    flat = _anchors(pe)
    pe.apply_formation_stance("T1", FormationStance.BASELINE, style=None)
    assert all(abs(_anchors(pe)[k] - flat[k]) < 1e-9 for k in flat)


# ── 4. AST guards. The omission class is invisible behaviourally. ───────

def _ast_tree(name):
    with open(os.path.join(REPO, name), encoding="utf-8") as fh:
        return ast.parse(fh.read())


def test_the_live_call_site_passes_style():
    """A defaulted argument is a silent omission. This is the same reasoning
    as the `attacks_right` and `shooter_name` guards in the suite, and it is
    why this behaviour cannot be pinned by any test that only calls the
    delta tables."""
    src = open(os.path.join(REPO, "match_engine.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "apply_formation_stance"]
    assert len(calls) >= 1, "no apply_formation_stance call site found"
    for c in calls:
        kws = {k.arg for k in c.keywords}
        assert "style" in kws, (
            "apply_formation_stance is called without style= -- the defensive "
            "temperament would be dead in a match, and the deltas above would "
            "all still pass")


def test_apply_formation_stance_stores_style_in_its_idempotency_key():
    """Without this the SECOND apply for a new style short-circuits and the
    team stays on the previous style's line -- a lever that looks wired."""
    import inspect
    src = inspect.getsource(PositionEngine.apply_formation_stance)
    key_line = [ln for ln in src.splitlines() if "key = (" in ln]
    assert key_line, "no idempotency key assignment found"
    assert "style" in key_line[0], (
        f"idempotency key omits style: {key_line[0].strip()}")


def test_the_false_docstring_is_gone():
    """The original defect was a docstring describing a dependency the code
    did not have, which is how it survived. Pin the correction.

    Non-vacuity first: if the docstring no longer MENTIONS the profile at
    all, the positive assertion below would pass vacuously, so both halves
    are checked -- the correction must be PRESENT and the old claim ABSENT.

    NB the wording of the corrected docstring deliberately does not quote
    the original sentence. A docstring that reproduces the false claim in
    order to refute it makes this guard untestable by text scan -- the same
    trap as a comment that quotes deleted code.
    """
    import inspect
    import tactical_shapes
    src = inspect.getsource(tactical_shapes.formation_stance_for)
    assert "does NOT read it" in src, (
        "formation_stance_for's docstring should now state plainly that it "
        "does not read the profile -- that correction is the point of it")
    assert "only .style is read" not in src, (
        "the false claim is back in the docstring")


# ── 5. The sign convention must match the tables it was copied from ────

def test_lateral_signs_match_the_existing_convention():
    """For a wide role resting LOW (LW), positive dy moves INWARD. Getting
    this backwards does not crash and does not shift depth -- it quietly
    swaps a pressing side's width for a block's."""
    sunk = style_defense_deltas("park_the_bus")
    high = style_defense_deltas("attacking")
    # Sunk: LW and RB come inside (dy toward the centre line).
    assert sunk["LW"][1] > 0 and sunk["RW"][1] < 0, sunk
    # High: LW and RB push back out to the touchline.
    assert high["LW"][1] < 0 and high["RW"][1] > 0, high