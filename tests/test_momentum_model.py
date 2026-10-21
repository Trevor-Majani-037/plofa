"""MATCH MOMENTUM — the threat model, and the guards that keep it true.

The defect this suite exists for
-------------------------------
`MomentumEngine` (match_engine.py) had exactly three writers to the momentum
state: a goal, a red card, and a per-minute decay toward zero. Its log was
therefore mathematically obliged to be flat whenever nothing was scored —
most minutes of a football match — and the chart drawn from it showed the
scoreboard rather than the match. The user-facing symptoms were exactly that:
the curve could not rise without a goal, and every goal was pinned at a fixed
height so a scrappy corner goal looked like a wave.

`momentum_model` replaces the REPORTED series with the published Opta/xT
algorithm. These tests pin the two properties the user asked for, plus the
frame and purity properties that make the first one mean anything.

The guards
----------
  * `momentum_model` must not be imported by `match_engine`. The module is a
    pure function of a finished result on purpose, so that reporting a real
    momentum series cannot change a single simulated event (which would
    invalidate every calibration number in AGENTS.md). A guard that walks
    only one file cannot catch the class — this project has hit that four
    times — so it walks the engine's static import closure.
  * the exporter must no longer read `result.momentum_log`. That log is still
    produced and still feeds gameplay; if the chart ever goes back to reading
    it the old defect returns silently and no behavioural test on the model
    would notice.
"""
from __future__ import annotations

import ast
import os
import random

import pytest

import momentum_model as mm
from match_engine import (EventType, MatchConfig, MatchEvent, MatchResult,
                          MatchState, SituationType, XGEngine)

HOME, AWAY = "Home FC", "Away FC"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 30 is the fixture length. The series always spans 90 and the EWMA window is
# 4, so asserting over all 90 rows measures the empty tail, not the fixture.
LAST = 30


# ── helpers ──────────────────────────────────────────────────────────────

def _ev(minute, etype, team, x, y, ex=None, ey=None, outcome=True,
        situation=SituationType.OPEN_PLAY, xg=0.0) -> MatchEvent:
    return MatchEvent(minute=minute, second=30, event_type=etype, team=team,
                      player="P1", secondary_player="P2", situation=situation,
                      location_x=x, location_y=y, end_x=ex, end_y=ey,
                      xg=xg, outcome=outcome)


def _result(timeline, added=0) -> MatchResult:
    st = MatchState()
    st.added_time = added
    return MatchResult(
        config=MatchConfig(home_team=HOME, away_team=AWAY), state=st,
        timeline=list(timeline), goals=[], cards=[], subs=[], squads={},
        threat=None, position_log=[], momentum_log=[])


def _at(result, minute) -> float:
    return next(r["momentum"] for r in result.series if r["minute"] == minute)


def _attack(minutes=range(1, LAST + 1), team=HOME, frm=(60, 30), to=(88, 32)):
    return [_ev(m, EventType.PASS, team, frm[0], frm[1], to[0], to[1])
            for m in minutes]


# ── 1. THE GRID ──────────────────────────────────────────────────────────

def test_grid_has_the_declared_shape():
    g = mm.default_grid()
    assert len(g) == mm.XT_ROWS == 12
    assert all(len(row) == mm.XT_COLS == 18 for row in g)


def test_grid_value_rises_monotonically_toward_the_attacking_goal():
    """Central row: own goal -> halfway -> edge of the box."""
    row = mm.default_grid()[6]
    far, mid, near = row[0], row[9], row[17]
    assert far < mid < near
    assert near > 0.2, f"a shot 3m out should be worth a lot, got {near}"


def test_grid_is_symmetric_about_the_halfway_line():
    """Left and right channels are the same channel. If they were not, the
    model would read a team's flank bias as a threat asymmetry."""
    g = mm.default_grid()
    for r in range(mm.XT_ROWS):
        for c in range(mm.XT_COLS):
            assert g[r][c] == pytest.approx(g[mm.XT_ROWS - 1 - r][c], abs=1e-12)


def test_grid_is_derived_from_the_engines_own_shot_model_at_cell_centres():
    """Not a typed table: every cell equals XGEngine at that cell's CENTRE.

    Asserting the CENTRE and not an arbitrary point matters — a binned lookup
    is quantised, so comparing it against a point value compares two different
    things. That was the first version of this test's mistake.
    """
    g = mm.default_grid()
    for r in range(mm.XT_ROWS):
        for c in range(mm.XT_COLS):
            cx = (c + 0.5) / mm.XT_COLS * mm.PITCH_X
            cy = (r + 0.5) / mm.XT_ROWS * mm.PITCH_Y
            expect = XGEngine.calculate_geometric(
                cx, cy, "right_foot", SituationType.OPEN_PLAY,
                under_pressure=False, attacks_right=True)
            assert g[r][c] == pytest.approx(expect, abs=1e-12)


def test_zone_lookup_is_binned_not_continuous():
    g = mm.default_grid()
    col, row = mm.cell_of(95.0, 34.0)
    cx = (col + 0.5) / mm.XT_COLS * mm.PITCH_X
    cy = (row + 0.5) / mm.XT_ROWS * mm.PITCH_Y
    assert mm.zone_value(g, 95.0, 33.0, True) == mm.zone_value(g, 95.0, 36.0, True)
    assert mm.zone_value(g, 95.0, 34.0, True) == pytest.approx(
        mm.zone_value(g, cx, cy, True), abs=1e-12)


# ── 2. FRAME ─────────────────────────────────────────────────────────────

def test_attacking_frame_normalises_x_for_each_side():
    assert mm.to_attacking_frame(90.0, 34.0, True) == (90.0, 34.0)
    # away attacks LEFT, so its own goal is at raw 105
    assert mm.to_attacking_frame(90.0, 34.0, False) == (15.0, 34.0)


def test_the_same_mirror_matched_attack_is_worth_the_same_to_both_teams():
    """The most important frame test: if this fails, one side of the chart is
    measured in a different frame than the other."""
    g = mm.default_grid()
    home = _ev(1, EventType.PASS, HOME, 60, 30, 88, 32)
    away = _ev(1, EventType.PASS, AWAY, 105 - 60, 30, 105 - 88, 32)
    assert mm.action_value(home, g, True) == pytest.approx(
        mm.action_value(away, g, False), abs=1e-12)


def test_an_identical_match_gives_momentum_of_exactly_zero():
    tl = []
    for m in range(1, LAST + 1):
        tl.append(_ev(m, EventType.PASS, HOME, 60, 30, 88, 32))
        tl.append(_ev(m, EventType.PASS, AWAY, 105 - 60, 30, 105 - 88, 32))
    r = mm.compute_momentum(_result(tl))
    assert all(abs(_at(r, m)) < 1e-9 for m in range(6, LAST + 1))


def test_the_mirror_of_a_one_sided_match_flips_the_sign_exactly():
    home_only = mm.compute_momentum(_result(_attack()))
    away_only = mm.compute_momentum(_result(_attack(
        team=AWAY, frm=(45, 30), to=(17, 32))))
    for m in range(6, LAST + 1):
        assert _at(home_only, m) > 0
        assert _at(away_only, m) < 0
        assert _at(home_only, m) == pytest.approx(-_at(away_only, m), abs=1e-12)


def test_off_pitch_coordinates_are_clamped_for_the_lookup_and_counted():
    """Nine coordinates a match sit off the pitch (to ~-12.8 and 119.7). The
    lookup must not raise on them, and the count must be reported rather than
    absorbed — a silent clamp hides a real datum."""
    tl = [_ev(m, EventType.PASS, HOME, 60, 30, 119.7, -12.8)
          for m in range(1, LAST + 1)]
    r = mm.compute_momentum(_result(tl))
    assert len(r.series) == 90
    assert r.diagnostics["clamped_coordinates"] > 0


# ── 3. ACTION VALUE ──────────────────────────────────────────────────────

def test_a_progressive_pass_is_positive_and_a_regressive_pass_is_negative():
    """The property that makes xT need no special case for bad football."""
    g = mm.default_grid()
    assert mm.action_value(_ev(1, EventType.PASS, HOME, 50, 34, 92, 34), g, True) > 0
    assert mm.action_value(_ev(1, EventType.PASS, HOME, 92, 34, 50, 34), g, True) < 0


def test_the_open_play_situation_bonus_is_exactly_zero():
    """A term meant to model the exceptional case must be zero everywhere
    else. The first version returned `end_value * 1.0` for every pass, so an
    open-play pass scored `2*end - start`, both teams pinned at the clip on
    identical ground, and the plotted difference was zero all match."""
    for s in (SituationType.OPEN_PLAY, SituationType.FAST_BREAK,
              SituationType.THROW_IN):
        assert mm._situation_bonus(s, 0.5) == 0.0
    assert mm._situation_bonus(SituationType.CORNER, 0.5) > 0.0


def test_a_set_piece_delivery_is_discounted_below_open_play():
    g = mm.default_grid()
    owp = _ev(1, EventType.GOAL, HOME, 88, 30)
    corner = _ev(1, EventType.GOAL, HOME, 88, 30, situation=SituationType.CORNER)
    assert mm.action_value(corner, g, True) < mm.action_value(owp, g, True)


def test_a_goal_is_valued_exactly_as_the_shot_it_was():
    """No scoring bonus. A goal IS a shot; if it scored more than the shot
    from the same place, the chart would re-create the defect."""
    g = mm.default_grid()
    for x, y, sit in ((94.0, 34.0, SituationType.OPEN_PLAY),
                      (88.0, 30.0, SituationType.CORNER),
                      (94.0, 10.0, SituationType.OPEN_PLAY)):
        goal = _ev(1, EventType.GOAL, HOME, x, y, situation=sit)
        shot = _ev(1, EventType.SHOT_ON_TARGET, HOME, x, y, situation=sit)
        assert mm.action_value(goal, g, True) == pytest.approx(
            mm.action_value(shot, g, True), abs=1e-12)


def test_a_scrappy_corner_goal_is_worth_far_less_than_a_clean_one():
    g = mm.default_grid()
    scrappy = mm.action_value(
        _ev(1, EventType.GOAL, HOME, 88, 30, situation=SituationType.CORNER),
        g, True)
    clean = mm.action_value(_ev(1, EventType.GOAL, HOME, 94, 34), g, True)
    assert scrappy < clean / 3.0


def test_woodwork_counts_as_threat_even_though_it_is_not_a_shot():
    """`MatchEvent.is_shot` omits HIT_WOODWORK (documented in AGENTS.md), but
    a ball on the frame is a shot in football. The threat grid must not
    inherit that omission — five hits off the woodwork is not zero threat."""
    g = mm.default_grid()
    wood = _ev(1, EventType.HIT_WOODWORK, HOME, 94, 34)
    assert not wood.is_shot
    assert mm.action_value(wood, g, True) > 0


def test_a_failed_ball_mover_without_an_end_point_scores_zero_not_a_guess():
    """`end_x`/`end_y` can be None. Omit, never invent."""
    g = mm.default_grid()
    assert mm.action_value(_ev(1, EventType.PASS, HOME, 80, 34), g, True) == 0.0
    r = mm.compute_momentum(_result(
        [_ev(m, EventType.PASS, HOME, 80, 34) for m in range(1, LAST + 1)]))
    assert r.diagnostics["actions_without_end_point"] == LAST


def test_defensive_and_container_events_never_score():
    """A defender's good work shows up as the ATTACKER's reduced threat. If
    both sides were charged, the two would cancel into a flat line."""
    g = mm.default_grid()
    for et in (EventType.POSSESSION_SEQUENCE, EventType.INTERCEPTION,
               EventType.CLEARANCE, EventType.TACKLE_WON, EventType.PRESS,
               EventType.BALL_RECEIPT, EventType.GOAL_KICK):
        assert mm.action_value(_ev(1, et, HOME, 88, 34, 92, 34), g, True) is None


# ── 4. THE SERIES ────────────────────────────────────────────────────────

def test_momentum_rises_with_no_goal_scored_anywhere_in_the_match():
    """THE defect. The old model's only writers were a goal and a red card,
    so this fixture was mathematically obliged to produce a flat line."""
    tl = []
    for m in range(1, LAST + 1):
        deep = 70 + (m / LAST) * 22.0
        tl.append(_ev(m, EventType.PASS, HOME, 60, 30, deep, 32))
    r = mm.compute_momentum(_result(tl))
    assert not any(e.event_type == EventType.GOAL for e in tl)
    assert _at(r, 28) > 2.0 * _at(r, 8)
    vals = [_at(r, m) for m in range(6, LAST + 1)]
    assert max(vals) - min(vals) > 0.01


def test_a_scrappy_corner_goal_lands_in_a_low_momentum_minute():
    """The user's own example: a goal for an outpossessed side, out of
    nowhere, with the bar for that minute low."""
    tl = []
    for m in range(1, LAST + 1):
        tl.append(_ev(m, EventType.PASS, HOME, 50, 34, 51, 33))
        tl.append(_ev(m, EventType.PASS, AWAY, 50, 34, 51, 33))
    tl += [_ev(25, EventType.CORNER_TAKEN, HOME, 105, 0.5, 88, 30,
               situation=SituationType.CORNER),
           _ev(25, EventType.GOAL, HOME, 88, 30, ex=105, ey=34,
               situation=SituationType.CORNER)]
    r = mm.compute_momentum(_result(tl))
    # a turgid midfield over 6-20 generates essentially nothing
    assert max(_at(r, m) for m in range(6, 21)) < 0.005
    assert _at(r, 25) < 0.02

    # ...and a CLEAN goal in that very same minute sits much higher
    clean = mm.compute_momentum(_result(
        tl[:-1] + [_ev(25, EventType.GOAL, HOME, 94, 34, ex=105, ey=34)]))
    assert _at(clean, 25) > 1.5 * _at(r, 25)


def test_adding_a_goal_moves_the_curve_only_by_the_shots_own_value():
    base = mm.compute_momentum(_result(_attack()))
    withg = mm.compute_momentum(_result(
        _attack() + [_ev(20, EventType.GOAL, HOME, 94, 34, ex=105, ey=34)]))
    assert 0.0 <= (_at(withg, 20) - _at(base, 20)) <= mm.XT_CLIP + 1e-12
    for m in (10, 15, 40, 60):          # outside the EWMA window
        assert _at(withg, m) == pytest.approx(_at(base, m), abs=1e-12)


def test_the_clip_bounds_any_single_minute():
    """Opta's 0.1 cap: without it a penalty (~0.79) flattens the match."""
    tl = [_ev(m, EventType.GOAL, HOME, 94, 34, ex=105, ey=34,
              situation=SituationType.PENALTY) for m in range(1, LAST + 1)]
    r = mm.compute_momentum(_result(tl))
    assert max(_at(r, m) for m in range(1, LAST + 1)) <= mm.XT_CLIP + 1e-12


def test_the_series_always_spans_the_full_match_including_added_time():
    tl = [_ev(m, EventType.PASS, HOME, 60, 30, 88, 32) for m in range(1, LAST + 1)]
    assert len(mm.compute_momentum(_result(tl)).series) == 90
    assert len(mm.compute_momentum(_result(tl, added=5)).series) == 95


def test_scoreline_columns_advance_with_the_goal_events():
    tl = _attack() + [
        _ev(10, EventType.GOAL, HOME, 94, 34, ex=105, ey=34),
        _ev(30, EventType.GOAL, AWAY, 11, 34, ex=0, ey=34),
    ]
    r = mm.compute_momentum(_result(tl))
    row = lambda m: next(x for x in r.series if x["minute"] == m)  # noqa: E731
    assert row(9)["home_goals"] == 0
    assert row(10)["home_goals"] == 1
    assert row(29)["home_goals"] == 1
    assert row(30)["away_goals"] == 1


def test_ewma_weights_are_normalised_and_favour_the_present():
    w = mm._ewma_weights(4, 0.25)
    assert sum(w) == pytest.approx(1.0)
    assert w[0] > w[1] > w[2] > w[3] > 0


def test_only_the_last_few_minutes_contribute():
    """The series at minute M must not depend on an event at minute M-10.
    This is what makes it a momentum model rather than a running total."""
    early = mm.compute_momentum(_result(_attack(minutes=range(1, 8))))
    late = mm.compute_momentum(_result(
        _attack(minutes=range(1, 8)) + _attack(minutes=range(40, 44))))
    assert _at(early, 20) == pytest.approx(0.0, abs=1e-12)
    assert _at(late, 20) == pytest.approx(0.0, abs=1e-12)


# ── 5. PURITY ────────────────────────────────────────────────────────────

def test_computing_momentum_consumes_no_randomness():
    tl = _attack()
    random.seed(1234)
    before = random.getstate()
    mm.compute_momentum(_result(tl))
    assert random.getstate() == before


def test_the_same_timeline_always_gives_the_same_series():
    tl = _attack()
    assert (mm.compute_momentum(_result(tl)).series ==
            mm.compute_momentum(_result(tl)).series)


def test_provenance_reports_which_grid_produced_the_numbers():
    d = mm.compute_momentum(_result(_attack())).diagnostics
    assert d["grid_source"] in ("prior", "fitted", "supplied")
    assert d["goals_affect_curve"] is False
    assert d["xt_clip"] == mm.XT_CLIP
    assert d["window_minutes"] == mm.WINDOW


# ── 6. THE GRID FIT REFUSES A THIN SAMPLE ────────────────────────────────

def test_fit_refuses_to_install_a_grid_from_too_few_shots():
    """A grid fitted from three shots per cell is not a model, it is noise
    wearing a model's clothes. It must refuse AND say why."""
    shots = [_ev(m, EventType.SHOT_OFF_TARGET, HOME, 94, 34) for m in range(1, 4)]
    grid, diag = mm.fit_grid_from_matches([_result(shots)])
    assert diag["fitted"] is False
    assert "PRIOR" in diag["reason"]
    assert grid == mm.default_grid()


# ── 7. AST GUARDS ────────────────────────────────────────────────────────

def _module_imports(path):
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_match_engine_does_not_import_momentum_model():
    """The dependency is deliberately ONE-WAY.

    `momentum_model` is a pure function of a finished MatchResult so that
    reporting a real momentum series cannot perturb a single simulated event.
    The day match_engine can import it, that guarantee is gone.
    """
    assert "momentum_model" not in _module_imports(
        os.path.join(ROOT, "match_engine.py"))


def test_no_engine_module_imports_momentum_model():
    """...and a guard that walks only match_engine.py is the 'one-file hole'
    this project has hit four times. It cannot see a lazy import inside a
    function, so this walks the engine's STATIC import closure instead.

    ONE NARROW EXEMPTION, AND IT IS ASSERTED BELOW.

    `threat_reward` is the TRAINING reward (the xT potential difference an
    evolved brain is scored against). It reads ``momentum_model.default_grid``
    and nothing else, because the grid must have exactly one definition: the
    whole point of deriving every cell from ``XGEngine.calculate_geometric``
    is that the reward and the engine's own xG cannot disagree. Re-deriving
    it here would be the second source of truth this project keeps getting
    bitten by.

    It reaches the closure through a TRAINING edge, not a live one:
    ``match_engine -> event_chain -> brain_integration -> consequence_decision
    -> critic_surrogate -> surrogate_collect -> threat_reward``. Everything
    from `surrogate_collect` onwards is training code; `critic_surrogate` uses
    nothing but the `FitnessSurrogate` dataclass.

    What the exemption costs is paid back by three assertions: the allowlist is
    EXACTLY this one file (a second entry, or a new offender, fails), and
    `threat_reward` may not draw randomness, write to momentum_model, or touch
    engine state. If any of those change, this guard stops being safe.
    """
    #: The only modules allowed to import momentum_model while being reachable
    #: from match_engine.py. One entry, deliberately, and it must be justified
    #: above before it is added here.
    PURE_TRAINING_CONSUMERS = {"threat_reward.py"}

    seen, stack, offenders = set(), [os.path.join(ROOT, "match_engine.py")], []
    while stack:
        cur = stack.pop()
        if cur in seen or not os.path.exists(cur):
            continue
        seen.add(cur)
        mods = _module_imports(cur)
        if "momentum_model" in mods:
            rel = os.path.relpath(cur, ROOT).replace(os.sep, "/")
            if rel not in PURE_TRAINING_CONSUMERS:
                offenders.append(rel)
        for m in mods:
            cand = os.path.join(ROOT, m.replace(".", os.sep) + ".py")
            if os.path.exists(cand) and cand not in seen:
                stack.append(cand)
    assert not offenders, \
        f"simulation modules must not import momentum_model: {offenders}"
    # Non-vacuity: the walk really did reach the exempt module. If a refactor
    # moves it out of the closure, this fails and the exemption should go too
    # rather than sit here pretending to be load-bearing.
    assert "threat_reward.py" in {os.path.relpath(p, ROOT).replace(os.sep, "/")
                                  for p in seen}


def test_the_training_reward_is_a_pure_reader_of_the_threat_grid():
    """`threat_reward` is allowed one read of momentum_model. Nothing else.

    A reward function that mutated the grid, seeded an RNG or touched engine
    state would make evolution non-reproducible AND make the exempted import
    above unsafe, so the exemption is only as good as this test.
    """
    path = os.path.join(ROOT, "threat_reward.py")
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src)

    imported = _module_imports(path)
    assert "momentum_model" in imported
    # The ONLY momentum_model names it may use: the pure grid helpers.
    allowed_mm = {"default_grid", "zone_value", "to_attacking_frame", "cell_of"}
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "_mm":
            used.add(node.attr)
    assert used <= allowed_mm, f"threat_reward reaches into momentum_model: {used - allowed_mm}"
    assert used, "threat_reward no longer uses momentum_model; drop the exemption"

    # No randomness, and no writes back into the grid.
    for banned in ("import random", "import numpy", "np.random", "set_grid_from"):
        assert banned not in src, f"threat_reward must stay pure: found {banned!r}"



def test_the_exporter_no_longer_plots_the_goal_driven_momentum_log():
    """`result.momentum_log` still exists and still feeds GAMEPLAY. It must
    not come back as the REPORTED series, because no behavioural test on
    momentum_model would notice that regression.

    This guard parses the AST and looks for real attribute ACCESSES. A
    text-scan version of it failed, for exactly the reason AGENTS.md records
    twice: `plot_momentum`'s own docstring quotes `result.momentum_log` in
    order to explain what changed, so a substring scan matched the
    explanation. Stripping `#` comments is not enough — the string was in a
    DOCSTRING. A name inside prose is not a call site, and only structure can
    tell the difference.
    """
    with open(os.path.join(ROOT, "exporter.py"), "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    accessed = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "momentum_log" not in accessed, \
        "the exporter reads result.momentum_log again — that is the " \
        "goal-driven series this whole module replaced"
    assert "momentum_series" in accessed