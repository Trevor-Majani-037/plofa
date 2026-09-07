"""
PLOFA 26/27 — Positional Realism Tests
======================================
test_positional_realism.py

Guards the Checkpoint 25 realism upgrades:

1. Ball-side squeeze — when a team is OUT of possession, the block (GK/CB/
   CDM/CM) slides laterally toward the ball every drift minute, graded by
   danger_level. Below the old defensive_block threshold (danger<25) the
   shape previously stood still laterally.
2. GK space-run fix — the GK branch in _attacker_space_run was dead code
   (GK not in attack_positions) and carried a broken conditional
   expression. It now offers a back-pass outlet in both attack directions.
"""

import pytest

from position_engine import PositionEngine


class _FakeProfile:
    def __init__(self, defensive_line=0.5, width=0.5, tempo=0.5,
                 directness=0.5, press_intensity=0.5):
        self.defensive_line = defensive_line
        self.width = width
        self.tempo = tempo
        self.directness = directness
        self.press_intensity = press_intensity


class _FakeDNA:
    def __init__(self, specialties=(), geometric_awareness=50.0):
        self.specialties = list(specialties)

        class _Mental:
            pass

        mental = _Mental()
        mental.geometric_awareness = geometric_awareness
        self.mental = mental


class _FakePlayer:
    def __init__(self, name, position, specialties=(), geometric_awareness=50.0):
        self.name = name
        self.position = position
        self.dna = _FakeDNA(specialties, geometric_awareness)


def _build_engine(team="Test FC", positions=("CB", "CDM", "CM", "GK"),
                  attacks_right=True):
    players = [
        _FakePlayer(f"P{i}", pos) for i, pos in enumerate(positions)
    ]
    pe = PositionEngine()
    pe.initialize_team(team, players, _FakeProfile(),
                       attacks_right=attacks_right)
    return pe


def _phase(name="second_open"):
    return type("P", (), {"value": name})()


def _paired_drift(positions, in_possession, ball_x, ball_y):
    """Run the same drift twice: danger=0 vs danger=high; return (base, hot)."""
    results = []
    for danger in (0.0, 80.0):
        pe = _build_engine(positions=positions)
        pe.drift_minute(
            "Test FC", _FakeProfile(), _phase(), minute=3,
            in_possession=in_possession, ball_x=ball_x, ball_y=ball_y,
            danger_level=danger,
        )
        results.append({
            n: (pe.states[n].position, pe.states[n].current_y)
            for n in pe.team_rosters["Test FC"]
        })
    return results[0], results[1]


# ─────────────────────────────────────────────
# 1. BALL-SIDE SQUEEZE (Checkpoint 25)
# ─────────────────────────────────────────────

def test_out_of_possession_defenders_squeeze_toward_ball():
    """CB/CDM/CM/GK slide laterally toward ball_y when defending under danger."""
    ball_y = 10.0
    base, hot = _paired_drift(("CB", "CDM", "CM", "GK"), False, 30.0, ball_y)
    for n in hot:
        pos, y_hot = hot[n]
        _, y_base = base[n]
        if pos in ("GK", "CB", "CDM", "CM"):
            assert abs(y_hot - ball_y) < abs(y_base - ball_y), (
                f"{n} ({pos}): danger must pull laterally toward the ball "
                f"(base y={y_base:.2f}, hot y={y_hot:.2f})"
            )


def test_in_possession_never_squeezes():
    """Attacking shape ignores danger — the team stretches, not squeezes."""
    base, hot = _paired_drift(("CB", "CDM", "CM", "GK"), True, 60.0, 10.0)
    for n in hot:
        pos, y_hot = hot[n]
        _, y_base = base[n]
        assert y_hot == y_base, (
            f"{n} ({pos}): danger={80} while in possession must equal danger=0"
        )


def test_squeeze_skips_wingers_and_fullbacks():
    """LW/RW/LB/RB anchor flanks; the block squeeze must not fight them."""
    base, hot = _paired_drift(("LW", "RW", "LB", "RB"), False, 30.0, 10.0)
    for n in hot:
        pos, y_hot = hot[n]
        _, y_base = base[n]
        assert y_hot == y_base, (
            f"{n} ({pos}): flank anchoring must stay untouched by the squeeze"
        )


# ─────────────────────────────────────────────
# 2. GK SPACE-RUN FIX (Checkpoint 25)
# ─────────────────────────────────────────────

def _team_with_high_awareness_gk(attacks_right):
    players = [
        _FakePlayer("Neuer", "GK", geometric_awareness=90.0),
        _FakePlayer("Stones", "CB", geometric_awareness=90.0),
    ]
    pe = PositionEngine()
    pe.initialize_team("Test FC", players, _FakeProfile(),
                       attacks_right=attacks_right)
    return pe


@pytest.mark.parametrize("attacks_right, ball_x, gk_x", [(True, 70.0, 60.0), (False, 35.0, 45.0)])
def test_gk_steps_up_as_outlet_when_ball_deep(attacks_right, ball_x, gk_x):
    pe = _team_with_high_awareness_gk(attacks_right)
    gk = pe.states["Neuer"]
    # Place GK already pushed up in the opponent half (the eligibility gate)
    gk.current_x = gk_x
    x0, y0 = gk.current_x, gk.current_y
    pe._attacker_space_run(
        "Test FC", ball_x, 34.0, def_players=[],
        position_engine=pe, attacks_right=attacks_right, minute=5,
    )
    # GK must have moved toward a central outlet position
    assert (gk.current_x, gk.current_y) != (x0, y0)
    assert 35.0 <= gk.current_x <= 70.0
    assert 20.0 <= gk.current_y <= 48.0


def test_gk_own_half_gate_regression_for_attacking_left():
    """The old conditional-expression evaluated ball_x<45 for away teams and
    ignored the in-opponent-half guard. With the ball in the away team's own
    defensive zone he must stay home (previously he'd get dragged to midfield)."""
    attacks_right, ball_x = False, 30.0   # away GK sits near x=97; ball_x=30 is their third
    pe = _team_with_high_awareness_gk(attacks_right)
    gk = pe.states["Neuer"]
    x0, y0 = gk.current_x, gk.current_y
    pe._attacker_space_run(
        "Test FC", ball_x, 34.0, def_players=[],
        position_engine=pe, attacks_right=attacks_right, minute=5,
    )
    assert (gk.current_x, gk.current_y) == (x0, y0), (
        "attacking-left guard must respect the in-opponent-half check"
    )


# ─────────────────────────────────────────────
# 3. PHYSICS UPGRADES (Checkpoint 26)
# ─────────────────────────────────────────────

from pitch_control import (PlayerInfluenceInput, PitchControlField,
                           BASE_SIGMA, SIGMA_PACE_SCALE)


def _influence_isotropic(p, x, y):
    """The legacy formula, before velocity physics."""
    sigma = max(3.0, BASE_SIGMA + p.pace * SIGMA_PACE_SCALE)
    if p.is_goalkeeper:
        sigma *= 0.6
    dx, dy = x - p.x, y - p.y
    return 1.0 / (1.0 + (dx * dx + dy * dy) / (2.0 * sigma * sigma))


def test_zero_velocity_influence_is_isotropic_unchanged():
    p = PlayerInfluenceInput(name="X", team="T", position="CB",
                             x=10.0, y=20.0, pace=70.0)
    for (qx, qy) in [(11.0, 19.0), (25.0, 40.0), (3.0, 5.0)]:
        assert abs(PitchControlField._influence(p, qx, qy)
                   - _influence_isotropic(p, qx, qy)) < 1e-9


def test_moving_player_controls_more_along_motion():
    static = PlayerInfluenceInput(name="X", team="T", position="CB",
                                  x=10.0, y=20.0, pace=70.0)
    moving = PlayerInfluenceInput(name="X", team="T", position="CB",
                                  x=10.0, y=20.0, pace=70.0, vx=240.0, vy=0.0)
    # ahead along motion (+x) — control is higher than the static blob
    along = moving.x + 8.0
    assert PitchControlField._influence(moving, along, moving.y) > \
           PitchControlField._influence(static, along, static.y)
    # perpendicular (pure y) — control equals the static blob (reach shifts
    # INTO the direction of motion; it is not drained perpendicular)
    perp = moving.y + 8.0
    assert abs(PitchControlField._influence(moving, moving.x, perp) -
               PitchControlField._influence(static, static.x, perp)) < 1e-9


def test_living_space_bonus_monotone_and_saturating():
    from position_engine import PositionEngine
    assert PositionEngine._living_space_bonus(None) == 1.0
    # below reach — no bonus (matches legacy behaviour)
    assert PositionEngine._living_space_bonus(3.0) == 1.0
    assert PositionEngine._living_space_bonus(8.0) == 1.0
    # beyond reach — monotone increasing ~ +45 slope of reach-excess
    b1 = PositionEngine._living_space_bonus(20.0)
    b2 = PositionEngine._living_space_bonus(35.0)
    assert 1.0 < b1 < b2
    assert abs(b2 - (1.0 + (35.0 - 8.0) / 45.0)) < 1e-9


def test_st_space_run_layered_with_living_space_bonus():
    """ST candidate scoring keeps legacy penalties and stacks the reach
    bonus — a run where evading the mark is amplified, not rewritten."""
    players = [
        _FakePlayer("Haaland", "ST", geometric_awareness=95.0),
        _FakePlayer("Van Dijk", "CB", geometric_awareness=95.0),
        _FakePlayer("Alisson", "GK", specialties={"no_nonsense_cb"}),
    ]
    players[1].name = "Van Dijk"
    players[2].name = "Alisson"
    pe = PositionEngine()
    pe.initialize_team("Liverpool", players, _FakeProfile(), attacks_right=True)
    st = pe.states["Haaland"]
    x0, y0 = st.current_x, st.current_y
    defs = [players[1], players[2]]
    pe._attacker_space_run(
        "Liverpool", ball_x=70.0, ball_y=30.0, def_players=defs,
        position_engine=pe, attacks_right=True, minute=10,
    )
    # a run must have resolved (layered bonus works through a full pass)
    assert (st.current_x, st.current_y) != (x0, y0)


# ─────────────────────────────────────────────
# 4. MOTION-AWARE FIELD WIRING (Checkpoint 26)
# ─────────────────────────────────────────────

def _11_players(prefix):
    positions = ["GK", "CB", "CB", "LB", "RB", "CDM", "CM", "CM", "CAM",
                 "LW", "RW"]
    return [_FakePlayer(f"{prefix}{i}", pos) for i, pos in enumerate(positions)]


def two_team_engine():
    pe = PositionEngine()
    pe.initialize_team("HOME", _11_players("H"), _FakeProfile(),
                       attacks_right=True)
    pe.initialize_team("AWAY", _11_players("A"), _FakeProfile(),
                       attacks_right=False)
    return pe


def test_velocity_emas_update_from_drift_snapshot():
    pe = two_team_engine()
    inputs0 = pe.influence_inputs("HOME")
    assert all(i.vx == 0.0 and i.vy == 0.0 for i in inputs0)
    before = pe.snapshot_positions("HOME")
    pe.drift_minute("HOME", _FakeProfile(), _phase(), in_possession=False,
                    minute=1, ball_x=20.0, ball_y=10.0, danger_level=90.0)
    pe.accumulate_drift_from_snapshot("HOME", before)
    inputs = pe.influence_inputs("HOME")
    assert any(abs(i.vx) + abs(i.vy) > 0 for i in inputs), (
        "drift must produce signed velocity deltas"
    )


def test_update_pitch_control_caches_field_and_result():
    pe = two_team_engine()
    pe.update_pitch_control("HOME", "AWAY", minute=7)
    assert pe.pitch_control_field is not None
    assert pe.pitch_control_result is not None
    assert pe.pitch_control_result.minute == 7


def test_winger_openness_falls_back_to_cached_field():
    """half_space_openness must read the engine's cached PC snapshot when no
    explicit args are passed — the Checkpoint 26 wiring."""
    from winger_behavior import WingerRegistry
    players = _11_players("H")
    registry = WingerRegistry()
    registry.register_team(players)
    profile = registry.get("H9")   # LW
    pe = two_team_engine()
    lw = pe.states["H9"]
    pe.update_pitch_control("HOME", "AWAY", minute=7)
    val = profile.half_space_openness(
        x=80.0, y=lw.home_y, attacks_right=True,
        defenders=[_FakePlayer("D1", "CB"), _FakePlayer("D2", "RB")],
        position_engine=pe,
        anchor_y=lw.home_y,
    )
    assert 0.0 <= val <= 1.0


# ─────────────────────────────────────────────
# 5. FORMATION GRAPH PHYSICS (Checkpoint 27)
# ─────────────────────────────────────────────

def test_graph_relaxation_converges_cb_pair_toward_home_rest():
    """Two CBs piled on the same point must spring apart toward home rest."""
    pe = two_team_engine()
    c1, c2 = "H1", "H2"   # the CB pair from _11_players
    s1, s2 = pe.states[c1], pe.states[c2]
    s1.current_x, s1.current_y = 40.0, 34.0
    s2.current_x, s2.current_y = 40.2, 34.1
    pe._graph_relaxation("HOME", in_possession=False)
    dist = abs(s1.current_y - s2.current_y) + abs(s1.current_x - s2.current_x)
    assert dist > 0.31, "CB pair must spring back toward home separation"


def test_graph_relaxation_out_of_possession_tighter():
    """Defending springs must converge a perturbed pair more than attacking."""
    reduction = {}
    for flag in (True, False):
        players = _11_players("Y")
        pe = PositionEngine()
        pe.initialize_team("T", players, _FakeProfile(), attacks_right=True)
        a, b = pe.states["Y1"], pe.states["Y2"]
        a.current_x, a.current_y = 50.0, 34.0   # CB dragged to midfield
        d0 = abs(a.current_x - b.current_x) + abs(a.current_y - b.current_y)
        pe._graph_relaxation("T", in_possession=flag)
        d1 = abs(a.current_x - b.current_x) + abs(a.current_y - b.current_y)
        reduction[flag] = d0 - d1
    assert reduction[False] > reduction[True], (
        "out-of-possession springs must tighten harder than attacking ones"
    )


def test_min_separation_repels_close_team_mates():
    """Two players piled on the same point must spread apart."""
    pe = two_team_engine()
    s1, s2 = pe.states["H1"], pe.states["H2"]
    s1.current_x, s1.current_y = 50.0, 34.0
    s2.current_x, s2.current_y = 50.4, 34.2
    before = abs(s1.current_x - s2.current_x) + abs(s1.current_y - s2.current_y)
    pe._graph_relaxation("HOME", in_possession=False)
    after = abs(s1.current_x - s2.current_x) + abs(s1.current_y - s2.current_y)
    assert after > before, "min-separation repulsion failed"


def test_flank_edges_stay_on_their_side():
    """LW must link to LB (same side) — if the gate broke, the RB-side drag
    would show up as a y-force toward the wrong flank."""
    players = _11_players("F")
    pe = PositionEngine()
    pe.initialize_team("T", players, _FakeProfile(), attacks_right=True)
    lw = pe.states["F9"]
    lb = pe.states["F3"]
    rb = pe.states["F4"]
    # LW dragged to central; RB nearly colocated with LW on centre-right, 
    # LB out at the left flank anchor (its home is already at the left side)
    lw.current_x, lw.current_y = 50.0, 34.0
    rb.current_x, rb.current_y = 50.0, 34.4
    y_before = lw.current_y
    pe._graph_relaxation("T", in_possession=False)
    # only force LW feels must come from the LB edge; sign of its y-delta
    # must point AWAY from the central pile toward LB's flank anchor
    anchor_dir = 1.0 if lb.home_y > 34.0 else -1.0
    delta = (lw.current_y - y_before) * anchor_dir
    assert delta > 0.0 or abs(lw.current_y - y_before) < 0.6, (
        "flank edge pulled LW toward the wrong flank"
    )


# ─────────────────────────────────────────────
# 6. SPACE-CREATION TARGETS (Checkpoint 28)
# ─────────────────────────────────────────────

def test_space_targets_empty_without_cached_field():
    pe = two_team_engine()
    st = pe.states["H9"]
    assert pe._space_targets("HOME", st.current_x, st.current_y) == []


def test_space_targets_after_cached_header_update():
    pe = two_team_engine()
    pe.update_pitch_control("HOME", "AWAY", minute=3)
    st = pe.states["H9"]
    assert pe._space_targets("HOME", st.current_x, st.current_y)


class _ProxyTarget:
    def __init__(self, x, y, score):
        self.x, self.y, self.score = x, y, score


def test_space_bonus_path():
    """No targets → 1.0; nearer target scores higher; window cutoff = 1.0."""
    pe = PositionEngine()
    targets = [_ProxyTarget(60.0, 34.0, 0.04),
               _ProxyTarget(65.0, 40.0, 0.02)]
    assert pe._space_bonus(None, 50.0, 34.0) == 1.0
    near = pe._space_bonus(targets, 60.0, 34.0)
    far = pe._space_bonus(targets, 100.0, 34.0)
    assert near > 1.0 and far == 1.0


def test_st_space_run_uses_space_bonus_when_field_cached():
    """When PC data is cached, ST run scoring stacks the space bonus without
    breaking the legacy penalties/living space lookups."""
    pe = two_team_engine()
    pe.update_pitch_control("HOME", "AWAY", minute=3)
    st = pe.states["H9"]
    x0, y0 = st.current_x, st.current_y
    defs = [_FakePlayer("D1", "CB"), _FakePlayer("D2", "RB")]
    pe._attacker_space_run("HOME", ball_x=70.0, ball_y=30.0, def_players=defs,
                           position_engine=pe, attacks_right=True, minute=3)
    assert (st.current_x, st.current_y) != (x0, y0)


# ─────────────────────────────────────────────
# P5. COMPACTNESS DIAL (horizontal compression)
# ─────────────────────────────────────────────

def _spread_of_backline(pe, team):
    ys = [
        pe.states[n].current_y
        for n in pe.team_rosters[team]
        if pe.states[n].position in ("CB", "LB", "RB", "CDM")
    ]
    return max(ys) - min(ys)


def _block_engine():
    """FT/LB/CB/CB/RB/CDM spread across the pitch, defending own 105.0 goal."""
    players = [
        _FakePlayer("LB", "LB"), _FakePlayer("CB1", "CB"),
        _FakePlayer("CB2", "CB"), _FakePlayer("RB", "RB"),
        _FakePlayer("CDM", "CDM"), _FakePlayer("GK", "GK"),
    ]
    pe = PositionEngine()
    pe.initialize_team("Test FC", players, _FakeProfile(), attacks_right=False)
    for n, y in (("LB", 14.0), ("CB1", 26.0), ("CB2", 42.0),
                 ("RB", 54.0), ("CDM", 34.0), ("GK", 34.0)):
        pe.states[n].current_y = y
    return pe


_BALL = dict(ball_x=80.0, ball_y=12.0, own_goal_x=105.0)


def _block_with_compactness(compactness):
    pe = _block_engine()
    pe.defensive_block(
        "Test FC", danger_level=75.0, pull_strength=0.5,
        compactness=compactness, **_BALL,
    )
    return pe


def test_compactness_default_is_noop_vs_zero():
    """compactness=0.0 (and the omitted default) must leave the block
    untouched laterally — the dial is opt-in only."""
    a = _block_with_compactness(0.0)
    b = _block_engine()  # default -> compactness 0.0 in defensive_block
    b.defensive_block("Test FC", danger_level=75.0, pull_strength=0.5, **_BALL)
    ays = {n: a.states[n].current_y for n in a.team_rosters["Test FC"]}
    bys = {n: b.states[n].current_y for n in b.team_rosters["Test FC"]}
    assert ays == bys


def test_compactness_squeezes_backline_laterally():
    """A packed side (compactness=1.0) leaves a strictly smaller y-spread
    across CB/LB/RB/CDM than a spread side (compactness=0.0)."""
    loose = _block_with_compactness(0.0)
    packed = _block_with_compactness(1.0)
    s_loose = _spread_of_backline(loose, "Test FC")
    s_packed = _spread_of_backline(packed, "Test FC")
    assert s_packed < s_loose, (
        f"compactness=1 y-spread {s_packed:.2f} must be < {s_loose:.2f}"
    )


def test_compactness_keeps_block_depth():
    """The compactness squeeze is purely lateral — the line must not cave
    toward its own goal more than the zero-compactness run does."""
    loose = _block_with_compactness(0.0)
    packed = _block_with_compactness(1.0)
    xr = lambda pe: min(pe.states[n].current_x
                        for n in pe.team_rosters["Test FC"]
                        if pe.states[n].position in ("LB", "RB", "CB", "CDM"))
    assert xr(packed) >= xr(loose) - 0.01
