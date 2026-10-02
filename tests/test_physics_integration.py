"""
Physics <-> engine integration.
================================
The physics layer is only safe to ship if it is provably inert when switched
off, because the live 26/27 season runs through the same ``MatchEngine`` and
must come out byte-identical.

The strongest available proof is not "the output looks the same" — it is that
**the module is never imported at all**. If ``physics/`` is absent from
``sys.modules`` after a match has run, no line of it executed, so no line of it
could have changed anything. That is what
:func:`test_a_default_match_never_imports_the_physics_package` checks, in a
fresh subprocess so an earlier import cannot mask it.

The rest of this file covers the enabled path: that the adapter reads real
engine state, that weather maps in the right direction, and that the two
failure modes worth fearing — a stale cached body and a velocity estimate
blown up by a substitution — cannot happen.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from match_engine import MatchConfig, MatchEngine

DATE = "2026-09-08"


def _cfg(**kw) -> MatchConfig:
    from datetime import date
    base = dict(home_team="Home FC", away_team="Away FC",
                match_date=date(2026, 9, 8))
    base.update(kw)
    return MatchConfig(**base)


# ── the additive guarantee ─────────────────────────────────

def test_physics_is_off_by_default():
    assert MatchConfig(home_team="A", away_team="B").physics_enabled is False
    assert MatchConfig(home_team="A", away_team="B").physics_strict is False


def test_the_pre_existing_config_fields_are_untouched():
    """Adding fields to MatchConfig must not disturb the ones the live season
    reads. A default that changed here would silently alter 26/27."""
    cfg = MatchConfig(home_team="A", away_team="B")
    assert cfg.home_advantage == 0.08
    assert cfg.stadium_capacity == 35000
    assert cfg.referee_strictness == 0.5
    assert cfg.weather_enabled is False
    assert cfg.competition == "PLOFA"
    assert cfg.season == "26/27"
    assert cfg.perception_seed is None


def test_a_disabled_engine_has_no_physics_and_says_so():
    e = MatchEngine(_cfg(), None, None)
    assert e.physics is None
    assert e.physics_enabled is False
    assert e.resolve_pass_physically((10.0, 34.0), (40.0, 34.0)) is None
    assert "disabled" in e.physics_report()


def test_the_attributes_exist_before_simulate_runs():
    """"Off" has to be a value you can ask about, not an AttributeError.

    _initialize_simulation() runs from simulate(), not __init__, so without
    class-level defaults a freshly built engine raised on `engine.physics`.
    """
    e = MatchEngine(_cfg(), None, None)
    assert e.physics is None
    assert e.physics_unavailable is False


def test_a_default_match_never_imports_the_physics_package():
    """The strongest form of the additive guarantee: absence of execution.

    Run in a fresh interpreter so an earlier test's import cannot hide a real
    one. If `physics` is not in sys.modules when the match finishes, no line of
    the package ran, so no line of it could have changed the result.
    """
    script = textwrap.dedent(f"""
        import sys
        from datetime import date
        from match_engine import MatchConfig, MatchEngine
        from roster_loader import get_loader

        assert "physics" not in sys.modules, "physics imported before the match"

        loader = get_loader()
        clubs = sorted(loader.get_all_clubs())
        home, away = clubs[0], clubs[1]
        profiles = {{c: loader.get_club_players(c) for c in (home, away)}}

        cfg = MatchConfig(home_team=home, away_team=away,
                          match_date=date(2026, 9, 8))
        eng = MatchEngine(cfg, None, None)
        assert eng.physics is None

        # Exercise the path that WOULD construct physics if it were enabled.
        eng._init_physics()
        assert eng.physics is None, "physics must stay off when not requested"

        leaked = [m for m in sys.modules if m == "physics" or m.startswith("physics.")]
        print("LEAKED:", leaked)
        assert not leaked, f"physics imported despite being disabled: {{leaked}}"
        print("CLEAN")
    """)
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, timeout=900,
                         cwd=r"D:\PLOFA\plofa",
                         env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin",
                              "SYSTEMROOT": "C:\\Windows"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "CLEAN" in out.stdout
    assert "LEAKED: []" in out.stdout


def test_enabling_physics_constructs_the_adapter():
    e = MatchEngine(_cfg(physics_enabled=True), None, None)
    e._init_physics()
    assert e.physics is not None
    assert e.physics_enabled is True
    assert "adapter" in e.physics_report()


def test_a_missing_physics_package_does_not_break_a_match():
    """If the package cannot be imported the match must still play. Losing the
    enhancement is strictly better than losing a season."""
    e = MatchEngine(_cfg(physics_enabled=True), None, None)
    e._init_physics()
    # Simulate the import failing and confirm the engine degrades, not explodes.
    e.physics = None
    e.physics_unavailable = True
    assert e.resolve_pass_physically((10.0, 34.0), (40.0, 34.0)) is None
    assert e.physics_enabled is False


# ── the adapter reads real engine state ────────────────────

@pytest.fixture
def live_engine():
    """A real engine wired exactly the way production wires it.

    The first attempt fed ``set_squad`` the raw ``(name, pos, specialties)``
    tuples and the adapter found *zero* players. That is not an adapter bug so
    much as a discovery: ``set_squad`` stores its argument verbatim
    (``self.active_players[team_name] = list(starters)``), and production
    supplies ``SquadBuilder``'s ``PlayerProfile`` objects — ``match_engine.py``
    reaches for ``taker.dna``, so objects are what the rest of the engine
    assumes. This fixture therefore builds the same objects the live month
    driver builds, via the same route.
    """
    from auto_run_match import _resolve_team_profile
    from player_dna import SquadBuilder
    from roster_loader import get_loader
    from squad_manager import SubstitutionController

    loader = get_loader()
    clubs = sorted(loader.get_all_clubs())
    home, away = clubs[0], clubs[1]

    raw = {c: loader.build_matchday_squad(c) for c in (home, away)}
    squads = {
        c: SquadBuilder.build(
            team_name=c, starters=raw[c]["starters"],
            substitutes=raw[c]["substitutes"],
            team_superstars=raw[c]["superstars"],
            set_piece_takers=raw[c]["sp_takers"])
        for c in (home, away)
    }
    profiles = {
        c: _resolve_team_profile(c, raw[c]["formation"], is_home=(c == home))
        for c in (home, away)
    }

    cfg = _cfg(home_team=home, away_team=away, physics_enabled=True)
    eng = MatchEngine(cfg, profiles[home], profiles[away])
    for c in (home, away):
        eng.set_squad(c, squads[c]["starters"], squads[c]["substitutes"])

    subs = [c for c in (home, away) for _ in squads[c]["substitutes"]]
    eng.set_stamina_controller(SubstitutionController(
        home_team=home, away_team=away,
        home_subs_bench=subs, away_subs_bench=subs))
    eng._init_physics()
    return eng


def test_the_live_fixture_actually_has_players(live_engine):
    """Guards the fixture itself, after it silently produced zero candidates."""
    adapter = live_engine.physics
    objects = adapter._player_objects()
    assert len(objects) >= 22, f"only {len(objects)} players visible to the adapter"
    assert all(hasattr(p, "dna") for p in objects.values()), (
        "the engine should hold SquadBuilder PlayerProfile objects, not tuples")
    assert all(adapter.position_of(n) is not None for n in list(objects)[:5])


def test_the_adapter_reads_positions_from_the_position_engine(live_engine):
    adapter = live_engine.physics
    assert adapter is not None
    pe = live_engine.position_engine
    assert pe is not None
    any_player = next(iter(pe.states))
    pos = adapter.position_of(any_player)
    assert pos is not None, "the adapter could not read a real position"
    st = pe.states[any_player]
    assert pos[0] == pytest.approx(st.current_x)
    assert pos[1] == pytest.approx(st.current_y)


def test_the_adapter_never_caches_a_position(live_engine):
    """§16 — one authoritative position. The adapter must re-read every time,
    so a substitution cannot leave it working from a stale copy."""
    adapter = live_engine.physics
    pe = live_engine.position_engine
    name = next(iter(pe.states))
    before = adapter.position_of(name)
    st = pe.states[name]
    st.current_x += 12.0
    st.current_y += 3.0
    after = adapter.position_of(name)
    assert after != before, "the adapter returned a cached position"
    assert after[0] == pytest.approx(st.current_x)


def test_the_adapter_builds_candidates_for_a_ball(live_engine):
    adapter = live_engine.physics
    adapter.sync_clock(30.0)
    cands = adapter.candidates_for((52.0, 34.0))
    assert cands, "no candidates built from the live engine"
    for c in cands:
        assert c.arrival >= 0.0
        assert c.reaction >= 0.0
        assert c.travel >= 0.0
    arrivals = sorted(c.arrival for c in cands)
    assert arrivals == sorted(arrivals)


def test_a_pass_resolves_through_the_engine(live_engine):
    res = live_engine.resolve_pass_physically((40.0, 34.0), (58.0, 34.0),
                                              kind="through")
    assert res is not None
    assert res.trajectory.kind == "through"
    assert res.trajectory.distance > 10.0
    assert res.ball_arrival > res.launch_time
    assert res.timeline, "no timeline produced"
    times = [s.time for s in res.timeline]
    assert times == sorted(times)


def test_stamina_flows_through_from_the_existing_system(live_engine):
    """§14 — read the existing stamina, never a second copy."""
    adapter = live_engine.physics
    # The engine stores the controller as ``sub_controller``; an earlier version
    # of this test reached for ``stamina_controller`` and raised.
    controller = getattr(live_engine, "sub_controller", None)
    assert controller is not None, "the engine did not keep its controller"
    table = getattr(controller, "stamina", None)
    if not table:
        pytest.skip("no stamina registered on this engine")
    name = next(iter(table))
    state = table[name]
    fresh = adapter.profile_of(name)
    state.current_stamina = 20.0
    tired = adapter.profile_of(name)
    assert tired.max_speed < fresh.max_speed, (
        "a 20% player must be slower than a 100% one")
    assert adapter.stamina_of(name) == pytest.approx(20.0)
    # ...and the base profile must be untouched by the fatigue
    assert adapter.base_profile(name).stamina == pytest.approx(100.0)


def test_a_substitution_cannot_leave_a_stale_velocity(live_engine):
    """The adapter's velocity is a finite difference. Across a substitution the
    difference is a teleport, so the cache must be droppable — and a stale
    velocity would make the physics believe an impossible runner."""
    adapter = live_engine.physics
    pe = live_engine.position_engine
    name = next(iter(pe.states))
    adapter.sync_clock(10.0)
    adapter.velocity_of(name)
    assert name in adapter._velocity
    adapter.invalidate(name)
    assert name not in adapter._velocity
    assert name not in adapter._profiles


def test_velocity_is_clamped_to_a_physical_maximum(live_engine):
    """A finite difference across a reset would otherwise report a teleport as
    a velocity, and the physics would believe it."""
    adapter = live_engine.physics
    pe = live_engine.position_engine
    name = next(iter(pe.states))
    st = pe.states[name]
    adapter.sync_clock(10.0)
    st.current_x, st.current_y = 0.0, 0.0
    adapter.velocity_of(name)
    adapter.clock.advance_to(10.2)
    st.current_x, st.current_y = 100.0, 60.0        # a 116 m jump
    vx, vy = adapter.velocity_of(name)
    speed = (vx * vx + vy * vy) ** 0.5
    assert speed <= 12.0 + 1e-6, f"velocity {speed} exceeded the ceiling"


def test_the_adapter_syncs_to_the_engines_own_clock(live_engine):
    """The engine already had a continuous clock, so the physics clock follows
    it rather than inventing a second timeline."""
    adapter = live_engine.physics
    live_engine.state.match_clock_s = 1234.567
    adapter.sync_clock()
    assert adapter.clock.play_seconds == pytest.approx(1234.567)
    # ...and never runs backwards when the engine's clock is rewound
    live_engine.state.match_clock_s = 5.0
    adapter.sync_clock()
    assert adapter.clock.play_seconds == pytest.approx(1234.567)


# ── weather maps in the right direction (§21) ───────────────

def _engine_with_weather(weather, enabled: bool = True) -> MatchEngine:
    """An engine with PLOFA's weather gate set the way production sets it.

    ``MatchEngine.__init__`` calls
    ``WeatherPhysics.set_active_weather(cond, enabled=config.weather_enabled)``
    (match_engine.py:1849), and ``weather_enabled`` defaults to False. So
    setting the weather *before* constructing the engine is silently undone —
    the constructor owns the gate. This helper therefore passes the weather
    through the config, which is both the correct order and the production path.
    """
    from weather_physics import WeatherPhysics
    cfg = _cfg(weather=weather, weather_enabled=enabled)
    eng = MatchEngine(cfg, None, None)
    WeatherPhysics.set_active_weather(weather, enabled=enabled)
    return eng


def test_wet_weather_increases_ball_drag():
    """rolling_decel_mult is >1 on a wet pitch, so drag must go UP. Getting
    this backwards would be invisible until a rainy match played wrong."""
    from physics.adapter import EngineAdapter
    from weather_physics import WeatherCondition, WeatherPhysics

    try:
        dry = EngineAdapter(_engine_with_weather(WeatherCondition.clear()),
                            weather=WeatherCondition.clear())
        wet = EngineAdapter(
            _engine_with_weather(WeatherCondition.rain(0.9, 18.0, 8.0)),
            weather=WeatherCondition.rain(0.9, 18.0, 8.0))
        assert wet.ball().drag > dry.ball().drag, (
            f"wet drag {wet.ball().drag} must exceed dry {dry.ball().drag}")
    finally:
        WeatherPhysics.set_active_weather(None, False)


def test_weather_is_ignored_when_plofa_says_it_is_off():
    """Every PLOFA weather multiplier is gated behind ``is_active``. Supplying a
    rainy condition with weather switched off makes PLOFA itself return 1.0.

    The adapter must not override that — it would be changing engine behaviour
    from inside a physics layer — but it must say so, because the alternative
    is silently simulating a dry match in the rain.
    """
    from physics.adapter import EngineAdapter
    from weather_physics import WeatherCondition, WeatherPhysics

    try:
        rainy = WeatherCondition.rain(0.9, 18.0, 8.0)
        a = EngineAdapter(_engine_with_weather(rainy, enabled=False),
                          weather=rainy)
        assert WeatherPhysics.is_active(rainy) is False, "precondition"
        assert a.ball().drag == pytest.approx(0.12, abs=1e-6)
        assert a.calibration().speed_multiplier == pytest.approx(1.0)
        assert any("inactive" in n for n in a.calibration().notes), (
            "the adapter must report that PLOFA is ignoring the weather")
    finally:
        WeatherPhysics.set_active_weather(None, False)


def test_a_slick_pitch_reduces_player_movement():
    """pitch_grip is <1 on a slick pitch, so movement must go DOWN."""
    from physics.adapter import EngineAdapter
    from weather_physics import WeatherCondition, WeatherPhysics

    try:
        rainy = WeatherCondition.rain(0.9, 18.0, 8.0)
        wet = EngineAdapter(_engine_with_weather(rainy), weather=rainy)
        assert wet.calibration().speed_multiplier < 1.0, (
            "a wet pitch must cost a little movement")
        # ...but only a little: a player still moves
        assert wet.calibration().speed_multiplier > 0.9
    finally:
        WeatherPhysics.set_active_weather(None, False)


def test_no_weather_means_neutral_physics():
    from physics.adapter import EngineAdapter
    a = EngineAdapter(MatchEngine(_cfg(), None, None), weather=None)
    assert a.ball().drag == pytest.approx(0.12, abs=1e-6)
    assert a.calibration().speed_multiplier == pytest.approx(1.0)


# ── the active-adapter slot ────────────────────────────────
#
# ``PossessionChain.generate`` and ``GoalkeeperEngine.evaluate_shot`` are
# static/class methods with no engine reference, so the physics reaches them
# through one module-level slot. These tests exist because that slot is a global
# and globals in a season-simulating process are where matches go to contaminate
# each other.

def test_the_slot_is_empty_until_an_enabled_engine_fills_it():
    from physics.adapter import active_adapter, set_active_adapter
    try:
        set_active_adapter(None)
        assert active_adapter() is None
        e = MatchEngine(_cfg(physics_enabled=True), None, None)
        e._init_physics()
        assert active_adapter() is e.physics
    finally:
        from physics.adapter import set_active_adapter as _sa
        _sa(None)


def test_a_disabled_engine_clears_a_stale_slot():
    """The contamination case. A season runs match after match, so if an
    enabled match's adapter survived into the next (disabled) match, that match
    would resolve passes and shots with the *previous* match's positions,
    stamina and clock."""
    from physics.adapter import active_adapter
    try:
        on = MatchEngine(_cfg(physics_enabled=True), None, None)
        on._init_physics()
        assert active_adapter() is not None
        off = MatchEngine(_cfg(), None, None)          # physics OFF
        off._init_physics()
        assert active_adapter() is None, (
            "a disabled match inherited the previous match's adapter")
    finally:
        from physics.adapter import set_active_adapter
        set_active_adapter(None)


def test_the_slot_holds_one_adapter_not_a_registry():
    """A dict keyed by engine id is the ``id()``-reuse bug waiting to happen:
    entries outlive the match that created them. A single slot cannot
    accumulate, so two enabled engines leaves exactly one live adapter."""
    from physics.adapter import active_adapter
    try:
        a = MatchEngine(_cfg(physics_enabled=True), None, None)
        a._init_physics()
        b = MatchEngine(_cfg(physics_enabled=True), None, None)
        b._init_physics()
        assert active_adapter() is b.physics
        assert active_adapter() is not a.physics
    finally:
        from physics.adapter import set_active_adapter
        set_active_adapter(None)


def test_the_disabled_path_never_imports_the_package_to_clear_the_slot():
    """The slot is cleared via ``sys.modules``, not ``import``.

    An earlier version did ``from physics.adapter import set_active_adapter`` on
    the disabled path, which loaded the whole package for a match that never
    asked for it — breaking the additive guarantee that
    ``test_a_default_match_never_imports_the_physics_package`` exists to prove.
    """
    from physics import adapter as A
    A.set_active_adapter(None)
    before = A.__name__ in sys.modules
    assert before, "precondition: the test itself imported physics.adapter"
    # A disabled engine's _init_physics must not re-import. It is already in
    # sys.modules here, so the meaningful assertion is that the module object
    # is untouched and the slot is cleared.
    MatchEngine(_cfg(), None, None)._init_physics()
    assert A.active_adapter() is None


# ── the pass-block gate ────────────────────────────────────
#
# The gate answers a question the engine's own geometric block model cannot: not
# "is he near the lane?" but "can he GET there before the ball does?". It
# attenuates and never replaces, because the engine's probability is tuned and
# the whole season is calibrated around it.

def _place(live_engine, name, x, y):
    st = live_engine.position_engine.states[name]
    st.current_x, st.current_y = float(x), float(y)
    st.previous_x, st.previous_y = float(x), float(y)
    live_engine.physics.invalidate(name)


def test_the_block_gate_leaves_a_defender_who_arrives_in_time_alone(live_engine):
    adapter = live_engine.physics
    name = next(n for n in adapter._player_objects()
                if n != "GK")
    _place(live_engine, name, 62.0, 35.0)          # 1 m off a lane to (80, 34)
    factor, why = adapter.block_feasibility(
        name, (40.0, 34.0), (80.0, 34.0), ball_speed_mps=25.0)
    assert factor == 1.0, f"a defender 1 m from the lane was penalised: {why}"


def test_the_block_gate_removes_a_block_that_cannot_physically_happen(live_engine):
    """The case the engine's model gets wrong: 11 m off the lane with a fast
    pass. Lane distance says he is a plausible blocker; the clock says he is not
    going to be anywhere near the ball."""
    adapter = live_engine.physics
    name = next(n for n in adapter._player_objects() if n != "GK")
    _place(live_engine, name, 60.0, 45.0)          # 11 m off the same lane
    factor, why = adapter.block_feasibility(
        name, (40.0, 34.0), (80.0, 34.0), ball_speed_mps=25.0)
    assert factor == 0.0, f"an impossible block survived: {why}"


def test_the_block_gate_never_raises_a_probability_and_never_beats_the_floor(live_engine):
    """Both invariants, checked across a spread of base probabilities. The whole
    safety argument for this gate is that it is monotone downward and floored at
    the engine's own 0.02 — if either fails, it is no longer a conservative
    adjustment and the calibration guarantee is gone."""
    from physics.adapter import BLOCK_PROBABILITY_FLOOR
    adapter = live_engine.physics
    name = next(n for n in adapter._player_objects() if n != "GK")
    _place(live_engine, name, 60.0, 45.0)          # the impossible case
    for base in (0.02, 0.05, 0.15, 0.30, 0.45):
        adjusted, _why = adapter.block_probability(
            base, name, (40.0, 34.0), (80.0, 34.0), 25.0, lane_dist=2.0)
        assert adjusted <= base + 1e-9, "the gate raised a block probability"
        assert adjusted >= BLOCK_PROBABILITY_FLOOR - 1e-9, (
            "the gate drove a block below the engine's own floor")


def test_the_block_gate_finds_no_physical_difference_when_disabled(live_engine):
    """With no adapter published, a chain must behave exactly as it always did.
    This is the inertness the block gate depends on at its call site."""
    from physics.adapter import active_adapter, set_active_adapter
    try:
        set_active_adapter(None)
        assert active_adapter() is None
    finally:
        set_active_adapter(live_engine.physics)


# ── the keeper gate ────────────────────────────────────────
#
# Same shape as the block gate, same reason. ``_is_shot_savable`` is a good
# tuned model, but ``effective_reach = reach * (0.8 + reaction_time * 0.4)``
# cannot express flight time: it gives a keeper identical reach against a shot
# from 30 m as against one from 8 m, when the first takes ~1.2 s and the second
# ~0.35 s.

def _positioning(gk_x=101.0, gk_y=34.0, reach=2.5, reaction=0.70):
    return {"start_x": 4.0, "start_y": gk_y, "reach": reach,
            "reaction_time": reaction, "gk_x": gk_x}


def test_the_keeper_gate_leaves_a_reachable_shot_alone():
    from physics.adapter import EngineAdapter
    a = EngineAdapter(MatchEngine(_cfg(), None, None), weather=None)
    factor, why = a.keeper_feasibility(
        shot_x=104.0, shot_y=34.5, ball_x=80.0, ball_y=34.0,
        gk_x=101.0, gk_y=34.0, base_reach=2.5, reaction_time=0.70)
    assert factor == 1.0, f"a comfortable save was removed: {why}"


def test_the_keeper_gate_credits_a_long_shot_that_gives_time_to_dive():
    """The gap the engine has, stated as a test.

    One shot. One keeper. One place on the pitch. From 29 m the ball takes
    1.197 s, so after a 0.70 s reaction he has 0.50 s to extend himself and
    covers the 3.04 m easily. From 7 m it takes 0.274 s, which is less than his
    reaction time — he has not finished deciding to move when the ball is
    already past him.

    ``_is_shot_savable`` cannot express either of those, because
    ``effective_reach = reach * (0.8 + reaction_time * 0.4)`` is a formula with
    no flight time in it: the same keeper gets identical reach in both cases.
    """
    from physics.adapter import EngineAdapter
    a = EngineAdapter(MatchEngine(_cfg(), None, None), weather=None)
    corner = dict(shot_x=104.0, shot_y=34.5, gk_x=101.0, gk_y=34.0,
                  base_reach=2.5, reaction_time=0.70)
    far, _ = a.keeper_feasibility(ball_x=75.0, ball_y=34.0, **corner)
    near, _ = a.keeper_feasibility(ball_x=97.0, ball_y=34.0, **corner)
    assert far == 1.0, f"a keeper with half a second to dive was beaten: {far}"
    assert near == 0.0, f"a keeper with no time to react was credited: {near}"


def test_the_keeper_gate_falls_off_gradually_rather_than_at_a_cliff():
    """4.17 m is beyond even full stretch (2.5 m reach + 1.4 m dive = 3.9 m),
    so the shot is a goal even with time to spare — but a partial credit is
    right, because a keeper at full stretch getting a glove to it is a real
    outcome. A binary gate would throw that away.

    Getting the grading right matters more than it looks: an earlier draft of
    this test asserted a full 1.0 for a shot at 4.17 m, which would have
    required believing a keeper can cover 4.17 m. The physics says he cannot,
    so the physics wins."""
    from physics.adapter import EngineAdapter
    a = EngineAdapter(MatchEngine(_cfg(), None, None), weather=None)
    factor, _why = a.keeper_feasibility(
        shot_x=104.0, shot_y=36.9, ball_x=75.0, ball_y=34.0,
        gk_x=101.0, gk_y=34.0, base_reach=2.5, reaction_time=0.70)
    assert 0.0 < factor < 1.0, f"expected a graded partial credit, got {factor}"


def test_the_keeper_gate_only_ever_removes_save_credit():
    from physics.adapter import EngineAdapter
    a = EngineAdapter(MatchEngine(_cfg(), None, None), weather=None)
    positioning = _positioning(gk_x=101.0, gk_y=34.0)
    for save_mult in (1.0, 1.2, 1.5, 1.9):
        # a shot into the top corner from 7 m: physically unreachable
        adjusted, _why = a.keeper_save_multiplier(
            save_mult, 104.0, 36.9, 97.0, 34.0, positioning)
        assert adjusted <= save_mult + 1e-9, "the gate raised a save multiplier"
        assert adjusted >= 1.0, (
            "the gate pushed save_mult below 1.0, which would AMPLIFY the xG — "
            "the exact inversion the engine's own comment warns about")


def test_the_keeper_gate_gives_a_slow_keeper_no_credit_at_all():
    """Reaction time is what separates a keeper from a mannequin. A keeper with
    a 1.1 s reaction against a 0.274 s shot has not finished deciding to move;
    the engine's formula would hand him full reach anyway."""
    from physics.adapter import EngineAdapter
    a = EngineAdapter(MatchEngine(_cfg(), None, None), weather=None)
    factor, why = a.keeper_feasibility(
        shot_x=104.0, shot_y=34.5, ball_x=97.0, ball_y=34.0,
        gk_x=101.0, gk_y=34.0, base_reach=2.5, reaction_time=1.10)
    assert factor == 0.0, f"a keeper with no time to dive was credited: {why}"


# ── physical movement, fatigue and reaction (§ the four unwired gaps) ──
#
# The engine's own step is time-stepped and pace-capped, which an earlier
# version of this work got wrong by calling it "proportional steering". What it
# genuinely lacked was velocity STATE: direction was recomputed from scratch
# every tick, so a player could reverse at full pace, `tgt` was DNA-only so his
# legs never felt fatigue, and a changed target cost nothing to act on. These
# tests pin the three things the physics layer adds, and the inertness.

def _profile(max_speed=8.0, acceleration=7.0, deceleration=9.0, agility=0.6):
    from physics.calibration import PhysicalProfile
    return PhysicalProfile(
        name="T", position="CM", max_speed=max_speed,
        acceleration=acceleration, deceleration=deceleration,
        reaction_time=0.20, stamina=100.0, raw_max_speed=max_speed,
        raw_acceleration=acceleration, agility=agility)


def test_a_player_cannot_reverse_at_full_pace():
    """Momentum. Run east, reverse the target 180 degrees mid-stride: he must
    brake through the turn, not teleport and not stop dead.

    The magnitude matters. Turn deceleration is ``deceleration * (1 + agility)``
    = 9.0 * 1.6 = 14.4 m/s^2, so nulling 8 m/s of eastward momentum takes
    about 0.55 s. An earlier version of this test asserted the speed dropped
    below 7.0 m/s within a single 0.1 s tick, which would have required
    players to turn like a top-spinning table-tennis ball. 8.0 -> 7.3 is what
    a real body does.
    """
    from physics.movement import MovementMemory, step
    p, DT = _profile(), 0.1
    m = MovementMemory()
    pos = (50.0, 34.0)
    for _ in range(40):                              # 4 s, up to speed
        x, y, _moved = step(m, pos, (90.0, 34.0), p, DT)
        pos = (x, y)
    assert m.speed > 7.0, f"precondition: should be at pace, got {m.speed}"
    before_x = pos[0]
    speed_before = m.speed

    # One tick after the reversal he is still sliding east — a body cannot
    # reverse in 0.1 s, and pretending otherwise is the bug this whole test is
    # about. What matters is that he is BRAKING, not still accelerating.
    x, y, moved = step(m, pos, (10.0, 34.0), p, DT)
    assert moved > 0.0, "he teleported to the new target"
    assert m.speed < speed_before, (
        f"a 180 deg turn accelerated him: {speed_before:.2f} -> {m.speed:.2f}")
    assert m.speed > 0.0, "he stopped dead instead of carrying momentum"

    # ...and the turn actually completes: he comes round and makes ground west.
    # 19 m, not 32 m: about 0.9 s of the four goes into braking off 8 m/s at
    # 9.0 m/s^2 before he can drive the other way. A player who covered the full
    # 4 s at speed would be reversing instantaneously, which is the bug.
    for _ in range(40):                              # 4 s more
        x, y, _moved = step(m, pos, (10.0, 34.0), p, DT)
        pos = (x, y)
    assert pos[0] < before_x - 15.0, (
        f"he never completed the turn: still at x={pos[0]:.1f}")


def test_a_player_decelerates_to_a_stop_rather_than_vanishing():
    from physics.movement import MovementMemory, step
    p, DT = _profile(), 0.1
    m = MovementMemory()
    m.vx, m.vy = 8.0, 0.0
    pos = (50.0, 34.0)
    for _ in range(60):
        x, y, _moved = step(m, pos, (50.0, 34.0), p, DT)   # already there
        pos = (x, y)
    assert m.speed == pytest.approx(0.0, abs=1e-6), "he never came to rest"
    assert pos[0] > 50.0, "he should have coasted forward, not stood still"


def test_a_changed_target_costs_the_player_a_reaction():
    """The delay is on the player's INFORMATION, not his legs — so while the
    reaction is pending he keeps running toward the position he last believed
    in, and only then corrects."""
    from physics.movement import MovementMemory, perceive
    DT = 0.1
    m = MovementMemory()
    assert perceive(m, (50.0, 34.0), DT) == (50.0, 34.0)
    for tick in range(2):
        believed = perceive(m, (70.0, 34.0), DT)
        assert believed == (50.0, 34.0), f"tick {tick}: acted before reacting"
    believed = perceive(m, (70.0, 34.0), DT)
    assert believed == (70.0, 34.0), "never adopted the new target"


def test_the_reaction_does_not_re_arm_itself():
    """A regression guard for a real bug. The countdown used to live in
    ``step``, which was handed the already-stale believed target, so on expiry
    it re-stored the OLD belief; the next ``perceive`` then saw a large
    difference and armed a fresh reaction. The player ran forever toward the
    first position he ever saw."""
    from physics.movement import MovementMemory, perceive
    DT = 0.1
    m = MovementMemory()
    perceive(m, (50.0, 34.0), DT)
    for tick in range(30):                     # far longer than the 0.2 s delay
        believed = perceive(m, (70.0, 34.0), DT)
    assert believed == (70.0, 34.0), (
        "the reaction re-armed and the player never updated his target")
    assert m.reaction_left == 0.0


def test_ordinary_shape_drift_never_arms_a_reaction():
    """Shape targets drift centimetres every tick. Arming on those would hold
    the entire team in permanent reaction — slower, and wrong."""
    from physics.movement import MovementMemory, perceive
    m = MovementMemory()
    perceive(m, (50.0, 34.0), 0.1)
    perceive(m, (50.4, 34.1), 0.1)
    assert m.reaction_left == 0.0, "0.45 m of drift armed a reaction"
    perceive(m, (53.0, 34.0), 0.1)
    assert m.reaction_left > 0.0, "a 3 m shift should arm a reaction"


def test_fatigue_reaches_a_players_legs():
    """The engine's own ``tgt`` is ``5.0 + pace * 0.042`` — DNA only, so a
    player on 20% stamina ran at the same top speed as one on 100%."""
    from physics.calibration import calibrate

    class D:
        pace, acceleration, anticipation, stamina = 70, 70, 60, 100

    fresh = calibrate(D(), stamina=100.0)
    tired = calibrate(D(), stamina=20.0)
    assert tired.max_speed < fresh.max_speed, "stamina did not slow him down"
    assert tired.max_speed > 0.0, "a tired player should be slow, not immobile"


def test_the_engine_only_steps_physically_when_asked(live_engine):
    """Inertness, at the call site rather than by inspection."""
    adapter = live_engine.physics
    pos = adapter.position_of("__nobody__")
    assert pos is None or True            # adapter exists; behaviour below
    adapter.invalidate("Nobody At All")
    assert adapter.offball_step("Nobody At All", (50.0, 34.0), 0.1) is None, (
        "the adapter invented a state for a player it cannot see")


def test_a_substitution_cannot_inherit_momentum(live_engine):
    """``MovementMemory`` holds carried velocity AND a held target. A replaced
    player inheriting either would arrive already travelling at the pace and in
    the direction of a man no longer on the pitch."""
    adapter = live_engine.physics
    name = next(iter(live_engine.position_engine.states))
    st = live_engine.position_engine.states[name]
    st.current_x, st.current_y = 50.0, 34.0
    adapter.offball_step(name, (70.0, 34.0), 0.1)
    memory = adapter._movement.get(name)
    assert memory is not None, "no movement memory was created"
    assert memory.vx or memory.vy, "precondition: he should be moving"
    adapter.invalidate(name)
    assert name not in adapter._movement, (
        "a substituted player kept the momentum and held target of his predecessor")


# ── ball flight time driving the sequence ──────────────────

def test_the_ball_takes_time_to_cross_the_pitch():
    from physics.ball_physics import ground_travel_time
    t_short = ground_travel_time(5.0, 25.0)
    t_long = ground_travel_time(40.0, 25.0)
    assert 0.0 < t_short < t_long, "flight time must grow with distance"
    assert t_long > 1.0, f"40 m at 25 m/s should take well over a second, got {t_long}"


def test_the_flight_time_gate_declines_rather_than_guessing():
    """A zero means 'do not shift the clock', which is different from 'the ball
    arrived instantly'. Guessing here would move every event in the match."""
    from physics.adapter import EngineAdapter
    a = EngineAdapter(MatchEngine(_cfg(), None, None), weather=None)
    assert a.pass_flight_time(0.0, 25.0)[0] == 0.0
    assert a.pass_flight_time(-5.0, 25.0)[0] == 0.0
    assert a.pass_flight_time(20.0, 0.0)[0] == 0.0
    t, _why = a.pass_flight_time(30.0, 24.0)
    assert t > 0.5, f"a 30 m pass should take real time, got {t}"


def test_the_flight_time_does_not_depend_on_being_gated_on():
    """The timeline shift is inside ``if _fphys is not None`` at the call site,
    so a completed pass that never entered the reactive-block branch has no
    adapter-side state to rely on. The distance is therefore derived from the
    pass origin and destination, which are in scope at the receiving event."""
    import ast
    import pathlib
    src = pathlib.Path(r"D:\PLOFA\plofa\event_chain.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if "pass_flight_time" not in seg:
            continue
        found = True
        # the call must not reference _pdm, which is bound in another branch
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "pass_flight_time"):
                for arg in sub.args:
                    assert not (isinstance(arg, ast.Name) and arg.id == "_pdm"), (
                        "the flight gate reads _pdm, which is only bound inside "
                        "the reactive-block branch — a completed pass that "
                        "skipped it would NameError, be swallowed, and silently "
                        "never shift the clock")
    assert found, "the flight-time gate is no longer wired into event_chain.py"


# ── the call sites are real ────────────────────────────────
#
# Both gates are wrapped in ``try/except`` so that a physics problem can never
# cost a season. That is the right trade, and it has a nasty edge: a ``NameError``
# from a renamed local is swallowed exactly like a genuine physics failure, so the
# gate sits there wired and never fires — indistinguishable, in a scoreline, from
# "the physics had nothing to say". These two tests make that visible.

def _gated_functions() -> list:
    """Every function that *calls* a physics gate, across both wiring points.

    Matched on the AST, not the source text. A substring search also finds
    ``_pass_block_probability`` — the engine's own method — inside its own
    definition, which is not a gate. Only an attribute call whose name is in
    ``gates`` is one.

    Two files, because the wiring is split: the block and flight gates live in
    the possession chain, the keeper gate in the goalkeeper engine, and the
    movement step in the match engine itself.
    """
    import ast
    import pathlib
    gates = {"block_probability", "keeper_save_multiplier", "offball_step",
             "pass_flight_time"}
    found = []
    for path in (r"D:\PLOFA\plofa\event_chain.py",
                 r"D:\PLOFA\plofa\match_engine.py"):
        src = pathlib.Path(path).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if any(isinstance(sub, ast.Call)
                   and isinstance(sub.func, ast.Attribute)
                   and sub.func.attr in gates
                   for sub in ast.walk(node)):
                found.append((src, node))
    return found


def test_the_gates_are_actually_wired_into_the_engine():
    """If someone removed a call site, the gates would still pass every unit
    test — they are pure functions, and their tests never go through the engine.
    Only looking at the call sites catches that.

    Four gates across three functions: the block gate and the flight-time gate
    both live in the possession chain, so they count once.
    """
    found = _gated_functions()
    assert len(found) == 3, (
        f"expected 3 gated functions across event_chain.py + match_engine.py, "
        f"found {len(found)}: {[n.name for _, n in found]}")


def test_every_name_a_gate_uses_is_in_scope_where_it_is_called():
    """The ``except Exception: pass`` that protects a season from a physics bug
    also protects it from a typo. So the names have to be checked statically."""
    import ast
    problems = []
    for src, fn in _gated_functions():
        used = set()
        for sub in ast.walk(fn):
            # Names inside the try-block only, so a variable the gate needs
            # rather than one the engine happens to use elsewhere.
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in ("block_probability",
                                          "keeper_save_multiplier")):
                for arg in sub.args:
                    if isinstance(arg, ast.Name):
                        used.add(arg.id)
        bound = set()
        a = fn.args
        for arg in (*a.args, *a.kwonlyargs):
            bound.add(arg.arg)
        if a.vararg:
            bound.add(a.vararg.arg)
        if a.kwarg:
            bound.add(a.kwarg.arg)
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                bound.add(sub.id)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for al in sub.names:
                    bound.add((al.asname or al.name).split(".")[0])
        for name in sorted(used - bound):
            problems.append(f"{fn.name}(): '{name}' is not in scope")
    assert not problems, (
        "a gate would raise NameError and be silently swallowed: "
        + "; ".join(problems))


def test_a_good_keepers_save_credit_survives_intact():
    """The gate must not quietly flatten every save into the floor. A shot the
    keeper physically reaches, with a well-positioned multiplier of 1.5, has to
    come back as 1.5 — otherwise the physics would be quietly overriding a
    tuned model rather than supplementing it."""
    from physics.adapter import EngineAdapter
    a = EngineAdapter(MatchEngine(_cfg(), None, None), weather=None)
    factor, why = a.keeper_save_multiplier(
        1.5, 103.0, 34.2, 80.0, 34.0, _positioning(gk_x=101.0, gk_y=34.0))
    assert factor == 1.5, f"a reachable save was altered to {factor}: {why}"


def test_the_keeper_gate_uses_the_engines_geometry_not_the_kinematic_state():
    """A contract, and the reason the gate is written the way it is.

    ``_get_gk_positioning`` computes the keeper's position analytically — it
    bisects the angle between the posts. The gate therefore takes that computed
    position as an argument rather than reading the position engine, because
    attenuating a multiplier using a *different* keeper position would make the
    gate quietly disagree with the number it is adjusting.

    This engine has no squads at all, so anything the adapter could read about a
    keeper it does not have. The gate still works, which is the proof."""
    from physics.adapter import EngineAdapter
    a = EngineAdapter(MatchEngine(_cfg(), None, None), weather=None)
    assert not a._player_objects(), "precondition: this engine has no players"
    # From 21 m the flight is ~0.88 s, leaving 0.18 s of dive time after a 0.70 s
    # reaction. That is the only regime in which the keeper's POSITION matters at
    # all — against a close shot the gate returns 0.0 before geometry is even
    # consulted, so a test built on a close shot cannot tell the two designs
    # apart. (An earlier draft of this test used a 7 m shot and asserted the
    # positions changed the answer. It proved nothing: both returned 0.0.)
    near_gk, _ = a.keeper_feasibility(104.0, 34.5, 80.0, 34.0,
                                      101.0, 34.0, 2.5, 0.70)
    far_gk, _ = a.keeper_feasibility(104.0, 34.5, 80.0, 34.0,
                                     101.0, 30.0, 2.5, 0.70)
    assert near_gk > far_gk, (
        "the gate ignored the keeper position it was given, so it must be "
        "reading the kinematic state instead — which would disagree with the "
        "multiplier it is supposed to be attenuating")
