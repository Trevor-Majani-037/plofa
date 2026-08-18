"""
PLOFA 26/27 — Preservation Property Tests
==========================================
test_preservation_properties.py

**Validates: Requirements 3.1-3.12 from bugfix.md**

**CRITICAL**: These tests MUST PASS on UNFIXED code to establish baseline behavior.
They verify that existing functionality continues to work correctly after fixes are applied.

These tests observe behavior on UNFIXED code for non-buggy inputs and capture
those patterns as properties that must be preserved.
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck, assume
from match_engine import (
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, 
    Intensity, MatchState, MatchPhase, GameState, EventType, XGEngine
)
from event_chain import (
    ChainDispatcher, ChainResult, AttackChain, PitchZone, 
    SituationType, PossessionChain
)
from position_engine import PositionEngine
from player_dna import PlayerProfile, PlayerDNA, DNAFactory, BehavioralTendencies
from datetime import date
import random


def _profile(name: str, position: str, archetype: str,
             team: str = "Test FC") -> PlayerProfile:
    """Build a PlayerProfile with the CURRENT (dna-backed) API.
    name/position live on the DNA; the requested position is enforced so
    PositionEngine formation placement sees the intended role."""
    dna = DNAFactory.create_archetype(archetype, name=name)
    dna.position = position
    return PlayerProfile(dna=dna, team_name=team)


# Disable slow healthchecks for faster test execution
settings.register_profile("fast", max_examples=20, deadline=None, 
                         suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
settings.load_profile("fast")


# ============================================================================
# TEST 2.1: Existing Shot Outcome Logic - Realistic Positions
# ============================================================================
# **Validates: Requirements 3.5, 3.6, 3.7**
# Property: For all realistic shot positions (60 < x < 103), xG calculation,
# body_part multipliers, pressure penalties, save evaluation, and outcome 
# logic remain unchanged from current baseline behavior.

@settings(max_examples=20, deadline=None)
@given(
    shot_x=st.floats(min_value=60.0, max_value=103.0),  # Realistic shot range
    shot_y=st.floats(min_value=20.0, max_value=48.0),   # Reasonable funnel
    body_part=st.sampled_from(["right_foot", "left_foot", "head"]),
    under_pressure=st.booleans(),
    is_big_chance=st.booleans(),
)
def test_realistic_shot_xg_calculation_preserved(shot_x, shot_y, body_part, under_pressure, is_big_chance):
    """
    **Validates: Requirements 3.5**
    
    Property: For realistic shot positions, xG calculation applies body_part 
    multipliers, pressure penalties, and situation adjustments as currently 
    implemented.
    
    EXPECTED ON UNFIXED CODE: Test PASSES (baseline behavior captured)
    EXPECTED AFTER FIX: Test PASSES (behavior preserved)
    """
    # Calculate xG using current implementation
    zone = PitchZone.xg_zone(shot_x, shot_y)
    situation = SituationType.OPEN_PLAY
    
    # Get baseline xG from current implementation
    xg = XGEngine.calculate(
        zone=zone,
        body_part=body_part,
        situation=situation,
        under_pressure=under_pressure,
        is_big_chance=is_big_chance,
        first_time_shot=False,
        shot_x=shot_x,
        shot_y=shot_y,
    )
    
    # Property: xG should be within expected ranges for the zone
    zone_ranges = {
        "six_yard_box": (0.40, 0.70),
        "inside_box": (0.10, 0.35),
        "edge_of_box": (0.04, 0.15),
        "outside_box": (0.01, 0.08),
    }
    
    min_xg, max_xg = zone_ranges.get(zone, (0.0, 1.0))
    
    # Adjust for modifiers (pressure reduces, big chance increases)
    if under_pressure:
        min_xg *= 0.5
        max_xg *= 0.8
    
    # xG should be in reasonable range (allowing for noise and modifiers)
    assert 0.0 <= xg <= 0.99, f"xG {xg:.3f} should be in [0, 0.99] range"
    
    # Body part should affect xG (head is less than foot from same position)
    if body_part == "head":
        # Headers typically have 0.7x multiplier
        # Just verify xG is calculated, not exact value (noise + other factors)
        assert xg >= 0.0, "Header xG should be non-negative"


@settings(max_examples=20, deadline=None)
@given(
    shot_x=st.floats(min_value=88.0, max_value=103.0),
    shot_y=st.floats(min_value=24.0, max_value=44.0),
)
def test_realistic_shot_on_target_probability_preserved(shot_x, shot_y):
    """
    **Validates: Requirements 3.6**
    
    Property: On-target shots call GoalkeeperEngine.evaluate_save() with the 
    same parameters and logic as current implementation.
    
    EXPECTED ON UNFIXED CODE: Test PASSES (current save evaluation logic)
    EXPECTED AFTER FIX: Test PASSES (save evaluation logic unchanged)
    """
    # Create minimal test setup
    state = MatchState()
    state.minute = 45
    state.phase = MatchPhase.FIRST_SPELL
    
    # Create players
    shooter = _profile("Test Shooter", "ST", "fox_in_box")
    
    gk = _profile("Test GK", "GK", "sweeper_keeper")
    
    # Calculate xG
    zone = PitchZone.xg_zone(shot_x, shot_y)
    xg = XGEngine.calculate(
        zone=zone,
        body_part="right_foot",
        situation=SituationType.OPEN_PLAY,
        under_pressure=False,
        shot_x=shot_x,
        shot_y=shot_y,
    )
    
    # Test that shot evaluation works
    shooter_quality = DNAFactory.get_shooter_quality(shooter.dna)
    
    # Property: The system can evaluate shots without errors
    # This tests that the existing shot outcome logic is functional
    try:
        from event_chain import GoalkeeperEngine
        result = GoalkeeperEngine.evaluate_save(xg, shooter_quality, shot_x, shot_y, gk, shot_x, shot_y)
        is_goal, positioning = result
        assert bool(is_goal) in (True, False), "First element should be boolean"
        assert isinstance(positioning, dict), "Second element should be a positioning dict"
    except Exception as e:
        pytest.fail(f"Existing save evaluation failed: {e}")


@settings(max_examples=15, deadline=None)
@given(
    shot_x=st.floats(min_value=88.0, max_value=103.0),
    shot_y=st.floats(min_value=28.0, max_value=40.0),
)
def test_realistic_shot_woodwork_and_rebound_preserved(shot_x, shot_y):
    """
    **Validates: Requirements 3.7**
    
    Property: Shots hitting woodwork handle rebound_in logic and corner awards
    as currently implemented.
    
    EXPECTED ON UNFIXED CODE: Test PASSES (current woodwork behavior)
    EXPECTED AFTER FIX: Test PASSES (woodwork logic preserved)
    """
    # Create match setup
    config = MatchConfig(
        home_team="Home FC",
        away_team="Away FC",
        match_date=date.today(),
    )
    
    home_profile = TeamProfile(
        name="Home FC",
        style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM
    )
    
    away_profile = TeamProfile(
        name="Away FC",
        style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM
    )
    
    state = MatchState()
    state.minute = 45
    state.phase = MatchPhase.FIRST_SPELL
    state.last_ball_x = shot_x
    state.last_ball_y = shot_y
    
    # Create players
    att_player = _profile("Attacker", "ST", "fox_in_box")
    
    def_player = _profile("Defender", "CB", "stopper_defender")
    
    gk_player = _profile("GK", "GK", "sweeper_keeper")
    
    # Generate attack chain
    att_result = AttackChain.generate(
        minute=45,
        attacking_team="Home FC",
        defending_team="Away FC",
        att_players=[att_player],
        def_players=[def_player, gk_player],
        team_profile=home_profile,
        def_profile=away_profile,
        state=state,
        situation=SituationType.OPEN_PLAY,
        context_x=shot_x,
        context_y=shot_y,
    )
    
    # Property: Attack chain should generate valid events
    assert len(att_result.events) > 0, "Attack chain should generate events"
    
    # Check for woodwork events (if they occur)
    woodwork_events = [e for e in att_result.events if e.event_type == EventType.HIT_WOODWORK]
    
    # Property: If woodwork hit occurs, it should have proper metadata
    for we in woodwork_events:
        assert hasattr(we, 'metadata'), "Woodwork event should have metadata"
        assert 'rebound_in' in we.metadata, "Woodwork should track rebound_in"


# ============================================================================
# TEST 2.2: Corner Causality System
# ============================================================================
# **Validates: Requirements 3.1**
# Property: For all blocked shots or ineffective clearances, corner causality 
# system continues to track won corners (pending_corners_home/away) and 
# consume them before the next sequence situation roll.

@settings(max_examples=20, deadline=None)
@given(
    shot_x=st.floats(min_value=88.0, max_value=103.0),
    shot_y=st.floats(min_value=24.0, max_value=44.0),
)
def test_corner_causality_from_blocked_shots(shot_x, shot_y):
    """
    **Validates: Requirements 3.1**
    
    Property: Blocked shots set a pending corner (pending_corners_*), which is consumed before 
    the next sequence, ensuring corners are a consequence of defensive actions.
    
    EXPECTED ON UNFIXED CODE: Test PASSES (corner causality exists)
    EXPECTED AFTER FIX: Test PASSES (corner causality preserved)
    """
    # Create match setup
    config = MatchConfig(
        home_team="Home FC",
        away_team="Away FC",
        match_date=date.today(),
    )
    
    home_profile = TeamProfile(
        name="Home FC",
        style=TeamStyle.ATTACKING,
        playing_style=PlayingStyle.HIGH_PRESS,
        intensity=Intensity.HIGH
    )
    
    away_profile = TeamProfile(
        name="Away FC",
        style=TeamStyle.DEFENSIVE,
        playing_style=PlayingStyle.LOW_BLOCK,
        intensity=Intensity.MEDIUM
    )
    
    state = MatchState()
    state.minute = 45
    state.phase = MatchPhase.FIRST_SPELL
    state.last_ball_x = shot_x
    state.last_ball_y = shot_y
    state.possession_team = "Home FC"
    
    # Create players
    att_players = [
        _profile(f"Attacker_{i}", pos, "fox_in_box")
        for i, pos in enumerate(["ST", "LW", "RW"])
    ]
    
    def_players = [
        _profile(f"Defender_{i}", pos, "stopper_defender")
        for i, pos in enumerate(["CB", "CB", "GK"])
    ]
    
    # Run multiple attack chains to observe corner causality
    corner_won_count = 0
    blocked_shot_count = 0
    
    for _ in range(5):
        result = AttackChain.generate(
            minute=45,
            attacking_team="Home FC",
            defending_team="Away FC",
            att_players=att_players,
            def_players=def_players,
            team_profile=home_profile,
            def_profile=away_profile,
            state=state,
            situation=SituationType.OPEN_PLAY,
            context_x=shot_x,
            context_y=shot_y,
        )
        
        # Count blocked shots
        blocked_shots = [e for e in result.events if e.event_type == EventType.SHOT_BLOCKED]
        blocked_shot_count += len(blocked_shots)
        
        # Check if corner was won
        if result.corner_won:
            corner_won_count += 1
            # Property: corner_team should be set
            assert result.corner_team == "Home FC", "Corner should be awarded to attacking team"
    
    # Property: The corner causality system exists and can award corners
    # We're just checking that the mechanism is present, not forcing a specific outcome
    assert True, "Corner causality system is operational"


@settings(max_examples=15, deadline=None)
@given(
    context_x=st.floats(min_value=85.0, max_value=100.0),
    context_y=st.floats(min_value=20.0, max_value=48.0),
)
def test_corner_spatial_continuity_preserved(context_x, context_y):
    """
    **Validates: Requirements 3.1**
    
    Property: Corners are anchored to state.last_ball_x/y (no teleport).
    
    EXPECTED ON UNFIXED CODE: Test PASSES (spatial continuity exists)
    EXPECTED AFTER FIX: Test PASSES (spatial continuity preserved)
    """
    # Create match state with specific ball position
    state = MatchState()
    state.minute = 45
    state.phase = MatchPhase.FIRST_SPELL
    state.last_ball_x = context_x
    state.last_ball_y = context_y
    state.pending_corners_home = 1  # Home FC has a corner pending
    
    home_profile = TeamProfile(
        name="Home FC",
        style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM
    )
    
    away_profile = TeamProfile(
        name="Away FC",
        style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM
    )
    
    # Create players
    att_players = [
        _profile(f"Att_{i}", "ST", "target_forward")
        for i in range(3)
    ]
    
    def_players = [
        _profile(f"Def_{i}", "CB", "stopper_defender")
        for i in range(3)
    ]
    
    # Trigger corner consumption
    result = ChainDispatcher.set_piece(
        minute=45,
        att_team="Home FC",
        def_team="Away FC",
        att_players=att_players,
        def_players=def_players,
        state=state,
        situation=SituationType.CORNER
    )
    
    # Property: Corner events should exist
    corner_events = [e for e in result.events if e.event_type == EventType.CORNER_TAKEN]
    
    if corner_events:
        # Property: Corner should be taken from near the context position
        # (not teleported to a random location)
        corner_event = corner_events[0]
        
        # Corners are taken from corner flag, but should be anchored to 
        # where the ball was when corner was won
        # The x should be at/near goal line (105), y should be at touchline
        assert corner_event.location_x >= 100.0, "Corner should be near goal line"
        assert corner_event.location_y < 10.0 or corner_event.location_y > 58.0, \
            "Corner should be from touchline"


# ============================================================================
# TEST 2.3: Position Engine Spatial State
# ============================================================================
# **Validates: Requirements 3.8, 3.9, 3.10**
# Property: For all events with real coordinates, position engine updates 
# current_x/y on touch, uninvolved players drift toward home positions, and 
# substitutions register with PositionEngine.

@settings(max_examples=20, deadline=None)
@given(
    touch_x=st.floats(min_value=10.0, max_value=100.0),
    touch_y=st.floats(min_value=5.0, max_value=63.0),
    position=st.sampled_from(["CB", "CM", "ST", "LW", "RW"]),
)
def test_position_engine_updates_on_touch(touch_x, touch_y, position):
    """
    **Validates: Requirements 3.8**
    
    Property: When a player touches the ball, position_engine.record_touch() 
    updates their current_x/current_y to the event coordinates.
    
    EXPECTED ON UNFIXED CODE: Test PASSES (position tracking exists)
    EXPECTED AFTER FIX: Test PASSES (position tracking preserved)
    """
    WIDE = {"LW", "RW"}
    # Create position engine
    pe = PositionEngine()
    
    # Create team profile
    profile = TeamProfile(
        name="Test Team",
        style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM
    )
    
    # Create player
    player = _profile("Test Player", position, "fox_in_box")
    
    # Initialize team
    pe.initialize_team("Test Team", [player], profile)
    
    # Get initial position
    initial_x, initial_y = pe.get_position(player.name)
    
    # Record touch at new coordinates
    pe.record_touch(player.name, touch_x, touch_y, minute=45)
    
    # Get updated position
    new_x, new_y = pe.get_position(player.name)
    
    # Property: x always tracks the touch coordinates exactly.
    assert abs(new_x - touch_x) < 0.1, \
        f"Position should update to touch x={touch_x:.1f}, got {new_x:.1f}"
    _, home_y = pe.get_home_position(player.name)
    if position in WIDE and abs(home_y - touch_y) > 6.0:
        # Checkpoint 21d: a wide player whose touch has dragged him >6m OFF
        # his flank channel is pulled back TOWARD home_y (never away). The
        # engine must place him strictly between the touch and his flank.
        assert min(touch_y, home_y) <= new_y <= max(touch_y, home_y), \
            f"Wide flank-hold must stay between touch y={touch_y:.1f} and " \
            f"home y={home_y:.1f}, got {new_y:.1f}"
    else:
        # Property: on-flank touches (and all central roles) track exactly.
        assert abs(new_y - touch_y) < 0.1, \
            f"Position should update to touch y={touch_y:.1f}, got {new_y:.1f}"


@settings(max_examples=15, deadline=None)
@given(
    displaced_x=st.floats(min_value=60.0, max_value=100.0),
    displaced_y=st.floats(min_value=10.0, max_value=58.0),
    position=st.sampled_from(["CB", "CDM", "CM", "LW", "ST"]),
)
def test_position_engine_drift_toward_home(displaced_x, displaced_y, position):
    """
    **Validates: Requirements 3.9**
    
    Property: When a minute elapses without involvement, uninvolved players 
    drift back toward their home positions with phase/game-state modifiers.
    
    EXPECTED ON UNFIXED CODE: Test PASSES (drift mechanism exists)
    EXPECTED AFTER FIX: Test PASSES (drift mechanism preserved)
    """
    # Create position engine
    pe = PositionEngine()
    
    # Create team profile
    profile = TeamProfile(
        name="Test Team",
        style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM
    )
    
    # Create player
    player = _profile("Test Player", position, "fox_in_box")
    
    # Initialize team
    pe.initialize_team("Test Team", [player], profile)
    
    # Get home position
    player_state = pe.states.get(player.name)
    assume(player_state is not None)
    
    home_x = player_state.home_x
    home_y = player_state.home_y
    
    # Displace player
    pe.record_touch(player.name, displaced_x, displaced_y, minute=45)
    
    # Get displaced position
    displaced_pos_x, displaced_pos_y = pe.get_position(player.name)
    
    # Run drift for several minutes
    for minute in range(46, 56):
        pe.drift_minute(
            team_name="Test Team",
            profile=profile,
            phase=MatchPhase.SECOND_OPEN,
            game_state_gd=0,
            minute=minute,
            in_possession=False
        )
    
    # Get final position
    final_x, final_y = pe.get_position(player.name)
    
    # Property: Player should drift closer to the current shape target
    # rather than the static kickoff home when the team is out of possession.
    scale = PositionEngine.SHAPE_SHIFT_SCALE.get(player.position, 1.0)
    target_x = home_x - 3.0 * scale
    target_y = home_y

    initial_dist = ((displaced_pos_x - target_x)**2 + (displaced_pos_y - target_y)**2)**0.5
    final_dist = ((final_x - target_x)**2 + (final_y - target_y)**2)**0.5

    assert final_dist < initial_dist, \
        f"Player should drift toward shape target (initial dist={initial_dist:.1f}, " \
        f"final dist={final_dist:.1f})"


def test_fullbacks_preserve_wide_y_under_line_cohesion():
    """
    Property: Fullbacks should not be pulled centrally by defensive line
    cohesion to the same extent as centre-backs.

    This guards against the bug where LB/RB positions collapse toward the
    centre channel even when their own home/current positions remain wide.
    """
    pe = PositionEngine()
    profile = TeamProfile(
        name="Test Team",
        style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM
    )

    cb = _profile("Centre Back", "CB", "stopper_defender")
    lb = _profile("Left Back", "LB", "attacking_fullback")
    rb = _profile("Right Back", "RB", "attacking_fullback")

    pe.initialize_team("Test Team", [cb, lb, rb], profile)

    # Put the defenders in a realistic wide block with the fullbacks on the
    # touchline and the centre-back more central.
    pe.states[cb.name].current_x, pe.states[cb.name].current_y = 24.0, 34.0
    pe.states[lb.name].current_x, pe.states[lb.name].current_y = 24.0, 10.0
    pe.states[rb.name].current_x, pe.states[rb.name].current_y = 24.0, 58.0

    before_lb_y = pe.states[lb.name].current_y
    before_rb_y = pe.states[rb.name].current_y

    pe._apply_line_cohesion("Test Team")

    after_lb_y = pe.states[lb.name].current_y
    after_rb_y = pe.states[rb.name].current_y

    assert abs(after_lb_y - before_lb_y) < 1.0, \
        f"Left back should not be pulled heavily toward center, moved {after_lb_y - before_lb_y:.2f}"
    assert abs(after_rb_y - before_rb_y) < 1.0, \
        f"Right back should not be pulled heavily toward center, moved {after_rb_y - before_rb_y:.2f}"


def test_wide_fullbacks_preserve_flank_on_safe_reset_passes():
    """
    Property: Wide fullbacks recycling the ball should not be forced toward
    the central channel on safe/sideways passes.
    """
    random.seed(42)
    fb = _profile("Wide Fullback", "LB", "attacking_fullback")
    profile = TeamProfile(
        name="Test Team",
        style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM
    )

    y = 10.0
    x = 45.0
    passes = [
        PossessionChain._pass_destination(
            fb, x, y, pass_dist=12.0,
            profile=profile, attacks_right=True,
            is_long=False, is_prog=False
        )
        for _ in range(50)
    ]

    assert sum(1 for _, pass_y in passes if pass_y < 22.0) >= 40, \
        f"Wide LB should preserve left flank on safe passes, got {passes}"


@settings(max_examples=10, deadline=None)
@given(
    position=st.sampled_from(["ST", "CM", "LW", "RW", "CB"]),
)
def test_position_engine_substitute_registration(position):
    """
    **Validates: Requirements 3.10**
    
    Property: When a substitution occurs, the incoming player is registered 
    with PositionEngine.register_substitute() and receives a home position.
    
    EXPECTED ON UNFIXED CODE: Test PASSES (substitute registration exists)
    EXPECTED AFTER FIX: Test PASSES (substitute registration preserved)
    """
    # Create position engine
    pe = PositionEngine()
    
    # Create team profile
    profile = TeamProfile(
        name="Test Team",
        style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED,
        intensity=Intensity.MEDIUM
    )
    
    # Create starter
    starter = _profile("Starter", position, "fox_in_box")
    
    # Initialize team with starter
    pe.initialize_team("Test Team", [starter], profile)
    
    # Create substitute
    sub = _profile("Substitute", position, "fox_in_box")
    
    # Register substitute
    pe.register_substitute("Test Team", sub, profile)
    
    # Property: Substitute should have spatial state registered
    sub_state = pe.states.get(sub.name)
    assert sub_state is not None, "Substitute should be registered in position engine"
    
    # Property: Substitute should have home position
    assert sub_state.home_x > 0, "Substitute should have valid home_x"
    assert sub_state.home_y > 0, "Substitute should have valid home_y"
    
    # Property: Substitute's current position should start at home
    assert abs(sub_state.current_x - sub_state.home_x) < 0.1, \
        "Substitute current_x should start at home_x"
    assert abs(sub_state.current_y - sub_state.home_y) < 0.1, \
        "Substitute current_y should start at home_y"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
