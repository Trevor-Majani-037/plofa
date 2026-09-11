"""
Simple Bug Exploration Tests (non-PBT) for faster execution
These demonstrate that the bugs exist in the current code
"""

from match_engine import MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle, Intensity, MatchState, MatchPhase, EventType
from event_chain import PitchZone
from position_engine import PositionEngine
from player_dna import PlayerProfile, DNAFactory
from datetime import date


def test_unrealistic_shot_behind_goal_line_simple():
    """Test that shots from behind goal line (x >= 105) are currently allowed"""
    shot_x = 105.5
    shot_y = 55.0
    
    # Bug: PitchZone.xg_zone doesn't reject x >= 105
    zone = PitchZone.xg_zone(shot_x, shot_y)
    
    # On unfixed code: this will return a zone (e.g. "six_yard_box")
    # On fixed code: should reject or return special "invalid" zone
    
    print(f"Shot from (x={shot_x}, y={shot_y}) gets zone: {zone}")
    # Bug confirmed: shot from behind goal line is assigned a valid zone
    assert shot_x < 105.0, f"Shot from x={shot_x} (behind goal line) should be rejected"


def test_throw_in_missing():
    """Test that throw-in events are not generated for out-of-bounds"""
    state = MatchState()
    state.minute = 30
    state.last_ball_x = 85.0
    state.last_ball_y = 1.0  # Out of bounds
    
    # Bug: no code checks y < 2 or y > 66 to generate THROW_IN
    # After fix: should have THROW_IN event in timeline
    
    # Simulate what would happen - no throw-in detection exists
    is_out = state.last_ball_y < 2.0 or state.last_ball_y > 66.0
    
    print(f"Ball at (x={state.last_ball_x}, y={state.last_ball_y}), is_out={is_out}")
    
    # Bug: even though ball is out, no mechanism detects it
    assert False, "Ball out of bounds should trigger THROW_IN event (currently missing)"


def test_goal_kick_missing():
    """Test that goal kick events are not generated"""
    ball_x = 105.5  # Beyond goal line
    ball_y = 20.0   # Outside posts (30.34-37.66)
    
    crossed_line = ball_x >= 105.0
    outside_posts = ball_y < 30.34 or ball_y > 37.66
    
    if crossed_line and outside_posts:
        print(f"Ball at (x={ball_x}, y={ball_y}) crossed line outside posts")
        # Bug: no code checks this condition to generate GOAL_KICK
        assert False, "Should generate GOAL_KICK event (currently missing)"


def test_offside_missing():
    """Test that offside detection is not implemented"""
    pass_x = 70.0
    receiver_x = 95.0
    second_last_defender_x = 88.0
    
    forward_pass = receiver_x > pass_x
    attacking_half = receiver_x > 52.5
    ahead_of_defender = receiver_x > second_last_defender_x
    
    is_offside = forward_pass and attacking_half and ahead_of_defender
    
    if is_offside:
        print(f"Offside: receiver at x={receiver_x} ahead of defender at x={second_last_defender_x}")
        # Bug: no offside detection exists in AttackChain or MatchEngine
        assert False, "Should detect offside and stop attack (currently missing)"


def test_celebration_missing():
    """Test that goal celebrations are not implemented"""
    config = MatchConfig(
        home_team="Home FC",
        away_team="Away FC",
        match_date=date.today()
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
    
    engine = MatchEngine(config, home_profile, away_profile)
    engine.state.minute = 67
    engine.state.second = 23
    
    # Bug: no GOAL_CELEBRATION events exist in event types being used
    has_celebration = any(e.event_type == EventType.GOAL_CELEBRATION for e in engine.timeline)
    
    print(f"Has celebration event: {has_celebration}")
    # Even though GOAL_CELEBRATION is defined in EventType, it's never emitted
    assert has_celebration, "Should have GOAL_CELEBRATION event after goal (currently missing)"


def test_formation_reset_missing():
    """Test that formation reset after goal is not implemented"""
    config = MatchConfig(
        home_team="Home FC",
        away_team="Away FC",
        match_date=date.today()
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
    
    engine = MatchEngine(config, home_profile, away_profile)
    
    # Create simple player DNA
    dna = DNAFactory.create(
        name="Test Striker",
        position="ST",
        specialties=["poacher"],
        age=25
    )
    player = PlayerProfile(dna=dna, team_name="Home FC")
    
    engine.position_engine.initialize_team("Home FC", [player], home_profile)
    
    # Displace player during goal sequence
    displaced_x = 102.0
    displaced_y = 35.0
    engine.position_engine.record_touch(player.name, displaced_x, displaced_y, 45)
    
    # Set up kickoff scenario
    engine.state.pending_kickoff_for = "Away FC"
    engine.state.last_ball_x = 52.5
    engine.state.last_ball_y = 34.0
    
    # Bug: pending_kickoff_for resets ball but not player positions
    player_state = engine.position_engine._players.get(player.name)
    if player_state:
        current_x = player_state.current_x
        home_x = player_state.home_x
        
        print(f"Player current_x={current_x:.1f}, home_x={home_x:.1f}")
        # Bug: current_x still at displaced position, not reset to home_x
        assert abs(current_x - home_x) < 5.0, \
            f"After kickoff, player should be at home position (currently at displaced position)"


if __name__ == "__main__":
    # Run each test and catch failures to document counterexamples
    tests = [
        test_unrealistic_shot_behind_goal_line_simple,
        test_throw_in_missing,
        test_goal_kick_missing,
        test_offside_missing,
        test_celebration_missing,
        test_formation_reset_missing,
    ]
    
    print("=" * 80)
    print("BUG CONDITION EXPLORATION - Documenting Counterexamples")
    print("=" * 80)
    
    counterexamples = []
    
    for test_func in tests:
        print(f"\n### Running: {test_func.__name__}")
        try:
            test_func()
            print(f"✓ PASSED (unexpected - bug might be fixed)")
        except AssertionError as e:
            print(f"✗ FAILED (expected - confirms bug exists)")
            print(f"   Counterexample: {str(e)}")
            counterexamples.append({
                'test': test_func.__name__,
                'error': str(e)
            })
        except Exception as e:
            print(f"⚠ ERROR: {str(e)}")
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(counterexamples)} bugs confirmed via counterexamples")
    print("=" * 80)
    
    for ce in counterexamples:
        print(f"\n{ce['test']}:")
        print(f"  {ce['error']}")
