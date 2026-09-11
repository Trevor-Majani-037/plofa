"""
PLOFA 26/27 — GOAL CELEBRATION, INJURY TIMELINE & TRAINING SYSTEM TESTS
=======================================================================
test_checkpoint7_subsystems.py

Covers the Checkpoint-7 polish subsystems layered onto the match engine:

    Goal celebration
      P1  A scored goal emits a GOAL_CELEBRATION event with duration metadata.
      P2  The celebration advances the match clock (10-30s) and the event
          carries that exact duration.
      P3  Celebration time lands in dead time, not the goal chain's play.

    Injury timeline
      P4  An injury-forced substitution emits an INJURY event before the
          SUBSTITUTION event.

    Training system
      P5  run_week produces one record per player and commits confidence /
          fatigue deltas into SeasonState.
      P6  Young, professional players develop attributes; the deltas are
          bounded (no overnight +10s).
      P7  A full match still runs end-to-end with the new events enabled
          (integration smoke test).

Running:  python -m pytest tests/test_checkpoint7_subsystems.py -v
"""
import random
from datetime import date

from match_engine import (
    MatchEngine, MatchConfig, TeamProfile, TeamStyle, PlayingStyle,
    Intensity, EventType,
)
from player_dna import DNAFactory
from player_personality import PersonalityFactory

from training_system import TrainingSystem, TrainingFocus

from season_manager import SeasonState


def _team_profile(name: str) -> TeamProfile:
    return TeamProfile(
        name=name, style=TeamStyle.BALANCED,
        playing_style=PlayingStyle.MIXED, intensity=Intensity.MEDIUM,
    )


def _make_engine(seed: int = 42) -> MatchEngine:
    random.seed(seed)
    config = MatchConfig(home_team="Home FC", away_team="Away FC",
                         match_date=date.today())
    return MatchEngine(config, _team_profile("Home FC"), _team_profile("Away FC"))


def _make_player(name: str, position: str, age: int = 24):
    dna = DNAFactory.create(name=name, position=position,
                            specialties=["workhorse"], age=age)
    prof = PlayerProfile(dna=dna, team_name="T")
    prof.personality = PersonalityFactory.create(["workhorse"])
    return prof


from player_dna import PlayerProfile


# ─────────────────────────────────────────────
# P1-P3 — GOAL CELEBRATION
# ─────────────────────────────────────────────

def _force_goal_via_absorb(engine: MatchEngine, minute: int = 60):
    """Drive an open-play-like goal through _absorb_chain without relying on
    the full stochastic match loop."""
    from event_chain import ChainResult
    from match_engine import GameState, MatchPhase
    result = ChainResult()
    result.goal_scored = True
    result.goal_team = "Home FC"
    result.goal_scorer = "Kofi"
    result.events.append(
        MatchEvent(minute=minute, second=0, event_type=EventType.GOAL,
                   team="Home FC", player="Kofi",
                   phase=MatchPhase.PEAK_INTENSITY, game_state=GameState.LEVEL)
    )
    clock_before = engine.state.match_clock_s
    engine._absorb_chain(result, minute)
    return clock_before


from match_engine import MatchEvent, GameState, MatchPhase


def test_goal_emits_celebration_event_p1():
    engine = _make_engine()
    engine._absorb_chain(_chain_goal("Home FC", "Kofi", 55), 55)
    celebrations = [e for e in engine.timeline
                    if e.event_type == EventType.GOAL_CELEBRATION]
    assert len(celebrations) == 1, "a goal must emit exactly one GOAL_CELEBRATION"
    assert "duration" in celebrations[0].metadata


def test_celebration_advances_clock_p2():
    engine = _make_engine()
    engine.state.match_clock_s = 0.0
    clock_before = engine.state.match_clock_s
    engine._absorb_chain(_chain_goal("Home FC", "Kofi", 55), 55)
    celebration = next(e for e in engine.timeline
                       if e.event_type == EventType.GOAL_CELEBRATION)
    dur = celebration.metadata["duration"]
    assert 10 <= dur <= 30
    assert engine.state.match_clock_s - clock_before == dur
    assert celebration.second == int(engine.state.match_clock_s) % 60


def test_clock_no_longer_locked_to_minute_p3():
    """Sanity: the celebration adds real seconds, so the engine's continuous
    clock and minute bucket diverge as designed after a goal."""
    engine = _make_engine()
    engine.state.match_clock_s = 67.0 * 60.0 + 23.0
    engine._absorb_chain(_chain_goal("Home FC", "Kofi", 68), 68)
    celebration = next(e for e in engine.timeline
                       if e.event_type == EventType.GOAL_CELEBRATION)
    assert celebration.minute >= 67
    assert engine.state.match_clock_s > 67.0 * 60.0 + 23.0


def _chain_goal(team: str, scorer: str, minute: int):
    from event_chain import ChainResult
    from match_engine import GameState, MatchPhase
    result = ChainResult()
    result.goal_scored = True
    result.goal_team = team
    result.goal_scorer = scorer
    result.events.append(
        MatchEvent(minute=minute, second=0, event_type=EventType.GOAL,
                   team=team, player=scorer,
                   phase=MatchPhase.PEAK_INTENSITY, game_state=GameState.LEVEL)
    )
    return result


# ─────────────────────────────────────────────
# P4 — INJURY TIMELINE EVENT
# ─────────────────────────────────────────────

def test_injury_sub_emits_injury_event_p4():
    engine = _make_engine()
    engine.active_players = {"Home FC": [_make_player("Injured Fred", "CM")],
                             "Away FC": [_make_player("Other", "ST")]}
    engine.squads = {
        "Home FC": {"substitutes": [_make_player("Replacement", "CM")]},
        "Away FC": {"substitutes": []},
    }
    engine.sub_controller = None
    sub = {
        "team": "Home FC", "minute": 70,
        "player_off": "Injured Fred", "player_on": "Replacement",
        "reason": "injury", "freshness": 1.1, "stamina_at_exit": 33.0,
    }
    before = len(engine.timeline)
    engine._execute_substitution(sub, 70)
    types = [e.event_type for e in engine.timeline]
    assert EventType.INJURY in types, "injury sub must emit an INJURY event"
    inj_idx = types.index(EventType.INJURY)
    sub_idx = types.index(EventType.SUBSTITUTION)
    assert inj_idx < sub_idx, "INJURY event must precede SUBSTITUTION"


# ─────────────────────────────────────────────
# P5-P6 — TRAINING SYSTEM
# ─────────────────────────────────────────────

def test_training_run_week_produces_records_and_commits_p5(tmp_path):
    state = SeasonState("26/27", str(tmp_path / "season_state.json"))
    players = [_make_player("Young Ace", "ST", age=19),
               _make_player("Veteran Pro", "CM", age=31)]
    for p in players:
        state.get_player_state(p.name)  # ensure persisted rows exist

    coach = TrainingSystem()
    records = coach.run_week(players, team_name="Home FC", matchday=2,
                             state=state)

    assert len(records) == len(players)
    for rec, p in zip(records, players):
        assert rec.player == p.name
        # Confidence / fatigue got committed into SeasonState
        row = state.get_player_state(p.name)
        assert "confidence" in row and "fatigue_level" in row
    # Young attacking player should develop on the team-blend focus
    gained = [r for r in records if r.attribute_deltas]
    assert all(v <= 1.5 for r in gained for v in
               r.attribute_deltas.values()), "gains must stay bounded"


def test_training_professional_young_player_develops_p6():
    young_pro = _make_player("Prodigy", "ST", age=20)
    young_pro.personality = PersonalityFactory.create(["workhorse"])
    # force high professionalism deterministically
    young_pro.personality.professionalism = 92.0

    veteran = _make_player("Old Head", "CB", age=34)
    veteran.personality.professionalism = 30.0

    coach = TrainingSystem()
    recs = coach.run_week([young_pro, veteran], team_name="T", matchday=1)

    young_record = next(r for r in recs if r.player == "Prodigy")
    vet_record = next(r for r in recs if r.player == "Old Head")
    assert any(v > 0 for v in young_record.attribute_deltas.values()), \
        "young professional should gain attributes"
    assert young_record.effectiveness > vet_record.effectiveness, \
        "professionalism must drive training effectiveness"


# ─────────────────────────────────────────────
# P7 — INTEGRATION SMOKE
# ─────────────────────────────────────────────

def test_full_match_runs_with_new_events_p7():
    from player_dna import SquadBuilder
    roles = [
        ("GK", ["sweeper_keeper"]), ("CB", ["stopper_defender"]),
        ("CB", ["ball_playing_cb"]), ("LB", ["aggressive_fullback"]),
        ("RB", ["overlapping_fullback"]), ("CDM", ["anchor_man"]),
        ("CM", ["engine"]), ("CM", ["box_box"]), ("CAM", ["creator"]),
        ("LW", ["winger"]), ("ST", ["fox_in_box"]),
    ]
    home_players = SquadBuilder.build("Home FC", [
        (f"Home {i}", pos, specs, 26) for i, (pos, specs) in enumerate(roles)
    ])["starters"]
    away_players = SquadBuilder.build("Away FC", [
        (f"Away {i}", pos, specs, 26) for i, (pos, specs) in enumerate(roles)
    ])["starters"]

    config = MatchConfig(home_team="Home FC", away_team="Away FC",
                         match_date=date.today())
    engine = MatchEngine(config, _team_profile("Home FC"),
                         _team_profile("Away FC"))
    engine.set_squad("Home FC", home_players, [])
    engine.set_squad("Away FC", away_players, [])

    random.seed(7)
    result = engine.simulate()
    # Exercises whatever happened (goals → celebrations, subs → injuries);
    # the simulation simply must complete and stay internally consistent.
    assert result is not None
    assert 0 <= result.home_goals <= 12 and 0 <= result.away_goals <= 12
    if any(e.event_type == EventType.GOAL_CELEBRATION
           for e in engine.timeline):
        for e in engine.timeline:
            if e.event_type == EventType.GOAL_CELEBRATION:
                assert 10 <= e.metadata.get("duration", 0) <= 30


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))