"""
Causal possession (Checkpoint) — a team that wins the ball in open play is
OWED the next sequence, instead of the possession being a memoryless coin
flip re-rolled every sequence.

Covers:
    1. _arm_possession_winner arms the recovering team for the next sequence
    2. ... but only on clean open-play wins — corner / throw-in / goal-kick /
       offside all hand the ball to a restart obligation, never a carry
    3. _resolve_sequence_attacker honours the carry and consumes it
       (single-shot), falling back to the possession-target split otherwise
    4. end-to-end smoke: a real seeded match keeps the invariant that no
       carry is ever left armed, and the split stays in the calibrated band
"""
from unittest import mock
import run_match as R
from match_engine import MatchEngine, MatchConfig
from player_dna import SquadBuilder


def _raw_engine():
    """Engine with a MatchState, no squads needed for primitive-level tests."""
    cfg = MatchConfig(
        home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM,
        match_date=R.MATCH_DATE, matchday=R.MATCHDAY, season=R.SEASON,
        competition=R.COMPETITION, venue=R.VENUE,
        stadium_capacity=R.CAPACITY, referee=R.REFEREE,
        referee_strictness=R.STRICTNESS, is_derby=R.IS_DERBY,
    )
    return MatchEngine(cfg, R.HOME_STYLE, R.AWAY_STYLE)


class _FakeResult:
    def __init__(self, **kwargs):
        self.corner_won = kwargs.get("corner_won", False)
        self.restart_required = kwargs.get("restart_required", False)
        self.offside_detected = kwargs.get("offside_detected", False)


def test_arm_possession_winner_sets_carry():
    eng = _raw_engine()
    assert eng.state.possession_winner == ""
    eng._arm_possession_winner(R.AWAY_TEAM, _FakeResult())
    assert eng.state.possession_winner == R.AWAY_TEAM


def test_arm_possession_winner_gates_restarts():
    eng = _raw_engine()
    # Corner: attacking team keeps the ball via the set piece — no carry.
    eng._arm_possession_winner(R.AWAY_TEAM, _FakeResult(corner_won=True))
    assert eng.state.possession_winner == ""
    # Throw-in / goal-kick: ball is dead, restart redirects it.
    eng._arm_possession_winner(R.AWAY_TEAM, _FakeResult(restart_required=True))
    assert eng.state.possession_winner == ""
    # Offside: defending team is owed a free kick, not an open-play carry.
    eng._arm_possession_winner(R.AWAY_TEAM, _FakeResult(offside_detected=True))
    assert eng.state.possession_winner == ""


def test_resolve_sequence_attacker_honours_carry_and_consumes():
    eng = _raw_engine()
    eng.POSSESSION_CARRY_PROB = 1.0
    eng.state.possession_winner = R.AWAY_TEAM
    with mock.patch("match_engine.random.random", return_value=0.0):
        att, dfd = eng._resolve_sequence_attacker(0.5, R.HOME_TEAM, R.AWAY_TEAM)
    assert att == R.AWAY_TEAM
    assert dfd == R.HOME_TEAM
    assert eng.state.possession_winner == "", "carry must be single-shot"


def test_resolve_sequence_attacker_carry_is_single_shot():
    eng = _raw_engine()
    eng.POSSESSION_CARRY_PROB = 1.0
    eng.state.possession_winner = R.AWAY_TEAM
    with mock.patch("match_engine.random.random", return_value=0.0):
        att1, _ = eng._resolve_sequence_attacker(0.9, R.HOME_TEAM, R.AWAY_TEAM)
    # Second call must fall back to the home-favoured split — the carry is gone.
    with mock.patch("match_engine.random.random", side_effect=[0.0, 0.99]):
        att2, _ = eng._resolve_sequence_attacker(0.9, R.HOME_TEAM, R.AWAY_TEAM)
    assert att1 == R.AWAY_TEAM
    assert att2 == R.HOME_TEAM


def test_resolve_sequence_attacker_rejected_carry_falls_back_to_split():
    eng = _raw_engine()
    eng.POSSESSION_CARRY_PROB = 0.0
    eng.state.possession_winner = R.AWAY_TEAM
    with mock.patch("match_engine.random.random", return_value=0.0):
        att, _ = eng._resolve_sequence_attacker(0.5, R.HOME_TEAM, R.AWAY_TEAM)
    # POSSESSION_CARRY_PROB == 0 → split decides (0.0 < 0.5 → home).
    assert att == R.HOME_TEAM
    assert eng.state.possession_winner == "", "rejected carry still consumed"


def test_full_match_invariant_and_split():
    """End-to-end guard: a real seeded match never leaves a carry armed and
    the measured split stays inside the calibrated 20-80% band."""
    import random
    random.seed(7)
    home_squad = SquadBuilder.build(
        team_name=R.HOME_TEAM, starters=R.HOME_STARTERS, substitutes=R.HOME_SUBS,
        team_superstars=R.HOME_SUPERSTARS, set_piece_takers=R.HOME_SP_TAKERS,
    )
    away_squad = SquadBuilder.build(
        team_name=R.AWAY_TEAM, starters=R.AWAY_STARTERS, substitutes=R.AWAY_SUBS,
        team_superstars=R.AWAY_SUPERSTARS, set_piece_takers=R.AWAY_SP_TAKERS,
    )
    for p in (home_squad["starters"] + home_squad["substitutes"]
              + away_squad["starters"] + away_squad["substitutes"]):
        if p.name in R.SOUL_PLAYERS:
            p.dna.soul = R.SOUL_PLAYERS[p.name]
    cfg = MatchConfig(
        home_team=R.HOME_TEAM, away_team=R.AWAY_TEAM, match_date=R.MATCH_DATE,
        matchday=R.MATCHDAY, season=R.SEASON, competition=R.COMPETITION,
        venue=R.VENUE, stadium_capacity=R.CAPACITY, referee=R.REFEREE,
        referee_strictness=R.STRICTNESS, is_derby=R.IS_DERBY,
    )
    eng = MatchEngine(cfg, R.HOME_STYLE, R.AWAY_STYLE)
    eng.set_squad(R.HOME_TEAM, home_squad["starters"], home_squad["substitutes"])
    eng.set_squad(R.AWAY_TEAM, away_squad["starters"], away_squad["substitutes"])
    res = eng.simulate()

    assert eng.state.possession_winner == "", "carry must be consumed/cleared"
    assert 20.0 <= res.home_possession_pct <= 80.0, res.home_possession_pct