"""
PLOFA 26/27 — Chance Creation Ledger tests.
Checks that key passes / assists / xA / second assists / SCA / fantasy
assists and big chances are derived from the REAL timeline, never invented.
"""
import pytest

from match_engine import MatchEvent, EventType, SituationType
from chance_creation import (
    ChanceCreationLedger, ChanceRecord,
    BIG_CHANCE_XG_FLOOR,
)


def evt(etype, team, player, **kw):
    base = dict(
        minute=0, second=0, secondary_player=None,
        situation=SituationType.OPEN_PLAY,
        location_x=50.0, location_y=34.0, end_x=None, end_y=None,
        xg=0.0, xa=0.0, outcome=True, body_part="right_foot",
        metadata={},
    )
    base.update(kw)
    return MatchEvent(event_type=etype, team=team, player=player, **base)


def run(timeline):
    return ChanceCreationLedger(timeline).compute()


def test_goal_assist_second_assist_and_xa():
    """The pass to a scorer is a Goal Assist; the pass to the assister is a
    Second Assist; xA of the delivery equals xG of the shot it created."""
    tl = [
        evt(EventType.PASS, "Home", "A", secondary_player="B"),
        evt(EventType.BALL_RECEIPT, "Home", "B"),
        evt(EventType.PASS, "Home", "B", secondary_player="S", xg=0.0),
        evt(EventType.BALL_RECEIPT, "Home", "S"),
        evt(EventType.SHOT_ON_TARGET, "Home", "S", xg=0.35),
        evt(EventType.GOAL, "Home", "S", xg=0.35),
    ]
    ledger = run(tl)
    assert len(ledger.records) == 1
    r = ledger.records[0]
    assert r.outcome == "goal"
    assert r.creator == "B"
    assert r.is_goal_assist is True
    assert r.second_assist == "A"
    assert r.xa == pytest.approx(0.35)
    assert r.xg == pytest.approx(0.35)
    p = ledger.per_player["B"]
    assert p["goal_assists"] == 1 and p["assists"] == 1
    assert p["chances_created"] == 1 and p["shot_assists"] == 0
    assert ledger.per_player["A"]["second_assists"] == 1


def test_shot_assist_on_saved_shot_not_goal_assist():
    tl = [
        evt(EventType.PASS, "Home", "B", secondary_player="S"),
        evt(EventType.BALL_RECEIPT, "Home", "S"),
        evt(EventType.SHOT_ON_TARGET, "Home", "S", xg=0.2),
        evt(EventType.SAVE, "Away", "GK"),
    ]
    ledger = run(tl)
    assert len(ledger.records) == 1
    r = ledger.records[0]
    assert r.outcome == "save"
    assert r.creator == "B"
    assert r.is_goal_assist is False
    p = ledger.per_player["B"]
    assert p["shot_assists"] == 1 and p["goal_assists"] == 0
    assert p["chances_created"] == 1


def test_rebound_tap_in_grants_fantasy_assist():
    """Shot saved, rebound tapped in by a DIFFERENT player: the original
    shooter gets a Fantasy Assist and the rebound goal is its own record."""
    tl = [
        evt(EventType.PASS, "Home", "C", secondary_player="D"),
        evt(EventType.BALL_RECEIPT, "Home", "D"),
        evt(EventType.SHOT_ON_TARGET, "Home", "D", xg=0.5),
        evt(EventType.SAVE, "Away", "GK"),
        evt(EventType.BALL_RECEIPT, "Home", "E"),
        evt(EventType.GOAL, "Home", "E", xg=0.8),
    ]
    ledger = run(tl)
    assert len(ledger.records) == 2
    saved, rebound = ledger.records
    assert saved.outcome == "save" and saved.shooter == "D"
    assert saved.fantasy_assist == "D"          # the original shot
    assert rebound.outcome == "goal" and rebound.shooter == "E"
    assert rebound.creator == ""                 # no pass directly to E
    p = ledger.per_player["D"]
    assert p["fantasy_assists"] == 1 and p["assists"] == 1
    assert ledger.per_player["C"]["chances_created"] == 1


def test_woodwork_upgraded_to_goal_by_same_player():
    """HIT_WOODWORK followed immediately by a GOAL from the SAME player is
    one shot upgraded to scored — not a fantasy-assist situation."""
    tl = [
        evt(EventType.PASS, "Home", "B", secondary_player="S"),
        evt(EventType.BALL_RECEIPT, "Home", "S"),
        evt(EventType.HIT_WOODWORK, "Home", "S", xg=0.4),
        evt(EventType.GOAL, "Home", "S", xg=0.4),
    ]
    ledger = run(tl)
    assert len(ledger.records) == 1
    r = ledger.records[0]
    assert r.outcome == "goal"
    assert r.shooter == "S"
    assert r.fantasy_assist == ""
    assert r.is_goal_assist is True


def test_no_shot_assist_across_turnover():
    """A turnover breaks the chain — a shot after it cannot be credited to a
    pass from before it."""
    tl = [
        evt(EventType.PASS, "Home", "B", secondary_player="S"),
        evt(EventType.BALL_RECEIPT, "Home", "S"),
        evt(EventType.TURNOVER, "Home", "S"),
        evt(EventType.RECOVERY, "Away", "X"),
        evt(EventType.SHOT_ON_TARGET, "Away", "Y", xg=0.3),
    ]
    ledger = run(tl)
    assert len(ledger.records) == 1
    assert ledger.records[0].creator == ""
    assert ledger.per_player.get("B") is None or ledger.per_player["B"]["shot_assists"] == 0


def test_big_chance_threshold():
    tl_low = [evt(EventType.SHOT_ON_TARGET, "Home", "S", xg=0.30)]
    tl_high = [evt(EventType.SHOT_ON_TARGET, "Home", "S", xg=0.50)]
    assert run(tl_low).records[0].is_big is False
    assert run(tl_high).records[0].is_big is True
    assert run(tl_high).records[0].is_big == (0.50 >= BIG_CHANCE_XG_FLOOR)


def test_sca_counts_last_two_offensive_actions():
    """SCA = the two offensive actions before the shot (e.g. dribbler + passer)."""
    tl = [
        evt(EventType.DRIBBLE_SUCCESS, "Home", "X"),
        evt(EventType.PASS, "Home", "X", secondary_player="S"),
        evt(EventType.BALL_RECEIPT, "Home", "S"),
        evt(EventType.SHOT_ON_TARGET, "Home", "S", xg=0.3),
    ]
    ledger = run(tl)
    r = ledger.records[0]
    assert r.creator == "X"
    assert set(r.sca_players) == {"X"}
    # dribbler and passer are the same player here — check a two-person build
    tl2 = [
        evt(EventType.DRIBBLE_SUCCESS, "Home", "X"),
        evt(EventType.PASS, "Home", "B", secondary_player="S"),
        evt(EventType.BALL_RECEIPT, "Home", "S"),
        evt(EventType.SHOT_ON_TARGET, "Home", "S", xg=0.3),
    ]
    r2 = run(tl2).records[0]
    assert r2.creator == "B"
    assert set(r2.sca_players) == {"B", "X"}


def test_penalty_after_won_foul_is_fantasy_assist():
    tl = [
        evt(EventType.DRIBBLE_SUCCESS, "Home", "X"),
        evt(EventType.FOUL_WON, "Home", "X"),
        evt(EventType.PENALTY_SCORED, "Home", "Y", xg=0.79,
            situation=SituationType.PENALTY),
    ]
    ledger = run(tl)
    r = ledger.records[0]
    assert r.outcome == "goal"
    assert r.fantasy_assist == "X"
    assert ledger.per_player["X"]["fantasy_assists"] == 1


def test_corner_delivery_is_shot_assist():
    """A corner delivery that leads directly to a shot is the shot assist, even
    with no receiver recorded."""
    tl = [
        evt(EventType.CORNER_TAKEN, "Home", "T", situation=SituationType.CORNER),
        evt(EventType.SHOT_ON_TARGET, "Home", "H", xg=0.2,
            situation=SituationType.CORNER),
        evt(EventType.SAVE, "Away", "GK"),
    ]
    ledger = run(tl)
    assert len(ledger.records) == 1
    r = ledger.records[0]
    assert r.creator == "T"
    assert ledger.per_player["T"]["shot_assists"] == 1


def test_aggregate_totals_across_multiple_chances():
    tl = [
        evt(EventType.PASS, "Home", "A", secondary_player="B"),
        evt(EventType.BALL_RECEIPT, "Home", "B"),
        evt(EventType.PASS, "Home", "B", secondary_player="S"),
        evt(EventType.BALL_RECEIPT, "Home", "S"),
        evt(EventType.SHOT_ON_TARGET, "Home", "S", xg=0.3),
        evt(EventType.SAVE, "Away", "GK"),
        evt(EventType.PASS, "Home", "C", secondary_player="T"),
        evt(EventType.BALL_RECEIPT, "Home", "T"),
        evt(EventType.SHOT_ON_TARGET, "Home", "T", xg=0.6),
        evt(EventType.GOAL, "Home", "T", xg=0.6),
    ]
    ledger = run(tl)
    assert ledger.per_player["B"]["shot_assists"] == 1
    assert ledger.per_player["C"]["goal_assists"] == 1
    assert ledger.per_player["C"]["big_chances_created"] == 1
    assert ledger.per_player["C"]["xa"] == pytest.approx(0.6)
    # SCA credits for both shots
    assert ledger.per_player["B"]["sca"] == 1
    assert ledger.per_player["C"]["sca"] == 1
    # the two set-up passes are marked for gold plotting
    assert len(ledger.shot_assist_event_indexes) == 2


def test_miss_off_target_with_no_shot_assist():
    tl = [evt(EventType.SHOT_OFF_TARGET, "Home", "S", xg=0.1)]
    ledger = run(tl)
    assert len(ledger.records) == 1
    r = ledger.records[0]
    assert r.outcome == "miss"
    assert r.creator == ""
    assert r.fantasy_assist == ""
    assert "S" not in ledger.per_player or ledger.per_player["S"]["chances_created"] == 0


def test_blocked_shot_counts_as_save_outcome_with_shot_assist():
    tl = [
        evt(EventType.PASS, "Home", "B", secondary_player="S"),
        evt(EventType.BALL_RECEIPT, "Home", "S"),
        evt(EventType.SHOT_BLOCKED, "Home", "S", xg=0.2),
        evt(EventType.BLOCK, "Away", "D"),
    ]
    ledger = run(tl)
    r = ledger.records[0]
    assert r.outcome == "block"
    assert r.creator == "B"
    assert ledger.per_player["B"]["shot_assists"] == 1


def test_key_pass_plot_endpoint_falls_back_to_shot_receiver_point():
    from player_maps import resolve_key_pass_plot_endpoints

    event = evt(EventType.PASS, "Home", "B", secondary_player="S",
                location_x=12.0, location_y=18.0, end_x=None, end_y=None)
    ledger_record = type("Rec", (), {
        "pass_event_index": 0,
        "pass_end_x": 12.0,
        "pass_end_y": 18.0,
        "shot_x": 72.0,
        "shot_y": 38.0,
    })()

    endpoint = resolve_key_pass_plot_endpoints(event, ledger_record, index=0)
    assert endpoint == (72.0, 38.0)
