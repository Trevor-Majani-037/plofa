"""Tests for the post-match brain observatory.

These are deliberately cheap: no match is simulated.  The observatory's job
is arithmetic over a decision ledger, so the ledger is built by hand and the
statistics are checked against cases with a known answer.

Run: python -m pytest tests/test_brain_observatory.py -q
"""
import json
import os
import types

import pytest

import brain_observatory as obs
from brain_integration import collect_decisions


# ─────────────────────────────────────────────────────────────
# Fakes: the smallest thing that looks like a MatchEvent
# ─────────────────────────────────────────────────────────────
class _EventType:
    def __init__(self, name):
        self.name = name


def _event(name, *, team="Home", player="A", minute=10.0, x=50.0, y=34.0,
           outcome=True, metadata=None):
    e = types.SimpleNamespace()
    e.event_type = _EventType(name)
    e.team = team
    e.player = player
    e.minute = minute
    e.second = 0
    e.match_clock_s = minute * 60.0
    e.location_x = x
    e.location_y = y
    e.end_x = None
    e.end_y = None
    e.outcome = outcome
    e.metadata = metadata or {}
    return e


def _stamped(event_name, intent, *, team="Home", player="A", minute=10.0,
             x=50.0, pressure=None, outcome=True):
    md = {"active_brain": {"action": "PASS", "intent": intent,
                           "confidence": 0.5, "reason": "test",
                           "decision_quality": 0.7, "is_error": False}}
    if pressure is not None:
        md["under_pressure"] = pressure
    return _event(event_name, team=team, player=player, minute=minute,
                  x=x, outcome=outcome, metadata=md)


def _engine(direction=None):
    return types.SimpleNamespace(
        position_engine=types.SimpleNamespace(
            team_attacks_right=direction or {"Home": True, "Away": False}))


# ─────────────────────────────────────────────────────────────
# collect_decisions
# ─────────────────────────────────────────────────────────────
def test_collect_decisions_ignores_unstamped_events():
    tl = [
        _event("PASS"),                       # no active_brain
        _stamped("PASS", "SAFE_PASS"),
        _event("BALL_RECEIPT"),
    ]
    recs = collect_decisions(tl, engine=_engine(), home_team="Home",
                             away_team="Away")
    assert len(recs) == 1
    assert recs[0]["intent"] == "SAFE_PASS"


def test_collect_decisions_assigns_third_from_attacking_direction():
    tl = [
        _stamped("PASS", "SAFE_PASS", x=10.0),   # own third for Home
        _stamped("PASS", "SAFE_PASS", x=80.0),   # attacking third for Home
        _stamped("PASS", "SAFE_PASS", team="Away", x=80.0),  # Away defends high x
    ]
    recs = collect_decisions(tl, engine=_engine(), home_team="Home",
                             away_team="Away")
    assert [r["zone"] for r in recs] == ["DEF", "ATT", "DEF"]


def test_collect_decisions_tracks_running_score():
    tl = [
        _stamped("PASS", "SAFE_PASS", minute=5),
        _event("GOAL", team="Home", minute=10),
        _stamped("PASS", "SAFE_PASS", minute=20),
        _event("GOAL", team="Away", minute=30),
        _stamped("PASS", "SAFE_PASS", minute=40),
    ]
    recs = collect_decisions(tl, engine=_engine(), home_team="Home",
                             away_team="Away")
    assert [r["score_state"] for r in recs] == ["level", "leading", "level"]


def test_collect_decisions_marks_shoot_intent_as_disobeyed_by_a_pass():
    """A SHOOT intent that the body turns into a pass is an override, and the
    ledger has to say so — this is the number the whole tool exists to show."""
    tl = [_stamped("PASS", "SHOOT")]
    recs = collect_decisions(tl, engine=_engine(), home_team="Home",
                             away_team="Away")
    assert recs[0]["family"] == "pass"
    assert recs[0]["obeyed"] is False


def test_collect_decisions_marks_obedience_when_family_matches():
    tl = [_stamped("CARRY", "CARRY")]
    recs = collect_decisions(tl, engine=_engine(), home_team="Home",
                             away_team="Away")
    assert recs[0]["obeyed"] is True


def test_collect_decisions_accepts_a_result_object():
    result = types.SimpleNamespace(timeline=[_stamped("PASS", "SAFE_PASS")])
    recs = collect_decisions(result, engine=_engine(), home_team="Home",
                             away_team="Away")
    assert len(recs) == 1


# ─────────────────────────────────────────────────────────────
# Information measures
# ─────────────────────────────────────────────────────────────
def test_entropy_of_a_constant_is_zero():
    assert obs._entropy([10]) == pytest.approx(0.0)


def test_entropy_of_a_fair_coin_is_one_bit():
    assert obs._entropy([50, 50]) == pytest.approx(1.0)


def test_mutual_information_is_zero_when_independent():
    pairs = [("A", "x"), ("A", "y"), ("B", "x"), ("B", "y")] * 10
    mi, _h = obs._mutual_information(pairs)
    assert mi == pytest.approx(0.0, abs=1e-9)


def test_mutual_information_is_full_entropy_when_deterministic():
    pairs = [("A", "x"), ("B", "y")] * 10
    mi, h_intent = obs._mutual_information(pairs)
    assert mi == pytest.approx(h_intent, abs=1e-9)


# ─────────────────────────────────────────────────────────────
# Shuffle test — must separate signal from luck
# ─────────────────────────────────────────────────────────────
def _records(pairs):
    return [{"team": "T", "player": "P", "zone": z, "pressure": None,
             "intent": i, "confidence": 0.5, "obeyed": True, "minute": 1.0}
            for z, i in pairs]


def test_shuffle_test_finds_signal_when_situation_decides():
    # ATT always shoots, DEF always recycles: I is maximal.
    pairs = ([("ATT", "SHOOT")] * 30) + ([("DEF", "RECYCLE")] * 30)
    res = obs._shuffle_test(_records(pairs), permutations=200, seed=0)
    assert res["p"] <= 0.01
    assert res["observed"] > res["null_p95"]


def test_shuffle_test_finds_nothing_when_choice_is_random():
    pairs = [("ATT", "SHOOT"), ("ATT", "RECYCLE"),
             ("DEF", "SHOOT"), ("DEF", "RECYCLE")] * 20
    res = obs._shuffle_test(_records(pairs), permutations=200, seed=0)
    assert res["p"] > 0.05


def test_shuffle_test_declines_tiny_samples():
    assert obs._shuffle_test(_records([("ATT", "SHOOT")] * 5)) is None


# ─────────────────────────────────────────────────────────────
# attach_null_test
# ─────────────────────────────────────────────────────────────
def _report(records):
    return {"thought_log": records, "match": {"home": "H", "away": "A",
                                              "score": "1-0"}}


def test_attach_null_test_builds_per_player_rows():
    recs = []
    for player in ("P1", "P2"):
        recs += [{"team": "T", "player": player, "zone": z, "pressure": None,
                  "intent": i, "confidence": 0.5, "obeyed": True, "minute": 1.0}
                 for z, i in ([("ATT", "SHOOT")] * 20 + [("DEF", "RECYCLE")] * 20)]
    report = obs.attach_null_test(_report(recs), permutations=50, seed=0)
    nt = report["null_test"]
    assert nt["overall"] is not None
    assert {r["player"] for r in nt["players"]} == {"P1", "P2"}
    assert all(r["p"] <= 0.05 for r in nt["players"])


def test_attach_null_test_skips_players_below_threshold():
    recs = [{"team": "T", "player": "Sparse", "zone": "ATT", "pressure": None,
             "intent": "SHOOT", "confidence": 0.5, "obeyed": True, "minute": 1.0}]
    report = obs.attach_null_test(_report(recs), permutations=20, seed=0)
    assert report["null_test"]["players"] == []


def test_attach_null_test_handles_an_empty_ledger():
    report = obs.attach_null_test(_report([]))
    assert report["null_test"]["overall"] is None


# ─────────────────────────────────────────────────────────────
# write_trace / round trip
# ─────────────────────────────────────────────────────────────
def test_write_trace_round_trips_through_analyse_match(tmp_path):
    tl = [_stamped("PASS", "SAFE_PASS", minute=m) for m in (10, 20, 30)]
    result = types.SimpleNamespace(
        timeline=tl,
        config=types.SimpleNamespace(home_team="Home", away_team="Away"),
        home_goals=1, away_goals=0,
    )
    path = os.path.join(str(tmp_path), "trace.json")
    obs.write_trace(result, path, engine=_engine())

    doc = json.load(open(path, encoding="utf-8"))
    assert doc["match"] == {"home": "Home", "away": "Away", "score": "1-0"}
    assert len(doc["decisions"]) == 3

    report = obs.analyse_match(doc)
    assert report["coverage"]["decisions"] == 3
    assert report["coverage"]["direction_known"] is True


# ─────────────────────────────────────────────────────────────
# The shot gate
# ─────────────────────────────────────────────────────────────
def _stamped_shot(event_name="CARRY", *, gate=None, shot_intent=True):
    md = {"active_brain": {"action": "SHOOT", "intent": "SHOOT",
                           "confidence": 0.6, "reason": "test",
                           "decision_quality": 0.7, "is_error": False}}
    if shot_intent:
        md["shot_intent"] = True
    if gate is not None:
        md["shot_gate"] = gate
    return _event(event_name, metadata=md)


def test_shot_dispatch_is_obeyed_not_a_disobeyed_carry():
    """A `shot_intent` event is recorded as CARRY for timeline ordering only.
    Reading it literally scored every successful shot as a disobeyed carry."""
    recs = collect_decisions([_stamped_shot()], engine=_engine(),
                             home_team="Home", away_team="Away")
    assert recs[0]["family"] == "shoot"
    assert recs[0]["obeyed"] is True
    assert recs[0]["dispatch"] is True


def test_collect_decisions_reads_the_gate_from_inside_the_decision():
    """An allowed shot carries its gate through to the SHOT_* event, where it
    lives inside active_brain rather than beside it."""
    md = {"active_brain": {"intent": "SHOOT", "confidence": 0.6,
                           "shot_gate": {"score": 0.44, "threshold": 0.2,
                                         "passed": True, "veto": None}}}
    ev = _event("SHOT_ON_TARGET", metadata=md)
    recs = collect_decisions([ev], engine=_engine(), home_team="Home",
                             away_team="Away")
    assert recs[0]["shot_gate"]["score"] == 0.44
    assert recs[0]["shot_gate"]["passed"] is True


def test_shot_gate_summary_separates_allowed_from_refused():
    gate_ok = {"score": 0.5, "threshold": 0.2, "passed": True, "veto": None}
    gate_no = {"score": 0.05, "threshold": 0.2, "passed": False,
               "veto": "geometry"}
    tl = [_stamped_shot(gate=gate_ok), _stamped_shot("PASS", gate=gate_no),
          _stamped_shot("PASS", gate=gate_no)]
    recs = collect_decisions(tl, engine=_engine(), home_team="Home",
                             away_team="Away")
    gate = obs._shot_gate_summary(recs)
    assert gate["asked"] == 3
    assert gate["passed"] == 1
    assert gate["vetoed"] == 2
    assert gate["veto_reasons"] == {"geometry": 2}
    assert gate["veto_score_median"] == 0.05
    assert gate["veto_score_max"] == 0.05


def test_shot_gate_summary_is_absent_without_shot_intents():
    recs = collect_decisions([_stamped("PASS", "SAFE_PASS")], engine=_engine(),
                             home_team="Home", away_team="Away")
    assert obs._shot_gate_summary(recs) is None


def test_render_includes_the_shot_gate_when_present():
    gate_no = {"score": 0.04, "threshold": 0.2, "passed": False,
               "veto": "geometry"}
    tl = [_stamped_shot("PASS", gate=gate_no) for _ in range(20)]
    recs = collect_decisions(tl, engine=_engine(), home_team="Home",
                             away_team="Away")
    report = obs.analyse_match({"match": {"home": "H", "away": "A",
                                          "score": "0-0"}, "decisions": recs})
    text = obs.render_markdown(report)
    assert "The shot gate" in text
    assert "asked to shoot" in text


# ─────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────
def test_render_includes_the_shuffle_test_verdict():
    recs = ([{"team": "T", "player": "P", "zone": "ATT", "pressure": None,
              "intent": "SHOOT", "confidence": 0.5, "obeyed": True, "minute": 1.0}] * 20
            + [{"team": "T", "player": "P", "zone": "DEF", "pressure": None,
                "intent": "RECYCLE", "confidence": 0.5, "obeyed": True, "minute": 1.0}] * 20)
    report = obs.analyse_match({"match": {"home": "H", "away": "A",
                                          "score": "0-0"}, "decisions": recs})
    obs.attach_null_test(report, permutations=50, seed=0)
    text = obs.render_markdown(report)
    assert "shuffle test" in text
    assert "p = " in text


def test_render_survives_an_empty_ledger():
    report = obs.analyse_match({"match": {"home": "H", "away": "A",
                                          "score": "0-0"}, "decisions": []})
    text = obs.render_markdown(report)
    assert "0 on-ball decisions" in text
