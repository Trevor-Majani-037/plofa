"""Step-7 (audit "Observability"): decision journal + ``why(player, minute)``.

Mirrors the Step-6 ``tests/test_tactics_context.py`` discipline byte-for-byte:
deterministic (no ``input()``, no RNG, no surprise), assert-gated, and the
journal is an **append-only sidecar** -- a fresh journal records nothing; the
production on-ball path is untouched unless a harness explicitly opts in with
the same ``record_trace=True`` trace the brain already produces. The journal
lifts those traces into ``why(player, minute)`` answers carrying the full
Step-7 field contract from PLOFA_ARCHITECTURE_AUDIT.md item 7:

  world state, perceived state, sensor values, DNA, tactics, probs,
  alternatives, chosen intent, result, immediate value, future value,
  final consequence.

Realism totals (goals/shots/possession/pass completion/turnovers/xG) are the
**consumer** of these records (Phase-17 realism diagnostics), NOT authored
here -- the journal is the boundary, never the template.
"""

import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decision_journal as dj


# -----------------------------------------------------------------------------
# The on-ball trace the brain already produces with record_trace=True.
# -----------------------------------------------------------------------------
def _trace(player: str, minute: float, chosen_idx: int = 0,
           note: str = "progressive pass completed") -> Dict[str, Any]:
    return {
        "player": player,
        "minute": minute,
        "chosen_idx": chosen_idx,
        "chosen_intent": "PROGRESSIVE_PASS",
        "chosen_probability": 0.57,
        "fatigue": 0.33,
        "temperature": 26.0,
        "visibility_floor": 0.90,
        "sensor_vector": [0.1 * (n % 3) for n in range(24)],
        "candidates": [
            {"intent": "PROGRESSIVE_PASS", "objective_value": 0.82,
             "risk": 0.30, "target": "CM8", "note": "line-breaking option"},
            {"intent": "SAFE_PASS", "objective_value": 0.40, "risk": 0.05,
             "target": "CB5", "note": "reliable short option"},
            {"intent": "SHOOT", "objective_value": 0.27, "risk": 0.55,
             "target": None, "note": "believes the window is there"},
        ],
        "perceived": [
            {"intent": "PROGRESSIVE_PASS", "perceived_value": 0.66,
             "noise": 0.05, "bias": 0.0},
            {"intent": "SAFE_PASS", "perceived_value": 0.34,
             "noise": 0.03, "bias": 0.0},
            {"intent": "SHOOT", "perceived_value": 0.27,
             "noise": 0.20, "bias": 0.0},
        ],
        "result": note,
        "immediate_value": 0.62,
    }


# -----------------------------------------------------------------------------
# The Step-7 enrichment the trace adapter folds in (world/perceived/DNA/
# tactics/future/final) -- optional overrides, never invented keys.
# -----------------------------------------------------------------------------
def _enrichment(world_score_diff: int = 1, future: float = 0.62,
                consequence: str = "led to a shot two minutes later") -> Dict[str, Any]:
    return {
        "world_state": {"score_diff": world_score_diff, "formation": "4-3-3"},
        "perceived_state": {"under_pressure": False, "space_ahead": 0.61},
        "dna": {"pace": 0.80, "composure": 0.70},
        "tactics": {"style": "balanced", "tempo": 0.62},
        "future_value": future,
        "final_consequence": consequence,
    }


# -----------------------------------------------------------------------------
# 1. A fresh journal records nothing and why(player, minute) answers cleanly
#    (Step-7 default is byte-identical: the journal is an empty sidecar).
# -----------------------------------------------------------------------------
def test_fresh_journal_is_empty_and_why_unanswered():
    j = dj.new_journal()
    assert len(j) == 0
    w = j.why("Striker9", 67.0)
    assert w["answered"] is False
    assert "explanation" in w and w["entry"] is None


# -----------------------------------------------------------------------------
# 2. journalize_trace builds the full Step-7 entry: every contract field.
# -----------------------------------------------------------------------------
def test_journalize_trace_carries_full_step7_contract():
    j = dj.new_journal()
    j.record(dj.journalize_trace(_trace("Striker9", 67.0), **_enrichment()))
    assert len(j) == 1
    e = j._entries[0]
    for key in ("minute", "player", "world_state", "perceived_state",
                "sensor_values", "dna", "tactics", "probs", "alternatives",
                "chosen_intent", "result", "immediate_value", "future_value",
                "final_consequence"):
        assert key in e, "Step-7 entry missing contract field {0}".format(key)
    assert e["player"] == "Striker9"
    assert e["chosen_intent"] == "PROGRESSIVE_PASS"
    assert e["world_state"]["score_diff"] == 1


# -----------------------------------------------------------------------------
# 3. record() then why(player, minute) explains the chosen intent with the
#    same perceived/DNA/tactics the brain recorded -- no re-perception.
# -----------------------------------------------------------------------------
def test_record_then_why_answers_chosen_intent():
    j = dj.new_journal()
    j.record(dj.journalize_trace(_trace("Striker9", 67.0), **_enrichment()))
    w = j.why("Striker9", 67.0)
    assert w["answered"] is True
    assert w["entry"] is not None
    assert "PROGRESSIVE_PASS" in w["explanation"]


# -----------------------------------------------------------------------------
# 4. why() honours the minute bound and the player identity (deterministic:
#    a decision at minute 67 is not an explanation for minute 10).
# -----------------------------------------------------------------------------
def test_why_honours_minute_and_player_bounds():
    j = dj.new_journal()
    j.record(dj.journalize_trace(_trace("Striker9", 67.0), **_enrichment()))
    assert j.why("Striker9", 10.0)["answered"] is False
    assert j.why("Nobody", 67.0)["answered"] is False
    assert j.why("Striker9", 67.0)["answered"] is True


# -----------------------------------------------------------------------------
# 5. Determinism: equal traces enrich to equal journals and identical why().
#    without RNG or input(). Also the journal exposes records for the realism
#    consumer (Phase-17) -- but never claims the realism totals itself.
# -----------------------------------------------------------------------------
def test_journal_deterministic_and_exposes_records_for_realism_consumer():
    a = dj.new_journal()
    b = dj.new_journal()
    for minute in (45.0, 61.0):
        for player in ("Striker9", "CM8"):
            a.record(dj.journalize_trace(_trace(player, minute), **_enrichment()))
            b.record(dj.journalize_trace(_trace(player, minute), **_enrichment()))
    assert a == b
    assert len(a) == 4
    assert a.since(60.0) == b.since(60.0)
    assert len(a.since(61.0)) == 2
    recs = a.to_records()
    assert len(recs) == 4
    assert all("goals" not in r and "possession" not in r for r in recs)


# -----------------------------------------------------------------------------
# 6. since() and to_records() feed the realism diagnostics without the journal
#    pretending to be that diagnostics template.
# -----------------------------------------------------------------------------
def test_recorded_result_minutes_are_since_queryable():
    j = dj.new_journal()
    for minute, chosen in ((45.0, 0), (67.0, 0), (82.0, 0)):
        pass
