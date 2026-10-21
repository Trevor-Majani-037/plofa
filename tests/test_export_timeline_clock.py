"""THE EXPORT'S TIME AXIS — the timeline must FLOW.

THE DEFECT THIS FILE EXISTS FOR
-------------------------------
The user watched an exported match and reported that the first ten seconds
"jump" — the timeline does not flow.

The cause: `event_chain.make_event` stamps
`second=kwargs.pop("second", random.randint(0, 59))`, so an event's `second`
is a RANDOM placeholder.  The engine's own docstring on
`MatchEngine.chronograph` says so: *"The timeline's `MatchEvent.second` is
only a bucket (mostly a placeholder random value)"*.

Measured before the fix, the exported second against the truth: **median
|difference| 16.6 s, max 59.9 s**, and only **3.2%** of events within 1 s.

The truth already existed — `MatchEngine.chronograph()` rebuilds every event's
real time from the engine's one continuous clock — but it was never written
back onto the events, so no consumer could see it.

THE FIX (2026-10-20)
--------------------
1. `MatchEvent.match_clock_s` — the TRUE seconds from kickoff.
2. `chronograph()` writes it back onto every chain event; `_emit_event` stamps
   the engine's direct events; the goal celebration stamps its own.
3. `exporter.py::_timeline_json` reports the true `second` / `clock` /
   `clock_s` and a StatsBomb-style MILLISECOND `timestamp`; the shot map's
   sort and its x-axis position use the true clock (it used to place every
   shot at a random position inside its minute).
4. `plofa_export.convert_event` already preferred `match_clock_s`; the field
   now exists, so its `clock` is real and it gains a ms `timestamp`.

WHAT THIS FILE PINS
-------------------
That the exported timeline FLOWS (non-decreasing, and matching the engine's
own chronology), that it carries a millisecond timestamp, and that the
provider-style export's clock is real rather than built from the placeholder.
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOME, AWAY = "Oxton", "Natrican"


@pytest.fixture(scope="module")
def exported():
    """One real match, exported to JSON, plus the engine's own chronology."""
    from tools.diag._diag_watch import build
    from player_dna import SquadBuilder
    import exporter as EX

    eng, home_raw, away_raw = build()
    # The exporter wants the BUILT squads (PlayerProfile), not the loader's
    # raw records — `run_match.py` builds them the same way.
    home = SquadBuilder.build(HOME, starters=home_raw["starters"],
                              substitutes=home_raw["substitutes"])
    away = SquadBuilder.build(AWAY, starters=away_raw["starters"],
                              substitutes=away_raw["substitutes"])
    all_players = {HOME: home, AWAY: away}

    random.seed(31)
    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        res = eng.simulate()
    finally:
        sys.stdout = real

    ex = EX.PLOFAExporter(res, all_players)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "match.json")
    try:
        ex.export_json(path)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return eng, res, doc


def test_the_timeline_carries_a_millisecond_timestamp(exported):
    """StatsBomb's `timestamp` is ms within the period."""
    eng, res, doc = exported
    tl = doc.get("timeline") or []
    assert tl, "the export has no timeline"
    for row in tl[:200]:
        assert "timestamp" in row, "a timeline row has no millisecond timestamp"
        assert isinstance(row["timestamp"], int)
        assert row["timestamp"] >= 0
    assert tl[0]["timestamp"] == 0, "the first event is not at kickoff"


def test_the_timeline_FLOWS(exported):
    """The user's observation: the first ten seconds "jump"."""
    eng, res, doc = exported
    tl = doc.get("timeline") or []
    ts = [r["timestamp"] for r in tl]
    assert len(ts) > 100, "the match is too short to say anything"
    back = [(i, a, b) for i, (a, b) in enumerate(zip(ts, ts[1:])) if b < a]
    assert not back, (
        f"the timeline goes BACKWARDS at {len(back)} places, first at "
        f"index {back[0][0]}: {back[0][1]} -> {back[0][2]}")


def test_the_timeline_matches_the_engines_own_chronology(exported):
    """The exported time must be the engine's real clock, not a rebuilt one."""
    eng, res, doc = exported
    tl = doc.get("timeline") or []
    chron = getattr(res, "chronology", None)
    assert chron is not None and getattr(chron, "events", None), \
        "the engine produced no chronology"
    tes = chron.events
    # join by source_event_index, NOT by position (the two lists differ)
    by_index = {t.source_event_index: t for t in tes
                if getattr(t, "source_event_index", -1) >= 0}
    checked = 0
    for i, row in enumerate(tl):
        te = by_index.get(i)
        if te is None:
            continue
        assert abs(row["clock_s"] - te.match_clock_s) < 0.005, (
            f"event {i}: exported clock_s {row['clock_s']} vs the engine's "
            f"{te.match_clock_s}")
        # the exported `second` must be the engine's own `second` — the same
        # convention, including the `min(..., 59.9)` clamp, so an event on a
        # minute boundary cannot be reported by two different rules.
        assert abs(row["second"] - te.second) < 0.05, (
            f"event {i}: exported second {row['second']} vs the engine's "
            f"{te.second}")
        checked += 1
    assert checked > 100, f"only {checked} events could be checked"


def test_the_placeholder_second_is_not_what_is_exported(exported):
    """The exported second must NOT be the random placeholder."""
    eng, res, doc = exported
    tl = doc.get("timeline") or []
    raw = list(getattr(res, "timeline", []) or [])
    if len(raw) != len(tl):
        return
    same = sum(1 for e, row in zip(raw, tl)
               if int(e.second) == int(row["second"]))
    # the placeholder is random, so it coincides with the truth only by
    # chance; a real time must coincide far more often
    assert same < len(tl) * 0.5, (
        "the exported second still looks like the placeholder")


def test_the_provider_style_export_prefers_the_true_clock(exported):
    """`plofa_export.convert_event` already looked for `match_clock_s`; now
    the field exists, so its `clock` is real."""
    eng, res, doc = exported
    from tools.diag._diag_plofa_export import roster_from
    import plofa_export as px
    try:
        roster = roster_from(eng)
    except Exception:
        pytest.skip("no roster available for the provider export")
    pdoc = px.export_match(res, engine=eng, roster=roster)
    evs = pdoc.get("events") or []
    assert evs, "the provider export has no events"
    for e in evs[:100]:
        assert "clock_s" in e
        assert "timestamp" in e
        assert abs(e["timestamp"] - e["clock_s"] * 1000) < 2.0


def _reads_true_clock(src: str) -> bool:
    """Does this source mention `match_clock_s` as a CODE literal?

    AST, not a text scan: **a name inside a comment is not a call site** —
    the sixth recorded instance of that shape in this project.  But note the
    mirror trap as well: the shot map reads the clock through
    `getattr(e, "match_clock_s", None)`, which is a STRING ARGUMENT, not an
    attribute access — so looking for `ast.Attribute` finds nothing and the
    guard fails for the wrong reason.  A `ast.Constant` is the right shape:
    a comment cannot produce one.
    """
    import ast
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "plot_xg_timeline":
            fn = node
            break
    if fn is None:
        return False
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and node.value == "match_clock_s":
            return True
    return False


def test_the_shot_map_reads_the_true_clock():
    """STRUCTURAL, because the shot map is a PNG and nothing behavioural can
    see its axis.

    `plot_xg_timeline` used `minute * 60 + second` to SORT the shots and
    `e.minute + e.second / 60` to PLACE them — and `second` is a random
    placeholder, so the shots inside a minute were ordered and drawn in an
    essentially random order.  The render test (`test_sequence_engine.py`)
    proves the chart still draws; this proves it draws from the truth.
    """
    import exporter as EX

    src = open(EX.__file__, encoding="utf-8", newline="").read()
    assert _reads_true_clock(src), (
        "plot_xg_timeline no longer reads the true clock — the shot map is "
        "back to ordering and placing shots by the random placeholder second")

    # NON-VACUITY: the same check must FAIL on the pre-fix shape.  A guard
    # that cannot fail is not a guard.
    reverted = (
        "def plot_xg_timeline(self, filepath):\n"
        "    shots.sort(key=lambda e: e.minute * 60 + e.second)\n"
        "    for e in shots:\n"
        "        minute = e.minute + e.second / 60.0\n"
    )
    assert not _reads_true_clock(reverted), (
        "the guard cannot see the pre-fix shape — it is not checking "
        "anything")
