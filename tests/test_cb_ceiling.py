"""CB attacking ceiling — the clamp AND the gate that lets it bind.

Two things are pinned here, and they are different claims:

  1. BEHAVIOUR. `PositionEngine.cb_advance_clamp` clamps depth in both
     directions of play, passes anything at or below the ceiling untouched,
     and leaves y alone.

  2. REACH. The clamp is applied in EVERY phase, not only in possession.
     This is the half that matters. The clamp existed, was live-called and
     bound hard (`_diag_cb_ceiling.py`: 4,400-8,653 real clamps per CB per
     match) and CBs STILL reached nx 97-105, because the call site was gated
     `pos == "CB" and has_ball`. Out of possession there was no ceiling at
     all, and the off-ball anchor is the player's own minute-start position,
     so a CB who stepped up ratcheted deeper and never returned.

     A behaviour test on `cb_advance_clamp` cannot see that -- the gate lives
     in `match_engine.py`, one file away. So (2) is an AST guard over the real
     call site, and it asserts NON-VACUITY first: if the call is ever renamed
     or removed the guard fails loudly instead of passing on nothing.
"""
from __future__ import annotations

import ast
import os

from position_engine import PositionEngine, PITCH_X

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCH_ENGINE = os.path.join(ROOT, "match_engine.py")


# ── 1. BEHAVIOUR ────────────────────────────────────────────────────────

def _pe() -> PositionEngine:
    return PositionEngine()


def test_clamp_passes_through_below_the_ceiling():
    pe = _pe()
    for tx in (0.0, 10.0, 24.0, float(pe.CB_MAX_ADVANCE_NX)):
        assert pe.cb_advance_clamp(tx, True) == tx


def test_clamp_binds_above_the_ceiling_attacking_right():
    pe = _pe()
    assert pe.cb_advance_clamp(90.0, True) == pe.CB_MAX_ADVANCE_NX
    assert pe.cb_advance_clamp(70.0, True) == pe.CB_MAX_ADVANCE_NX


def test_clamp_binds_above_the_ceiling_attacking_left():
    pe = _pe()
    # Attacking left, depth is measured from x=105: raw x=15 is depth 90.
    assert pe.cb_advance_clamp(15.0, False) == PITCH_X - pe.CB_MAX_ADVANCE_NX
    assert pe.cb_advance_clamp(20.0, False) == PITCH_X - pe.CB_MAX_ADVANCE_NX
    # Raw x=60 is depth 45 -- below the ceiling, untouched.
    assert pe.cb_advance_clamp(60.0, False) == 60.0


def test_clamp_is_depth_only():
    """y has no ceiling -- re-stamping a CB's width would be a second,
    unrequested opinion about shape (same rule rest_defence_clamp follows)."""
    pe = _pe()
    import inspect
    sig = inspect.signature(pe.cb_advance_clamp)
    assert list(sig.parameters) == ["tx", "attacks_right"]


def test_ceiling_is_past_the_halfway_line_only_just():
    pe = _pe()
    assert PITCH_X / 2.0 < pe.CB_MAX_ADVANCE_NX < PITCH_X / 2.0 + 10.0


# ── 2. REACH (AST guard over the live call site) ────────────────────────

def _find_clamp_calls(tree: ast.AST):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "cb_advance_clamp":
                out.append(node)
    return out


def _ancestors(tree: ast.AST, target: ast.AST):
    chain = []
    stack = [(tree, ())]
    while stack:
        node, anc = stack.pop()
        if node is target:
            return anc
        for child in ast.iter_child_nodes(node):
            stack.append((child, anc + (node,)))
    return None


def _test_names(node: ast.AST):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def test_the_clamp_is_called_from_match_engine_at_all():
    """Non-vacuity, asserted before anything is claimed about the gate."""
    with open(MATCH_ENGINE, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    calls = _find_clamp_calls(tree)
    assert len(calls) >= 1, "cb_advance_clamp is not called from match_engine.py"


def test_the_clamp_is_not_gated_on_possession():
    """THE GUARD. The call site must not be conditional on `has_ball`.

    Gated on possession there is no ceiling out of possession, which is the
    exact defect this file exists to stop recurring -- and it is invisible to
    every behavioural test because it lives in the caller.
    """
    with open(MATCH_ENGINE, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    calls = _find_clamp_calls(tree)
    assert calls, "cb_advance_clamp is not called from match_engine.py"

    offenders = []
    for call in calls:
        for anc in _ancestors(tree, call) or ():
            if isinstance(anc, ast.If) and "has_ball" in _test_names(anc.test):
                offenders.append(anc.lineno)
    assert not offenders, (
        "cb_advance_clamp is possession-gated at match_engine.py line(s) "
        f"{offenders} -- the ceiling then does not apply out of possession"
    )
