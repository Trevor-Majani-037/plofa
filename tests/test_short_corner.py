"""Pins for the short-corner mechanic.

Why a new file and not additions to `tests/test_set_piece_routines.py`: that
file's 23 tests pass and are worth nothing against this work. It asserts
`corner_delivery(SHORT_CORNER)["target_zone"] == "edge"` and nothing else about
short corners — no `short` flag, no delivery height, no support run. All three
defects found on 2026-10-05 (the height clamp overwriting the ground ball, the
receiver 18-34 m away, nobody coming short) sail straight through it. A suite
that is green and silent about a feature is not evidence about that feature.

These are fast: no match, no engine. The one that earns its keep is the AST
guard, because defect A was exactly "the line I wrote was correct and the next
line undid it", which no behavioural test on `corner_delivery` could see.
"""
import ast
import pathlib

import pytest

from event_chain import (SHORT_CORNER_HEIGHT_M, SHORT_CORNER_SUPPORT_M,
                         SetPieceChain)
from set_piece_routines import SetPieceRoutine, corner_delivery

EVENT_CHAIN = pathlib.Path(__file__).resolve().parent.parent / "event_chain.py"

# Parsed ONCE and cached. The second version of this helper re-parsed inside
# `_is_short_guard`, so it compared node identity between two DIFFERENT trees
# and could never match -- the guard then reported every clamp unguarded.
# Node identity only means anything within one parse.
_CHAIN_FN = None


def _corner_chain_ast():
    global _CHAIN_FN
    if _CHAIN_FN is None:
        tree = ast.parse(EVENT_CHAIN.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SetPieceChain":
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef,
                                         ast.AsyncFunctionDef)) \
                            and sub.name == "_corner_chain":
                        _CHAIN_FN = sub
                        break
    assert _CHAIN_FN is not None, (
        "_corner_chain not found in event_chain.py")
    return _CHAIN_FN


# ── 1. the flag is set on the short routine and nowhere else ──────────────
def test_short_flag_is_set_only_on_the_short_corner():
    assert corner_delivery(SetPieceRoutine.SHORT_CORNER).get("short") is True


@pytest.mark.parametrize("routine", [r for r in SetPieceRoutine
                                     if r is not SetPieceRoutine.SHORT_CORNER])
def test_no_other_routine_claims_to_be_short(routine):
    assert not corner_delivery(routine).get("short")


def test_baseline_is_never_short():
    """`routine=None` must keep perfect baseline behaviour."""
    assert not corner_delivery(None).get("short")


# ── 2. a short pass is a ground pass, and goes a short distance ───────────
def test_short_pass_is_played_along_the_ground():
    assert SHORT_CORNER_HEIGHT_M < 1.0, (
        "a short corner is a ground pass; above 1 m it is a low cross")


def test_short_corner_support_is_in_a_short_pass_band():
    """The receiver runs out to support, so the pass stays 8-18 m."""
    assert 8.0 <= SHORT_CORNER_SUPPORT_M <= 18.0


# ── 3. THE GUARD THAT EARNS THIS FILE ────────────────────────────────────
def _is_short_guard(node):
    """Is this node lexically inside `if not is_short:`?

    The first version of this helper walked `ast.iter_child_nodes(node)`,
    which walks DOWN from the node -- so it searched the clamp's descendants
    for an enclosing `if`, found none, and reported the clamp unguarded. The
    test failed on a real pass. `ast` has no parent pointers, so the guard set
    is built once by collecting every node under a matching `if`.
    """
    for cur in ast.walk(_corner_chain_ast()):
        if not isinstance(cur, ast.If):
            continue
        t = cur.test
        if not (isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not)
                and isinstance(t.operand, ast.Name)
                and t.operand.id == "is_short"):
            continue
        for branch in (cur.body, cur.orelse):
            for sub in ast.walk(ast.Module(body=list(branch), type_ignores=[])):
                if sub is node:
                    return True
    return False


def test_the_height_clamp_is_not_applied_to_a_short_pass():
    """DEFECT A, as a regression guard.

    `corner_height = 0.35` was immediately followed by
    `corner_height = max(2.0, min(3.0, ...))`, which undid it. Measured: a
    forced short corner had apex 3.4-9.1 m. A behavioural test on
    `corner_delivery` cannot see this -- the clamp lives in the chain, not in
    the params -- so the guard has to be structural.
    """
    fn = _corner_chain_ast()
    clamps = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "max"
              and n.args
              and isinstance(n.args[0], ast.Constant)
              and n.args[0].value == 2.0]
    assert clamps, ("the [2.0, 3.0] delivery clamp is gone from _corner_chain "
                    "-- if it was renamed, this guard is now vacuous and the "
                    "height regression it guards can come back")
    unguarded = [n for n in clamps if not _is_short_guard(n)]
    assert not unguarded, (
        f"{len(unguarded)} corner_height clamp(s) are not inside "
        "`if not is_short:` -- a short corner will be lofted back into a cross")


def test_short_corner_branch_exists_in_the_chain():
    """The chain must actually consult the flag, not merely define it."""
    src = EVENT_CHAIN.read_text(encoding="utf-8")
    body = src[src.index("def _corner_chain("):]
    body = body[:body.index("\n    def ", 10)]
    assert "is_short" in body, (
        "_corner_chain never reads `is_short` -- the routine is flagged and "
        "then ignored, which is the exact state SHORT_CORNER was found in")


def test_short_corner_does_not_pack_the_box():
    """A short corner exists BECAUSE the box is empty."""
    src = EVENT_CHAIN.read_text(encoding="utf-8")
    assert "if is_corner and not is_short:" in src, (
        "the packed-box branch is no longer gated on `not is_short`, so a "
        "short corner would run thirteen men into a box it exists to avoid")