"""
Reproducibility of the match engine — what holds, and what does not.
=================================================================

Findings behind these tests, all measured rather than assumed
(``physics/probe_determinism.py``):

1. **The engine replays exactly, given identical inputs.** Three runs of the
   same seed in one process, and three *separate processes*, all produced a
   byte-identical 3,283-event timeline, sha ``e59c7889d1dbad5a``.

2. **It survives ``PYTHONHASHSEED`` far less well.** The same fixture under
   ``PYTHONHASHSEED=1`` produced 3,024 events — 8% less football. This traces
   to exactly three ``hash()``-on-string sites, and those are frozen on
   purpose: changing them changes the live 26/27 season.

3. **The production squad path does not replay across processes.** Three fresh
   interpreters, same seed, same ``PYTHONHASHSEED``, real rosters: 3,057 /
   3,060 / 3,095 events. Player *positions* differ from minute 1:40 while the
   iteration *order* of those players is identical — so it is allocation-
   dependent (``id()``, or an unseeded generator), not set-order or ``hash()``.

   A control experiment isolates this: ``roster_loader.build_matchday_squad``
   returns byte-identical squads when called three times in a row, so the
   loader is not the source.

Point 3 is the open bug. It is **not** the "minute 28 off-ball divergence"
previously recorded — that diagnosis was wrong, and these tests exist partly so
the corrected version is the one that survives.

The tests below are AST checks rather than match simulations. A full-match
reproducibility test costs ~160 s for the two runs it needs, which does not
belong in the fast suite; the scripts in ``physics/`` are the tool for that,
and the checks here stop the *known* causes from coming back.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(r"D:\PLOFA\plofa")

#: Modules on the live match path. Not the whole repo — the evolution and
#: analysis tooling is allowed to be loose, the season is not.
MATCH_PATH = (
    "match_engine.py", "event_chain.py", "possession_physics.py",
    "geometry_engine.py", "ball_vision.py", "perception.py", "chance_creation.py",
    "football_brain.py", "manager_brain.py", "offball_brain_wiring.py",
    "position_engine.py", "weather_physics.py", "squad_manager.py",
    "decision_brain.py", "pitch_control.py",
)


def _tree(name: str) -> ast.Module:
    return ast.parse((ROOT / name).read_text(encoding="utf-8"))


# ── the PYTHONHASHSEED dependency, pinned ─────────────────

def test_the_hash_seed_dependent_sites_are_still_exactly_three():
    """``hash()`` on a string depends on PYTHONHASHSEED, so every such call on
    the match path is a reproducibility hazard and a place where a 26/27
    calibration quietly depends on the environment.

    Three are expected, all deliberate:

      ``ball_vision.py``    perception seed for a vision query
      ``perception.py``  x2 the same, in the two draw paths

    ``perception.py`` already carries a comment explaining that new code should
    use ``zlib.crc32`` instead and that these were left alone because changing
    them changes the live season. This test does not demand they be fixed — it
    demands that the count stays *known*, so a fourth one shows up as a failure
    rather than as a mystery divergence months later.
    """
    found = []
    for name in MATCH_PATH:
        src = (ROOT / name).read_text(encoding="utf-8")
        for node in ast.walk(_tree(name)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "hash" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found.append(f"{name}:{node.lineno} hash(<string literal>)")
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "hash" and node.args
                  and isinstance(node.args[0], (ast.Name, ast.Attribute))):
                found.append(f"{name}:{node.lineno} hash(<name/attr>)")
    # Literal-argument calls are constants and harmless; only dynamic ones count.
    dynamic = [f for f in found if "<string literal>)" not in f]
    assert len(dynamic) == 3, (
        f"expected 3 dynamic hash() calls on the match path, found "
        f"{len(dynamic)}: {dynamic}")
    assert sorted(dynamic) == sorted([
        "ball_vision.py:124 hash(<name/attr>)",
        "perception.py:422 hash(<name/attr>)",
        "perception.py:519 hash(<name/attr>)",
    ]), f"the hash-dependent sites moved: {dynamic}"


def test_the_perception_module_explains_why_it_uses_crc32():
    """The next person to add a draw should follow the existing note, not
    rediscover the trap. This asserts the warning is still in the source."""
    src = (ROOT / "perception.py").read_text(encoding="utf-8")
    assert "PYTHONHASHSEED" in src, (
        "perception.py no longer documents its hash() dependency — the note is "
        "the only thing stopping the next draw from reintroducing it")


# ── unseeded generators, pinned ──────────────────────────

def test_no_unseeded_numpy_generator_on_the_match_path():
    """``np.random.default_rng()`` with no argument seeds from OS entropy, so
    the object it produces differs in every process.

    Four such calls exist in ``football_brain.py`` and two in
    ``manager_brain.py`` — all inside ``mutate()`` and ``crossover()``, which
    are evolution operators and never run during a match. That is a legitimate
    reason for them to be loose, but it is invisible: nothing at the call site
    says "this is fine because this is an evolution operator". So the list is
    pinned, and a new unseeded generator in a match-path function fails.
    """
    allowed = {
        ("football_brain.py", "mutate"), ("football_brain.py", "crossover"),
        ("manager_brain.py", "mutate"), ("manager_brain.py", "crossover"),
    }
    offenders = []
    for name in MATCH_PATH:
        tree = _tree(name)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(fn):
                if not (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "default_rng" and not sub.args):
                    continue
                if (name, fn.name) in allowed:
                    continue
                offenders.append(f"{name}:{sub.lineno} in {fn.name}()")
    assert not offenders, (
        "unseeded np.random.default_rng() on the match path — it reads OS "
        f"entropy, so this object differs every process: {offenders}")


def test_brain_constructors_default_to_no_seed():
    """``seed: Optional[int] = None`` forwarded to ``default_rng`` means an
    unseeded brain, because ``default_rng(None)`` reads OS entropy.

    The default is only safe because nothing in the match path constructs a
    brain that way — brains are either loaded from ``brains_offball/*.json``
    or built by ``.random(seed=...)``. This pins that, so a future caller
    cannot reach for the constructor and inherit an unseeded generator.
    """
    tree = _tree("football_brain.py")
    defaults_to_none = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for a in (*fn.args.args, *fn.args.kwonlyargs):
            if a.annotation is not None and "None" in ast.unparse(a.annotation):
                defaults_to_none.add((fn.name, a.arg))

    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for sub in ast.walk(fn):
            if not (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "default_rng" and sub.args):
                continue
            arg = sub.args[0]
            if isinstance(arg, ast.Name) and (fn.name, arg.id) in defaults_to_none:
                offenders.append(
                    f"{fn.name}(): default_rng({arg.id}) where {arg.id} "
                    f"defaults to None -> OS entropy")
    assert not offenders, offenders


# ── the corrected diagnosis, recorded ────────────────────

def test_the_determinism_report_states_the_corrected_diagnosis():
    """The physics report previously recorded "a player-iteration ordering
    divergence at minute 28, localised to the off-ball path". That was wrong:
    iteration order is stable — what moves is player *positions*, and only on
    the production squad path, and only across processes.

    Left uncorrected, it would keep pointing the next person at the off-ball
    path, which is not where the problem is.
    """
    doc = (ROOT / "docs" / "PHYSICS-LAYER.md").read_text(encoding="utf-8")
    assert "minute 28" in doc, "the old claim should still be visible for history"
    assert "cross-process" in doc or "across processes" in doc, (
        "the report does not mention the cross-process finding, which is the "
        "one that is actually supported by measurement")
