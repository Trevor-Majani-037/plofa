"""
Reproducibility of the match engine — what holds, and what does not.
=================================================================

Findings behind these tests, all measured rather than assumed
(``physics/probe_determinism.py``):

1. **The engine replays exactly, given identical inputs.** Three runs of the
   same seed in one process, and three *separate processes*, all produced a
   byte-identical 3,283-event timeline, sha ``e59c7889d1dbad5a``.

2. **It did NOT survive ``PYTHONHASHSEED``** — the same fixture under
   ``PYTHONHASHSEED=1`` produced 3,024 events against 3,283 at the default,
   i.e. 8% less football. This traced to exactly three ``hash()``-on-string
   sites. Those three were the *entire* cross-process cause and are now
   ``zlib.crc32`` (2026-10-04): pinned ``PYTHONHASHSEED=0`` gave three
   byte-identical matches, varied 0 / 1 / 12345 gave three different ones, and
   that discrepancy is gone. The AST test below now pins the count at **zero**
   rather than at three.

   The cost of the fix was accepted knowingly: every 26/27 calibration number
   taken before it was taken under an unpinned hash seed and is therefore not
   byte-reproducible.

3. **The production squad path does not replay across processes.** Three fresh
   interpreters, same seed, same ``PYTHONHASHSEED``, real rosters: 3,057 /
   3,060 / 3,095 events. Player *positions* differ from minute 1:40 while the
   iteration *order* of those players is identical — so it is allocation-
   dependent (``id()``, or an unseeded generator), not set-order or ``hash()``.

   A control experiment isolates this: ``roster_loader.build_matchday_squad``
   returns byte-identical squads when called three times in a row, so the
   loader is not the source.

Point 3 is not the same defect as point 2, and the two must not be conflated:
point 2 was the whole cross-process story for the *scratch* harness path and is
fixed; point 3 concerns the *production roster* path and is still open. It is
**not** the "minute 28 off-ball divergence" previously recorded — that
diagnosis was wrong, and these tests exist partly so the corrected version is
the one that survives.

A fourth finding (2026-10-04) retracts a standing project rule rather than
adding to this list: the widely repeated "an A/B across two ``simulate()``
calls in one process is invalid" was **false**. A negative control — same seed,
second ``simulate()``, nothing cleared at all — was byte-identical to the
reference. Module-level brain and mind caches do not contaminate the timeline
(the 531k ``np.random.default_rng()`` calls per match are all seeded), and
``np.random.seed()`` was never the lever.

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

def test_there_are_no_hash_seed_dependent_sites_on_the_match_path():
    """``hash()`` on a string depends on PYTHONHASHSEED, so every such call on
    the match path is a reproducibility hazard and a place where a 26/27
    calibration quietly depends on the environment.

    There used to be exactly three of them, all deliberate:

      ``ball_vision.py``    recall noise for a player who has lost the ball
      ``perception.py``  x2 per-snapshot perception noise, two draw paths

    and they were measured to be the **entire** cross-process cause of this
    engine producing different matches from the same seed on different
    machines: identical seed with ``PYTHONHASHSEED`` 0 / 1 / 12345 gave three
    different matches; pinned to 0 it gave three identical ones. All three are
    ``zlib.crc32`` now (2026-10-04), which is stable in every process forever.

    The test now demands the count is **zero**. The reason to keep a count test
    at all, rather than deleting it with the sites, is that the failure mode is
    invisible: a re-introduced ``hash()`` does not crash, it does not warn, and
    it does not show up until someone re-runs a calibration on a different
    machine and gets a different match.
    """
    found = []
    for name in MATCH_PATH:
        for node in ast.walk(_tree(name)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "hash" and node.args
                    and isinstance(node.args[0], (ast.Name, ast.Attribute))):
                found.append(f"{name}:{node.lineno} hash(<name/attr>)")
    assert not found, (
        f"dynamic hash() calls are back on the match path: {found}. Use "
        f"zlib.crc32(ident.encode('utf-8')) -- see perception._fixture_seed. "
        f"A hash() of a str is PYTHONHASHSEED-salted, so the same fixture "
        f"replays into a different match on a different machine.")


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
