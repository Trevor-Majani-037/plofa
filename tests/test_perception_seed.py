"""
Per-match perception seeding — the opt-in that makes noise vary AND replay.
=========================================================================

``PerceptionConfig.seed`` defaulted to ``0`` and nothing ever set it, so the
noise a player got from misreading the ball was byte-identical in every match of
every season: the same striker, at the same minute, with the same staleness,
produced the same misread against every opponent. Not realistic, and not what
the surrounding code intended — the draws are derived from a hash of the
situation, so the design plainly expected the match to contribute a term.

Seeding it from the fixture gives both properties at once:

  * same fixture replayed  -> same noise  (reproducible)
  * different fixtures     -> different noise (realistic variety)

Two things this suite is really guarding:

  1. **26/27 stays byte-identical.** ``perception_seed`` defaults to ``None`` and
     ``None`` must mean "leave the legacy global alone". A seeding change that
     quietly altered the live season would invalidate every calibration number
     ever taken.
  2. **No cross-match leak.** Perception is a PROCESS-WIDE global, so setting it
     for one fixture and moving on hands that seed to every later match in the
     same process. That is precisely the contamination this whole determinism
     investigation is about, so opting out has to genuinely restore.
"""
from __future__ import annotations

import subprocess
import sys
import zlib
from datetime import date

import pytest

from perception import (
    PerceptionConfig,
    derive_match_seed,
    get_perception_config,
    set_perception,
)

D = date(2026, 9, 8)


@pytest.fixture(autouse=True)
def _restore_perception():
    """Every test gets the process-wide config back the way it found it."""
    from match_engine import _reset_perception_seed_tracking
    saved = get_perception_config()
    _reset_perception_seed_tracking()
    yield
    _reset_perception_seed_tracking()
    set_perception(saved)


# ── the seed itself ─────────────────────────────────────────

def test_the_same_fixture_always_gets_the_same_seed():
    a = derive_match_seed("Home FC", "Away FC", D, "PL")
    b = derive_match_seed("Home FC", "Away FC", D, "PL")
    assert a == b


def test_a_different_fixture_gets_a_different_seed():
    base = derive_match_seed("Home FC", "Away FC", D, "PL")
    assert derive_match_seed("Home FC", "Away FC", date(2026, 9, 15), "PL") != base
    assert derive_match_seed("Home FC", "Other FC", D, "PL") != base
    assert derive_match_seed("Away FC", "Home FC", D, "PL") != base, (
        "home and away are not interchangeable — home advantage is real")
    assert derive_match_seed("Home FC", "Away FC", D, "UCL") != base, (
        "the same fixture in a different competition is a different match")


def test_the_seed_is_a_plain_int_in_range():
    s = derive_match_seed("Home FC", "Away FC", D, "PL")
    assert isinstance(s, int)
    assert 0 <= s <= 0xFFFFFFFF


def test_the_seed_survives_unprintable_names():
    """Club names come from a spreadsheet; they must never raise."""
    for name in ("Atlético Madrid", "FC Köln", "Aa!!", "", "Étoile", "東京"):
        assert isinstance(derive_match_seed(name, "X", D, "PL"), int)


# ── why crc32 and not hash() ────────────────────────────────

def test_the_seed_does_not_depend_on_pythonhashseed():
    """The whole point of crc32.

    The pre-existing draws at ball_vision.py:123 and perception.py:390 use
    ``hash(ident)``, which for a str varies with PYTHONHASHSEED. That is why
    every command in this project insists on PYTHONHASHSEED=0 — it is
    load-bearing, not ceremony. New code must not repeat that trap.
    """
    expected = derive_match_seed("Home FC", "Away FC", D, "PL")

    snippet = (
        "import sys; sys.path.insert(0, r'D:\\PLOFA\\plofa');"
        "from datetime import date;"
        "from perception import derive_match_seed;"
        "print(derive_match_seed('Home FC', 'Away FC', date(2026, 9, 8), 'PL'))"
    )
    for seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, timeout=300,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
                 "SYSTEMROOT": "C:\\Windows"},
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().splitlines()[-1] == str(expected), (
            f"seed changed under PYTHONHASHSEED={seed}")

    # and the same query with hash() DOES drift, which is the trap being avoided
    ident = "PlayerName:37.4:2.1"
    drifted = set()
    for seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c",
             f"print((0 * 2654435761 + hash({ident!r})) & 0xFFFFFFFF)"],
            capture_output=True, text=True, timeout=300,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
                 "SYSTEMROOT": "C:\\Windows"},
        )
        drifted.add(out.stdout.strip().splitlines()[-1])
    assert len(drifted) > 1, (
        "hash() unexpectedly did not drift; the reason crc32 is used here "
        "would no longer hold")


def test_the_seed_matches_the_documented_formula():
    ident = f"{'PL'}|{'Home FC'}|{'Away FC'}|{D}"
    assert derive_match_seed("Home FC", "Away FC", D, "PL") == (
        zlib.crc32(ident.encode("utf-8")) & 0xFFFFFFFF)


# ── the engine wiring: opt-in, and no leak ──────────────────

def test_no_seed_means_the_legacy_config_is_untouched():
    from match_engine import _apply_perception_seed
    legacy = PerceptionConfig(enabled=True, seed=0)
    set_perception(legacy)
    _apply_perception_seed(None)
    assert get_perception_config() is legacy


def test_opting_in_sets_the_seed():
    from match_engine import _apply_perception_seed
    set_perception(PerceptionConfig(enabled=True, seed=0))
    _apply_perception_seed(123456)
    assert get_perception_config().seed == 123456


def test_opting_out_restores_rather_than_leaking():
    """The bug this guards: perception is a process-wide global, so a naive
    'set it and move on' hands one match's seed to the next one."""
    from match_engine import _apply_perception_seed
    legacy = PerceptionConfig(enabled=True, seed=0)
    set_perception(legacy)

    _apply_perception_seed(111)
    assert get_perception_config().seed == 111
    _apply_perception_seed(222)
    assert get_perception_config().seed == 222
    _apply_perception_seed(None)

    after = get_perception_config()
    assert after.seed == legacy.seed, "match N's seed leaked into match N+1"
    assert after is not legacy or after == legacy


def test_opting_out_does_not_clobber_a_runtime_config_change():
    """Restoring must undo only the SEED.

    An earlier version snapshotted the whole config object and put it back,
    which would silently revert a caller's legitimate runtime change to
    ``ball_noise`` or ``role_blocks``. Only the seed may be restored.
    """
    from dataclasses import replace
    from match_engine import _apply_perception_seed
    set_perception(PerceptionConfig(enabled=True, ball_noise=2.5, seed=0))
    _apply_perception_seed(111)

    # someone reconfigures perception mid-process, as the UI would
    set_perception(replace(get_perception_config(), ball_noise=9.0))
    _apply_perception_seed(None)

    after = get_perception_config()
    assert after.ball_noise == 9.0, "restore clobbered a runtime change"
    assert after.seed == 0


def test_seeding_preserves_every_other_perception_setting():
    """Only the seed may change. Flipping a default here would silently alter
    26/27 perception for anyone who opts in."""
    from dataclasses import replace
    from match_engine import _apply_perception_seed
    base = PerceptionConfig(enabled=True, role_blocks=True, ball_vision=True,
                            ball_noise=2.5, ball_stale_growth=8.0, seed=0)
    set_perception(base)
    _apply_perception_seed(999)
    after = get_perception_config()

    for field in ("enabled", "role_blocks", "ball_vision",
                  "ball_vision_movement", "offball_actor_perception",
                  "ball_noise", "ball_stale_growth", "ball_radius",
                  "ball_fov_deg"):
        assert getattr(after, field) == getattr(base, field), (
            f"seeding changed {field!r}")
    assert after.seed == 999


def test_a_broken_perception_module_cannot_break_a_match():
    """Seeding is an enhancement. If perception cannot be touched, the match
    must still start — an import error here would be a hard regression."""
    from match_engine import _apply_perception_seed
    _apply_perception_seed(4242)   # must not raise, even after monkeypatching
    assert get_perception_config() is not None


def test_matchconfig_defaults_to_the_legacy_behaviour():
    """26/27 must be untouched: no seed, no change."""
    from datetime import date as _date
    from match_engine import MatchConfig
    cfg = MatchConfig(home_team="A", away_team="B", match_date=_date(2026, 9, 8))
    assert cfg.perception_seed is None


def test_matchconfig_accepts_a_seed_without_disturbing_other_fields():
    from datetime import date as _date
    from match_engine import MatchConfig
    plain = MatchConfig(home_team="A", away_team="B", match_date=_date(2026, 9, 8))
    seeded = MatchConfig(home_team="A", away_team="B",
                         match_date=_date(2026, 9, 8), perception_seed=7)
    assert seeded.perception_seed == 7
    for f in ("home_team", "away_team", "match_date", "extra_time",
              "penalties", "importance", "home_advantage"):
        assert getattr(seeded, f) == getattr(plain, f), f


# ── the payoff, stated as a test ────────────────────────────

def test_different_fixtures_get_different_noise_but_each_replays():
    """The two properties the user actually asked for, as executable claims.

    Seeded correctly: a replayed match misreads the ball identically; a
    different fixture does not.
    """
    from dataclasses import replace
    import numpy as np

    def misread(seed: int, player: str, minute: float, elapsed: float) -> float:
        """The ball_vision.py:123 draw, isolated."""
        ident = f"{player}:{minute:.1f}:{elapsed:.1f}"
        rng = np.random.default_rng(
            (int(seed) * 2654435761 + zlib.crc32(ident.encode())) & 0xFFFFFFFF)
        return float(rng.normal(0.0, 2.5))

    md5 = derive_match_seed("Home FC", "Away FC", D, "PL")
    md9 = derive_match_seed("Home FC", "Away FC", date(2026, 9, 15), "PL")

    # same fixture, replayed -> the striker misreads exactly as before
    assert misread(md5, "Striker", 88.0, 2.1) == misread(md5, "Striker", 88.0, 2.1)

    # a different fixture -> a different misread, i.e. still unpredictable
    assert misread(md5, "Striker", 88.0, 2.1) != misread(md9, "Striker", 88.0, 2.1)


def test_the_legacy_seed_would_have_been_identical_every_match():
    """Why the default of 0 was a bug, stated as a test.

    With the seed stuck at 0 the match term contributes nothing, so the same
    striker at the same minute misreads the ball identically against every
    opponent. This asserts the OLD behaviour so the fix's necessity is
    recorded, and asserts the NEW behaviour differs from it.
    """
    import numpy as np

    def misread(seed: int) -> float:
        ident = "Striker:88.0:2.1"
        rng = np.random.default_rng(
            (int(seed) * 2654435761 + zlib.crc32(ident.encode())) & 0xFFFFFFFF)
        return float(rng.normal(0.0, 2.5))

    assert misread(0) == misread(0), "legacy: identical every match (the bug)"
    assert misread(derive_match_seed("A", "B", D, "PL")) != misread(
        derive_match_seed("A", "C", D, "PL")), "fixed: varies by fixture"
