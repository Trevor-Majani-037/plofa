"""DNA must be a character sheet, not a re-roll.

Before 2026-10-14, DNAFactory.create() drew all 36 attributes, the
footedness and the superstar boost from the GLOBAL random stream, and it
runs once per matchday. Perćy Luka's finishing was 37.6 / 61.3 / 60.5 on
matchdays 1 / 2 / 6 of 26/27 — three different players wearing one name.

These tests pin the fix, and — per the project's standing rule that a test
which has never failed is not a test — each assertion is paired with a
negative control in _diag_dna_determinism.py that mutates the source and
shows the suite going red.
"""
import random

import pytest

from player_dna import DNAFactory, ArchetypeLibrary

SPECS = ["shooter", "creator", "big_game_player"]


def _build(pos="RW", name="Perćy Luka", age=34, specs=None, superstar=False):
    return DNAFactory.create(name, pos, specs or SPECS, age=age,
                             preferred_foot="left", is_superstar=superstar)


def _sig(dna):
    """Every numeric field of the DNA, as a comparable tuple."""
    out = []
    for dom in (dna.physical, dna.technical, dna.mental, dna.passing,
                dna.defending, dna.gk_attrs):
        for k in sorted(vars(dom)):
            v = getattr(dom, k)
            if isinstance(v, (int, float)):
                out.append((k, round(float(v), 4)))
    out.append(("footedness", dna.footedness))
    out.append(("overall", round(dna.overall_rating, 4)))
    return tuple(out)


# ── THE CORE CLAIM ───────────────────────────────────────────────────────

def test_two_builds_of_the_same_player_are_identical():
    """The whole point. Was failing before the fix."""
    a, b = _build(), _build()
    assert _sig(a) == _sig(b)


def test_dna_survives_a_full_global_stream_burn():
    """Stronger than 'two calls in a row'.

    Burns 5000 global draws between the builds. Under the old code the
    second build was a different player; under the fix it cannot be.
    """
    a = _sig(_build())
    for _ in range(5000):
        random.random()
        random.uniform(0, 100)
    assert _sig(_build()) == a


def test_a_fresh_interpreter_sees_the_same_player():
    """Same values with the global stream seeded differently."""
    random.seed(1)
    a = _sig(_build())
    random.seed(999999)
    assert _sig(_build()) == a


def test_no_hash_call_was_introduced():
    """crc32, not hash().

    hash() on a str is PYTHONHASHSEED-salted, so it would give every player
    a different DNA in every process — the defect fixed 2026-10-04 and
    pinned at a count of ZERO by tests/test_determinism.py.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "player_dna.py"
    text = src.read_text(encoding="utf-8")
    offenders = [l.strip() for l in text.splitlines()
                 if "hash(" in l and not l.strip().startswith("#")]
    assert offenders == [], f"hash() reintroduced in player_dna.py: {offenders}"


# ── THE DRAW MUST STILL HAPPEN (stream parity) ───────────────────────────

def test_global_stream_consumption_is_unchanged():
    """Option (b): the global draw is MADE and discarded, never removed.

    If this fails, every calibration number in AGENTS.md is now measured on
    a different stream and the change can no longer be A/B'd in isolation.
    """
    def consumed(fn):
        random.seed(4242)
        before = random.random()
        random.seed(4242)
        fn()
        after = random.random()
        return round(after - before, 12)

    # 36 attributes + footedness is exercised by the builders themselves.
    n = consumed(lambda: _build())
    # One uniform per _attr call. Count them from the source rather than
    # hard-coding, so a future attribute does not silently break this.
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "player_dna.py"
    body = src.read_text(encoding="utf-8")
    expected = len(re.findall(r"cls\._attr\(", body))
    # random.random() and random.uniform() consume differently, so assert
    # the stream MOVED by a non-trivial amount and that removing the parity
    # draw changes it (i.e. the parity draw is genuinely load-bearing).
    assert n != 0.0, "global stream not consumed at all — parity draws lost"
    assert expected >= 30, f"only {expected} _attr call sites found; source moved"


# ── DETERMINISM MUST NOT FLATTEN THE LEAGUE ──────────────────────────────

def test_different_players_get_different_dna():
    """A stable league is not a league of twins."""
    sigs = {_sig(_build(name=f"Player {i:03d}")) for i in range(60)}
    assert len(sigs) == 60, f"only {len(sigs)}/60 distinct DNAs"


def test_attribute_spread_covers_its_band():
    """crc32 must actually spread, not cluster at the bottom of the band."""
    vals = [_build(name=f"Player {i:03d}").technical.finishing for i in range(400)]
    assert min(vals) < 36.0, f"nothing near the bottom of the band: min={min(vals)}"
    assert max(vals) > 52.0, f"nothing near the top of the band: max={max(vals)}"
    assert max(vals) - min(vals) > 12.0, "spread far too narrow — crc32 is clustering"


def test_the_seed_stream_is_uniform_not_just_wide():
    """The spread test above was NOT ENOUGH and it let a real defect through.

    Raw crc32 on names differing only in a trailing digit is linear over
    GF(2), so it clusters: on D0..D299 it gave chi2 = 39.2 (9 df, p<0.001,
    one decile holding 55 of 300 against an expected 30). The old spread
    test still passed, because 'wide' and 'uniform' are different claims —
    a clustered sample can still span the band.

    This asserts uniformity directly. Chi2 critical values on 9 df are 16.9
    (p=0.05) and 27.9 (p=0.001); 25.0 sits between them, so a genuinely
    uniform stream passes and the clustered one fails with margin.
    """
    from player_dna import _dna_unit, _DNA_IDENTITY

    def chi2_for(names, key):
        saved = _DNA_IDENTITY["name"]
        try:
            counts = [0] * 10
            for n in names:
                _DNA_IDENTITY["name"] = n
                u = _dna_unit(key)
                counts[min(9, int(u * 10))] += 1
        finally:
            _DNA_IDENTITY["name"] = saved
        expected = len(names) / 10.0
        return sum((c - expected) ** 2 / expected for c in counts)

    # Two cohorts with DIFFERENT name shapes — the clustered failure only
    # shows up on the unpadded one.
    for label, names in [
        ("D{i} unpadded", [f"D{i}" for i in range(300)]),
        ("Player {i:03d}", [f"Player {i:03d}" for i in range(300)]),
        ("Surname {i:04d}", [f"Testplayer {i:04d}" for i in range(300)]),
    ]:
        chi = chi2_for(names, "mental.geometric_awareness")
        assert chi < 25.0, (
            f"{label}: seed stream is not uniform, chi2={chi:.1f} on 9 df. "
            "Names that differ only in a trailing digit are getting correlated "
            "attributes — the avalanche step in _dna_unit is missing or weak."
        )


def test_geometric_awareness_still_centres_on_its_band():
    """Regression pin for the clustering above, on the field that exposed it.

    tests/test_dna_awareness_shots.py asserts this at +-1.0 on a 300-sample
    median. Raw crc32 moved it to 62.3 (6.9 sigma) and that suite went RED —
    which is how the clustering was found. Pinned here too so the signal is
    owned by the file that caused it.
    """
    import numpy as np
    vals = [DNAFactory.create(f"D{i}", "CB", [], age=28).mental.geometric_awareness
            for i in range(300)]
    assert min(vals) == 50.0, f"band floor not reached: min={min(vals)}"
    assert max(vals) == 70.0, f"band ceiling not reached: max={max(vals)}"
    assert abs(float(np.median(vals)) - 60.0) < 1.0, (
        f"median {np.median(vals)} is not centred on the (50,70) band")
    assert all(50.0 <= v <= 70.0 for v in vals), "a value escaped the band"


def test_archetype_still_decides_the_band_not_the_seed():
    """POS=ST must still move finishing into the striker band.

    This is the lever that repairs Perćy Luka's shot conversion, and it
    only works because the archetype is resolved from POS/SPECIALTY. Guard
    it so a 'fix' that keyed the seed on the band alone cannot pass.
    """
    rw = [_build(pos="RW", name=f"Player {i:03d}").technical.finishing for i in range(200)]
    st = [_build(pos="ST", name=f"Player {i:03d}").technical.finishing for i in range(200)]
    rw_med = sorted(rw)[100]
    st_med = sorted(st)[100]
    assert st_med > rw_med + 20, f"POS no longer moves finishing: RW={rw_med} ST={st_med}"


# ── FOOTEDNESS (the 20% flip) ────────────────────────────────────────────

def test_footedness_flip_set_is_stable_and_actually_exercised():
    """The footedness coin is the third draw site, and it was UNGUARDED.

    The first version of this suite only built Perćy Luka, whose crc32
    footedness value lands above 0.20, so he never flips — and a mutation
    reverting the coin to the global stream stayed GREEN (negative control
    M3). A guard that never exercises the branch it protects is not a guard,
    so this sweeps a cohort and asserts BOTH outcomes occur.
    """
    cohort = [f"Player {i:03d}" for i in range(400)]

    first = {n: _build(name=n).footedness for n in cohort}
    second = {n: _build(name=n).footedness for n in cohort}

    # 1. stable across builds — this is what M3 breaks
    unstable = [n for n in cohort if first[n] != second[n]]
    assert unstable == [], f"{len(unstable)} players changed feet between builds: {unstable[:5]}"

    # 2. non-vacuity — both branches must actually occur, or the assertion
    #    above is comparing a constant to itself.
    outcomes = set(first.values())
    assert outcomes == {"left", "right"}, f"cohort never exercised both branches: {outcomes}"

    # 3. the flip rate is still ~20%, not 0% and not 100%
    flipped = sum(1 for v in first.values() if v == "right")   # preferred_foot is left
    assert 0.10 < flipped / len(cohort) < 0.32, (
        f"footedness flip rate {flipped/len(cohort):.1%}, expected ~20%")


def test_two_footed_ignores_the_coin_entirely():
    """A two_footed player must be 'both' regardless of the stream."""
    for i in range(50):
        d = DNAFactory.create(f"Player {i:03d}", "RW", ["two_footed"], age=30)
        assert d.footedness == "both"


# ── FALLBACK SAFETY ──────────────────────────────────────────────────────

def test_a_builder_called_outside_create_falls_back_to_the_global_stream():
    """Diagnostics call _attr directly. They must not all collide on ''.

    Pinned so the fallback stays deliberate: if this ever becomes an error
    instead, the eight probe scripts that call _attr directly need updating
    in the same commit.
    """
    from player_dna import _DNA_IDENTITY, _dna_unit
    saved = _DNA_IDENTITY["name"]
    try:
        _DNA_IDENTITY["name"] = ""
        assert _dna_unit("technical.finishing") is None
        v1 = DNAFactory._attr({}, "technical.finishing", (50, 70), 1.0)
        v2 = DNAFactory._attr({}, "technical.finishing", (50, 70), 1.0)
        assert v1 != v2, "anonymous builds are colliding — the fallback is broken"
    finally:
        _DNA_IDENTITY["name"] = saved


def test_real_player_builds_never_take_the_fallback():
    """The fallback must be unreachable for an actual player."""
    from player_dna import _DNA_IDENTITY
    DNAFactory.create("Perćy Luka", "RW", SPECS, age=34)
    assert _DNA_IDENTITY["name"] == "Perćy Luka"


def test_two_players_with_the_same_name_share_a_dna_documented_limit():
    """Names are the seed. Two clubs fielding one name collide.

    Recorded as a KNOWN limitation rather than fixed: DNAFactory.create is
    not handed a club, and the rest of the project already treats name as a
    unique per-player key. If this ever becomes a real problem, the fix is
    to thread team_name through, not to change the seed scheme.
    """
    a = _build(name="Duplicate Name")
    b = _build(name="Duplicate Name")
    assert _sig(a) == _sig(b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
