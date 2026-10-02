"""Off-ball actor-honesty tests (2026-09-20): the off-ball vision gate.

The off-ball press brains used to see all 21 other players at TRUE
positions.  ``perceive_offball_actors`` now gates/degrades those lists with
the same cone/range/top-k/noise rules as the on-ball path, measured from
the runner facing the opponent goal.

  1. Disabled -> identity (original lists + engine, byte-identical).
  2. Actors behind the runner / outside range are dropped.
  3. Perceived positions are noisy but deterministic per snapshot.
  4. The production off-ball path routes through the gate (e2e counter).

Run:  python -m tests.test_offball_actor_perception
"""
from __future__ import annotations

import random
from types import SimpleNamespace

from perception import (get_perception_config, set_perception,
                        PerceptionConfig, perceive_offball_actors)


class _Pos:
    def __init__(self, positions):
        self._p = positions

    def get_position(self, name):
        return self._p.get(name, (50.0, 34.0))


class _Player:
    def __init__(self, name, position="LW"):
        self.name = name
        self.position = position
        self.dna = SimpleNamespace(mental=SimpleNamespace(
            vision=60.0, anticipation=60.0, composure=60.0))


def _scene():
    runner = _Player("R", "LW")
    t1 = _Player("T1")          # in FRONT (80,34)
    t2 = _Player("T2")          # BEHIND (20,34)
    d1 = _Player("D1")          # near (52,34)
    d2 = _Player("D2")          # far side (90,40)
    engine = _Pos({
        "R": (50.0, 34.0), "T1": (80.0, 34.0), "T2": (20.0, 34.0),
        "D1": (52.0, 34.0), "D2": (90.0, 40.0),
    })
    return runner, engine, [t1, t2], [d1, d2]


# ─────────────────────────────────────────────────────────────
# 1. Disabled -> identity
# ─────────────────────────────────────────────────────────────

def test_disabled_is_identity():
    old = get_perception_config()
    set_perception(PerceptionConfig(offball_actor_perception=False))
    try:
        runner, engine, team, defs = _scene()
        tms, defs2, eng = perceive_offball_actors(runner, 50, 34, team, defs,
                                                  engine, True)
        assert tms is team and defs2 is defs and eng is engine, (
            "disabled must pass the ORIGINAL objects")
        assert [p.name for p in tms] == ["T1", "T2"]
        assert [p.name for p in defs2] == ["D1", "D2"]
    finally:
        set_perception(old)


# ─────────────────────────────────────────────────────────────
# 2. Gating drops actors behind / out of range
# ─────────────────────────────────────────────────────────────

def test_gate_drops_behind_actors():
    old = get_perception_config()
    set_perception(PerceptionConfig(offball_actor_perception=True,
                                    role_blocks=False, range_max=45.0,
                                    fov_deg=150.0))
    try:
        runner, engine, team, defs = _scene()
        tms, df, eng, meta = perceive_offball_actors(
            runner, 50, 34, team, defs, engine, True, return_meta=True)
        seen = {p.name for p in tms}
        assert "T2" not in seen, "the (20,34) teammate is behind -> unseen"
        assert "T1" in seen, "the (80,34) teammate is ahead -> seen"
        # radius = (8 + (45-8)*0.6) ~= 30 m for this mid-vision runner, so
        # D1 (2 m) is seen and D2 (40 m) is out of range -> dropped.
        seen_defs = {p.name for p in df}
        assert "D1" in seen_defs, df
        assert "D2" not in seen_defs, "D2 at ~40 m exceeds the ~30 m radius"
        assert meta["seen"] == len(tms) + len(df)
        assert meta["radius"] > 0.0
    finally:
        set_perception(old)


# ─────────────────────────────────────────────────────────────
# 3. Noise + determinism
# ─────────────────────────────────────────────────────────────

def test_noise_and_determinism():
    old = get_perception_config()
    # noise_base > 0 and a mid vision player => degraded positions.
    set_perception(PerceptionConfig(offball_actor_perception=True,
                                    role_blocks=False, noise_base=3.0,
                                    range_max=100.0))
    try:
        runner, engine, team, defs = _scene()
        tms, _df, eng, meta = perceive_offball_actors(
            runner, 50, 34, team, defs, engine, True, return_meta=True)
        assert meta["sigma"] > 0.0, "mid vision must be degraded"
        # The noisy engine re-reads degraded positions for the seen actors.
        px1 = eng.get_position("T1")
        px2 = eng.get_position("T1")
        assert px1 == px2, "deterministic per snapshot"
        assert px1[0] != 80.0 or px1[1] != 34.0, "positions must be noisy"
    finally:
        set_perception(old)


# ─────────────────────────────────────────────────────────────
# 4. e2e: production off-ball path routes through the gate
# ─────────────────────────────────────────────────────────────

def test_offball_production_path_uses_actor_gate():
    from pitch_replay import run_scratch_match
    from perception import perceive_offball_actors as real
    old = get_perception_config()
    set_perception(PerceptionConfig(offball_actor_perception=True))
    try:
        calls = []
        def counting(*args, **kwargs):
            out = real(*args, **kwargs)
            calls.append(out)
            return out
        import perception as pc
        pc.perceive_offball_actors = counting
        try:
            run_scratch_match(seed=7, verbose=False)
        finally:
            pc.perceive_offball_actors = real
        assert len(calls) > 0, "production off-ball path never gated actors"
    finally:
        set_perception(old)


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(123)
    tests = [
        test_disabled_is_identity,
        test_gate_drops_behind_actors,
        test_noise_and_determinism,
        test_offball_production_path_uses_actor_gate,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise SystemExit(1)
    print(f"\n{passed}/{len(tests)} off-ball actor-perception tests passed.")