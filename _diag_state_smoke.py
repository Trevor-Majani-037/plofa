"""Smoke: the new state + memory modules, without a match.

Run:  .venv\\Scripts\\python.exe _diag_state_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import player_state as PS
import memory_features as MF


def main() -> int:
    print("PLAN_COUNT =", PS.PLAN_COUNT)

    # ── the plan's hysteresis ────────────────────────────────────────
    tp = PS.TeamPlan(team="Home")
    assert tp.plan_id == -1, tp.plan_id
    assert tp.propose(1, 0.9) is False, "one confident touch is not a plan"
    assert tp.plan_id == -1, tp.plan_id
    assert tp.propose(1, 0.9) is True, "the second one is"
    assert tp.plan_id == 1, tp.plan_id
    assert tp.switches == 1, tp.switches
    # a different plan needs its own two touches
    assert tp.propose(2, 0.95) is False
    assert tp.propose(2, 0.95) is True
    assert tp.plan_id == 2, tp.plan_id
    print("plan hysteresis OK  switches =", tp.switches)

    # a low-confidence proposal never lands
    tp2 = PS.TeamPlan(team="Away")
    for _ in range(10):
        tp2.propose(3, 0.4)
    assert tp2.plan_id == -1, tp2.plan_id
    print("low-confidence plan refused OK")

    # the plan does not survive a possession change
    assert tp.reset_possession(99) is True
    assert tp.plan_id == -1, tp.plan_id
    assert tp.reset_possession(99) is False
    print("possession reset OK")

    # ── the per-player state ─────────────────────────────────────────
    st = PS.PlayerNeuralState(player_name="CB1")
    h = st.ensure_hidden(8)
    assert h.shape == (8,)
    h[:] = 1.0
    h2 = st.ensure_hidden(8)
    assert h2 is h, "the same size must keep the SAME array"
    h3 = st.ensure_hidden(16)
    assert h3.shape == (16,) and h3.sum() == 0.0, "a size change drops it"
    st.record_touch(7, 12.0, 3.5)
    assert st.touches == 1 and st.last_intent == 7
    print("player state OK")

    store = PS.NeuralStateStore()
    assert store.player("X").player_name == "X"
    store.remove_player("X")
    assert "X" not in store.players
    print("store OK")

    # ── the memory block ─────────────────────────────────────────────
    from cognition.mind import clear_minds, register_mind, new_mind
    clear_minds()
    m = new_mind()
    m.engaged = False
    register_mind("Test Player", m)

    f = MF.memory_features("Test Player")
    assert f.shape == (MF.MEMORY_FEATURE_D,), f.shape
    assert np.allclose(f[:6], 0.0), f
    print("empty memory block:", np.round(f, 3).tolist())

    class Ev:
        def __init__(self, tag, severity=1.0, age=0.0):
            self.tag = tag
            self.severity = severity
            self.age = age
            self.location = None

    m.memory.store("clean_tackle_win", severity=1.0)
    m.memory.store("clean_tackle_win", severity=1.0)
    m.memory.store("lost_duel", severity=1.0)
    m.memory.store("big_chance_missed", severity=1.6)
    m.memory.store("big_chance_scored", severity=2.0)
    m.memory.store("foul_in_box", severity=1.2)
    for e in m.memory.events:
        e.age = 5.0
    m.experience.apply_memory(m.memory)

    f = MF.memory_features("Test Player")
    print("full memory block:", np.round(f, 3).tolist())
    assert f[0] > 0.5, "two wins one loss must read positive"
    assert 0.0 < f[1] < 1.0, f[1]
    assert f[2] > 0.0, f[2]
    assert f[3] > 0.0, f[3]
    assert f[4] > 0.0, f[4]
    assert f[5] > 0.0, f[5]
    print("memory features OK")

    # a player with NO mind is zeros, not an error
    clear_minds()
    f0 = MF.memory_features("Nobody")
    assert np.allclose(f0, 0.0), f0
    print("no-mind block is zeros OK")

    # ── the auto-provisioned mind is INERT for the decision path ────
    clear_minds()
    m2 = MF.ensure_mind("Auto")
    assert m2 is not None and m2.engaged is False, "must be UNENGAGED"
    from cognition.mind import get_mind
    assert get_mind("Auto") is m2
    print("auto-provisioned mind is unengaged OK")

    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
