"""SITUATIONAL CONTEXT — the four blocks the situational brain reads.

WHAT THIS ASSEMBLES
-------------------
::

    state   (24)   the shared block  — the same one the shipped brain reads
    role     (9)   the player's own role block, padded to a fixed width
    memory   (8)   what this player remembers of his match so far
    plan     (6)   the possession objective the team is executing

The role block is padded to a FIXED nine slots because the families emit
seven, eight or nine features and they mean DIFFERENT things in each
slot — which is correct, because the brain is per position.  A fixed
width is what lets one network shape serve all eleven positions.

ZERO IS A VALUE, NOT A HOLE
----------------------------
A player with no mind, a team with no plan and an unknown role all read
as zeros, and the network learns what to do with a zero block.  They are
not errors and they are not silently replaced with a neutral guess: a
zero memory block means "he has done nothing yet", which is a real and
different situation from "he is playing well".
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from role_features import role_block
from situational_brain import MEMORY_D, PLAN_D, ROLE_D, STATE_D


def role_tail(player: Any, x: float, y: float,
              teammates: List[Any], defenders: List[Any],
              position_engine: Any, attacks_right: bool) -> np.ndarray:
    """The role block at a FIXED width of ``ROLE_D``.

    An unknown role (no family) gives zeros — the same convention the
    v2 schema uses for an unknown-role player.
    """
    out = np.zeros(ROLE_D, dtype=np.float64)
    block = role_block(player, x, y, teammates, defenders,
                       position_engine, attacks_right)
    if block is None:
        return out
    arr = np.asarray(block, dtype=np.float64)
    n = min(ROLE_D, arr.shape[0])
    out[:n] = arr[:n]
    return out


def memory_tail(player_name: str) -> np.ndarray:
    """The player's memory block (``memory_features.memory_features``)."""
    try:
        from memory_features import memory_features as _mf

        return _mf(player_name)
    except Exception:
        return np.zeros(MEMORY_D, dtype=np.float64)


def plan_tail(position_engine: Any, team: str) -> np.ndarray:
    """The team's plan summary, or zeros when there is no plan."""
    out = np.zeros(PLAN_D, dtype=np.float64)
    if position_engine is None or not team:
        return out
    try:
        store = position_engine.neural_states
        return store.team(team).summary()
    except Exception:
        return out


def situation_blocks(player: Any, x: float, y: float,
                     teammates: List[Any], defenders: List[Any],
                     position_engine: Any, attacks_right: bool,
                     state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The role, memory and plan blocks, as one call.

    ``state`` is the shared 24-d block, already built by the perception
    layer — it is NOT rebuilt here, so the situational brain reads the
    SAME world the shipped brain would have read.
    """
    role = role_tail(player, x, y, teammates, defenders,
                     position_engine, attacks_right)
    mem = memory_tail(getattr(player, "name", "") or "")
    pl = plan_tail(position_engine, getattr(player, "team", "") or "")
    return role, mem, pl


def assemble(state: np.ndarray, role: np.ndarray, memory: np.ndarray,
             plan: np.ndarray) -> np.ndarray:
    """Concatenate the four blocks into the network's situation vector."""
    role_arr = np.asarray(role, dtype=np.float64).reshape(-1)
    if role_arr.shape[0] < ROLE_D:
        role_arr = np.pad(role_arr, (0, ROLE_D - role_arr.shape[0]))
    elif role_arr.shape[0] > ROLE_D:
        role_arr = role_arr[:ROLE_D]
    return np.concatenate([
        np.asarray(state, dtype=np.float64).reshape(STATE_D),
        role_arr.reshape(ROLE_D),
        np.asarray(memory, dtype=np.float64).reshape(MEMORY_D),
        np.asarray(plan, dtype=np.float64).reshape(PLAN_D),
    ])
