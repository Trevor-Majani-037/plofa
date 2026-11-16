"""BPTT must invert the forward pass EXACTLY — for every role width.

WHY THIS EXISTS
---------------
``SituationalBrain.backward`` sliced the situation vector with the MODULE
constant ``ROLE_D`` (19) instead of the width the brain actually uses.  Every
shipped brain uses a NINE-wide role block (``w_q`` is ``(47, 32)``), so the
slice was 57 into a 79-long vector, ``dctx`` came out 22 long instead of 32,
and the GRU gate gradients were written from a truncated vector.  No exception,
no warning, wrong weights.

That is the failure mode this file exists to make impossible.  A gradient check
is the only test that can catch it: it compares the analytic gradient against a
finite difference of the function it claims to be the derivative of, so a
mislabelled slice cannot hide behind a plausible-looking training curve.

Both widths are checked.  The 19-wide case is the one that used to pass, and
keeping it here is deliberate — a fix that breaks the working case is not a fix.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from situational_brain import (  # noqa: E402
    EMBED_D, HIDDEN_D, MEMORY_D, OUTPUT_SIZE, PLAN_D, PLAN_OUT_D, STATE_D,
    TOKEN_D, SituationalBrain,
)


def _role_width(input_size: int) -> int:
    return input_size - STATE_D - MEMORY_D - PLAN_D


def _make(input_size: int, seed: int = 7) -> SituationalBrain:
    return SituationalBrain.random(seed=seed, hidden_size=HIDDEN_D,
                                   input_size=input_size)


def _batch(brain: SituationalBrain, input_size: int, T: int, seed: int = 11):
    rng = np.random.default_rng(seed)
    rw = _role_width(input_size)
    n_tok = 6
    tokens = rng.normal(0, 0.5, (T, n_tok, TOKEN_D))
    mask = np.ones((T, n_tok))
    mask[:, -2:] = 0.0                      # exercise the attention mask
    state = rng.normal(0, 0.5, (T, STATE_D))
    role = rng.normal(0, 0.5, (T, rw))
    memory = rng.normal(0, 0.5, (T, MEMORY_D))
    plan = rng.normal(0, 0.5, (T, PLAN_D))
    return tokens, mask, state, role, memory, plan


def _loss_and_caches(brain, batch, dlogits, dplan, h0):
    """L = <logits, dlogits> + <plan_logits, dplan>, and its caches."""
    tokens, mask, state, role, memory, plan = batch
    logits, plan_logits, _h, caches = brain.forward_sequence(
        tokens, mask, state, role, memory, plan, h0)
    return float((logits * dlogits).sum() + (plan_logits * dplan).sum()), caches


def _numeric_grad(brain, batch, dlogits, dplan, h0, key, eps=1e-5):
    """Central finite difference of the loss w.r.t. one parameter array."""
    arr = brain.w[key]
    flat = arr.ravel()
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        lp, _ = _loss_and_caches(brain, batch, dlogits, dplan, h0)
        flat[i] = orig - eps
        lm, _ = _loss_and_caches(brain, batch, dlogits, dplan, h0)
        flat[i] = orig
        grad[i] = (lp - lm) / (2 * eps)
    return grad.reshape(arr.shape)


@pytest.mark.parametrize("input_size", [47, 57],
                         ids=["role_width_9_shipped", "role_width_19_default"])
def test_backward_matches_finite_differences(input_size):
    brain = _make(input_size)
    T = 3
    batch = _batch(brain, input_size, T)
    rng = np.random.default_rng(3)
    dlogits = rng.normal(0, 0.3, (T, OUTPUT_SIZE))
    dplan = rng.normal(0, 0.3, (T, PLAN_OUT_D))
    h0 = np.zeros(HIDDEN_D)

    _l, caches = _loss_and_caches(brain, batch, dlogits, dplan, h0)
    grads = brain.backward(dlogits, dplan, caches)

    # The output head and one GRU gate weight: if the situation width were
    # sliced wrong, the gate gradients are exactly what would be wrong.
    for key in ("w_o", "b_o", "w_z", "u_n", "w_q", "w_e"):
        num = _numeric_grad(brain, batch, dlogits, dplan, h0, key)
        err = float(np.max(np.abs(grads[key] - num)))
        scale = max(1e-8, float(np.max(np.abs(num))))
        assert err < 1e-5 * max(1.0, scale * 100), (
            f"{key} at role width {_role_width(input_size)}: analytic and "
            f"numeric gradients disagree (max abs err {err:.3e}, "
            f"numeric scale {scale:.3e}). The backward pass is not the "
            f"derivative of the forward pass."
        )


@pytest.mark.parametrize("input_size", [47, 57],
                         ids=["role_width_9_shipped", "role_width_19_default"])
def test_input_gradients_land_on_the_right_blocks(input_size):
    """d_state / d_role / d_memory / d_plan must cover the vector exactly.

    This is the slice bug itself, tested where it lives.  If the blocks are
    misaligned, the role gradient is contaminated with memory values (or vice
    versa) and no amount of finite differencing on PARAMETERS would notice,
    because the parameters are downstream of the misalignment.
    """
    brain = _make(input_size, seed=5)
    T = 2
    batch = _batch(brain, input_size, T, seed=19)
    rng = np.random.default_rng(4)
    dlogits = rng.normal(0, 0.3, (T, OUTPUT_SIZE))
    dplan = rng.normal(0, 0.3, (T, PLAN_OUT_D))
    h0 = np.zeros(HIDDEN_D)

    tokens, mask, state, role, memory, plan = batch
    rw = _role_width(input_size)
    d_state = np.zeros_like(state)
    d_role = np.zeros_like(role)
    d_memory = np.zeros_like(memory)
    d_plan = np.zeros_like(plan)

    _l, caches = _loss_and_caches(brain, batch, dlogits, dplan, h0)
    brain.backward(dlogits, dplan, caches, d_state=d_state, d_role=d_role,
                   d_memory=d_memory, d_plan=d_plan)

    assert d_role.shape == (T, rw)
    assert d_memory.shape == (T, MEMORY_D)
    assert d_plan.shape == (T, PLAN_D)

    # Numeric check on one row of each block.
    eps = 1e-5
    for name, arr, grad, width in (
        ("state", state, d_state, STATE_D),
        ("role", role, d_role, rw),
        ("memory", memory, d_memory, MEMORY_D),
        ("plan", plan, d_plan, PLAN_D),
    ):
        for j in range(width):
            orig = arr[0, j]
            arr[0, j] = orig + eps
            lp, _ = _loss_and_caches(brain, batch, dlogits, dplan, h0)
            arr[0, j] = orig - eps
            lm, _ = _loss_and_caches(brain, batch, dlogits, dplan, h0)
            arr[0, j] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(grad[0, j] - num) < 1e-5 * max(1.0, abs(num) * 100), (
                f"{name} block, element {j}: analytic {grad[0, j]:.6f} vs "
                f"numeric {num:.6f}. The situation vector is being sliced at "
                f"the wrong offsets."
            )


def test_the_shipped_brain_accepts_its_own_gradient():
    """The real ST brain must produce finite, correctly-shaped gradients.

    A shape mismatch here is the 22-vs-32 error that started this: the shipped
    role block is 9 wide, and any code that assumes the module default is
    wrong about the brain that actually plays.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "brains", "ST.json")
    if not os.path.exists(path):
        pytest.skip("shipped ST brain not present")
    brain = SituationalBrain.load(path)
    rw = int(brain.w_q.shape[0]) - STATE_D - MEMORY_D - PLAN_D
    assert rw == 9, f"expected the shipped 9-wide role block, got {rw}"

    T = 4
    tokens = np.zeros((T, 6, TOKEN_D))
    tokens[:, :, 0] = 1.0
    mask = np.ones((T, 6))
    state = np.zeros((T, STATE_D))
    role = np.zeros((T, rw))
    memory = np.zeros((T, MEMORY_D))
    plan = np.zeros((T, PLAN_D))
    dlogits = np.zeros((T, OUTPUT_SIZE))
    dlogits[:, 4] = 1.0                     # push one intent
    dplan = np.zeros((T, PLAN_OUT_D))

    logits, plan_logits, _h, caches = brain.forward_sequence(
        tokens, mask, state, role, memory, plan, np.zeros(HIDDEN_D))
    grads = brain.backward(dlogits, dplan, caches)
    for k, g in grads.items():
        assert g.shape == brain.w[k].shape
        assert np.all(np.isfinite(g)), f"{k} gradient is not finite"
