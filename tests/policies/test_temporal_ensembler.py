#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused tests for the extracted TemporalEnsembler and the wrapper's queue proxy.

The full online-vs-offline equivalence of the underlying algorithm is already
covered by ``tests/policies/test_policies.py::test_act_temporal_ensembler``
(which now exercises the extracted class through ACT's re-export). These tests
add coverage for the extraction itself and for the wrapper's _action_queue
proxy that the relative-action processor and SA both rely on.
"""

import torch

from lerobot.configs.temporal_ensemble import TemporalEnsembleConfig
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.policies.temporal_ensemble_wrapper import _TEActionQueueProxy
from lerobot.policies.temporal_ensembler import TemporalEnsembler


def test_act_re_export_is_extracted_class():
    """ACTTemporalEnsembler should be the same class as TemporalEnsembler.

    Guarantees the extraction is a pure no-op for existing ACT users — any
    code importing ACTTemporalEnsembler from modeling_act gets the same
    object as importing TemporalEnsembler from the new module.
    """
    assert ACTTemporalEnsembler is TemporalEnsembler


def test_reset_returns_to_fresh_state():
    """Feed N chunks → reset → feed M chunks should equal a fresh ensembler fed M chunks.

    Verifies no cross-episode state leakage.
    """
    coeff = 0.01
    chunk_size = 8
    action_dim = 4

    def feed(ens, n_chunks, base=0.0):
        # Use deterministic inputs so the equivalence is bit-exact.
        outputs = []
        for i in range(n_chunks):
            chunk = torch.full((1, chunk_size, action_dim), base + 0.1 * i)
            outputs.append(ens.update(chunk))
        return outputs

    # Path A: fresh, fed M chunks.
    fresh = TemporalEnsembler(coeff, chunk_size)
    a = feed(fresh, 3, base=10.0)

    # Path B: fed N chunks → reset → fed M chunks.
    dirty = TemporalEnsembler(coeff, chunk_size)
    _ = feed(dirty, 5, base=99.0)
    dirty.reset()
    b = feed(dirty, 3, base=10.0)

    assert len(a) == len(b)
    for ya, yb in zip(a, b, strict=True):
        torch.testing.assert_close(ya, yb, rtol=0, atol=0)


def test_skip_aligns_buffer_with_real_time():
    """K>1 cadence: skip(K-1) before update() must align the new chunk with
    the buffer entries representing the same future timesteps.

    Scenario: K=5, chunk_size=10. After update #0 with a chunk that contains a
    unique marker at each future timestep, the buffer should map directly to
    those timesteps. After skip(4) + update #1 with a new chunk, position 0
    (= the popped action) should represent the average of *the same future
    timestep* from both chunks — not a mix of timesteps 1 and 5.
    """
    coeff = 0.0  # uniform weighting → averaging is just (a + b) / 2 for two predictions
    chunk_size = 10
    K = 5  # noqa: N806
    ens = TemporalEnsembler(coeff, chunk_size)

    # Chunk 0: predictions for real times [0..9]. Encode time as the value.
    chunk0 = torch.arange(chunk_size, dtype=torch.float32).reshape(1, chunk_size, 1)
    popped0 = ens.update(chunk0)
    torch.testing.assert_close(popped0, torch.tensor([[0.0]]))  # action for real time 0

    # Wrapper would consume buffer[0..K-2] = times 1..4 directly via indexing.
    # Now we're at wrapper step K=5; the wrapper calls skip(K-1=4) then update.
    ens.skip(K - 1)

    # Chunk 1: predictions for real times [5..14]. Same encoding.
    chunk1 = torch.arange(K, K + chunk_size, dtype=torch.float32).reshape(1, chunk_size, 1)
    popped1 = ens.update(chunk1)

    # The popped action should be the action for real time 5, averaged from:
    #   chunk0[5] (chunk0's prediction for time 5) = 5.0
    #   chunk1[0] (chunk1's prediction for time 5) = 5.0
    # With coeff=0, the running mean of [5, 5] is 5.0. The pre-skip buffer entry
    # at position K-1+1=5 was 5.0 from chunk0; after skip it became position 0;
    # then update integrated chunk1[0]=5.0; pop returns 5.0.
    torch.testing.assert_close(popped1, torch.tensor([[5.0]]))

    # Sanity check that with the OLD (broken) behaviour the popped value would
    # be 1.0 = avg(chunk0[1], chunk1[0]) = avg(1, 5) / 2 = 3.0 — distinctly
    # different. The 5.0 value here proves alignment is correct.


def test_skip_with_nonzero_coeff_preserves_real_time_alignment():
    """Same alignment property must hold for the real ACT coefficient.

    With coeff=0.01 the math is a weighted (not uniform) mean, but the *set*
    of predictions being averaged for a given real-time slot must still be
    predictions for that same real-time slot from different chunks.
    """
    coeff = 0.01
    chunk_size = 10
    K = 3  # noqa: N806
    ens = TemporalEnsembler(coeff, chunk_size)

    # Each chunk: prediction for real time t is encoded as (t * 1.0).
    chunk0 = torch.arange(chunk_size, dtype=torch.float32).reshape(1, chunk_size, 1)
    _ = ens.update(chunk0)
    ens.skip(K - 1)

    chunk1 = torch.arange(K, K + chunk_size, dtype=torch.float32).reshape(1, chunk_size, 1)
    popped1 = ens.update(chunk1)

    # Both chunks predict the same value for real time K (because they encode
    # time directly). Their weighted average must therefore equal K exactly,
    # regardless of the weights — proving alignment, not just smoothing.
    torch.testing.assert_close(popped1, torch.tensor([[float(K)]]), rtol=1e-5, atol=1e-5)


def test_uniform_weighting_returns_constant_input():
    """With coeff=0 (uniform), feeding the same constant chunk forever should
    pop that constant from the very first call onward.

    Sanity check on the weighting math: an average of identical values must
    equal that value, regardless of how many predictions accumulate.
    """
    chunk_size = 8
    const = torch.full((1, chunk_size, 1), 0.42)
    ens = TemporalEnsembler(temporal_ensemble_coeff=0.0, chunk_size=chunk_size)
    # First chunk: popped action equals const[0].
    out0 = ens.update(const)
    torch.testing.assert_close(out0, const[:, 0])
    # After many overlapping passes the popped action should still equal the constant.
    for _ in range(chunk_size * 3):
        out = ens.update(const)
        torch.testing.assert_close(out, const[:, 0], rtol=1e-6, atol=1e-6)


def _MockWrapper(n_action_steps: int):  # noqa: N802
    """Minimal stand-in for TemporalEnsemblePolicyWrapper.

    The proxy only reads _intra_chunk_step and _n_action_steps off its
    wrapper; build the minimum needed instead of instantiating a real
    wrapper (which would require a full PreTrainedPolicy stack).
    """

    class _W:
        _intra_chunk_step = 0

    w = _W()
    w._n_action_steps = n_action_steps
    return w


def test_action_queue_proxy_len_reflects_intra_chunk_step():
    """The proxy's __len__ must return 0 at chunk boundaries (so the relative-
    action processor refreshes its anchor) and positive mid-chunk."""
    w = _MockWrapper(n_action_steps=4)
    q = _TEActionQueueProxy(w)

    w._intra_chunk_step = 0
    assert len(q) == 0  # chunk boundary

    w._intra_chunk_step = 1
    assert len(q) > 0
    w._intra_chunk_step = 3
    assert len(q) > 0


def test_action_queue_proxy_clear_resets_counter():
    """The proxy's clear() must reset _intra_chunk_step to 0 so the next call
    fetches a fresh chunk. This is what SharedAutonomyPolicyWrapper._flush_inner_
    action_queue relies on for SA+TE coexistence."""
    w = _MockWrapper(n_action_steps=4)
    q = _TEActionQueueProxy(w)

    w._intra_chunk_step = 2
    assert len(q) > 0
    q.clear()
    assert w._intra_chunk_step == 0
    assert len(q) == 0


def test_temporal_ensemble_config_defaults():
    cfg = TemporalEnsembleConfig()
    assert cfg.enabled is False
    assert cfg.coeff == 0.01
