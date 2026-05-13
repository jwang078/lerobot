#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Temporal-ensembling wrapper configuration.

Temporal ensembling is a pure inference-time technique introduced by ACT
(Algorithm 2 of https://huggingface.co/papers/2304.13705). At each step the
policy predicts a full chunk of actions, and the executed action is an
exponentially-weighted average of all predictions made about that future
timestep so far. Smooths chunk boundaries and reduces action-prediction jerk.

Originally implemented inline inside ACTPolicy; this config powers the
policy-agnostic ``TemporalEnsemblePolicyWrapper`` so pi05, diffusion, and
any other chunk-predicting policy can opt in via a CLI flag.
"""

from dataclasses import dataclass


@dataclass
class TemporalEnsembleConfig:
    """Configuration for the policy-agnostic temporal-ensemble wrapper.

    The weight assigned to a prediction at age ``i`` (i=0 = oldest) is
    ``w_i = exp(-coeff * i)``. Positive ``coeff`` (the ACT default) weights
    *older* predictions more heavily on the grounds that they've been
    smoothed through more averaging passes and represent a settled estimate.
    """

    enabled: bool = False
    # Exponential weight coefficient. ACT paper uses 0.01. Set to 0 for a
    # uniform average; negative values weight newer predictions more heavily
    # (typically detrimental — see the docstring of ACTTemporalEnsembler).
    coeff: float = 0.01
    # ACT-only opt-in: by default, ACT checkpoints with a non-None
    # ``temporal_ensemble_coeff`` use the legacy inline path and this wrapper
    # is skipped (see the deprecation warning in ``ACTConfig.__post_init__``).
    # Set ``force_act_to_wrapper_mode=True`` to apply this wrapper anyway —
    # primarily useful for A/B-testing wrapper equivalence on ACT. The inline
    # ensembler still runs inside ACT.select_action (its output is discarded
    # by the wrapper), so this is correct but slightly wasteful.
    force_act_to_wrapper_mode: bool = False
    # Stochastic-policy diagnostic. For policies that sample fresh noise on
    # every ``predict_action_chunk`` (pi05 flow-matching, diffusion), the
    # successive chunks fed to the ensembler are independent draws from the
    # action distribution. On multimodal tasks this causes the ensembler's
    # average to fall between modes — actions point nowhere useful and the
    # rollout makes slow / wrong-direction progress (no flailing, just mush).
    #
    # When ``pin_noise=True``, the wrapper samples one noise tensor (shape
    # inferred from the inner config's ``chunk_size`` + ``max_action_dim``)
    # and reuses it across every ``predict_action_chunk`` call for the next
    # ``chunk_size`` outer steps, then re-samples and repeats. Within one
    # window the ensembler smooths along a single coherent flow trajectory;
    # at window boundaries the noise shifts, which lets the policy pick a
    # different mode and escape absorbing states. ``chunk_size`` is the
    # natural unit because the TE buffer holds ~chunk_size past predictions,
    # so within any window all contributing predictions came from the same
    # noise — boundary jumps are minimally visible after ensembler smoothing.
    # Noise is also reset on ``reset()`` so each episode starts fresh.
    #
    # Has no effect on deterministic policies (ACT) — ``predict_action_chunk``
    # ignores any ``noise`` kwarg.
    pin_noise: bool = False
