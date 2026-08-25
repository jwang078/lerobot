"""Positional anchoring ("inpainting") spec for action-chunk samplers.

A ChunkAnchor pins selected positions of a generated action chunk to known
values (typically expert guidance) during the denoising / flow-integration
loop. Both the DDPM sampler (diffusion policy) and the flow-matching sampler
(pi05) consume the same spec, so anchor semantics stay identical across
policy families:

  * ``every_step=True`` (classic inpainting): after EVERY sampler step the
    masked positions are overwritten with the anchor values forward-noised to
    the sampler's next noise level, and after the FINAL step they are clamped
    to the clean values exactly. The emitted chunk is GUARANTEED to equal the
    anchor at the masked positions; the model harmonizes the unmasked
    positions against them.
  * ``every_step=False`` (soft injection): the masked positions are
    overwritten ONCE, after the first sampler step, at the next noise level —
    the model conditions on them for the remaining steps but may drift them.
    No exactness guarantee.

The mask is defined over CHUNK positions (0 = first action of this chunk
build), so placement is arbitrary: prefix (seam continuity with the previous
plan), suffix (guarantee the chunk terminates on the expert corridor —
rejoin-by-construction for blend rollouts), or any other pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ChunkAnchor:
    """Anchor ``values`` at ``mask`` positions of a generated action chunk."""

    values: Tensor  # (B, T, D) clean anchor actions in the chunk frame
    mask: Tensor  # (T, D) bool — per-position, PER-DIMENSION anchor mask
    every_step: bool = True

    @classmethod
    def build(
        cls,
        guidance_chunk: Tensor,
        *,
        prefix_steps: int,
        suffix_steps: int,
        chunk_step: int = 0,
        action_dim: int | None = None,
        every_step: bool = True,
        max_abs: float | None = None,
    ) -> ChunkAnchor | None:
        """Build the prefix/suffix anchor for one chunk, or None if empty.

        ``prefix_steps`` anchors positions ``[chunk_step, chunk_step + k)``
        (the next-to-execute actions at build time); ``suffix_steps`` anchors
        the LAST M positions of the chunk. Positions whose guidance is not
        finite (e.g. padding beyond the demo's end) are dropped from the mask
        so an anchor can never inject NaN/Inf into the sampler. When
        ``max_abs`` is given (the encoded-guidance clip bound), positions
        whose values SATURATE that bound are dropped too: a clipped encode is
        a per-dimension-truncated, direction-distorted target, and pinning it
        exactly would enforce the wrong point — better to leave the position
        unconstrained.
        """
        if prefix_steps <= 0 and suffix_steps <= 0:
            return None
        values = guidance_chunk if action_dim is None else guidance_chunk[:, :, :action_dim]
        n = values.shape[1]
        pos = torch.zeros(n, dtype=torch.bool, device=values.device)
        if prefix_steps > 0:
            lo = max(0, min(int(chunk_step), n))
            pos[lo : min(n, lo + int(prefix_steps))] = True
        if suffix_steps > 0:
            pos[max(0, n - int(suffix_steps)) :] = True
        # Per-DIMENSION validity: a dimension that is non-finite or that
        # saturates the encode clip is left unconstrained at that position
        # while the healthy dimensions stay pinned. (Concretely: a
        # zero-variance gripper dim normalizes to a constant ±1.0 — dropping
        # whole positions for it would disable the anchor entirely.)
        valid = torch.isfinite(values).all(dim=0)  # (T, D)
        if max_abs is not None:
            valid &= (values.abs() < (float(max_abs) - 1e-6)).all(dim=0)
        mask = pos.unsqueeze(-1) & valid  # (T, D)
        if not bool(mask.any()):
            return None
        return cls(values=values, mask=mask, every_step=bool(every_step))

    def trimmed(self, n: int) -> ChunkAnchor | None:
        """The anchor restricted to the first ``n`` chunk positions, or None if empty there."""
        if n >= self.values.shape[1]:
            return self
        mask = self.mask[:n]
        if not bool(mask.any()):
            return None
        return ChunkAnchor(values=self.values[:, :n], mask=mask, every_step=self.every_step)

    def snap(self, chunk: Tensor) -> Tensor:
        """Post-hoc clamp of ``chunk`` (B, T, D') to the anchor at masked positions.

        The non-iterative analog of in-loop anchoring, for glass-box blend
        strategies with no sampler to anchor inside (INTERPOLATE).
        """
        n = min(chunk.shape[1], self.values.shape[1])
        d = min(chunk.shape[2], self.values.shape[2])
        m = self.mask[:n, :d]
        if bool(m.any()):
            chunk = chunk.clone()
            region = chunk[:, :n, :d]
            region[:, m] = self.values[:, :n, :d][:, m].to(chunk.dtype)
            chunk[:, :n, :d] = region
        return chunk
