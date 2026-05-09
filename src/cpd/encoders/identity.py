"""Identity φ encoder — passes raw state through as float32 tensor."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from cpd.core.trajectory import RawState


@dataclass
class IdentityEncoder:
    """φ(s) = s. Used by toy envs where the raw state already lives in ℝ^d."""

    latent_dim: int

    def encode(self, raw_states: RawState | Sequence[RawState]) -> torch.Tensor:
        z = torch.as_tensor(raw_states, dtype=torch.float32)
        if z.ndim == 1:
            if z.shape[0] != self.latent_dim:
                raise ValueError(
                    f"IdentityEncoder expected last dim {self.latent_dim}, "
                    f"got shape {tuple(z.shape)}"
                )
        else:
            if z.shape[-1] != self.latent_dim:
                raise ValueError(
                    f"IdentityEncoder expected last dim {self.latent_dim}, "
                    f"got shape {tuple(z.shape)}"
                )
        return z
