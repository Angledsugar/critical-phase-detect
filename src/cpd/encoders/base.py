"""PhiEncoder protocol — φ : raw state → ℝ^d. Paper §3.3."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from cpd.core.trajectory import RawState


class PhiEncoder(Protocol):
    """Maps raw env states to fixed-dim latent vectors.

    Implementations: TLDR (default), QRL, HILP. See §6.4 ablation #2.
    """

    @property
    def latent_dim(self) -> int: ...

    def encode(self, raw_states: Sequence[RawState]) -> torch.Tensor:
        """Encode T raw states → (T, d) latent tensor."""
        ...
