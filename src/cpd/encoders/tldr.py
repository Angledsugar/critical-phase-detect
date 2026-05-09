"""TLDR (Temporal-Distance Latent Representation) encoder. Paper §3.3.

φ : raw state → ℝ^d such that ||φ(s_a) − φ(s_b)|| tracks |t_a − t_b| along
expert demos. Trained with triplet contrastive loss (cpd.encoders.tldr_train).
After training, φ(s_T) gives a "task progress" embedding consumed by G2.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cpd.core.trajectory import RawState


def _to_tensor(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


class TLDREncoder(nn.Module):
    """MLP encoder with LayerNorm + ReLU stack, final linear projection.

    Conforms to PhiEncoder protocol. encode() handles both batched and
    unbatched inputs (1-D state -> (latent_dim,), 2-D batch -> (B, latent_dim)).
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self._state_dim = int(state_dim)
        self._latent_dim = int(latent_dim)
        self._hidden_dim = int(hidden_dim)
        self._num_layers = int(num_layers)

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def hyperparams(self) -> dict[str, int]:
        return {
            "state_dim": self._state_dim,
            "latent_dim": self._latent_dim,
            "hidden_dim": self._hidden_dim,
            "num_layers": self._num_layers,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def encode(self, raw_states: RawState | Sequence[RawState]) -> torch.Tensor:
        """Encode raw state(s) → latent.

        Accepts a single state (1-D, shape (state_dim,)) or a batch
        (2-D, shape (T, state_dim)). Returns matching rank in latent space.
        Inputs may be Tensor, numpy array, or list/tuple of scalars.
        """
        param = next(self.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        dtype = param.dtype if param is not None else torch.float32
        x = _to_tensor(raw_states, device=device, dtype=dtype)
        # Allow list-of-list / list-of-array batch input.
        if x.ndim == 0:
            x = x.unsqueeze(0)
        squeezed = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeezed = True
        z = self.forward(x)
        if squeezed:
            z = z.squeeze(0)
        return z

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "hyperparams": self.hyperparams},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: Any = "cpu") -> TLDREncoder:
        ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
        model = cls(**ckpt["hyperparams"])
        model.load_state_dict(ckpt["state_dict"])
        return model
