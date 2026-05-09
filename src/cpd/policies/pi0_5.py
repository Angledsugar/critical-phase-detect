"""π_0.5 (openpi) wrapped as a frozen base Policy. Paper §6.1.2.

Refinement happens via Pi0RefinedPolicy (PPO + AdapterHead), not here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_INSTALL_HINT = (
    "openpi is required for Pi05Policy. Install with `uv sync --extra vla` "
    "(see pyproject.toml [project.optional-dependencies] vla)."
)


class Pi05Policy:
    """Frozen π_0.5 base policy.

    Args:
        model_id: openpi config name (e.g. "pi05_libero").
        checkpoint_dir: local or remote (gs://) checkpoint path.
        device: torch device for the underlying model.
        default_prompt: optional prompt injected when obs lacks one.
    """

    def __init__(
        self,
        model_id: str = "pi05_libero",
        checkpoint_dir: str | Path | None = None,
        *,
        device: str = "cuda",
        default_prompt: str | None = None,
    ) -> None:
        try:
            from openpi.policies import policy_config as _policy_config
            from openpi.training import config as _config
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        if checkpoint_dir is None:
            raise ValueError(
                f"checkpoint_dir is required for {model_id}; pass a local dir or gs:// uri."
            )

        train_config = _config.get_config(model_id)
        self._policy = _policy_config.create_trained_policy(
            train_config,
            str(checkpoint_dir),
            default_prompt=default_prompt,
            pytorch_device=device,
        )
        self.model_id = model_id
        self.device = device

    def predict(self, obs: Any) -> torch.Tensor:
        """Sample actions from openpi for a single (un-batched) obs dict."""
        out = self._policy.infer(obs)
        return torch.as_tensor(out["actions"])

    def refine_step(self, batch: Any) -> dict[str, float]:  # noqa: ARG002
        """Frozen base — refinement flows through Pi0RefinedPolicy (PPO + adapter)."""
        return {}
