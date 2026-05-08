"""Policy protocol — VLA inference + RL refinement adapter. Paper §6.1.2."""
from __future__ import annotations

from typing import Any, Protocol


class Policy(Protocol):
    """VLA backbone with PPO refinement hook.

    Implementations: pi0 (openpi), pi0_5 (openpi), ppo (sb3 baseline).
    """

    def predict(self, obs: Any) -> Any:
        """Action from observation."""
        ...

    def refine_step(self, batch: Any) -> dict[str, float]:
        """One PPO update step. Returns metrics (loss, kl, success_rate, ...)."""
        ...
