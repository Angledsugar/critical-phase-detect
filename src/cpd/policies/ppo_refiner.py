"""PPO refiner for VLA base policies. Paper §6.1.2 + §3.5.

AdapterHead is a low-rank residual added to the frozen base action.
Pi0RefinedPolicy runs ONE clipped-ratio PPO update per call to refine_step,
using r(t) = reward_fn.per_step(traj.latents[t]) as the dense per-step reward.

Single-trajectory on-policy update: this is per-task RL refinement, not full RL.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from cpd.core.trajectory import Trajectory


def _to_feat(x: Any) -> torch.Tensor:
    """Best-effort coercion of an obs into a 1-D float feature tensor.

    Supports raw tensors, np arrays, and dicts with a "feat"/"latent"/"state" key.
    """
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, dict):
        for k in ("feat", "latent", "state"):
            if k in x:
                return _to_feat(x[k])
        raise KeyError(f"obs dict has none of feat/latent/state; got keys {list(x.keys())}")
    else:
        t = torch.as_tensor(x)
    return t.reshape(-1).float()


class AdapterHead(nn.Module):
    """Low-rank residual: action ← base_action + W_up @ relu(W_down @ obs_feat).

    Zero-initialized W_up so the initial residual is exactly 0 — i.e. the refined
    policy's output equals the base policy's output before any PPO step.
    """

    def __init__(self, in_dim: int, action_dim: int, rank: int = 8) -> None:
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, action_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.up(torch.relu(self.down(feat)))

    def dist(self, feat: torch.Tensor, base_action: torch.Tensor) -> torch.distributions.Normal:
        mean = base_action + self.forward(feat)
        std = torch.exp(self.log_std).clamp_min(1e-4)
        return torch.distributions.Normal(mean, std)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    epochs: int = 4
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0


def _gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """Generalized advantage estimation; bootstraps with V=0 at terminal step."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else torch.zeros_like(values[0])
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv


class Pi0RefinedPolicy:
    """Frozen base + low-rank adapter, refined by single-traj PPO.

    Args:
        base: frozen Policy (e.g. Pi0Policy / Pi05Policy / any Policy implementer).
        adapter: AdapterHead matching base's obs feature and action dims.
        reward_fn: object exposing `per_step(latent: Tensor) -> Tensor` (CPD reward).
        config: PPOConfig (clip_eps, gamma, lam, lr, epochs).
    """

    def __init__(
        self,
        base: Any,
        adapter: AdapterHead,
        reward_fn: Any,
        config: PPOConfig | None = None,
    ) -> None:
        self.base = base
        self.adapter = adapter
        self.reward_fn = reward_fn
        self.config = config or PPOConfig()
        self.opt = torch.optim.Adam(self.adapter.parameters(), lr=self.config.lr)
        self._value = nn.Linear(adapter.down.in_features, 1)
        self.opt.add_param_group({"params": self._value.parameters()})

    def predict(self, obs: Any) -> torch.Tensor:
        feat = _to_feat(obs)
        base_action = torch.as_tensor(self.base.predict(obs)).float()
        with torch.no_grad():
            residual = self.adapter(feat)
        return base_action + residual.reshape(base_action.shape)

    def refine_step(self, batch: Trajectory) -> dict[str, float]:
        """One PPO update on a single trajectory using CPD per-step reward."""
        cfg = self.config
        latents = batch.latents.float()
        T = latents.shape[0]
        if T == 0:
            return {"loss": 0.0, "skipped": 1.0}

        # Stack base actions; require them on the trajectory (replay) — fall back to
        # asking the base policy for each raw_state if actions are absent.
        if len(batch.actions) == T:
            base_actions = torch.stack(
                [torch.as_tensor(a).float().reshape(-1) for a in batch.actions]
            )
        else:
            base_actions = torch.stack(
                [
                    torch.as_tensor(self.base.predict(s)).float().reshape(-1)
                    for s in batch.raw_states
                ]
            )

        with torch.no_grad():
            rewards = torch.as_tensor(self.reward_fn.per_step(latents)).float().reshape(-1)
            old_dist = self.adapter.dist(latents, base_actions)
            sampled = old_dist.sample()
            old_logp = old_dist.log_prob(sampled).sum(-1)
            values_old = self._value(latents).squeeze(-1)
            advantages = _gae(rewards, values_old, cfg.gamma, cfg.lam)
            returns = advantages + values_old
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        last_loss = torch.zeros(())
        for _ in range(cfg.epochs):
            new_dist = self.adapter.dist(latents, base_actions)
            new_logp = new_dist.log_prob(sampled).sum(-1)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv_norm
            clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_norm
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_pred = self._value(latents).squeeze(-1)
            value_loss = 0.5 * (returns - value_pred).pow(2).mean()
            entropy = new_dist.entropy().sum(-1).mean()
            loss = policy_loss + value_loss - cfg.entropy_coef * entropy

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.adapter.parameters()) + list(self._value.parameters()),
                cfg.max_grad_norm,
            )
            self.opt.step()
            last_loss = loss.detach()

        with torch.no_grad():
            kl = (old_logp - new_logp).mean().item()

        return {
            "loss": float(last_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(kl),
            "reward_mean": float(rewards.mean().item()),
        }


def make_adapter(
    in_dim: int, action_dim: int, rank: int = 8
) -> AdapterHead:
    """Convenience factory matching the configs/policy/ppo_refiner.yaml schema."""
    return AdapterHead(in_dim=in_dim, action_dim=action_dim, rank=rank)


__all__ = [
    "AdapterHead",
    "PPOConfig",
    "Pi0RefinedPolicy",
    "make_adapter",
]


# Type: reward_fn protocol noted for clarity (no runtime dep).
RewardFn = Callable[[torch.Tensor], torch.Tensor]
