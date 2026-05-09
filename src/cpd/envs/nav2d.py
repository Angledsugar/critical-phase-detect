"""2D navigation toy environment + greedy demo source."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from cpd.core.trajectory import Trajectory


@dataclass
class Nav2DEnv:
    """Point on [0, 1]^2 navigating to a goal. Implements cpd.envs.base.Env."""

    grid_size: float = 1.0
    goal_pos: tuple[float, float] = (0.9, 0.9)
    step_size: float = 0.05
    max_steps: int = 50
    success_radius: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        self._goal = torch.tensor(self.goal_pos, dtype=torch.float32)
        self._gen = torch.Generator()
        if self.seed is not None:
            self._gen.manual_seed(int(self.seed))
        self._pos = torch.zeros(2, dtype=torch.float32)
        self._steps = 0

    @property
    def task_id(self) -> str:
        return "nav2d"

    def reset(self, *, start: tuple[float, float] | None = None) -> torch.Tensor:
        if start is None:
            self._pos = torch.rand(2, generator=self._gen) * float(self.grid_size)
        else:
            self._pos = torch.tensor(start, dtype=torch.float32)
        self._steps = 0
        return self._pos.clone()

    def step(self, action) -> tuple[torch.Tensor, float, bool, dict]:
        a = torch.as_tensor(action, dtype=torch.float32).reshape(2)
        a = a.clamp(min=-self.step_size, max=self.step_size)
        new_pos = (self._pos + a).clamp(min=0.0, max=float(self.grid_size))
        self._pos = new_pos
        self._steps += 1
        dist = float(torch.linalg.norm(self._pos - self._goal).item())
        success = dist < self.success_radius
        reward = -dist
        done = success or (self._steps >= self.max_steps)
        info = {"success": success, "dist": dist, "steps": self._steps}
        return self._pos.clone(), reward, done, info


@dataclass
class Nav2DDemoSource:
    """Greedy gradient-toward-goal demonstrator for Nav2DEnv."""

    env: Nav2DEnv
    seed: int = 0

    def demos(self, task_id: str = "nav2d", n: int = 20) -> list[Trajectory]:
        if task_id != self.env.task_id:
            raise ValueError(
                f"DemoSource configured for {self.env.task_id!r}, got {task_id!r}"
            )
        gen = torch.Generator()
        gen.manual_seed(int(self.seed))
        goal = torch.tensor(self.env.goal_pos, dtype=torch.float32)
        out: list[Trajectory] = []
        for _ in range(n):
            start = (torch.rand(2, generator=gen) * float(self.env.grid_size)).tolist()
            obs = self.env.reset(start=tuple(start))
            states: list[torch.Tensor] = [obs.clone()]
            actions: list[torch.Tensor] = []
            done = False
            success = False
            while not done:
                direction = goal - obs
                norm = float(torch.linalg.norm(direction).item())
                if norm > 1e-9:
                    action = direction / norm * self.env.step_size
                else:
                    action = torch.zeros(2, dtype=torch.float32)
                obs, _r, done, info = self.env.step(action)
                states.append(obs.clone())
                actions.append(action)
                success = bool(info["success"])
            latents = torch.stack(states, dim=0).to(torch.float32)
            out.append(
                Trajectory(
                    raw_states=[s.tolist() for s in states],
                    latents=latents,
                    actions=(),
                    success=success,
                )
            )
        return out
