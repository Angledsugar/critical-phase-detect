"""Env + DemoSource protocols. Paper §6.1.

Env: episodic environment (typically a gymnasium.Env wrapper).
DemoSource: provides expert demonstrations for G2 labeler init (§6.1.3).
"""
from __future__ import annotations

from typing import Any, Protocol

from cpd.core.trajectory import Trajectory


class Env(Protocol):
    """Episodic environment. Concrete impls typically wrap gymnasium.Env."""

    @property
    def task_id(self) -> str: ...

    def reset(self) -> Any:
        """Return initial observation."""
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """(obs, reward, done, info)."""
        ...


class DemoSource(Protocol):
    """Source of expert demos for G2 labeler init. Paper §6.1.3.

    For LIBERO this wraps the official benchmark's demo loader.
    """

    def demos(self, task_id: str, n: int) -> list[Trajectory]:
        """Return n expert demo trajectories for the given task."""
        ...
