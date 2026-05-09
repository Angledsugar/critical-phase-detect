"""LIBERO benchmark wrapper. Paper §6.1.

Wraps `libero.libero.envs.OffScreenRenderEnv` as a `cpd.envs.base.Env` and
loads expert hdf5 demos as `Trajectory` instances via `DemoSource`.

Suite naming follows LIBERO upstream: `libero_spatial`, `libero_object`,
`libero_goal`, `libero_10` (= LIBERO-Long, paper main), `libero_90`. The
paper's "libero_long" alias maps to `libero_10`.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from cpd.core.trajectory import Trajectory

_SUITE_ALIASES = {
    "libero_long": "libero_10",
}


def _resolve_suite(suite: str) -> str:
    return _SUITE_ALIASES.get(suite, suite)


def _get_benchmark(suite: str):
    from libero.libero import benchmark as _bench

    resolved = _resolve_suite(suite)
    cls = _bench.get_benchmark(resolved)
    return cls()


class LiberoEnv:
    """LIBERO `OffScreenRenderEnv` adapted to the project's `Env` protocol."""

    def __init__(
        self,
        suite: str = "libero_long",
        task_id: int = 0,
        image_size: int = 128,
        render: bool = False,
    ) -> None:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.suite = suite
        self._task_index = int(task_id)
        self.image_size = int(image_size)
        self.render = bool(render)

        self._benchmark = _get_benchmark(suite)
        n = self._benchmark.get_num_tasks()
        if not (0 <= self._task_index < n):
            raise IndexError(
                f"task_id {self._task_index} out of range [0, {n}) for suite {suite}"
            )
        self._task = self._benchmark.get_task(self._task_index)
        bddl_path = os.path.join(
            get_libero_path("bddl_files"),
            self._task.problem_folder,
            self._task.bddl_file,
        )
        self._bddl_path = bddl_path

        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            has_renderer=render,
        )

    @property
    def task_id(self) -> str:
        return f"{_resolve_suite(self.suite)}/{self._task.name}"

    @property
    def task_name(self) -> str:
        return self._task.name

    @property
    def language_instruction(self) -> str:
        return self._task.language

    def reset(self) -> dict[str, Any]:
        self._env.reset()
        init_states = self._benchmark.get_task_init_states(self._task_index)
        obs = self._env.set_init_state(init_states[0])
        return obs

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict]:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


class LiberoDemoSource:
    """Loads expert hdf5 demonstrations from LIBERO's dataset folder."""

    def __init__(
        self,
        suite: str = "libero_long",
        task_id: int = 0,
        max_demos: int | None = None,
    ) -> None:
        self.suite = suite
        self._task_index = int(task_id)
        self.max_demos = max_demos
        self._benchmark = _get_benchmark(suite)
        n = self._benchmark.get_num_tasks()
        if not (0 <= self._task_index < n):
            raise IndexError(
                f"task_id {self._task_index} out of range [0, {n}) for suite {suite}"
            )
        self._task = self._benchmark.get_task(self._task_index)

    def _hdf5_path(self) -> str:
        from libero.libero import get_libero_path

        rel = self._benchmark.get_task_demonstration(self._task_index)
        return os.path.join(get_libero_path("datasets"), rel)

    def demos(self, task_id: str | None = None, n: int | None = None) -> list[Trajectory]:
        import h5py

        path = self._hdf5_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LIBERO demo hdf5 not found: {path}. Run "
                "`python benchmark_scripts/download_libero_datasets.py` from the LIBERO repo."
            )

        cap = n if n is not None else self.max_demos
        out: list[Trajectory] = []
        with h5py.File(path, "r") as f:
            ep_keys = sorted(
                list(f["data"].keys()), key=lambda k: int(k.split("_")[-1])
            )
            if cap is not None:
                ep_keys = ep_keys[:cap]
            for ep in ep_keys:
                grp = f[f"data/{ep}"]
                actions = np.asarray(grp["actions"][()])
                T = actions.shape[0]
                states = self._collect_states(grp, T)
                out.append(self.to_trajectory(states, list(actions), success=True))
        return out

    @staticmethod
    def _collect_states(grp, T: int) -> list[dict[str, Any]]:
        obs_grp = grp["obs"]
        keys = list(obs_grp.keys())
        states: list[dict[str, Any]] = []
        cached = {k: np.asarray(obs_grp[k][()]) for k in keys}
        for t in range(T):
            states.append({k: cached[k][t] for k in keys})
        return states

    @staticmethod
    def to_trajectory(
        states: list[Any],
        actions: list[Any],
        *,
        success: bool = True,
    ) -> Trajectory:
        T = len(states)
        return Trajectory(
            raw_states=list(states),
            latents=torch.zeros((T, 0)),
            actions=tuple(actions),
            success=success,
        )
