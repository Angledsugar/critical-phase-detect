"""Collect π_0.5_libero rollouts on a LIBERO task — must run in .venv-libero.

Server (main venv):
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv/bin/python /home/engineer/openpi/scripts/serve_policy.py \\
        --env LIBERO --port 8000

Client (this script, .venv-libero):
    MUJOCO_GL=egl .venv-libero/bin/python scripts/collect_libero_rollouts.py \\
        --suite libero_10 --task-id 0 --num-episodes 10 \\
        --out-dir /media/engineer/DATA/datasets/cpd_rollouts/pi05_libero
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import pathlib
import time

import numpy as np

# torch 2.x weights_only default shim (LIBERO calls torch.load without specifying it)
import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**{"weights_only": False}, **kw})

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools, websocket_client_policy

DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RES = 256
RESIZE = 224
REPLAN_STEPS = 5
NUM_WAIT_STEPS = 10

# upstream max-steps per suite (longest training demo)
MAX_STEPS_PER_SUITE = {
    "libero_10": 520, "libero_90": 400,
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
}


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    q[3] = max(-1.0, min(1.0, q[3]))
    den = np.sqrt(1.0 - q[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def proprio(obs: dict) -> np.ndarray:
    return np.concatenate([
        obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]
    ]).astype(np.float32)


def run_episode(env, client, task_description: str, init_state: np.ndarray, max_steps: int):
    env.reset()
    obs = env.set_init_state(init_state)
    plan = collections.deque()
    agent_imgs, wrist_imgs, states, actions, rewards = [], [], [], [], []
    done = False
    t = 0
    while t < max_steps + NUM_WAIT_STEPS:
        if t < NUM_WAIT_STEPS:
            obs, _, done, _ = env.step(DUMMY_ACTION)
            t += 1
            continue
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        st = proprio(obs)
        if not plan:
            elem = {
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, RESIZE, RESIZE)),
                "observation/wrist_image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist, RESIZE, RESIZE)),
                "observation/state": st,
                "prompt": task_description,
            }
            chunk = client.infer(elem)["actions"]
            plan.extend(chunk[:REPLAN_STEPS])
        action = plan.popleft()
        agent_imgs.append(img)
        wrist_imgs.append(wrist)
        states.append(st)
        actions.append(np.asarray(action, dtype=np.float32))
        obs, r, done, _ = env.step(action.tolist())
        rewards.append(float(r))
        t += 1
        if done:
            break
    return {
        "agentview": np.stack(agent_imgs).astype(np.uint8),     # (T, 256, 256, 3)
        "wrist": np.stack(wrist_imgs).astype(np.uint8),
        "state": np.stack(states).astype(np.float32),            # (T, 9)
        "action": np.stack(actions).astype(np.float32),          # (T, 7)
        "reward": np.asarray(rewards, dtype=np.float32),
        "success": bool(done),
        "length": int(len(actions)),
        "task_description": task_description,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--start-idx", type=int, default=0,
                   help="skip init_states[0:start_idx]; resume mid-run")
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap; default = upstream max for suite")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--out-dir", default="/media/engineer/DATA/datasets/cpd_rollouts/pi05_libero")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    bm = benchmark.get_benchmark_dict()[args.suite]()
    task = bm.get_task(args.task_id)
    init_states = bm.get_task_init_states(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    out_dir = pathlib.Path(args.out_dir) / args.suite / f"task{args.task_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_steps = args.max_steps or MAX_STEPS_PER_SUITE.get(args.suite, 400)
    print(f"[collect] suite={args.suite} task={args.task_id} task_name={task.name}")
    print(f"[collect] description: {task.language!r}")
    print(f"[collect] num_episodes={args.num_episodes} max_steps={max_steps}")
    print(f"[collect] out_dir={out_dir}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=ENV_RES, camera_widths=ENV_RES,
    )
    env.seed(args.seed)

    client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)

    start = args.start_idx
    end = start + args.num_episodes
    n = end - start
    n_init = len(init_states)
    n_succ = 0
    t0 = time.time()
    prev_cycle = -1
    for i in range(start, end):
        cycle = i // n_init
        if cycle != prev_cycle:
            env.seed(args.seed + cycle)
            prev_cycle = cycle
        init_idx = i % n_init
        ep_t0 = time.time()
        ep = run_episode(env, client, task.language, init_states[init_idx], max_steps)
        n_succ += int(ep["success"])
        out_path = out_dir / f"ep{i:03d}.npz"
        np.savez_compressed(out_path, **ep)
        done_so_far = i - start + 1
        rate = n_succ / done_so_far
        print(f"[ep {i:03d}] init={init_idx:02d} cycle={cycle} T={ep['length']:3d} success={ep['success']} "
              f"(this ep={time.time()-ep_t0:.0f}s) "
              f"running success rate: {n_succ}/{done_so_far} = {rate:.2f}",
              flush=True)

    elapsed = time.time() - t0
    print(f"[done] {n_succ}/{n} success ({n_succ/n:.2f}); total {elapsed:.0f}s")

    summary = out_dir / "summary.txt"
    summary.write_text(
        f"suite={args.suite}\ntask_id={args.task_id}\ntask={task.language}\n"
        f"num_episodes={n}\nnum_success={n_succ}\nsuccess_rate={n_succ/n:.4f}\n"
        f"max_steps={max_steps}\nseed={args.seed}\nelapsed_seconds={elapsed:.1f}\n"
    )

    env.close()


if __name__ == "__main__":
    main()
