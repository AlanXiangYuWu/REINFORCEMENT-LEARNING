"""Shared helpers for training and evaluation scripts.

Adds the project root to sys.path so child processes spawned by
SubprocVecEnv can import `env.py` regardless of CWD.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.monitor import Monitor  # noqa: E402

from env import SequentialDeliveryEnv  # noqa: E402


# Shared default env config -- keep in sync with docs/PRD.md §9.
DEFAULT_ENV_KWARGS = dict(
    grid_size=6,
    n_landmarks=3,
    max_steps=100,
    view_size=5,
    reward_shaping=False,
    wrong_visit_penalty=0.1,
)


def make_env_fn(seed: int, monitor_dir: str | None = None,
                **env_kwargs) -> Callable[[], "Monitor"]:
    """Factory used by SubprocVecEnv / DummyVecEnv."""
    cfg = {**DEFAULT_ENV_KWARGS, **env_kwargs}

    def _init():
        env = SequentialDeliveryEnv(**cfg)
        if monitor_dir:
            os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=os.path.join(monitor_dir, f"seed_{seed}"))
        else:
            env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def fmt_steps(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)
