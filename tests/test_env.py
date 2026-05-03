"""Smoke tests + gymnasium check_env."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gymnasium.utils.env_checker import check_env

from env import SequentialDeliveryEnv


def test_check_env():
    env = SequentialDeliveryEnv(grid_size=8, n_landmarks=3, max_steps=80)
    check_env(env, skip_render_check=True)


def test_random_rollout_terminates():
    env = SequentialDeliveryEnv(grid_size=8, n_landmarks=3, max_steps=50)
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(10):  # 10 episodes
        done = False
        steps = 0
        while not done:
            a = int(rng.integers(0, env.action_space.n))
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            done = term or trunc
            assert steps <= env.max_steps + 1
        env.reset(seed=int(rng.integers(0, 10_000)))


def test_reward_signs():
    """Correct visit gives a positive reward, wrong visit a clearly worse one."""
    env = SequentialDeliveryEnv(grid_size=6, n_landmarks=3, max_steps=50)
    env.reset(seed=1)
    # Force the agent onto a landmark and call VISIT.
    env.robot_pos = env.landmark_pos[int(env.permutation[0])].copy()
    _, r_correct, term, _, info = env.step(4)
    assert info["last_outcome"] == "correct"
    assert r_correct > 0

    env.reset(seed=2)
    # Stand on a non-target landmark.
    wrong_idx = [i for i in range(env.n_landmarks)
                 if i != int(env.permutation[0])][0]
    env.robot_pos = env.landmark_pos[wrong_idx].copy()
    _, r_wrong, term, _, info = env.step(4)
    assert info["last_outcome"] == "wrong"
    assert r_wrong < 0


if __name__ == "__main__":
    test_check_env()
    test_random_rollout_terminates()
    test_reward_signs()
    print("all env tests passed.")
