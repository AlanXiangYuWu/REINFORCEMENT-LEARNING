"""Evaluate a trained agent (or a random one) on a fixed seed bank.

Reports:
  - mean / std episode reward
  - success rate (full sequence completed)
  - mean episode length
  - mean number of wrong-visit attempts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from agents.common import DEFAULT_ENV_KWARGS  # noqa: E402
from env import SequentialDeliveryEnv  # noqa: E402

ALGOS = ("random", "ppo", "recurrent_ppo", "dqn", "qrdqn")


def load_policy(algo: str, model_path: str | None,
                env: SequentialDeliveryEnv) -> Callable:
    if algo == "random":
        def policy(_obs, _state=None):
            return env.action_space.sample(), None
        return policy

    if model_path is None:
        raise ValueError(f"--model required for algo={algo}")

    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    elif algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path)
    elif algo == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
    elif algo == "qrdqn":
        from sb3_contrib import QRDQN
        model = QRDQN.load(model_path)
    else:
        raise ValueError(f"unknown algo {algo}")

    if algo == "recurrent_ppo":
        def policy(obs, state):
            action, state = model.predict(
                obs, state=state, episode_start=np.array([state is None]),
                deterministic=True,
            )
            return int(action), state
        return policy
    else:
        def policy(obs, _state=None):
            action, _ = model.predict(obs, deterministic=True)
            return int(action), None
        return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS, default="random")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=20_000,
                        help="seeds [seed_base .. seed_base+n) form the eval bank")
    parser.add_argument("--out", type=str, default=None,
                        help="optional path to dump metrics as JSON")
    args = parser.parse_args()

    env = SequentialDeliveryEnv(**DEFAULT_ENV_KWARGS)
    policy = load_policy(args.algo, args.model, env)

    rewards = []
    lengths = []
    successes = []
    wrongs = []

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed_base + ep)
        state = None
        total_r = 0.0
        n_wrong = 0
        done = False
        while not done:
            action, state = policy(obs, state)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            if info["last_outcome"] == "wrong":
                n_wrong += 1
            done = term or trunc
        rewards.append(total_r)
        lengths.append(env.t)
        successes.append(env.k == env.n_landmarks)
        wrongs.append(n_wrong)

    metrics = {
        "algo": args.algo,
        "model": args.model,
        "n_episodes": args.n_episodes,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "wrong_visits_mean": float(np.mean(wrongs)),
    }

    width = 22
    print(f"\n=== eval: {args.algo} ===")
    for k, v in metrics.items():
        print(f"{k:<{width}} {v}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"wrote metrics -> {args.out}")


if __name__ == "__main__":
    main()
