"""Train DQN baseline on SequentialDeliveryEnv (single env)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from agents.common import make_env_fn, DEFAULT_ENV_KWARGS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--logdir", type=str, default="runs/dqn")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reward-shaping", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = {**DEFAULT_ENV_KWARGS, "reward_shaping": args.reward_shaping}

    train_env = DummyVecEnv([make_env_fn(seed=args.seed, **env_kwargs)])
    train_env = VecMonitor(train_env)

    eval_env = SubprocVecEnv([
        make_env_fn(seed=10_000 + i, **env_kwargs) for i in range(4)
    ])
    eval_env = VecMonitor(eval_env)

    run_dir = Path(args.logdir) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=50_000,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )

    model = DQN(
        policy="MultiInputPolicy",
        env=train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tb"),
        device=args.device,
        learning_rate=2.5e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(total_timesteps=args.total_steps, callback=eval_cb,
                progress_bar=True)
    model.save(str(run_dir / "final.zip"))
    print(f"saved final model -> {run_dir / 'final.zip'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
