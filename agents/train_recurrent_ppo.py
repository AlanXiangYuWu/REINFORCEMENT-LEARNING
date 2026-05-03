"""Train RecurrentPPO (LSTM) on SequentialDeliveryEnv.

This is the principled POMDP solution: the policy/value LSTM learns
its own belief over the hidden ordering rather than relying on the
hand-crafted history vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from agents.common import make_env_fn, DEFAULT_ENV_KWARGS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--logdir", type=str, default="runs/recurrent_ppo")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reward-shaping", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = {**DEFAULT_ENV_KWARGS, "reward_shaping": args.reward_shaping}

    train_env = SubprocVecEnv([
        make_env_fn(seed=args.seed * 1000 + i, **env_kwargs)
        for i in range(args.n_envs)
    ])
    train_env = VecMonitor(train_env)

    eval_env = SubprocVecEnv([
        make_env_fn(seed=10_000 + i, **env_kwargs)
        for i in range(8)
    ])
    eval_env = VecMonitor(eval_env)

    run_dir = Path(args.logdir) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )

    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tb"),
        device=args.device,
        n_steps=256,
        batch_size=256,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            lstm_hidden_size=128,
            n_lstm_layers=1,
        ),
    )

    model.learn(total_timesteps=args.total_steps, callback=eval_cb,
                progress_bar=True)
    model.save(str(run_dir / "final.zip"))
    print(f"saved final model -> {run_dir / 'final.zip'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
