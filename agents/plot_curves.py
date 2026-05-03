"""Plot training curves from SB3 EvalCallback logs.

EvalCallback writes evaluations.npz to {logdir}/seed_{i}/eval/.
We read all seeds for an algo, align on the timestep grid, and plot
mean ± std.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def load_runs(algo_dir: Path):
    """Load all evaluations.npz files under algo_dir/seed_*/eval/."""
    runs = []
    for seed_dir in sorted(algo_dir.glob("seed_*")):
        npz = seed_dir / "eval" / "evaluations.npz"
        if not npz.exists():
            continue
        data = np.load(npz)
        # data["timesteps"]: (T,)
        # data["results"]: (T, n_eval_episodes) per-episode rewards
        runs.append({
            "timesteps": data["timesteps"],
            "rewards": data["results"].mean(axis=1),
        })
    return runs


def plot_algos(algo_dirs: Dict[str, Path], out: Path) -> None:
    plt.figure(figsize=(8, 5))

    for name, d in algo_dirs.items():
        runs = load_runs(d)
        if not runs:
            print(f"[skip] no runs found for {name} at {d}")
            continue

        # Align on the shortest timestep grid across seeds.
        min_len = min(len(r["timesteps"]) for r in runs)
        ts = runs[0]["timesteps"][:min_len]
        stack = np.stack([r["rewards"][:min_len] for r in runs], axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)

        plt.plot(ts, mean, label=f"{name} (n={len(runs)})")
        plt.fill_between(ts, mean - std, mean + std, alpha=0.2)

    plt.xlabel("environment steps")
    plt.ylabel("mean episode reward (eval)")
    plt.title("Sequential Delivery POMDP — training curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--algos", nargs="+",
                        default=["ppo", "recurrent_ppo", "dqn", "qrdqn"])
    parser.add_argument("--out", type=str, default="runs/curves.png")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    algo_dirs = {a: runs_root / a for a in args.algos}
    plot_algos(algo_dirs, Path(args.out))


if __name__ == "__main__":
    main()
