"""Record a video of an agent (random or trained) playing the env.

Outputs an mp4 (via imageio + ffmpeg). Requires SDL_VIDEODRIVER=dummy
on a headless server; render.py sets it by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

from agents.common import DEFAULT_ENV_KWARGS  # noqa: E402
from agents.eval import load_policy, ALGOS  # noqa: E402
from env import SequentialDeliveryEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS, default="random")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--repeat-last-frame", type=int, default=8,
                        help="hold the final frame this many times for visual breathing room")
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    env = SequentialDeliveryEnv(render_mode="rgb_array", **DEFAULT_ENV_KWARGS)
    policy = load_policy(args.algo, args.model, env)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed_base + ep)
        state = None
        frames.append(env.render())
        done = False
        while not done:
            action, state = policy(obs, state)
            obs, r, term, trunc, info = env.step(action)
            frames.append(env.render())
            done = term or trunc
        # hold final frame
        for _ in range(args.repeat_last_frame):
            frames.append(frames[-1])

    imageio.mimsave(str(out), frames, fps=args.fps,
                    macro_block_size=1)
    env.close()
    print(f"wrote {len(frames)} frames -> {out} ({args.fps} fps)")


if __name__ == "__main__":
    main()
