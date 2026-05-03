"""Sequential Multi-Target Delivery POMDP environment.

A delivery robot navigates an N x N grid and must visit K landmarks in a
hidden order that is randomized at the start of each episode. The agent
only learns whether a visit was correct or wrong after attempting it,
so the optimal policy must remember which landmarks have been tried and
which were rejected -- making this a POMDP.

This file is the gymnasium-style environment required by the project
spec (CA6126 final project).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Tile encoding (channel index in the ego-view tensor)
# ---------------------------------------------------------------------------
# Channel 0          : empty
# Channels 1 .. K    : landmark i (one-hot)
# Channel K + 1      : wall / out-of-grid
# ---------------------------------------------------------------------------

ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "VISIT")
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class SequentialDeliveryEnv(gym.Env):
    """Sequential multi-target delivery POMDP.

    Args:
        grid_size: side length N of the square grid.
        n_landmarks: number of landmarks K (default 4 -> 24 orderings).
        max_steps: episode horizon T.
        view_size: side length of the (odd) ego-centric observation window.
        reward_shaping: if True, add a small potential-based bonus toward
            the nearest plausible landmark to speed up early learning.
        render_mode: "rgb_array" returns RGB frames; "human" opens a window.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 8}

    def __init__(
        self,
        grid_size: int = 12,
        n_landmarks: int = 4,
        max_steps: int = 200,
        view_size: int = 5,
        reward_shaping: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if view_size % 2 == 0:
            raise ValueError("view_size must be odd")
        if n_landmarks < 2:
            raise ValueError("n_landmarks must be >= 2")
        if grid_size < view_size:
            raise ValueError("grid_size must be >= view_size")

        self.grid_size = int(grid_size)
        self.n_landmarks = int(n_landmarks)
        self.max_steps = int(max_steps)
        self.view_size = int(view_size)
        self.reward_shaping = bool(reward_shaping)
        self.render_mode = render_mode

        self._n_channels = 1 + self.n_landmarks + 1  # empty + K + wall
        self._half = self.view_size // 2

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "view": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.view_size, self.view_size, self._n_channels),
                dtype=np.float32,
            ),
            "history": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.n_landmarks,), dtype=np.float32,
            ),
            "progress": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "steps_left": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        # State, set in reset()
        self.robot_pos: np.ndarray = np.zeros(2, dtype=np.int32)
        self.landmark_pos: np.ndarray = np.zeros((self.n_landmarks, 2), dtype=np.int32)
        self.permutation: np.ndarray = np.arange(self.n_landmarks, dtype=np.int32)
        self.history: np.ndarray = np.zeros(self.n_landmarks, dtype=np.float32)
        self.k: int = 0
        self.t: int = 0
        self._last_action: int = -1
        self._last_visit_outcome: str = "none"  # "none" | "correct" | "wrong" | "noop"
        self._prev_potential: float = 0.0

        # Rendering state (lazy-initialized in render.py)
        self._renderer = None

    # ------------------------------------------------------------------
    # Core gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        rng = self.np_random

        # Sample n_landmarks + 1 distinct cells: robot first, landmarks rest.
        n_cells = self.grid_size * self.grid_size
        idx = rng.choice(n_cells, size=self.n_landmarks + 1, replace=False)
        rows, cols = np.divmod(idx, self.grid_size)
        positions = np.stack([rows, cols], axis=1).astype(np.int32)

        self.robot_pos = positions[0].copy()
        self.landmark_pos = positions[1:].copy()
        self.permutation = rng.permutation(self.n_landmarks).astype(np.int32)
        self.history = np.zeros(self.n_landmarks, dtype=np.float32)
        self.k = 0
        self.t = 0
        self._last_action = -1
        self._last_visit_outcome = "none"
        self._prev_potential = self._potential()

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        self._last_action = action
        reward = -0.01  # per-step cost
        terminated = False
        truncated = False
        outcome = "none"

        if action < 4:
            di, dj = DELTAS[action]
            new_r = int(self.robot_pos[0] + di)
            new_c = int(self.robot_pos[1] + dj)
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                self.robot_pos[0] = new_r
                self.robot_pos[1] = new_c
            else:
                reward -= 0.05  # wall bump
        else:
            # VISIT
            on_landmark = -1
            for i in range(self.n_landmarks):
                if (self.robot_pos[0] == self.landmark_pos[i, 0]
                        and self.robot_pos[1] == self.landmark_pos[i, 1]):
                    on_landmark = i
                    break

            if on_landmark < 0:
                reward -= 0.05  # visit on empty cell
                outcome = "noop"
            else:
                target = int(self.permutation[self.k])
                if on_landmark == target:
                    self.k += 1
                    self.history[on_landmark] = 1.0
                    reward += 1.0
                    outcome = "correct"
                    if self.k >= self.n_landmarks:
                        reward += 5.0
                        terminated = True
                else:
                    # Don't overwrite a previously-correct mark.
                    if self.history[on_landmark] != 1.0:
                        self.history[on_landmark] = -1.0
                    reward -= 0.5
                    outcome = "wrong"

        self._last_visit_outcome = outcome
        self.t += 1

        # Reward shaping (potential-based; preserves optimal policy in MDP
        # sense; here used as a soft hint).
        if self.reward_shaping and not terminated:
            phi = self._potential()
            reward += 0.99 * phi - self._prev_potential
            self._prev_potential = phi

        if not terminated and self.t >= self.max_steps:
            truncated = True
            reward -= 1.0

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        view = np.zeros(
            (self.view_size, self.view_size, self._n_channels),
            dtype=np.float32,
        )
        wall_ch = self._n_channels - 1
        for di in range(self.view_size):
            for dj in range(self.view_size):
                gi = int(self.robot_pos[0]) - self._half + di
                gj = int(self.robot_pos[1]) - self._half + dj
                if 0 <= gi < self.grid_size and 0 <= gj < self.grid_size:
                    cell_ch = 0  # empty by default
                    for k in range(self.n_landmarks):
                        if gi == self.landmark_pos[k, 0] and gj == self.landmark_pos[k, 1]:
                            cell_ch = k + 1
                            break
                    view[di, dj, cell_ch] = 1.0
                else:
                    view[di, dj, wall_ch] = 1.0

        return {
            "view": view,
            "history": self.history.copy(),
            "progress": np.array(
                [self.k / self.n_landmarks], dtype=np.float32),
            "steps_left": np.array(
                [(self.max_steps - self.t) / self.max_steps], dtype=np.float32),
        }

    def _get_info(self) -> dict:
        next_target = int(self.permutation[self.k]) if self.k < self.n_landmarks else -1
        return {
            "robot_pos": self.robot_pos.copy(),
            "landmark_pos": self.landmark_pos.copy(),
            "permutation": self.permutation.copy(),
            "k": self.k,
            "t": self.t,
            "next_target": next_target,
            "last_outcome": self._last_visit_outcome,
        }

    def _potential(self) -> float:
        """-d_min(robot, nearest plausible landmark), normalized to [-1, 0].

        A landmark is plausible if it has not been rejected (history != -1)
        and not yet correctly visited (history != 1).
        """
        plausible = [
            i for i in range(self.n_landmarks)
            if self.history[i] == 0.0
        ]
        if not plausible:
            return 0.0
        min_d = None
        for i in plausible:
            d = (abs(int(self.robot_pos[0]) - int(self.landmark_pos[i, 0]))
                 + abs(int(self.robot_pos[1]) - int(self.landmark_pos[i, 1])))
            if min_d is None or d < min_d:
                min_d = d
        max_d = 2.0 * (self.grid_size - 1)
        return -float(min_d) / max_d

    # ------------------------------------------------------------------
    # Rendering (lazy-imports pygame)
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from render import PygameRenderer  # local import to avoid hard pygame dep
            self._renderer = PygameRenderer(
                grid_size=self.grid_size,
                n_landmarks=self.n_landmarks,
                view_size=self.view_size,
                mode=self.render_mode,
            )
        return self._renderer.render(self)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Convenience for `gymnasium.make`-like usage.
def make(**kwargs: Any) -> SequentialDeliveryEnv:
    return SequentialDeliveryEnv(**kwargs)


if __name__ == "__main__":  # smoke test
    env = SequentialDeliveryEnv()
    obs, info = env.reset(seed=0)
    print("obs keys:", list(obs.keys()))
    print("view shape:", obs["view"].shape)
    print("permutation (hidden):", info["permutation"])
    total = 0.0
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print("total reward over random rollout:", round(total, 3))
