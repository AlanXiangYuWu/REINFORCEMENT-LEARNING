"""Pygame renderer for SequentialDeliveryEnv.

Designed to run headless on a server (set SDL_VIDEODRIVER=dummy if
no display is available). Returns an HxWx3 uint8 array when called
in rgb_array mode.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Use dummy SDL driver if no display is available. Caller may override.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

# Color palette for up to 6 landmarks.
LANDMARK_COLORS = [
    (220, 60, 60),    # red
    (60, 170, 90),    # green
    (240, 200, 50),   # yellow
    (140, 90, 230),   # purple
    (50, 150, 220),   # blue
    (240, 130, 60),   # orange
]
LANDMARK_LABELS = ["A", "B", "C", "D", "E", "F"]

ROBOT_COLOR = (40, 40, 40)
GRID_COLOR = (200, 200, 200)
BG_COLOR = (250, 250, 250)
HUD_BG = (235, 235, 240)
TEXT_COLOR = (30, 30, 30)
WRONG_OVERLAY = (220, 60, 60)
DONE_OVERLAY = (60, 170, 90)


class PygameRenderer:
    def __init__(
        self,
        grid_size: int,
        n_landmarks: int,
        view_size: int = 5,
        cell_px: int = 48,
        hud_h: int = 80,
        side_w: int = 220,
        mode: str = "rgb_array",
    ) -> None:
        if n_landmarks > len(LANDMARK_COLORS):
            raise ValueError(
                f"renderer supports up to {len(LANDMARK_COLORS)} landmarks"
            )
        self.grid_size = grid_size
        self.n_landmarks = n_landmarks
        self.view_size = view_size
        self.cell_px = cell_px
        self.hud_h = hud_h
        self.side_w = side_w
        self.mode = mode

        self.grid_px = grid_size * cell_px
        self.W = self.grid_px + side_w
        self.H = self.grid_px + hud_h

        pygame.init()
        try:
            pygame.font.init()
        except Exception:
            pass

        if mode == "human":
            self.surface = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Sequential Delivery POMDP")
        else:
            self.surface = pygame.Surface((self.W, self.H))

        self.font_big = pygame.font.SysFont("dejavusans", 22, bold=True)
        self.font = pygame.font.SysFont("dejavusans", 16)
        self.font_small = pygame.font.SysFont("dejavusans", 13)

    # ------------------------------------------------------------------
    def render(self, env: Any) -> np.ndarray | None:
        s = self.surface
        s.fill(BG_COLOR)

        self._draw_grid(s, env)
        self._draw_landmarks(s, env)
        self._draw_robot(s, env)
        self._draw_hud(s, env)
        self._draw_side(s, env)

        if self.mode == "human":
            pygame.display.flip()
            return None

        # rgb_array
        rgb = pygame.surfarray.array3d(s)
        return np.transpose(rgb, (1, 0, 2))  # (W,H,3) -> (H,W,3)

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    def _cell_rect(self, r: int, c: int) -> pygame.Rect:
        return pygame.Rect(
            c * self.cell_px,
            self.hud_h + r * self.cell_px,
            self.cell_px,
            self.cell_px,
        )

    def _draw_grid(self, s: pygame.Surface, env: Any) -> None:
        # White background for grid area
        pygame.draw.rect(
            s, (255, 255, 255),
            pygame.Rect(0, self.hud_h, self.grid_px, self.grid_px),
        )
        # Grid lines
        for i in range(self.grid_size + 1):
            x = i * self.cell_px
            pygame.draw.line(
                s, GRID_COLOR,
                (x, self.hud_h), (x, self.hud_h + self.grid_px), 1,
            )
            y = self.hud_h + i * self.cell_px
            pygame.draw.line(
                s, GRID_COLOR,
                (0, y), (self.grid_px, y), 1,
            )

        # Highlight the agent's ego-view footprint.
        half = self.view_size // 2
        r0 = int(env.robot_pos[0]) - half
        c0 = int(env.robot_pos[1]) - half
        for di in range(self.view_size):
            for dj in range(self.view_size):
                gi, gj = r0 + di, c0 + dj
                if 0 <= gi < self.grid_size and 0 <= gj < self.grid_size:
                    rect = self._cell_rect(gi, gj)
                    overlay = pygame.Surface(
                        (rect.width, rect.height), pygame.SRCALPHA)
                    overlay.fill((100, 150, 255, 30))
                    s.blit(overlay, rect.topleft)

    def _draw_landmarks(self, s: pygame.Surface, env: Any) -> None:
        for i in range(env.n_landmarks):
            r = int(env.landmark_pos[i, 0])
            c = int(env.landmark_pos[i, 1])
            rect = self._cell_rect(r, c)
            color = LANDMARK_COLORS[i]

            # Base square
            inner = rect.inflate(-6, -6)
            pygame.draw.rect(s, color, inner, border_radius=8)

            # Status overlay
            h = float(env.history[i])
            if h == 1.0:
                # Done: green check overlay
                pygame.draw.rect(
                    s, DONE_OVERLAY,
                    inner, width=4, border_radius=8,
                )
            elif h == -1.0:
                # Wrong: red X
                pygame.draw.line(
                    s, (255, 255, 255),
                    (inner.left + 6, inner.top + 6),
                    (inner.right - 6, inner.bottom - 6), 4,
                )
                pygame.draw.line(
                    s, (255, 255, 255),
                    (inner.right - 6, inner.top + 6),
                    (inner.left + 6, inner.bottom - 6), 4,
                )

            # Letter
            label = self.font_big.render(
                LANDMARK_LABELS[i], True, (255, 255, 255))
            s.blit(label, label.get_rect(center=rect.center))

    def _draw_robot(self, s: pygame.Surface, env: Any) -> None:
        r = int(env.robot_pos[0])
        c = int(env.robot_pos[1])
        rect = self._cell_rect(r, c)
        cx, cy = rect.center
        radius = self.cell_px // 3
        pygame.draw.circle(s, ROBOT_COLOR, (cx, cy), radius)
        pygame.draw.circle(s, (255, 255, 255), (cx, cy), radius, 2)

    def _draw_hud(self, s: pygame.Surface, env: Any) -> None:
        pygame.draw.rect(s, HUD_BG, pygame.Rect(0, 0, self.W, self.hud_h))
        pygame.draw.line(
            s, GRID_COLOR,
            (0, self.hud_h - 1), (self.W, self.hud_h - 1), 1,
        )
        from env import ACTION_NAMES
        last = ACTION_NAMES[env._last_action] if env._last_action >= 0 else "—"
        outcome = env._last_visit_outcome
        outcome_color = {
            "correct": (60, 170, 90),
            "wrong": (220, 60, 60),
            "noop": (180, 130, 30),
            "none": TEXT_COLOR,
        }.get(outcome, TEXT_COLOR)

        lines = [
            (f"Sequential Delivery POMDP — step {env.t}/{env.max_steps}", TEXT_COLOR),
            (f"progress: {env.k}/{env.n_landmarks}    last action: {last}    outcome: {outcome}", outcome_color),
        ]
        for i, (text, color) in enumerate(lines):
            surf = self.font.render(text, True, color)
            s.blit(surf, (12, 10 + i * 24))

    def _draw_side(self, s: pygame.Surface, env: Any) -> None:
        x0 = self.grid_px
        pygame.draw.rect(
            s, HUD_BG,
            pygame.Rect(x0, self.hud_h, self.side_w, self.grid_px),
        )
        pygame.draw.line(
            s, GRID_COLOR,
            (x0, self.hud_h), (x0, self.hud_h + self.grid_px), 1,
        )

        # Title
        title = self.font.render("Belief / History", True, TEXT_COLOR)
        s.blit(title, (x0 + 12, self.hud_h + 10))

        # Per-landmark status
        for i in range(env.n_landmarks):
            yy = self.hud_h + 40 + i * 32
            color = LANDMARK_COLORS[i]
            pygame.draw.rect(
                s, color, pygame.Rect(x0 + 12, yy, 22, 22), border_radius=4)
            label = self.font.render(
                LANDMARK_LABELS[i], True, (255, 255, 255))
            s.blit(label, label.get_rect(center=(x0 + 23, yy + 11)))

            h = float(env.history[i])
            status = {1.0: "correct ✓", -1.0: "rejected ✗", 0.0: "unknown"}[h]
            scol = {1.0: (60, 130, 70), -1.0: (180, 50, 50), 0.0: TEXT_COLOR}[h]
            sf = self.font.render(status, True, scol)
            s.blit(sf, (x0 + 42, yy + 4))

        # Hidden ground truth (small, faded — debug only)
        gt = "→".join(LANDMARK_LABELS[i] for i in env.permutation.tolist())
        gt_label = self.font_small.render(
            f"hidden order: {gt}", True, (160, 160, 160))
        s.blit(gt_label, (x0 + 12, self.hud_h + self.grid_px - 26))
