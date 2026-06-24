"""Gymnasium Snake environment for reinforcement-learning agents."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snake.config import SnakeConfig, load_config

# Internal absolute headings: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = 0, 1, 2, 3
DIR_VECTORS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.int32)

# Relative actions (relative to current heading)
ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2

# Grid cell codes (numpy int8 board)
CELL_EMPTY = 0
CELL_SNAKE = 1
CELL_FOOD = 2


class SnakeEnv(gym.Env):
    """Snake game wrapped for Gymnasium.

    Game parameters live in ``config.yaml`` (see ``snake/config.py``). Pass
    ``config_path=`` or a ``SnakeConfig`` instance to override the default file.

    Action space: Discrete(3) — straight, turn left, turn right relative to the
    current heading.

    Observation space: ``Dict`` of raw game state (no feature engineering).
    Agents compute their own observations via ``rl_agent.observation.ObservationBuilder``.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 10,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        config: SnakeConfig | None = None,
        config_path: str | Path | None = None,
        grid_width: int | None = None,
        grid_height: int | None = None,
        max_episode_steps: int | None = None,
        cell_size_px: int | None = None,
        render_fps: int | None = None,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']} or None, "
                f"got {render_mode!r}"
            )

        cfg = config if config is not None else load_config(config_path)

        self.render_mode = render_mode
        self._cfg = cfg
        self.grid_width = grid_width if grid_width is not None else cfg.grid_width
        self.grid_height = grid_height if grid_height is not None else cfg.grid_height
        self.max_episode_steps = (
            max_episode_steps if max_episode_steps is not None else cfg.max_episode_steps
        )
        self.cell_size_px = cell_size_px if cell_size_px is not None else cfg.cell_size_px
        self.render_fps = render_fps if render_fps is not None else cfg.render_fps

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "snake_body_coords": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(self.grid_width),
                            spaces.Discrete(self.grid_height),
                        )
                    ),
                ),
                "food_coords": spaces.Tuple(
                    (spaces.Discrete(self.grid_width), spaces.Discrete(self.grid_height))
                ),
                "current_direction": spaces.Discrete(4),
                "grid_dimensions": spaces.Box(
                    low=np.array([1, 1], dtype=np.int32),
                    high=np.array([self.grid_width, self.grid_height], dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32,
                ),
            }
        )

        self._rng = np.random.default_rng()
        self._grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self._snake: deque[tuple[int, int]] = deque()
        self._direction = cfg.start_direction
        self._food: tuple[int, int] = (0, 0)
        self._score = 0
        self._steps = 0

        self._pygame = None
        self._screen = None
        self._clock = None
        self._window_size = (
            self.grid_width * self.cell_size_px,
            self.grid_height * self.cell_size_px,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._score = 0
        self._steps = 0
        self._direction = self._cfg.start_direction
        self._snake = self._build_initial_snake()

        self._sync_grid()
        self._spawn_food()
        self._sync_grid()

        state = self._get_state()
        info = self._build_info()
        return state, info

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self._steps += 1
        self._direction = self._resolve_direction(action)

        head_x, head_y = self._snake[0]
        delta = DIR_VECTORS[self._direction]
        new_head = (int(head_x + delta[0]), int(head_y + delta[1]))

        terminated = False
        reward = self._cfg.reward_step

        if self._is_collision(new_head):
            reward = self._cfg.reward_death
            terminated = True
        else:
            self._snake.appendleft(new_head)
            if new_head == self._food:
                reward = self._cfg.reward_food
                self._score += 1
                self._spawn_food()
            else:
                self._snake.pop()

            self._sync_grid()

        truncated = self._steps >= self.max_episode_steps and not terminated

        state = self._get_state()
        info = self._build_info(collision=terminated)

        if self.render_mode in ("human", "ansi"):
            self.render()

        return state, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode is None:
            return
        if self.render_mode == "ansi":
            self._render_ansi()
            return
        if self.render_mode == "human":
            self._render_pygame()

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None

    # ------------------------------------------------------------------
    # Core game logic (no rendering)
    # ------------------------------------------------------------------

    def _build_initial_snake(self) -> deque[tuple[int, int]]:
        start_x = self.grid_width // 2
        start_y = self.grid_height // 2
        segments: list[tuple[int, int]] = [(start_x, start_y)]

        back = -DIR_VECTORS[self._cfg.start_direction]
        x, y = start_x, start_y
        for _ in range(self._cfg.initial_snake_length - 1):
            x = int(x + back[0])
            y = int(y + back[1])
            segments.append((x, y))

        return deque(segments)

    def _resolve_direction(self, action: int) -> int:
        if action == ACTION_STRAIGHT:
            return self._direction
        if action == ACTION_LEFT:
            return (self._direction - 1) % 4
        if action == ACTION_RIGHT:
            return (self._direction + 1) % 4
        raise ValueError(f"Unhandled action: {action}")

    def _is_collision(self, position: tuple[int, int]) -> bool:
        x, y = position
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        if len(self._snake) > 1 and position in list(self._snake)[:-1]:
            return True
        return False

    def _spawn_food(self) -> None:
        empty_cells = np.argwhere(self._grid == CELL_EMPTY)
        if len(empty_cells) == 0:
            return

        index = int(self._rng.integers(0, len(empty_cells)))
        y, x = empty_cells[index]
        self._food = (int(x), int(y))

    def _sync_grid(self) -> None:
        self._grid.fill(CELL_EMPTY)
        for x, y in self._snake:
            self._grid[y, x] = CELL_SNAKE
        fx, fy = self._food
        if self._grid[fy, fx] == CELL_EMPTY:
            self._grid[fy, fx] = CELL_FOOD

    def _get_state(self) -> dict[str, Any]:
        """Return pure game state for agent-side observation builders."""
        return {
            "snake_body_coords": list(self._snake),
            "food_coords": self._food,
            "current_direction": self._direction,
            "grid_dimensions": (self.grid_width, self.grid_height),
        }

    def _build_info(self, *, collision: bool = False) -> dict[str, Any]:
        return {
            "score": self._score,
            "steps": self._steps,
            "snake_length": len(self._snake),
            "direction": self._direction,
            "food": self._food,
            "collision": collision,
        }

    # ------------------------------------------------------------------
    # Optional rendering (decoupled from step logic above)
    # ------------------------------------------------------------------

    def _render_ansi(self) -> None:
        cfg = self._cfg
        chars = []
        for y in range(self.grid_height):
            row = []
            for x in range(self.grid_width):
                if (x, y) == self._snake[0]:
                    row.append(cfg.ansi_head)
                elif (x, y) == self._food:
                    row.append(cfg.ansi_food)
                elif (x, y) in self._snake:
                    row.append(cfg.ansi_body)
                else:
                    row.append(cfg.ansi_empty)
            chars.append("".join(row))

        frame = "\n".join(chars)
        print(f"\n--- step {self._steps} | score {self._score} ---\n{frame}\n")

    def _init_pygame(self) -> None:
        if self._pygame is not None:
            return
        import pygame

        pygame.init()
        pygame.display.set_caption(self._cfg.window_title)
        self._pygame = pygame
        self._screen = pygame.display.set_mode(self._window_size)
        self._clock = pygame.time.Clock()

    def _render_pygame(self) -> None:
        cfg = self._cfg
        self._init_pygame()
        assert self._pygame is not None and self._screen is not None and self._clock is not None

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.close()
                raise SystemExit("Window closed by user")

        self._screen.fill(cfg.color_background)

        for x in range(self.grid_width + 1):
            px = x * self.cell_size_px
            self._pygame.draw.line(
                self._screen, cfg.color_grid_line, (px, 0), (px, self._window_size[1])
            )
        for y in range(self.grid_height + 1):
            py = y * self.cell_size_px
            self._pygame.draw.line(
                self._screen, cfg.color_grid_line, (0, py), (self._window_size[0], py)
            )

        fx, fy = self._food
        self._draw_cell(fx, fy, cfg.color_food)

        for index, (x, y) in enumerate(self._snake):
            color = cfg.color_head if index == 0 else cfg.color_snake
            self._draw_cell(x, y, color)

        self._pygame.display.flip()
        self._clock.tick(self.render_fps)

    def _draw_cell(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        assert self._pygame is not None and self._screen is not None
        padding = self._cfg.cell_padding_px
        rect = self._pygame.Rect(
            x * self.cell_size_px + padding,
            y * self.cell_size_px + padding,
            self.cell_size_px - 2 * padding,
            self.cell_size_px - 2 * padding,
        )
        self._pygame.draw.rect(self._screen, color, rect)


if __name__ == "__main__":
    cfg = load_config()
    env = SnakeEnv(render_mode="human")
    num_episodes = cfg.demo_num_episodes

    try:
        for episode in range(num_episodes):
            state, info = env.reset(seed=episode)
            terminated = False
            truncated = False
            total_reward = 0.0

            print(
                f"Episode {episode + 1}/{num_episodes} — "
                f"state keys {sorted(state.keys())}"
            )

            while not (terminated or truncated):
                action = env.action_space.sample()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

            print(
                f"Episode {episode + 1} finished: score={info['score']}, "
                f"steps={info['steps']}, total_reward={total_reward:.1f}"
            )
            time.sleep(cfg.demo_pause_between_episodes_sec)
    finally:
        env.close()
