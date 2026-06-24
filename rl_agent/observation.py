"""Agent-side observation builders for raw SnakeEnv state."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

# Absolute headings — must match snake.snake_env constants.
DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = 0, 1, 2, 3
DIR_VECTORS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.int32)

STANDARD_OBS_SIZE = 11


class RawState(TypedDict):
    snake_body_coords: list[tuple[int, int]]
    food_coords: tuple[int, int]
    current_direction: int
    grid_dimensions: tuple[int, int]


class ObservationBuilder:
    """Convert raw environment state into a fixed-size feature vector.

    The default ``build()`` reproduces the classic 11-feature Snake observation:
    danger (straight/left/right), heading one-hot, and food direction flags
    relative to the current heading.
    """

    def __init__(self, *, feature_size: int = STANDARD_OBS_SIZE) -> None:
        self.feature_size = feature_size

    def build(self, state: RawState | dict[str, Any]) -> np.ndarray:
        snake: list[tuple[int, int]] = list(state["snake_body_coords"])
        food_x, food_y = state["food_coords"]
        direction: int = int(state["current_direction"])
        grid_width, grid_height = state["grid_dimensions"]

        head_x, head_y = snake[0]

        danger_straight = float(self._danger_ahead(snake, direction, grid_width, grid_height))
        danger_left = float(
            self._danger_ahead(snake, (direction - 1) % 4, grid_width, grid_height)
        )
        danger_right = float(
            self._danger_ahead(snake, (direction + 1) % 4, grid_width, grid_height)
        )

        heading = np.zeros(4, dtype=np.float32)
        heading[direction] = 1.0

        food_flags = self._food_relative_flags(head_x, head_y, food_x, food_y, direction)

        obs = np.concatenate(
            [
                np.array([danger_straight, danger_left, danger_right], dtype=np.float32),
                heading,
                food_flags,
            ]
        )
        if obs.shape[0] != self.feature_size:
            raise ValueError(
                f"ObservationBuilder produced {obs.shape[0]} features, "
                f"expected {self.feature_size}"
            )
        return obs

    @staticmethod
    def _is_collision(
        position: tuple[int, int],
        snake: list[tuple[int, int]],
        grid_width: int,
        grid_height: int,
    ) -> bool:
        x, y = position
        if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
            return True
        if len(snake) > 1 and position in snake[:-1]:
            return True
        return False

    def _danger_ahead(
        self,
        snake: list[tuple[int, int]],
        direction: int,
        grid_width: int,
        grid_height: int,
    ) -> bool:
        head_x, head_y = snake[0]
        delta = DIR_VECTORS[direction]
        probe = (int(head_x + delta[0]), int(head_y + delta[1]))
        return self._is_collision(probe, snake, grid_width, grid_height)

    @staticmethod
    def _food_relative_flags(
        head_x: int,
        head_y: int,
        food_x: int,
        food_y: int,
        direction: int,
    ) -> np.ndarray:
        food_flags = np.zeros(4, dtype=np.float32)
        dx = food_x - head_x
        dy = food_y - head_y

        if direction == DIR_UP:
            food_flags[0] = float(dx < 0)
            food_flags[1] = float(dx > 0)
            food_flags[2] = float(dy < 0)
            food_flags[3] = float(dy > 0)
        elif direction == DIR_RIGHT:
            food_flags[0] = float(dy < 0)
            food_flags[1] = float(dy > 0)
            food_flags[2] = float(dx > 0)
            food_flags[3] = float(dx < 0)
        elif direction == DIR_DOWN:
            food_flags[0] = float(dx > 0)
            food_flags[1] = float(dx < 0)
            food_flags[2] = float(dy > 0)
            food_flags[3] = float(dy < 0)
        elif direction == DIR_LEFT:
            food_flags[0] = float(dy > 0)
            food_flags[1] = float(dy < 0)
            food_flags[2] = float(dx < 0)
            food_flags[3] = float(dx > 0)
        else:
            raise ValueError(f"Unhandled direction: {direction}")

        return food_flags
