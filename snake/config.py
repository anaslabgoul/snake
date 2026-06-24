"""Load and validate Snake environment settings from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"

_DIRECTION_NAMES = {"up": 0, "right": 1, "down": 2, "left": 3}


@dataclass(frozen=True)
class SnakeConfig:
    grid_width: int
    grid_height: int
    max_episode_steps: int
    reward_food: float
    reward_death: float
    reward_step: float
    initial_snake_length: int
    start_direction: int
    render_fps: int
    cell_size_px: int
    cell_padding_px: int
    window_title: str
    color_background: tuple[int, int, int]
    color_grid_line: tuple[int, int, int]
    color_snake: tuple[int, int, int]
    color_head: tuple[int, int, int]
    color_food: tuple[int, int, int]
    ansi_head: str
    ansi_body: str
    ansi_food: str
    ansi_empty: str
    observation_size: int
    demo_num_episodes: int
    demo_pause_between_episodes_sec: float


def _require_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Config missing required section '{name}'")
    return section


def _parse_direction(value: str) -> int:
    key = value.strip().lower()
    if key not in _DIRECTION_NAMES:
        allowed = ", ".join(sorted(_DIRECTION_NAMES))
        raise ValueError(f"Invalid start_direction {value!r}; expected one of: {allowed}")
    return _DIRECTION_NAMES[key]


def _parse_rgb(value: list[Any], *, field: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must be a list of three integers [R, G, B]")
    try:
        r, g, b = (int(value[0]), int(value[1]), int(value[2]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain integer RGB values") from exc
    for channel, name in ((r, "R"), (g, "G"), (b, "B")):
        if not 0 <= channel <= 255:
            raise ValueError(f"{field} {name} must be in 0..255, got {channel}")
    return (r, g, b)


def _parse_char(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 1:
        raise ValueError(f"{field} must be a single character string")
    return value


def load_config(path: str | Path | None = None) -> SnakeConfig:
    """Load settings from YAML. Defaults to snake/config.yaml next to this module."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    grid = _require_section(raw, "grid")
    episode = _require_section(raw, "episode")
    rewards = _require_section(raw, "rewards")
    snake = _require_section(raw, "snake")
    render = _require_section(raw, "render")
    colors = _require_section(raw, "colors")
    ansi = _require_section(raw, "ansi")
    observation = _require_section(raw, "observation")
    demo = _require_section(raw, "demo")

    initial_length = int(snake["initial_length"])
    if initial_length < 1:
        raise ValueError("snake.initial_length must be at least 1")

    observation_size = int(observation["size"])
    if observation_size < 1:
        raise ValueError("observation.size must be at least 1")

    return SnakeConfig(
        grid_width=int(grid["width"]),
        grid_height=int(grid["height"]),
        max_episode_steps=int(episode["max_steps"]),
        reward_food=float(rewards["food"]),
        reward_death=float(rewards["death"]),
        reward_step=float(rewards["step"]),
        initial_snake_length=initial_length,
        start_direction=_parse_direction(str(snake["start_direction"])),
        render_fps=int(render["fps"]),
        cell_size_px=int(render["cell_size_px"]),
        cell_padding_px=int(render["cell_padding_px"]),
        window_title=str(render["window_title"]),
        color_background=_parse_rgb(colors["background"], field="colors.background"),
        color_grid_line=_parse_rgb(colors["grid_line"], field="colors.grid_line"),
        color_snake=_parse_rgb(colors["snake"], field="colors.snake"),
        color_head=_parse_rgb(colors["head"], field="colors.head"),
        color_food=_parse_rgb(colors["food"], field="colors.food"),
        ansi_head=_parse_char(ansi["head"], field="ansi.head"),
        ansi_body=_parse_char(ansi["body"], field="ansi.body"),
        ansi_food=_parse_char(ansi["food"], field="ansi.food"),
        ansi_empty=_parse_char(ansi["empty"], field="ansi.empty"),
        observation_size=observation_size,
        demo_num_episodes=int(demo["num_episodes"]),
        demo_pause_between_episodes_sec=float(demo["pause_between_episodes_sec"]),
    )
