"""Snake reinforcement-learning environment."""

from snake.config import DEFAULT_CONFIG_PATH, SnakeConfig, load_config
from snake.snake_env import SnakeEnv

__all__ = ["DEFAULT_CONFIG_PATH", "SnakeConfig", "SnakeEnv", "load_config"]
