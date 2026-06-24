"""Run a Snake agent against SnakeEnv.

This file is the place to put *your* agent logic. The environment (snake_env.py)
only defines the game rules — it does not include a trained brain.

Usage (from repo root):
    python -m snake.play                  # random agent + pygame window
    python -m snake.play --agent rule     # simple hand-coded agent
    python -m snake.play --episodes 10 --seed 0
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

import numpy as np

from rl_agent.observation import ObservationBuilder
from snake.snake_env import SnakeEnv

# Observation layout (length 11) — see README_CODE_GUIDE.md
IDX_DANGER_STRAIGHT = 0
IDX_DANGER_LEFT = 1
IDX_DANGER_RIGHT = 2
# indices 3–6: heading one-hot [up, right, down, left]
# indices 7–10: food relative [left, right, ahead, behind]

ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2


class Agent(ABC):
    """Implement act() to plug your policy into the game loop below."""

    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """Return 0=straight, 1=left, 2=right."""


class RandomAgent(Agent):
    """Picks a random legal action each step — not intelligent, just for testing."""

    def __init__(self, action_size: int = 3) -> None:
        self._rng = np.random.default_rng()

    def act(self, observation: np.ndarray) -> int:
        return int(self._rng.integers(0, 3))


class RuleBasedAgent(Agent):
    """Tiny hand-coded policy: turn toward food when safe, else avoid walls."""

    def act(self, observation: np.ndarray) -> int:
        danger_straight = observation[IDX_DANGER_STRAIGHT]
        danger_left = observation[IDX_DANGER_LEFT]
        danger_right = observation[IDX_DANGER_RIGHT]

        food_left, food_right, food_ahead, food_behind = observation[7:11]

        # Prefer moving toward food if that direction is not dangerous.
        if food_ahead and not danger_straight:
            return ACTION_STRAIGHT
        if food_left and not danger_left:
            return ACTION_LEFT
        if food_right and not danger_right:
            return ACTION_RIGHT

        # Survival fallback: pick any non-dangerous move.
        if not danger_straight:
            return ACTION_STRAIGHT
        if not danger_left:
            return ACTION_LEFT
        if not danger_right:
            return ACTION_RIGHT

        return ACTION_STRAIGHT  # doomed — no safe move


AGENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "rule": RuleBasedAgent,
}


def run_episode(
    env: SnakeEnv,
    agent: Agent,
    obs_builder: ObservationBuilder,
    *,
    seed: int | None = None,
) -> tuple[float, dict]:
    state, info = env.reset(seed=seed)
    observation = obs_builder.build(state)
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.act(observation)
        state, reward, terminated, truncated, info = env.step(action)
        observation = obs_builder.build(state)
        total_reward += reward

    return total_reward, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Snake with a chosen agent.")
    parser.add_argument(
        "--agent",
        choices=sorted(AGENTS),
        default="random",
        help="Which agent implementation to use (default: random)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render",
        choices=["human", "ansi", "none"],
        default="human",
        help="human=pygame window, ansi=terminal, none=headless",
    )
    args = parser.parse_args()

    render_mode = None if args.render == "none" else args.render
    env = SnakeEnv(render_mode=render_mode)
    agent = AGENTS[args.agent]()
    obs_builder = ObservationBuilder()

    try:
        for episode in range(args.episodes):
            total_reward, info = run_episode(
                env, agent, obs_builder, seed=args.seed + episode
            )
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"agent={args.agent} | score={info['score']} | "
                f"steps={info['steps']} | reward={total_reward:.1f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
