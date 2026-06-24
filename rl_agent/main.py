"""Train or evaluate a DQN agent on SnakeEnv.

Usage (from repository root):
    python -m rl_agent.main train --episodes 500
    python -m rl_agent.main test --checkpoint rl_agent/checkpoints/dqn_final.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from rl_agent.agent import DQNAgent
from rl_agent.observation import ObservationBuilder, STANDARD_OBS_SIZE
from snake.snake_env import SnakeEnv

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"


def train(
    *,
    episodes: int,
    max_steps: int | None,
    save_every: int,
    checkpoint_dir: Path,
    seed: int,
    log_every: int,
    render: bool,
    render_fps: int | None,
) -> None:
    """DQN training loop. Pass render=True to watch the agent learn (slower)."""
    env_kwargs: dict = {}
    if render:
        env_kwargs["render_mode"] = "human"
        if render_fps is not None:
            env_kwargs["render_fps"] = render_fps

    env = SnakeEnv(**env_kwargs)
    obs_builder = ObservationBuilder()
    agent = DQNAgent(
        state_dim=STANDARD_OBS_SIZE,
        action_dim=env.action_space.n,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scores: list[int] = []
    start = time.perf_counter()

    try:
        for episode in range(episodes):
            state, _info = env.reset(seed=seed + episode)
            observation = obs_builder.build(state)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            loss_values: list[float] = []

            while not (terminated or truncated):
                if max_steps is not None and steps >= max_steps:
                    break

                action = agent.select_action(observation)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_observation = obs_builder.build(next_state)
                done = terminated or truncated

                agent.store_transition(
                    observation, action, reward, next_observation, done
                )
                loss = agent.train_step()
                if loss is not None:
                    loss_values.append(loss)

                observation = next_observation
                total_reward += reward
                steps += 1

            scores.append(int(info["score"]))

            if (episode + 1) % log_every == 0:
                window = scores[-log_every:]
                mean_score = float(np.mean(window))
                mean_loss = float(np.mean(loss_values)) if loss_values else float("nan")
                elapsed = time.perf_counter() - start
                print(
                    f"Episode {episode + 1}/{episodes} | "
                    f"score={info['score']} | reward={total_reward:.1f} | "
                    f"epsilon={agent.epsilon:.3f} | "
                    f"avg_score({log_every})={mean_score:.2f} | "
                    f"loss={mean_loss:.4f} | "
                    f"elapsed={elapsed:.1f}s"
                )

            if save_every > 0 and (episode + 1) % save_every == 0:
                ckpt = checkpoint_dir / f"dqn_ep{episode + 1}.pt"
                agent.save(ckpt)
                print(f"Saved checkpoint: {ckpt}")

        final_path = checkpoint_dir / "dqn_final.pt"
        agent.save(final_path)
        print(f"Training complete. Final checkpoint: {final_path}")
    finally:
        env.close()


def test(
    *,
    checkpoint: Path,
    episodes: int,
    seed: int,
    render_fps: int | None,
) -> None:
    """Load weights and watch the agent play with pygame rendering."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    kwargs: dict = {"render_mode": "human"}
    if render_fps is not None:
        kwargs["render_fps"] = render_fps

    env = SnakeEnv(**kwargs)
    obs_builder = ObservationBuilder()
    agent = DQNAgent(
        state_dim=STANDARD_OBS_SIZE,
        action_dim=env.action_space.n,
    )
    agent.load(checkpoint)
    agent.epsilon = 0.0

    try:
        for episode in range(episodes):
            state, _info = env.reset(seed=seed + episode)
            observation = obs_builder.build(state)
            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action = agent.select_action(observation, greedy=True)
                state, reward, terminated, truncated, info = env.step(action)
                observation = obs_builder.build(state)
                total_reward += reward

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"score={info['score']} | steps={info['steps']} | "
                f"reward={total_reward:.1f}"
            )
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or test a DQN Snake agent.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="DQN training (headless by default)")
    train_parser.add_argument("--episodes", type=int, default=500)
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on steps per episode (defaults to env max)",
    )
    train_parser.add_argument("--save-every", type=int, default=100)
    train_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument(
        "--render",
        action="store_true",
        help="Open a Pygame window and show each step while training (much slower)",
    )
    train_parser.add_argument(
        "--render-fps",
        type=int,
        default=None,
        help="Override render FPS from snake/config.yaml (only with --render)",
    )

    test_parser = subparsers.add_parser("test", help="Evaluate with pygame rendering")
    test_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR / "dqn_final.pt",
    )
    test_parser.add_argument("--episodes", type=int, default=5)
    test_parser.add_argument("--seed", type=int, default=0)
    test_parser.add_argument(
        "--render-fps",
        type=int,
        default=None,
        help="Override render FPS from snake/config.yaml",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "train":
        train(
            episodes=args.episodes,
            max_steps=args.max_steps,
            save_every=args.save_every,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            log_every=args.log_every,
            render=args.render,
            render_fps=args.render_fps,
        )
    elif args.mode == "test":
        test(
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            seed=args.seed,
            render_fps=args.render_fps,
        )
    else:
        raise ValueError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()
