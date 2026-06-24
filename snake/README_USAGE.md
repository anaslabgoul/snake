# SnakeEnv — Usage Guide

A headless-first Snake environment for reinforcement learning, built on [Gymnasium](https://gymnasium.farama.org/).

## Dependencies

```bash
pip install gymnasium numpy pyyaml pygame torch
```

Pygame is only required when you use `render_mode="human"`. The environment loads `snake/config.yaml` automatically via PyYAML. Headless training needs gymnasium, numpy, and pyyaml only.

For a full DQN training pipeline, see **`rl_agent/README.md`**.

## Configuration

All gameplay and rendering defaults live in `snake/config.yaml` (grid size, rewards, colors, FPS, etc.). Edit that file to tune the environment without changing Python code.

Use a custom config path:

```python
env = SnakeEnv(config_path="path/to/my_config.yaml")
```

Or load and pass a config object:

```python
from snake import load_config, SnakeEnv

cfg = load_config("path/to/my_config.yaml")
env = SnakeEnv(config=cfg)
```

Constructor keyword arguments (`grid_width=`, `render_fps=`, …) still override individual values from the config file.

## Watch the game run

There is **no trained AI** in this repo. When you run the demo, the snake moves **randomly** — that is only to show the environment works.

| What runs | Where | What it does |
|-----------|-------|----------------|
| Random demo | `python -m snake.snake_env` | `env.action_space.sample()` — random moves |
| Agent runner | `python -m snake.play` | Same random agent by default; try `--agent rule` for a simple hand-coded policy |

To build your own agent, implement the `Agent` class in `snake/play.py` (or your own script) and call `env.step(action)` in a loop. See **Code your own agent** below.

### Pygame window (recommended for humans)

From the repository root:

```bash
python -m snake.snake_env
```

Or from Python:

```python
from snake.snake_env import SnakeEnv

env = SnakeEnv(render_mode="human")
state, info = env.reset(seed=0)

done = False
while not done:
    action = env.action_space.sample()  # random agent
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

### Terminal (ASCII) rendering

```python
env = SnakeEnv(render_mode="ansi")
```

Each step prints a text grid to the terminal.

### Headless training (default)

```python
env = SnakeEnv()  # render_mode=None — fastest for RL loops
```

## Integrate with an RL library

### Stable Baselines3 example

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from snake.snake_env import SnakeEnv

def make_env():
    return SnakeEnv()

vec_env = make_vec_env(make_env, n_envs=8)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("snake_ppo")

# Evaluate with rendering
eval_env = SnakeEnv(render_mode="human")
obs, _ = eval_env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, _ = eval_env.reset()
eval_env.close()
```

### Custom agent loop

The environment returns **raw game state** each step. Build your own feature vector (or use `rl_agent.observation.ObservationBuilder`) before passing data to a policy.

```python
from rl_agent.observation import ObservationBuilder
from snake.snake_env import SnakeEnv

class MyAgent:
    def act(self, observation):
        # observation: numpy array of shape (11,) when using ObservationBuilder
        return 0  # replace with your logic or neural network

env = SnakeEnv(render_mode="human")
agent = MyAgent()
obs_builder = ObservationBuilder()
state, info = env.reset(seed=42)
obs = obs_builder.build(state)

while True:
    action = agent.act(obs)
    state, reward, terminated, truncated, info = env.step(action)
    obs = obs_builder.build(state)
    if terminated or truncated:
        state, info = env.reset()
        obs = obs_builder.build(state)
```

A ready-made runner with `RandomAgent` and `RuleBasedAgent` lives in `snake/play.py`:

```bash
python -m snake.play --agent random --render human
python -m snake.play --agent rule --episodes 10
```

Subclass `Agent` in that file (or import it) to drop in your own policy.

## Action space

| Index | Meaning |
|------:|---------|
| 0 | Go straight (keep current heading) |
| 1 | Turn left (90° relative to heading) |
| 2 | Turn right (90° relative to heading) |

Relative actions prevent instant 180° turns and typically learn faster than absolute Up/Down/Left/Right controls.

## Rewards

| Event | Reward |
|-------|--------:|
| Eat food | +10 |
| Die (wall or self) | -10 |
| Each surviving step | -0.1 |
