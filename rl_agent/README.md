# DQN Agent for Snake

Modular Deep Q-Network (DQN) training and evaluation for the Gymnasium `SnakeEnv`. The environment returns **raw game state**; this package owns all feature engineering, the neural policy, and training loops.

> **Preview tip:** Open this file with **Markdown: Open Preview to the Side** (`Ctrl+K V`) or **Markdown Preview Enhanced** (`Ctrl+Shift+P` → “Markdown Preview Enhanced: Open Preview”) to render the math formulas below.

## Architecture

```
snake/snake_env.py          →  raw state dict (no ML features)
        │
        ▼
rl_agent/observation.py     →  ObservationBuilder → 11-dim vector
        │
        ▼
rl_agent/agent.py           →  DQN (MLP + replay buffer + target net)
        │
        ▼
rl_agent/main.py            →  train (headless) / test (rendered)
```

| File | Role |
|------|------|
| `observation.py` | `ObservationBuilder` — maps raw state to engineered features |
| `agent.py` | `QNetwork`, `ReplayBuffer`, `DQNAgent` |
| `main.py` | CLI entry point for training and evaluation |
| `checkpoints/` | Saved model weights (created during training) |

---

## Setup

From the repository root, activate your virtual environment and install dependencies:

```bash
pip install gymnasium numpy pyyaml torch pygame
```

- **Training** needs: `gymnasium`, `numpy`, `pyyaml`, `torch` (Pygame is not loaded).
- **Testing / rendering** also needs: `pygame`.

Game parameters (grid size, rewards, episode length) live in `snake/config.yaml` and are loaded automatically by `SnakeEnv`.

---

## Quick start

### Train (headless, fastest)

No window opens — `render_mode=None` keeps Pygame unloaded:

```bash
python -m rl_agent.main train --episodes 500 --save-every 100
```

Checkpoints are written to `rl_agent/checkpoints/` (`dqn_ep100.pt`, …, `dqn_final.pt`).

### Train with live visualization

Add `--render` to watch the agent learn step-by-step in a Pygame window. Learning still runs each step; only the display is added:

```bash
python -m rl_agent.main train --episodes 50 --render
python -m rl_agent.main train --episodes 50 --render --render-fps 6
```

This is **much slower** than headless training because each frame waits for the display clock (`render.fps` in `snake/config.yaml`, default 10 FPS). Use it for debugging or short demos; use headless mode for long runs.

### Test (watch a trained agent play)

Loads weights and opens a Pygame window:

```bash
python -m rl_agent.main test --checkpoint rl_agent/checkpoints/dqn_final.pt --episodes 5
```

Use `--render-fps 15` to slow down or speed up the window.

---

## Raw state vs. observations

`SnakeEnv.reset()` and `step()` return a dictionary:

| Key | Type | Description |
|-----|------|-------------|
| `snake_body_coords` | `list[(x, y)]` | Head at index 0, tail at the end |
| `food_coords` | `(x, y)` | Food cell |
| `current_direction` | `int` | 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT |
| `grid_dimensions` | `(width, height)` | Board size |

Build the classic 11-feature vector in your agent code:

```python
from rl_agent.observation import ObservationBuilder
from snake.snake_env import SnakeEnv

env = SnakeEnv()
obs_builder = ObservationBuilder()

state, info = env.reset(seed=0)
obs = obs_builder.build(state)  # shape (11,)
```

### Standard 11-feature layout

| Index | Feature |
|------:|---------|
| 0–2 | Danger straight / left / right (1.0 = collision one cell ahead) |
| 3–6 | Heading one-hot [UP, RIGHT, DOWN, LEFT] |
| 7–10 | Food relative to heading [left, right, ahead, behind] |

To experiment with new representations, subclass or replace `ObservationBuilder.build()` without touching the environment.

---

## DQN mathematics

This agent implements **vanilla DQN** with experience replay and a **target network** for stable Q-learning on a discrete action space $\mathcal{A} = \{0, 1, 2\}$ (straight, left, right).

| Symbol | Meaning |
|--------|---------|
| $s$, $s'$ | Current and next state (11-dim feature vector) |
| $a$, $a'$ | Current and candidate actions |
| $r$ | Immediate reward |
| $\gamma$ | Discount factor (default `0.9`) |
| $\theta$ | Policy-network weights |
| $\bar{\theta}$ | Target-network weights (lagged copy of $\theta$) |
| $Q^*(s, a)$ | Optimal action-value function |
| $Q_\theta(s, a)$ | Neural-network approximation of $Q^*$ |

---

### Markov Decision Process

At each timestep $t$ the agent observes state $s_t$, selects action $a_t$, receives reward $r_t$, and transitions to $s_{t+1}$. The goal is to learn the optimal action-value function:

$$
Q^*(s, a) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s,\; a_t = a\right]
$$

$Q^*(s, a)$ is the expected discounted sum of future rewards when starting in state $s$, taking action $a$, and behaving optimally thereafter.

Discount factor $\gamma \in (0, 1]$ (default `0.9`).

---

### Bellman optimality equation

The optimal Q-function satisfies:

$$
Q^*(s, a) = \mathbb{E}_{s' \sim p(s' \mid s,a)}\left[r + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a')\right]
$$

The value of $(s, a)$ equals the immediate reward $r$ plus the discounted value of the best action in the next state $s'$.

DQN approximates $Q^*(s,a)$ with a neural network $Q_\theta(s,a)$ parameterized by $\theta$ (the **policy network**).

---

### Temporal-difference target

For a transition $(s, a, r, s', \text{done})$ sampled from the replay buffer, the **one-step Bellman target** is:

$$
y = r + \gamma (1 - \text{done}) \max_{a' \in \mathcal{A}} Q_{\bar{\theta}}(s', a')
$$

- If the episode ended ($\text{done} = 1$): $y = r$
- Otherwise: $y = r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a')$

$Q_{\bar{\theta}}$ is the **target network** — a lagged copy of the policy network. Weights $\bar{\theta}$ are synced from $\theta$ every `target_update_freq` gradient steps (default 100). Using $\bar{\theta}$ instead of $\theta$ on the bootstrap term stabilizes training.

In code (`agent.py`):

```python
next_q = target_net(next_states).max(dim=1).values
targets = rewards + gamma * next_q * (1.0 - dones)
```

The `dones` mask zeroes out bootstrapping on terminal transitions.

---

### Loss function (MSE)

Training minimizes the mean squared error between the predicted Q-value for the taken action and the Bellman target:

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \left( Q_\theta(s_i, a_i) - y_i \right)^2
$$

$B$ is the minibatch size (default 64). Gradients flow only through $Q_\theta(s_i, a_i)$ (the action that was actually taken), not through the max in the target — standard semi-gradient Q-learning.

---

### Experience replay

Transitions are stored in a FIFO replay buffer (capacity 100,000) and uniformly sampled each step. Replay breaks temporal correlation in consecutive frames and reuses past data, improving sample efficiency.

---

### Epsilon-greedy exploration

During training, with probability $\epsilon$ the agent picks a random action; otherwise:

$$
a_t = \arg\max_{a \in \mathcal{A}} Q_\theta(s_t, a)
$$

$\epsilon$ starts at `1.0`, multiplies by `0.995` after each `train_step`, and is floored at `0.01`. During evaluation (`test` mode), $\epsilon = 0$ (purely greedy).

---

### Network architecture

`QNetwork` is a 3-layer MLP:

$$
\mathbb{R}^{11} \xrightarrow{\text{Linear + ReLU}} \mathbb{R}^{256} \xrightarrow{\text{Linear + ReLU}} \mathbb{R}^{256} \xrightarrow{\text{Linear}} \mathbb{R}^{3}
$$

| Layer | Size | Activation |
|-------|-----:|------------|
| Input | 11 | — (observation vector) |
| Hidden 1 | 256 | ReLU |
| Hidden 2 | 256 | ReLU |
| Output | 3 | none (one Q-value per action) |

Output dimension 3 matches the relative action space (straight, left, right).

---

## Training hyperparameters

Defaults are set in `DQNAgent.__init__` (`agent.py`):

| Parameter | Default | Meaning |
|-----------|--------:|---------|
| `learning_rate` | 1e-3 | Adam step size |
| `gamma` | 0.9 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Per-train-step multiplicative decay |
| `buffer_capacity` | 100,000 | Replay buffer size |
| `batch_size` | 64 | Minibatch size |
| `target_update_freq` | 100 | Steps between target-net syncs |
| `hidden_dim` | 256 | MLP hidden width |

Tune these by constructing `DQNAgent(...)` with custom kwargs in a fork of `main.py`, or extend the CLI.

---

## Programmatic usage

```python
from rl_agent.agent import DQNAgent
from rl_agent.observation import ObservationBuilder
from snake.snake_env import SnakeEnv

env = SnakeEnv(render_mode=None)
builder = ObservationBuilder()
agent = DQNAgent(state_dim=11, action_dim=3)

state, _ = env.reset()
obs = builder.build(state)

for _ in range(1000):
    action = agent.select_action(obs)
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_obs = builder.build(next_state)
    done = terminated or truncated

    agent.store_transition(obs, action, reward, next_obs, done)
    agent.train_step()

    obs = next_obs
    if done:
        state, _ = env.reset()
        obs = builder.build(state)

agent.save("my_checkpoint.pt")
env.close()
```

---

## Extending for research

1. **New observations** — Add a method to `ObservationBuilder` (e.g. grid flatten, distance features) and update `state_dim` in `DQNAgent`.
2. **Double DQN / Dueling / PER** — Extend `DQNAgent.train_step()`; the replay buffer and env decoupling stay the same.
3. **Different algorithms** — Keep `observation.py` and swap `agent.py` for PPO, A2C, etc.
4. **Curriculum** — Pass `grid_width=` / `grid_height=` to `SnakeEnv(...)` in `main.py`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: torch` | `pip install torch` |
| Pygame window during training | Use `train` mode only; never pass `render_mode="human"` in the training loop |
| Poor scores after few episodes | DQN needs hundreds of episodes on Snake; increase `--episodes` |
| Checkpoint not found | Pass `--checkpoint` with the path printed at end of training |

---

## Related docs

- Environment usage: `snake/README_USAGE.md`
- Environment internals: `snake/README_CODE_GUIDE.md`
