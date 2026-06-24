# SnakeEnv — Developer Code Guide

This document explains how `snake_env.py` is structured so you can tune gameplay, observations, and rendering without spelunking the whole file.

## Architecture overview

The environment splits cleanly into three layers:

```
┌─────────────────────────────────────────┐
│  Gymnasium API                          │
│  reset() / step() / render() / close()  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  Core game logic (headless, numpy)      │
│  _resolve_direction, _is_collision,     │
│  _spawn_food, _sync_grid, _get_state    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  Optional rendering                     │
│  _render_ansi(), _render_pygame()       │
└─────────────────────────────────────────┘
```

**Decoupling rule:** `step()` updates game state through the `_`-prefixed helpers only. Rendering methods read `self._grid`, `self._snake`, and `self._food` but never mutate gameplay. Pygame is imported lazily inside `_init_pygame()` so headless training never loads it.

Rendering is triggered only when `render_mode` is set:

- `None` (default): no output; fastest path for training.
- `"ansi"`: `_render_ansi()` prints a character grid each step.
- `"human"`: `_render_pygame()` opens a window; `step()` calls `render()` automatically.

---

## Configuration file (`config.yaml`)

All tunable game parameters live in **`snake/config.yaml`**. The loader in **`snake/config.py`** parses YAML into a frozen `SnakeConfig` dataclass. `SnakeEnv` loads the default file automatically on construction.

| Goal | YAML path |
|------|-----------|
| Grid width / height | `grid.width`, `grid.height` |
| Episode length cap | `episode.max_steps` |
| Visual speed (FPS) | `render.fps` |
| Pixel size of cells | `render.cell_size_px` |
| Cell draw padding | `render.cell_padding_px` |
| Window title | `render.window_title` |
| Food / death / step rewards | `rewards.food`, `rewards.death`, `rewards.step` |
| Starting snake length | `snake.initial_length` |
| Starting direction | `snake.start_direction` (`up` / `right` / `down` / `left`) |
| Pygame colors (RGB) | `colors.background`, `colors.snake`, `colors.head`, `colors.food`, `colors.grid_line` |
| Terminal characters | `ansi.head`, `ansi.body`, `ansi.food`, `ansi.empty` |
| Observation vector length | `observation.size` (used by `rl_agent` baseline builder) |
| Demo loop settings | `demo.num_episodes`, `demo.pause_between_episodes_sec` |

Example — edit `config.yaml`:

```yaml
grid:
  width: 30
  height: 30
render:
  fps: 6
rewards:
  food: 15.0
```

Or override at runtime without editing the file:

```python
env = SnakeEnv(render_mode="human", grid_width=30, grid_height=30, render_fps=6)
env = SnakeEnv(config_path="experiments/hard_mode.yaml")
```

Programmatic access:

```python
from snake import load_config, DEFAULT_CONFIG_PATH

print(DEFAULT_CONFIG_PATH)  # .../snake/config.yaml
cfg = load_config()
print(cfg.reward_food, cfg.grid_width)
```

---

## Core game state

| Variable | Type | Role |
|----------|------|------|
| `self._grid` | `np.ndarray (H, W)` int8 | Board snapshot: `CELL_EMPTY`, `CELL_SNAKE`, `CELL_FOOD` |
| `self._snake` | `deque[(x, y)]` | Head at index 0, tail at the end |
| `self._direction` | `int 0–3` | Absolute heading (`DIR_UP` … `DIR_LEFT`) |
| `self._food` | `(x, y)` | Food cell coordinates |
| `self._score` | `int` | Foods eaten this episode |
| `self._steps` | `int` | Steps taken (used for truncation) |

`_sync_grid()` rebuilds `_grid` from snake and food after every valid move. Collision checks use coordinates directly via `_is_collision()` rather than reading the grid, so logic stays correct even mid-step.

---

## Action space — why Discrete(3)?

```python
self.action_space = spaces.Discrete(3)
# 0 = straight, 1 = left turn, 2 = right turn
```

**Why not Discrete(4) absolute directions?**

1. **Invalid actions removed:** With absolute controls, “down” while heading up is a no-op or must be filtered. Relative turns make every action meaningful.
2. **Smaller branching factor:** Three actions vs four speeds up early exploration.
3. **Common RL Snake convention:** Matches widely used tutorial environments where the policy chooses “continue / turn left / turn right”.

Implementation lives in `_resolve_direction()`:

```python
if action == ACTION_STRAIGHT:
    return self._direction
if action == ACTION_LEFT:
    return (self._direction - 1) % 4
if action == ACTION_RIGHT:
    return (self._direction + 1) % 4
```

To switch to absolute Discrete(4), replace `_resolve_direction()` with a mapping from action → `DIR_*` and add a guard that rejects 180° reversals.

---

## Observation space — raw state (decoupled)

```python
self.observation_space = spaces.Dict({...})  # raw game state
```

`reset()` and `step()` return a **state dictionary**, not engineered features:

| Key | Meaning |
|-----|---------|
| `snake_body_coords` | List of `(x, y)` from head to tail |
| `food_coords` | `(x, y)` food position |
| `current_direction` | `0–3` absolute heading |
| `grid_dimensions` | `(width, height)` |

Feature engineering lives in **`rl_agent/observation.py`** (`ObservationBuilder`). The default builder reproduces the classic 11-feature vector:

| Index | Name | Meaning |
|------:|------|---------|
| 0 | danger straight | `1.0` if wall or body directly ahead |
| 1 | danger left | `1.0` if hazard after a left turn |
| 2 | danger right | `1.0` if hazard after a right turn |
| 3–6 | heading one-hot | `[UP, RIGHT, DOWN, LEFT]` — exactly one `1.0` |
| 7–10 | food direction | Relative to current heading: `[left, right, ahead, behind]` |

### Tweaking observations

Edit `ObservationBuilder.build()` in `rl_agent/observation.py` — no environment changes required.

| Change | Where to edit |
|--------|----------------|
| Add full grid flatten | Extend `ObservationBuilder.build()`; update DQN `state_dim` |
| Add distance to food | Compute in `ObservationBuilder` |
| Remove heading one-hot | Drop from concat; set `state_dim` to 7 |
| Binary → normalized grid | Return grid channels from builder |

---

## Reward logic

Applied in `step()` using values from `self._cfg` (loaded from `config.yaml`):

```python
reward = self._cfg.reward_step   # default each step
if collision:
    reward = self._cfg.reward_death
elif ate_food:
    reward = self._cfg.reward_food
```

Edit `rewards.*` in `config.yaml`. For shaped rewards (e.g. distance-to-food bonus), add a helper and combine it in the `else` branch of the collision check in `step()`.

---

## Episode termination

| Flag | Condition |
|------|-----------|
| `terminated` | Wall or self collision (`_is_collision`) |
| `truncated` | `self._steps >= self.max_episode_steps` without dying |

Adjust `episode.max_steps` in `config.yaml` or pass `max_episode_steps=` to `SnakeEnv(...)`.

---

## Rendering internals

### ANSI (`render_mode="ansi"`)

`_render_ansi()` maps cells to characters:

- `@` head, `#` body, `*` food, `.` empty

Change characters or add borders in that method only.

### Pygame (`render_mode="human"`)

- Lazy init: `_init_pygame()` imports pygame and creates the window.
- Draw loop: `_render_pygame()` → grid lines → `_draw_cell()` for food and snake.
- Frame rate: `self._clock.tick(self.render_fps)` — from `render.fps` in config.
- Colors: edit the `colors.*` section in `config.yaml`.

Window size is derived: `(grid_width * cell_size_px, grid_height * cell_size_px)`.

---

## Randomness and reproducibility

`reset(seed=...)` reseeds `self._rng` (`numpy.random.Generator`). Food placement in `_spawn_food()` uses `self._rng.integers`. For vectorized or parallel envs, always pass distinct seeds per worker.

---

## File map

| Section | Methods / symbols |
|---------|-------------------|
| `config.yaml` | All tunable game parameters |
| `config.py` | `SnakeConfig`, `load_config()` |
| `SnakeEnv.__init__` | Loads config, spaces, grid buffers |
| `reset` / `step` | Public API |
| Game logic | `_resolve_direction` through `_build_info` |
| Raw state | `_get_state` |
| Feature engineering | `rl_agent/observation.py` — `ObservationBuilder` |
| Rendering | `_render_ansi`, `_init_pygame`, `_render_pygame`, `_draw_cell` |
| Demo | `if __name__ == "__main__"` random agent loop |

---

## Suggested next customizations

1. **Grid observation:** Return a one-hot or multi-channel `(H, W, C)` tensor for CNN policies.
2. **Variable speed:** Skip render tick and add `steps_per_action > 1` for arcade-style pacing during human play only.
3. **Curriculum:** Start with smaller `grid_width`/`grid_height` in training scripts, then increase via env kwargs.
