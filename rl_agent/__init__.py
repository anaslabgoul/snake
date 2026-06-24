"""Deep Q-Network agent and observation builders for SnakeEnv."""

from rl_agent.observation import ObservationBuilder, RawState, STANDARD_OBS_SIZE

__all__ = [
    "DQNAgent",
    "ObservationBuilder",
    "QNetwork",
    "RawState",
    "ReplayBuffer",
    "STANDARD_OBS_SIZE",
]


def __getattr__(name: str):
    if name in ("DQNAgent", "QNetwork", "ReplayBuffer"):
        from rl_agent.agent import DQNAgent, QNetwork, ReplayBuffer

        exports = {"DQNAgent": DQNAgent, "QNetwork": QNetwork, "ReplayBuffer": ReplayBuffer}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
