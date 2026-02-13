import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor


def make_lunarlander_env(
    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 0.0,
    turbulence_power: float = 0.0,
    max_episode_steps: int = 600,
    seed: int | None = None,
    render_mode: str | None = None,
):
    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        render_mode=render_mode,
    )
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env
