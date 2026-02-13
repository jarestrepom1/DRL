import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack, TransformReward


def make_atari_env(
    game_id: str,
    seed: int,
    frame_skip: int = 4,
    clip_rewards: bool = False,
    render_mode: str | None = None,
):
    env = gym.make(
        game_id,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode,
    )
    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        grayscale_obs=True,
        screen_size=84,
        scale_obs=False,
        terminal_on_life_loss=False,
    )
    if clip_rewards:
        env = TransformReward(env, lambda r: max(-1.0, min(1.0, r)))
    env = FrameStack(env, num_stack=4)
    env.reset(seed=seed)
    return env
