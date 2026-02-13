import os
from gymnasium.wrappers import RecordVideo


def wrap_record_video(env, video_dir: str, name_prefix: str = "eval"):
    os.makedirs(video_dir, exist_ok=True)
    return RecordVideo(env, video_dir=video_dir, name_prefix=name_prefix)
