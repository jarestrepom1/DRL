import json
import os
import torch


def save_checkpoint(state: dict, checkpoint_dir: str, step: int) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pth")
    torch.save(state, ckpt_path)
    return ckpt_path


def load_checkpoint(ckpt_path: str, device: str):
    return torch.load(ckpt_path, map_location=device)


def save_config(config: dict, checkpoint_dir: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path
