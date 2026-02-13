import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.replay_buffers.prioritized_replay import PrioritizedReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@dataclass
class EpsilonSchedule:
    eps_start: float
    eps_end: float
    decay_steps: int

    def value(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.eps_end
        frac = step / float(self.decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)


@dataclass
class PERBetaSchedule:
    beta_start: float
    beta_end: float
    beta_steps: int

    def value(self, step: int) -> float:
        if step >= self.beta_steps:
            return self.beta_end
        frac = step / float(self.beta_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)


class DQNPerAgent:
    def __init__(
        self,
        obs_shape,
        num_actions: int,
        device: str,
        gamma: float = 0.99,
        lr: float = 1e-4,
        target_update_interval: int = 10_000,
        buffer_size: int = 200_000,
        alpha: float = 0.6,
        eps_schedule: EpsilonSchedule | None = None,
        beta_schedule: PERBetaSchedule | None = None,
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.step_count = 0

        in_channels = obs_shape[0]
        self.online = QNetwork(in_channels, num_actions).to(device)
        self.target = QNetwork(in_channels, num_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = PrioritizedReplayBuffer(buffer_size, alpha=alpha)

        self.eps_schedule = eps_schedule or EpsilonSchedule(1.0, 0.05, 1_000_000)
        self.beta_schedule = beta_schedule or PERBetaSchedule(0.4, 1.0, 1_000_000)

    def select_action(self, obs: np.ndarray) -> int:
        eps = self.eps_schedule.value(self.step_count)
        if np.random.rand() < eps:
            return np.random.randint(self.num_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch_size: int):
        if len(self.replay) < batch_size:
            return None

        beta = self.beta_schedule.value(self.step_count)
        obs, actions, rewards, next_obs, dones, indices, weights = self.replay.sample(batch_size, beta)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online(obs_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target(next_obs_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        td_error = target - q_values
        loss = F.smooth_l1_loss(q_values, target, reduction="none")
        loss = (loss * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        priorities = td_error.detach().abs().cpu().numpy().squeeze() + 1e-6
        self.replay.update_priorities(indices, priorities)

        if self.step_count % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item()), float(td_error.detach().abs().mean().item())

    def save_state(self):
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "replay": self.replay.state_dict(),
            "gamma": self.gamma,
            "target_update_interval": self.target_update_interval,
        }

    def load_state(self, state):
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = int(state["step_count"])
        self.replay.load_state_dict(state["replay"])
        self.gamma = float(state.get("gamma", self.gamma))
        self.target_update_interval = int(state.get("target_update_interval", self.target_update_interval))
