import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float):
        idx = self.data_pointer + self.capacity - 1
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        return idx

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, value: float):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.size = 0
        self.obs = None
        self.next_obs = None
        self.actions = None
        self.rewards = None
        self.dones = None

    def _init_storage(self, obs_shape, obs_dtype):
        self.obs = np.zeros((self.capacity,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((self.capacity,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        if self.obs is None:
            self._init_storage(obs.shape, obs.dtype)

        idx = self.tree.data_pointer
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        priority = self.max_priority ** self.alpha
        self.tree.add(priority)
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        batch = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, priority, data_idx = self.tree.get(value)
            indices[i] = idx
            priorities[i] = priority
            batch.append(data_idx)

        probs = priorities / self.tree.total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max() + 1e-8

        obs = self.obs[batch]
        next_obs = self.next_obs[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        dones = self.dones[batch]

        return obs, actions, rewards, next_obs, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority = float(priority)
            priority = max(priority, 1e-6)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)

    def __len__(self):
        return self.size

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "max_priority": self.max_priority,
            "size": self.size,
            "tree": self.tree.tree,
            "data_pointer": self.tree.data_pointer,
            "obs": self.obs,
            "next_obs": self.next_obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
        }

    def load_state_dict(self, state):
        self.capacity = int(state["capacity"])
        self.alpha = float(state["alpha"])
        self.max_priority = float(state["max_priority"])
        self.size = int(state["size"])
        self.tree = SumTree(self.capacity)
        self.tree.tree = state["tree"]
        self.tree.data_pointer = int(state["data_pointer"])
        self.obs = state["obs"]
        self.next_obs = state["next_obs"]
        self.actions = state["actions"]
        self.rewards = state["rewards"]
        self.dones = state["dones"]
