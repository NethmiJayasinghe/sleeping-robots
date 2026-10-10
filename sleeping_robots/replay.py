"""Replay buffer for the wake phase.

Wake is ordinary off-policy SAC, so it keeps a standard transition buffer.
Note that this buffer is **discarded at the end of each wake phase** - it is
not the per-skill memory. Sleep never sees transitions, only unordered states
(see :mod:`sleeping_robots.sleep`).
"""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device: str):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(obs).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_obs).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)
