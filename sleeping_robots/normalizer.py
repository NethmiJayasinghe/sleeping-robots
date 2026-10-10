"""Running observation normalizer (Welford's online algorithm).

One shared normalizer tracks every state the agent has ever seen, and is
applied at each forward pass as ``(obs - mean) / (std + eps)``.

A per-skill **snapshot** is taken at the end of every wake phase. This matters:
a frozen critic Q_k must be queried under the observation distribution it was
trained on. Evaluating a frozen reach actor under a later skill's normalizer
once dropped success from 93% to 3.3% with no weight change at all.
"""

import torch


class RunningNormalizer:
    def __init__(self, obs_dim: int, device: str, epsilon: float = 1e-8):
        self.obs_dim = obs_dim
        self.device = device
        self.epsilon = epsilon
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = 0

    @torch.no_grad()
    def update(self, obs: torch.Tensor):
        """Fold a batch of observations ``(N, obs_dim)`` into the statistics."""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        batch_count = obs.shape[0]

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta.pow(2) * self.count * batch_count / total
        ) / total
        self.count = total

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / (self.var.sqrt() + self.epsilon)

    def state_dict(self):
        return {'mean': self.mean.cpu(), 'var': self.var.cpu(), 'count': self.count}

    def load_state_dict(self, d: dict):
        self.mean = d['mean'].to(self.device)
        self.var = d['var'].to(self.device)
        self.count = d['count']

    @classmethod
    def from_state_dict(cls, d: dict, obs_dim: int, device: str):
        """Build a normalizer directly from a saved state dict."""
        n = cls(obs_dim, device)
        n.load_state_dict(d)
        return n
