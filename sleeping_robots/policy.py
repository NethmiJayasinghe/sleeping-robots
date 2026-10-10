"""Shared actor and per-skill critic.

One actor is shared by every skill in the stream; there are no adapters, no
LoRA, no mixture-of-experts gating, and no per-skill parameters. Continual
learning therefore has to happen *in* the shared weights.

Architecture::

    obs -> [Linear -> Excitability] -> [Linear -> Excitability] -> mean, log_std

Each skill contributes one frozen twin critic ``Q_k``, which becomes that
skill's entire memory during sleep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .excitability import ExcitabilityLayer

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class ExcitableMLPBlock(nn.Module):
    """One hidden layer followed by its per-neuron excitability gate."""

    def __init__(self, in_dim, out_dim, **exc_kwargs):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.exc = ExcitabilityLayer(out_dim, **exc_kwargs)

    def forward(self, x):
        return self.exc(self.linear(x))

    def homeostatic_update(self, x):
        with torch.no_grad():
            activations = torch.relu(self.linear(x))
            self.exc.homeostatic_update(activations)

    def reset_for_new_skill(self):
        self.exc.reset_for_new_skill()


class SleepingRobotsActor(nn.Module):
    """Squashed-Gaussian SAC actor with per-neuron excitability."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256, **exc_kwargs):
        super().__init__()
        self.block1 = ExcitableMLPBlock(obs_dim, hidden_dim, **exc_kwargs)
        self.block2 = ExcitableMLPBlock(hidden_dim, hidden_dim, **exc_kwargs)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        # Bias log_std toward -1 => initial std ~0.37. Default Linear init can
        # give std up to ~7, i.e. an effectively uniform policy at step 0.
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def forward(self, obs):
        x = self.block2(self.block1(obs))
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, obs):
        """Returns ``(sampled_action, log_prob, deterministic_action)``."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def homeostatic_update(self, obs_batch):
        self.block1.homeostatic_update(obs_batch)
        x = self.block1(obs_batch).detach()
        self.block2.homeostatic_update(x)

    def set_excitability_rate(self, eta_e):
        self.block1.exc.set_wake_rate(eta_e)
        self.block2.exc.set_wake_rate(eta_e)

    def reset_for_new_skill(self):
        """Re-engage homeostatic re-tuning at a skill boundary.

        Touches the excitability mechanism only, never the policy weights.
        """
        self.block1.reset_for_new_skill()
        self.block2.reset_for_new_skill()

    def flat_params(self):
        """Flat copy of the trainable parameter vector (used by sleep/anchor)."""
        return torch.cat([p.data.flatten().clone() for p in self.parameters()]).detach()


class SkillCritic(nn.Module):
    """Twin Q-network for one skill, frozen at the end of that skill's wake.

    ``forward`` returns ``min(Q1, Q2)`` (conservative, used by actor updates and
    by sleep); ``both`` returns the pair for the twin-critic training loss.

    Once frozen this critic *is* skill ``k``'s memory: sleep reads a gradient
    signal out of it and never touches the skill's data again.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        def _make_net():
            return nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = _make_net()
        self.q2 = _make_net()
        self.frozen = False

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return torch.min(self.q1(x), self.q2(x))

    def both(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.frozen = True

    def td_error(self, obs, action, target):
        with torch.no_grad():
            return F.mse_loss(self.forward(obs, action), target).item()
