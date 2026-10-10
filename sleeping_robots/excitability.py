"""Per-neuron excitability - the homeostatic plasticity mechanism.

Each hidden unit carries a scalar gain ``e_i`` applied multiplicatively after
the activation function::

    y_i = e_i * relu(z_i)

``e_i`` is **not** a learnable parameter and receives no gradient. It is moved
only by a homeostatic rule that drives each unit's long-run mean activation
toward ``alpha_target``::

    e_i <- clip(e_i + eta_e * (alpha_target - ema_activation_i), e_min, e_max)

``eta_e`` differs between wake and sleep (see :mod:`sleeping_robots.config`).
"""

import torch
import torch.nn as nn


class ExcitabilityLayer(nn.Module):
    """Multiplicative per-neuron gain with a homeostatic update rule.

    Args:
        n_neurons: width of the layer this gate is attached to.
        alpha_target: target long-run mean activation per unit.
        eta_e: homeostatic step size. Set per phase via :meth:`set_wake_rate`
            and :meth:`set_sleep_rate`.
        beta: EMA coefficient for the activation estimate.
        e_min, e_max: clipping bounds on the gain.
    """

    def __init__(self, n_neurons, alpha_target=0.10, eta_e=1e-3, beta=0.01,
                 e_min=0.1, e_max=5.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.alpha_target = alpha_target
        self.eta_e = eta_e
        self.beta = beta
        self.e_min = e_min
        self.e_max = e_max

        # Registered as buffers, not parameters: these are moved by the
        # homeostatic rule only and must never receive a gradient.
        self.register_buffer('e', torch.ones(n_neurons))
        self.register_buffer('ema_activation', torch.full((n_neurons,), alpha_target))

    def forward(self, pre_activation):
        activated = torch.relu(pre_activation)
        return self.e * activated

    @torch.no_grad()
    def homeostatic_update(self, activations):
        """Nudge ``e`` so mean activation drifts toward ``alpha_target``.

        Args:
            activations: ``(batch, n_neurons)`` post-ReLU activations.
        """
        mean_act = activations.mean(dim=0)
        self.ema_activation = (1 - self.beta) * self.ema_activation + self.beta * mean_act
        self.e = torch.clamp(
            self.e + self.eta_e * (self.alpha_target - self.ema_activation),
            self.e_min, self.e_max
        )

    def set_sleep_rate(self, eta_e_sleep):
        self.eta_e = eta_e_sleep

    def set_wake_rate(self, eta_e_wake):
        self.eta_e = eta_e_wake

    @torch.no_grad()
    def reset_for_new_skill(self):
        """Clear the activity estimate and gain at a skill boundary.

        Re-engages per-neuron re-tuning when the activation distribution shifts
        abruptly on the arrival of a new skill. Policy weights are untouched -
        this resets the excitability mechanism only.
        """
        self.ema_activation.fill_(self.alpha_target)
        self.e.fill_(1.0)
