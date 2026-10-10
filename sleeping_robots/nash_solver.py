"""Nash bargaining over per-skill gradients during sleep.

A sequential adaptation of Nash-MTL (Navon et al., 2022). Given per-skill
gradients ``g_1..g_K`` of the sleep objectives, find convex weights ``alpha``
and return the combined direction::

    g* = sum_k alpha_k * g_hat_k

where ``g_hat_k`` is ``g_k`` normalised to unit norm. ``alpha`` is found by
Frank-Wolfe ascent on ``F(alpha) = sum_k w_k log <g_k, g_alpha>``.
"""

import torch


class NashSolver:
    def __init__(self, n_skills=0, fw_iters=20, normalize_gradients=False):
        """
        Args:
            n_skills: number of skills to initialise uniform weights for.
            fw_iters: number of Frank-Wolfe iterations.
            normalize_gradients: normalise each per-skill gradient to unit norm
                before bargaining, so weights reflect direction agreement rather
                than gradient magnitude. The default config passes True.
        """
        self.fw_iters = fw_iters
        self.normalize_gradients = normalize_gradients
        self.alpha = torch.tensor([]) if n_skills == 0 else torch.ones(n_skills) / n_skills

    def add_skill(self, K_new):
        """Warm-start weights when skill ``K_new`` joins (see module note)."""
        if len(self.alpha) == 0:
            self.alpha = torch.ones(1)
            return

        new_weight = 1.0 / (K_new ** 2)
        scale = 1.0 - new_weight
        new_alpha = torch.cat([self.alpha * scale, torch.tensor([new_weight])])
        self.alpha = new_alpha / new_alpha.sum()

    def solve(self, gradients, skill_weights=None):
        """Bargain over ``gradients`` and return the combined direction.

        Args:
            gradients: list of K flat tensors of shape ``(D,)``.
            skill_weights: optional K positive weights ``w_k`` giving skill k
                more bargaining power (asymmetric Nash). None means uniform.

        Returns:
            ``g_nash`` of shape ``(D,)``.
        """
        K = len(gradients)
        assert K > 0

        if K == 1:
            if self.normalize_gradients:
                g = gradients[0]
                return g / (g.norm() + 1e-8)
            return gradients[0]

        G = torch.stack(gradients)                      # (K, D)
        if self.normalize_gradients:
            G = G / (G.norm(dim=1, keepdim=True) + 1e-8)

        gram = G @ G.T                                  # (K, K)

        if skill_weights is None:
            w = torch.ones(K, device=G.device)
        else:
            w = torch.as_tensor(skill_weights, dtype=G.dtype, device=G.device)
            assert w.shape == (K,), f"skill_weights shape {w.shape} != (K={K},)"
            assert (w > 0).all(), "skill_weights must be positive"

        alpha = self.alpha.clone().to(G.device)
        if len(alpha) != K:
            alpha = torch.ones(K, device=G.device) / K

        # Frank-Wolfe ascent on F(alpha) = sum_k w_k log <g_k, g_alpha>.
        #   dF/dalpha_i = sum_k w_k * gram[i,k] / <g_k, g_alpha>
        # Step toward the vertex e_{k*} maximising the linear approximation.
        for t in range(self.fw_iters):
            g_alpha = alpha @ G                                      # (D,)
            scores = (G @ g_alpha).clamp(min=1e-8)                   # (K,) floored
            grad_F = (gram * (w / scores).unsqueeze(0)).sum(dim=1)   # (K,)

            k_star = torch.argmax(grad_F)
            e_k = torch.zeros(K, device=G.device)
            e_k[k_star] = 1.0

            gamma = 2.0 / (t + 2)                                    # 1.0 at t=0
            alpha = (1 - gamma) * alpha + gamma * e_k

        alpha = torch.clamp(alpha, min=0)
        alpha = alpha / alpha.sum()

        # Retain as the warm state for the next call.
        self.alpha = alpha.detach().cpu()

        return alpha @ G

    def get_alpha(self):
        return self.alpha.clone()
