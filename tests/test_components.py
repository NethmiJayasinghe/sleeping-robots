"""Unit tests for the method's components (no Meta-World required)."""

import numpy as np
import torch

from sleeping_robots.excitability import ExcitabilityLayer
from sleeping_robots.metrics import (average_final_success, average_forgetting,
                                     backward_transfer, pairwise_reliability,
                                     pairwise_reliability_detail)
from sleeping_robots.nash_solver import NashSolver
from sleeping_robots.normalizer import RunningNormalizer
from sleeping_robots.policy import SkillCritic, SleepingRobotsActor
from sleeping_robots.spi import spi_index_paired, spi_subspace


def test_nash_weights_are_a_simplex():
    torch.manual_seed(0)
    solver = NashSolver(fw_iters=50, normalize_gradients=True)
    grads = [torch.randn(128) for _ in range(4)]
    solver.solve(grads)
    alpha = solver.get_alpha()
    assert alpha.shape == (4,)
    assert torch.all(alpha >= 0)
    assert abs(alpha.sum().item() - 1.0) < 1e-6


def test_nash_output_is_a_convex_combination_of_unit_gradients():
    torch.manual_seed(0)
    solver = NashSolver(fw_iters=50, normalize_gradients=True)
    grads = [torch.randn(64) * s for s in (1.0, 1000.0, 0.001)]
    g = solver.solve(grads)
    unit = torch.stack([x / x.norm() for x in grads])
    expected = solver.get_alpha() @ unit
    assert torch.allclose(g, expected, atol=1e-6)


def test_nash_is_invariant_to_gradient_rescaling():
    """Scale invariance holds for the pre-bargaining normalisation, as claimed."""
    grads = [torch.randn(64) for _ in range(3)]
    torch.manual_seed(1)
    a = NashSolver(fw_iters=50, normalize_gradients=True)
    g1 = a.solve([x.clone() for x in grads])
    torch.manual_seed(1)
    b = NashSolver(fw_iters=50, normalize_gradients=True)
    g2 = b.solve([x.clone() * 97.0 for x in grads])
    assert torch.allclose(g1, g2, atol=1e-6)


def test_single_skill_nash_returns_the_unit_gradient():
    solver = NashSolver(fw_iters=50, normalize_gradients=True)
    g = torch.randn(32) * 5
    out = solver.solve([g])
    assert torch.allclose(out, g / (g.norm() + 1e-8), atol=1e-7)


def test_excitability_is_a_buffer_and_never_receives_a_gradient():
    """``e`` is homeostatic state, not a learnable parameter."""
    layer = ExcitabilityLayer(8)
    assert list(layer.parameters()) == []          # nothing here is learnable
    assert 'e' in dict(layer.named_buffers())
    assert not layer.e.requires_grad

    # Backprop through the layer must leave ``e`` untouched while still passing
    # gradient to the input.
    x = torch.randn(4, 8, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert layer.e.grad is None


def test_excitability_moves_gain_toward_target_and_clips():
    layer = ExcitabilityLayer(4, alpha_target=0.1, eta_e=1.0, beta=1.0,
                              e_min=0.5, e_max=1.5)
    layer.homeostatic_update(torch.full((2, 4), 10.0))   # far above target
    assert torch.all(layer.e >= 0.5) and torch.all(layer.e <= 1.5)
    assert torch.all(layer.e < 1.0)                       # pushed down


def test_excitability_reset_restores_initial_state():
    layer = ExcitabilityLayer(6, alpha_target=0.1)
    layer.homeostatic_update(torch.rand(8, 6))
    layer.reset_for_new_skill()
    assert torch.allclose(layer.e, torch.ones(6))
    assert torch.allclose(layer.ema_activation, torch.full((6,), 0.1))


def test_actor_action_is_bounded_and_shaped():
    actor = SleepingRobotsActor(9, 4, hidden_dim=32)
    action, log_prob, deterministic = actor.get_action(torch.randn(5, 9))
    assert action.shape == (5, 4) and log_prob.shape == (5, 1)
    assert torch.all(action.abs() <= 1.0) and torch.all(deterministic.abs() <= 1.0)


def test_frozen_critic_takes_no_gradient():
    critic = SkillCritic(9, 4, hidden_dim=32)
    critic.freeze()
    assert critic.frozen
    assert all(not p.requires_grad for p in critic.parameters())


def test_critic_forward_is_the_conservative_minimum():
    critic = SkillCritic(9, 4, hidden_dim=32)
    obs, act = torch.randn(3, 9), torch.randn(3, 4)
    q1, q2 = critic.both(obs, act)
    assert torch.allclose(critic(obs, act), torch.min(q1, q2))


def test_normalizer_matches_batch_statistics():
    rng = np.random.RandomState(0)
    data = rng.randn(500, 7).astype(np.float32) * 3 + 2
    n = RunningNormalizer(7, 'cpu')
    for chunk in np.array_split(data, 25):
        n.update(torch.from_numpy(chunk))
    assert np.allclose(n.mean.numpy(), data.mean(axis=0), atol=1e-4)
    assert np.allclose(n.var.numpy(), data.var(axis=0), atol=1e-3)


def test_normalizer_round_trips_through_state_dict():
    n = RunningNormalizer(5, 'cpu')
    n.update(torch.randn(50, 5))
    m = RunningNormalizer.from_state_dict(n.state_dict(), 5, 'cpu')
    assert torch.allclose(n.mean, m.mean) and torch.allclose(n.var, m.var)
    assert n.count == m.count


def test_prs_never_exceeds_afsr():
    """min <= mean, so this must hold for every table we ever print."""
    rng = np.random.RandomState(0)
    for _ in range(200):
        row = {k: float(rng.rand()) for k in range(5)}
        assert pairwise_reliability(row, 5) <= average_final_success(row, 5) + 1e-12


def test_prs_is_none_before_any_pair_is_scorable():
    assert pairwise_reliability({0: 1.0}, 1) is None


def test_one_dead_skill_can_zero_several_pairs():
    """A skill at index 2 sits in three of the six pairs."""
    row = {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0}
    detail = {d['pair']: d['score']
              for d in pairwise_reliability_detail(row, 5)}
    assert sum(1 for v in detail.values() if v == 0.0) == 3
    assert abs(pairwise_reliability(row, 5) - 0.5) < 1e-12


def test_backward_transfer_is_negative_under_forgetting():
    rows = {1: {0: 1.0}, 2: {0: 0.4, 1: 1.0}}
    assert backward_transfer(rows, 2) == -0.6
    assert average_forgetting(rows, 2) == 0.6


def test_backward_transfer_refuses_to_guess_without_the_diagonal():
    assert backward_transfer({2: {0: 0.4, 1: 1.0}}, 2) is None
    assert average_forgetting({2: {0: 0.4, 1: 1.0}}, 2) is None


def test_subspace_spi_is_invariant_to_component_reordering():
    """Subspace SPI ignores the ordering of components inside the subspace."""
    rng = np.random.RandomState(0)
    a = rng.randn(400, 10)
    basis = np.linalg.qr(rng.randn(10, 10))[0]
    b = a @ basis.T                        # same subspace, rotated coordinates
    assert spi_subspace(a, a) > 0.999
    # A permutation of the reference's own PCs leaves the subspace identical...
    pcs = np.linalg.svd(a - a.mean(0), full_matrices=False)[2][:3]
    swapped = np.vstack([pcs[1], pcs[0], pcs[2]])
    proj = (a - a.mean(0)) @ pcs.T @ swapped + a.mean(0)
    assert spi_subspace(a, proj) > 0.999
    # ...but the index-paired measure is damaged by exactly that swap.
    assert spi_index_paired(a, proj) < 0.9
    assert b.shape == a.shape
