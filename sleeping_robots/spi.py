"""Synergy Preservation Index (SPI).

SPI compares the top-``m`` principal directions of the observation trajectories
produced on a reference skill before and after consolidation.

Two estimators are provided. :func:`spi_subspace` is the rotation- and
order-invariant subspace overlap and is the default::

    SPI = ||U^T V||_F^2 / m

:func:`spi_index_paired` compares the principal directions component by
component.
"""

import itertools

import numpy as np

M_DEFAULT = 3


def top_pcs(obs, m: int = M_DEFAULT):
    """Top-``m`` right singular vectors of mean-centred ``obs`` (orthonormal)."""
    oc = obs - obs.mean(axis=0)
    _, _, Vt = np.linalg.svd(oc, full_matrices=False)
    return Vt[:m]


def spi_subspace(obs_before, obs_after, m: int = M_DEFAULT) -> float:
    """Rotation- and order-invariant subspace overlap (default)."""
    A, B = top_pcs(obs_before, m), top_pcs(obs_after, m)
    return float(np.linalg.norm(A @ B.T, 'fro') ** 2 / m)


def spi_index_paired(obs_before, obs_after, m: int = M_DEFAULT) -> float:
    """Index-paired variant: mean |cos| between matched principal directions."""
    A, B = top_pcs(obs_before, m), top_pcs(obs_after, m)
    return float(np.mean([
        abs(A[j] @ B[j] / (np.linalg.norm(A[j]) * np.linalg.norm(B[j]) + 1e-8))
        for j in range(m)]))


def spi_best_permutation(obs_before, obs_after, m: int = M_DEFAULT) -> float:
    """Index-paired under the best component matching.

    Diagnostic for separating a reordering of components from a change of
    subspace.
    """
    A, B = top_pcs(obs_before, m), top_pcs(obs_after, m)
    C = np.abs(A @ B.T)
    return float(max(np.mean([C[j, p[j]] for j in range(m)])
                     for p in itertools.permutations(range(m))))
