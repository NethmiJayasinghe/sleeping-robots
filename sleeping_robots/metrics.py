"""Evaluation metrics.

All metrics operate on a **success table** ``rows``, a mapping
``{K: {skill_index: success_rate}}`` where ``rows[K][k]`` is the success rate of
skill ``k`` measured on the policy saved after learning ``K`` skills. Building
that table from separate checkpoints (rather than from one final checkpoint) is
what makes BWT and AF trustworthy - see :func:`backward_transfer`.

Metrics
-------
``AFSR``  average final success rate over the K skills (higher better)
``PRS``   pairwise reliability over the six pre-registered pairs (higher better)
``BWT``   backward transfer; negative means forgetting
``AF``    average forgetting from each skill's peak (lower better)
``MC``    memory cost in MB

PRS
---
PRS averages ``min(S_i, S_j)`` over fixed related-skill pairs, with both skills
evaluated separately under the same shared policy.
"""

import numpy as np

from .stream import PREREG_PAIRS


def average_final_success(row: dict, K: int) -> float:
    """AFSR: mean success over skills ``0..K-1`` on the final policy."""
    return float(np.mean([row[k] for k in range(K)]))


def pairwise_reliability(row: dict, K: int):
    """PRS over the pre-registered pairs whose both members are below ``K``.

    Returns ``None`` when no pair is scorable yet (i.e. ``K < 2``).
    """
    pairs = [(i, j) for (i, j, _) in PREREG_PAIRS if i < K and j < K]
    if not pairs:
        return None
    return float(sum(min(row[i], row[j]) for i, j in pairs) / len(pairs))


def pairwise_reliability_detail(row: dict, K: int):
    """Per-pair PRS contributions, for reporting *which* relations survive."""
    out = []
    for (i, j, name) in PREREG_PAIRS:
        if i < K and j < K:
            out.append({'pair': name, 'i': i, 'j': j,
                        's_i': row[i], 's_j': row[j],
                        'score': min(row[i], row[j])})
    return out


def backward_transfer(rows: dict, K: int):
    """BWT = mean over earlier skills of ``final - at_acquisition``.

    Requires the diagonal entries ``rows[j][j-1]`` (skill ``j-1`` measured right
    after it was learned). If those rows are missing, BWT is silently inflated,
    which is why this function returns ``None`` rather than guessing.
    """
    if K < 2 or not all(j in rows for j in range(1, K)):
        return None
    return float(sum(rows[K][j - 1] - rows[j][j - 1]
                     for j in range(1, K)) / (K - 1))


def average_forgetting(rows: dict, K: int):
    """AF = mean over earlier skills of ``peak_over_time - final``."""
    if K < 2 or not all(j in rows for j in range(1, K)):
        return None
    return float(sum(max(rows[t][j - 1] for t in range(j, K + 1) if t in rows)
                     - rows[K][j - 1] for j in range(1, K)) / (K - 1))


def memory_cost(critics, state_buffers, obs_dim: int = 39):
    """MC: stored critic parameters plus stored states.

    Returns ``(n_params, n_states, total_mb)`` assuming float32. This counts the
    method's retained memory only - the replay buffer used inside a wake phase
    is discarded and is not part of it.
    """
    total_params = sum(sum(p.numel() for p in c.parameters()) for c in critics)
    total_states = sum(len(s) for s in state_buffers)
    total_mb = (total_params * 4 + total_states * obs_dim * 4) / 1e6
    return total_params, total_states, total_mb


def mean_and_stderr(values):
    """``(mean, standard error)`` over a list, ignoring ``None``."""
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    if len(v) == 1:
        return float(v[0]), 0.0
    return float(np.mean(v)), float(np.std(v, ddof=1) / len(v) ** 0.5)
