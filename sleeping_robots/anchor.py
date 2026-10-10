"""Proximal anchor and the drift detector that gates it.

During sleep the update can optionally be pulled toward ``theta_anchor`` - the
previous post-sleep parameter vector, i.e. the last point at which several
skills were jointly alive::

    g_step = g_nash + lambda * (theta - theta_anchor)

Whether the anchor fires at a given K is decided by one of four modes:

``never``
    Never anchor.
``always``
    Anchor at every sleep with K >= 2.
``manual``
    Anchor at an explicit set of K values.
``detector`` (default)
    Anchor iff the minimum parameter drift between the current policy and any
    older skill's wake-end snapshot falls below ``threshold``, *and* K < K_max.
"""

import torch


def min_drift_to_older_skills(theta_cur, wake_end_snapshots, K_now,
                              actor_factory):
    """Smallest L2 distance from ``theta_cur`` to any older wake-end snapshot.

    Args:
        theta_cur: flat current parameter vector.
        wake_end_snapshots: actor state dicts captured at each wake end.
        K_now: 1-indexed number of skills learned so far.
        actor_factory: zero-arg callable returning a fresh actor, used to read
            the snapshot's parameters in canonical order (buffers excluded).

    Returns:
        The minimum drift, or ``inf`` if no snapshot could be read.
    """
    min_drift = float('inf')
    for k in range(K_now - 1):
        if k >= len(wake_end_snapshots):
            continue
        try:
            tmp = actor_factory()
            tmp.load_state_dict(wake_end_snapshots[k])
            theta_k_flat = torch.cat(
                [p.data.flatten().clone() for p in tmp.parameters()]
            ).detach()
            drift = (theta_cur - theta_k_flat).norm().item()
            del tmp
            min_drift = min(min_drift, drift)
        except Exception as exc:
            print(f"  [detector] warning: could not compute drift for k={k}: {exc}",
                  flush=True)
    return min_drift


def should_anchor(K_now, K_max, theta_cur, wake_end_snapshots, actor_factory,
                  mode='detector', threshold=150.0, manual_ks=None,
                  verbose=True):
    """Decide whether the proximal anchor fires at the ``K_now``-th sleep.

    ``K_now`` is 1-indexed (it equals ``len(critics)`` at the sleep call).
    """
    if mode == 'never':
        return False
    if mode == 'always':
        return K_now >= 2            # the K=1 sleep is skipped anyway
    if mode == 'manual':
        return manual_ks is not None and K_now in manual_ks

    # detector
    if K_now < 2 or K_now >= K_max:  # K_max rule: never anchor the last sleep
        return False

    min_drift = min_drift_to_older_skills(theta_cur, wake_end_snapshots,
                                          K_now, actor_factory)
    trigger = min_drift < threshold
    if verbose:
        print(f"  [detector] K={K_now}: min_drift over older critics = "
              f"{min_drift:.2f} (threshold {threshold}); "
              f"{'TRIGGER -> apply Option D' if trigger else 'no trigger -> unanchored sleep'}",
              flush=True)
    return trigger
