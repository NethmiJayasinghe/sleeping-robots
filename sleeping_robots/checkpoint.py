"""Self-contained per-skill checkpoints.

Two checkpoints are written per skill: ``skill{k}_post_wake.pt`` (straight after
acquisition, before consolidation) and ``skill{k}_post_sleep.pt`` (after
consolidation). Each one carries everything needed to re-evaluate later without
retraining, which is what makes the post-hoc evaluation pipeline possible.
"""

import os

import torch


def save_checkpoint(actor, critics, state_buffers, nash_solver, cfg: dict, path: str,
                    success_matrix=None, spi_baseline_obs=None, eval_log=None,
                    normalizer=None, extra=None):
    """Write a checkpoint containing the full continual-learning context.

    Args:
        actor: the shared actor.
        critics: frozen per-skill critics, in stream order.
        state_buffers: per-skill raw (unnormalised) state buffers.
        nash_solver: solver whose bargaining weights are recorded.
        cfg: the config dict, stored verbatim so evaluation can rebuild the
            architecture exactly.
        path: destination; parent directories are created.
        extra: additional keys merged into the saved dict (the trainer uses this
            for ``normalizers_per_skill`` and ``wake_end_snapshots``).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'actor': actor.state_dict(),
        'critics': [c.state_dict() for c in critics],
        'state_buffers': [s.cpu() for s in state_buffers],
        'cfg': cfg,
    }
    if nash_solver is not None:
        ckpt['nash_alpha'] = nash_solver.get_alpha()
    if success_matrix is not None:
        # Flatten nested defaultdicts so the file is always pickle-safe.
        ckpt['success_matrix'] = {t: dict(tasks) for t, tasks in success_matrix.items()}
    if spi_baseline_obs is not None:
        ckpt['spi_baseline_obs'] = spi_baseline_obs
    if eval_log is not None:
        ckpt['eval_log'] = list(eval_log)
    if normalizer is not None:
        ckpt['normalizer'] = normalizer.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    print(f"  [ckpt] Saved {path}", flush=True)


def load_checkpoint(path: str, device: str):
    """Load a checkpoint dict written by :func:`save_checkpoint`."""
    return torch.load(path, map_location=device, weights_only=False)


def build_actor(obs_dim: int, action_dim: int, cfg: dict, device: str):
    """Construct an actor with the excitability settings recorded in ``cfg``."""
    from .policy import SleepingRobotsActor
    return SleepingRobotsActor(
        obs_dim, action_dim,
        hidden_dim=int(cfg.get('hidden_dim', 256)),
        alpha_target=float(cfg.get('alpha_target', 0.10)),
        eta_e=float(cfg.get('eta_e_wake', 1e-3)),
        beta=float(cfg.get('exc_beta', 0.01)),
        e_min=float(cfg.get('e_min', 0.1)),
        e_max=float(cfg.get('e_max', 5.0)),
    ).to(device)


def build_critics(state_dicts, obs_dim: int, action_dim: int, cfg: dict, device: str):
    """Rebuild and freeze the per-skill critics from saved state dicts."""
    from .policy import SkillCritic
    critics = []
    for sd in state_dicts:
        c = SkillCritic(obs_dim, action_dim,
                        hidden_dim=int(cfg.get('hidden_dim', 256))).to(device)
        c.load_state_dict(sd)
        c.freeze()
        critics.append(c)
    return critics
