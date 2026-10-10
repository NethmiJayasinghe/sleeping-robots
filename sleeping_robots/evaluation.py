"""Post-hoc evaluation of saved checkpoints.

Training is run with ``--skip-eval`` and evaluated afterwards from checkpoints.
That is not just a speed optimisation: metrics computed from the stored
per-skill checkpoints (``skill1_post_sleep.pt`` ... ``skillK_post_sleep.pt``) use
the same policy snapshots and the same episode count for every skill and every
K, so BWT and AF have the diagonal entries they need.

Default protocol:

* **three evaluation seeds** (70000 / 80000 / 90000),
* **N = 100** episodes per skill,
* PRS over the six pre-registered pairs.
"""

import os
import random

import numpy as np
import torch

from .checkpoint import build_actor, build_critics, load_checkpoint
from .config import DEVICE
from .metrics import (average_final_success, average_forgetting,
                      backward_transfer, memory_cost, pairwise_reliability,
                      pairwise_reliability_detail)
from .normalizer import RunningNormalizer
from .spi import spi_subspace
from .stream import MT5_STREAM, make_env_with_tasks

EVAL_SEEDS = (70000, 80000, 90000)
N_EPISODES = 100


def eval_skill(actor, normalizer, skill_name: str, n_episodes: int, seed: int,
               device: str = DEVICE):
    """Roll out ``n_episodes`` on one skill. Returns ``(success_rate, obs)``.

    ``obs`` is every observation visited, used as the SPI trajectory sample.
    """
    random.seed(seed)
    env, tasks = make_env_with_tasks(skill_name, seed)
    successes, all_obs = [], []
    for _ in range(n_episodes):
        env.set_task(random.choice(tasks))
        obs, _ = env.reset()
        done, success = False, False
        while not done:
            all_obs.append(obs.copy())
            obs_t = normalizer.normalize(
                torch.FloatTensor(obs).unsqueeze(0).to(device))
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_t)
            obs, _, term, trunc, info = env.step(action.cpu().numpy()[0])
            done = term or trunc
            if info.get('success', 0) > 0.5:
                success = True
        successes.append(float(success))
    env.close()
    return float(np.mean(successes)), np.array(all_obs)


def _probe_dims(skill_name: str):
    env, _ = make_env_with_tasks(skill_name, 0)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()
    return obs_dim, action_dim


def evaluate_checkpoint(ckpt_path: str, skills: list, n_episodes: int, seed: int,
                        device: str = DEVICE, spi_episodes: int = 30):
    """Evaluate one checkpoint on the skills it has learned.

    Returns a dict with per-skill success rates, PRS detail, MC, and subspace
    SPI (when the checkpoint carries a stored SPI baseline).
    """
    print(f"\n{'='*70}\nEvaluating: {ckpt_path}\n{'='*70}", flush=True)
    ckpt = load_checkpoint(ckpt_path, device)
    cfg = ckpt.get('cfg', {})

    obs_dim, action_dim = _probe_dims(skills[0])

    if not any('.exc.' in k for k in ckpt['actor']):
        raise ValueError(
            f"{ckpt_path} does not look like a Sleeping Robots actor "
            "(no excitability buffers). This repository ships the method only.")

    actor = build_actor(obs_dim, action_dim, cfg, device)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    normalizer = RunningNormalizer(obs_dim, device)
    if 'normalizer' in ckpt:
        normalizer.load_state_dict(ckpt['normalizer'])

    critics = build_critics(ckpt.get('critics', []), obs_dim, action_dim, cfg, device)
    state_buffers = [s.to(device) for s in ckpt.get('state_buffers', [])]

    K = len(critics)
    if K == 0:
        raise ValueError(f"{ckpt_path} contains no critics; cannot infer K.")
    eval_skills = skills[:K]
    print(f"  K = {K} (stream has {len(skills)} skills)", flush=True)

    print(f"\n  Per-skill eval (N={n_episodes} episodes each):", flush=True)
    row = {}
    for k, skill_name in enumerate(eval_skills):
        sr, _ = eval_skill(actor, normalizer, skill_name, n_episodes,
                           seed=seed + k, device=device)
        row[k] = sr
        print(f"    Skill {k+1} [{skill_name}]: {sr:.3f}", flush=True)

    _, _, mc_mb = memory_cost(critics, state_buffers, obs_dim=obs_dim)

    spi = None
    baseline = ckpt.get('spi_baseline_obs')
    if K >= 2 and baseline is not None:
        _, obs_now = eval_skill(actor, normalizer, eval_skills[0], spi_episodes,
                                seed=seed + 1000, device=device)
        spi = spi_subspace(np.asarray(baseline), np.asarray(obs_now))

    return {
        'ckpt_path': ckpt_path,
        'K': K,
        'n_episodes': n_episodes,
        'eval_seed': seed,
        'success_per_skill': {eval_skills[k]: row[k] for k in range(K)},
        'row': row,
        'prs_detail': pairwise_reliability_detail(row, K),
        'mc_mb': mc_mb,
        'spi_subspace': spi,
    }


def evaluate_run(ckpt_dir: str, skills: list = None, seed: int = 70000,
                 n_episodes: int = N_EPISODES, device: str = DEVICE):
    """Evaluate every ``skill{k}_post_sleep.pt`` in a run under one eval seed.

    Returns ``{K: metrics}``. Computing every K from its own checkpoint is what
    gives BWT and AF their required diagonal.
    """
    skills = skills or MT5_STREAM
    present = [k for k in range(1, len(skills) + 1)
               if os.path.exists(os.path.join(ckpt_dir, f'skill{k}_post_sleep.pt'))]
    if not present:
        raise FileNotFoundError(f"no skill*_post_sleep.pt checkpoints in {ckpt_dir}")

    rows, spis, mcs = {}, {}, {}
    for k in present:
        r = evaluate_checkpoint(os.path.join(ckpt_dir, f'skill{k}_post_sleep.pt'),
                                skills[:k], n_episodes, seed, device)
        rows[k] = r['row']
        spis[k] = r['spi_subspace']
        mcs[k] = r['mc_mb']

    out = {}
    for k in present:
        out[k] = dict(
            AFSR=average_final_success(rows[k], k),
            PRS=pairwise_reliability(rows[k], k),
            BWT=backward_transfer(rows, k),
            AF=average_forgetting(rows, k),
            SPI=spis[k],
            MC_MB=mcs[k],
            per_skill=[rows[k][j] for j in range(k)],
        )
    return out
