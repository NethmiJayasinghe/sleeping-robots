"""Stream orchestration: wake -> sleep -> checkpoint, for each skill in turn.

For every skill ``k`` the trainer:

1. runs **wake** on skill ``k`` (shared weights, no protection for old skills),
2. snapshots the normalizer and the wake-end parameters,
3. saves ``skill{k}_post_wake.pt``,
4. runs **sleep** over all ``k`` frozen memories (skipped at ``k = 1``, where a
   single critic makes negotiation vacuous),
5. saves ``skill{k}_post_sleep.pt``.

Checkpoints are written before *and* after the optional inline evaluation, so a
crash inside evaluation cannot lose the consolidation work.

The recommended workflow is to run this with ``skip_eval=True`` and then
evaluate the saved checkpoints with :mod:`sleeping_robots.evaluation`. The
inline evaluation here is a convenience for watching a run: it uses fewer
episodes.
"""

import os
from collections import defaultdict

import torch

from .anchor import should_anchor
from .checkpoint import build_actor, load_checkpoint, save_checkpoint
from .config import DEVICE, default_cfg
from .evaluation import eval_skill
from .metrics import average_final_success, pairwise_reliability
from .nash_solver import NashSolver
from .normalizer import RunningNormalizer
from .policy import SkillCritic
from .phases.sleep import sleep as sleep_phase
from .stream import (MT5_STREAM, SKILL_BUDGETS, make_env_with_tasks,
                     seed_everything)
from .phases.wake import wake as wake_phase


def _flat_params(actor):
    return torch.cat([p.data.flatten().clone() for p in actor.parameters()]).detach()


def train_stream(cfg: dict = None, ckpt_dir: str = 'runs/sr_mt5',
                 skills: list = None, resume: str = None,
                 anchor_mode: str = 'detector', detector_threshold: float = 150.0,
                 manual_ks=None, anchor_ckpt: str = None,
                 skip_eval: bool = True, device: str = DEVICE):
    """Train the shared policy across the whole skill stream.

    Args:
        cfg: config dict; defaults to :func:`sleeping_robots.config.default_cfg`.
        ckpt_dir: directory for per-skill checkpoints.
        skills: skill stream; defaults to :data:`MT5_STREAM`.
        resume: path to a ``skill{k}_post_wake.pt`` or ``skill{k}_post_sleep.pt``
            checkpoint to continue from. RNG state is not restored.
        anchor_mode: ``detector`` | ``always`` | ``never`` | ``manual``.
        detector_threshold: drift threshold for ``detector`` mode.
        manual_ks: set of K values to anchor at, for ``manual`` mode.
        anchor_ckpt: override the anchor with this checkpoint's parameters. Use
            when resuming from a *post_wake* checkpoint, where the live
            parameters are wake-end (already damaged) rather than the previous
            multi-skill equilibrium.
        skip_eval: skip inline evaluation (recommended; evaluate post-hoc).
    """
    cfg = dict(cfg) if cfg is not None else default_cfg()
    skills = skills or MT5_STREAM
    K_max = len(skills)
    hidden = int(cfg['hidden_dim'])

    print(f"  ckpt_dir={ckpt_dir}", flush=True)
    print(f"  seed={cfg['seed']}", flush=True)
    print(f"  skip_eval={skip_eval}", flush=True)
    if anchor_mode == 'detector':
        print(f"  [anchor] detector mode: drift threshold {detector_threshold}; "
              f"K_max rule applied (no anchor at the final sleep)", flush=True)
    elif anchor_mode == 'always':
        print("  [anchor] anchor at every K>=2 sleep", flush=True)
    elif anchor_mode == 'never':
        print("  [anchor] anchor disabled at every sleep", flush=True)
    elif anchor_mode == 'manual':
        print(f"  [anchor] manual anchor at K in {sorted(manual_ks or [])}", flush=True)

    seed_everything(int(cfg['seed']))

    env0, _ = make_env_with_tasks(skills[0], int(cfg['seed']))
    obs_dim = env0.observation_space.shape[0]
    action_dim = env0.action_space.shape[0]
    env0.close()
    print(f"Sleeping Robots | obs_dim={obs_dim}, action_dim={action_dim}, "
          f"device={device}", flush=True)
    print(f"  budgets={ {s: SKILL_BUDGETS[s] for s in skills} }", flush=True)
    print(f"  sleep: lr={cfg['sleep_lr']}, max_iters={cfg['sleep_max_iters']}, "
          f"delta={cfg['sleep_delta']}, window={cfg['sleep_n_window']}, "
          f"fw_iters={cfg['nash_fw_iters']}, "
          f"anchor_lambda={cfg.get('sleep_anchor_lambda', 0.0)}", flush=True)

    actor = build_actor(obs_dim, action_dim, cfg, device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(cfg['lr_actor']))
    # normalize_gradients=True: bargaining must be scale-invariant, since
    # per-skill Q magnitudes differ by orders of magnitude across skills.
    nash_solver = NashSolver(fw_iters=int(cfg['nash_fw_iters']),
                             normalize_gradients=True)
    normalizer = RunningNormalizer(obs_dim, device)

    def actor_factory():
        return build_actor(obs_dim, action_dim, cfg, device)

    critics: list = []
    state_buffers_raw: list = []
    normalizers_per_skill: list = []
    wake_end_snapshots: list = []
    all_rewards: dict = {}
    success_matrix: dict = defaultdict(lambda: defaultdict(float))
    eval_log: list = []
    spi_baseline_obs = None
    theta_anchor_flat = None
    start_idx = 0

    # ── Resume ──────────────────────────────────────────────────────────────
    if resume:
        ckpt = load_checkpoint(resume, device)
        actor.load_state_dict(ckpt['actor'])
        for sd in ckpt['critics']:
            c = SkillCritic(obs_dim, action_dim, hidden).to(device)
            c.load_state_dict(sd)
            c.freeze()
            critics.append(c)
        state_buffers_raw = [s.to(device) for s in ckpt['state_buffers']]
        if 'normalizer' in ckpt:
            normalizer.load_state_dict(ckpt['normalizer'])
        for n_sd in ckpt.get('normalizers_per_skill', []):
            n_k = RunningNormalizer(obs_dim, device)
            n_k.load_state_dict(n_sd)
            normalizers_per_skill.append(n_k)
        if not normalizers_per_skill:
            print(f"  [resume] no per-skill normalizers stored; falling back to the "
                  f"shared normalizer for all {len(critics)} skills", flush=True)
            for _ in critics:
                fallback = RunningNormalizer(obs_dim, device)
                fallback.load_state_dict(normalizer.state_dict())
                normalizers_per_skill.append(fallback)
        wake_end_snapshots = [{k: v.to(device) for k, v in sd.items()}
                              for sd in ckpt.get('wake_end_snapshots', [])]
        for t, tasks in ckpt.get('success_matrix', {}).items():
            for k, v in tasks.items():
                success_matrix[int(t)][int(k)] = v
        eval_log = list(ckpt.get('eval_log', []))
        spi_baseline_obs = ckpt.get('spi_baseline_obs')
        for k in range(len(critics)):
            nash_solver.add_skill(k + 1)
        start_idx = len(critics)

        if spi_baseline_obs is None and start_idx >= 1:
            print(f"  [resume] no stored SPI baseline; recomputing on {skills[0]}",
                  flush=True)
            _, spi_baseline_obs = eval_skill(
                actor, normalizer, skills[0],
                int(cfg['spi_baseline_n_episodes']), 777, device)
        print(f"  [resume] loaded {resume}; resuming from skill "
              f"{start_idx + 1}/{K_max}", flush=True)

        if start_idx >= 1:
            theta_anchor_flat = _flat_params(actor)
            print(f"  [anchor] resume captured the anchor "
                  f"(flat-dim={theta_anchor_flat.numel()})", flush=True)
            if anchor_ckpt is not None:
                a_ck = load_checkpoint(anchor_ckpt, device)
                tmp = actor_factory()
                tmp.load_state_dict(a_ck['actor'])
                theta_anchor_flat = _flat_params(tmp)
                del tmp
                print(f"  [anchor] anchor OVERRIDDEN from {anchor_ckpt}", flush=True)

        # Resuming from a post_wake checkpoint means skill `start_idx` was
        # acquired but never consolidated, so its sleep is still pending and is
        # run here, before the next wake.
        resumed_from_post_wake = 'post_wake' in os.path.basename(resume)
        if resumed_from_post_wake and start_idx >= 2:
            print(f"  [resume] post_wake checkpoint for skill {start_idx}: running "
                  f"its pending sleep now", flush=True)
            interp_snaps = wake_end_snapshots[:-1] if wake_end_snapshots else []
            use_anchor = should_anchor(start_idx, K_max, _flat_params(actor),
                                       interp_snaps, actor_factory,
                                       mode=anchor_mode,
                                       threshold=detector_threshold,
                                       manual_ks=manual_ks)
            sleep_phase(actor, actor_optimizer, critics, state_buffers_raw,
                        normalizer, nash_solver, normalizers_per_skill,
                        interp_snaps, cfg, device,
                        theta_anchor_flat=(theta_anchor_flat if use_anchor else None))
            theta_anchor_flat = _flat_params(actor)
            _save(actor, critics, state_buffers_raw, nash_solver, cfg, ckpt_dir,
                  f'skill{start_idx}_post_sleep', normalizers_per_skill,
                  wake_end_snapshots, spi_baseline_obs, normalizer,
                  success_matrix, eval_log)
        elif start_idx >= 1:
            print(f"  [resume] post_sleep checkpoint: skill {start_idx} is already "
                  f"consolidated; not re-running its sleep", flush=True)

    # ── Stream ──────────────────────────────────────────────────────────────
    for skill_idx in range(start_idx, K_max):
        skill_name = skills[skill_idx]
        critic, state_buffer_raw, ep_rewards = wake_phase(
            skill_name, actor, actor_optimizer, normalizer, skill_idx,
            int(cfg['seed']) + skill_idx, cfg, device)
        critics.append(critic)
        state_buffers_raw.append(state_buffer_raw)
        nash_solver.add_skill(skill_idx + 1)
        all_rewards[f'wake_{skill_name}'] = ep_rewards

        # Snapshot the normalizer so this skill's critic keeps a valid input
        # distribution during every later sleep, and the parameters for the
        # drift detector.
        norm_snap = RunningNormalizer(obs_dim, device)
        norm_snap.load_state_dict(normalizer.state_dict())
        normalizers_per_skill.append(norm_snap)
        wake_end_snapshots.append({k: v.detach().clone()
                                   for k, v in actor.state_dict().items()})

        if skill_idx == 0:
            _, spi_baseline_obs = eval_skill(
                actor, normalizer, skills[0],
                int(cfg['spi_baseline_n_episodes']), 777, device)

        _save(actor, critics, state_buffers_raw, nash_solver, cfg, ckpt_dir,
              f'skill{skill_idx+1}_post_wake', normalizers_per_skill,
              wake_end_snapshots, spi_baseline_obs, normalizer)

        if len(critics) >= 2:
            # Interpolation init should look at *previous* wake ends, not the
            # one just taken (which equals the current parameters).
            interp_snaps = wake_end_snapshots[:-1]
            use_anchor = should_anchor(len(critics), K_max, _flat_params(actor),
                                       interp_snaps, actor_factory,
                                       mode=anchor_mode,
                                       threshold=detector_threshold,
                                       manual_ks=manual_ks)
            sleep_phase(actor, actor_optimizer, critics, state_buffers_raw,
                        normalizer, nash_solver, normalizers_per_skill,
                        interp_snaps, cfg, device,
                        theta_anchor_flat=(theta_anchor_flat if use_anchor else None))
        else:
            print(f"  [skip sleep] K={len(critics)} - nothing to negotiate with; "
                  f"keeping the wake-trained policy", flush=True)

        # The post-sleep parameters are the equilibrium the next sleep anchors to.
        theta_anchor_flat = _flat_params(actor)

        label = f'skill{skill_idx+1}_post_sleep'
        _save(actor, critics, state_buffers_raw, nash_solver, cfg, ckpt_dir, label,
              normalizers_per_skill, wake_end_snapshots, spi_baseline_obs, normalizer)

        if not skip_eval:
            metrics = _inline_eval(actor, normalizer, skills, skill_idx + 1,
                                   success_matrix, cfg, device)
            eval_log.append(metrics)
        else:
            print(f"  [skip_eval] no inline eval after skill {skill_idx+1}; "
                  f"checkpoint saved for post-hoc evaluation", flush=True)

        _save(actor, critics, state_buffers_raw, nash_solver, cfg, ckpt_dir, label,
              normalizers_per_skill, wake_end_snapshots, spi_baseline_obs, normalizer,
              success_matrix, eval_log)

        print(f"\nCompleted skill {skill_idx+1}/{K_max}: {skill_name}", flush=True)

    final_path = os.path.join(ckpt_dir, 'final.pt')
    torch.save({
        'actor': actor.state_dict(),
        'critics': [c.state_dict() for c in critics],
        'state_buffers_raw': [s.cpu() for s in state_buffers_raw],
        'nash_alpha': nash_solver.get_alpha(),
        'all_rewards': all_rewards,
        'eval_log': eval_log,
        'success_matrix': {t: dict(v) for t, v in success_matrix.items()},
        'normalizer': normalizer.state_dict(),
        'normalizers_per_skill': [n.state_dict() for n in normalizers_per_skill],
        'wake_end_snapshots': [{k: v.cpu() for k, v in sd.items()}
                               for sd in wake_end_snapshots],
        'cfg': cfg,
    }, final_path)
    print(f"\nDone. Final summary -> {final_path}", flush=True)
    for i, m in enumerate(eval_log):
        print(f"  After skill {i+1}: AFSR={m['afsr']:.3f} PRS={m['prs']:.3f}")
    return ckpt_dir


def _save(actor, critics, state_buffers_raw, nash_solver, cfg, ckpt_dir, label,
          normalizers_per_skill, wake_end_snapshots, spi_baseline_obs, normalizer,
          success_matrix=None, eval_log=None):
    extra = {
        'normalizers_per_skill': [n.state_dict() for n in normalizers_per_skill],
        'wake_end_snapshots': [{k: v.cpu() for k, v in sd.items()}
                               for sd in wake_end_snapshots],
    }
    save_checkpoint(actor, critics, state_buffers_raw, nash_solver, cfg,
                    os.path.join(ckpt_dir, f'{label}.pt'),
                    success_matrix=success_matrix,
                    spi_baseline_obs=spi_baseline_obs,
                    normalizer=normalizer,
                    eval_log=eval_log,
                    extra=extra)


def _inline_eval(actor, normalizer, skills, K, success_matrix, cfg, device):
    """Cheap progress check. Not the reporting path - see module docstring."""
    print(f"\n{'='*60}\nINLINE EVAL after {K} skill(s)\n{'='*60}", flush=True)
    n_ep = int(cfg['eval_n_episodes'])
    row = {}
    for k in range(K):
        sr, _ = eval_skill(actor, normalizer, skills[k], n_ep, 900 + (K - 1) * 10 + k,
                           device)
        row[k] = sr
        success_matrix[K][k] = sr
        print(f"  Skill {k+1} [{skills[k]}]: {sr:.3f}", flush=True)
    afsr = average_final_success(row, K)
    prs = pairwise_reliability(row, K) or 0.0
    print(f"\n  AFSR={afsr:.3f} | PRS={prs:.3f}", flush=True)
    return dict(afsr=afsr, prs=prs, n_skills=K)
