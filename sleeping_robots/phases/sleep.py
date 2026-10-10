"""Sleep phase - offline consolidation with no new data.

Sleep is where continual learning actually happens. It sees **no environment**,
no new experience, and no stored transitions: only the K frozen critics and the
K raw state buffers. For each skill it forms a gradient of
``L_k = -mean(Q_k(s, pi(s)))`` over that skill's states, bargains the K
gradients into a single direction, and takes one step.

The resulting equilibrium is a **new** K-skill compromise, not a return to the
previous one. Sleep is not a replay buffer and not a distillation pass; the
negotiation is the mechanism.

Two normalizers are in play on purpose. ``Q_k`` is queried through skill k's
**own** wake-end normalizer, because that is the observation distribution it was
trained on; the actor is queried through the **current shared** normalizer,
because that is what inference will use.
"""

from collections import defaultdict, deque

import torch


def sleep(actor, actor_optimizer, critics: list, state_buffers_raw: list,
          normalizer, nash_solver, normalizers_per_skill: list,
          wake_end_snapshots: list, cfg: dict, device: str,
          theta_anchor_flat=None):
    """Consolidate over the K frozen skill memories, in place on ``actor``.

    Args:
        critics: frozen per-skill critics ``Q_1..Q_K``.
        state_buffers_raw: per-skill raw state buffers (no actions, no rewards).
        normalizer: current shared observation normalizer (actor input).
        normalizers_per_skill: wake-end normalizer snapshots (critic input).
        wake_end_snapshots: wake-end actor states, used only by interpolation
            init (disabled by default).
        theta_anchor_flat: if given and ``sleep_anchor_lambda > 0``, pull the
            update toward this parameter vector. Pass ``None`` to disable at
            this K; see :mod:`sleeping_robots.anchor` for the gating rule.
    """
    sleep_iters = int(cfg['sleep_max_iters'])
    sleep_window = int(cfg['sleep_n_window'])
    sleep_lr = float(cfg['sleep_lr'])

    K = len(critics)
    print(f"\n=== SLEEP: negotiating over {K} skills ===", flush=True)

    actor.set_excitability_rate(float(cfg['eta_e_sleep']))

    # ── Optional interpolation init (disabled in the default config) ───────
    interp_alpha = float(cfg['sleep_interp_alpha'])
    if interp_alpha > 0.0 and wake_end_snapshots:
        with torch.no_grad():
            mean_snap = {}
            ref_state = actor.state_dict()
            for key in ref_state:
                # Skip excitability buffers - they have their own dynamics.
                if '.exc.' in key:
                    continue
                stacked = torch.stack([snap[key] for snap in wake_end_snapshots])
                mean_snap[key] = stacked.mean(dim=0)
            for key, val_mean in mean_snap.items():
                ref_state[key] = (interp_alpha * val_mean
                                  + (1 - interp_alpha) * ref_state[key])
            actor.load_state_dict(ref_state)
        print(f"  sleep init: alpha={interp_alpha:.2f} interpolation toward mean of "
              f"{len(wake_end_snapshots)} wake-end snapshot(s)", flush=True)

    # ── Optimizer state ─────────────────────────────────────────────────────
    # Adam is hand-rolled rather than torch.optim.Adam because the step is taken
    # on a bargained direction that is not the gradient of any scalar loss, so
    # there is no loss to call .backward() on at this point.
    use_adam = cfg.get('sleep_optimizer', 'sgd') == 'adam'
    if use_adam:
        adam_m = [torch.zeros_like(p) for p in actor.parameters()]
        adam_v = [torch.zeros_like(p) for p in actor.parameters()]
        adam_t = 0
        b1, b2, eps_a = 0.9, 0.999, 1e-8
    print(f"  sleep optimizer: {'adam' if use_adam else 'sgd'}, lr={sleep_lr}, "
          f"max_iters={sleep_iters}", flush=True)

    sleep_anchor_lambda = float(cfg.get('sleep_anchor_lambda', 0.0))
    use_sleep_anchor = sleep_anchor_lambda > 0.0 and theta_anchor_flat is not None
    if use_sleep_anchor:
        print(f"  [anchor] proximal anchor active: lambda={sleep_anchor_lambda:.1e}, "
              f"anchor flat-dim={theta_anchor_flat.numel()}", flush=True)

    loss_history: dict = defaultdict(lambda: deque(maxlen=sleep_window))

    for it in range(sleep_iters):
        gradients = []
        skill_losses = []

        for k, (critic, S_k_raw) in enumerate(zip(critics, state_buffers_raw)):
            n = min(64, len(S_k_raw))
            idx = torch.randint(0, len(S_k_raw), (n,))
            raw_batch = S_k_raw[idx]

            # Q_k under its own wake-end normalizer; actor under the shared one.
            if (normalizers_per_skill is not None
                    and k < len(normalizers_per_skill)
                    and normalizers_per_skill[k] is not None):
                obs_for_critic = normalizers_per_skill[k].normalize(raw_batch)
            else:
                obs_for_critic = normalizer.normalize(raw_batch)
            obs_for_actor = normalizer.normalize(raw_batch)

            actor_optimizer.zero_grad()
            actions, _, _ = actor.get_action(obs_for_actor)
            L_k = -critic(obs_for_critic, actions).mean()
            L_k.backward(retain_graph=(k < K - 1))

            g_k = torch.cat([p.grad.flatten() for p in actor.parameters()
                             if p.grad is not None])
            gradients.append(g_k.detach())
            skill_losses.append(L_k.item())
            actor_optimizer.zero_grad()

        g_nash = nash_solver.solve(gradients)

        # Proximal anchor: L2 pull toward theta_anchor, added to the bargained
        # direction before the optimizer step. Applied uniformly to the shared
        # parameter vector, not per skill.
        if use_sleep_anchor:
            with torch.no_grad():
                theta_flat = torch.cat([p.data.flatten() for p in actor.parameters()])
                anchor_grad = sleep_anchor_lambda * (theta_flat - theta_anchor_flat)
            g_step = g_nash + anchor_grad
        else:
            g_step = g_nash

        with torch.no_grad():
            offset = 0
            if use_adam:
                adam_t += 1
                bc1 = 1 - b1 ** adam_t
                bc2 = 1 - b2 ** adam_t
                for i, p in enumerate(actor.parameters()):
                    numel = p.numel()
                    g_p = g_step[offset:offset + numel].view(p.shape)
                    adam_m[i].mul_(b1).add_(g_p, alpha=(1 - b1))
                    adam_v[i].mul_(b2).addcmul_(g_p, g_p, value=(1 - b2))
                    m_hat = adam_m[i] / bc1
                    v_hat = adam_v[i] / bc2
                    p.data -= sleep_lr * m_hat / (v_hat.sqrt() + eps_a)
                    offset += numel
            else:
                for p in actor.parameters():
                    numel = p.numel()
                    p.data -= sleep_lr * g_step[offset:offset + numel].view(p.shape)
                    offset += numel

        # Homeostatic update over the union of all skills' states, normalised by
        # the shared normalizer (matching inference).
        all_states_raw = torch.cat(state_buffers_raw, dim=0)
        n_all = min(256, len(all_states_raw))
        idx_all = torch.randint(0, len(all_states_raw), (n_all,))
        with torch.no_grad():
            actor.homeostatic_update(normalizer.normalize(all_states_raw[idx_all]))

        for k, L in enumerate(skill_losses):
            loss_history[k].append(L)

        if it >= sleep_window:
            max_val_change = max(
                abs(list(loss_history[k])[0] - list(loss_history[k])[-1])
                for k in range(K)
            )
            if max_val_change < float(cfg['sleep_delta']):
                print(f"  Converged at iter {it} (val_change={max_val_change:.2e})",
                      flush=True)
                break
            if g_nash.norm().item() < float(cfg['sleep_eps']):
                print(f"  Converged at iter {it} (grad_norm={g_nash.norm():.2e})",
                      flush=True)
                break

        if (it + 1) % 100 == 0:
            losses_str = " | ".join(f"L_{k}={L:.3f}" for k, L in enumerate(skill_losses))
            print(f"  iter {it+1} | {losses_str} | ||g_nash||={g_nash.norm():.4f}",
                  flush=True)

    actor.set_excitability_rate(float(cfg['eta_e_wake']))
