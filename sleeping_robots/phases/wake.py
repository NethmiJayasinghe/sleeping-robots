"""Wake phase - acquiring one new skill.

Wake is ordinary SAC on the new skill only, updating the **shared** weights with
no protection for earlier skills. That is deliberate: on an RL substrate wake is
*expected* to damage old skills, and recovering them is sleep's job. Nothing
here is a continual-learning mechanism.

Wake produces exactly two things that outlive it:

1. a **frozen twin critic** ``Q_k``, validated against a TD-error gate, and
2. a **raw state buffer** sampled from the last 20% of episodes.

Those two objects are the skill's entire memory. No transitions, actions,
rewards, done flags or task losses are retained.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

from ..config import (LOG_ALPHA_MAX, LOG_ALPHA_MIN, RANDOM_EXPLORATION_STEPS,
                     REWARD_SCALE, TARGET_ENTROPY, TARGET_ENTROPY_PER_SKILL)
from ..policy import SkillCritic
from ..replay import ReplayBuffer
from ..stream import SKILL_BUDGETS, early_stop_rule, make_env_with_tasks


def validate_critic(critic, actor, obs_b, act_b, rew_b, next_obs_b, done_b,
                    r_max: float, alpha: float, gamma: float,
                    threshold_fraction: float = 0.1):
    """TD-error gate on a held-out batch. Returns ``(passes, td_err)``."""
    with torch.no_grad():
        next_actions, next_log_pi, _ = actor.get_action(next_obs_b)
        next_q = critic(next_obs_b, next_actions) - alpha * next_log_pi
        target_q = rew_b + (1 - done_b) * gamma * next_q
        td_err = F.mse_loss(critic(obs_b, act_b), target_q).item()
    return td_err < threshold_fraction * r_max, td_err


def wake(skill_name: str, actor, actor_optimizer, normalizer, skill_idx: int,
         seed: int, cfg: dict, device: str):
    """Train the shared actor on ``skill_name`` and freeze its critic.

    Returns:
        ``(critic, state_buffer_raw, episode_rewards)``.
    """
    hidden = int(cfg['hidden_dim'])
    batch = int(cfg['batch_size'])
    utd = int(cfg['utd_ratio'])
    state_buf_size = int(cfg['state_buffer_size'])
    gamma = float(cfg['gamma'])
    tau = float(cfg['tau'])

    n_episodes = SKILL_BUDGETS[skill_name]
    print(f"\n=== WAKE: skill {skill_idx+1} - {skill_name} ({n_episodes} eps) ===",
          flush=True)

    env, tasks = make_env_with_tasks(skill_name, seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"  task_pool_size={len(tasks)} (randomized per reset)", flush=True)

    critic = SkillCritic(obs_dim, action_dim, hidden).to(device)
    critic_target = SkillCritic(obs_dim, action_dim, hidden).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg['lr_critic']))

    target_entropy = TARGET_ENTROPY_PER_SKILL.get(skill_name, TARGET_ENTROPY)
    print(f"  target_entropy={target_entropy:.1f}", flush=True)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

    replay = ReplayBuffer(int(cfg['replay_size']))

    # Novelty-driven plasticity boost: for a new skill (not the first), run the
    # excitability rate above baseline for `transition_eps` episodes, then decay.
    if skill_idx > 0:
        actor.set_excitability_rate(float(cfg['eta_e_transition']))
        print(f"  novelty-boost: eta_e={cfg['eta_e_transition']:.1e} for first "
              f"{cfg['transition_eps']} eps", flush=True)
    else:
        actor.set_excitability_rate(float(cfg['eta_e_wake']))

    all_episode_obs_buffers: list = []
    episode_rewards: list = []
    late_threshold = float(cfg['late_episode_fraction'])
    total_env_steps = 0

    best_mean_reward = -float('inf')
    eps_since_improvement = 0
    es = early_stop_rule(skill_name)
    es_min_eps, es_patience = es['min_eps'], es['patience']
    es_improvement, es_window = es['improvement'], es['window']
    print(f"  early_stop[{skill_name}]: min_eps={es_min_eps}, patience={es_patience}, "
          f"improvement={es_improvement}, window={es_window}", flush=True)

    for ep in range(n_episodes):
        env.set_task(random.choice(tasks))
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        obs_buffer: list = []

        while not done:
            obs_buffer.append(obs.copy())
            if total_env_steps < RANDOM_EXPLORATION_STEPS:
                action_np = env.action_space.sample().astype(np.float32)
            else:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                obs_norm = normalizer.normalize(obs_t)
                with torch.no_grad():
                    action, _, _ = actor.get_action(obs_norm)
                action_np = action.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            # float(terminated), not float(done): do not bootstrap-kill on a
            # time-limit truncation.
            replay.push(obs, action_np, float(reward) / REWARD_SCALE,
                        next_obs, float(terminated))
            obs = next_obs
            ep_reward += float(reward)
            total_env_steps += 1

            if len(replay) >= batch and total_env_steps >= RANDOM_EXPLORATION_STEPS:
                for _ in range(utd):
                    obs_b, act_b, rew_b, next_obs_b, done_b = replay.sample(batch, device)
                    obs_b = normalizer.normalize(obs_b)
                    next_obs_b = normalizer.normalize(next_obs_b)
                    alpha = log_alpha.exp().item()

                    with torch.no_grad():
                        next_act, next_log_pi, _ = actor.get_action(next_obs_b)
                        next_q = critic_target(next_obs_b, next_act) - alpha * next_log_pi
                        target_q = rew_b + (1 - done_b) * gamma * next_q
                    q1, q2 = critic.both(obs_b, act_b)
                    q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                    critic_optimizer.zero_grad()
                    q_loss.backward()
                    critic_optimizer.step()

                    act_new, log_pi, _ = actor.get_action(obs_b)
                    q_val = critic(obs_b, act_new)
                    actor_loss = (alpha * log_pi - q_val).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    log_alpha.data.clamp_(min=LOG_ALPHA_MIN, max=LOG_ALPHA_MAX)

                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

                with torch.no_grad():
                    obs_b_h, *_ = replay.sample(batch, device)
                    actor.homeostatic_update(normalizer.normalize(obs_b_h))

            normalizer.update(torch.FloatTensor(np.array(obs_buffer[-1:])).to(device))

        episode_rewards.append(ep_reward)
        all_episode_obs_buffers.append(obs_buffer)

        if (ep + 1) % 50 == 0:
            mean_r = float(np.mean(episode_rewards[-50:]))
            print(f"  ep {ep+1}/{n_episodes} | reward={mean_r:.1f} | "
                  f"alpha={log_alpha.exp().item():.4f}", flush=True)

        if skill_idx > 0 and (ep + 1) == int(cfg['transition_eps']):
            actor.set_excitability_rate(float(cfg['eta_e_wake']))
            print(f"  novelty-boost decayed at ep {ep+1}: eta_e back to "
                  f"{cfg['eta_e_wake']:.1e}", flush=True)

        if ep >= es_window - 1:
            current_mean = float(np.mean(episode_rewards[-es_window:]))
            if ep >= es_min_eps:
                if current_mean > best_mean_reward * (1 + es_improvement):
                    best_mean_reward = current_mean
                    eps_since_improvement = 0
                else:
                    eps_since_improvement += 1
                    if eps_since_improvement >= es_patience:
                        print(f"  [Early stop] ep {ep+1}: no >{es_improvement*100:.1f}% "
                              f"improvement in {es_patience} eps; "
                              f"final {es_window}-ep mean={current_mean:.1f}", flush=True)
                        break
            else:
                if current_mean > best_mean_reward:
                    best_mean_reward = current_mean

    # State buffer: the last (1 - late_threshold) of *completed* episodes, so the
    # rule holds whether wake exited by early stop or by exhausting its budget.
    n_completed = len(all_episode_obs_buffers)
    n_late_start = int(n_completed * late_threshold)
    late_episode_states: list = []
    for ep_obs in all_episode_obs_buffers[n_late_start:]:
        late_episode_states.extend(ep_obs)
    print(f"  [state buffer] {n_completed} eps completed; taking last "
          f"{n_completed - n_late_start} eps ({len(late_episode_states)} obs)",
          flush=True)

    # Critic quality gate: the critic is about to become this skill's only
    # memory, so extend training until its TD error is small relative to r_max.
    r_max = max(abs(r) for r in episode_rewards) / REWARD_SCALE + 1e-6
    threshold = float(cfg['critic_val_threshold']) * r_max
    td_err = float('inf')
    passes = False
    for round_idx in range(10):
        obs_h, act_h, rew_h, next_h, done_h = replay.sample(min(batch, len(replay)), device)
        obs_h = normalizer.normalize(obs_h)
        next_h = normalizer.normalize(next_h)
        passes, td_err = validate_critic(critic, actor, obs_h, act_h, rew_h,
                                         next_h, done_h, r_max,
                                         log_alpha.exp().item(), gamma,
                                         float(cfg['critic_val_threshold']))
        if passes:
            break
        print(f"  [gate] TD err {td_err:.4f} > thresh {threshold:.4f}; extending "
              f"200 twin-critic updates (round {round_idx+1}/10)", flush=True)
        for _ in range(200):
            obs_b, act_b, rew_b, next_obs_b, done_b = replay.sample(batch, device)
            obs_b = normalizer.normalize(obs_b)
            next_obs_b = normalizer.normalize(next_obs_b)
            with torch.no_grad():
                next_act, next_log_pi, _ = actor.get_action(next_obs_b)
                next_q = (critic_target(next_obs_b, next_act)
                          - log_alpha.exp().item() * next_log_pi)
                target_q = rew_b + (1 - done_b) * gamma * next_q
            q1, q2 = critic.both(obs_b, act_b)
            q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            critic_optimizer.zero_grad()
            q_loss.backward()
            critic_optimizer.step()
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

    critic.freeze()
    print(f"  Critic frozen | TD error={td_err:.4f} | r_max={r_max:.2f} | "
          f"thresh={threshold:.4f} | {'PASS' if passes else 'WARN:failed'}", flush=True)

    # Store RAW states; normalisation happens at retrieval, because which
    # normalizer applies depends on whether the consumer is Q_k or the actor.
    n_sample = min(state_buf_size, len(late_episode_states))
    idx = np.random.choice(len(late_episode_states), n_sample, replace=False)
    state_buffer_raw = torch.FloatTensor(
        np.array([late_episode_states[i] for i in idx])
    ).to(device)

    env.close()
    return critic, state_buffer_raw, episode_rewards
