"""Hyperparameters for the MT5 experiments.

``DEFAULT_CFG`` holds the default values. It is deliberately a plain ``dict``: it is written verbatim into every checkpoint
under the ``cfg`` key, and the evaluation code reads architecture settings back
out of saved checkpoints, so the key names are part of the on-disk format.

Use :func:`default_cfg` to get a fresh copy rather than mutating the module
global.
"""

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── SAC / wake-phase constants ───────────────────────────────────────────────

REWARD_SCALE = 1.0
"""Raw Meta-World rewards. A larger scale makes Q small relative to the entropy
term ``alpha * log_pi``, which lets the entropy coefficient run away."""

TARGET_ENTROPY = -2.0
"""Less negative than the canonical ``-action_dim``. Meta-World's multi-goal
tasks have a near-deterministic optimum per state, so ``-action_dim`` makes the
auto-tuner chase unreachable entropy."""

TARGET_ENTROPY_PER_SKILL: dict = {}
"""Per-skill overrides. Empty for the MT5 stream - every skill uses
``TARGET_ENTROPY``. (The longer streams used ``push-v3: -4.0``, which needs more
exploration than the MT5 skills.)"""

RANDOM_EXPLORATION_STEPS = 5000
"""Uniform random actions for the first N env steps of each skill, so the replay
buffer holds diverse experience before policy updates begin."""

LOG_ALPHA_MIN, LOG_ALPHA_MAX = -10.0, 4.0
"""Clamp on ``log_alpha`` => ``alpha`` in ``[4.5e-5, 54.6]``; prevents runaway."""

# ── Everything else ──────────────────────────────────────────────────────────

DEFAULT_CFG: dict = dict(
    # Architecture
    hidden_dim=256,

    # Excitability
    alpha_target=0.10,
    eta_e_wake=1e-3,
    eta_e_sleep=1e-2,
    eta_e_transition=5e-3,   # elevated rate for the first `transition_eps`
    transition_eps=500,      # episodes of each new skill, then decays to wake
    exc_beta=0.01,
    e_min=0.1,
    e_max=5.0,

    # SAC
    lr_actor=3e-4,
    lr_critic=3e-4,
    gamma=0.99,
    tau=0.005,
    utd_ratio=5,
    replay_size=1_000_000,
    batch_size=256,

    # Sleep
    sleep_max_iters=4000,    # 5000+ destabilises at this lr
    sleep_lr=2e-3,
    sleep_optimizer='adam',  # decouples step size from gradient magnitude
    sleep_interp_alpha=0.0,  # interpolation init disabled (it broke door-open)
    sleep_delta=1e-3,        # loss-plateau convergence threshold
    sleep_n_window=100,      # window for that threshold
    sleep_eps=1e-5,          # ||g_nash|| convergence threshold
    sleep_anchor_lambda=1e-2,  # proximal anchor strength; 0 disables
    nash_fw_iters=50,

    # Memory / validation
    critic_val_threshold=0.1,
    state_buffer_size=1000,
    late_episode_fraction=0.8,   # keep states from the last 20% of episodes

    # Eval
    eval_n_episodes=20,
    spi_baseline_n_episodes=30,

    seed=42,
)


def default_cfg(**overrides) -> dict:
    """Fresh copy of :data:`DEFAULT_CFG`, with optional overrides applied."""
    cfg = dict(DEFAULT_CFG)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg
