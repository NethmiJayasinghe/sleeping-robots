"""Sleeping Robots - offline skill consolidation for a shared robot policy.

A single shared policy learns a stream of skills one at a time. After each new
skill is acquired (**wake**), the policy enters an offline **sleep** phase with
no new data, where the skills' frozen memories bargain over the shared weights
to reach a new K-skill equilibrium.

Typical use::

    from sleeping_robots import default_cfg, train_stream
    train_stream(cfg=default_cfg(seed=42), ckpt_dir='runs/seed42')

See ``docs/REPRODUCTION.md`` for the training and evaluation commands.
"""

__version__ = '1.0.0'

from .anchor import should_anchor
from .config import DEFAULT_CFG, DEVICE, default_cfg
from .evaluation import evaluate_checkpoint, evaluate_run, eval_skill
from .metrics import (average_final_success, average_forgetting,
                      backward_transfer, memory_cost, pairwise_reliability)
from .nash_solver import NashSolver
from .normalizer import RunningNormalizer
from .policy import SkillCritic, SleepingRobotsActor
from .replay import ReplayBuffer
from .phases.sleep import sleep
from .spi import spi_subspace
from .stream import MT5_STREAM, PREREG_PAIRS, SKILL_BUDGETS, make_env_with_tasks
from .trainer import train_stream
from .phases.wake import wake

__all__ = [
    'DEFAULT_CFG', 'DEVICE', 'default_cfg',
    'MT5_STREAM', 'PREREG_PAIRS', 'SKILL_BUDGETS', 'make_env_with_tasks',
    'SleepingRobotsActor', 'SkillCritic', 'NashSolver', 'RunningNormalizer',
    'ReplayBuffer', 'wake', 'sleep', 'should_anchor', 'train_stream',
    'eval_skill', 'evaluate_checkpoint', 'evaluate_run',
    'average_final_success', 'pairwise_reliability', 'backward_transfer',
    'average_forgetting', 'memory_cost', 'spi_subspace',
]
