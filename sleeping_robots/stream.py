"""The MT5 skill stream: task list, budgets, stopping rules, and PRS pairs.

Skills arrive one at a time, in this fixed order, and are never revisited::

    reach -> door-open -> drawer-open -> window-close -> button-press-topdown

Episode cost varies a lot by skill (door ~44 ep/h, drawer ~103 ep/h on a single
GPU), so the budgets below are not interchangeable.
"""

import random

import metaworld

MT5_STREAM = [
    'reach-v3',
    'door-open-v3',
    'drawer-open-v3',
    'window-close-v3',
    'button-press-topdown-v3',
]

SKILL_BUDGETS = {
    'reach-v3':                 1500,
    'door-open-v3':             4000,
    'drawer-open-v3':           3000,
    'window-close-v3':          2000,
    'button-press-topdown-v3':  2500,
}
"""Episode budget per skill; 13,000 episodes for the full K=5 stream."""

# Fallbacks for a skill with no entry in EARLY_STOP_PER_SKILL.
EARLY_STOP_MIN_EPS = 500
EARLY_STOP_PATIENCE = 500
EARLY_STOP_IMPROVEMENT = 0.02
EARLY_STOP_WINDOW = 100

EARLY_STOP_PER_SKILL = {
    'reach-v3':                 {'min_eps': 500,  'patience': 500,  'improvement': 0.02, 'window': 100},
    'door-open-v3':             {'min_eps': 1000, 'patience': 1000, 'improvement': 0.02, 'window': 100},
    'drawer-open-v3':           {'min_eps': 1000, 'patience': 1000, 'improvement': 0.02, 'window': 100},
    'window-close-v3':          {'min_eps': 500,  'patience': 500,  'improvement': 0.02, 'window': 100},
    'button-press-topdown-v3':  {'min_eps': 800,  'patience': 800,  'improvement': 0.02, 'window': 100},
}
"""Per-skill early stopping, tuned to each skill's reward dynamics."""

PREREG_PAIRS = [
    (0, 1, "reach->door-open"),
    (1, 2, "door-open->drawer-open"),
    (0, 2, "reach->drawer-open"),
    (1, 3, "door-open->window-close"),
    (2, 4, "drawer-open->button-press-topdown"),
    (3, 4, "window-close->button-press-topdown"),
]
"""The six related-skill pairs used by PRS, fixed *a priori* from shared
interaction structure (gripper approach, handle/drawer contact) before the K=4
wake was run. PRS scores ``min(S_i, S_j)`` over exactly these pairs.

Note what PRS does and does not measure: the two skills in a pair are evaluated
**separately** under the same shared policy. PRS is pairwise co-retention, not
chained or joint execution.
"""


def early_stop_rule(skill_name: str) -> dict:
    """Early-stopping thresholds for ``skill_name``, falling back to defaults."""
    return EARLY_STOP_PER_SKILL.get(skill_name, {
        'min_eps': EARLY_STOP_MIN_EPS,
        'patience': EARLY_STOP_PATIENCE,
        'improvement': EARLY_STOP_IMPROVEMENT,
        'window': EARLY_STOP_WINDOW,
    })


def make_env_with_tasks(task_name: str, seed: int = 0):
    """Build a Meta-World env plus its full task (goal) pool.

    Returns ``(env, tasks)``. The caller **must** call
    ``env.set_task(random.choice(tasks))`` before each ``env.reset()``; without
    it the policy only ever sees one goal and the task is far easier than the
    benchmark intends.
    """
    mt10 = metaworld.MT10()
    env_cls = mt10.train_classes[task_name]
    env = env_cls()
    tasks = [t for t in mt10.train_tasks if t.env_name == task_name]
    env.set_task(tasks[0])
    env.seed(seed)
    return env, tasks


def seed_everything(seed: int):
    """Seed torch, numpy and python ``random``."""
    import numpy as np
    import torch
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
