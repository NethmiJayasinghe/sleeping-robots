#!/usr/bin/env python
"""Train Sleeping Robots on the Meta-World MT5 skill stream.

Full K=5 run from scratch (13,000 episodes, ~2-3 days on one modern GPU)::

    python scripts/train_mt5.py --seed 42 --ckpt-dir runs/seed42

Evaluate afterwards with ``scripts/evaluate.py``; see docs/REPRODUCTION.md.
"""

import argparse
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleeping_robots import default_cfg, train_stream                 # noqa: E402
from sleeping_robots.config import DEFAULT_CFG                        # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, default=int(DEFAULT_CFG['seed']),
                   help='training seed, applied to torch, numpy, random and env '
                        'task sampling (default: %(default)s)')
    p.add_argument('--ckpt-dir', default='runs/sr_mt5',
                   help='directory for per-skill checkpoints (default: %(default)s)')
    p.add_argument('--resume', default=None,
                   help='checkpoint to continue from (RNG state is not restored)')
    p.add_argument('--anchor-mode', default='detector',
                   choices=['detector', 'always', 'never', 'manual'],
                   help='when the sleep-time proximal anchor fires '
                        '(default: %(default)s)')
    p.add_argument('--detector-threshold', type=float, default=150.0,
                   help='drift threshold for detector mode (default: %(default)s)')
    p.add_argument('--anchor-Ks', default=None,
                   help='comma-separated K values to anchor at; implies '
                        '--anchor-mode manual')
    p.add_argument('--anchor-ckpt', default=None,
                   help='override the anchor with this checkpoint\'s parameters; '
                        'use when resuming from a post_wake checkpoint')
    p.add_argument('--sleep-lr', type=float, default=None,
                   help='override the sleep learning rate (default: config value '
                        f"{DEFAULT_CFG['sleep_lr']})")
    p.add_argument('--eval-inline', action='store_true',
                   help='run a cheap evaluation after each skill (off by default)')
    a = p.parse_args()

    cfg = default_cfg(seed=a.seed)
    if a.sleep_lr is not None:
        cfg['sleep_lr'] = a.sleep_lr

    manual_ks = None
    anchor_mode = a.anchor_mode
    if a.anchor_Ks:
        manual_ks = {int(x) for x in a.anchor_Ks.split(',') if x.strip()}
        anchor_mode = 'manual'

    train_stream(cfg=cfg, ckpt_dir=a.ckpt_dir, resume=a.resume,
                 anchor_mode=anchor_mode, detector_threshold=a.detector_threshold,
                 manual_ks=manual_ks, anchor_ckpt=a.anchor_ckpt,
                 skip_eval=not a.eval_inline)


if __name__ == '__main__':
    main()
