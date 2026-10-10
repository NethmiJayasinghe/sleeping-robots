#!/usr/bin/env python
"""Evaluate trained checkpoints and emit the metric table.

Default protocol: three evaluation seeds, N=100 episodes per skill, PRS over
the six pre-registered pairs, rotation-invariant subspace SPI::

    python scripts/evaluate.py --run-dirs runs/seed42 --out results/seed42

Writes ``<out>.json`` (full per-seed detail) and ``<out>.md`` (a table).
"""

import argparse
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleeping_robots.evaluation import (EVAL_SEEDS, N_EPISODES,  # noqa: E402
                                        evaluate_run)
from sleeping_robots.metrics import mean_and_stderr                   # noqa: E402

METRICS = ('AFSR', 'PRS', 'BWT', 'AF', 'SPI', 'MC_MB')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dirs', required=True,
                   help='comma-separated checkpoint directories to evaluate')
    p.add_argument('--seeds', default=','.join(map(str, EVAL_SEEDS)),
                   help='comma-separated evaluation seeds (default: %(default)s)')
    p.add_argument('--n', type=int, default=N_EPISODES,
                   help='episodes per skill (default: %(default)s)')
    p.add_argument('--out', default='results/evaluation',
                   help='output path prefix (default: %(default)s)')
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(',')]
    run_dirs = [d.strip() for d in a.run_dirs.split(',') if d.strip()]

    results = {}
    for d in run_dirs:
        if not os.path.isdir(d):
            print(f'[skip] no such directory: {d}', flush=True)
            continue
        print(f'\n{"#"*72}\n# {d}\n{"#"*72}', flush=True)
        per_seed = {}
        for s in seeds:
            print(f'\n--- evaluation seed {s} ---', flush=True)
            per_seed[s] = evaluate_run(d, seed=s, n_episodes=a.n)

        Ks = sorted(per_seed[seeds[0]].keys())
        K_final = max(Ks)
        summary = {}
        for m in METRICS:
            mu, se = mean_and_stderr([per_seed[s][K_final][m] for s in seeds])
            summary[m] = {'mean': mu, 'se': se}
        results[d] = {
            'final_K': K_final,
            'per_seed': {str(s): {str(k): v for k, v in per_seed[s].items()}
                         for s in seeds},
            'summary': summary,
        }
        print('\n>>> ' + d + f' @K={K_final}: ' + '  '.join(
            f'{m}={summary[m]["mean"]:.3f}+-{summary[m]["se"]:.3f}'
            if summary[m]['mean'] is not None else f'{m}=n/a'
            for m in METRICS), flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(f'{a.out}.json', 'w') as fh:
        json.dump({'eval_seeds': seeds, 'n_episodes': a.n,
                   'prs_rule': '6 pre-registered pairs',
                   'spi_variant': 'subspace (rotation-invariant)',
                   'results': results}, fh, indent=2)

    md = [f'# Evaluation - {len(seeds)} evaluation seeds, N={a.n}', '',
          f'Evaluation seeds: {seeds}. PRS over the 6 pre-registered pairs. '
          'SPI is the rotation-invariant subspace variant. Uncertainty is the '
          'standard error across evaluation seeds.', '']
    cols = [d for d in run_dirs if d in results]
    if cols:
        md.append('| Metric | ' + ' | '.join(cols) + ' |')
        md.append('|---' * (len(cols) + 1) + '|')
        for m in METRICS:
            cells = []
            for d in cols:
                s = results[d]['summary'][m]
                cells.append('n/a' if s['mean'] is None
                             else f'{s["mean"]:.3f} ± {s["se"]:.3f}')
            md.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    with open(f'{a.out}.md', 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    print(f'\nsaved -> {a.out}.json\n         {a.out}.md', flush=True)


if __name__ == '__main__':
    main()
