"""End-to-end smoke test: a miniature two-skill stream.

Runs the real pipeline - wake, sleep, checkpointing, and post-hoc evaluation -
on drastically reduced budgets, to catch wiring errors that unit tests cannot.
It does **not** check result quality; the budgets are far too small to learn
anything.

Needs Meta-World and takes a few minutes, so it is opt-in::

    SR_RUN_SMOKE=1 pytest tests/test_smoke_e2e.py -s
"""

import os
import tempfile

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('SR_RUN_SMOKE') != '1',
                                reason='set SR_RUN_SMOKE=1 to run (slow)')


def test_two_skill_stream_trains_and_evaluates(monkeypatch):
    import sleeping_robots.stream as stream
    import sleeping_robots.phases.wake as wake_mod
    from sleeping_robots import default_cfg
    from sleeping_robots.evaluation import evaluate_run
    from sleeping_robots.trainer import train_stream

    skills = stream.MT5_STREAM[:2]

    # Miniature budgets, and a short warmup so some updates actually happen.
    monkeypatch.setitem(stream.SKILL_BUDGETS, skills[0], 12)
    monkeypatch.setitem(stream.SKILL_BUDGETS, skills[1], 12)
    monkeypatch.setattr(wake_mod, 'RANDOM_EXPLORATION_STEPS', 600)

    cfg = default_cfg(seed=0)
    cfg.update(sleep_max_iters=20, sleep_n_window=5,
               spi_baseline_n_episodes=2, eval_n_episodes=2,
               state_buffer_size=200)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = os.path.join(tmp, 'run')
        train_stream(cfg=cfg, ckpt_dir=ckpt_dir, skills=skills,
                     skip_eval=True, device='cpu')

        # Both skills produced both checkpoints.
        for k in (1, 2):
            for phase in ('post_wake', 'post_sleep'):
                assert os.path.exists(os.path.join(ckpt_dir, f'skill{k}_{phase}.pt')), \
                    f'missing skill{k}_{phase}.pt'
        assert os.path.exists(os.path.join(ckpt_dir, 'final.pt'))

        # Post-hoc evaluation runs and returns a well-formed metric table.
        out = evaluate_run(ckpt_dir, skills=skills, seed=70000, n_episodes=2,
                           device='cpu')
        assert set(out) == {1, 2}
        assert 0.0 <= out[2]['AFSR'] <= 1.0
        assert out[2]['PRS'] is not None          # pair (0,1) is scorable at K=2
        assert out[2]['BWT'] is not None          # diagonal present from ckpt 1
        assert out[2]['SPI'] is not None          # baseline stored at skill 1
        assert out[2]['MC_MB'] > 0
        assert len(out[2]['per_skill']) == 2
        print('\nsmoke metrics @K=2:', {k: v for k, v in out[2].items()
                                        if k != 'per_skill'})
