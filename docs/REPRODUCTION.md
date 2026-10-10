# Reproduction

## Protocol

* **Training:** one continuous run from scratch per seed.
* **Evaluation:** post-hoc from the saved per-skill checkpoints.
* **Evaluation seeds:** 70000, 80000, 90000.
* **Episodes:** N=100 per skill.
* **PRS:** the six pre-registered pairs in `stream.PREREG_PAIRS`.

## Commands

Full method, seed 42:

```bash
python scripts/train_mt5.py --seed 42 --ckpt-dir runs/seed42
python scripts/evaluate.py  --run-dirs runs/seed42 --out results/seed42
```

Several training seeds, evaluated into one table:

```bash
for S in 42 100 123; do
  python scripts/train_mt5.py --seed $S --ckpt-dir runs/seed$S
done
python scripts/evaluate.py --run-dirs runs/seed42,runs/seed100,runs/seed123 \
                           --out results/training_seeds
```

Anchor variants (the anchor is a flag on the same trainer, not a separate
script):

```bash
python scripts/train_mt5.py --seed 42 --anchor-mode never  --ckpt-dir runs/no_anchor
python scripts/train_mt5.py --seed 42 --anchor-mode always --ckpt-dir runs/anchor_always
python scripts/train_mt5.py --seed 42 --anchor-Ks 4        --ckpt-dir runs/anchor_k4
```

## Configuration

`sleeping_robots/config.py` holds the default values:

| | |
|---|---|
| stream | reach 1500 → door 4000 → drawer 3000 → window 2000 → button 2500 eps |
| SAC | lr 3e-4, γ 0.99, τ 0.005, UTD 5, batch 256, replay 1e6 |
| target entropy | −2.0 |
| warmup | 5000 uniform-random env steps per skill |
| sleep | Adam, lr 2e-3, ≤4000 iters, 50 Frank-Wolfe iters |
| anchor | λ=1e-2, detector mode, threshold 150 |
| excitability | target 0.10, η wake 1e-3 / sleep 1e-2, transition 5e-3 for 500 eps |
| memory | 1000 raw states per skill, from the last 20% of episodes |

## Compute

One K=5 chain is roughly 2-3 days on a single modern GPU (13,000 episodes).
Several chains can run concurrently on one GPU, since the bottleneck is largely
simulation on CPU. Evaluation takes about 30 minutes per run.
