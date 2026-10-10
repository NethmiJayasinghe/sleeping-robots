# Sleeping Robots

**Offline skill consolidation for a shared robot policy.**

Code for the CoRL 2026 paper *When Robots Sleep: Offline Skill Consolidation for
Shared-Policy Robot Learning*.

[Paper (arXiv)](https://arxiv.org/abs/2606.17493) | [Project page](https://nethmijayasinghe.github.io/projects/when-robots-sleep.html)

A single policy learns a stream of skills one at a time. Each new skill is
acquired in a **wake** phase that updates the shared weights. The policy then
enters an offline **sleep** phase with no new data, where each skill's frozen
memory bargains over the shared parameters until they reach a new equilibrium
that serves all of them.

This repository contains the method and the **Meta-World MT5** experiment.

---

## How it works

For a stream of skills `1..K` sharing one actor `π_θ`:

**Wake (skill k).** Ordinary SAC on skill `k` only, updating `θ` directly. Two
things survive the phase: a **frozen twin critic** `Q_k` and a **raw state
buffer** `S_k` sampled from the last 20% of episodes. No transitions, actions,
rewards, or task losses are kept.

**Sleep (offline).** For each skill `k ≤ K`, form

```
L_k(θ) = −E_{s∼S_k}[ Q_k(s, π_θ(s)) ]
```

take its gradient, normalise all `K` gradients to unit length, and combine them
into one direction with a Frank-Wolfe solver for the Nash bargaining objective
`F(α) = Σ_k log⟨g_k, g_α⟩`:

```
g* = Σ_k α_k ĝ_k
```

Optionally add a proximal anchor `λ(θ − θ_anchor)` toward the previous
equilibrium, then step with Adam. Repeat until the per-skill losses plateau.

**Per-neuron excitability.** Every hidden unit carries a homeostatic gain `e_i`
that receives no gradient and is driven only by its own activation statistics.
It operates in both phases, at different rates.

---

## Install

```bash
conda env create -f environment.yml
conda activate sleeping-robots
pip install -e .
```

Or into an existing environment:

```bash
pip install -r requirements.txt && pip install -e .
```

Needs Meta-World with the 3.0 task API (`reach-v3`, `door-open-v3`, ...) and a
working MuJoCo. On a headless machine set `MUJOCO_GL=egl` (the scripts do this
for you).

---

## Quickstart

Train the full five-skill stream, then evaluate the saved checkpoints:

```bash
python scripts/train_mt5.py --seed 42 --ckpt-dir runs/seed42
python scripts/evaluate.py  --run-dirs runs/seed42 --out results/seed42
```

The stream is `reach → door-open → drawer-open → window-close →
button-press-topdown`, 13,000 episodes total. Expect roughly 2-3 days on one
modern GPU.

Training writes two checkpoints per skill:

```
runs/seed42/
  skill1_post_wake.pt    skill1_post_sleep.pt
  ...                    skillK_post_sleep.pt
  final.pt
```

Evaluation is run afterwards from those checkpoints. See
`docs/REPRODUCTION.md` for more commands and the configuration.

---

## Metrics

| Metric | Meaning |
|---|---|
| **AFSR** | average final success rate over the K skills |
| **PRS** | pairwise reliability: `min(S_i, S_j)` over six pre-registered related-skill pairs |
| **BWT** | backward transfer; negative means forgetting |
| **AF** | average forgetting from each skill's peak |
| **MC** | retained memory in MB |
| **SPI** | synergy preservation index: overlap of the top-3 observation principal subspaces before vs. after |

---

## Results

Meta-World MT5 after learning five skills, as reported in the paper
(mean ± standard error):

| Metric | Sleeping Robots |
|---|---|
| AFSR | 0.578 ± 0.016 |
| PRS | 0.245 ± 0.020 |
| BWT | −0.280 ± 0.020 |
| SPI | 0.767 ± 0.017 |

See the paper for baselines, ablations, and the SurgicAI results.

---

## Repository layout

```
sleeping_robots/
  config.py        hyperparameters
  stream.py        MT5 task stream, budgets, stopping rules, PRS pairs
  policy.py        shared actor + per-skill frozen twin critic
  excitability.py  per-neuron homeostatic gain
  nash_solver.py   Frank-Wolfe Nash bargaining over skill gradients
  anchor.py        proximal anchor and the drift detector that gates it
  phases/wake.py   acquisition (SAC) + critic gate + state-buffer sampling
  phases/sleep.py  offline consolidation
  trainer.py       stream orchestration and checkpointing
  evaluation.py    post-hoc evaluation of saved checkpoints
  metrics.py       AFSR, PRS, BWT, AF, MC
  spi.py           synergy preservation index
scripts/           train_mt5.py, evaluate.py
tests/             unit tests and an end-to-end smoke test
docs/              REPRODUCTION.md
```

Scope: the shared-policy method on MT5. The SurgicAI imitation-learning
experiments, the comparison baselines, and the physical-robot demonstration are
not included here.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/test_components.py -q
SR_RUN_SMOKE=1 pytest tests/test_smoke_e2e.py -s     # miniature end-to-end run
```

---

## Citation

```bibtex
@inproceedings{jayasinghe2026sleeping,
  title     = {When Robots Sleep: Offline Skill Consolidation for Shared-Policy
               Robot Learning},
  author    = {Jayasinghe, Nethmi and Gontero, Diana and Trivedi, Amit Ranjan},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  eprint    = {2606.17493},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO}
}
```

## Contact

Questions about the code or the experiments: Nethmi Jayasinghe (wjayas3@uic.edu).

## License

MIT - see `LICENSE`.
