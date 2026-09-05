# RL From Scratch

This repository is a long-term reinforcement learning playground. The goal is
to build core RL algorithms step by step, starting with small environments where
the math is visible, then moving toward more scalable PyTorch implementations.

The first project is a NumPy implementation of REINFORCE on `CartPole-v1`.

## Roadmap

- REINFORCE with manual NumPy gradients
- REINFORCE with a learned value baseline
- Advantage actor-critic
- PyTorch versions of the same algorithms
- PPO
- Larger experiments with cleaner result tracking

## Projects

| Project | Environment | Implementation | Status |
| --- | --- | --- | --- |
| [REINFORCE NumPy](reinforce_numpy/README.md) | CartPole-v1 | NumPy + Gymnasium | Solves CartPole |

## Current Result

The first implementation trains a hidden-layer stochastic policy with manual
backpropagation and batched REINFORCE updates.

One training run reached:

```text
Solved at batch 215 with recent average reward 484.84
Evaluation average reward: 500.00
```

## Repository Layout

```text
rl-from-scratch/
├── README.md
├── requirements.txt
├── reinforce_numpy/
│   ├── README.md
│   └── reinforce_cartpole.py
├── results/
└── notes/
```

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the current REINFORCE project:

```bash
python reinforce_numpy/reinforce_cartpole.py
```

## Why This Repo

The point is not just to get high rewards. The point is to make each algorithm
explainable:

- What objective is being optimized?
- What data is collected from the environment?
- How are returns or advantages computed?
- Where do the gradients come from?
- What changes when we add a baseline or move to actor-critic?

The early projects intentionally avoid high-level RL libraries so the learning
signal, gradient flow, and training behavior stay visible.
