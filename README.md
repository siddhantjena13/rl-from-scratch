# REINFORCE From Scratch on CartPole

This project implements the REINFORCE policy-gradient algorithm on the
`CartPole-v1` environment from Gymnasium.

The goal is to build the reinforcement learning pipeline from scratch in a
single Python file, using NumPy for the model math and Gymnasium only for the
environment.

## Finished Project Goal

By the end, this repository will contain a complete implementation of:

- A stochastic policy for CartPole
- A small neural network written with NumPy
- Softmax action probabilities
- Episode rollout collection
- Discounted return computation
- The REINFORCE policy-gradient update
- Manual gradient calculations and parameter updates
- Training logs showing learning progress
- Evaluation of the trained policy

## Current Implementation

The project currently includes:

- A working CartPole training loop
- A NumPy linear policy
- Softmax action sampling
- Episode trajectory collection
- Discounted return calculation
- Return normalization
- Manual policy-gradient computation
- Batched REINFORCE updates
- Evaluation using the trained policy

Recent training runs reach substantially better-than-random performance, with
moving-average rewards in the hundreds of timesteps.

## Environment

CartPole has:

- Observation space: 4 continuous values
- Action space: 2 discrete actions
- Reward: `+1` for every step the pole remains balanced
- Maximum episode length: 500 steps for `CartPole-v1`

The policy receives the observation:

```text
[cart position, cart velocity, pole angle, pole angular velocity]
```

and outputs a probability distribution over:

```text
0 = push left
1 = push right
```

## Why This Project

This is a learning-focused implementation. Instead of using a deep learning
framework like PyTorch or TensorFlow, the core reinforcement learning logic is
implemented directly so that each part of REINFORCE is visible and explainable.

## Current Status

The current policy is linear:

```text
observation -> action probabilities
```

The next milestone is to replace it with a small hidden-layer neural network:

```text
observation -> hidden layer -> tanh -> action probabilities
```

This will make the model more expressive while still keeping every part of the
forward pass, backpropagation, and parameter update implemented from scratch in
NumPy.

## Running

Install dependencies:

```bash
python -m pip install gymnasium numpy
```

Run the current scaffold:

```bash
python reinforce_cartpole.py
```
