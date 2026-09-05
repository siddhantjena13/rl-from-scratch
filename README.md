# REINFORCE From Scratch on CartPole

This project implements the REINFORCE policy-gradient algorithm on the
`CartPole-v1` environment from Gymnasium.

The goal is to build the reinforcement learning pipeline from scratch in a
single Python file, using NumPy for the model math and Gymnasium only for the
environment.

## Project Features

This repository contains a complete implementation of:

- A stochastic policy for CartPole
- A small hidden-layer neural network written with NumPy
- Softmax action probabilities
- Episode rollout collection
- Discounted return computation
- The REINFORCE policy-gradient update
- Manual gradient calculations and parameter updates
- Batched policy-gradient updates
- Best-policy checkpointing during training
- Early stopping when the environment is solved
- Evaluation of the trained policy

## Current Implementation

The policy network uses one hidden layer:

```text
observation -> hidden layer -> tanh -> action probabilities
```

The implementation is intentionally framework-free. Gymnasium provides the
CartPole environment, and NumPy handles the policy network, softmax, discounted
returns, backpropagation, and parameter updates.

One training run solved the environment with:

```text
Solved at batch 215 with recent average reward 484.84
Evaluation average reward: 500.00
```

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

## Running

Install dependencies:

```bash
python -m pip install gymnasium numpy
```

Run training:

```bash
python reinforce_cartpole.py
```
