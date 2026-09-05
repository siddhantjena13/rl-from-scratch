# REINFORCE NumPy on CartPole

This project implements the REINFORCE policy-gradient algorithm on
`CartPole-v1` from Gymnasium.

The implementation is intentionally framework-free: Gymnasium provides the
environment, and NumPy handles the policy network, softmax, discounted returns,
manual backpropagation, and parameter updates.

## Features

- Stochastic policy for CartPole
- One-hidden-layer neural network written with NumPy
- Softmax action sampling during training
- Greedy action selection during evaluation
- Episode trajectory collection
- Discounted return computation
- Return normalization
- Manual REINFORCE gradient calculation
- Batched policy-gradient updates
- Best-policy checkpointing during training
- Early stopping when the environment is solved
- Evaluation of the trained policy

## Policy Architecture

```text
observation -> hidden layer -> tanh -> action probabilities
```

CartPole observations contain:

```text
[cart position, cart velocity, pole angle, pole angular velocity]
```

The policy outputs probabilities for:

```text
0 = push left
1 = push right
```

## Result

One training run solved the environment with:

```text
Solved at batch 215 with recent average reward 484.84
Evaluation average reward: 500.00
```

Because REINFORCE is a high-variance algorithm, individual runs can vary. The
implementation restores the best policy seen during training before evaluation.

## Run

From the repository root:

```bash
python reinforce_numpy/reinforce_cartpole.py
```

## Next Experiments

- Plot episode reward curves and save them under `results/`
- Compare per-episode return normalization with batch return normalization
- Report final-policy evaluation and best-policy evaluation separately
- Add a learned value baseline
- Port the same algorithm to PyTorch
