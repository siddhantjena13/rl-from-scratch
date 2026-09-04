import gymnasium as gym
import numpy as np

def softmax(logits):
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def initialize_policy(input_dim, output_dim, rng):
    weights = rng.normal(loc=0.0, scale=0.01, size=(input_dim, output_dim))
    bias = np.zeros(output_dim)
    return weights, bias


def policy_forward(obs, weights, bias):
    logits = obs @ weights + bias
    probs = softmax(logits)
    return probs

def main():
    env = gym.make("CartPole-v1")

    rng = np.random.default_rng(42)
    weights, bias = initialize_policy(input_dim=4, output_dim=2, rng=rng)

    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0

    while not done:
        probs = policy_forward(obs, weights, bias)
        action = rng.choice(2, p=probs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    print("Random policy reward:", total_reward)
    env.close()


if __name__ == "__main__":
    main()