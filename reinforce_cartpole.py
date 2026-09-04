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

    episode_obs = []
    episode_actions = []
    episode_rewards = []

    while not done:
        probs = policy_forward(obs, weights, bias)
        action = rng.choice(2, p=probs)

        episode_obs.append(obs)
        episode_actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)

        total_reward += reward
        done = terminated or truncated
    
    print("Episode length:", len(episode_rewards))
    print("First observation:", episode_obs[0])
    print("First action:", episode_actions[0])
    print("First reward:", episode_rewards[0])

    print("Random policy reward:", total_reward)
    env.close()


if __name__ == "__main__":
    main()