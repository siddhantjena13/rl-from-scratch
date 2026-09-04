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

def compute_discounted_returns(rewards, gamma):
    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return np.array(returns)

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

def update_policy(episode_obs, episode_actions, returns, weights, bias, learning_rate):
    grad_weights = np.zeros_like(weights)
    grad_bias = np.zeros_like(bias)

    for obs, action, return_t in zip(episode_obs, episode_actions, returns):
        probs = policy_forward(obs, weights, bias)

        dlogits = probs.copy()
        dlogits[action] -= 1
        dlogits *= return_t

        grad_weights += np.outer(obs, dlogits)
        grad_bias += dlogits

    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias

    return weights, bias

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
    
    gamma = 0.99
    returns = compute_discounted_returns(episode_rewards, gamma)
    returns = normalize(returns)

    learning_rate = 0.01
    weights, bias = update_policy(
        episode_obs,
        episode_actions,
        returns,
        weights,
        bias,
        learning_rate,
    )

    print("Updated weights:", weights)
    print("Updated bias:", bias)
    env.close()


if __name__ == "__main__":
    main()