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

def update_policy(episode_observations, episode_actions, returns, weights, bias, learning_rate):
    grad_weights = np.zeros_like(weights)
    grad_bias = np.zeros_like(bias)

    for obs, action, return_t in zip(episode_observations, episode_actions, returns):
        probs = policy_forward(obs, weights, bias)

        dlogits = probs.copy()
        dlogits[action] -= 1
        dlogits *= return_t

        grad_weights += np.outer(obs, dlogits)
        grad_bias += dlogits

    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias

    return weights, bias

def run_episode(env, weights, bias, rng):
    obs, info = env.reset()
    done = False
    total_reward = 0

    episode_observations = []
    episode_actions = []
    episode_rewards = []

    while not done:
        probs = policy_forward(obs, weights, bias)
        action = rng.choice(2, p=probs)

        episode_observations.append(obs)
        episode_actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)
        total_reward += reward
        done = terminated or truncated

    return episode_observations, episode_actions, episode_rewards, total_reward

def main():
    env = gym.make("CartPole-v1")

    rng = np.random.default_rng(42)
    weights, bias = initialize_policy(input_dim=4, output_dim=2, rng=rng)

    num_episodes = 1000
    gamma = 0.99
    learning_rate = 0.005

    episode_rewards_history = []

    for episode in range(num_episodes):
        episode_observations, episode_actions, episode_rewards, total_reward = run_episode(
            env,
            weights,
            bias,
            rng,
        )

        returns = compute_discounted_returns(episode_rewards, gamma)
        returns = normalize(returns)

        episode_rewards_history.append(total_reward)

        weights, bias = update_policy(
            episode_observations,
            episode_actions,
            returns,
            weights,
            bias,
            learning_rate,
        )

        if (episode + 1) % 50 == 0:
            recent_average = np.mean(episode_rewards_history[-50:])
            print(
                f"Episode {episode + 1}: "
                f"reward = {total_reward}, "
                f"average reward = {recent_average:.2f}"
            )

    env.close()


if __name__ == "__main__":
    main()
