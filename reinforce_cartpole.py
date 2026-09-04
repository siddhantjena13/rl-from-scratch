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

def compute_policy_gradients(episode_observations, episode_actions, returns, weights, bias):
    grad_weights = np.zeros_like(weights)
    grad_bias = np.zeros_like(bias)

    for obs, action, return_t in zip(episode_observations, episode_actions, returns):
        probs = policy_forward(obs, weights, bias)

        dlogits = probs.copy()
        dlogits[action] -= 1
        dlogits *= return_t

        grad_weights += np.outer(obs, dlogits)
        grad_bias += dlogits

    return grad_weights, grad_bias

def update_policy(weights, bias, grad_weights, grad_bias, learning_rate):
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

def evaluate_policy(env, weights, bias, num_episodes):
    rewards = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            probs = policy_forward(obs, weights, bias)
            action = np.argmax(probs)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return np.mean(rewards)

def main():
    env = gym.make("CartPole-v1")

    rng = np.random.default_rng(42)
    weights, bias = initialize_policy(input_dim=4, output_dim=2, rng=rng)

    num_batches = 100
    batch_size = 10
    gamma = 0.99
    learning_rate = 0.05

    episode_rewards_history = []

    for batch in range(num_batches):
        batch_grad_weights = np.zeros_like(weights)
        batch_grad_bias = np.zeros_like(bias)

        batch_rewards = []

        for episode in range(batch_size):
            episode_observations, episode_actions, episode_rewards, total_reward = run_episode(
                env,
                weights,
                bias,
                rng,
            )

            returns = compute_discounted_returns(episode_rewards, gamma)
            returns = normalize(returns)

            grad_weights, grad_bias = compute_policy_gradients(
                episode_observations,
                episode_actions,
                returns,
                weights,
                bias,
            )

            batch_grad_weights += grad_weights
            batch_grad_bias += grad_bias
            batch_rewards.append(total_reward)
            episode_rewards_history.append(total_reward)

        batch_grad_weights /= batch_size
        batch_grad_bias /= batch_size

        weights, bias = update_policy(
            weights,
            bias,
            batch_grad_weights,
            batch_grad_bias,
            learning_rate,
        )

        if (batch + 1) % 5 == 0:
            recent_average = np.mean(episode_rewards_history[-50:])
            batch_average = np.mean(batch_rewards)
            print(
                f"Batch {batch + 1}: "
                f"batch average reward = {batch_average:.2f}, "
                f"recent average reward = {recent_average:.2f}"
            )
    
    evaluation_reward = evaluate_policy(
        env,
        weights,
        bias,
        num_episodes=20,
    )

    print(f"Evaluation average reward: {evaluation_reward:.2f}")


    env.close()


if __name__ == "__main__":
    main()
