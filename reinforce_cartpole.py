import gymnasium as gym
import numpy as np

def softmax(logits):
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def initialize_policy(input_dim, hidden_dim, output_dim, rng):
    w1 = rng.normal(loc=0.0, scale=0.01, size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)

    w2 = rng.normal(loc=0.0, scale=0.01, size=(hidden_dim, output_dim))
    b2 = np.zeros(output_dim)

    return w1, b1, w2, b2


def policy_forward(obs, w1, b1, w2, b2):
    hidden_pre = obs @ w1 + b1
    hidden = np.tanh(hidden_pre)

    logits = hidden @ w2 + b2
    probs = softmax(logits)

    return probs, hidden

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

def compute_policy_gradients(episode_observations, episode_actions, returns, w1, b1, w2, b2):
    grad_w1 = np.zeros_like(w1)
    grad_b1 = np.zeros_like(b1)
    grad_w2 = np.zeros_like(w2)
    grad_b2 = np.zeros_like(b2)

    for obs, action, return_t in zip(episode_observations, episode_actions, returns):
        probs, hidden = policy_forward(obs, w1, b1, w2, b2)

        dlogits = probs.copy()
        dlogits[action] -= 1
        dlogits *= return_t

        grad_w2 += np.outer(hidden, dlogits)
        grad_b2 += dlogits

        dhidden = w2 @ dlogits
        dhidden_pre = dhidden * (1 - hidden ** 2)

        grad_w1 += np.outer(obs, dhidden_pre)
        grad_b1 += dhidden_pre

    return grad_w1, grad_b1, grad_w2, grad_b2

def update_policy(w1, b1, w2, b2, grad_w1, grad_b1, grad_w2, grad_b2, learning_rate):
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2

    return w1, b1, w2, b2

def run_episode(env, w1, b1, w2, b2, rng):
    obs, info = env.reset()
    done = False
    total_reward = 0

    episode_observations = []
    episode_actions = []
    episode_rewards = []

    while not done:
        probs, _ = policy_forward(obs, w1, b1, w2, b2)
        action = rng.choice(2, p=probs)

        episode_observations.append(obs)
        episode_actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)
        total_reward += reward
        done = terminated or truncated

    return episode_observations, episode_actions, episode_rewards, total_reward

def evaluate_policy(env, w1, b1, w2, b2, num_episodes):
    rewards = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            probs, _ = policy_forward(obs, w1, b1, w2, b2)
            action = np.argmax(probs)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return np.mean(rewards)

def main():
    env = gym.make("CartPole-v1")

    rng = np.random.default_rng(42)
    w1, b1, w2, b2 = initialize_policy(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        rng=rng,
    )

    num_batches = 100
    batch_size = 10
    gamma = 0.99
    learning_rate = 0.02

    episode_rewards_history = []

    for batch in range(num_batches):
        batch_grad_w1 = np.zeros_like(w1)
        batch_grad_b1 = np.zeros_like(b1)
        batch_grad_w2 = np.zeros_like(w2)
        batch_grad_b2 = np.zeros_like(b2)

        batch_rewards = []

        for episode in range(batch_size):
            episode_observations, episode_actions, episode_rewards, total_reward = run_episode(
                env,
                w1,
                b1,
                w2,
                b2,
                rng,
            )

            returns = compute_discounted_returns(episode_rewards, gamma)
            returns = normalize(returns)

            grad_w1, grad_b1, grad_w2, grad_b2 = compute_policy_gradients(
                episode_observations,
                episode_actions,
                returns,
                w1,
                b1,
                w2,
                b2,
            )

            batch_grad_w1 += grad_w1
            batch_grad_b1 += grad_b1
            batch_grad_w2 += grad_w2
            batch_grad_b2 += grad_b2

            batch_rewards.append(total_reward)
            episode_rewards_history.append(total_reward)

        batch_grad_w1 /= batch_size
        batch_grad_b1 /= batch_size
        batch_grad_w2 /= batch_size
        batch_grad_b2 /= batch_size

        w1, b1, w2, b2 = update_policy(
            w1,
            b1,
            w2,
            b2,
            batch_grad_w1,
            batch_grad_b1,
            batch_grad_w2,
            batch_grad_b2,
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
        w1,
        b1,
        w2,
        b2,
        num_episodes=20,
    )

    print(f"Evaluation average reward: {evaluation_reward:.2f}")


    env.close()


if __name__ == "__main__":
    main()
