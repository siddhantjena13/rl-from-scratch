import gymnasium as gym
import numpy as np


def softmax(logits):
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def make_env(seed):
    # seed the env and the action sampler so runs are repeatable
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def initialize_policy(input_dim, hidden_dim, output_dim, rng):
    w1 = rng.normal(loc=0.0, scale=0.1, size=(input_dim, hidden_dim))
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


def initialize_value_fn(input_dim, hidden_dim, rng):
    vw1 = rng.normal(loc=0.0, scale=0.1, size=(input_dim, hidden_dim))
    vb1 = np.zeros(hidden_dim)

    vw2 = rng.normal(loc=0.0, scale=0.01, size=(hidden_dim, 1))
    vb2 = np.zeros(1)

    return vw1, vb1, vw2, vb2


def value_forward(obs, vw1, vb1, vw2, vb2):
    hidden_pre = obs @ vw1 + vb1
    hidden = np.tanh(hidden_pre)

    value = hidden @ vw2 + vb2

    return value[0], hidden


def compute_td_errors(episode_observations, episode_rewards, episode_next_observations,
                      episode_terminated, gamma, vw1, vb1, vw2, vb2):
    td_errors = []
    values = []

    for obs, reward, next_obs, terminated in zip(
        episode_observations, episode_rewards, episode_next_observations, episode_terminated
    ):
        value, _ = value_forward(obs, vw1, vb1, vw2, vb2)

        if terminated:
            next_value = 0.0
        else:
            next_value, _ = value_forward(next_obs, vw1, vb1, vw2, vb2)

        values.append(value)
        td_errors.append(reward + gamma * next_value - value)

    return np.array(td_errors), np.array(values)


def compute_value_gradients(batch_observations, batch_returns, vw1, vb1, vw2, vb2):
    grad_vw1 = np.zeros_like(vw1)
    grad_vb1 = np.zeros_like(vb1)
    grad_vw2 = np.zeros_like(vw2)
    grad_vb2 = np.zeros_like(vb2)

    for obs, actual_return in zip(batch_observations, batch_returns):
        predicted_value, hidden = value_forward(obs, vw1, vb1, vw2, vb2)

        # derivative of (1/2)(V(s) - G)^2 with respect to V(s)
        d_value = predicted_value - actual_return

        grad_vw2 += np.outer(hidden, d_value)
        grad_vb2 += d_value

        d_hidden = vw2.flatten() * d_value
        d_hidden_pre = d_hidden * (1 - hidden ** 2)

        grad_vw1 += np.outer(obs, d_hidden_pre)
        grad_vb1 += d_hidden_pre

    # average over timesteps
    num_steps = len(batch_observations)

    grad_vw1 /= num_steps
    grad_vb1 /= num_steps
    grad_vw2 /= num_steps
    grad_vb2 /= num_steps

    return grad_vw1, grad_vb1, grad_vw2, grad_vb2


def update_value_fn(vw1, vb1, vw2, vb2, grad_vw1, grad_vb1, grad_vw2, grad_vb2, learning_rate):
    vw1 -= learning_rate * grad_vw1
    vb1 -= learning_rate * grad_vb1
    vw2 -= learning_rate * grad_vw2
    vb2 -= learning_rate * grad_vb2

    return vw1, vb1, vw2, vb2


def compute_policy_gradients(batch_observations, batch_actions, batch_hidden, batch_probs, weights, w1, b1, w2, b2):
    grad_w1 = np.zeros_like(w1)
    grad_b1 = np.zeros_like(b1)
    grad_w2 = np.zeros_like(w2)
    grad_b2 = np.zeros_like(b2)

    for obs, action, hidden, probs, weight in zip(batch_observations, batch_actions, batch_hidden, batch_probs, weights):
        # gradient of -log(prob of the action we took). for a softmax this is
        # just the probs with a 1 subtracted at the chosen action.
        dlogits = probs.copy()
        dlogits[action] -= 1
        dlogits *= weight

        grad_w2 += np.outer(hidden, dlogits)
        grad_b2 += dlogits

        dhidden = w2 @ dlogits
        dhidden_pre = dhidden * (1 - hidden ** 2)

        grad_w1 += np.outer(obs, dhidden_pre)
        grad_b1 += dhidden_pre

    # average over timesteps, not episodes
    num_steps = len(batch_observations)

    grad_w1 /= num_steps
    grad_b1 /= num_steps
    grad_w2 /= num_steps
    grad_b2 /= num_steps

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

    episode_old_probs = []
    episode_observations = []
    episode_actions = []
    episode_rewards = []
    episode_hidden = []
    episode_probs = []

    # need the next state so the critic can estimate the rest of the future
    episode_next_observations = []

    # terminated = the pole fell, so there is no future left to value.
    # truncated = we only hit the 500-step limit, which is not the same thing.
    episode_terminated = []

    while not done:
        probs, hidden = policy_forward(obs, w1, b1, w2, b2)
        action = rng.choice(2, p=probs)

        episode_observations.append(obs)
        episode_actions.append(action)
        episode_hidden.append(hidden)
        episode_probs.append(probs)
        episode_old_probs.append(probs[action])

        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)
        episode_next_observations.append(next_obs)
        episode_terminated.append(terminated)

        total_reward += reward
        done = terminated or truncated
        obs = next_obs

    return (episode_observations, episode_actions, episode_rewards, episode_hidden,
        episode_probs, episode_next_observations, episode_terminated,
        episode_old_probs, total_reward)


def evaluate_policy(env, w1, b1, w2, b2, num_episodes, seed):
    rewards = []

    for i in range(num_episodes):
        # fixed seeds so evaluation gives the same numbers every time
        obs, info = env.reset(seed=seed + i)
        done = False
        total_reward = 0

        while not done:
            probs, _ = policy_forward(obs, w1, b1, w2, b2)
            action = np.argmax(probs)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def episodes_to_solve(episode_rewards_history, threshold=475.0, window=50):
    # first episode where the average of the last 50 hits the threshold
    if len(episode_rewards_history) < window:
        return None

    for end in range(window, len(episode_rewards_history) + 1):
        if np.mean(episode_rewards_history[end - window:end]) >= threshold:
            return end

    return None


def train(seed=0, normalization="batch", num_batches=200, batch_size=10, gamma=0.99,
          learning_rate=0.5, value_learning_rate=0.1, clip_epsilon=0.2, ppo_epochs=4,
          advantage_estimator="td", gae_lambda=0.95,
          hidden_dim=16, eval_episodes=20, verbose=True):
    env = make_env(seed)
    eval_env = make_env(seed + 10000)
    rng = np.random.default_rng(seed)

    w1, b1, w2, b2 = initialize_policy(input_dim=4, hidden_dim=hidden_dim, output_dim=2, rng=rng)
    vw1, vb1, vw2, vb2 = initialize_value_fn(input_dim=4, hidden_dim=hidden_dim, rng=rng)

    episode_rewards_history = []
    best_recent_average = -np.inf
    best_params = (w1.copy(), b1.copy(), w2.copy(), b2.copy())

    for batch in range(num_batches):
        batch_observations = []
        batch_actions = []
        batch_hidden = []
        batch_probs = []
        batch_weights = []
        batch_returns = []
        batch_rewards = []
        batch_old_probs = []

        for episode in range(batch_size):
            episode_observations, episode_actions, episode_rewards, episode_hidden, episode_probs, episode_next_observations, episode_terminated, episode_old_probs, total_reward = run_episode(
                env, w1, b1, w2, b2, rng,)

            # weight each step by the advantage instead of the raw return. the
            # critic is used here before it gets updated further down.
            td_errors, values = compute_td_errors(
                episode_observations, episode_rewards, episode_next_observations,
                episode_terminated, gamma, vw1, vb1, vw2, vb2,
            )

            if advantage_estimator == "gae":
                advantages = compute_gae(td_errors, gamma, gae_lambda)
            else:
                advantages = td_errors

            # advantage + value gives the return back, which is what the
            # critic should be predicting
            value_targets = advantages + values   
            # "episode": normalize inside each episode
            # "batch": normalize once across the whole batch
            if normalization == "episode":
                advantages = normalize(advantages)

            batch_observations.extend(episode_observations)
            batch_actions.extend(episode_actions)
            batch_hidden.extend(episode_hidden)
            batch_probs.extend(episode_probs)
            batch_weights.extend(advantages)
            batch_returns.extend(value_targets)
            batch_old_probs.extend(episode_old_probs)

            batch_rewards.append(total_reward)
            episode_rewards_history.append(total_reward)

        batch_weights = np.array(batch_weights)
        batch_returns = np.array(batch_returns)
        batch_old_probs = np.array(batch_old_probs)

        if normalization == "batch":
            batch_weights = normalize(batch_weights)

        # several passes over the same batch. the clip is what stops the policy
        # from drifting too far from the one that collected the data.
        for epoch in range(ppo_epochs):
            grad_w1, grad_b1, grad_w2, grad_b2 = compute_ppo_policy_gradients(
                batch_observations, batch_actions, batch_old_probs, batch_weights,
                w1, b1, w2, b2, clip_epsilon,
            )

            w1, b1, w2, b2 = update_policy(
                w1, b1, w2, b2, grad_w1, grad_b1, grad_w2, grad_b2, learning_rate,
            )

            grad_vw1, grad_vb1, grad_vw2, grad_vb2 = compute_value_gradients(
                batch_observations, batch_returns, vw1, vb1, vw2, vb2,
            )

            vw1, vb1, vw2, vb2 = update_value_fn(
                vw1, vb1, vw2, vb2, grad_vw1, grad_vb1, grad_vw2, grad_vb2, value_learning_rate,
            )

        recent_average = np.mean(episode_rewards_history[-50:])

        if len(episode_rewards_history) >= 50 and recent_average > best_recent_average:
            best_recent_average = recent_average
            best_params = (w1.copy(), b1.copy(), w2.copy(), b2.copy())

        if verbose and (batch + 1) % 10 == 0:
            print(
                f"Batch {batch + 1}: "
                f"batch average reward = {np.mean(batch_rewards):.2f}, "
                f"recent average reward = {recent_average:.2f}"
            )

    # evaluate the final policy and the best one we saved
    final_mean, final_std = evaluate_policy(eval_env, w1, b1, w2, b2, eval_episodes, seed + 10000)

    best_w1, best_b1, best_w2, best_b2 = best_params
    best_mean, best_std = evaluate_policy(eval_env, best_w1, best_b1, best_w2, best_b2, eval_episodes, seed + 10000)

    env.close()
    eval_env.close()

    return {
        "algorithm": "ppo",
        "normalization": normalization,
        "seed": seed,
        "episode_returns": episode_rewards_history,
        "episodes_to_solve": episodes_to_solve(episode_rewards_history),
        "eval_final_mean": final_mean,
        "eval_final_std": final_std,
        "eval_best_mean": best_mean,
        "eval_best_std": best_std,
    }

def compute_ppo_policy_gradients(batch_observations, batch_actions, batch_old_probs,
                                 weights, w1, b1, w2, b2, clip_epsilon):
    grad_w1 = np.zeros_like(w1)
    grad_b1 = np.zeros_like(b1)
    grad_w2 = np.zeros_like(w2)
    grad_b2 = np.zeros_like(b2)

    for obs, action, old_prob, advantage in zip(batch_observations, batch_actions, batch_old_probs, weights):
        # the policy changes every epoch, so the forward pass has to be redone
        probs, hidden = policy_forward(obs, w1, b1, w2, b2)

        ratio = probs[action] / old_prob

        unclipped = ratio * advantage
        clipped = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage

        # PPO takes the smaller of the two terms. if the clipped one wins, the
        # objective is flat and this step gets no gradient.
        if unclipped <= clipped:
            d_objective_d_ratio = advantage
        else:
            d_objective_d_ratio = 0.0

        # chain rule: d(ratio)/d(logits) = ratio * d(log p)/d(logits)
        dlogits = probs.copy()
        dlogits[action] -= 1
        dlogits *= d_objective_d_ratio * ratio

        grad_w2 += np.outer(hidden, dlogits)
        grad_b2 += dlogits

        dhidden = w2 @ dlogits
        dhidden_pre = dhidden * (1 - hidden ** 2)

        grad_w1 += np.outer(obs, dhidden_pre)
        grad_b1 += dhidden_pre

    num_steps = len(batch_observations)

    grad_w1 /= num_steps
    grad_b1 /= num_steps
    grad_w2 /= num_steps
    grad_b2 /= num_steps

    return grad_w1, grad_b1, grad_w2, grad_b2

def compute_gae(td_errors, gamma, lam):
    advantages = []
    running_advantage = 0.0

    for delta in reversed(td_errors):
        running_advantage = delta + gamma * lam * running_advantage
        advantages.append(running_advantage)

    advantages.reverse()
    return np.array(advantages)


def main():
    result = train(seed=0, normalization="batch")

    solved = result["episodes_to_solve"]

    print()
    print(f"Algorithm: {result['algorithm']}, normalization: {result['normalization']}, seed: {result['seed']}")
    print(f"Episodes to solve: {solved if solved is not None else 'not solved'}")
    print(f"Evaluation, final policy:    {result['eval_final_mean']:.2f} +/- {result['eval_final_std']:.2f}")
    print(f"Evaluation, best checkpoint: {result['eval_best_mean']:.2f} +/- {result['eval_best_std']:.2f}")


if __name__ == "__main__":
    main()