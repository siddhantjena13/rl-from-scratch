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
    episode_action_probs = []  #remember how likely each action was, at the time we took it

    while not done:
        probs, _ = policy_forward(obs, w1, b1, w2, b2)
        action = rng.choice(2, p=probs)

        episode_observations.append(obs)
        episode_actions.append(action)

        #save the probability the policy assigned to the action we picked.
        # this is our "old" probability — a snapshot, frozen at collection time
        episode_action_probs.append(probs[action])

        obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)
        total_reward += reward
        done = terminated or truncated

    return episode_observations, episode_actions, episode_rewards, episode_action_probs, total_reward

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

def initialize_value_fn(input_dim, hidden_dim, rng):
    vw1 = rng.normal(loc=0.0, scale=0.01, size=(input_dim, hidden_dim))
    vb1 = np.zeros(hidden_dim)

    vw2 = rng.normal(loc=0.0, scale=0.01, size=(hidden_dim, 1))
    vb2 = np.zeros(1)

    return vw1, vb1, vw2, vb2


def value_forward(obs, vw1, vb1, vw2, vb2):
    hidden_pre = obs @ vw1 + vb1
    hidden = np.tanh(hidden_pre)

    value = hidden @ vw2 + vb2

    return value[0], hidden

def compute_advantages(episode_observations, returns, vw1, vb1, vw2, vb2):
    advantages = []

    for obs, actual_return in zip(episode_observations, returns):
        # ask the value network: "how good did you think this state was?"
        predicted_value, _ = value_forward(obs, vw1, vb1, vw2, vb2)

        # the advantage is how much better (or worse) the real outcome was
        advantage = actual_return - predicted_value
        advantages.append(advantage)

    return np.array(advantages)

def compute_value_gradients(episode_observations, returns, vw1, vb1, vw2, vb2):
    grad_vw1 = np.zeros_like(vw1)
    grad_vb1 = np.zeros_like(vb1)
    grad_vw2 = np.zeros_like(vw2)
    grad_vb2 = np.zeros_like(vb2)

    for obs, actual_return in zip(episode_observations, returns):
        predicted_value, hidden = value_forward(obs, vw1, vb1, vw2, vb2)

        # how far off was our prediction? this is the gradient of the
        # squared error loss with respect to the network's output
        d_value = predicted_value - actual_return

        # gradient for the second layer (hidden -> output)
        grad_vw2 += np.outer(hidden, d_value)
        grad_vb2 += d_value

        # backpropagate the error through the second layer's weights
        d_hidden = vw2.flatten() * d_value

        # backpropagate through the tanh activation
        # (derivative of tanh(x) is 1 - tanh(x)^2)
        d_hidden_before_activation = d_hidden * (1 - hidden ** 2)

        # gradient for the first layer (input -> hidden)
        grad_vw1 += np.outer(obs, d_hidden_before_activation)
        grad_vb1 += d_hidden_before_activation

    return grad_vw1, grad_vb1, grad_vw2, grad_vb2

def update_value_fn(vw1, vb1, vw2, vb2, grad_vw1, grad_vb1, grad_vw2, grad_vb2, learning_rate):
    # move the weights a little bit in the direction that reduces
    # the value network's prediction error
    vw1 -= learning_rate * grad_vw1
    vb1 -= learning_rate * grad_vb1
    vw2 -= learning_rate * grad_vw2
    vb2 -= learning_rate * grad_vb2

    return vw1, vb1, vw2, vb2

def compute_ppo_policy_gradients(episode_observations, episode_actions, old_action_probs, advantages, w1, b1, w2, b2, clip_epsilon):
    grad_w1 = np.zeros_like(w1)
    grad_b1 = np.zeros_like(b1)
    grad_w2 = np.zeros_like(w2)
    grad_b2 = np.zeros_like(b2)

    for obs, action, old_prob, advantage in zip(episode_observations, episode_actions, old_action_probs, advantages):
        # ask the CURRENT (possibly already-updated) policy what it thinks now
        probs, hidden = policy_forward(obs, w1, b1, w2, b2)
        new_prob = probs[action]

        # the ratio: how much more (or less) likely is this action now,
        # compared to when we collected the data?
        ratio = new_prob / old_prob

        # the two candidate objectives PPO chooses between
        unclipped_objective = ratio * advantage
        clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        clipped_objective = clipped_ratio * advantage

        # PPO takes whichever is SMALLER. this is the whole trick:
        # it removes the incentive to keep pushing the ratio further
        # once clipping has kicked in
        if unclipped_objective <= clipped_objective:
            d_objective_d_ratio = advantage
        else:
            d_objective_d_ratio = 0.0

        # chain rule: d(ratio)/d(logits) = ratio * (onehot(action) - probs)
        # this comes from differentiating softmax - same math you already
        # used in compute_policy_gradients, just multiplied by "ratio" now
        dlogits = probs.copy()
        dlogits[action] -= 1
        dlogits *= -(d_objective_d_ratio * ratio)  # negative sign flips this into "descent" form, matching your existing w -= lr * grad pattern

        grad_w2 += np.outer(hidden, dlogits)
        grad_b2 += dlogits

        dhidden = w2 @ dlogits
        dhidden_pre = dhidden * (1 - hidden ** 2)

        grad_w1 += np.outer(obs, dhidden_pre)
        grad_b1 += dhidden_pre

    return grad_w1, grad_b1, grad_w2, grad_b2

def main():
    env = gym.make("CartPole-v1")

    rng = np.random.default_rng(42)
    w1, b1, w2, b2 = initialize_policy(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        rng=rng,
    )

    # NEW: create the value network's starting weights too
    vw1, vb1, vw2, vb2 = initialize_value_fn(
        input_dim=4,
        hidden_dim=16,
        rng=rng,
    )

    num_batches = 300
    batch_size = 10
    gamma = 0.99
    learning_rate = 0.04
    value_learning_rate = 0.001  
    solved_reward = 475

    best_recent_average = -np.inf
    best_params = None

    episode_rewards_history = []

    for batch in range(num_batches):
        batch_grad_w1 = np.zeros_like(w1)
        batch_grad_b1 = np.zeros_like(b1)
        batch_grad_w2 = np.zeros_like(w2)
        batch_grad_b2 = np.zeros_like(b2)

        # NEW: separate accumulators for the value network's gradients
        batch_grad_vw1 = np.zeros_like(vw1)
        batch_grad_vb1 = np.zeros_like(vb1)
        batch_grad_vw2 = np.zeros_like(vw2)
        batch_grad_vb2 = np.zeros_like(vb2)

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

            # NEW: ask the value network how good it thought each state was,
            # and use the difference from the real return as the advantage
            advantages = compute_advantages(
                episode_observations,
                returns,
                vw1,
                vb1,
                vw2,
                vb2,
            )
            advantages = normalize(advantages)

            # CHANGED: the policy gradient now uses advantages, not raw returns
            grad_w1, grad_b1, grad_w2, grad_b2 = compute_policy_gradients(
                episode_observations,
                episode_actions,
                advantages,
                w1,
                b1,
                w2,
                b2,
            )

            # NEW: also compute how the value network should update itself,
            # using the un-normalized returns as its training target
            grad_vw1, grad_vb1, grad_vw2, grad_vb2 = compute_value_gradients(
                episode_observations,
                returns,
                vw1,
                vb1,
                vw2,
                vb2,
            )

            batch_grad_w1 += grad_w1
            batch_grad_b1 += grad_b1
            batch_grad_w2 += grad_w2
            batch_grad_b2 += grad_b2

            # NEW: accumulate the value network's gradients too
            batch_grad_vw1 += grad_vw1
            batch_grad_vb1 += grad_vb1
            batch_grad_vw2 += grad_vw2
            batch_grad_vb2 += grad_vb2

            batch_rewards.append(total_reward)
            episode_rewards_history.append(total_reward)

        batch_grad_w1 /= batch_size
        batch_grad_b1 /= batch_size
        batch_grad_w2 /= batch_size
        batch_grad_b2 /= batch_size

        # NEW: average the value network's gradients over the batch too
        batch_grad_vw1 /= batch_size
        batch_grad_vb1 /= batch_size
        batch_grad_vw2 /= batch_size
        batch_grad_vb2 /= batch_size

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

        # NEW: update the value network's weights too
        vw1, vb1, vw2, vb2 = update_value_fn(
            vw1,
            vb1,
            vw2,
            vb2,
            batch_grad_vw1,
            batch_grad_vb1,
            batch_grad_vw2,
            batch_grad_vb2,
            value_learning_rate,
        )

        # (the rest of the loop — printing progress, checkpointing best_params,
        # checking solved_reward — stays exactly the same as before)

        if (batch + 1) % 5 == 0:
            recent_average = np.mean(episode_rewards_history[-50:])
            batch_average = np.mean(batch_rewards)

            print(
                f"Batch {batch + 1}: "
                f"batch average reward = {batch_average:.2f}, "
                f"recent average reward = {recent_average:.2f}"
            )

            if recent_average > best_recent_average:
                best_recent_average = recent_average
                best_params = (
                    w1.copy(),
                    b1.copy(),
                    w2.copy(),
                    b2.copy(),
                )

            if recent_average >= solved_reward:
                print(
                    f"Solved at batch {batch + 1} "
                    f"with recent average reward {recent_average:.2f}"
                )
                break
            
    print("Training complete.")

    if best_params is not None:
        w1, b1, w2, b2 = best_params
        print(f"Restored best policy with recent average reward {best_recent_average:.2f}")

    evaluation_reward = evaluate_policy(
        env,
        w1,
        b1,
        w2,
        b2,
        num_episodes=20,
    )

    print(f"Evaluation average reward: {evaluation_reward:.2f}")
    print(f"Best recent average reward: {best_recent_average:.2f}")


    env.close()


if __name__ == "__main__":
    main()
