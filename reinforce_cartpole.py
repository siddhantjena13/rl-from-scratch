import gymnasium as gym
import numpy as np


def main():
    env = gym.make("CartPole-v1")

    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    print("Random policy reward:", total_reward)
    env.close()


if __name__ == "__main__":
    main()