from stable_baselines3 import PPO
import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")

model = PPO.load("ppo_cartpole")

for episode in range(3):

    obs, info = env.reset()

    while True:

        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.02)

        if terminated or truncated:
            break

env.close()