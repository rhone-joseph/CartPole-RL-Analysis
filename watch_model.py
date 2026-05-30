from stable_baselines3 import DQN
import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")

model = DQN.load("center_aware_model")  # or "cartpole_model"

num_episodes = 3

for episode in range(num_episodes):

    print(f"\nEpisode {episode + 1}")

    obs, info = env.reset()

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.02)

        if terminated or truncated:
            print(f"Episode {episode + 1} finished")
            time.sleep(1)  # pause before next episode
            break

env.close()