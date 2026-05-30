from stable_baselines3 import DQN
from custom_cartpole import CenterAwareCartPole
import gymnasium as gym

env = gym.make("CartPole-v1")
env = CenterAwareCartPole(env)

model = DQN.load("center_aware_model")

scores = []

for episode in range(100):

    obs, info = env.reset()
    total_reward = 0

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    scores.append(total_reward)

    print(f"Episode {episode + 1}: {total_reward:.2f}")

print("\n===== RESULTS =====")
print(f"Average Reward: {sum(scores)/len(scores):.2f}")
print(f"Best Reward: {max(scores):.2f}")
print(f"Worst Reward: {min(scores):.2f}")

env.close()