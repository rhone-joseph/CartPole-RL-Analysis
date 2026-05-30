from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")

model = PPO.load("ppo_cartpole")

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

    print(f"Episode {episode + 1}: {total_reward}")

print("\n===== PPO RESULTS =====")
print(f"Average Reward: {sum(scores)/len(scores):.2f}")
print(f"Best Reward: {max(scores):.2f}")
print(f"Worst Reward: {min(scores):.2f}")

env.close()