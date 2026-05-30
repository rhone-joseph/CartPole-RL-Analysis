from stable_baselines3 import DQN
import gymnasium as gym

env = gym.make("CartPole-v1")

model = DQN.load("cartpole_model")

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
    print(f"Episode {episode+1}: {total_reward}")

print("\nAverage Reward:", sum(scores)/len(scores))
print("Best Reward:", max(scores))
print("Worst Reward:", min(scores))