from stable_baselines3 import DQN
import gymnasium as gym

env = gym.make("CartPole-v1")

model = DQN.load("center_aware_model")

obs, info = env.reset()

left = 0
right = 0
velocities = []

while True:

    velocities.append(abs(obs[1]))

    action, _ = model.predict(obs, deterministic=True)

    if action == 0:
        left += 1
    else:
        right += 1

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

print("Left actions :", left)
print("Right actions:", right)
print("Average Velocity:", sum(velocities)/len(velocities))
print("Max Velocity:", max(velocities))