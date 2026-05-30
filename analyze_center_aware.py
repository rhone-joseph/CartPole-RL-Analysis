from stable_baselines3 import DQN
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

model = DQN.load("center_aware_model")

obs, info = env.reset()

print("Initial Cart Position:", obs[0])
print("Initial Pole Angle:", obs[2])

cart_positions = []
pole_angles = []

while True:

    action, _ = model.predict(obs, deterministic=True)

    cart_positions.append(obs[0])
    pole_angles.append(obs[2])

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

plt.figure(figsize=(10, 5))

plt.plot(cart_positions, label="Cart Position")
plt.plot(pole_angles, label="Pole Angle")

plt.title("Cart Position and Pole Angle During Episode")
plt.xlabel("Time Step")
plt.ylabel("Value")

plt.legend()
plt.grid(True)

plt.show()

env.close()