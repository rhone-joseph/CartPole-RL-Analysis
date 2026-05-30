from stable_baselines3 import DQN
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

model = DQN.load("cartpole_model")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()