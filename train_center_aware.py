from stable_baselines3 import DQN
from custom_cartpole import CenterAwareCartPole
import gymnasium as gym

env = gym.make("CartPole-v1")
env = CenterAwareCartPole(env)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0001,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    verbose=1
)

model.learn(total_timesteps=100000)

model.save("center_aware_model")

print("Center-Aware Training Complete!")