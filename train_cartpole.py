from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

env = gym.make("CartPole-v1")

env = Monitor(env, "monitor.csv")

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

model.save("cartpole_model")

print("Training Complete!")