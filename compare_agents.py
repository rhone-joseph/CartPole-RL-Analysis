from stable_baselines3 import DQN
import gymnasium as gym

env = gym.make("CartPole-v1")

# Load trained AI
model = DQN.load("cartpole_model")

NUM_EPISODES = 100

# -------------------------
# Random Agent Evaluation
# -------------------------

random_rewards = []

for episode in range(NUM_EPISODES):

    obs, info = env.reset()
    total_reward = 0

    while True:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    random_rewards.append(total_reward)

random_avg = sum(random_rewards) / len(random_rewards)

# -------------------------
# DQN Agent Evaluation
# -------------------------

dqn_rewards = []

for episode in range(NUM_EPISODES):

    obs, info = env.reset()
    total_reward = 0

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    dqn_rewards.append(total_reward)

dqn_avg = sum(dqn_rewards) / len(dqn_rewards)

# -------------------------
# Results
# -------------------------

improvement = ((dqn_avg - random_avg) / random_avg) * 100

print("\n===== RESULTS =====\n")

print(f"Random Agent Average Reward : {random_avg:.2f}")
print(f"DQN Agent Average Reward    : {dqn_avg:.2f}")
print(f"Improvement                 : {improvement:.2f}%")

env.close()