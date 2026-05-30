import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("monitor.csv", skiprows=1)

print(df.head())

plt.figure(figsize=(10, 5))

rolling_mean = df["r"].rolling(window=50).mean()

plt.plot(rolling_mean)

plt.title("CartPole DQN Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.grid(True)

plt.savefig("learning_curve.png")

plt.show()