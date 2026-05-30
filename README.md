# CartPole Reinforcement Learning Analysis

## Overview

This project investigates Reinforcement Learning approaches for solving the CartPole-v1 control problem.

The project compares:

* Deep Q Network (DQN)
* Reward-Shaped DQN
* Proximal Policy Optimization (PPO)

The primary objective was not only to maximize reward but also to analyze policy behaviour, investigate positional drift, and evaluate the impact of reward engineering.

---

## Technologies Used

* Python
* Gymnasium
* Stable-Baselines3
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Streamlit

---

## Project Workflow

### 1. DQN Agent

A Deep Q Network (DQN) agent was trained on the CartPole-v1 environment.

Observation:

* Successfully balanced the pole
* Developed significant positional drift
* Gradually moved toward track boundaries

---

### 2. Reward Engineering

Several custom reward functions were designed to reduce drift.

Experiments included:

* Position penalties
* Quadratic penalties
* Boundary penalties
* Center-aware rewards

Observation:

* Drift was reduced
* Behaviour became more stable
* Raw reward decreased due to added penalties

---

### 3. PPO Agent

A PPO agent was trained and evaluated.

Results:

* Average Reward: 500
* Best Reward: 500
* Worst Reward: 500
* Solved 100/100 evaluation episodes

Observation:

* No significant drift
* Stable balancing behaviour
* Consistently reached the environment time limit

---

## Key Findings

* DQN learned an unintended drifting strategy.
* Reward engineering reduced drift but did not completely eliminate it.
* PPO achieved perfect performance and stable control.
* Algorithm selection had a larger impact than reward engineering alone.

---

## Dashboard

A Streamlit dashboard was developed to visualize:

* Model performance
* Drift analysis
* PPO evaluation results
* Project findings
* Reinforcement learning experiments

---

## Results Summary

| Model             | Average Reward | Drift Behaviour |
| ----------------- | -------------- | --------------- |
| DQN               | 300.85         | High            |
| Reward-Shaped DQN | 217.79         | Reduced         |
| PPO               | 500.00         | Minimal         |

---

## Future Improvements

* Deploy dashboard online
* Compare additional RL algorithms
* Extend experiments to more complex environments
* Explore continuous-action control problems

---

## Author

Rhone Thomas Joseph

BTech Artificial Intelligence and Data Science
