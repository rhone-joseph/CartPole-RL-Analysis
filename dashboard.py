import streamlit as st
import pandas as pd

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="CartPole RL Dashboard",
    page_icon="🎯",
    layout="wide"
)

# ==================================================
# DATA
# ==================================================

results = pd.DataFrame({
    "Model": [
        "DQN",
        "Reward-Shaped DQN",
        "PPO"
    ],
    "Average Reward": [
        300.85,
        217.79,
        500.00
    ],
    "Drift Behaviour": [
        "High",
        "Reduced",
        "Minimal"
    ]
})

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🎮 Navigation")

selected_model = st.sidebar.selectbox(
    "Select Model",
    [
        "DQN",
        "Reward-Shaped DQN",
        "PPO"
    ]
)

st.sidebar.markdown("---")

st.sidebar.header("📊 Project Stats")

st.sidebar.metric("Episodes Evaluated", "100")
st.sidebar.metric("Best PPO Reward", "500")
st.sidebar.metric("Worst PPO Reward", "500")
st.sidebar.metric("Environment Limit", "500 Steps")

# ==================================================
# HEADER
# ==================================================

st.title("🎯 CartPole Reinforcement Learning Dashboard")

st.markdown("""
This project investigates CartPole balancing using
three Reinforcement Learning approaches:

- Deep Q Network (DQN)
- Reward-Shaped DQN
- Proximal Policy Optimization (PPO)
""")

# ==================================================
# PROJECT OBJECTIVE
# ==================================================

st.info("""
### Project Objective

The goal of this project was not simply to maximize reward.

The project investigated why a DQN agent developed a
persistent cart-drift behaviour, explored reward-engineering
solutions to reduce drift, and compared the results against PPO.

This makes the project an investigation into policy behaviour,
not just a reinforcement learning implementation.
""")

# ==================================================
# IMPORTANT NOTE
# ==================================================

st.warning("""
### Important Interpretation

The Reward-Shaped DQN uses a modified reward function.

Its lower reward DOES NOT necessarily mean worse behaviour.

The modified reward explicitly penalizes cart drift and rewards
centre-aware behaviour.

As a result, the agent may sacrifice raw reward in exchange for
better positioning and stability.

Therefore, reward values should be interpreted together with
behaviour analysis and drift measurements.
""")

# ==================================================
# KPI CARDS
# ==================================================

st.header("🏆 Final Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "DQN",
        "300.85",
        delta="-199.15 vs PPO"
    )

with col2:
    st.metric(
        "Reward-Shaped DQN",
        "217.79",
        delta="Reduced Drift"
    )

with col3:
    st.metric(
        "PPO",
        "500.00",
        delta="Perfect Score"
    )

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Performance",
    "📈 Analysis",
    "📋 Project Summary"
])

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("Algorithm Comparison")

    st.dataframe(
        results,
        use_container_width=True
    )

    st.subheader("Average Reward Comparison")

    st.bar_chart(
        results.set_index("Model")["Average Reward"]
    )

    st.success("""
### PPO Evaluation Results

Average Reward : 500

Best Reward : 500

Worst Reward : 500

Solved Episodes : 100 / 100

The PPO agent consistently survived until the
500-step CartPole environment limit.
""")

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.header("Policy Analysis")

    st.error("""
### Key Discovery

The original DQN learned a drifting strategy.

Although it successfully balanced the pole,
it gradually moved toward one side of the track.

Eventually the cart reached the environment boundary
and failed.

Reward engineering reduced the drift.

PPO completely solved the issue and learned a stable policy.
""")

    st.subheader("PPO Analysis")

    st.image(
        "ppo_analysis.png",
        caption="PPO Cart Position and Pole Angle",
        use_container_width=True
    )

    st.subheader("DQN Drift Analysis")

    st.image(
        "dqn_drift.png",
        caption="DQN Drift Behaviour",
        use_container_width=True
    )

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.header("Project Timeline")

    st.markdown("""
    ✅ Built DQN Agent

    ✅ Evaluated DQN Performance

    ✅ Identified Persistent Drift Behaviour

    ✅ Analysed Cart Position Trends

    ✅ Designed Reward-Shaping Strategy

    ✅ Reduced Drift Through Reward Engineering

    ✅ Compared Multiple RL Algorithms

    ✅ Trained PPO Agent

    ✅ Achieved Perfect PPO Performance

    ✅ Built Interactive Dashboard
    """)

    st.header("Key Findings")

    st.markdown("""
- DQN successfully balanced the pole but developed drift.
- Drift caused the cart to gradually approach track boundaries.
- Reward engineering reduced drift and improved centre awareness.
- Lower reward in Reward-Shaped DQN was expected because drift penalties were added.
- PPO achieved perfect performance and eliminated observable drift.
- Algorithm selection had a larger impact than reward engineering alone.
""")

# ==================================================
# MODEL EXPLORER
# ==================================================

st.markdown("---")

st.header(f"🔍 Selected Model: {selected_model}")

if selected_model == "DQN":

    st.info("""
### DQN

Average Reward: 300.85

Characteristics:

- Learned balancing behaviour
- Persistent cart drift
- Eventually reaches track boundary
- Moderate overall performance

Main Observation:

The agent discovered a strategy that balanced
the pole but did not maintain centre position.
""")

elif selected_model == "Reward-Shaped DQN":

    st.info("""
### Reward-Shaped DQN

Average Reward: 217.79

Characteristics:

- Reduced drift
- Better centre awareness
- Explicit drift penalties
- Improved behavioural stability

Important:

The lower reward is expected because the reward
function intentionally penalizes drift.

Behaviourally, this model often performed better
than the original DQN despite receiving lower reward.
""")

elif selected_model == "PPO":

    st.success("""
### PPO

Average Reward: 500

Characteristics:

- Perfect performance
- Stable policy
- No significant drift
- Solved CartPole consistently

Result:

100 / 100 successful evaluation episodes.
""")

# ==================================================
# ENVIRONMENT
# ==================================================

st.markdown("---")

st.header("🖥 Environment Information")

st.markdown("""
### CartPole-v1

State Variables:

1. Cart Position
2. Cart Velocity
3. Pole Angle
4. Pole Angular Velocity

Actions:

- 0 → Move Left
- 1 → Move Right

Success Condition:

Keep the pole balanced for 500 time steps.
""")

# ==================================================
# TECH STACK
# ==================================================

st.markdown("---")

st.header("🛠 Technology Stack")

st.code("""
Python
Gymnasium
Stable-Baselines3
PyTorch
NumPy
Pandas
Matplotlib
Streamlit
""")

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.success(
    "🏅 Final Conclusion: PPO achieved the best combination of stability, control quality and reward performance."
)