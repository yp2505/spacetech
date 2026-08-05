import os
import numpy as np
from stable_baselines3 import PPO
from satellite_env import MultiSatelliteEnv, SingleAgentWrapper

# Dummy training progress based on your exact last output
progress_data = [
    [1, 5000, -0.770, -1.123],
    [2, 10000, -0.133, -0.059],
    [3, 15000, 0.940, 1.259],
    [4, 20000, 1.829, 1.979],
    [5, 25000, 2.232, 2.216],
    [6, 30000, 3.068, 2.257],
    [7, 35000, 3.355, 2.660],
    [8, 40000, 3.093, 2.903],
    [9, 45000, 3.001, 2.661],
    [10, 50000, 3.318, 2.651],
    [11, 55000, 3.907, 3.171],
    [12, 60000, 4.032, 3.801],
    [13, 65000, 3.688, 3.830],
    [14, 70000, 3.828, 3.882],
    [15, 75000, 3.961, 3.402],
    [16, 80000, 4.135, 3.624],
    [17, 85000, 4.299, 3.669],
    [18, 90000, 4.293, 3.425],
    [19, 95000, 4.134, 3.629],
    [20, 100000, 4.579, 3.719]
]

multi_env0 = MultiSatelliteEnv(max_steps=50)
env1 = SingleAgentWrapper(multi_env0, agent_idx=0)
env2 = SingleAgentWrapper(multi_env0, agent_idx=1)

model1 = PPO.load("ppo_satellite_1", env=env1)
model2 = PPO.load("ppo_satellite_2", env=env2)

env = MultiSatelliteEnv(max_steps=50)
obs_list, _ = env.reset()

for step in range(50):
    a1, _ = model1.predict(obs_list[0], deterministic=True)
    a2, _ = model2.predict(obs_list[1], deterministic=True)
    obs_list, _, _, truncated, _ = env.step([int(a1), int(a2)])
    if truncated:
        break

export_dir = "matlab_export"
os.makedirs(export_dir, exist_ok=True)

# 1. reward_history.csv
with open(os.path.join(export_dir, "reward_history.csv"), "w") as f:
    f.write("step,sat1_reward,sat2_reward,sat1_cumulative_reward,sat2_cumulative_reward\n")
    for s in range(len(env.reward_history[0])):
        f.write(f"{s+1},{env.reward_history[0][s]:.4f},{env.reward_history[1][s]:.4f},{env.cumulative_reward_history[0][s]:.4f},{env.cumulative_reward_history[1][s]:.4f}\n")

# 2. positions_history.csv
with open(os.path.join(export_dir, "positions_history.csv"), "w") as f:
    f.write("step,sat1_position,sat2_position,target_positions,target_types\n")
    for s in range(len(env.agent_pos_history[0])):
        t_arr = env.targets_history[s]
        t_pos = [str(idx) for idx, val in enumerate(t_arr) if val != 0]
        t_types = [str(val) for val in t_arr if val != 0]
        f.write(f"{s+1},{env.agent_pos_history[0][s]},{env.agent_pos_history[1][s]},\"{','.join(t_pos)}\",\"{','.join(t_types)}\"\n")

# 3. training_progress.csv
with open(os.path.join(export_dir, "training_progress.csv"), "w") as f:
    f.write("cycle,steps_completed,sat1_reward_mean,sat2_reward_mean\n")
    for row in progress_data:
        f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.4f}\n")
