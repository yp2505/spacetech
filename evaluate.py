"""
Quick evaluation script — loads the already-trained models and runs
the 10-episode evaluation + visualization without retraining.
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from satellite_env import MultiSatelliteEnv, SingleAgentWrapper
from train import run_evaluation, print_summary


def main():
    print("\nLoading saved models…")
    env1 = SingleAgentWrapper(num_positions=20, max_steps=50, agent_idx=0)
    env2 = SingleAgentWrapper(num_positions=20, max_steps=50, agent_idx=1)

    try:
        model1 = PPO.load("ppo_satellite_1", env=env1)
        model2 = PPO.load("ppo_satellite_2", env=env2)
        print("  ppo_satellite_1.zip and ppo_satellite_2.zip loaded.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    print("\n" + "="*62)
    print("  EVALUATION: 10 episodes, last episode shown live")
    print("="*62)

    t_coll, t_fuel, t_rew, _ = run_evaluation(
        model1, model2, num_episodes=10, ep_steps=50, render_last=True
    )
    print_summary("TRAINED (100k co-training steps)", t_coll, t_fuel, t_rew)


if __name__ == "__main__":
    main()
