"""
Quick evaluation script — loads the already-trained models and runs
the 10-episode evaluation + visualization without retraining.
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from satellite_env import SingleAgentWrapper
from memory import EpisodicMemory
from train import run_evaluation, print_summary


def main():
    print("\nLoading saved models…")
    memory = EpisodicMemory()
    env1 = SingleAgentWrapper(num_positions=360, max_steps=360, agent_idx=0, memory=memory)
    env2 = SingleAgentWrapper(num_positions=360, max_steps=360, agent_idx=1, memory=memory)

    try:
        model1 = PPO.load("ppo_satellite_1", env=env1)
        model2 = PPO.load("ppo_satellite_2", env=env2)
        print("  ppo_satellite_1.zip and ppo_satellite_2.zip loaded.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    print("\n" + "="*62)
    print("  EVALUATION: 10 episodes")
    print("="*62)

    t_coll, t_fuel, t_rew, _ = run_evaluation(
        model1, model2, memory, num_episodes=10, ep_steps=360, render_last=False
    )
    print_summary("TRAINED EVALUATION", t_coll, t_fuel, t_rew)


if __name__ == "__main__":
    main()
