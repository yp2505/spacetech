"""
train.py
--------
Phase 4: Generalized Swarm Training script.
Trains a SINGLE generalized policy to control all N satellites.
Loads the old 2-satellite brain and pads it to support N-satellite features.
Includes evaluation, CSV export, and EWC protection.
"""

import os
import io
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from satellite_env import SingleAgentWrapper, MultiSatelliteEnv, LOCAL_DIM, GLOBAL_DIM
from ctde_policy import CTDEPolicy, load_old_weights_with_padding
from memory import EpisodicMemory
from ewc import EWC

NUM_SATELLITES = 10
EWC_LAMBDA = 5000.0
MODEL_PATH = "ppo_swarm_brain"
OLD_MODEL_PATH = "ppo_satellite_1.zip"
EWC_PATH = "ewc_fisher_swarm.pkl"
OLD_EWC_PATH = "ewc_fisher_sat1.pkl"

# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(model, memory: EpisodicMemory, num_episodes=10, ep_steps=360, render_last=True):
    collisions_per_ep  = [[] for _ in range(NUM_SATELLITES)]
    fuel_outs_per_ep   = [[] for _ in range(NUM_SATELLITES)]
    mean_reward_per_ep = [[] for _ in range(NUM_SATELLITES)]

    for ep in range(num_episodes):
        env = MultiSatelliteEnv(max_steps=ep_steps, num_satellites=NUM_SATELLITES)
        obs_raw, _ = env.reset()
        do_render = render_last and (ep == num_episodes - 1)

        for step in range(ep_steps):
            if do_render:
                env.render()

            # Predict actions for all satellites using the single Swarm Brain
            actions = []
            for i in range(NUM_SATELLITES):
                ctx = memory.get_context() if memory else np.zeros(4, dtype=np.float32)
                local_obs = np.concatenate([obs_raw[i]["local"], ctx]).astype(np.float32)
                obs_dict = {"local": local_obs, "global": obs_raw[i]["global"]}
                a, _ = model.predict(obs_dict, deterministic=True)
                actions.append(a)

            obs_raw, rewards, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

        for i in range(NUM_SATELLITES):
            collisions_per_ep[i].append(env.collisions[i])
            fuel_outs_per_ep[i].append(env.fuel_outs[i])
            mean_reward_per_ep[i].append(np.sum(env.reward_history[i]))

            # Record episode into episodic memory (only recording for agent 0 as representative)
            if i == 0:
                memory.record(
                    total_reward=np.sum(env.reward_history[i]),
                    collisions=env.collisions[i],
                    fuel_outs=env.fuel_outs[i],
                    steps=env.current_step,
                    was_eclipse=any(env.eclipse_mode),
                    was_weather=env.space_weather_active,
                )

        if do_render:
            print("Animation complete. Close the window to exit.")
            plt.ioff()
            plt.show()
            env.close()

    return collisions_per_ep, fuel_outs_per_ep, mean_reward_per_ep, env

def print_summary(label, collisions_per_ep, fuel_outs_per_ep, mean_reward_per_ep):
    W = 62
    print(f"\n{'─'*W}")
    print(f"  {label} — 10-Episode Evaluation Summary")
    print(f"{'─'*W}")
    print(f"  {'Satellite':<18} {'Collisions':>10} {'Fuel Outs':>10} {'Avg Total Rew':>15}")
    print(f"  {'─'*(W-2)}")
    for i in range(NUM_SATELLITES):
        avg_coll = np.mean(collisions_per_ep[i])
        avg_fuel = np.mean(fuel_outs_per_ep[i])
        avg_rew  = np.mean(mean_reward_per_ep[i])
        print(f"  Sat {i:<14} {avg_coll:>10.2f} {avg_fuel:>10.2f} {avg_rew:>15.2f}")
    print(f"{'─'*W}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true', help='Skip training and just evaluate')
    args = parser.parse_args()

    print("="*60)
    print(f" INITIALIZING GENERALIZED SWARM BRAIN ({NUM_SATELLITES} Satellites)")
    print("="*60)

    memory = EpisodicMemory()
    print(f"\n[Memory] {memory.stats}")

    env = SingleAgentWrapper(num_positions=360, max_steps=360, agent_idx=0, memory=memory, num_satellites=NUM_SATELLITES)

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing Swarm Brain from {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=env, custom_objects={'policy_class': CTDEPolicy})
    else:
        print("Creating fresh Swarm Brain...")
        model = PPO(
            CTDEPolicy, env,
            verbose=0,
            learning_rate=1e-4, 
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
        load_old_weights_with_padding(model, OLD_MODEL_PATH)

    env.set_other_model(model)

    if os.path.exists(OLD_EWC_PATH) and not os.path.exists(EWC_PATH):
        print("Re-computing EWC Fisher for padded model to protect old knowledge...")
        ewc = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC_PATH)
        ewc.compute_fisher(model, env, n_samples=300)
    else:
        ewc = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC_PATH)

    if not args.eval_only:
        print("\nStarting Swarm Training...")
        cycles = 100
        steps_per_cycle = 5000
        progress_data = []

        for cycle in range(1, cycles + 1):
            model.learn(total_timesteps=steps_per_cycle, reset_num_timesteps=False)

            if ewc.is_active():
                loss = ewc.apply_correction(model, n_steps=5)

            model.save(MODEL_PATH)

            buf = model.ep_info_buffer
            max_r = np.max([ep["r"] for ep in buf]) if len(buf) > 0 else 0.0

            progress_data.append([cycle, cycle * steps_per_cycle, max_r])
            print(f" Cycle {cycle}/{cycles} | Max Reward: {max_r:+.2f} | Steps: {cycle*steps_per_cycle}")

    print(f"\nModels saved → {MODEL_PATH}.zip")
    memory.save()

    print("\n" + "="*62)
    print("  EVALUATION: Trained Swarm (10 episodes, last shown live)")
    print("="*62)

    t_coll, t_fuel, t_rew, final_env = run_evaluation(model, memory, num_episodes=10, ep_steps=360, render_last=True)
    print_summary("TRAINED SWARM", t_coll, t_fuel, t_rew)

    print("\n" + "="*62)
    print("  EXPORTING DATA TO MATLAB")
    print("="*62)

    export_dir = "matlab_export"
    os.makedirs(export_dir, exist_ok=True)

    with open(os.path.join(export_dir, "reward_history.csv"), "w") as f:
        headers = ["step"] + [f"sat{i}_reward" for i in range(NUM_SATELLITES)]
        f.write(",".join(headers) + "\n")
        for s in range(len(final_env.reward_history[0])):
            row = [f"{s+1}"] + [f"{final_env.reward_history[i][s]:.4f}" for i in range(NUM_SATELLITES)]
            f.write(",".join(row) + "\n")
    print(f"  ✓ Saved {export_dir}/reward_history.csv")

    with open(os.path.join(export_dir, "positions_history.csv"), "w") as f:
        headers = ["step"] + [f"sat{i}_position" for i in range(NUM_SATELLITES)] + ["debris_positions"]
        f.write(",".join(headers) + "\n")
        for s in range(len(final_env.agent_pos_history[0])):
            d_arr = final_env.debris_history[s] if s < len(final_env.debris_history) else []
            d_pos = [str(val) for val in d_arr]
            row = [f"{s+1}"] + [f"{final_env.agent_pos_history[i][s]}" for i in range(NUM_SATELLITES)] + [f"\"{','.join(d_pos)}\""]
            f.write(",".join(row) + "\n")
    print(f"  ✓ Saved {export_dir}/positions_history.csv")

    print("  Done.")

if __name__ == "__main__":
    main()
