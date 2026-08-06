"""
train_swarm_colab.py
--------------------
Phase 4: Generalized Swarm Training script for Google Colab.
Trains a SINGLE generalized policy to control all N satellites.
Loads the old 2-satellite brain and pads it to support N-satellite features.
"""

import os
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from satellite_env import SingleAgentWrapper, LOCAL_DIM, GLOBAL_DIM
from ctde_policy import CTDEPolicy, load_old_weights_with_padding
from memory import EpisodicMemory
from ewc import EWC

NUM_SATELLITES = 10
EWC_LAMBDA = 5000.0
MODEL_PATH = "ppo_swarm_brain"
OLD_MODEL_PATH = "ppo_satellite_1.zip"
EWC_PATH = "ewc_fisher_swarm.pkl"
OLD_EWC_PATH = "ewc_fisher_sat1.pkl"

def main():
    print("="*60)
    print(f" INITIALIZING GENERALIZED SWARM BRAIN ({NUM_SATELLITES} Satellites)")
    print("="*60)
    
    memory = EpisodicMemory()
    
    # We create a single env wrapper that trains on agent 0, but during step() 
    # it uses the same model to predict for all other agents!
    env = SingleAgentWrapper(num_positions=360, max_steps=360, agent_idx=0, memory=memory, num_satellites=NUM_SATELLITES)
    
    # 1. Create or Load Swarm Model
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing Swarm Brain from {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=env, custom_objects={'policy_class': CTDEPolicy})
    else:
        print("Creating fresh Swarm Brain...")
        model = PPO(
            CTDEPolicy, env,
            verbose=0,
            learning_rate=1e-4, # lower LR since we are fine-tuning
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
        # Pad and load old weights
        load_old_weights_with_padding(model, OLD_MODEL_PATH)

    # Make the environment use this model for ALL other satellites in the swarm
    env.set_other_model(model)
    
    # 2. EWC Setup
    # Load old EWC fisher to protect old weights
    if os.path.exists(OLD_EWC_PATH) and not os.path.exists(EWC_PATH):
        # We need to adapt the old EWC tensors just like we adapted the model!
        # EWC requires matching shapes. But our old model shapes changed for the first layer.
        # We will re-compute Fisher for the padded model!
        print("Re-computing EWC Fisher for padded model to protect old knowledge...")
        ewc = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC_PATH)
        ewc.compute_fisher(model, env, n_samples=300)
    else:
        ewc = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC_PATH)
    
    # 3. Training Loop
    print("\nStarting Swarm Training...")
    cycles = 50
    steps_per_cycle = 5000
    
    for cycle in range(1, cycles + 1):
        # Train the single swarm brain
        model.learn(total_timesteps=steps_per_cycle, reset_num_timesteps=False)
        
        # Apply EWC Penalty to prevent forgetting old 2-satellite behavior
        if ewc.is_active():
            loss = ewc.apply_correction(model, n_steps=5)
            
        model.save(MODEL_PATH)
        
        # Extract max episode reward
        buf = model.ep_info_buffer
        max_r = np.max([ep["r"] for ep in buf]) if len(buf) > 0 else 0.0
        
        print(f" Cycle {cycle}/{cycles} | Max Reward: {max_r:+.2f} | Steps: {cycle*steps_per_cycle}")
        
    print(f"\nSwarm training complete! Saved to {MODEL_PATH}.zip")

if __name__ == "__main__":
    main()
