import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from agentic_env import AgenticEnv
from ctde_policy import CTDEPolicy # For loading the motor brain

def main():
    print("Loading Frozen Motor Brain...")
    # The Motor Brain will be saved as 'ppo_satellite_1' when the current background training finishes
    try:
        motor_brain = PPO.load("ppo_satellite_1")
    except Exception as e:
        print("Motor brain not found yet. Make sure the background training has finished!")
        return
        
    print("Initializing Agentic Environment...")
    env = AgenticEnv(motor_brain=motor_brain, num_positions=360, max_steps=360)
    
    # Check if the custom env follows the gym interface
    # check_env(env)
    
    print("Creating High-Level Agentic Brain...")
    # Agentic Brain uses a standard MLP policy, since it doesn't need CTDE (only Agent B is learning)
    agentic_brain = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[128, 128]))
    
    print("Training Agentic Brain to perform Rendezvous via Hierarchical RL...")
    agentic_brain.learn(total_timesteps=50000)
    
    agentic_brain.save("ppo_agentic_brain")
    print("Agentic Brain saved successfully!")

if __name__ == "__main__":
    main()
