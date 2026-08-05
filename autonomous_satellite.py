import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

# ── Part 1: Custom Environment Layout ─────────────────────────────────────────

class AutonomousSatelliteEnv(gym.Env):
    """
    A continuous multi-objective satellite environment for training advanced AOCS 
    (Attitude and Orbit Control Systems). 
    
    State (10-dim):
      0: x position error
      1: y position error
      2: z position error
      3: vx velocity error
      4: vy velocity error
      5: vz velocity error
      6: w1 reaction wheel speed
      7: w2 reaction wheel speed
      8: w3 reaction wheel speed
      9: battery SoC (0.0 to 1.0)
      
    Action (3-dim):
      [torque_1, torque_2, torque_3] applied to the reaction wheels, clamped [-1.0, 1.0]
    """
    
    metadata = {"render_modes": [None]}
    
    def __init__(self):
        super().__init__()
        
        # Action space: 3 continuous torques for reaction wheels
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 10 continuous states
        # Box bounds set wide enough to cover normalized operational limits
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(10,), dtype=np.float32
        )
        
        # Physics Parameters (Simplified Hill's Equations / Linearized dynamics)
        self.dt = 1.0 # time step (seconds)
        self.n = 0.00113 # Mean motion (rad/s) for LEO (~400km)
        
        # Battery Parameters
        self.base_power_draw = 0.001 # Base drain per step
        self.torque_power_cost = 0.002 # Additional drain per unit of torque
        
        # Reward Weights
        self.w_pos = 1.0
        self.w_vel = 0.5
        self.w_effort = 0.1
        self.w_battery = 0.5
        
        # Horizons
        self.max_steps = 2048
        self.current_step = 0
        
        # Internal State array
        self.state = np.zeros(10, dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize with random tracking errors
        pos_err = self.np_random.uniform(low=-1.0, high=1.0, size=(3,))
        vel_err = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        rw_speeds = np.zeros(3)
        soc = np.array([1.0]) # Full battery
        
        self.state = np.concatenate([pos_err, vel_err, rw_speeds, soc]).astype(np.float32)
        
        return self.state, {}
        
    def step(self, action):
        self.current_step += 1
        
        # Parse state
        x, y, z, vx, vy, vz, w1, w2, w3, soc = self.state
        
        # Clip action torques
        tau = np.clip(action, -1.0, 1.0)
        
        # ── Physical Dynamics Emulation ───────────────────────────────────────
        
        # 1. Update reaction wheels (simplified integration: w_new = w_old + torque * dt)
        # Note: In a real system, the torque applied by the RW to the satellite is -tau.
        # Here we just track the RW speeds.
        w1_new = w1 + tau[0] * self.dt
        w2_new = w2 + tau[1] * self.dt
        w3_new = w3 + tau[2] * self.dt
        
        # 2. Orbital Dynamics (Simplified Hill's Equations / Clohessy-Wiltshire)
        # For this demonstration, we assume the torques directly translate to attitude
        # which translates to thrust vectoring. We will use a simplified linear integrator
        # where the torques influence the velocity errors directly for the sake of the exercise.
        # In a true 6DOF sim, attitude decouples from position, requiring 2 separate loops.
        ax = tau[0] * 0.1 - 2 * self.n * vy + 3 * self.n**2 * x
        ay = tau[1] * 0.1 + 2 * self.n * vx
        az = tau[2] * 0.1 - self.n**2 * z
        
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        vz_new = vz + az * self.dt
        
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        z_new = z + vz_new * self.dt
        
        # 3. Battery Management
        torque_magnitude = np.sum(np.abs(tau))
        soc_new = soc - self.base_power_draw - (self.torque_power_cost * torque_magnitude)
        soc_new = max(0.0, min(1.0, soc_new)) # clamp to [0, 1]
        
        # Reconstruct state
        self.state = np.array([
            x_new, y_new, z_new, 
            vx_new, vy_new, vz_new, 
            w1_new, w2_new, w3_new, 
            soc_new
        ], dtype=np.float32)
        
        # ── Multi-Objective Reward Function ───────────────────────────────────
        
        pos_error = np.sqrt(x_new**2 + y_new**2 + z_new**2)
        vel_error = np.sqrt(vx_new**2 + vy_new**2 + vz_new**2)
        control_effort = torque_magnitude
        
        # Base penalties
        reward = -(self.w_pos * pos_error) - (self.w_vel * vel_error) - (self.w_effort * control_effort)
        
        # Battery bonus/penalty
        if soc_new > 0.20:
            reward += self.w_battery * 1.0
        elif soc_new < 0.15:
            reward -= 5.0 # Heavy penalty
            
        # ── Termination & Truncation ──────────────────────────────────────────
        
        terminated = False
        truncated = False
        
        if soc_new <= 0.0:
            terminated = True
            reward -= 10.0 # Catastrophic failure
            
        if pos_error > 5.0:
            terminated = True
            reward -= 10.0 # Lost in space
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self.state, float(reward), terminated, truncated, {}


# ── Part 2: Vectorization & Training Configuration ────────────────────────────

def make_env():
    """Utility function for multiprocessing"""
    def _init():
        return AutonomousSatelliteEnv()
    return _init

if __name__ == "__main__":
    print("Initializing Autonomous Satellite Engineering Pipeline...")
    
    # 1. Vectorized Environment (4 workers)
    num_envs = 4
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # 2. Research-Grade PPO Hyperparameters
    # We use a massive [256, 256, 256] network architecture as requested by the user
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,           # Per environment worker
        batch_size=64,
        n_epochs=10,
        gamma=0.99,             # Long horizon
        gae_lambda=0.95,
        ent_coef=0.01,          # Encourage exploration
        verbose=1,
        tensorboard_log="./satellite_ppo_tensorboard/",
        policy_kwargs=policy_kwargs
    )
    
    print("Starting 1,000,000 timestep training run...")
    # 3. Train the model
    # Note: total_timesteps is across all environments (e.g. 1M / 4 = 250k steps per env)
    model.learn(total_timesteps=1_000_000)
    
    # 4. Save the model
    model.save("autonomous_satellite_brain_ppo")
    print("Training complete! Model saved as autonomous_satellite_brain_ppo.zip")
