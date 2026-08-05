import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from satellite_env import MultiSatelliteEnv, LOCAL_DIM, TARGET_SLOT_GAP

class AgenticEnv(gym.Env):
    """
    Hierarchical RL Environment.
    This environment trains the HIGH-LEVEL 'Agentic Brain'.
    The Agentic Brain does NOT control thrusters. It outputs a Target Coordinate.
    The LOW-LEVEL frozen 'Motor Brain' executes the thrusters to reach that coordinate.
    """
    def __init__(self, motor_brain, num_positions=360, max_steps=360):
        super().__init__()
        self.base_env = MultiSatelliteEnv(num_positions=num_positions, max_steps=max_steps)
        self.motor_brain = motor_brain
        
        # Agentic Brain Action: Choose a target offset (0 to 359 degrees) for the Motor Brain to chase
        self.action_space = spaces.Discrete(num_positions)
        
        # Agentic Brain Observation: Needs to see Agent B's status and Agent A's actual position
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = max_steps // 10 # High level brain thinks 10x slower
        
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.base_env.reset(seed=seed)
        
        # In Rendezvous, Agent A is disabled and drifting. We place it at a random spot.
        self.base_env.agent_pos[0] = self.base_env.np_random.integers(0, self.base_env.num_positions)
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        # The Agentic Brain only cares about relative distance to target, fuel, battery, etc.
        pos_b = self.base_env.agent_pos[1] / self.base_env.num_positions
        pos_a = self.base_env.agent_pos[0] / self.base_env.num_positions
        fuel_b = self.base_env.agent_fuel[1] / 100.0
        bat_b = self.base_env.agent_battery[1] / 100.0
        
        obs = np.array([
            pos_b, pos_a, fuel_b, bat_b, 
            0.0, 0.0, 0.0, 0.0 # Padding for future radar additions
        ], dtype=np.float32)
        return obs
        
    def step(self, action):
        self.current_step += 1
        
        # Action is the target coordinate the Agentic Brain wants to reach
        target_coord = int(action)
        
        total_reward = 0
        
        # Run 10 sub-steps using the frozen Motor Brain
        for _ in range(10):
            # We SPOOF the radar for the Motor Brain!
            # The Motor Brain thinks it needs to be 90 degrees away from its partner.
            # So, to make it go to `target_coord`, we tell it the partner is at `target_coord - 90`!
            spoofed_partner_pos = (target_coord - TARGET_SLOT_GAP) % self.base_env.num_positions
            
            # Get real base obs for Agent B
            base_obs = self.base_env._get_base_local_obs(1)
            
            # Inject the spoofed position into Agent B's radar
            # Assuming radar gap is the first feature (forward gap)
            fwd_gap = spoofed_partner_pos - self.base_env.agent_pos[1]
            if fwd_gap < 0: fwd_gap += self.base_env.num_positions
            base_obs[0] = fwd_gap / self.base_env.num_positions
            
            # Build the 12-dim local obs required by the motor brain
            motor_obs = np.concatenate([base_obs, np.zeros(4, dtype=np.float32)]).astype(np.float32)
            
            # Ask the frozen Motor Brain for the thruster action
            # We don't train it, we just ask for predictions!
            with torch.no_grad():
                motor_action, _ = self.motor_brain.predict(
                    {"local": motor_obs, "global": np.zeros(11, dtype=np.float32)}, 
                    deterministic=True
                )
            
            # Agent A is disabled (action 1 = Stay)
            real_actions = [1, int(motor_action)]
            
            # Step the underlying physics engine
            self.base_env.step(real_actions)
            
            # Calculate Rendezvous Reward for Agentic Brain
            dist_to_target = abs(self.base_env.agent_pos[1] - self.base_env.agent_pos[0])
            circular_dist = min(dist_to_target, self.base_env.num_positions - dist_to_target)
            
            if circular_dist <= 5:
                total_reward += 1.0 # Hovering close to the disabled satellite!
            else:
                total_reward -= (circular_dist * 0.01)
                
            if self.base_env.collision_flags[1]:
                total_reward -= 5.0 # Don't crash into it!
                
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), total_reward, terminated, truncated, {}
