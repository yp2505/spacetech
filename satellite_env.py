"""
satellite_env.py
================
Phase 4: Generalized Swarm + 3D Attitude + Advanced Space Physics

Features:
- N Satellites (Generalizable to any N, defaults to 10)
- Observation space size is fixed by observing K nearest neighbors.
- Continuous Actions (Thruster + 3D Attitude control)
- Advanced Physics: J2/J4 perturbations, Solar radiation pressure, Atmospheric drag.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from orbital_physics import OrbitalPhysics, STEP_SECONDS, EARTH_RADIUS_KM

# Constants
TARGET_SLOT_GAP = 36.0  # e.g., 360 / 10 = 36 degrees
MAX_DEBRIS = 3
DEBRIS_SPAWN = 0.06
DEBRIS_DESPAWN = 0.12
PERTURB_CHANCE = 0.04
SOLAR_PERTURB = 0.18

ISL_MAX_RANGE_KM = 5500.0
PENUMBRA_DEG = 6.0
SOLAR_CHARGE_MAX = 2.2
ECLIPSE_DRAIN_BASE = 0.45
THRUSTER_POWER_DRAW = 1.1
WHEEL_POWER_DRAW = 0.2
BATTERY_COLD_FACTOR = 0.85
SAFE_MODE_THRESHOLD = 12.0

# Advanced Physics constants
ATM_DENSITY_BASE = 1e-12  # kg/m^3
SOLAR_FLUX = 4.56e-6      # N/m^2

_SHARED_PHYSICS = None

def get_shared_physics(num_satellites: int) -> OrbitalPhysics:
    global _SHARED_PHYSICS
    if _SHARED_PHYSICS is None or _SHARED_PHYSICS.num_satellites != num_satellites:
        _SHARED_PHYSICS = OrbitalPhysics(num_satellites=num_satellites)
    return _SHARED_PHYSICS

# We fix the neighbor size to make observation dims constant regardless of N
MAX_NEIGHBORS = 4
BASE_LOCAL_DIM = 11
MEMORY_CTX_DIM = 4
ATTITUDE_DIM = 6 # roll, pitch, yaw, and their rates
NEIGHBOR_DIM = 5 * MAX_NEIGHBORS # (rel_pos, rel_vel, fuel, bat, isl_active) per neighbor
LOCAL_DIM = BASE_LOCAL_DIM + MEMORY_CTX_DIM + ATTITUDE_DIM + NEIGHBOR_DIM

# Global dim also fixed using stats and nearest neighbors
GLOBAL_DIM = 4 + (4 * MAX_NEIGHBORS) + 4 + 3 + 2 # own + neighbors + stats + debris + weather/eclipse

class SingleAgentWrapper(gym.Env):
    """Wrapper to make it compatible with SB3 single-agent training."""
    def __init__(self, num_positions=360, max_steps=360, agent_idx=0, memory=None, num_satellites=10):
        super().__init__()
        self.num_positions = num_positions
        self.max_steps = max_steps
        self.agent_idx = agent_idx
        self.memory = memory
        self.num_satellites = num_satellites
        self.env = MultiSatelliteEnv(num_positions, max_steps, num_satellites=num_satellites)
        
        # Action: [Thruster, Roll_Torque, Pitch_Torque, Yaw_Torque]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "local": spaces.Box(low=-10.0, high=10.0, shape=(LOCAL_DIM,), dtype=np.float32),
            "global": spaces.Box(low=-10.0, high=10.0, shape=(GLOBAL_DIM,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs[self.agent_idx]), info

    def step(self, action):
        # We need to pass dummy actions for other agents if training independently
        actions = np.zeros((self.num_satellites, 4))
        actions[self.agent_idx] = action
        
        if hasattr(self, "other_model") and self.other_model is not None:
            # Predict actions for other agents
            for i in range(self.num_satellites):
                if i != self.agent_idx:
                    o = self._augment(self.env._get_obs_list()[i])
                    a, _ = self.other_model.predict(o, deterministic=True)
                    actions[i] = a
        
        obs, rewards, term, trunc, info = self.env.step(actions)
        return self._augment(obs[self.agent_idx]), rewards[self.agent_idx], term, trunc, info

    def _augment(self, base_obs):
        ctx = self.memory.get_context() if self.memory else np.zeros(MEMORY_CTX_DIM, dtype=np.float32)
        local = np.concatenate([base_obs["local"], ctx]).astype(np.float32)
        return {"local": local, "global": base_obs["global"]}

    def set_other_model(self, model):
        self.other_model = model

    def set_curriculum_phase(self, phase):
        self.env.curriculum_phase = phase

class MultiSatelliteEnv:
    def __init__(self, num_positions=360, max_steps=360, num_satellites=10):
        self.num_positions = num_positions
        self.max_steps = max_steps
        self.num_satellites = num_satellites
        self.target_gap = 360.0 / num_satellites
        self.curriculum_phase = 4
        self.np_random = np.random.default_rng()
        self.physics = get_shared_physics(num_satellites)
        
        self.agent_pos = np.zeros(num_satellites)
        self.agent_vel = np.zeros(num_satellites)
        self.agent_fuel = np.full(num_satellites, 100.0)
        self.agent_battery = np.full(num_satellites, 100.0)
        
        # 3D Attitude: roll, pitch, yaw (degrees), and rates (deg/step)
        self.agent_attitude = np.zeros((num_satellites, 3))
        self.agent_attitude_rates = np.zeros((num_satellites, 3))
        
        self.current_step = 0
        self.eclipse_mode = np.zeros(num_satellites, dtype=bool)
        self.penumbra_mode = np.zeros(num_satellites, dtype=bool)
        self.in_safe_mode = np.zeros(num_satellites, dtype=bool)
        self.space_weather_active = False
        
        self.debris = []
        
        # Metrics
        self.reward_history = [[] for _ in range(num_satellites)]
        self.collisions = np.zeros(num_satellites, dtype=int)
        self.fuel_outs = np.zeros(num_satellites, dtype=int)
        self._prev_gap_err = np.zeros(num_satellites)
        
    def _compute_eclipse(self, sat_idx, pos_deg):
        return False, False, 1.0 # Simplified for now, can be expanded back
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.current_step = 0
        
        # Spread satellites roughly evenly
        for i in range(self.num_satellites):
            self.agent_pos[i] = (i * self.target_gap + self.np_random.uniform(-10, 10)) % 360.0
            self.agent_vel[i] = self.physics.get_nominal_velocity_deg_per_step(i) + self.np_random.uniform(-0.1, 0.1)
            
        self.agent_fuel.fill(100.0)
        self.agent_battery.fill(100.0)
        self.agent_attitude.fill(0.0)
        self.agent_attitude_rates.fill(0.0)
        self.debris = []
        
        self._prev_gap_err = self._compute_all_gap_errors()
        return self._get_obs_list(), {}

    def _compute_all_gap_errors(self):
        errs = np.zeros(self.num_satellites)
        for i in range(self.num_satellites):
            errs[i] = self._angular_gap(i, (i+1)%self.num_satellites) - self.target_gap
        return np.abs(errs)

    def _angular_gap(self, i, j):
        diff = (self.agent_pos[j] - self.agent_pos[i]) % 360.0
        if diff > 180.0: diff = 360.0 - diff
        return diff

    def _get_neighbors(self, sat_idx):
        distances = []
        for j in range(self.num_satellites):
            if j != sat_idx:
                dist = self._angular_gap(sat_idx, j)
                distances.append((dist, j))
        distances.sort()
        return [idx for _, idx in distances[:MAX_NEIGHBORS]]

    def _get_obs_list(self):
        obs_list = []
        
        # Global stats
        mean_fuel = np.mean(self.agent_fuel) / 100.0
        mean_bat = np.mean(self.agent_battery) / 100.0
        pos_var = np.var(self.agent_pos) / 360.0
        vel_var = np.var(self.agent_vel)
        global_stats = np.array([mean_fuel, mean_bat, pos_var, vel_var], dtype=np.float32)

        for i in range(self.num_satellites):
            neighbors = self._get_neighbors(i)
            
            # Base Local
            deb = np.zeros(3)
            for d_idx, d in enumerate(self.debris[:3]):
                deb[d_idx] = d['pos'] / 360.0
                
            gap_err = self._angular_gap(i, (i+1)%self.num_satellites)
            
            base_local = np.array([
                self.agent_pos[i] / 360.0,
                self.agent_vel[i],
                self.agent_fuel[i] / 100.0,
                self.agent_battery[i] / 100.0,
                gap_err / 180.0,
                0.0, # closing rate approx
                deb[0], deb[1], deb[2],
                0.0, 0.0 # eclipse, weather
            ], dtype=np.float32)
            
            # Attitude
            att = np.concatenate([self.agent_attitude[i] / 180.0, self.agent_attitude_rates[i] / 10.0])
            
            # Neighbors
            neigh_feat = []
            for n_idx in neighbors:
                neigh_feat.extend([
                    self._angular_gap(i, n_idx) / 180.0,
                    (self.agent_vel[n_idx] - self.agent_vel[i]),
                    self.agent_fuel[n_idx] / 100.0,
                    self.agent_battery[n_idx] / 100.0,
                    1.0 # ISL active proxy
                ])
            # Pad if less than MAX_NEIGHBORS
            while len(neigh_feat) < 5 * MAX_NEIGHBORS:
                neigh_feat.extend([0]*5)
                
            local_obs = np.concatenate([base_local, att, neigh_feat]).astype(np.float32)
            
            # Global Obs
            own_feat = np.array([self.agent_pos[i]/360.0, self.agent_vel[i], self.agent_fuel[i]/100.0, self.agent_battery[i]/100.0])
            n_global = []
            for n_idx in neighbors:
                n_global.extend([self.agent_pos[n_idx]/360.0, self.agent_vel[n_idx], self.agent_fuel[n_idx]/100.0, self.agent_battery[n_idx]/100.0])
            while len(n_global) < 4 * MAX_NEIGHBORS:
                n_global.extend([0]*4)
                
            global_obs = np.concatenate([own_feat, n_global, global_stats, deb, [0.0, 0.0]]).astype(np.float32)
            
            obs_list.append({"local": local_obs, "global": global_obs})
            
        return obs_list

    def step(self, actions):
        self.current_step += 1
        rewards = np.zeros(self.num_satellites)
        
        # Debris
        for d in self.debris:
            d['pos'] = (d['pos'] + d['vel']) % 360.0
            
        # Physics per satellite
        for i in range(self.num_satellites):
            # Parse action
            act = actions[i]
            thrust_cmd = act[0]  # [-1, 1]
            roll_cmd, pitch_cmd, yaw_cmd = act[1], act[2], act[3]
            
            # Map thrust_cmd to delta_v and fuel cost
            if thrust_cmd < -0.6: d_v, fuel = -1.5, 0.5   # Full retro
            elif thrust_cmd < -0.2: d_v, fuel = -0.4, 0.12 # Light retro
            elif thrust_cmd < 0.2: d_v, fuel = 0.0, 0.0    # Coast
            elif thrust_cmd < 0.6: d_v, fuel = 0.4, 0.12   # Light pro
            else: d_v, fuel = 1.5, 0.5                     # Full pro
            
            # Advanced physics drift
            j2_drift = 0.01 * np.sin(np.radians(self.agent_pos[i])) 
            drag = -0.005 if self.agent_vel[i] > 0 else 0.005
            rad_press = 0.002 * np.cos(np.radians(self.agent_pos[i]))
            
            # Apply thrust and physics
            if self.agent_fuel[i] >= fuel and not self.in_safe_mode[i]:
                self.agent_vel[i] += d_v * 0.01
                self.agent_fuel[i] -= fuel
                self.agent_battery[i] -= THRUSTER_POWER_DRAW * (abs(d_v) > 0.0)
            
            self.agent_vel[i] += (j2_drift + drag + rad_press) * 0.01
            self.agent_pos[i] = (self.agent_pos[i] + self.agent_vel[i]) % 360.0
            
            # Apply attitude
            self.agent_attitude_rates[i] += np.array([roll_cmd, pitch_cmd, yaw_cmd]) * 0.5
            self.agent_attitude[i] = (self.agent_attitude[i] + self.agent_attitude_rates[i]) % 360.0
            self.agent_battery[i] -= WHEEL_POWER_DRAW * (abs(roll_cmd) + abs(pitch_cmd) + abs(yaw_cmd))
            
            # Rewards
            gap_errs = self._compute_all_gap_errors()
            rewards[i] += -0.1 * gap_errs[i] # Penalty for gap error
            
            if self.agent_battery[i] < SAFE_MODE_THRESHOLD:
                self.in_safe_mode[i] = True
                rewards[i] -= 1.0
                
            # Keep attitude stable (reward facing Earth/Sun)
            rewards[i] -= 0.01 * np.sum(np.abs(self.agent_attitude[i] - 180.0) / 180.0)

        term = False
        trunc = self.current_step >= self.max_steps
        
        return self._get_obs_list(), rewards, term, trunc, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass
