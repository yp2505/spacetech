"""
satellite_env.py
================
Phase A–F: Config-driven Multi-Orbit, Multi-Mission RL Environment.

Phase A: Single-orbit, slot-keeping, N satellites (done)
Phase B: Multi-orbit (LEO/MEO/GEO) + config-driven parameters (done)
Phase C: Multi-mission objectives — Earth Obs, Comms, Science (done)
Phase D: Ground station blackout scheduling + data-buffer downlink (done)
Phase E: Fault injection — wheel saturation, thruster degradation,
         sensor noise + autonomous recovery rewards (done)
Phase F: Thermal dynamics + multi-plane Walker Delta constellations (done)

Additional features:
- ISL encrypted data relay with rolling-XOR + SHA-256 key derivation
- ISL mesh network with gossip-based experience sharing
- Hierarchical commander goals (3D vector) for high-level task conditioning
- Orbital maneuvering: Hohmann transfer, debris avoidance, deorbit
- Delta-V budget management per satellite
- Keplerian debris tracking with collision penalties
- Space weather effects on thruster/solar performance

Observation dims (must match ctde_policy.py + fsw/ai_brain/adapter.py):
  LOCAL_DIM_BASE = 44   (before memory context: 11 base + 6 attitude + 20 neighbor + 6 config + 1 recovery)
  LOCAL_DIM      = 48   (after 4-dim memory context appended by SingleAgentWrapper)
  GLOBAL_DIM     = 63   (6 own + 40 neighbor + 6 global_stats + 3 debris + 2 weather/eclipse + 6 config)

Action space: 8D continuous [thrust, roll, pitch, yaw, relay, hohmann, avoidance, deorbit]
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List, Dict, Any
from collections import deque
import logging
import json
import time

try:
    from astropy import constants as const
    from astropy import units as u
    GM_EARTH = const.GM_earth.to(u.km**3 / u.s**2).value
    EARTH_RADIUS_KM = const.R_earth.to(u.km).value
except ImportError:
    GM_EARTH = 398600.4418
    EARTH_RADIUS_KM = 6378.1

from simulation.sat_config import SatelliteConfig, PRESETS, MissionType
from simulation.orbital_physics import (
    get_walker_delta_params, anomaly_to_ecef, check_ground_station_los, 
    GROUND_STATIONS, latlon_to_ecef
)
# ISL mesh network
from fsw.hal.isl_mesh import ISLMeshNetwork, ISLPacketCrypto, PacketType, ISLPacket

# ─────────────────────────────────────────────────────────────────────────────
#  Fixed environment constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_DEBRIS           = 3
DEBRIS_SPAWN         = 0.06
DEBRIS_DESPAWN       = 0.12
SAFE_MODE_THRESHOLD  = 12.0     # % battery below which safe mode triggers
THRUSTER_POWER_DRAW  = 1.1      # % battery consumed per full-thrust step
WHEEL_POWER_DRAW     = 0.2      # % battery consumed per attitude step

# ── Observation space dimension constants ─────────────────────────────────────
MAX_NEIGHBORS    = 4

# LOCAL obs breakdown (44 dims total before memory context):
#   base_local: 11 dims (pos, altitude, eclipse_frac, gs_los, debris, delta_v, temp, battery, commander_goal[3])
#   attitude: 6 dims (roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate)
#   neighbors: 5 × 4 = 20 dims (rel_pos, rel_vel, fuel, battery, isl_active)
#   config: 6 dims (orbit_type, thruster_type, mission_type, mass, solar_area, battery)
#   recovery: 1 dim (in_recovery flag)
#   = 11 + 6 + 20 + 6 + 1 = 44
BASE_LOCAL_DIM   = 11
ATTITUDE_DIM     = 6
NEIGHBOR_DIM     = 5 * MAX_NEIGHBORS   # 20
CONFIG_DIM       = 6
RECOVERY_DIM     = 1   # binary flag: AI is in a recovery manoeuvre
MEMORY_CTX_DIM   = 4

LOCAL_DIM_BASE   = BASE_LOCAL_DIM + ATTITUDE_DIM + NEIGHBOR_DIM + CONFIG_DIM + RECOVERY_DIM  # 44
LOCAL_DIM        = LOCAL_DIM_BASE + MEMORY_CTX_DIM                                            # 48

# GLOBAL obs breakdown (63 dims):
#   own: 6 dims (pos, closing_rate, fuel, battery, temp, data)
#   neighbors: 4 × 10 = 40 dims (pos, closing, fuel, battery, temp, data, task_queue, reward, link_quality, valid_decryption)
#   global_stats: 6 dims (mean_fuel, mean_battery, pos_var, vel_var, mean_temp, mean_data)
#   debris: 3 dims (normalized true anomalies)
#   weather/eclipse: 2 dims (space_weather_active, mean_eclipse_fraction)
#   config: 6 dims
#   = 6 + 40 + 6 + 3 + 2 + 6 = 63
GLOBAL_DIM       = 6 + (10 * MAX_NEIGHBORS) + 6 + 3 + 2 + CONFIG_DIM   # 63


# ─────────────────────────────────────────────────────────────────────────────
#  SingleAgentWrapper (SB3-compatible gym.Env)
# ─────────────────────────────────────────────────────────────────────────────
class SingleAgentWrapper(gym.Env):
    """
    Wraps MultiSatelliteEnv for SB3 training.
    Each satellite is trained in turn against a frozen copy of itself (self-play).
    """

    def __init__(
        self,
        config:    SatelliteConfig = None,
        max_steps: int = 360,
        agent_idx: int = 0,
        memory     = None,
    ):
        super().__init__()
        self.config    = config or PRESETS["starlink_leo"]
        self.max_steps = max_steps
        self.agent_idx = agent_idx
        self.memory    = memory
        self.env       = MultiSatelliteEnv(config=self.config, max_steps=max_steps)

        # Continuous 4D action: [thrust, roll_torque, pitch_torque, yaw_torque]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "local":  spaces.Box(low=-10.0, high=10.0, shape=(LOCAL_DIM,),  dtype=np.float32),
            "global": spaces.Box(low=-10.0, high=10.0, shape=(GLOBAL_DIM,), dtype=np.float32),
        })
        self.other_model      = None
        self.curriculum_phase = 1

    def reset(self, seed=None, options=None):
        self.env.curriculum_phase = self.curriculum_phase
        obs_raw, info = self.env.reset(seed=seed)
        return self._augment(obs_raw[self.agent_idx]), info

    def step(self, action):
        n = self.config.num_satellites
        actions = np.zeros((n, 8), dtype=np.float32)
        actions[self.agent_idx] = action

        if self.other_model is not None:
            obs_raw_all = self.env._get_obs_list()
            for i in range(n):
                if i != self.agent_idx:
                    o = self._augment(obs_raw_all[i])
                    a, _ = self.other_model.predict(o, deterministic=True)
                    actions[i] = a

        obs_raw, rewards, term, trunc, info = self.env.step(actions)
        return self._augment(obs_raw[self.agent_idx]), rewards[self.agent_idx], term, trunc, info

    def _augment(self, base_obs: dict) -> dict:
        """Append 4-dim episodic memory context to local observation."""
        ctx = (self.memory.get_context() if self.memory is not None
               else np.zeros(MEMORY_CTX_DIM, dtype=np.float32))
        local = np.concatenate([base_obs["local"], ctx]).astype(np.float32)
        return {"local": local, "global": base_obs["global"]}

    def set_other_model(self, model):
        self.other_model = model

    def set_curriculum_phase(self, p: int):
        self.curriculum_phase = p
        self.env.curriculum_phase = p


# ─────────────────────────────────────────────────────────────────────────────
#  MultiSatelliteEnv (core physics + reward engine)
# ─────────────────────────────────────────────────────────────────────────────
class MultiSatelliteEnv:
    """
    Multi-satellite orbital mechanics environment covering Phases A-F.
    All behaviour is driven by a SatelliteConfig instance.
    """

    def __init__(self, config: SatelliteConfig = None, max_steps: int = 360):
        self.config          = config or PRESETS["starlink_leo"]
        self.num_satellites  = self.config.num_satellites
        self.max_steps       = max_steps
        self.curriculum_phase = 1
        self.np_random       = np.random.default_rng()

        self._step_sec    = self.config.orbit.step_seconds
        self._nominal_vel = self._compute_nominal_vel()

        # Walker Delta orbit distribution — Phase F
        self._walker_params = get_walker_delta_params(
            self.num_satellites,
            self.config.planes,
            inclination_deg=(self.config.tle.inclination_deg if self.config.tle else self.config.inclination_deg),
            raan_offset_deg=(self.config.tle.raan_deg if self.config.tle else 0.0),
            anomaly_offset_deg=(self.config.tle.mean_anomaly_deg if self.config.tle else 0.0),
        )
        # Compute per-plane target gap (360 / sats per plane)
        sats_per_plane = max(1, self.num_satellites // self.config.planes)
        self._target_gap = 360.0 / sats_per_plane

        # ── [PERF] Cache walker params as NumPy arrays (avoid repeated dict→array) ──
        self._wp_plane_ids    = np.array([wp["plane_id"] for wp in self._walker_params], dtype=np.int32)
        self._wp_inclinations = np.array([wp["inclination_deg"] for wp in self._walker_params], dtype=np.float64)
        self._wp_raans        = np.array([wp["raan_deg"] for wp in self._walker_params], dtype=np.float64)
        self._wp_anomalies    = np.array([wp["anomaly_offset_deg"] for wp in self._walker_params], dtype=np.float64)

        # ── State arrays ──────────────────────────────────────────────────────
        n = self.num_satellites
        self.agent_pos            = np.zeros(n)
        self.agent_vel            = np.zeros(n)
        self.agent_fuel           = np.zeros(n)
        self.agent_battery        = np.zeros(n)
        self.agent_attitude       = np.zeros((n, 3))    # roll, pitch, yaw
        self.agent_attitude_rates = np.zeros((n, 3))    # angular rates

        # Phase F: Thermal state
        self.agent_temp    = np.zeros(n)
        
        # Part 2 & 4: Delta-V and Commander Goal
        self.agent_delta_v = np.ones(n) * 1000.0
        self.commander_goal = np.zeros((n, 3)) # 3D vector for mission commander goal

        # Phase D: Data buffer
        self.agent_data    = np.zeros(n)
        self.gs_los        = np.zeros(n, dtype=bool)
        self.gs_blackout   = np.zeros(n, dtype=bool)
        self.gs_name       = [""] * n

        # Phase E: Fault state flags
        self.fault_wheel     = np.zeros(n, dtype=bool)
        self.fault_thruster  = np.zeros(n, dtype=bool)
        self.fault_sensor    = np.zeros(n, dtype=bool)
        self.in_recovery     = np.zeros(n, dtype=bool)
        self.fault_recovery_progress = np.zeros(n, dtype=int)
        self.faults_recovered = np.zeros(n, dtype=int)

        # General state
        self.eclipse_mode         = np.zeros(n, dtype=bool)
        self.eclipse_fraction     = np.zeros(n, dtype=np.float32)
        self.in_safe_mode         = np.zeros(n, dtype=bool)
        self.space_weather_active = False
        
        # Keplerian Debris objects: [{"a": semi_major, "e": ecc, "i": inc, "omega": arg_pe, "nu": true_anomaly}, ...]
        self.debris               = []
        self.earth_rot_deg        = 0.0
        self.current_step         = 0

        # Episode metrics — 2D NumPy array for O(1) per-step append
        self._reward_buf    = np.zeros((n, max_steps + 1), dtype=np.float64)
        self._reward_len    = 0   # current write index into _reward_buf
        self.collisions     = np.zeros(n, dtype=int)
        self.fuel_outs      = np.zeros(n, dtype=int)
        self.faults_logged  = np.zeros(n, dtype=int)

        # ── ISL encrypted relay crypto registry ──────────────────────────────
        # One ISLPacketCrypto object per directed link (i -> j).
        # Keys are (src, dst) tuples; created lazily on first use.
        self._isl_crypto: dict = {}

        # ── ISL Mesh Network for inter-satellite communication ────────────────
        self.isl_mesh = ISLMeshNetwork(self.num_satellites)
        self._isl_mesh_enabled = True
        self._experience_sharing_interval = 10  # Share experiences every N steps
        self._last_experience_share = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def reward_history(self):
        """Backward-compatible view: returns list of per-satellite reward arrays.
        
        External code accesses env.reward_history[i] and calls np.sum() on it,
        so we return sliced views of the 2D buffer.
        """
        length = min(self._reward_len, self._reward_buf.shape[1])
        return [self._reward_buf[i, :length] for i in range(self.num_satellites)]

    def _compute_nominal_vel(self) -> float:
        """Compute degrees-per-step for a circular orbit at this altitude."""
        import math
        r_km = EARTH_RADIUS_KM + self.config.orbit.altitude_km
        T_s  = 2 * math.pi * math.sqrt(r_km**3 / EARTH_MU)
        return 360.0 * self._step_sec / T_s

    def _compute_eclipse(self, pos_deg: float) -> tuple[bool, float]:
        """Determine if satellite is in eclipse and return (is_eclipse, fraction)."""
        if self.curriculum_phase < 2:
            return False, 0.0
        eclipse_half = self.config.orbit.eclipse_arc_deg / 2.0
        relative_pos = (pos_deg - self._sun_angle_deg) % 360.0
        
        dist_to_anti_sun = abs(180.0 - relative_pos)
        
        is_eclipse = dist_to_anti_sun < eclipse_half
        
        # Calculate fraction
        if dist_to_anti_sun < eclipse_half * 0.8:
            fraction = 1.0 # deep eclipse
        elif dist_to_anti_sun < eclipse_half:
            fraction = (eclipse_half - dist_to_anti_sun) / (eclipse_half * 0.2)
        else:
            fraction = 0.0
            
        return is_eclipse, fraction

    def _gap_error(self, i: int) -> float:
        """
        Angular gap error for satellite i vs the next satellite in the SAME plane.
        Compares against ideal spacing within the plane (target_gap).
        """
        plane_id = self._wp_plane_ids[i]
        same_plane_mask = (self._wp_plane_ids == plane_id) & (np.arange(self.num_satellites) != i)
        same_plane_indices = np.where(same_plane_mask)[0]
        
        if len(same_plane_indices) == 0:
            return 0.0
        
        gaps = (self.agent_pos[same_plane_indices] - self.agent_pos[i]) % 360.0
        min_gap = float(np.min(gaps))
        return abs(min_gap - self._target_gap)

    def _gap_errors_all(self) -> np.ndarray:
        """[PERF] Compute gap errors for ALL satellites in one vectorized pass.
        
        Returns:
            (N,) array of angular gap errors.
        """
        N = self.num_satellites
        result = np.zeros(N, dtype=np.float64)
        unique_planes = np.unique(self._wp_plane_ids)
        
        for pid in unique_planes:
            mask = self._wp_plane_ids == pid
            indices = np.where(mask)[0]
            if len(indices) <= 1:
                continue
            # Positions of satellites in this plane
            plane_pos = self.agent_pos[indices]
            # For each satellite in the plane, find the minimum forward gap
            # to any other satellite in the same plane
            # plane_pos[:, None] - plane_pos[None, :] gives (K, K) matrix
            diff = (plane_pos[None, :] - plane_pos[:, None]) % 360.0  # (K, K)
            np.fill_diagonal(diff, 360.0)  # exclude self
            min_gaps = np.min(diff, axis=1)  # (K,)
            result[indices] = np.abs(min_gaps - self._target_gap)
        
        return result

    def _get_neighbors(self, sat_idx: int) -> list[int]:
        """Return indices of the MAX_NEIGHBORS nearest satellites by angular distance."""
        dists = []
        for j in range(self.num_satellites):
            if j == sat_idx:
                continue
            diff = abs(self.agent_pos[j] - self.agent_pos[sat_idx])
            dists.append((min(diff, 360.0 - diff), j))
        dists.sort()
        return [idx for _, idx in dists[:MAX_NEIGHBORS]]

    def _get_all_neighbors(self) -> np.ndarray:
        """[PERF] Compute nearest MAX_NEIGHBORS for ALL satellites at once.
        
        Returns:
            (N, MAX_NEIGHBORS) int array of neighbor indices.  Rows with
            fewer than MAX_NEIGHBORS real neighbors are padded with -1.
        """
        N = self.num_satellites
        # (N, N) angular distance matrix
        diff = np.abs(self.agent_pos[:, None] - self.agent_pos[None, :])  # (N, N)
        ang = np.minimum(diff, 360.0 - diff)                              # (N, N)
        np.fill_diagonal(ang, np.inf)  # exclude self
        
        k = min(MAX_NEIGHBORS, N - 1)
        if k <= 0:
            return np.full((N, MAX_NEIGHBORS), -1, dtype=np.int32)
        
        # argpartition is O(N) vs O(N log N) for full sort
        part_idx = np.argpartition(ang, k, axis=1)[:, :k]  # (N, k)
        # Sort the k neighbors by distance for deterministic ordering
        row_idx = np.arange(N)[:, None]
        part_dists = ang[row_idx, part_idx]                 # (N, k)
        sorted_order = np.argsort(part_dists, axis=1)       # (N, k)
        result = part_idx[row_idx, sorted_order]             # (N, k)
        
        # Pad to MAX_NEIGHBORS if k < MAX_NEIGHBORS
        if k < MAX_NEIGHBORS:
            pad = np.full((N, MAX_NEIGHBORS - k), -1, dtype=np.int32)
            result = np.concatenate([result, pad], axis=1)
        
        return result.astype(np.int32)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_step = 0
        self.earth_rot_deg = 0.0
        self.debris = []
        # Keplerian Debris Initialization
        for _ in range(self.np_random.integers(5, 11)):
            self.debris.append({
                "a": self.np_random.uniform(EARTH_RADIUS_KM + 300, EARTH_RADIUS_KM + 1000),
                "e": self.np_random.uniform(0.0, 0.1),
                "i": self.np_random.uniform(0, 180),
                "omega": self.np_random.uniform(0, 360),
                "nu": self.np_random.uniform(0, 360)
            })

        n = self.num_satellites
        
        self._reward_buf[:] = 0.0
        self._reward_len    = 0
        self.collisions.fill(0)
        self.fuel_outs.fill(0)
        self.faults_logged.fill(0)
        self.faults_recovered.fill(0)
        self.in_safe_mode.fill(False)
        self.in_recovery.fill(False)
        self.space_weather_active = False
        self.agent_attitude.fill(0.0)
        self.agent_attitude_rates.fill(0.0)
        self.fault_wheel.fill(False)
        self.fault_thruster.fill(False)
        self.fault_sensor.fill(False)
        self.fault_recovery_progress.fill(0)
        self.gs_los.fill(False)
        self.gs_blackout.fill(False)
        self.gs_name = [""] * n

        # Sun angle randomized every episode
        self._sun_angle_deg = float(self.np_random.uniform(0, 360))

        # ── [VEC] Thruster efficiency — domain randomization (vectorized) ─────
        eff_range = (0.75, 1.25) if self.curriculum_phase >= 2 else (1.0, 1.0)
        self._thruster_eff = self.np_random.uniform(*eff_range, size=n)

        # ── [VEC] Initialize satellite states (all at once, no per-sat loop) ──
        temp_mid = (self.config.max_temp_c + self.config.min_temp_c) / 2.0
        pos_noise = self.np_random.uniform(-6, 6, size=n)
        self.agent_pos[:]     = (self._wp_anomalies + pos_noise) % 360.0
        self.agent_vel[:]     = self._nominal_vel + self.np_random.uniform(-0.12, 0.12, size=n)
        self.agent_fuel[:]    = 100.0
        self.agent_battery[:] = 100.0
        self.agent_temp[:]    = temp_mid
        self.agent_data[:]    = self.np_random.uniform(0.0, self.config.data_capacity_gb * 0.1, size=n)

        # Adversarial scenario injection (Phase 3+)
        if self.curriculum_phase >= 3:
            if self.np_random.random() < 0.18:
                self.space_weather_active = True
            if self.np_random.random() < 0.08:
                low = int(self.np_random.integers(0, n))
                self.agent_fuel[low] = float(self.np_random.uniform(5.0, 25.0))
            if self.np_random.random() < 0.10:
                dead = int(self.np_random.integers(0, n))
                self.agent_battery[dead] = float(self.np_random.uniform(8.0, 25.0))

        # ── [VEC] Phase 5+: Thermal stress injection (vectorized) ─────────────
        if self.curriculum_phase >= 5:
            stress_mask = self.np_random.random(n) < 0.12
            stress_offsets = self.np_random.uniform(0, 5, size=n)
            self.agent_temp[stress_mask] = self.config.max_temp_c - stress_offsets[stress_mask]

        # ── [VEC] Phase 6: Fault injection — vectorized random draws ──────────
        if self.curriculum_phase >= 6:
            wheel_faults    = self.np_random.random(n) < 0.07
            thruster_faults = self.np_random.random(n) < 0.07
            sensor_faults   = self.np_random.random(n) < 0.07
            self.fault_wheel[:]     = wheel_faults
            self.fault_thruster[:]  = thruster_faults
            self.fault_sensor[:]    = sensor_faults
            self.faults_logged[:]  += wheel_faults.astype(int) + thruster_faults.astype(int) + sensor_faults.astype(int)

        return self._get_obs_list(), {}

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_satellites, 8)
        if actions.shape != expected_shape:
            raise ValueError(f"actions must have shape {expected_shape}, got {actions.shape}")
        actions = np.clip(actions, -1.0, 1.0)
        self.current_step += 1
        # Earth rotates ~360°/86400s per second of sim time
        self.earth_rot_deg = (self.earth_rot_deg + (self._step_sec / 86400.0) * 360.0) % 360.0
        rewards = np.zeros(self.num_satellites, dtype=np.float64)

        # ── [VEC] Advance Keplerian debris — vectorized across all debris objects ──
        if self.debris:
            deb_a   = np.array([d["a"] for d in self.debris], dtype=np.float64)
            deb_nu  = np.array([d["nu"] for d in self.debris], dtype=np.float64)
            n_means = np.sqrt(GM_EARTH / (deb_a * 1000.0) ** 3)
            delta_nus = np.degrees(n_means * self._step_sec)
            deb_nu = (deb_nu + delta_nus) % 360.0
            for k, d in enumerate(self.debris):
                d["nu"] = float(deb_nu[k])

        N   = self.num_satellites
        thr = self.config.thruster
        orbit = self.config.orbit

        # ── [VEC] Fault overrides — boolean masks on action columns ───────────
        thrust_cmds = actions[:, 0].copy()          # (N,)
        roll_cmds   = actions[:, 1].copy()
        pitch_cmds  = actions[:, 2].copy()
        yaw_cmds    = actions[:, 3].copy()

        # Reaction wheel fault — freeze roll axis
        roll_cmds[self.fault_wheel] = 0.0
        # Thruster fault — 30% effectiveness
        thrust_cmds[self.fault_thruster] *= 0.3

        # ── [VEC] Fault detection & recovery progress ─────────────────────────
        any_fault = self.fault_wheel | self.fault_thruster | self.fault_sensor  # (N,) bool
        recovery_cmd = (
            (np.abs(thrust_cmds) < 0.2) &
            (np.linalg.norm(np.stack([roll_cmds, pitch_cmds, yaw_cmds], axis=1), axis=1) < 0.35)
        )  # (N,) bool
        self.in_recovery = any_fault & recovery_cmd
        self.fault_recovery_progress = np.where(
            self.in_recovery,
            self.fault_recovery_progress + 1,
            0,
        )
        # Successful recovery: 5 consecutive recovery steps
        recovered_mask = any_fault & (self.fault_recovery_progress >= 5)
        if recovered_mask.any():
            self.fault_wheel[recovered_mask]     = False
            self.fault_thruster[recovered_mask]  = False
            self.fault_sensor[recovered_mask]    = False
            self.in_recovery[recovered_mask]     = False
            self.fault_recovery_progress[recovered_mask] = 0
            self.faults_recovered[recovered_mask] += 1

        # ── [VEC] Thruster mapping — np.select across 5 bands ────────────────
        #  Bands: full-retro | light-retro | coast | light-pro | full-pro
        dv_vals   = np.array([
            -thr.max_dv_per_step,
            -thr.max_dv_per_step * 0.25,
             0.0,
             thr.max_dv_per_step * 0.25,
             thr.max_dv_per_step,
        ])
        fuel_vals = np.array([
            thr.fuel_cost_full,
            thr.fuel_cost_light,
            0.0,
            thr.fuel_cost_light,
            thr.fuel_cost_full,
        ])
        conds = [
            thrust_cmds < -0.6,
            (thrust_cmds >= -0.6) & (thrust_cmds < -0.2),
            (thrust_cmds >= -0.2) & (thrust_cmds <  0.2),
            (thrust_cmds >=  0.2) & (thrust_cmds <  0.6),
        ]
        d_v_arr  = np.select(conds, dv_vals[:4],  default=dv_vals[4])
        fuel_arr = np.select(conds, fuel_vals[:4], default=fuel_vals[4])

        # Space weather penalty
        if self.space_weather_active:
            d_v_arr *= 0.6

        # Apply thrust where fuel available and not in safe mode
        thr_eff = np.asarray(self._thruster_eff, dtype=np.float64)
        can_thrust = (self.agent_fuel >= fuel_arr) & (~self.in_safe_mode)
        fuel_out   = (fuel_arr > 0.0) & (self.agent_fuel < 0.5)

        self.agent_vel  += np.where(can_thrust, d_v_arr * thr_eff, 0.0)
        self.agent_fuel  = np.where(can_thrust, self.agent_fuel - fuel_arr, self.agent_fuel)
        # Battery and temp cost for firing thrusters
        fired = can_thrust & (fuel_arr > 0.0)
        self.agent_battery -= np.where(fired, THRUSTER_POWER_DRAW, 0.0)
        self.agent_temp    += np.where(fired, 2.0, 0.0)
        self.fuel_outs     += fuel_out.astype(int)

        # ── [VEC] Orbital perturbations (J2 + drag) ───────────────────────────
        j2_drift = orbit.j2_strength * np.sin(np.radians(self.agent_pos))
        drag     = -orbit.drag_coeff * np.sign(self.agent_vel - self._nominal_vel)
        self.agent_vel  = np.clip(
            self.agent_vel + j2_drift + drag,
            self._nominal_vel - 3.0,
            self._nominal_vel + 3.0,
        )
        self.agent_pos = (self.agent_pos + self.agent_vel) % 360.0

        # ── [VEC] 3D Attitude control ─────────────────────────────────────────
        att_cmds = np.stack([roll_cmds, pitch_cmds, yaw_cmds], axis=1)  # (N,3)
        self.agent_attitude_rates = np.clip(
            self.agent_attitude_rates + att_cmds * 0.5, -10.0, 10.0
        )
        self.agent_attitude = (self.agent_attitude + self.agent_attitude_rates) % 360.0
        wheel_power = WHEEL_POWER_DRAW * np.sum(np.abs(att_cmds), axis=1)  # (N,)
        self.agent_battery -= wheel_power

        # ── [VEC] Phase F: Thermal dynamics + eclipse ─────────────────────────
        # Vectorized eclipse check: relative position vs anti-sun
        eclipse_half = orbit.eclipse_arc_deg / 2.0
        if self.curriculum_phase >= 2:
            rel_pos   = (self.agent_pos - self._sun_angle_deg) % 360.0
            dist_anti = np.abs(180.0 - rel_pos)
            ecl_mask  = dist_anti < eclipse_half
            # Eclipse fraction
            deep_mask   = dist_anti < eclipse_half * 0.8
            partial_mask = (~deep_mask) & ecl_mask
            frac = np.where(
                deep_mask, 1.0,
                np.where(
                    partial_mask,
                    (eclipse_half - dist_anti) / (eclipse_half * 0.2 + 1e-9),
                    0.0,
                )
            )
        else:
            ecl_mask = np.zeros(N, dtype=bool)
            frac     = np.zeros(N, dtype=np.float32)

        self.eclipse_mode[:]     = ecl_mask
        self.eclipse_fraction[:] = frac.astype(np.float32)

        # Eclipse: drain battery + cool; sunlit: charge + heat
        self.agent_battery += np.where(ecl_mask,
                                       -(orbit.eclipse_fraction * 2.0),
                                        self.config.solar_charge_rate)
        self.agent_temp    += np.where(ecl_mask, -0.8, 0.4)
        # Wheel heat
        self.agent_temp    += wheel_power * 0.5
        self.agent_battery  = np.clip(self.agent_battery, 0.0, 100.0)

        # ── [VEC] Ground Station LOS — vectorized batch computation ───
        prev_safe = self.in_safe_mode.copy()
        
        # Pre-compute all satellite ECEF positions in batch (cached arrays)
        pos_arr = self.agent_pos
        incl_arr = self._wp_inclinations
        raan_arr = self._wp_raans
        
        r_km = EARTH_RADIUS_KM + orbit.altitude_km
        theta = np.radians(pos_arr)
        inc = np.radians(incl_arr)
        
        # Orbital plane coordinates (vectorized)
        x_orbit = r_km * np.cos(theta)
        y_orbit = r_km * np.sin(theta)
        
        # Rotate by inclination
        y_inc = y_orbit * np.cos(inc)
        z_inc = y_orbit * np.sin(inc)
        
        # Rotate by RAAN (with Earth rotation)
        raan_rad = np.radians(raan_arr - self.earth_rot_deg)
        x_ecef = x_orbit * np.cos(raan_rad) - y_inc * np.sin(raan_rad)
        y_ecef = x_orbit * np.sin(raan_rad) + y_inc * np.cos(raan_rad)
        z_ecef = z_inc
        
        # Batch LOS check against all ground stations
        min_elevation_deg = 5.0
        gs_los = np.zeros(N, dtype=bool)
        gs_name_arr = np.empty(N, dtype=object)
        gs_name_arr[:] = ""
        
        for gs in GROUND_STATIONS:
            gs_ecef = latlon_to_ecef(gs["lat"], gs["lon"])
            vec_gs_to_sat = np.column_stack([x_ecef, y_ecef, z_ecef]) - gs_ecef
            gs_norm = np.linalg.norm(gs_ecef)
            vec_norms = np.linalg.norm(vec_gs_to_sat, axis=1)
            
            valid = (gs_norm > 1e-6) & (vec_norms > 1e-6)
            if not np.any(valid):
                continue
                
            zenith = gs_ecef / gs_norm
            sat_dir = vec_gs_to_sat[valid] / vec_norms[valid, None]
            cos_angle = np.clip(np.dot(sat_dir, zenith), -1.0, 1.0)
            elevation_deg = 90.0 - np.degrees(np.arccos(cos_angle))
            
            mask = elevation_deg >= min_elevation_deg
            valid_indices = np.where(valid)[0]
            gs_los[valid_indices[mask]] = True
            gs_name_arr[valid_indices[mask]] = gs["name"]
        
        # Apply blackouts
        gs_blackout = np.zeros(N, dtype=bool)
        for i in range(N):
            if gs_los[i]:
                gs_blackout[i] = self.config.station_is_blacked_out(self.current_step, gs_name_arr[i])
        
        self.gs_los = gs_los & (~gs_blackout)
        self.gs_blackout = gs_blackout
        self.gs_name = gs_name_arr.tolist()

        # ── [VEC] Safe-mode detection ─────────────────────────────────────────
        low_power = self.agent_battery < SAFE_MODE_THRESHOLD
        over_heat = self.agent_temp    > self.config.max_temp_c
        cold_soak = self.agent_temp    < self.config.min_temp_c
        self.in_safe_mode[:] = low_power | over_heat | cold_soak

        # ── [VEC] Data generation (Earth Obs) ─────────────────────────────────
        if self.config.mission_type == MissionType.EARTH_OBS:
            gen_mask = self.agent_data < self.config.data_capacity_gb
            self.agent_data += np.where(gen_mask, 5.0, 0.0)

        # ── [VEC] Phase D: Downlink ───────────────────────────────────────────
        can_downlink  = self.gs_los & (~self.in_safe_mode)
        downlinked    = np.minimum(self.agent_data, 20.0) * can_downlink
        self.agent_data = np.maximum(0.0, self.agent_data - downlinked)
        rewards      += np.where(downlinked > 0, 0.3, 0.0)

        # ── [VEC] GNC Maneuvers (Hohmann / avoidance / deorbit) ──────────────
        current_r = (EARTH_RADIUS_KM + orbit.altitude_km) * 1000.0
        current_v = np.sqrt(GM_EARTH / current_r)

        if actions.shape[1] > 5:
            hohmann_cmds  = actions[:, 5]
            avoidance_cmds = actions[:, 6] if actions.shape[1] > 6 else np.full(N, -1.0)
            deorbit_cmds   = actions[:, 7] if actions.shape[1] > 7 else np.full(N, -1.0)

            # Hohmann
            target_r_h = current_r + 50_000.0
            dv_h = abs(current_v - np.sqrt(GM_EARTH / target_r_h)) * 2.0
            do_h = (hohmann_cmds > 0) & (self.agent_delta_v > dv_h)
            self.agent_delta_v -= np.where(do_h, dv_h, 0.0)
            rewards += np.where(do_h, 0.5, 0.0)

            # Avoidance
            dv_av = 15.0
            do_av = (avoidance_cmds > 0) & (self.agent_delta_v > dv_av)
            self.agent_delta_v -= np.where(do_av, dv_av, 0.0)
            rewards += np.where(do_av, 1.0, 0.0)

            # Deorbit
            dv_de = abs(current_v - np.sqrt(GM_EARTH / (EARTH_RADIUS_KM * 1000.0)))
            do_de = (deorbit_cmds > 0) & (self.agent_delta_v > dv_de)
            self.agent_delta_v -= np.where(do_de, dv_de, 0.0)
            rewards += np.where(do_de, 2.0, 0.0)

        # ── [VEC] Commander goal update (every 10 steps) ──────────────────────
        if self.current_step % 10 == 0:
            low_bat  = self.agent_battery < 30.0
            low_dv   = (~low_bat) & (self.agent_delta_v < 100.0)
            nominal  = (~low_bat) & (~low_dv)
            self.commander_goal[low_bat]  = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.commander_goal[low_dv]   = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.commander_goal[nominal]  = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # ── [VEC] ISL Link-quality matrix — N×N array ────────────────────────
        # pos_i (N,1) - pos_j (1,N) → angular distance matrix
        pos_col = self.agent_pos[:, None]         # (N,1)
        pos_row = self.agent_pos[None, :]         # (1,N)
        diff_mat   = np.abs(pos_col - pos_row)    # (N,N)
        ang_mat    = np.minimum(diff_mat, 360.0 - diff_mat)
        lq_mat     = np.clip(1.0 - ang_mat / 45.0, 0.0, 1.0)  # (N,N)
        np.fill_diagonal(lq_mat, 0.0)             # no self-link

        # ── [VEC] Best relay neighbor selection ───────────────────────────────
        # Candidates: neighbor has GS LOS, not in safe-mode, lq > 0.3
        relay_eligible = (
            self.gs_los[None, :]       &          # (1,N) broadcast → (N,N)
            (~self.in_safe_mode[None, :]) &
            (lq_mat > 0.3)
        )  # (N,N)
        # For each satellite i, find its best relay neighbor j
        lq_candidates = np.where(relay_eligible, lq_mat, 0.0)  # (N,N)
        best_neighbor_arr = np.argmax(lq_candidates, axis=1)    # (N,) — index of best j
        best_lq_arr       = lq_candidates[
            np.arange(N), best_neighbor_arr
        ]  # (N,) — link quality of that neighbor
        has_relay = best_lq_arr > 0.0  # (N,) — True if a valid neighbor exists

        # ── ISL Encrypted relay — serial only for crypto handshake ────────────
        # Routing decisions above are vectorized; only per-link crypto stays serial.
        relay_actions = actions[:, 4] if actions.shape[1] > 4 else np.full(N, -1.0)
        for i in range(N):
            if (relay_actions[i] > 0.0 and self.agent_data[i] > 0
                    and not self.in_safe_mode[i] and not self.gs_los[i]
                    and has_relay[i]):
                best_neighbor = int(best_neighbor_arr[i])
                best_link     = float(best_lq_arr[i])
                relayed_gb    = min(self.agent_data[i], 10.0 * best_link)

                crypto_key = (i, best_neighbor)
                if ISLPacketCrypto is not None:
                    if crypto_key not in self._isl_crypto:
                        self._isl_crypto[crypto_key] = ISLPacketCrypto(i, best_neighbor)
                    crypto = self._isl_crypto[crypto_key]

                    raw       = ISLPacketCrypto.pack_data_packet(
                        src_idx=i, data_gb=relayed_gb,
                        timestamp=float(self.current_step),
                    )
                    encrypted = crypto.encrypt(raw)
                    rx_key    = (best_neighbor, i)
                    if rx_key not in self._isl_crypto:
                        self._isl_crypto[rx_key] = ISLPacketCrypto(best_neighbor, i)
                    decrypted = self._isl_crypto[crypto_key].decrypt(encrypted)
                    packet    = ISLPacketCrypto.unpack_data_packet(decrypted) if decrypted else None

                    if packet and abs(packet["data_gb"] - relayed_gb) < 0.01:
                        self.agent_data[i]             -= relayed_gb
                        self.agent_data[best_neighbor] += relayed_gb
                        rewards[i] += 0.4
                        logging.debug(
                            "ISL RELAY SAT%d\u2192SAT%d  %.2f GB  lq=%.2f  encrypted+verified",
                            i, best_neighbor, relayed_gb, best_link,
                        )
                    else:
                        logging.warning(
                            "ISL RELAY SAT%d\u2192SAT%d: packet verification FAILED",
                            i, best_neighbor,
                        )
                else:
                    self.agent_data[i]             -= relayed_gb
                    self.agent_data[best_neighbor] += relayed_gb
                    rewards[i] += 0.1

        # ── [VEC] Buffer overflow penalty ─────────────────────────────────────
        rewards -= np.where(self.agent_data >= self.config.data_capacity_gb, 0.2, 0.0)

        # ── [VEC] Safe-mode penalty ───────────────────────────────────────────
        just_entered = self.in_safe_mode & (~prev_safe)
        stuck_safe   = self.in_safe_mode & prev_safe
        rewards -= np.where(just_entered, 10.0, 0.0)
        rewards -= np.where(stuck_safe,    0.1, 0.0)

        # ── [VEC] Phase C: Mission-specific rewards ───────────────────────────
        if self.config.mission_type == MissionType.COMMS:
            gap_errs = self._gap_errors_all()
            rew_comms = np.select(
                [gap_errs <= 2.0,
                 gap_errs <= 10.0,
                 gap_errs <= 30.0],
                [1.0,
                 0.8 - (gap_errs - 2.0) * 0.05,
                 0.4 * (1.0 - (gap_errs - 10.0) / 20.0)],
                default=0.0,
            )
            rewards += rew_comms
            # Coast bonus
            rewards += np.where((gap_errs <= 10.0) & (np.abs(thrust_cmds) < 0.2), 0.1, 0.0)

        elif self.config.mission_type == MissionType.EARTH_OBS:
            roll_deg  = self.agent_attitude[:, 0]
            pitch_deg = self.agent_attitude[:, 1]
            roll_err  = np.minimum(roll_deg,  360.0 - roll_deg)
            pitch_err = np.minimum(pitch_deg, 360.0 - pitch_deg)
            pointing_err = roll_err + pitch_err
            rew_obs = np.select(
                [pointing_err < 3.0,
                 pointing_err < 10.0,
                 pointing_err < 30.0],
                [1.0, 0.6, 0.2],
                default=-0.2,
            )
            rewards += rew_obs
            rewards -= np.where(self.agent_data >= self.config.data_capacity_gb, 0.5, 0.0)

        elif self.config.mission_type == MissionType.SCIENCE:
            yaw_rates  = self.agent_attitude_rates[:, 2]
            spin_err   = np.abs(yaw_rates - 5.0)
            rew_sci    = np.select(
                [spin_err < 0.5, spin_err < 2.0],
                [1.0, 0.4],
                default=-0.1,
            )
            rewards += rew_sci
            stable_rp = (
                (np.abs(self.agent_attitude_rates[:, 0]) < 0.3) &
                (np.abs(self.agent_attitude_rates[:, 1]) < 0.3)
            )
            rewards += np.where(stable_rp, 0.1, 0.0)

        # ── [VEC] Phase E: Fault recovery bonus ──────────────────────────────
        rewards += np.where(any_fault & self.in_recovery, 0.2, 0.0)
        if self.config.mission_type == MissionType.EARTH_OBS:
            roll_deg  = self.agent_attitude[:, 0]
            pitch_deg = self.agent_attitude[:, 1]
            good_point = (
                (np.minimum(roll_deg, 360.0 - roll_deg) +
                 np.minimum(pitch_deg, 360.0 - pitch_deg)) < 15.0
            )
            rewards += np.where(any_fault & self.in_recovery & good_point, 0.3, 0.0)

        # ── [VEC] Phase F: Thermal margin reward ─────────────────────────────
        temp_range = self.config.max_temp_c - self.config.min_temp_c + 1e-6
        temp_norm  = (self.agent_temp - self.config.min_temp_c) / temp_range
        rewards   += np.where((temp_norm > 0.25) & (temp_norm < 0.75), 0.05, 0.0)

        # ── [VEC] Debris collision penalty ────────────────────────────────────
        if self.debris:
            deb_nu_arr = np.array([d["nu"] for d in self.debris], dtype=np.float64)  # (D,)
            # pos_i (N,1) - deb_nu (1,D) → (N,D) proximity matrix
            diff_nd = np.abs(self.agent_pos[:, None] - deb_nu_arr[None, :])           # (N,D)
            prox_nd = np.minimum(diff_nd, 360.0 - diff_nd)                            # (N,D)
            collision_mask = prox_nd < 2.0   # (N,D) bool
            close_mask     = prox_nd < 5.0   # (N,D) bool
            # Any debris within collision range
            n_collisions = collision_mask.sum(axis=1)   # (N,)
            rewards -= (n_collisions * 10.0).astype(np.float64)
            self.collisions += n_collisions.astype(int)
            # Close proximity penalty (only for non-collision close passes)
            close_only = close_mask & (~collision_mask)
            rewards -= (close_only.sum(axis=1) * 0.1).astype(np.float64)

        # ── Record reward history (vectorized: column write into 2D buffer) ──
        idx = self._reward_len
        if idx < self._reward_buf.shape[1]:
            self._reward_buf[:, idx] = rewards
        self._reward_len = idx + 1

        # ── ISL Mesh Network Step ─────────────────────────────────────────────
        if self._isl_mesh_enabled:
            self._step_isl_mesh(lq_mat)

        # ── Experience Sharing via Gossip ─────────────────────────────────────
        self._share_experiences()

        term  = False
        trunc = self.current_step >= self.max_steps
        info  = {
            "ground_station_blackouts": self.gs_blackout.copy(),
            "faults_recovered":         self.faults_recovered.copy(),
            "isl_stats": self.isl_mesh.get_all_stats() if self._isl_mesh_enabled else {},
        }
        return self._get_obs_list(), rewards, term, trunc, info

    # ── ISL Mesh Network Methods ─────────────────────────────────────────────────
    def _step_isl_mesh(self, lq_mat: np.ndarray = None):
        """Step the ISL mesh network with current link qualities.
        
        Args:
            lq_mat: Pre-computed (N,N) link-quality matrix from step().
                    If None, recomputes it (e.g. when called standalone).
        """
        if lq_mat is None:
            # Vectorized fallback: recompute the N×N link-quality matrix
            pos_col  = self.agent_pos[:, None]
            pos_row  = self.agent_pos[None, :]
            diff_mat = np.abs(pos_col - pos_row)
            ang_mat  = np.minimum(diff_mat, 360.0 - diff_mat)
            lq_mat   = np.clip(1.0 - ang_mat / 45.0, 0.0, 1.0)
            np.fill_diagonal(lq_mat, 0.0)

        N = self.num_satellites
        # Convert (N,N) matrix to dict expected by mesh.step()
        link_qualities = {
            (i, j): float(lq_mat[i, j])
            for i in range(N) for j in range(N) if i != j
        }
        # Add noise in-place using vectorized op, then rebuild the dict
        noise = self.np_random.uniform(0.9, 1.1, size=(N, N))
        lq_mat_noisy = np.clip(lq_mat * noise, 0.0, 1.0)
        np.fill_diagonal(lq_mat_noisy, 0.0)
        link_qualities = {
            (i, j): float(lq_mat_noisy[i, j])
            for i in range(N) for j in range(N) if i != j
        }

        # Step the mesh network
        self.isl_mesh.step(link_qualities)

        # Process relay actions using already-computed lq_mat
        self._process_agent_relay_actions(lq_mat_noisy)

    def _process_agent_relay_actions(self, lq_mat: np.ndarray = None):
        """Process relay actions and inject packets into mesh using vectorized link-quality."""
        N = self.num_satellites

        # Build lq_mat if not provided
        if lq_mat is None:
            pos_col  = self.agent_pos[:, None]
            pos_row  = self.agent_pos[None, :]
            diff_mat = np.abs(pos_col - pos_row)
            ang_mat  = np.minimum(diff_mat, 360.0 - diff_mat)
            lq_mat   = np.clip(1.0 - ang_mat / 45.0, 0.0, 1.0)
            np.fill_diagonal(lq_mat, 0.0)

        # Vectorized: for each sat, find best neighbor with GS LOS (quality > 0.3)
        gs_los_row = self.gs_los[None, :]                     # (1,N)
        safe_row   = (~self.in_safe_mode)[None, :]            # (1,N)
        eligible   = (lq_mat > 0.3) & gs_los_row & safe_row  # (N,N)
        lq_cands   = np.where(eligible, lq_mat, 0.0)          # (N,N)
        best_j     = np.argmax(lq_cands, axis=1)              # (N,)
        best_q     = lq_cands[np.arange(N), best_j]           # (N,)

        for i in range(N):
            if (self.agent_data[i] > 1.0 and not self.gs_los[i]
                    and not self.in_safe_mode[i] and best_q[i] > 0.0):
                best_neighbor = int(best_j[i])
                relayed_gb    = min(self.agent_data[i], 10.0 * float(best_q[i]))
                packet = ISLPacket(
                    packet_type=PacketType.DATA_RELAY,
                    src_sat_idx=i,
                    dst_sat_idx=best_neighbor,
                    seq_num=self.isl_mesh.nodes[i]._seq_counter + 1,
                    payload=ISLPacketCrypto.pack_data_packet(i, relayed_gb, self.current_step),
                )
                self.isl_mesh.nodes[i].tx_queue.append((best_neighbor, packet))
                self.agent_data[i]             -= relayed_gb
                self.agent_data[best_neighbor] += relayed_gb

    def _share_experiences(self):
        """Share experiences across satellites via ISL mesh gossip."""
        if not hasattr(self, 'episodic_memory') or self.episodic_memory is None:
            return
        
        # Share every N steps
        if self.current_step - self._last_experience_share < self._experience_sharing_interval:
            return
        
        self._last_experience_share = self.current_step
        
        # Get recent episodes from memory
        if hasattr(self.episodic_memory, 'episodes') and self.episodic_memory.episodes:
            recent_episodes = self.episodic_memory.episodes[-3:]  # Last 3 episodes
            
            for episode in recent_episodes:
                # Broadcast episode to neighbors via mesh
                episode_data = {
                    "type": "EXPERIENCE_SHARE",
                    "episode_id": episode.episode_id,
                    "total_reward": episode.total_reward,
                    "collisions": episode.collisions,
                    "fuel_outs": episode.fuel_outs,
                    "steps": episode.steps,
                    "was_eclipse": episode.was_eclipse,
                    "was_weather": episode.was_weather,
                    "was_fault": episode.was_fault,
                    "salience": episode.salience,
                    "orbital_state": getattr(episode, 'orbital_state', {}),
                    "commander_goal": getattr(episode, 'commander_goal', []),
                    "action_sequence": getattr(episode, 'action_sequence', []),
                    "maneuver_performed": getattr(episode, 'maneuver_performed', ""),
                    "fuel_cost": getattr(episode, 'fuel_cost', 0.0),
                    "data_routed_via_isl": getattr(episode, 'data_routed_via_isl', False),
                    "timestamp_step": self.current_step,
                }
                
                # Send from a random node to a random neighbor (simulate gossip)
                import random
                src_node = random.randint(0, self.num_satellites - 1)
                dst_node = random.randint(0, self.num_satellites - 1)
                if src_node != dst_node:
                    payload = json.dumps(episode_data).encode()
                    self.isl_mesh.nodes[src_node].send_data(dst_node, payload, PacketType.TELEMETRY_BEACON)

    def receive_shared_experiences(self, sat_idx: int) -> List[Dict]:
        """Receive and decode shared experiences from ISL mesh for a specific satellite."""
        experiences = []
        node = self.isl_mesh.nodes[sat_idx]
        packets = node.get_received_packets()
        
        for packet in packets:
            if packet.packet_type == PacketType.TELEMETRY_BEACON:
                try:
                    data = json.loads(packet.payload.decode())
                    if data.get("type") == "EXPERIENCE_SHARE":
                        experiences.append(data)
                except Exception:
                    pass
        return experiences

    # ── Observations (vectorized batch build) ─────────────────────────────────
    def _get_obs_list(self) -> list[dict]:
        N          = self.num_satellites
        config_vec = self.config.to_obs_vector()   # 6-dim, same for all sats

        # Fleet-wide global stats (6 dims) — computed once, shared
        global_stats = np.array([
            float(np.mean(self.agent_fuel))    / 100.0,
            float(np.mean(self.agent_battery)) / 100.0,
            float(np.var(self.agent_pos))      / (360.0 ** 2),
            float(np.var(self.agent_vel)),
            float(np.mean(self.agent_temp))    / 100.0,
            float(np.mean(self.agent_data))    / (self.config.data_capacity_gb + 1e-6),
        ], dtype=np.float32)

        # ── [VEC] Batch neighbor computation (N×MAX_NEIGHBORS) ────────────────
        all_neighbors = self._get_all_neighbors()  # (N, MAX_NEIGHBORS) int32

        # ── [VEC] Debris proximity for all satellites at once ─────────────────
        deb_vec = np.zeros(3, dtype=np.float32)
        min_deb_dists = np.ones(N, dtype=np.float64)
        if self.debris:
            deb_nu = np.array([d["nu"] for d in self.debris[:3]], dtype=np.float64)
            deb_vec[:len(deb_nu)] = deb_nu / 360.0
            # (N, D) proximity matrix
            diff_nd = np.abs(self.agent_pos[:, None] - deb_nu[None, :])  # (N, D)
            prox_nd = np.minimum(diff_nd, 360.0 - diff_nd) / 180.0      # (N, D)
            min_deb_dists = np.min(prox_nd, axis=1)                      # (N,)

        # ── [VEC] Sensor noise (applied to observed pos/vel) ──────────────────
        obs_pos = self.agent_pos.copy()       # (N,)
        obs_vel = self.agent_vel.copy()       # (N,)
        sensor_mask = self.fault_sensor       # (N,) bool
        n_faulty = int(np.sum(sensor_mask))
        if n_faulty > 0:
            obs_pos[sensor_mask] += self.np_random.normal(0, 5.0, size=n_faulty)
            obs_vel[sensor_mask] += self.np_random.normal(0, 1.0, size=n_faulty)

        closing_rates = obs_vel - self._nominal_vel  # (N,)

        # ── [VEC] Attitude features — (N, 6) ─────────────────────────────────
        att_norm = (self.agent_attitude % 360.0) / 180.0 - 1.0   # (N, 3)
        rate_norm = np.clip(self.agent_attitude_rates / 10.0, -1.0, 1.0)  # (N, 3)
        att_all = np.concatenate([att_norm, rate_norm], axis=1).astype(np.float32)  # (N, 6)

        # ── [VEC] Build local neighbor features — (N, NEIGHBOR_DIM) ───────────
        neigh_all = np.zeros((N, NEIGHBOR_DIM), dtype=np.float32)
        for i in range(N):
            nbrs = all_neighbors[i]
            valid = nbrs[nbrs >= 0]
            k = len(valid)
            if k > 0:
                rel_p = ((self.agent_pos[valid] - obs_pos[i]) % 360.0) / 180.0 - 1.0
                rel_v = np.clip((self.agent_vel[valid] - obs_vel[i]) / 3.0, -1.0, 1.0)
                fuel_n = self.agent_fuel[valid] / 100.0
                batt_n = self.agent_battery[valid] / 100.0
                isl_n = np.ones(k, dtype=np.float32)
                flat = np.column_stack([rel_p, rel_v, fuel_n, batt_n, isl_n]).flatten()
                neigh_all[i, :min(len(flat), NEIGHBOR_DIM)] = flat[:NEIGHBOR_DIM]

        # ── [VEC] Base LOCAL obs — (N, 11) ────────────────────────────────────
        base_local = np.column_stack([
            self.agent_pos / 360.0,
            np.full(N, self.config.orbit.altitude_km / 10000.0),
            self.eclipse_fraction.astype(np.float32),
            self.gs_los.astype(np.float32),
            min_deb_dists,
            self.agent_delta_v / 1000.0,
            np.clip(self.agent_temp / 100.0, -2.0, 2.0),
            self.agent_battery / 100.0,
            self.commander_goal[:, 0],
            self.commander_goal[:, 1],
            self.commander_goal[:, 2],
        ]).astype(np.float32)  # (N, 11)

        # ── [VEC] Recovery flag — (N, 1) ──────────────────────────────────────
        recovery_flags = self.in_recovery.astype(np.float32).reshape(N, 1)

        # ── [VEC] Config vector broadcast — (N, 6) ────────────────────────────
        config_tiled = np.tile(config_vec, (N, 1))  # (N, 6)

        # ── Assemble LOCAL obs — (N, 44) ──────────────────────────────────────
        local_all = np.concatenate([
            base_local, att_all, neigh_all, config_tiled, recovery_flags
        ], axis=1).astype(np.float32)  # (N, LOCAL_DIM_BASE)

        assert local_all.shape == (N, LOCAL_DIM_BASE), (
            f"LOCAL dim mismatch: expected (N, {LOCAL_DIM_BASE}), got {local_all.shape}"
        )

        # ── [VEC] Global own features — (N, 6) ────────────────────────────────
        data_cap = self.config.data_capacity_gb + 1e-6
        own_feat = np.column_stack([
            self.agent_pos / 360.0,
            np.clip(closing_rates / 3.0, -1.0, 1.0),
            self.agent_fuel / 100.0,
            self.agent_battery / 100.0,
            np.clip(self.agent_temp / 100.0, -2.0, 2.0),
            np.clip(self.agent_data / data_cap, 0.0, 1.0),
        ]).astype(np.float32)  # (N, 6)

        # ── [VEC] Global neighbor features — (N, 40) ─────────────────────────
        n_global_all = np.zeros((N, 10 * MAX_NEIGHBORS), dtype=np.float32)
        for i in range(N):
            nbrs = all_neighbors[i]
            valid = nbrs[nbrs >= 0]
            k = len(valid)
            if k > 0:
                n_close = self.agent_vel[valid] - self._nominal_vel
                ang_d = np.abs(self.agent_pos[i] - self.agent_pos[valid])
                ang_d = np.minimum(ang_d, 360.0 - ang_d)
                lq = np.clip(1.0 - ang_d / 45.0, 0.0, 1.0)
                flat = np.column_stack([
                    self.agent_pos[valid] / 360.0,
                    np.clip(n_close / 3.0, -1.0, 1.0),
                    self.agent_fuel[valid] / 100.0,
                    self.agent_battery[valid] / 100.0,
                    np.clip(self.agent_temp[valid] / 100.0, -2.0, 2.0),
                    np.clip(self.agent_data[valid] / data_cap, 0.0, 1.0),
                    np.zeros(k),   # task_queue_size
                    np.zeros(k),   # last_episode_reward
                    lq,
                    np.ones(k),    # valid_decryption
                ]).flatten()
                n_global_all[i, :min(len(flat), 10 * MAX_NEIGHBORS)] = flat[:10 * MAX_NEIGHBORS]

        # ── [VEC] Weather/eclipse — shared (2,) ───────────────────────────────
        weather_eclipse = np.array([
            float(self.space_weather_active),
            float(np.mean(self.eclipse_fraction)),
        ], dtype=np.float32)

        # ── [VEC] Assemble GLOBAL obs — (N, 63) ──────────────────────────────
        # Tile shared vectors: global_stats (6), deb_vec (3), weather_eclipse (2), config_vec (6)
        shared_tail = np.concatenate([global_stats, deb_vec, weather_eclipse, config_vec])  # (17,)
        shared_tiled = np.tile(shared_tail, (N, 1))  # (N, 17)

        global_all = np.concatenate([
            own_feat, n_global_all, shared_tiled
        ], axis=1).astype(np.float32)  # (N, GLOBAL_DIM)

        assert global_all.shape == (N, GLOBAL_DIM), (
            f"GLOBAL dim mismatch: expected (N, {GLOBAL_DIM}), got {global_all.shape}"
        )

        # ── Build output list ─────────────────────────────────────────────────
        return [
            {"local": local_all[i], "global": global_all[i]}
            for i in range(N)
        ]


# Import to ensure they are accessible from satellite_env for train.py
EARTH_RADIUS_KM = 6371.0
EARTH_MU        = 398600.4418
