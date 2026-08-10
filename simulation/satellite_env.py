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

Observation dims (must match ctde_policy.py):
  LOCAL_DIM_BASE = 49   (before memory context)
  LOCAL_DIM      = 53   (after 4-dim memory context appended by SingleAgentWrapper)
  GLOBAL_DIM     = 47
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

from simulation.sat_config import SatelliteConfig, PRESETS, MissionType
from simulation.orbital_physics import (
    get_walker_delta_params, anomaly_to_ecef, check_ground_station_los
)

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

# LOCAL obs breakdown (49 dims total before memory context):
#   pos, closing_rate, fuel, bat, gap_err, deb×3, eclipse,
#   weather, temp, data, fault_wheel, fault_thruster, fault_sensor,
#   gs_los = 16 base dims
#   + 3 roll/pitch/yaw + 3 rates = 6 attitude dims
#   + 5 × 4 neighbors = 20 neighbor dims
#   + 6 config dims
#   = 16 + 6 + 20 + 6 + 1 (recovery flag) = 49
BASE_LOCAL_DIM   = 16
ATTITUDE_DIM     = 6
NEIGHBOR_DIM     = 5 * MAX_NEIGHBORS   # 20
CONFIG_DIM       = 6
RECOVERY_DIM     = 1   # binary flag: AI is in a recovery manoeuvre
MEMORY_CTX_DIM   = 4

LOCAL_DIM_BASE   = BASE_LOCAL_DIM + ATTITUDE_DIM + NEIGHBOR_DIM + CONFIG_DIM + RECOVERY_DIM  # 49
LOCAL_DIM        = LOCAL_DIM_BASE + MEMORY_CTX_DIM                                            # 53

# GLOBAL obs breakdown (47 dims):
#   own (6) + 4 neighbors × 6 (24) + global_stats (6) + debris (3) + weather+eclipse (2) + config (6)
GLOBAL_DIM       = 6 + (6 * MAX_NEIGHBORS) + 6 + 3 + 2 + CONFIG_DIM   # 47


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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

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
        actions = np.zeros((n, 4), dtype=np.float32)
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
        self.in_safe_mode         = np.zeros(n, dtype=bool)
        self.space_weather_active = False
        self.debris               = []
        self.earth_rot_deg        = 0.0
        self.current_step         = 0

        # Episode metrics
        self.reward_history = [[] for _ in range(n)]
        self.collisions     = np.zeros(n, dtype=int)
        self.fuel_outs      = np.zeros(n, dtype=int)
        self.faults_logged  = np.zeros(n, dtype=int)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _compute_nominal_vel(self) -> float:
        """Compute degrees-per-step for a circular orbit at this altitude."""
        import math
        r_km = EARTH_RADIUS_KM + self.config.orbit.altitude_km
        T_s  = 2 * math.pi * math.sqrt(r_km**3 / EARTH_MU)
        return 360.0 * self._step_sec / T_s

    def _compute_eclipse(self, pos_deg: float) -> bool:
        """Determine if satellite is in eclipse based on sun angle and orbit arc."""
        if self.curriculum_phase < 2:
            return False
        eclipse_half = self.config.orbit.eclipse_arc_deg / 2.0
        relative_pos = (pos_deg - self._sun_angle_deg) % 360.0
        return (180.0 - eclipse_half) < relative_pos < (180.0 + eclipse_half)

    def _gap_error(self, i: int) -> float:
        """
        Angular gap error for satellite i vs the next satellite in the SAME plane.
        Compares against ideal spacing within the plane (target_gap).
        """
        plane_id = self._walker_params[i]["plane_id"]
        # Find the next satellite in the same plane
        same_plane = [j for j in range(self.num_satellites)
                      if j != i and self._walker_params[j]["plane_id"] == plane_id]
        if not same_plane:
            return 0.0

        # Find the one with the smallest positive angular gap ahead
        gaps = [(self.agent_pos[j] - self.agent_pos[i]) % 360.0 for j in same_plane]
        min_gap = min(gaps)
        return abs(min_gap - self._target_gap)

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

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_step = 0
        self.earth_rot_deg = 0.0
        self.debris = []
        n = self.num_satellites

        self.reward_history = [[] for _ in range(n)]
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

        # Thruster efficiency — domain randomization
        eff_range = (0.75, 1.25) if self.curriculum_phase >= 2 else (1.0, 1.0)
        self._thruster_eff = [
            float(self.np_random.uniform(*eff_range)) for _ in range(n)
        ]

        # Initialize satellite states
        temp_mid = (self.config.max_temp_c + self.config.min_temp_c) / 2.0
        for i in range(n):
            wp = self._walker_params[i]
            self.agent_pos[i]     = (wp["anomaly_offset_deg"] + float(self.np_random.uniform(-6, 6))) % 360.0
            self.agent_vel[i]     = self._nominal_vel + float(self.np_random.uniform(-0.12, 0.12))
            self.agent_fuel[i]    = 100.0
            self.agent_battery[i] = 100.0
            self.agent_temp[i]    = temp_mid
            self.agent_data[i]    = float(self.np_random.uniform(0.0, self.config.data_capacity_gb * 0.1))

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

        # Phase 5+: Thermal stress injection
        if self.curriculum_phase >= 5:
            for i in range(n):
                if self.np_random.random() < 0.12:
                    # Start some satellites near thermal limit
                    self.agent_temp[i] = self.config.max_temp_c - float(self.np_random.uniform(0, 5))

        # Phase 6: Fault injection — Phase E hardware faults
        if self.curriculum_phase >= 6:
            for i in range(n):
                if self.np_random.random() < 0.07:
                    self.fault_wheel[i] = True
                    self.faults_logged[i] += 1
                if self.np_random.random() < 0.07:
                    self.fault_thruster[i] = True
                    self.faults_logged[i] += 1
                if self.np_random.random() < 0.07:
                    self.fault_sensor[i] = True
                    self.faults_logged[i] += 1

        return self._get_obs_list(), {}

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_satellites, 4)
        if actions.shape != expected_shape:
            raise ValueError(f"actions must have shape {expected_shape}, got {actions.shape}")
        actions = np.clip(actions, -1.0, 1.0)
        self.current_step += 1
        # Earth rotates ~360°/86400s per second of sim time
        self.earth_rot_deg = (self.earth_rot_deg + (self._step_sec / 86400.0) * 360.0) % 360.0
        rewards = np.zeros(self.num_satellites)

        # Advance debris
        for d in self.debris:
            d["pos"] = (d["pos"] + d["vel"]) % 360.0

        # Spawn / despawn debris (Phase 3+)
        if self.curriculum_phase >= 3:
            if len(self.debris) < MAX_DEBRIS and self.np_random.random() < DEBRIS_SPAWN:
                self.debris.append({
                    "pos": float(self.np_random.uniform(0, 360)),
                    "vel": float(self.np_random.uniform(-2.0, 2.0)),
                })
            self.debris = [d for d in self.debris if self.np_random.random() > DEBRIS_DESPAWN]

        for i in range(self.num_satellites):
            act = np.array(actions[i], dtype=np.float32)
            thrust_cmd                 = float(act[0])
            roll_cmd, pitch_cmd, yaw_cmd = float(act[1]), float(act[2]), float(act[3])

            # ── Phase E: Fault overrides ──────────────────────────────────────
            if self.fault_wheel[i]:
                roll_cmd = 0.0   # Reaction wheel saturation — roll axis frozen

            if self.fault_thruster[i]:
                thrust_cmd *= 0.3   # Degraded thruster — reduced effectiveness

            # ── Fault detection and autonomous recovery (Phase E) ─────────────
            # A recovery is an explicit low-activity safe-hold/detumble command,
            # sustained long enough for onboard diagnostics and reset to finish.
            any_fault = self.fault_wheel[i] or self.fault_thruster[i] or self.fault_sensor[i]
            recovery_cmd = (abs(thrust_cmd) < 0.2 and
                            np.linalg.norm([roll_cmd, pitch_cmd, yaw_cmd]) < 0.35)
            self.in_recovery[i] = any_fault and recovery_cmd
            if self.in_recovery[i]:
                self.fault_recovery_progress[i] += 1
            else:
                self.fault_recovery_progress[i] = 0
            if any_fault and self.fault_recovery_progress[i] >= 5:
                # A successful reset restores only the degraded subsystem(s);
                # resource/thermal damage remains part of the physical state.
                self.fault_wheel[i] = False
                self.fault_thruster[i] = False
                self.fault_sensor[i] = False
                self.in_recovery[i] = False
                self.fault_recovery_progress[i] = 0
                self.faults_recovered[i] += 1

            # ── Thruster mapping ──────────────────────────────────────────────
            thr = self.config.thruster
            if   thrust_cmd < -0.6: d_v, fuel = -thr.max_dv_per_step,        thr.fuel_cost_full
            elif thrust_cmd < -0.2: d_v, fuel = -thr.max_dv_per_step * 0.25, thr.fuel_cost_light
            elif thrust_cmd <  0.2: d_v, fuel =  0.0,                         0.0
            elif thrust_cmd <  0.6: d_v, fuel =  thr.max_dv_per_step * 0.25, thr.fuel_cost_light
            else:                   d_v, fuel =  thr.max_dv_per_step,         thr.fuel_cost_full

            if self.space_weather_active:
                d_v *= 0.6   # Space weather reduces thruster effectiveness

            if self.agent_fuel[i] >= fuel and not self.in_safe_mode[i]:
                self.agent_vel[i]   += d_v * self._thruster_eff[i]
                self.agent_fuel[i]  -= fuel
                if fuel > 0.0:
                    self.agent_battery[i] -= THRUSTER_POWER_DRAW
                    self.agent_temp[i]    += 2.0   # Thruster heat (Phase F)
            elif fuel > 0.0 and self.agent_fuel[i] < 0.5:
                self.fuel_outs[i] += 1

            # ── Orbital perturbations (J2 + drag) ─────────────────────────────
            orbit    = self.config.orbit
            j2_drift = orbit.j2_strength * np.sin(np.radians(self.agent_pos[i]))
            drag     = -orbit.drag_coeff * np.sign(self.agent_vel[i] - self._nominal_vel)
            self.agent_vel[i] += j2_drift + drag
            self.agent_vel[i]  = float(np.clip(
                self.agent_vel[i], self._nominal_vel - 3.0, self._nominal_vel + 3.0
            ))
            self.agent_pos[i] = (self.agent_pos[i] + self.agent_vel[i]) % 360.0

            # ── 3D Attitude control ───────────────────────────────────────────
            self.agent_attitude_rates[i] += np.array([roll_cmd, pitch_cmd, yaw_cmd]) * 0.5
            self.agent_attitude_rates[i]  = np.clip(self.agent_attitude_rates[i], -10.0, 10.0)
            self.agent_attitude[i]         = (self.agent_attitude[i] + self.agent_attitude_rates[i]) % 360.0
            wheel_power = WHEEL_POWER_DRAW * float(np.sum(np.abs([roll_cmd, pitch_cmd, yaw_cmd])))
            self.agent_battery[i] -= wheel_power

            # ── Phase F: Thermal dynamics ─────────────────────────────────────
            ecl = self._compute_eclipse(self.agent_pos[i])
            self.eclipse_mode[i] = ecl
            if ecl:
                self.agent_battery[i] -= orbit.eclipse_fraction * 2.0
                self.agent_temp[i]    -= 0.8   # Radiative cooling in shadow
            else:
                self.agent_battery[i] += self.config.solar_charge_rate
                self.agent_temp[i]    += 0.4   # Solar heating

            # Attitude control wheels also generate heat
            self.agent_temp[i] += wheel_power * 0.5

            self.agent_battery[i] = float(np.clip(self.agent_battery[i], 0.0, 100.0))
            # Temperature: no hard clip — the env just penalizes and enters safe mode

            # ── Phase D: Ground Station LOS + Data Buffer ─────────────────────
            wp = self._walker_params[i]
            ecef_pos = anomaly_to_ecef(
                self.agent_pos[i],
                orbit.altitude_km,
                wp["inclination_deg"],
                wp["raan_deg"],
                self.earth_rot_deg
            )
            is_los, gs_name = check_ground_station_los(ecef_pos)
            self.gs_blackout[i] = bool(is_los and self.config.station_is_blacked_out(
                self.current_step, gs_name
            ))
            self.gs_los[i]  = is_los and not self.gs_blackout[i]
            self.gs_name[i] = gs_name

            # Evaluate safe-mode before downlink; a thermally/power-disabled
            # spacecraft must not transmit during the step that caused the trip.
            prev_safe = self.in_safe_mode[i]
            low_power = self.agent_battery[i] < SAFE_MODE_THRESHOLD
            over_heat = self.agent_temp[i] > self.config.max_temp_c
            cold_soak = self.agent_temp[i] < self.config.min_temp_c
            self.in_safe_mode[i] = low_power or over_heat or cold_soak

            # Data generation (Phase C missions)
            if self.config.mission_type == MissionType.EARTH_OBS:
                if self.agent_data[i] < self.config.data_capacity_gb:
                    self.agent_data[i] += 5.0

            # Phase D: Downlink when in LOS (ground station blackout = no LOS)
            if self.gs_los[i] and not self.in_safe_mode[i]:
                downlinked = min(self.agent_data[i], 20.0)
                self.agent_data[i] = max(0.0, self.agent_data[i] - downlinked)
                if downlinked > 0:
                    rewards[i] += 0.3   # Reward for successful downlink

            # ── Safe mode (power or thermal) ───────────────────────────────────
            if self.in_safe_mode[i]:
                if not prev_safe:
                    rewards[i] -= 10.0   # Gradient spike on entering safe mode
                else:
                    rewards[i] -= 0.1    # Bleed while stuck in safe mode

            # ── Phase C: Mission-specific rewards ─────────────────────────────
            gap_err = self._gap_error(i)

            if self.config.mission_type == MissionType.COMMS:
                # COMMS: Keep constellation gap tight for ISL connectivity
                if gap_err <= 2.0:
                    rewards[i] += 1.0
                elif gap_err <= 10.0:
                    rewards[i] += 0.8 - (gap_err - 2.0) * 0.05
                elif gap_err <= 30.0:
                    rewards[i] += 0.4 * (1.0 - (gap_err - 10.0) / 20.0)

                # Bonus: coast when already on target (fuel conservation)
                if gap_err <= 10.0 and abs(thrust_cmd) < 0.2:
                    rewards[i] += 0.1

            elif self.config.mission_type == MissionType.EARTH_OBS:
                # EARTH OBS: Maintain strict Nadir pointing (roll=0, pitch=0)
                roll  = self.agent_attitude[i][0]
                pitch = self.agent_attitude[i][1]
                roll_err  = min(roll, 360.0 - roll)
                pitch_err = min(pitch, 360.0 - pitch)
                pointing_err = roll_err + pitch_err

                if pointing_err < 3.0:
                    rewards[i] += 1.0
                elif pointing_err < 10.0:
                    rewards[i] += 0.6
                elif pointing_err < 30.0:
                    rewards[i] += 0.2
                else:
                    rewards[i] -= 0.2

                # Penalize buffer overflow
                if self.agent_data[i] >= self.config.data_capacity_gb:
                    rewards[i] -= 0.5

            elif self.config.mission_type == MissionType.SCIENCE:
                # SCIENCE: Requires a stable spin rate on the Yaw axis
                # (e.g. magnetometer scanning)
                yaw_rate = self.agent_attitude_rates[i][2]
                target_spin = 5.0   # deg/step target spin rate
                spin_err = abs(yaw_rate - target_spin)
                if spin_err < 0.5:
                    rewards[i] += 1.0
                elif spin_err < 2.0:
                    rewards[i] += 0.4
                else:
                    rewards[i] -= 0.1

                # Also needs stable roll/pitch
                roll_rate_mag = abs(self.agent_attitude_rates[i][0])
                pitch_rate_mag = abs(self.agent_attitude_rates[i][1])
                if roll_rate_mag < 0.3 and pitch_rate_mag < 0.3:
                    rewards[i] += 0.1

            # ── Phase E: Fault recovery bonus ─────────────────────────────────
            if any_fault and self.in_recovery[i]:
                # Reward conservative behaviour while faulted
                rewards[i] += 0.2
                # Extra reward if we are also in a good pointing state despite fault
                if self.config.mission_type == MissionType.EARTH_OBS:
                    roll  = self.agent_attitude[i][0]
                    pitch = self.agent_attitude[i][1]
                    if min(roll, 360.0 - roll) + min(pitch, 360.0 - pitch) < 15.0:
                        rewards[i] += 0.3

            # ── Phase F: Thermal margin reward ────────────────────────────────
            # Reward staying in the safe thermal zone (not just "not dead")
            temp_norm = (self.agent_temp[i] - self.config.min_temp_c) / (
                self.config.max_temp_c - self.config.min_temp_c + 1e-6
            )
            if 0.25 < temp_norm < 0.75:   # Sweet spot in middle of range
                rewards[i] += 0.05

            # ── Debris collision penalty (Phase 3+) ───────────────────────────
            for d in self.debris:
                prox = min(
                    abs(self.agent_pos[i] - d["pos"]),
                    360.0 - abs(self.agent_pos[i] - d["pos"])
                )
                if prox < 2.0:
                    rewards[i] -= 10.0
                    self.collisions[i] += 1
                elif prox < 5.0:
                    rewards[i] -= 0.1

            self.reward_history[i].append(float(rewards[i]))

        term  = False
        trunc = self.current_step >= self.max_steps
        info = {
            "ground_station_blackouts": self.gs_blackout.copy(),
            "faults_recovered": self.faults_recovered.copy(),
        }
        return self._get_obs_list(), rewards, term, trunc, info

    # ── Observations ──────────────────────────────────────────────────────────
    def _get_obs_list(self) -> list[dict]:
        obs_list   = []
        config_vec = self.config.to_obs_vector()   # 6-dim

        # Fleet-wide global stats (6 dims)
        global_stats = np.array([
            float(np.mean(self.agent_fuel))    / 100.0,
            float(np.mean(self.agent_battery)) / 100.0,
            float(np.var(self.agent_pos))      / (360.0 ** 2),  # normalised variance
            float(np.var(self.agent_vel)),
            float(np.mean(self.agent_temp))    / 100.0,
            float(np.mean(self.agent_data))    / (self.config.data_capacity_gb + 1e-6),
        ], dtype=np.float32)

        for i in range(self.num_satellites):
            neighbors = self._get_neighbors(i)

            # Debris positions (3 slots, normalised)
            deb = np.zeros(3, dtype=np.float32)
            for d_idx, d in enumerate(self.debris[:3]):
                deb[d_idx] = float(d["pos"]) / 360.0

            # Phase E: Apply sensor noise to position/velocity if sensor fault
            obs_pos = float(self.agent_pos[i])
            obs_vel = float(self.agent_vel[i])
            if self.fault_sensor[i]:
                obs_pos += float(self.np_random.normal(0, 5.0))
                obs_vel += float(self.np_random.normal(0, 1.0))

            gap_err      = self._gap_error(i)
            closing_rate = obs_vel - self._nominal_vel

            # ── Base LOCAL obs — 16 dims ──────────────────────────────────────
            # Note: NO duplicate closing_rate. Count carefully.
            base_local = np.array([
                (obs_pos % 360.0) / 360.0,           # 1: normalised position
                np.clip(closing_rate / 3.0, -1, 1),  # 2: velocity error (normed)
                self.agent_fuel[i] / 100.0,           # 3: fuel remaining
                self.agent_battery[i] / 100.0,        # 4: battery level
                gap_err / 180.0,                      # 5: gap error
                deb[0], deb[1], deb[2],               # 6-8: debris positions
                float(self.eclipse_mode[i]),           # 9: eclipse flag
                float(self.space_weather_active),      # 10: space weather flag
                np.clip(self.agent_temp[i] / 100.0, -2.0, 2.0),  # 11: temperature
                np.clip(self.agent_data[i] / (self.config.data_capacity_gb + 1e-6), 0.0, 1.0),  # 12: data buffer
                float(self.fault_wheel[i]),            # 13: wheel fault
                float(self.fault_thruster[i]),         # 14: thruster fault
                float(self.fault_sensor[i]),           # 15: sensor fault
                float(self.gs_los[i]),                 # 16: ground station LOS
            ], dtype=np.float32)

            # ── Attitude obs — 6 dims ─────────────────────────────────────────
            att = np.array([
                *[((a % 360.0) / 180.0 - 1.0) for a in self.agent_attitude[i]],
                *np.clip(self.agent_attitude_rates[i] / 10.0, -1.0, 1.0),
            ], dtype=np.float32)

            # ── Neighbor obs — 20 dims ────────────────────────────────────────
            neigh_feat = []
            for n_idx in neighbors:
                rel_pos = ((self.agent_pos[n_idx] - obs_pos) % 360.0) / 180.0 - 1.0
                rel_vel = np.clip((self.agent_vel[n_idx] - obs_vel) / 3.0, -1.0, 1.0)
                neigh_feat.extend([
                    rel_pos,
                    rel_vel,
                    self.agent_fuel[n_idx] / 100.0,
                    self.agent_battery[n_idx] / 100.0,
                    1.0,   # ISL active flag
                ])
            while len(neigh_feat) < NEIGHBOR_DIM:
                neigh_feat.extend([0.0] * 5)

            # ── Recovery flag — 1 dim ─────────────────────────────────────────
            recovery_flag = np.array([float(self.in_recovery[i])], dtype=np.float32)

            # ── Assemble LOCAL obs — 16 + 6 + 20 + 6 + 1 = 49 dims ───────────
            local_obs = np.concatenate([
                base_local, att, neigh_feat, config_vec, recovery_flag
            ]).astype(np.float32)

            # Sanity check (development only — remove for production)
            assert local_obs.shape == (LOCAL_DIM_BASE,), (
                f"LOCAL dim mismatch: expected {LOCAL_DIM_BASE}, got {local_obs.shape[0]}"
            )

            # ── Global obs — 6 + 24 + 6 + 3 + 2 + 6 = 47 dims ───────────────
            own_feat = np.array([
                self.agent_pos[i] / 360.0,
                np.clip(closing_rate / 3.0, -1.0, 1.0),
                self.agent_fuel[i] / 100.0,
                self.agent_battery[i] / 100.0,
                np.clip(self.agent_temp[i] / 100.0, -2.0, 2.0),
                np.clip(self.agent_data[i] / (self.config.data_capacity_gb + 1e-6), 0.0, 1.0),
            ], dtype=np.float32)

            n_global = []
            for n_idx in neighbors:
                n_closing = self.agent_vel[n_idx] - self._nominal_vel
                n_global.extend([
                    self.agent_pos[n_idx] / 360.0,
                    np.clip(n_closing / 3.0, -1.0, 1.0),
                    self.agent_fuel[n_idx] / 100.0,
                    self.agent_battery[n_idx] / 100.0,
                    np.clip(self.agent_temp[n_idx] / 100.0, -2.0, 2.0),
                    np.clip(self.agent_data[n_idx] / (self.config.data_capacity_gb + 1e-6), 0.0, 1.0),
                ])
            while len(n_global) < 6 * MAX_NEIGHBORS:
                n_global.extend([0.0] * 6)

            weather_eclipse = np.array([
                float(self.space_weather_active),
                float(np.mean(self.eclipse_mode)),
            ], dtype=np.float32)

            global_obs = np.concatenate([
                own_feat, n_global, global_stats, deb, weather_eclipse, config_vec
            ]).astype(np.float32)

            assert global_obs.shape == (GLOBAL_DIM,), (
                f"GLOBAL dim mismatch: expected {GLOBAL_DIM}, got {global_obs.shape[0]}"
            )

            obs_list.append({"local": local_obs, "global": global_obs})

        return obs_list


# Import to ensure they are accessible from satellite_env for train.py
EARTH_RADIUS_KM = 6371.0
EARTH_MU        = 398600.4418
