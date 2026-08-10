"""
fsw/ai_brain/adapter.py
-----------------------
Observation adapter for the AI policy.
Decouples the FSW from the simulation's private `_get_obs_list()` method.
Constructs the identical 49-dim Local and 47-dim Global feature vectors 
using strongly typed `SubsystemState` and `ISLNeighborState`.
"""

import numpy as np
from typing import Optional
from fsw.core.telemetry import SubsystemState
from simulation.sat_config import SatelliteConfig, PRESETS

class AIObservationAdapter:
    def __init__(self, config: Optional[SatelliteConfig] = None,
                 config_name: str = "starlink_leo", nominal_vel: float = 0.0):
        # We need the static config to emit the 6-dim config vector
        self.config: SatelliteConfig = config or PRESETS.get(config_name, PRESETS["starlink_leo"])
        self.config_vec = self.config.to_obs_vector()
        
        # We need nominal_vel to calculate closing rates.
        # In actual flight, this is computed dynamically based on current orbit altitude.
        self.nominal_vel = nominal_vel
        self.max_neighbors = 4

    def build_observation(self, state: SubsystemState, is_in_recovery: bool) -> dict:
        """
        Builds the Dict space for SB3 inference.
        Returns:
            {"local": ndarray(49), "global": ndarray(47)}
        Note: The 4-dim memory context must be appended by the agent if required.
        """
        
        # 1. Base Local (16 dims)
        obs_pos = state.adcs.position_eci_km[0] / 100.0 # Mock un-mapping from anomaly
        obs_vel = state.adcs.velocity_eci_kms[0]
        closing_rate = obs_vel - self.nominal_vel
        
        # Mock eclipse based on current sun state
        eclipse_flag = 1.0 if state.eps.solar_power_w == 0.0 else 0.0
        
        deb = np.zeros(3, dtype=np.float32)
        for i, d_pos in enumerate(state.global_fleet.debris_positions[:3]):
            deb[i] = d_pos / 360.0

        base_local = np.array([
            (obs_pos % 360.0) / 360.0,
            np.clip(closing_rate / 3.0, -1, 1),
            np.clip(state.adcs.fuel_percent / 100.0, 0.0, 1.0),
            state.eps.battery_charge_percent / 100.0,
            0.0, # Gap error (mocked 0 for standalone FSW logic)
            deb[0], deb[1], deb[2],
            eclipse_flag,
            float(state.global_fleet.space_weather_active),
            np.clip(state.thermal.battery_temp_c / 100.0, -2.0, 2.0),
            np.clip(state.comms.data_buffer_gb / (self.config.data_capacity_gb + 1e-6), 0.0, 1.0),
            float(state.faults.wheel_fault),
            float(state.faults.thruster_fault),
            float(state.faults.sensor_fault),
            float(state.comms.has_ground_los),
        ], dtype=np.float32)

        # 2. Attitude (6 dims)
        att = np.array([
            *[((a % 360.0) / 180.0 - 1.0) for a in state.adcs.attitude_deg],
            *np.clip(np.array(state.adcs.rates_deg_s) / 10.0, -1.0, 1.0),
        ], dtype=np.float32)

        # 3. Neighbors (20 dims)
        neigh_feat = []
        for n in state.neighbors[:self.max_neighbors]:
            n_pos = n.position_eci_km[0] / 100.0
            n_vel = n.velocity_eci_kms[0]
            rel_pos = ((n_pos - obs_pos) % 360.0) / 180.0 - 1.0
            rel_vel = np.clip((n_vel - obs_vel) / 3.0, -1.0, 1.0)
            
            neigh_feat.extend([
                rel_pos,
                rel_vel,
                n.fuel_percent / 100.0,
                n.battery_charge_percent / 100.0,
                float(n.isl_active),
            ])
        while len(neigh_feat) < (self.max_neighbors * 5):
            neigh_feat.extend([0.0] * 5)

        # 4. Recovery Flag (1 dim)
        recovery_flag = np.array([float(is_in_recovery)], dtype=np.float32)

        # LOCAL: 16 + 6 + 20 + 6 + 1 = 49
        local_obs = np.concatenate([
            base_local, att, neigh_feat, self.config_vec, recovery_flag
        ]).astype(np.float32)


        # 5. Own Features (6 dims)
        own_feat = np.array([
            (obs_pos % 360.0) / 360.0,
            np.clip(closing_rate / 3.0, -1.0, 1.0),
            np.clip(state.adcs.fuel_percent / 100.0, 0.0, 1.0),
            state.eps.battery_charge_percent / 100.0,
            np.clip(state.thermal.battery_temp_c / 100.0, -2.0, 2.0),
            np.clip(state.comms.data_buffer_gb / (self.config.data_capacity_gb + 1e-6), 0.0, 1.0),
        ], dtype=np.float32)

        # 6. Global Neighbors (24 dims)
        n_global = []
        for n in state.neighbors[:self.max_neighbors]:
            n_pos = n.position_eci_km[0] / 100.0
            n_vel = n.velocity_eci_kms[0]
            n_closing = n_vel - self.nominal_vel
            n_global.extend([
                (n_pos % 360.0) / 360.0,
                np.clip(n_closing / 3.0, -1.0, 1.0),
                n.fuel_percent / 100.0,
                n.battery_charge_percent / 100.0,
                0.0, # neighbor temp not shared in ISL mock currently
                0.0, # neighbor data not shared in ISL mock currently
            ])
        while len(n_global) < (self.max_neighbors * 6):
            n_global.extend([0.0] * 6)

        # 7. Global Stats (6 dims)
        global_stats = np.array([
            state.global_fleet.mean_fuel_percent / 100.0,
            state.global_fleet.mean_battery_percent / 100.0,
            state.global_fleet.pos_variance,
            state.global_fleet.vel_variance,
            state.global_fleet.mean_temp_c / 100.0,
            state.global_fleet.mean_data_gb / (self.config.data_capacity_gb + 1e-6),
        ], dtype=np.float32)

        # 8. Weather / Eclipse (2 dims)
        weather_eclipse = np.array([
            float(state.global_fleet.space_weather_active),
            state.global_fleet.eclipse_fraction,
        ], dtype=np.float32)

        # GLOBAL: 6 + 24 + 6 + 3 + 2 + 6 = 47
        global_obs = np.concatenate([
            own_feat, n_global, global_stats, deb, weather_eclipse, self.config_vec
        ]).astype(np.float32)

        if local_obs.shape != (49,) or global_obs.shape != (47,):
            raise ValueError("AI observation schema mismatch")
        return {"local": local_obs, "global": global_obs}
