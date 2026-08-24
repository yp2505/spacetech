"""
fsw/hal/sim_backend.py
----------------------
Simulation backend for the HAL.
This connects the FSW to the `MultiSatelliteEnv` for offline testing.
It extracts telemetry from the environment's internal arrays.
"""

from fsw.hal.interfaces import AbstractEPS, AbstractThermal, AbstractADCS, AbstractRadio, AbstractFaultMonitor
from fsw.core.telemetry import (
    EPSTelemetry, ThermalTelemetry, ADCSTelemetry, CommsTelemetry, FaultTelemetry,
    GlobalFleetState, ISLNeighborState, SubsystemState,
)
from fsw.core.commands import ActuatorCommand
from fsw.hal.isl_mesh import ISLMeshNetwork, ISLPacketCrypto, ISLMeshNode
import time
import numpy as np
import base64

class SimEPS(AbstractEPS):
    def __init__(self, sim_env, agent_idx: int):
        self.env = sim_env
        self.idx = agent_idx

    def get_telemetry(self) -> EPSTelemetry:
        return EPSTelemetry(
            battery_charge_percent=self.env.agent_battery[self.idx],
            battery_voltage_v=24.0 + (self.env.agent_battery[self.idx] / 100.0) * 4.0, # fake voltage curve
            solar_power_w=self.env.config.solar_charge_rate * 10.0 if not self.env.eclipse_mode[self.idx] else 0.0,
            system_draw_w=15.0 # base draw
        )

class SimThermal(AbstractThermal):
    def __init__(self, sim_env, agent_idx: int):
        self.env = sim_env
        self.idx = agent_idx

    def get_telemetry(self) -> ThermalTelemetry:
        t = self.env.agent_temp[self.idx]
        return ThermalTelemetry(battery_temp_c=t, obc_temp_c=t-2.0, payload_temp_c=t+5.0)

class SimADCS(AbstractADCS):
    def __init__(self, sim_env, agent_idx: int):
        self.env = sim_env
        self.idx = agent_idx
        # We store the requested command so the main sim loop can pick it up
        self.last_command = [0.0] * 8

    def get_telemetry(self) -> ADCSTelemetry:
        # In sim, agent_pos is just anomaly degrees. We'll map to mock ECI/Body
        pos_deg = self.env.agent_pos[self.idx]
        vel = self.env.agent_vel[self.idx]
        att = self.env.agent_attitude[self.idx]
        rates = self.env.agent_attitude_rates[self.idx]
        
        return ADCSTelemetry(
            position_eci_km=(pos_deg * 100.0, 0.0, 0.0), # X encodes anomaly for mock
            velocity_eci_kms=(vel, 0.0, 0.0), # X encodes velocity
            attitude_deg=tuple(att),
            rates_deg_s=tuple(rates),
            delta_v_remaining=float(self.env.agent_delta_v[self.idx]) if hasattr(self.env, "agent_delta_v") else 1000.0,
            fuel_percent=float(self.env.agent_fuel[self.idx]),
        )

    def enrich_state(self, state: SubsystemState) -> None:
        """Populate optional fleet telemetry for SIL without exposing private AI APIs."""
        # We no longer overwrite state.neighbors here; SimRadio.get_neighbor_telemetry provides it with link quality and delay simulation.
        state.global_fleet = GlobalFleetState(
            mean_fuel_percent=float(self.env.agent_fuel.mean()),
            mean_battery_percent=float(self.env.agent_battery.mean()),
            mean_temp_c=float(self.env.agent_temp.mean()),
            mean_data_gb=float(self.env.agent_data.mean()),
            pos_variance=float(self.env.agent_pos.var()) / (360.0 ** 2),
            vel_variance=float(self.env.agent_vel.var()),
            debris_positions=[float(d.get("nu", 0.0)) for d in self.env.debris],
            space_weather_active=bool(self.env.space_weather_active),
            eclipse_fraction=float(self.env.eclipse_mode.mean()),
        )
        if hasattr(self.env, "commander_goal"):
            state.commander_goal = tuple(float(x) for x in self.env.commander_goal[self.idx])

    def command_actuators(self, cmd: ActuatorCommand) -> bool:
        self.last_command = [
            cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq,
            cmd.relay_action, cmd.hohmann, cmd.avoidance, cmd.deorbit
        ]
        return True

class SimRadio(AbstractRadio):
    def __init__(self, sim_env, agent_idx: int, isl_mesh: ISLMeshNetwork = None):
        self.env = sim_env
        self.idx = agent_idx
        self.isl_mesh = isl_mesh
        self.mesh_node = isl_mesh.get_node(agent_idx) if isl_mesh else None
        
        # Legacy transmission buffer for backward compatibility
        if not hasattr(self, 'transmission_buffer'):
            self.transmission_buffer = {}

    def get_telemetry(self) -> CommsTelemetry:
        return CommsTelemetry(
            has_ground_los=self.env.gs_los[self.idx],
            active_station_name=self.env.gs_name[self.idx],
            data_buffer_gb=self.env.agent_data[self.idx]
        )
        
    def downlink_data(self, amount_gb: float) -> bool:
        if self.env.gs_los[self.idx]:
            return True
        return False
        
    def get_global_telemetry(self) -> GlobalFleetState:
        """Simulates ISL mesh gathering global fleet statistics."""
        import numpy as np
        return GlobalFleetState(
            mean_fuel_percent=float(np.mean(self.env.agent_fuel)),
            mean_battery_percent=float(np.mean(self.env.agent_battery)),
            mean_temp_c=float(np.mean(self.env.agent_temp)),
            mean_data_gb=float(np.mean(self.env.agent_data)),
            pos_variance=float(np.var(self.env.agent_pos)) / (360.0**2),
            vel_variance=float(np.var(self.env.agent_vel)),
            debris_positions=[float(d.get("nu", 0.0)) for d in self.env.debris],
            space_weather_active=self.env.space_weather_active,
            eclipse_fraction=float(np.mean(self.env.eclipse_mode))
        )
        
    def get_neighbor_telemetry(self) -> list[ISLNeighborState]:
        """Simulates ISL mesh gathering neighbor states with realistic delay and encryption."""
        if self.mesh_node is None:
            return self._legacy_get_neighbor_telemetry()
        
        # Use mesh network for realistic ISL
        neighbors = []
        link_states = self.mesh_node.get_link_states()
        
        # Update mesh with current link qualities
        neighbor_indices = self.env._get_neighbors(self.idx)
        for n_idx in neighbor_indices:
            diff = abs(self.env.agent_pos[self.idx] - self.env.agent_pos[n_idx])
            angular_dist = min(diff, 360.0 - diff)
            link_quality = max(0.0, 1.0 - (angular_dist / 45.0))
            self.mesh_node.update_link_quality(n_idx, link_quality)
        
        # Get received packets from mesh
        received_packets = self.mesh_node.get_received_packets()
        
        for n_idx in neighbor_indices:
            link = link_states.get(n_idx)
            link_quality = link.link_quality if link else 0.0
            is_active = link.is_active if link else False
            
            # Simulated data extraction
            temp = float(self.env.agent_temp[n_idx])
            data_vol = float(self.env.agent_data[n_idx])
            task_queue = float(len(self.env.targets)) if hasattr(self.env, 'targets') else 0.0
            
            # Get last episode reward from shared memory
            reward = 0.0
            ep_mem = getattr(self.env, "episodic_memory", None)
            if ep_mem is not None:
                if isinstance(ep_mem, dict):
                    episodes = ep_mem.get(n_idx, [])
                    if episodes:
                        last = episodes[-1]
                        reward = float(last.get("total_reward", 0.0)) if isinstance(last, dict) else float(getattr(last, "total_reward", 0.0))
                elif hasattr(ep_mem, "get_last_reward"):
                    reward = float(ep_mem.get_last_reward(n_idx))
            
            # Encrypted payload using mesh crypto
            raw_payload = f"SAT_{n_idx}_DATA".encode()
            # Use the mesh node's key manager for compatibility
            key = self.mesh_node.key_manager.derive_key(self.idx, n_idx)
            seq = link.tx_seq if link else 0
            encrypted = self.mesh_node.key_manager.encrypt(key, raw_payload, seq)
            b64_enc = base64.b64encode(encrypted).decode()
            
            state = ISLNeighborState(
                position_eci_km=(float(self.env.agent_pos[n_idx])*100.0, 0.0, 0.0),
                velocity_eci_kms=(float(self.env.agent_vel[n_idx]), 0.0, 0.0),
                battery_charge_percent=float(self.env.agent_battery[n_idx]),
                fuel_percent=float(self.env.agent_fuel[n_idx]),
                isl_active=is_active,
                neighbor_temp_c=temp,
                data_buffer_gb=data_vol,
                task_queue_size=task_queue,
                last_episode_reward=reward,
                link_quality=link_quality,
                encrypted_payload=b64_enc
            )
            neighbors.append(state)
        
        return neighbors
    
    def _legacy_get_neighbor_telemetry(self) -> list[ISLNeighborState]:
        """Legacy implementation for backward compatibility."""
        neighbors = []
        neighbor_indices = self.env._get_neighbors(self.idx)
        for n_idx in neighbor_indices:
            diff = abs(self.env.agent_pos[self.idx] - self.env.agent_pos[n_idx])
            angular_dist = min(diff, 360.0 - diff)
            link_quality = max(0.0, 1.0 - (angular_dist / 45.0))
            
            raw_payload = f"SAT_{n_idx}_DATA".encode()
            key = 0xAA
            encrypted = bytearray([b ^ key for b in raw_payload])
            b64_enc = base64.b64encode(encrypted).decode()
            
            temp = float(self.env.agent_temp[n_idx])
            data_vol = float(self.env.agent_data[n_idx])
            task_queue = float(len(self.env.targets)) if hasattr(self.env, 'targets') else 0.0
            
            reward = 0.0
            ep_mem = getattr(self.env, "episodic_memory", None)
            if ep_mem is not None:
                if isinstance(ep_mem, dict):
                    episodes = ep_mem.get(n_idx, [])
                    if episodes:
                        last = episodes[-1]
                        reward = float(last.get("total_reward", 0.0)) if isinstance(last, dict) else float(getattr(last, "total_reward", 0.0))
                elif hasattr(ep_mem, "get_last_reward"):
                    reward = float(ep_mem.get_last_reward(n_idx))
            
            state = ISLNeighborState(
                position_eci_km=(float(self.env.agent_pos[n_idx])*100.0, 0.0, 0.0),
                velocity_eci_kms=(float(self.env.agent_vel[n_idx]), 0.0, 0.0),
                battery_charge_percent=float(self.env.agent_battery[n_idx]),
                fuel_percent=float(self.env.agent_fuel[n_idx]),
                isl_active=link_quality > 0.1,
                neighbor_temp_c=temp,
                data_buffer_gb=data_vol,
                task_queue_size=task_queue,
                last_episode_reward=reward,
                link_quality=link_quality,
                encrypted_payload=b64_enc
            )
            neighbors.append(state)
        return neighbors

class SimFaultMonitor(AbstractFaultMonitor):
    def __init__(self, sim_env, agent_idx: int):
        self.env = sim_env
        self.idx = agent_idx

    def get_telemetry(self) -> FaultTelemetry:
        return FaultTelemetry(
            wheel_fault=self.env.fault_wheel[self.idx],
            thruster_fault=self.env.fault_thruster[self.idx],
            sensor_fault=self.env.fault_sensor[self.idx]
        )
