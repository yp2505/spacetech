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
import time

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
        self.last_command = [0.0, 0.0, 0.0, 0.0]

    def get_telemetry(self) -> ADCSTelemetry:
        # In sim, agent_pos is just anomaly degrees. We'll map to mock ECI/Body
        pos_deg = self.env.agent_pos[self.idx]
        vel = self.env.agent_vel[self.idx]
        att = self.env.agent_attitude[self.idx]
        rates = self.env.agent_attitude_rates[self.idx]
        
        return ADCSTelemetry(
            position_eci_km=(pos_deg * 100.0, 0.0, 0.0), # mock
            velocity_eci_kms=(vel, 0.0, 0.0),
            attitude_deg=tuple(att),
            rates_deg_s=tuple(rates),
            fuel_percent=float(self.env.agent_fuel[self.idx]),
        )

    def enrich_state(self, state: SubsystemState) -> None:
        """Populate optional fleet telemetry for SIL without exposing private AI APIs."""
        neighbors = self.env._get_neighbors(self.idx)
        state.neighbors = [
            ISLNeighborState(
                position_eci_km=(float(self.env.agent_pos[j]) * 100.0, 0.0, 0.0),
                velocity_eci_kms=(float(self.env.agent_vel[j]), 0.0, 0.0),
                battery_charge_percent=float(self.env.agent_battery[j]),
                fuel_percent=float(self.env.agent_fuel[j]),
                isl_active=True,
            )
            for j in neighbors
        ]
        state.global_fleet = GlobalFleetState(
            mean_fuel_percent=float(self.env.agent_fuel.mean()),
            mean_battery_percent=float(self.env.agent_battery.mean()),
            mean_temp_c=float(self.env.agent_temp.mean()),
            mean_data_gb=float(self.env.agent_data.mean()),
            pos_variance=float(self.env.agent_pos.var()) / (360.0 ** 2),
            vel_variance=float(self.env.agent_vel.var()),
            debris_positions=[float(d["pos"]) for d in self.env.debris],
            space_weather_active=bool(self.env.space_weather_active),
            eclipse_fraction=float(self.env.eclipse_mode.mean()),
        )

    def command_actuators(self, cmd: ActuatorCommand) -> bool:
        self.last_command = [cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq]
        return True

class SimRadio(AbstractRadio):
    def __init__(self, sim_env, agent_idx: int):
        self.env = sim_env
        self.idx = agent_idx

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
            debris_positions=[float(d["pos"]) for d in self.env.debris],
            space_weather_active=self.env.space_weather_active,
            eclipse_fraction=float(np.mean(self.env.eclipse_mode))
        )
        
    def get_neighbor_telemetry(self) -> list[ISLNeighborState]:
        """Simulates ISL mesh gathering neighbor states."""
        neighbors = []
        # Get nearest neighbors using sim's helper
        neighbor_indices = self.env._get_neighbors(self.idx)
        for n_idx in neighbor_indices:
            neighbors.append(ISLNeighborState(
                position_eci_km=(self.env.agent_pos[n_idx]*100.0, 0.0, 0.0),
                velocity_eci_kms=(self.env.agent_vel[n_idx], 0.0, 0.0),
                battery_charge_percent=self.env.agent_battery[n_idx],
                fuel_percent=self.env.agent_fuel[n_idx],
                isl_active=True
            ))
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
