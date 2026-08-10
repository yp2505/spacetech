"""
fsw/core/telemetry.py
---------------------
Structured telemetry models with explicit SI units, freshness timestamps,
and validation checks for the flight software.
"""

from dataclasses import dataclass, field
import time

@dataclass
class TelemetryPoint:
    """A telemetry sample stamped by a monotonic clock and a source sequence."""
    timestamp: float = field(default_factory=time.monotonic)
    sequence: int = 0
    valid: bool = True
    
    @property
    def is_stale(self) -> bool:
        """Returns True if telemetry is invalid or older than five seconds."""
        return not self.valid or (time.monotonic() - self.timestamp) > 5.0

@dataclass
class EPSTelemetry(TelemetryPoint):
    battery_charge_percent: float = 0.0
    battery_voltage_v: float = 0.0
    solar_power_w: float = 0.0
    system_draw_w: float = 0.0

@dataclass
class ThermalTelemetry(TelemetryPoint):
    battery_temp_c: float = 0.0
    obc_temp_c: float = 0.0
    payload_temp_c: float = 0.0

@dataclass
class ADCSTelemetry(TelemetryPoint):
    # ECI frame
    position_eci_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_eci_kms: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Body frame (Roll, Pitch, Yaw in degrees for simplicity in this sim)
    attitude_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rates_deg_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fuel_percent: float = 0.0

@dataclass
class CommsTelemetry(TelemetryPoint):
    has_ground_los: bool = False
    active_station_name: str = ""
    data_buffer_gb: float = 0.0

@dataclass
class FaultTelemetry(TelemetryPoint):
    wheel_fault: bool = False
    thruster_fault: bool = False
    sensor_fault: bool = False

@dataclass
class ISLNeighborState:
    """Inter-Satellite Link telemetry from a neighbor."""
    position_eci_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_eci_kms: tuple[float, float, float] = (0.0, 0.0, 0.0)
    battery_charge_percent: float = 0.0
    fuel_percent: float = 0.0
    isl_active: bool = False

@dataclass
class GlobalFleetState(TelemetryPoint):
    """Aggregate fleet telemetry, typically uplinked or derived via ISL mesh."""
    mean_fuel_percent: float = 0.0
    mean_battery_percent: float = 0.0
    mean_temp_c: float = 0.0
    mean_data_gb: float = 0.0
    pos_variance: float = 0.0
    vel_variance: float = 0.0
    # Global environment
    debris_positions: list[float] = field(default_factory=list)
    space_weather_active: bool = False
    eclipse_fraction: float = 0.0

@dataclass
class SubsystemState:
    """The complete spacecraft state at a single point in time."""
    eps: EPSTelemetry = field(default_factory=EPSTelemetry)
    thermal: ThermalTelemetry = field(default_factory=ThermalTelemetry)
    adcs: ADCSTelemetry = field(default_factory=ADCSTelemetry)
    comms: CommsTelemetry = field(default_factory=CommsTelemetry)
    faults: FaultTelemetry = field(default_factory=FaultTelemetry)
    
    # ISL/Network additions for the AI observation adapter
    neighbors: list[ISLNeighborState] = field(default_factory=list)
    global_fleet: GlobalFleetState = field(default_factory=GlobalFleetState)
    
    @property
    def any_stale(self) -> bool:
        return (self.eps.is_stale or self.thermal.is_stale or 
                self.adcs.is_stale or self.comms.is_stale or 
                self.faults.is_stale or self.global_fleet.is_stale)
