"""
sat_config.py
=============
Phase C-F: Universal Configuration Engine with Multi-Mission, Thermal, 
and Multi-Plane support.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Sequence
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────────────────────────

class OrbitType(Enum):
    LEO = "leo"   # Low Earth Orbit:      200 – 2,000 km
    MEO = "meo"   # Medium Earth Orbit: 2,000 – 35,786 km
    GEO = "geo"   # Geostationary Orbit:       35,786 km


class ThrusterType(Enum):
    COLD_GAS = "cold_gas"
    CHEMICAL = "chemical"
    ION      = "ion"
    HALL     = "hall"


class MissionType(Enum):
    EARTH_OBS = "earth_obs"    # Requires strict nadir pointing
    COMMS     = "comms"        # Requires LOS to ground & ISL
    SCIENCE   = "science"      # Requires specific off-axis pointing or spin


# ─────────────────────────────────────────────────────────────────────────────
#  Orbit Physics Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrbitParams:
    altitude_km:      float
    period_min:       float
    step_seconds:     float
    j2_strength:      float
    drag_coeff:       float
    eclipse_fraction: float
    eclipse_arc_deg:  float


@dataclass(frozen=True)
class TLEElements:
    """The orbital elements used by the simulator, parsed from a NORAD TLE."""

    name: str
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    argument_of_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_per_day: float


def parse_tle(lines: str | Sequence[str]) -> TLEElements:
    """Parse and validate a two- or three-line NORAD TLE without network access.

    The simulator currently uses a circular-orbit approximation, so eccentricity
    and argument of perigee are retained for traceability while mean motion,
    inclination, RAAN, and mean anomaly drive the configured initial orbit.
    """
    raw = [line.rstrip("\n") for line in (lines.splitlines() if isinstance(lines, str) else lines)
           if line.strip()]
    if len(raw) == 2:
        name, line1, line2 = "TLE satellite", raw[0], raw[1]
    elif len(raw) == 3:
        name, line1, line2 = raw
    else:
        raise ValueError("A TLE must contain exactly two element lines, optionally preceded by a name.")
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("Invalid TLE line identifiers.")
    for line in (line1, line2):
        if len(line) < 69 or not line[-1].isdigit():
            raise ValueError("Malformed TLE line.")
        checksum = sum((int(c) if c.isdigit() else 1 if c == "-" else 0) for c in line[:68]) % 10
        if checksum != int(line[68]):
            raise ValueError("TLE checksum validation failed.")
    if line1[2:7] != line2[2:7]:
        raise ValueError("TLE satellite numbers do not match.")
    try:
        return TLEElements(
            name=name.strip() or "TLE satellite",
            inclination_deg=float(line2[8:16]),
            raan_deg=float(line2[17:25]),
            eccentricity=float(f"0.{line2[26:33].strip()}"),
            argument_of_perigee_deg=float(line2[34:42]),
            mean_anomaly_deg=float(line2[43:51]),
            mean_motion_rev_per_day=float(line2[52:63]),
        )
    except ValueError as exc:
        raise ValueError("Malformed numeric field in TLE.") from exc


def orbit_params_from_tle(tle: TLEElements, step_seconds: Optional[float] = None) -> OrbitParams:
    """Derive circular-orbit simulation parameters from a TLE mean motion."""
    if tle.mean_motion_rev_per_day <= 0:
        raise ValueError("TLE mean motion must be positive.")
    mu = 398600.4418
    earth_radius = 6371.0
    period_min = 1440.0 / tle.mean_motion_rev_per_day
    semi_major_km = (mu * (period_min * 60.0 / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)
    altitude_km = semi_major_km - earth_radius
    if altitude_km <= 0:
        raise ValueError("TLE describes an orbit below Earth's surface.")
    # Retain the existing regime-specific perturbation model but use real timing.
    orbit_type = OrbitType.LEO if altitude_km < 2_000 else OrbitType.MEO if altitude_km < 35_000 else OrbitType.GEO
    base = ORBIT_PARAMS[orbit_type]
    return OrbitParams(altitude_km, period_min, step_seconds or base.step_seconds,
                       base.j2_strength, base.drag_coeff, base.eclipse_fraction,
                       base.eclipse_arc_deg)


ORBIT_PARAMS: dict[OrbitType, OrbitParams] = {
    OrbitType.LEO: OrbitParams(
        altitude_km=550.0, period_min=95.6, step_seconds=15.9,
        j2_strength=0.035, drag_coeff=0.008, eclipse_fraction=0.37, eclipse_arc_deg=36.0,
    ),
    OrbitType.MEO: OrbitParams(
        altitude_km=20_200.0, period_min=718.7, step_seconds=119.8,
        j2_strength=0.005, drag_coeff=0.0001, eclipse_fraction=0.28, eclipse_arc_deg=28.0,
    ),
    OrbitType.GEO: OrbitParams(
        altitude_km=35_786.0, period_min=1_440.0, step_seconds=240.0,
        j2_strength=0.0005, drag_coeff=0.0, eclipse_fraction=0.05, eclipse_arc_deg=8.7,
    ),
}


@dataclass(frozen=True)
class ThrusterParams:
    max_dv_per_step:  float
    specific_impulse: float
    fuel_cost_full:   float
    fuel_cost_light:  float


THRUSTER_PARAMS: dict[ThrusterType, ThrusterParams] = {
    ThrusterType.COLD_GAS: ThrusterParams(0.4,  70,   0.30, 0.08),
    ThrusterType.CHEMICAL: ThrusterParams(1.5,  300,  0.50, 0.12),
    ThrusterType.ION:      ThrusterParams(0.15, 3000, 0.04, 0.01),
    ThrusterType.HALL:     ThrusterParams(0.6,  1500, 0.12, 0.03),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Satellite Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SatelliteConfig:
    name:             str
    mission_type:     MissionType
    orbit_type:       OrbitType
    thruster_type:    ThrusterType
    mass_kg:          float
    battery_wh:       float
    solar_area_m2:    float
    num_satellites:   int   = 10
    target_gap_deg:   float = None
    inclination_deg:  float = 53.0
    
    # Phase F: Multi-plane Constellation topology (Walker Delta style)
    planes:           int   = 1
    sats_per_plane:   int   = None
    
    # Phase F: Thermal limits (°C)
    min_temp_c:       float = -40.0
    max_temp_c:       float = 85.0
    
    # Phase D: Data capacity (Gigabytes)
    data_capacity_gb: float = 1000.0

    # Phase D: Optional real orbit source and deterministic station outages.
    tle: Optional[TLEElements] = None
    orbit_params: Optional[OrbitParams] = None
    ground_station_blackouts: tuple[tuple[int, int, Optional[str]], ...] = ()

    def __post_init__(self):
        if self.num_satellites < 1:
            raise ValueError("num_satellites must be at least one.")
        if self.planes < 1 or self.planes > self.num_satellites:
            raise ValueError("planes must be between 1 and num_satellites.")
        if self.num_satellites % self.planes:
            raise ValueError("num_satellites must divide evenly across planes.")
        if not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError("inclination_deg must be in [0, 180].")
        if self.min_temp_c >= self.max_temp_c:
            raise ValueError("min_temp_c must be lower than max_temp_c.")
        if self.data_capacity_gb <= 0:
            raise ValueError("data_capacity_gb must be positive.")
        if self.target_gap_deg is None:
            self.target_gap_deg = 360.0 / self.sats_per_plane if self.sats_per_plane else 360.0 / (self.num_satellites // self.planes)
        if self.sats_per_plane is None:
            self.sats_per_plane = self.num_satellites // self.planes
        elif self.sats_per_plane * self.planes != self.num_satellites:
            raise ValueError("sats_per_plane * planes must equal num_satellites.")
        for start, end, _station in self.ground_station_blackouts:
            if start < 0 or end <= start:
                raise ValueError("Blackout intervals must satisfy 0 <= start < end.")

    @property
    def orbit(self) -> OrbitParams:
        if self.orbit_params is not None:
            return self.orbit_params
        if self.tle is not None:
            return orbit_params_from_tle(self.tle)
        return ORBIT_PARAMS[self.orbit_type]

    @classmethod
    def from_tle(cls, tle_lines: str | Sequence[str], **kwargs) -> "SatelliteConfig":
        """Build a configuration whose orbit is derived from a validated TLE."""
        tle = parse_tle(tle_lines)
        orbit = orbit_params_from_tle(tle)
        orbit_type = OrbitType.LEO if orbit.altitude_km < 2_000 else OrbitType.MEO if orbit.altitude_km < 35_000 else OrbitType.GEO
        return cls(name=kwargs.pop("name", tle.name), orbit_type=kwargs.pop("orbit_type", orbit_type),
                   tle=tle, orbit_params=orbit, **kwargs)

    @property
    def thruster(self) -> ThrusterParams:
        return THRUSTER_PARAMS[self.thruster_type]

    @property
    def solar_charge_rate(self) -> float:
        return 2.2 * (self.solar_area_m2 / 8.0)

    def station_is_blacked_out(self, step: int, station_name: str) -> bool:
        """Return whether a configured blackout disables this station this step."""
        return any(start <= step < end and (station is None or station == station_name)
                   for start, end, station in self.ground_station_blackouts)

    def to_obs_vector(self) -> np.ndarray:
        """Returns a 6-dim normalized config feature vector."""
        orbit_enc  = {OrbitType.LEO: 0.0, OrbitType.MEO: 0.5, OrbitType.GEO: 1.0}
        thrust_enc = {
            ThrusterType.COLD_GAS: 0.0, ThrusterType.CHEMICAL: 0.33,
            ThrusterType.ION: 0.67,     ThrusterType.HALL: 1.0,
        }
        mission_enc = {
            MissionType.EARTH_OBS: 0.0, MissionType.COMMS: 0.5, MissionType.SCIENCE: 1.0
        }
        
        return np.array([
            orbit_enc[self.orbit_type],
            thrust_enc[self.thruster_type],
            mission_enc[self.mission_type],
            np.clip(self.mass_kg      / 8_000.0, 0.0, 1.0),
            np.clip(self.solar_area_m2 / 50.0,   0.0, 1.0),
            np.clip(self.battery_wh   / 1_000.0, 0.0, 1.0),
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Preset Configurations
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, SatelliteConfig] = {
    "starlink_leo": SatelliteConfig(
        name="Starlink LEO", mission_type=MissionType.COMMS, orbit_type=OrbitType.LEO,
        thruster_type=ThrusterType.HALL, mass_kg=260.0, battery_wh=120.0, solar_area_m2=8.0, 
        num_satellites=10, planes=2, inclination_deg=53.0, min_temp_c=-40.0, max_temp_c=85.0, data_capacity_gb=500.0
    ),
    "gps_meo": SatelliteConfig(
        name="GPS MEO", mission_type=MissionType.COMMS, orbit_type=OrbitType.MEO,
        thruster_type=ThrusterType.ION, mass_kg=845.0, battery_wh=400.0, solar_area_m2=14.0, 
        num_satellites=6, planes=3, inclination_deg=55.0, min_temp_c=-50.0, max_temp_c=70.0, data_capacity_gb=50.0
    ),
    "landsat_obs": SatelliteConfig(
        name="Landsat OBS", mission_type=MissionType.EARTH_OBS, orbit_type=OrbitType.LEO,
        thruster_type=ThrusterType.CHEMICAL, mass_kg=2_700.0, battery_wh=600.0, solar_area_m2=20.0, 
        num_satellites=4, planes=1, inclination_deg=98.2, min_temp_c=-20.0, max_temp_c=50.0, data_capacity_gb=4000.0
    ),
    "cubesat_sci": SatelliteConfig(
        name="CubeSat Science", mission_type=MissionType.SCIENCE, orbit_type=OrbitType.LEO,
        thruster_type=ThrusterType.COLD_GAS, mass_kg=12.0, battery_wh=15.0, solar_area_m2=0.06, 
        num_satellites=12, planes=1, inclination_deg=51.6, min_temp_c=-30.0, max_temp_c=60.0, data_capacity_gb=100.0
    ),
    "geo_comms": SatelliteConfig(
        name="GEO Comms", mission_type=MissionType.COMMS, orbit_type=OrbitType.GEO,
        thruster_type=ThrusterType.CHEMICAL, mass_kg=3_500.0, battery_wh=800.0, solar_area_m2=40.0, 
        num_satellites=4, planes=1, inclination_deg=0.0, min_temp_c=-60.0, max_temp_c=100.0, data_capacity_gb=10000.0
    ),
}

def list_presets() -> None:
    print("\nAvailable satellite presets:")
    print(f"  {'Name':<20} {'Mission':<10} {'Orbit':<6} {'Thruster':<10} {'Mass':>6}kg")
    print("  " + "─" * 65)
    for key, cfg in PRESETS.items():
        print(f"  {key:<20} {cfg.mission_type.value:<10} {cfg.orbit_type.value:<6} "
              f"{cfg.thruster_type.value:<10} {cfg.mass_kg:>6.0f}")

if __name__ == "__main__":
    list_presets()
