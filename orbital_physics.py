"""
orbital_physics.py
------------------
Phase 4: Multi-Shell Orbital Mechanics for N-Satellite Swarm via Skyfield + SGP4.
"""

import numpy as np
try:
    from skyfield.api import load, EarthSatellite
    _SKYFIELD_OK = True
except ImportError:
    _SKYFIELD_OK = False

STEP_SECONDS = 30
EARTH_RADIUS_KM = 6371.0
EARTH_MU        = 398600.4418  # km³/s²

# Advanced physics constants
J2 = 1.08262668e-3  # Earth's J2 coefficient
J4 = -1.61e-6       # Earth's J4 coefficient
SOLAR_FLUX_P = 4.56e-6  # N/m^2 (solar radiation pressure at 1 AU)
SATELLITE_AREA = 30.0   # m^2 (Starlink solar panel area)
SATELLITE_MASS = 260.0  # kg

# Pre-defined base shells (altitude, inclination)
BASE_SHELLS = [
    (550.0, 53.0),
    (570.0, 70.0),
    (560.0, 97.6),
    (540.0, 42.0),
    (530.0, 38.0)
]

def _orbit_xyz_ring_keplerian(altitude_km: float, inclination_deg: float,
                               raan_deg: float, num_points: int = 360) -> np.ndarray:
    r = EARTH_RADIUS_KM + altitude_km
    nu = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    i = np.radians(inclination_deg)
    x1 = x_orb
    y1 = y_orb * np.cos(i)
    z1 = y_orb * np.sin(i)
    raan = np.radians(raan_deg)
    x2 = x1 * np.cos(raan) - y1 * np.sin(raan)
    y2 = x1 * np.sin(raan) + y1 * np.cos(raan)
    z2 = z1
    return np.stack([x2, y2, z2], axis=1)

def _angle_to_xyz_keplerian(angle_deg: float, altitude_km: float,
                             inclination_deg: float, raan_deg: float) -> np.ndarray:
    r = EARTH_RADIUS_KM + altitude_km
    nu = np.radians(angle_deg)
    xo = r * np.cos(nu)
    yo = r * np.sin(nu)
    i = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    x1 = xo; y1 = yo * np.cos(i); z1 = yo * np.sin(i)
    x2 = x1 * np.cos(raan) - y1 * np.sin(raan)
    y2 = x1 * np.sin(raan) + y1 * np.cos(raan)
    return np.array([x2, y2, z1])

class OrbitalPhysics:
    def __init__(self, num_satellites: int = 10):
        self.num_satellites = num_satellites
        self._keplerian_params = []
        
        # Distribute N satellites across the base shells with varying RAAN
        for i in range(num_satellites):
            shell = BASE_SHELLS[i % len(BASE_SHELLS)]
            alt_km, inc_deg = shell
            # Spread RAAN evenly around 360 degrees for satellites in the same shell
            raan_deg = (i * (360.0 / num_satellites)) % 360.0
            
            self._keplerian_params.append({
                "inclination_deg": inc_deg,
                "raan_deg": raan_deg,
                "altitude_km": alt_km
            })

        print(f"[OrbitalPhysics] Initialized {num_satellites} satellites across {len(BASE_SHELLS)} shells.")
        
        # Precompute orbit rings
        self._rings = [
            self.get_orbit_xyz_ring(i, num_points=360) for i in range(num_satellites)
        ]

    def get_xyz(self, sat_idx: int, step: int) -> np.ndarray:
        kp = self._keplerian_params[sat_idx]
        T_s = 2 * np.pi * np.sqrt((EARTH_RADIUS_KM + kp['altitude_km'])**3 / EARTH_MU)
        angle_deg = (step * STEP_SECONDS / T_s * 360.0) % 360.0
        return _angle_to_xyz_keplerian(
            angle_deg, kp['altitude_km'], kp['inclination_deg'], kp['raan_deg']
        )

    def get_orbit_xyz_ring(self, sat_idx: int, num_points: int = 360) -> np.ndarray:
        kp = self._keplerian_params[sat_idx]
        return _orbit_xyz_ring_keplerian(
            kp['altitude_km'], kp['inclination_deg'], kp['raan_deg'], num_points
        )

    def angle_to_xyz_ring(self, sat_idx: int, angle_deg: float) -> np.ndarray:
        ring = self._rings[sat_idx]
        idx  = int(round(angle_deg)) % len(ring)
        return ring[idx]

    def get_orbital_speed_km_s(self, sat_idx: int) -> float:
        kp = self._keplerian_params[sat_idx]
        return float(np.sqrt(EARTH_MU / (EARTH_RADIUS_KM + kp['altitude_km'])))

    def get_nominal_velocity_deg_per_step(self, sat_idx: int) -> float:
        kp = self._keplerian_params[sat_idx]
        T_s = 2 * np.pi * np.sqrt((EARTH_RADIUS_KM + kp['altitude_km'])**3 / EARTH_MU)
        return 360.0 * STEP_SECONDS / T_s
