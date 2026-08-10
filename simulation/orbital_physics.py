"""
orbital_physics.py
------------------
Phase C-F: Orbital Physics Engine.
Includes 3D Keplerian mapping, Ground Station LOS (Line-of-Sight) math,
and Multi-plane Walker Delta constellation distribution.

Fixed: inclination is now stored per-satellite in walker params and passed
correctly to anomaly_to_ecef().
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0
EARTH_MU        = 398600.4418

# ─────────────────────────────────────────────────────────────────────────────
#  Static Ground Stations (Lat, Lon) — Phase D
# ─────────────────────────────────────────────────────────────────────────────
GROUND_STATIONS = [
    {"name": "Svalbard",     "lat": 78.2298,  "lon":  15.4078},
    {"name": "Punta Arenas", "lat": -53.1403, "lon": -70.9063},
    {"name": "Hawaii",       "lat":  19.8968, "lon": -155.5828},
    {"name": "Singapore",    "lat":   1.3521, "lon":  103.8198},
    {"name": "Bangalore",    "lat":  12.9716, "lon":   77.5946},
]


def latlon_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    """Convert Lat/Lon to Earth-Centered Earth-Fixed (ECEF) 3D coordinate."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    r = EARTH_RADIUS_KM + alt_km
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return np.array([x, y, z])


def check_ground_station_los(
    sat_ecef: np.ndarray,
    min_elevation_deg: float = 5.0
) -> tuple[bool, str]:
    """
    Check if the satellite is visible to ANY ground station.
    Returns (is_visible, station_name).
    Uses dot-product horizon check with a minimum elevation mask.
    """
    for gs in GROUND_STATIONS:
        gs_ecef = latlon_to_ecef(gs["lat"], gs["lon"])
        vec_gs_to_sat = sat_ecef - gs_ecef
        gs_norm = np.linalg.norm(gs_ecef)
        vec_norm = np.linalg.norm(vec_gs_to_sat)

        if gs_norm < 1e-6 or vec_norm < 1e-6:
            continue

        zenith = gs_ecef / gs_norm
        sat_dir = vec_gs_to_sat / vec_norm
        cos_angle = np.clip(np.dot(zenith, sat_dir), -1.0, 1.0)
        elevation_deg = 90.0 - np.degrees(np.arccos(cos_angle))

        if elevation_deg >= min_elevation_deg:
            return True, gs["name"]

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
#  Keplerian to Cartesian (ECEF) Math
# ─────────────────────────────────────────────────────────────────────────────
def anomaly_to_ecef(
    true_anomaly_deg: float,
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    earth_rotation_offset_deg: float = 0.0
) -> np.ndarray:
    """
    Maps a satellite's true anomaly and orbit parameters to a 3D ECEF coordinate.
    Includes simple Earth rotation offset to simulate ground track movement.

    Args:
        true_anomaly_deg:       Position in the orbital ring (0-360 deg)
        altitude_km:            Orbital altitude above Earth surface
        inclination_deg:        Orbit inclination (e.g. 53 deg for Starlink)
        raan_deg:               Right Ascension of the Ascending Node
        earth_rotation_offset:  Earth has rotated this many degrees since epoch
    """
    r = EARTH_RADIUS_KM + altitude_km
    theta = np.radians(true_anomaly_deg)
    inc   = np.radians(inclination_deg)

    # 1. Coordinates in the orbital plane (perifocal frame, circular orbit)
    x_orbit = r * np.cos(theta)
    y_orbit = r * np.sin(theta)

    # 2. Rotate by inclination around the X axis
    y_inc = y_orbit * np.cos(inc)
    z_inc = y_orbit * np.sin(inc)

    # 3. Rotate by RAAN around the Z axis (adjusted for Earth rotation)
    raan_rad = np.radians(raan_deg - earth_rotation_offset_deg)
    x_ecef = x_orbit * np.cos(raan_rad) - y_inc * np.sin(raan_rad)
    y_ecef = x_orbit * np.sin(raan_rad) + y_inc * np.cos(raan_rad)
    z_ecef = z_inc

    return np.array([x_ecef, y_ecef, z_ecef])


# ─────────────────────────────────────────────────────────────────────────────
#  Walker Delta Constellation Topology — Phase F
# ─────────────────────────────────────────────────────────────────────────────
def get_walker_delta_params(
    num_satellites: int,
    planes: int,
    inclination_deg: float = 53.0,
    raan_offset_deg: float = 0.0,
    anomaly_offset_deg: float = 0.0,
) -> list[dict]:
    """
    Generates orbital parameters for a Walker Delta Constellation.
    Each satellite gets: raan_deg, anomaly_offset_deg, inclination_deg, plane_id

    A Walker Delta T/P/F constellation distributes T total satellites across
    P planes with F phasing factor. We use F=1 (standard Walker Star is F=0).

    Args:
        num_satellites:   Total number of satellites in constellation
        planes:           Number of orbital planes (sets RAAN spacing)
        inclination_deg:  Orbital inclination shared across all planes

    Returns:
        List of dicts with orbit parameters per satellite.
    """
    if num_satellites < 1 or planes < 1 or planes > num_satellites:
        raise ValueError("Require 1 <= planes <= num_satellites.")
    if num_satellites % planes:
        raise ValueError("Walker Delta requires num_satellites divisible by planes.")
    params = []
    sats_per_plane = num_satellites // planes

    for i in range(num_satellites):
        plane_idx          = i // sats_per_plane
        sat_idx_in_plane   = i % sats_per_plane

        # RAAN is evenly spread across 180 degrees for polar Walker Delta
        # (use 360 for Walker Star where planes don't overlap at poles)
        raan = (raan_offset_deg + plane_idx * (180.0 / planes)) % 360.0

        # True anomaly is evenly spread within each plane.
        # Add inter-plane phasing to distribute satellites more uniformly.
        phasing = plane_idx * (360.0 / num_satellites)
        anomaly = (anomaly_offset_deg + sat_idx_in_plane * (360.0 / sats_per_plane) + phasing) % 360.0

        params.append({
            "raan_deg":         raan,
            "anomaly_offset_deg": anomaly,
            "inclination_deg":  inclination_deg,
            "plane_id":         plane_idx,
        })

    return params
