"""
orbital_physics.py
------------------
Phase C-F: Orbital Physics Engine.
Includes 3D Keplerian mapping, Ground Station LOS (Line-of-Sight) math,
and Multi-plane Walker Delta constellation distribution.

Step E upgrades:
- sgp4_propagate(): Real TLE-based orbit propagation (sgp4 library)
- skyfield_eclipse_check(): Precise Sun/Earth/shadow geometry (Skyfield)
- skyfield_los_check(): Accurate topocentric elevation for ground station LOS
All three degrade gracefully to the original math if the libraries are absent.

Fixed: inclination is now stored per-satellite in walker params and passed
correctly to anomaly_to_ecef().
"""

import math
import numpy as np
import datetime

# ── sgp4 integration (Step E) ────────────────────────────────────────────────
try:
    from sgp4.api import Satrec, WGS72
    _SGP4_AVAILABLE = True
except ImportError:
    _SGP4_AVAILABLE = False

# ── Skyfield integration (Step E) ────────────────────────────────────────────
try:
    from skyfield.api import load, wgs84, EarthSatellite
    from skyfield.framelib import ecliptic_frame
    _ts = load.timescale()
    _SKYFIELD_AVAILABLE = True
except Exception:
    _SKYFIELD_AVAILABLE = False


EARTH_RADIUS_KM = 6371.0
EARTH_MU        = 398600.4418


# ─────────────────────────────────────────────────────────────────────────────
#  Step E: sgp4 Orbit Propagator
# ─────────────────────────────────────────────────────────────────────────────
def sgp4_propagate(
    tle_line1: str,
    tle_line2: str,
    minutes_since_epoch: float,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Propagate a satellite position and velocity using the real SGP4 algorithm.

    Args:
        tle_line1:             TLE line 1 string (69 chars)
        tle_line2:             TLE line 2 string (69 chars)
        minutes_since_epoch:   Minutes elapsed since the TLE epoch

    Returns:
        (pos_km, vel_km_s): TEME-frame position (km) and velocity (km/s)
        or (None, None) if sgp4 is not available or TLE is invalid.
    """
    if not _SGP4_AVAILABLE:
        return None, None
    try:
        sat = Satrec.twoline2rv(tle_line1, tle_line2)
        # sgp4 wants whole minutes + fractional days offset
        whole_min = int(minutes_since_epoch)
        frac_day  = (minutes_since_epoch - whole_min) / 1440.0
        e, pos, vel = sat.sgp4(0.0, minutes_since_epoch / 1440.0)
        if e != 0:
            return None, None
        return np.array(pos), np.array(vel)
    except Exception:
        return None, None


def teme_to_ecef(pos_teme: np.ndarray, gst_rad: float) -> np.ndarray:
    """
    Rotate a TEME (True Equator Mean Equinox) position vector to ECEF.

    Args:
        pos_teme:  (3,) position vector in km, TEME frame
        gst_rad:   Greenwich Sidereal Time in radians

    Returns:
        (3,) position vector in km, ECEF frame
    """
    cos_g = math.cos(gst_rad)
    sin_g = math.sin(gst_rad)
    x_ecef =  cos_g * pos_teme[0] + sin_g * pos_teme[1]
    y_ecef = -sin_g * pos_teme[0] + cos_g * pos_teme[1]
    z_ecef =  pos_teme[2]
    return np.array([x_ecef, y_ecef, z_ecef])


# ─────────────────────────────────────────────────────────────────────────────
#  Step E: Skyfield Eclipse & LOS
# ─────────────────────────────────────────────────────────────────────────────
def skyfield_eclipse_check(
    tle_line1: str,
    tle_line2: str,
    t_utc: datetime.datetime,
) -> tuple[bool, float]:
    """
    Check whether the satellite is in Earth's shadow using Skyfield's precise
    Sun/Earth/shadow geometry.

    Args:
        tle_line1, tle_line2: TLE strings
        t_utc:               UTC datetime for the check

    Returns:
        (is_eclipse, shadow_fraction)  — fraction 0.0 (sunlit) to 1.0 (full shadow)
        Falls back to (None, None) if Skyfield is unavailable.
    """
    if not _SKYFIELD_AVAILABLE:
        return None, None
    try:
        sat = EarthSatellite(tle_line1, tle_line2, ts=_ts)
        t   = _ts.from_datetime(t_utc.replace(tzinfo=datetime.timezone.utc))
        sunlit = sat.at(t).is_sunlit(load('de421.bsp'))
        return (not sunlit), (0.0 if sunlit else 1.0)
    except Exception:
        return None, None


def skyfield_los_check(
    tle_line1: str,
    tle_line2: str,
    t_utc: datetime.datetime,
    lat_deg: float,
    lon_deg: float,
    min_elevation_deg: float = 5.0,
) -> tuple[bool, float]:
    """
    Check ground-station line-of-sight using Skyfield's topocentric elevation.
    More accurate than the dot-product method for stations at high latitudes.

    Args:
        tle_line1, tle_line2: TLE strings
        t_utc:               UTC datetime
        lat_deg, lon_deg:    Ground station geodetic coordinates
        min_elevation_deg:   Minimum elevation angle for LOS

    Returns:
        (is_visible, elevation_deg)
        Falls back to (None, None) if Skyfield is unavailable.
    """
    if not _SKYFIELD_AVAILABLE:
        return None, None
    try:
        sat = EarthSatellite(tle_line1, tle_line2, ts=_ts)
        t   = _ts.from_datetime(t_utc.replace(tzinfo=datetime.timezone.utc))
        gs  = wgs84.latlon(lat_deg, lon_deg)
        diff      = sat - gs
        topocentric = diff.at(t)
        alt, az, distance = topocentric.altaz()
        elev = alt.degrees
        return (elev >= min_elevation_deg), float(elev)
    except Exception:
        return None, None


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
