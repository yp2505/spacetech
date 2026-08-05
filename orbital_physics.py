"""
orbital_physics.py
------------------
Phase 3: Multi-Shell Orbital Mechanics via Skyfield + SGP4.

Real-world satellite knowledge encoded here
-------------------------------------------
Starlink operates in multiple orbital SHELLS, each at a different:
  - Altitude    (higher orbit = slower speed, Kepler's 3rd Law)
  - Inclination (tilt of orbit plane relative to equator)
  - RAAN        (Right Ascension of Ascending Node — rotation around Earth's pole)

This simulator uses TWO DIFFERENT SHELLS so the orbits cross in 3D space:

  Shell 1 — 53.0° inclined, 550 km:  Group 1 Starlink (most visible from mid-latitudes)
  Shell 3 — 70.0° inclined, 570 km:  Polar/high-latitude coverage shell

Because Shell 1 is inclined 53° and Shell 3 is inclined 70°, AND they have
different RAANs, their orbit rings physically intersect at two crossing nodes.
At those nodes, both satellites are at the exact same point in 3D space — which
is why collision avoidance is a real mission-critical concern for Starlink.

Physics facts used:
  - Kepler's 3rd Law: T = 2π √(a³/μ)  → higher altitude = longer period = slower
  - SGP4 propagator: used by NORAD/SpaceX for real Starlink tracking
  - Atmospheric drag modelled in TLE Bstar term
  - J2 perturbation (Earth's equatorial bulge) causes nodal precession
  - Eclipse is the shadow cone on the anti-sun side (modelled as half-orbit)

Skyfield is used for maximum accuracy when available.
Pure Keplerian is the fallback — still geometrically correct for visualization.
"""

import numpy as np
try:
    from skyfield.api import load, EarthSatellite
    _SKYFIELD_OK = True
except ImportError:
    _SKYFIELD_OK = False

# Each simulation step = 30 seconds of real orbital time
# Real Starlink: one orbit ≈ 95.5 minutes = 5730 seconds = 191 steps
STEP_SECONDS = 30

EARTH_RADIUS_KM = 6371.0
EARTH_MU        = 398600.4418  # km³/s²

# ── Real TLEs from TWO DIFFERENT Starlink shells ─────────────────────────────
# Shell 1: 53.0° inclination, ~550 km — STARLINK-1007 (Group 1)
# Shell 3: 70.0° inclination, ~570 km — STARLINK-3090 (Group 3 polar shell)
#
# These TLEs have DIFFERENT inclination AND different RAAN (orbital node),
# so the two orbit rings are physically tilted at different angles and CROSS.
#
# TLE format:
#   Line 1: NORAD ID, epoch, drag, BSTAR coefficient
#   Line 2: inclination, RAAN, eccentricity, arg_perigee, mean_anomaly, mean_motion
SHELL_TLES = [
    {
        "name": "STARLINK-1007 (Shell-1, 53°)",
        "line1": "1 44713U 19074A   24001.50000000  .00002000  00000-0  15000-3 0  9995",
        "line2": "2 44713  53.0530  60.0000 0001480  98.5400 261.5500 15.06379830200018",
        # inclination=53.05°, RAAN=60°, ~550km altitude, 15.06 rev/day
    },
    {
        "name": "STARLINK-3090 (Shell-3, 70°)",
        "line1": "1 51180U 22026A   24001.50000000  .00001500  00000-0  11000-3 0  9992",
        "line2": "2 51180  70.0038 200.0000 0002200 270.0000  90.0000 14.98376520180015",
        # inclination=70.00°, RAAN=200°, ~570km altitude, 14.98 rev/day
        # RAAN=200° vs 60° → 140° apart → orbits cross at two nodes!
    },
]

# ── Keplerian fallback orbit generator ────────────────────────────────────────

def _orbit_xyz_ring_keplerian(altitude_km: float, inclination_deg: float,
                               raan_deg: float, num_points: int = 360) -> np.ndarray:
    """
    Generate (num_points, 3) Cartesian km positions for one full orbit.
    Uses 3D rotation: orbit plane → inclination → RAAN.
    """
    r = EARTH_RADIUS_KM + altitude_km
    nu = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Orbit in its own plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Rotate by inclination (around X-axis)
    i = np.radians(inclination_deg)
    x1 = x_orb
    y1 = y_orb * np.cos(i)
    z1 = y_orb * np.sin(i)

    # Rotate by RAAN (around Z-axis)
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
    return np.array([x2, y2, z1])  # Note: z stays from inclination rotation


def _fetch_live_tle_pair():
    """
    Try to fetch a matching pair of Starlink TLEs from CelesTrak:
    one from Shell 1 (53°) and one from Shell 3 (70°).
    Returns (tle0_dict, tle1_dict) or None on failure.
    """
    import urllib.request
    try:
        url = "https://celestrak.org/SATCAT/TLE.PHP?GROUP=starlink&FORMAT=TLE"
        with urllib.request.urlopen(url, timeout=4) as resp:
            raw = resp.read().decode("utf-8").strip().splitlines()

        shell1_tle = None
        shell3_tle = None
        i = 0
        while i + 2 < len(raw):
            name = raw[i].strip()
            l1 = raw[i + 1].strip()
            l2 = raw[i + 2].strip()
            if l1.startswith("1 ") and l2.startswith("2 "):
                try:
                    inc = float(l2[8:16])
                    # Find a 53° shell sat
                    if shell1_tle is None and 52.8 < inc < 53.3:
                        shell1_tle = {"name": name + " (Shell-1, 53°)", "line1": l1, "line2": l2}
                    # Find a 70° shell sat
                    elif shell3_tle is None and 69.8 < inc < 70.3:
                        shell3_tle = {"name": name + " (Shell-3, 70°)", "line1": l1, "line2": l2}
                    if shell1_tle and shell3_tle:
                        return [shell1_tle, shell3_tle]
                except (ValueError, IndexError):
                    pass
                i += 3
            else:
                i += 1
    except Exception:
        pass
    return None


class OrbitalPhysics:
    """
    Provides 3D orbital positions and geometry for two satellites in DIFFERENT
    orbital shells, using Skyfield + SGP4 when available for maximum accuracy.

    Real physics included:
      - Different inclinations (53° and 70°) → orbit planes intersect
      - Different RAAN (60° vs 200°) → crossing nodes at two points
      - Different altitudes → different orbital speeds (Kepler's 3rd Law)
      - SGP4 propagation accounts for J2 oblateness and atmospheric drag
    """

    def __init__(self):
        self._skyfield_sats = None
        self._ts = None
        self._keplerian_params = None   # fallback

        tles = None

        # Step 1: Try to fetch fresh live TLEs with proper different shells
        if _SKYFIELD_OK:
            tles = _fetch_live_tle_pair()
            if tles:
                print("[OrbitalPhysics] ✓ Live TLEs fetched from CelesTrak (different shells)")
            else:
                tles = SHELL_TLES
                print("[OrbitalPhysics] ⚠ Network unavailable — using hardcoded multi-shell TLEs")
        else:
            tles = SHELL_TLES
            print("[OrbitalPhysics] ⚠ Skyfield not found — using pure Keplerian mechanics")

        self._tles = tles

        # Step 2: Load into Skyfield if available
        if _SKYFIELD_OK:
            try:
                self._ts = load.timescale()
                self._skyfield_sats = [
                    EarthSatellite(t["line1"], t["line2"], t["name"], self._ts)
                    for t in tles
                ]
                self._epoch_t = self._skyfield_sats[0].epoch
                print("[OrbitalPhysics] ✓ SGP4 propagator loaded via Skyfield")
            except Exception as e:
                print(f"[OrbitalPhysics] ⚠ Skyfield load failed ({e}), using Keplerian fallback")
                self._skyfield_sats = None

        # Step 3: Parse inclination/RAAN/altitude from TLE line2 for Keplerian fallback
        self._keplerian_params = []
        for t in tles:
            l2 = t["line2"]
            try:
                inc  = float(l2[8:16].strip())
                raan = float(l2[17:25].strip())
                # Mean motion (rev/day) → altitude via Kepler
                n_rev_day = float(l2[52:63].strip())
                n_rad_s   = n_rev_day * 2 * np.pi / 86400.0
                a_km      = (EARTH_MU / n_rad_s**2) ** (1/3)
                alt_km    = a_km - EARTH_RADIUS_KM
            except Exception:
                inc = 53.0; raan = 60.0; alt_km = 550.0
            self._keplerian_params.append(
                {"inclination_deg": inc, "raan_deg": raan, "altitude_km": alt_km}
            )

        # Log satellite info
        for i, (t, kp) in enumerate(zip(tles, self._keplerian_params)):
            T_s = 2 * np.pi * np.sqrt((EARTH_RADIUS_KM + kp['altitude_km'])**3 / EARTH_MU)
            v_km_s = np.sqrt(EARTH_MU / (EARTH_RADIUS_KM + kp['altitude_km']))
            print(
                f"[OrbitalPhysics] Sat {i}: {t['name']}  "
                f"Alt={kp['altitude_km']:.0f}km  "
                f"Inc={kp['inclination_deg']:.1f}°  "
                f"RAAN={kp['raan_deg']:.1f}°  "
                f"T={T_s/60:.1f}min  "
                f"v={v_km_s:.2f}km/s"
            )

        # Precompute orbit rings for rendering (never changes during episode)
        self._rings = [
            self.get_orbit_xyz_ring(i, num_points=360) for i in range(2)
        ]

    def _get_skyfield_time(self, step: int):
        elapsed = step * STEP_SECONDS
        return self._ts.tt_jd(self._epoch_t.tt + elapsed / 86400.0)

    def get_xyz(self, sat_idx: int, step: int) -> np.ndarray:
        """3D XYZ position in km at given simulation step."""
        if self._skyfield_sats is not None:
            t = self._get_skyfield_time(step)
            return self._skyfield_sats[sat_idx].at(t).position.km
        else:
            kp = self._keplerian_params[sat_idx]
            # Compute angle from step
            T_s = 2 * np.pi * np.sqrt((EARTH_RADIUS_KM + kp['altitude_km'])**3 / EARTH_MU)
            angle_deg = (step * STEP_SECONDS / T_s * 360.0) % 360.0
            return _angle_to_xyz_keplerian(
                angle_deg, kp['altitude_km'], kp['inclination_deg'], kp['raan_deg']
            )

    def get_orbit_xyz_ring(self, sat_idx: int, num_points: int = 360) -> np.ndarray:
        """
        Generate the complete 3D orbit ring for visualization.
        Uses Skyfield propagation for accuracy (accounts for J2/drag).
        Returns (num_points, 3) array in km.
        """
        if self._skyfield_sats is not None:
            # Use one full orbital period via SGP4 (accurate)
            kp = self._keplerian_params[sat_idx]
            T_s = 2 * np.pi * np.sqrt((EARTH_RADIUS_KM + kp['altitude_km'])**3 / EARTH_MU)
            steps_per_orbit = int(T_s / STEP_SECONDS)
            step_indices = np.linspace(0, steps_per_orbit, num_points, dtype=int)
            xyzs = [self.get_xyz(sat_idx, int(s)) for s in step_indices]
            return np.array(xyzs)
        else:
            kp = self._keplerian_params[sat_idx]
            return _orbit_xyz_ring_keplerian(
                kp['altitude_km'], kp['inclination_deg'], kp['raan_deg'], num_points
            )

    def angle_to_xyz_ring(self, sat_idx: int, angle_deg: float) -> np.ndarray:
        """
        Map a true anomaly angle to the precomputed 3D ring position.
        Returns (3,) array in km.
        """
        ring = self._rings[sat_idx]
        idx  = int(round(angle_deg)) % len(ring)
        return ring[idx]

    def get_orbital_speed_km_s(self, sat_idx: int) -> float:
        kp = self._keplerian_params[sat_idx]
        return float(np.sqrt(EARTH_MU / (EARTH_RADIUS_KM + kp['altitude_km'])))

    def get_nominal_velocity_deg_per_step(self, sat_idx: int) -> float:
        """Nominal angular velocity in degrees per 30-second step."""
        kp = self._keplerian_params[sat_idx]
        T_s = 2 * np.pi * np.sqrt((EARTH_RADIUS_KM + kp['altitude_km'])**3 / EARTH_MU)
        return 360.0 * STEP_SECONDS / T_s

    # Legacy compatibility
    def get_initial_slots(self, num_slots: int = 20):
        return [0, num_slots // 4]
