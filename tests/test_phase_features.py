"""Fast, deterministic regression tests for Phase B/D/F configuration features."""

import unittest

from simulation.orbital_physics import get_walker_delta_params
from simulation.sat_config import (
    MissionType, OrbitType, SatelliteConfig, ThrusterType, parse_tle,
    orbit_params_from_tle, PRESETS,
)


def _tle_line(prefix: str) -> str:
    """Add a NORAD checksum to a 68-column element line."""
    assert len(prefix) == 68
    checksum = sum(int(c) if c.isdigit() else 1 if c == "-" else 0 for c in prefix) % 10
    return prefix + str(checksum)


class TestPhaseFeatures(unittest.TestCase):
    def test_tle_parsing_and_orbit_derivation(self):
        line1 = _tle_line("1 25544U 98067A   24001.00000000  .00000000  00000+0  00000+0 0  999")
        line2 = _tle_line("2 25544  51.6400  20.0000 0005000  30.0000  40.0000 15.50000000    1")
        tle = parse_tle(["ISS test", line1, line2])
        orbit = orbit_params_from_tle(tle)
        self.assertEqual(tle.name, "ISS test")
        self.assertAlmostEqual(tle.inclination_deg, 51.64)
        self.assertGreater(orbit.altitude_km, 300)
        self.assertLess(orbit.altitude_km, 600)

    def test_blackout_windows_are_station_scoped(self):
        config = SatelliteConfig(
            name="test", mission_type=MissionType.COMMS, orbit_type=OrbitType.LEO,
            thruster_type=ThrusterType.HALL, mass_kg=100, battery_wh=100,
            solar_area_m2=1, ground_station_blackouts=((5, 10, "Hawaii"), (20, 25, None)),
        )
        self.assertTrue(config.station_is_blacked_out(6, "Hawaii"))
        self.assertFalse(config.station_is_blacked_out(6, "Bangalore"))
        self.assertTrue(config.station_is_blacked_out(22, "Bangalore"))
        self.assertFalse(config.station_is_blacked_out(10, "Hawaii"))

    def test_walker_topology_and_invalid_layout(self):
        params = get_walker_delta_params(12, 3, inclination_deg=55)
        self.assertEqual({p["plane_id"] for p in params}, {0, 1, 2})
        self.assertTrue(all(p["inclination_deg"] == 55 for p in params))
        with self.assertRaises(ValueError):
            get_walker_delta_params(10, 3)

    def test_config_rejects_ambiguous_plane_layout(self):
        with self.assertRaises(ValueError):
            SatelliteConfig(
                name="bad", mission_type=MissionType.COMMS, orbit_type=OrbitType.LEO,
                thruster_type=ThrusterType.HALL, mass_kg=100, battery_wh=100,
                solar_area_m2=1, num_satellites=10, planes=3,
            )


if __name__ == "__main__":
    unittest.main()
