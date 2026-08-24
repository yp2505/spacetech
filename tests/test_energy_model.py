"""
test_energy_model.py
====================
Step A0 Regression Tests - Energy Model Sanity.

Permanent regression tests:
  1. Zero-action policy must never drop below SAFE_MODE_THRESHOLD (12%) battery
     on any orbit preset when the energy model is physically consistent.
  2. Full-orbit energy balance must be positive (net gain) for all orbit presets.
  3. V(s) slope check helper used by the smoke-test runner.

Run with:
    cd spacetech
    python -m unittest tests.test_energy_model -v
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.satellite_env import MultiSatelliteEnv, SAFE_MODE_THRESHOLD
from simulation.sat_config import PRESETS


class TestEnergyModelIdlePolicy(unittest.TestCase):
    """
    Step A0 Item 4: Simulate a zero-action policy for one full episode (360 steps)
    and assert no satellite ever drops below SAFE_MODE_THRESHOLD from energy drain alone.
    """

    def _run_zero_action_episode(self, preset_key, phase=2):
        config = PRESETS[preset_key]
        env = MultiSatelliteEnv(config=config, max_steps=360)
        env.curriculum_phase = phase
        env.reset(seed=42)
        min_battery = np.full(config.num_satellites, 100.0)
        safe_triggered = np.zeros(config.num_satellites, dtype=bool)
        for _ in range(360):
            zero_actions = np.zeros((config.num_satellites, 8), dtype=np.float32)
            _, _, term, trunc, _ = env.step(zero_actions)
            min_battery = np.minimum(min_battery, env.agent_battery)
            safe_triggered |= env.in_safe_mode
            if term or trunc:
                break
        return {"min_battery": float(np.min(min_battery)), "any_safe_mode": bool(np.any(safe_triggered))}

    def test_starlink_leo_phase1_idle_never_safe_mode(self):
        r = self._run_zero_action_episode("starlink_leo", phase=1)
        self.assertFalse(r["any_safe_mode"],
            f"starlink_leo Phase1: idle triggered safe mode! min_battery={r['min_battery']:.2f}%")
        self.assertGreater(r["min_battery"], SAFE_MODE_THRESHOLD,
            f"starlink_leo Phase1: battery below threshold ({r['min_battery']:.2f}% < {SAFE_MODE_THRESHOLD}%)")

    def test_starlink_leo_phase2_idle_never_safe_mode(self):
        r = self._run_zero_action_episode("starlink_leo", phase=2)
        self.assertFalse(r["any_safe_mode"],
            f"starlink_leo Phase2 (eclipse ON): idle triggered safe mode! min_battery={r['min_battery']:.2f}%. "
            f"ENERGY MODEL BUG: eclipse drain exceeds solar charge for idle policy.")
        self.assertGreater(r["min_battery"], SAFE_MODE_THRESHOLD,
            f"starlink_leo Phase2: battery below threshold ({r['min_battery']:.2f}% < {SAFE_MODE_THRESHOLD}%)")

    def test_gps_meo_phase2_idle_never_safe_mode(self):
        r = self._run_zero_action_episode("gps_meo", phase=2)
        self.assertFalse(r["any_safe_mode"],
            f"gps_meo Phase2: idle triggered safe mode! min_battery={r['min_battery']:.2f}%")

    def test_geo_comms_phase2_idle_never_safe_mode(self):
        r = self._run_zero_action_episode("geo_comms", phase=2)
        self.assertFalse(r["any_safe_mode"],
            f"geo_comms Phase2: idle triggered safe mode! min_battery={r['min_battery']:.2f}%")


class TestEnergyBalance(unittest.TestCase):
    """Verify full-orbit net energy balance is positive for all presets."""

    def _compute_orbit_balance(self, preset_key):
        config = PRESETS[preset_key]
        orbit = config.orbit
        steps_per_orbit = round(360.0 / (360.0 * orbit.step_seconds / (orbit.period_min * 60.0)))
        shadow_steps = round(steps_per_orbit * (orbit.eclipse_arc_deg / 360.0))
        sunlit_steps = steps_per_orbit - shadow_steps
        drain = orbit.eclipse_fraction * 0.97
        net = sunlit_steps * config.solar_charge_rate - shadow_steps * drain
        print(f"\n  [{preset_key}] steps/orbit={steps_per_orbit} shadow={shadow_steps} "
              f"sunlit={sunlit_steps} drain={drain:.3f}%/step "
              f"charge={config.solar_charge_rate:.3f}%/step NET={net:+.2f}%/orbit")
        return net

    def test_all_presets_positive_energy_balance(self):
        for key in PRESETS:
            with self.subTest(preset=key):
                net = self._compute_orbit_balance(key)
                self.assertGreater(net, 0.0,
                    f"{key}: negative orbit energy balance ({net:.2f}%/orbit) - ENERGY MODEL BUG")


class TestVsSlopeHelper(unittest.TestCase):
    """V(s) slope computation used in smoke tests."""

    def test_flat_slope_near_zero(self):
        vs = np.ones(15) * -5.0
        slope = float(np.polyfit(np.arange(15), vs, 1)[0])
        self.assertLess(abs(slope), 0.01, f"Flat V(s) slope={slope:.6f} should be ~0")

    def test_diverging_slope_detected(self):
        vs = np.linspace(-0.03, -33.1, 200)
        slope = float(np.polyfit(np.arange(200), vs, 1)[0])
        self.assertGreaterEqual(abs(slope), 0.01,
            f"Diverging V(s) should have |slope|>=0.01, got {slope:.6f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
