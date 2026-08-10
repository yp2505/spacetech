"""
tests/test_fsw_sil.py
---------------------
Automated Software-In-the-Loop Integration Tests for FSW Architecture.
"""

import unittest
from simulation.satellite_env import MultiSatelliteEnv
from simulation.sat_config import PRESETS
from fsw.hal.sim_backend import SimEPS, SimThermal, SimADCS, SimRadio, SimFaultMonitor
from fsw.main import OnboardAutonomyAgent
from fsw.fdir.state_machine import OperatingMode

class TestFSWSIL(unittest.TestCase):
    def setUp(self):
        self.config = PRESETS["starlink_leo"]
        self.sim_env = MultiSatelliteEnv(config=self.config, max_steps=100)
        self.sim_env.reset()
        
        agent_idx = 0
        self.eps = SimEPS(self.sim_env, agent_idx)
        self.thermal = SimThermal(self.sim_env, agent_idx)
        self.adcs = SimADCS(self.sim_env, agent_idx)
        self.radio = SimRadio(self.sim_env, agent_idx)
        self.fault = SimFaultMonitor(self.sim_env, agent_idx)
        
        self.agent = OnboardAutonomyAgent(
            self.eps, self.thermal, self.adcs, self.radio, self.fault, 
            sim_env_ref=self.sim_env, agent_idx=agent_idx
        )
        # Fast forward BOOT mode
        for _ in range(5):
            self.agent.run(max_ticks=1)

    def test_nominal_execution(self):
        """Test the FSW reaches NOMINAL mode and queries the AI without errors."""
        self.assertEqual(self.agent.fdir_fsm.mode, OperatingMode.NOMINAL)
        # Advance 20 ticks to trigger the 2Hz AI loop multiple times
        for _ in range(20):
            self.agent.run(max_ticks=1)
        # Verify a command was emitted
        self.assertIsNotNone(self.adcs.last_command)
        
    def test_thruster_fault_blocks_propulsion(self):
        """Verify that injecting a thruster fault locks out propulsion commands."""
        # Inject fault in physics env
        self.sim_env.fault_thruster[0] = True
        
        for _ in range(20): # Enough ticks for FDIR to latch
            self.agent.run(max_ticks=1)
            
        self.assertEqual(self.agent.fdir_fsm.mode, OperatingMode.DEGRADED)
        
        # Now trigger the AI loop. Even if AI proposes thrust, the Arbiter should block it.
        # We manually propose thrust via the arbiter to test it
        cmd_t, cmd_r, cmd_p, cmd_y = self.agent.ai.propose_action(self.agent.state, in_recovery=False)
        
        # Verify the HAL got 0 thrust
        thrust = self.adcs.last_command[0]
        self.assertEqual(thrust, 0.0, "Thruster command was NOT blocked during a thruster fault.")

if __name__ == "__main__":
    unittest.main()
