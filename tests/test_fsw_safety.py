"""Deterministic safety and FDIR regression tests for the onboard-agent scaffold."""

import unittest

from fsw.ai_brain.adapter import AIObservationAdapter
from fsw.core.commands import ActuatorCommand
from fsw.core.scheduler import CyclicScheduler
from fsw.core.telemetry import (
    ADCSTelemetry, EPSTelemetry, FaultTelemetry, SubsystemState, ThermalTelemetry,
)
from fsw.fdir.state_machine import FDIRStateMachine, OperatingMode
from fsw.safety.arbiter import CommandArbiter
from fsw.safety.supervisor import SafetySupervisor


def healthy_state() -> SubsystemState:
    return SubsystemState(
        eps=EPSTelemetry(battery_charge_percent=80.0, solar_power_w=10.0),
        thermal=ThermalTelemetry(battery_temp_c=20.0),
        adcs=ADCSTelemetry(rates_deg_s=(0.0, 0.0, 0.0), fuel_percent=90.0),
        faults=FaultTelemetry(),
    )


class TestFlightSafety(unittest.TestCase):
    def test_thruster_fault_and_degraded_mode_block_propulsion(self):
        supervisor = SafetySupervisor()
        state = healthy_state()
        state.faults.thruster_fault = True
        cmd = supervisor.evaluate_command(OperatingMode.DEGRADED, state, ActuatorCommand("AI", thrust=0.4))
        self.assertFalse(cmd.approved)
        self.assertEqual(cmd.thrust, 0.0)
        self.assertIn("Propulsion blocked", cmd.reason_code)
        state.faults.thruster_fault = False
        cmd = supervisor.evaluate_command(OperatingMode.DEGRADED, state, ActuatorCommand("AI", thrust=0.4))
        self.assertFalse(cmd.approved)
        self.assertEqual(cmd.thrust, 0.0)
        self.assertIn("DEGRADED", cmd.reason_code)

    def test_safe_mode_zero_rates_do_not_create_spin(self):
        result = CommandArbiter(SafetySupervisor()).process_tick(
            OperatingMode.SAFE_MODE, healthy_state(), ActuatorCommand("AI", roll_tq=1.0)
        )
        self.assertTrue(result.approved)
        self.assertEqual((result.roll_tq, result.pitch_tq, result.yaw_tq), (0.0, 0.0, 0.0))
        self.assertEqual(result.thrust, 0.0)

    def test_fdir_faults_decay_and_authenticated_recovery_is_required(self):
        fdir = FDIRStateMachine()
        state = healthy_state()
        state.faults.wheel_fault = True
        for _ in range(3):
            fdir.update(state)
        self.assertEqual(fdir.mode, OperatingMode.DEGRADED)
        self.assertFalse(fdir.ground_command_clear_fault("wheel_fault"))
        self.assertTrue(fdir.ground_command_clear_fault("wheel_fault", authenticated=True))
        state.faults.wheel_fault = False
        for _ in range(3):
            fdir.update(state)
        self.assertEqual(fdir.mode, OperatingMode.NOMINAL)

    def test_observation_adapter_matches_policy_schema(self):
        observation = AIObservationAdapter().build_observation(healthy_state(), False)
        self.assertEqual(observation["local"].shape, (49,))
        self.assertEqual(observation["global"].shape, (47,))

    def test_scheduler_rejects_invalid_task_rate(self):
        scheduler = CyclicScheduler(hz=10)
        with self.assertRaises(ValueError):
            scheduler.add_task("bad", lambda: None, 3)


if __name__ == "__main__":
    unittest.main()
