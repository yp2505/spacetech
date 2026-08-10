"""
fsw/safety/supervisor.py
------------------------
Safety Supervisor.
Evaluates proposed actions against deterministic safety limits.
Has final authority to reject or modify commands before they reach the HAL.
"""

import logging
from fsw.core.telemetry import SubsystemState
from fsw.core.commands import ActuatorCommand
from fsw.fdir.state_machine import OperatingMode

class SafetySupervisor:
    def __init__(self):
        # Configurable limits
        self.min_battery_percent = 15.0
        self.max_battery_temp_c = 45.0
        self.max_rate_deg_s = 5.0
        self.max_thrust_cmd = 1.0

    def evaluate_command(self, 
                         mode: OperatingMode, 
                         state: SubsystemState, 
                         cmd: ActuatorCommand) -> ActuatorCommand:
        """
        Evaluate and potentially clip/reject a command.
        Returns the evaluated ActuatorCommand with approval status and reason.
        """
        cmd.approved = False

        # 1. Stale Telemetry
        if state.any_stale:
            cmd.reason_code = "REJECTED: Stale telemetry"
            cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq = 0.0, 0.0, 0.0, 0.0
            return cmd

        # 2. State/Mode Rules
        if mode == OperatingMode.SAFE_MODE:
            cmd.reason_code = "REJECTED: Spacecraft in SAFE_MODE. Actuators locked to classical detumble."
            cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq = 0.0, 0.0, 0.0, 0.0
            return cmd
            
        if mode in [OperatingMode.BOOT, OperatingMode.COMMISSIONING]:
            cmd.reason_code = "REJECTED: Autonomous maneuvers disabled during BOOT/COMM."
            cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq = 0.0, 0.0, 0.0, 0.0
            return cmd

        # 3. Thruster Faults / Degraded mode propulsion block
        if cmd.thrust != 0.0:
            if state.faults.thruster_fault:
                cmd.reason_code = "REJECTED: Propulsion blocked due to active thruster fault."
                cmd.thrust = 0.0
                return cmd
            if mode == OperatingMode.DEGRADED:
                # A latched degraded condition needs an explicit recovery path,
                # not continued autonomous propulsion.
                cmd.reason_code = "REJECTED: Propulsion blocked in DEGRADED mode."
                cmd.thrust = 0.0
                return cmd

        if state.faults.wheel_fault and any((cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq)):
            cmd.reason_code = "REJECTED: Attitude actuation blocked due to wheel fault."
            cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq = 0.0, 0.0, 0.0, 0.0
            return cmd

        if cmd.is_expired:
            cmd.reason_code = "REJECTED: Command expired before arbitration."
            cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq = 0.0, 0.0, 0.0, 0.0
            return cmd

        # 4. Resource Rules
        if state.eps.battery_charge_percent < self.min_battery_percent:
            cmd.reason_code = f"REJECTED: Battery critically low ({state.eps.battery_charge_percent:.1f}%)."
            cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq = 0.0, 0.0, 0.0, 0.0
            return cmd

        if state.thermal.battery_temp_c > self.max_battery_temp_c:
            cmd.reason_code = f"REJECTED: Thermal limit exceeded ({state.thermal.battery_temp_c:.1f}C)."
            cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq = 0.0, 0.0, 0.0, 0.0
            return cmd

        # 5. Kinematic Rules (Prevent spin-outs)
        if any(abs(rate) > self.max_rate_deg_s for rate in state.adcs.rates_deg_s):
            cmd.reason_code = "REJECTED: Angular rates exceed safety limits."
            cmd.thrust, cmd.roll_tq, cmd.pitch_tq, cmd.yaw_tq = 0.0, 0.0, 0.0, 0.0
            return cmd

        # 6. Command Clipping
        orig_thrust = cmd.thrust
        cmd.thrust = max(-self.max_thrust_cmd, min(self.max_thrust_cmd, cmd.thrust))
        if cmd.thrust != orig_thrust:
            logging.info(f"SAFETY: Thrust clipped from {orig_thrust} to {cmd.thrust}")
            
        cmd.roll_tq = max(-1.0, min(1.0, cmd.roll_tq))
        cmd.pitch_tq = max(-1.0, min(1.0, cmd.pitch_tq))
        cmd.yaw_tq = max(-1.0, min(1.0, cmd.yaw_tq))

        cmd.approved = True
        cmd.reason_code = "APPROVED"
        return cmd
