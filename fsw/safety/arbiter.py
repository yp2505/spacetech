"""
fsw/safety/arbiter.py
---------------------
Command Arbiter.
Prioritizes and routes commands.
Priority: 1. Hardware/Safety Override -> 2. FDIR Recovery -> 3. AI Planner
"""

import logging
from typing import Tuple
from fsw.fdir.state_machine import OperatingMode
from fsw.safety.supervisor import SafetySupervisor
from fsw.core.telemetry import SubsystemState
from fsw.core.commands import ActuatorCommand
from fsw.gnc.classical import ClassicalControllers

class CommandArbiter:
    def __init__(self, supervisor: SafetySupervisor):
        self.supervisor = supervisor

    def process_tick(self, 
                     mode: OperatingMode, 
                     state: SubsystemState, 
                     ai_cmd: ActuatorCommand) -> ActuatorCommand:
        """
        Determine final commands for actuators for this tick.
        """
        final_cmd = ActuatorCommand(source="ARBITER")

        if mode == OperatingMode.SAFE_MODE:
            # In safe mode, we execute a classical deterministic detumble/sun-pointing
            # Ignoring AI completely.
            logging.debug("ARBITER: Executing Classical Detumble (SAFE_MODE).")
            roll_tq, pitch_tq, yaw_tq = ClassicalControllers.detumble(state.adcs.rates_deg_s)
            
            final_cmd.source = "CLASSICAL_SAFE_MODE"
            final_cmd.thrust = 0.0 # Never thrust in safe mode
            final_cmd.roll_tq = roll_tq
            final_cmd.pitch_tq = pitch_tq
            final_cmd.yaw_tq = yaw_tq
            final_cmd.approved = True
            final_cmd.reason_code = "SAFE_MODE_OVERRIDE"

        elif mode in [OperatingMode.NOMINAL, OperatingMode.DEGRADED]:
            # AI proposes action
            logging.debug("ARBITER: Evaluating AI Proposed Action.")
            eval_cmd = self.supervisor.evaluate_command(mode, state, ai_cmd)
            
            if eval_cmd.approved:
                final_cmd = eval_cmd
            else:
                logging.warning(f"ARBITER: AI Command Rejected -> {eval_cmd.reason_code}")
                # Fallback to zero command (coasting)
                final_cmd.source = "ARBITER_FALLBACK"
                final_cmd.approved = False
                final_cmd.reason_code = eval_cmd.reason_code
                final_cmd.thrust, final_cmd.roll_tq, final_cmd.pitch_tq, final_cmd.yaw_tq = 0.0, 0.0, 0.0, 0.0
                
        else:
            # BOOT, COMM, GROUND_HOLD
            final_cmd.source = mode.name
            final_cmd.approved = False
            final_cmd.reason_code = f"{mode.name}_COAST"

        return final_cmd
