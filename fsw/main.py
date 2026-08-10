"""
fsw/main.py
-----------
Main Entry Point for the Onboard Autonomy Agent.
Binds the Scheduler, FDIR, Safety Supervisor, and AI Brain together.
"""

import logging
import sys
from fsw.core.scheduler import CyclicScheduler
from fsw.core.telemetry import SubsystemState
from fsw.fdir.state_machine import FDIRStateMachine, OperatingMode
from fsw.fdir.monitors import TelemetryMonitor
from fsw.safety.supervisor import SafetySupervisor
from fsw.safety.arbiter import CommandArbiter
from fsw.gnc.estimators import AttitudeEstimator
from fsw.ai_brain.agent import AIBrain
from fsw.ai_brain.validation import AIObservationValidator
from fsw.core.commands import ActuatorCommand

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class OnboardAutonomyAgent:
    def __init__(self, hal_eps, hal_thermal, hal_adcs, hal_radio, hal_fault, sim_env_ref=None, agent_idx=0):
        # Hardware Abstraction Layer references
        self.eps = hal_eps
        self.thermal = hal_thermal
        self.adcs = hal_adcs
        self.radio = hal_radio
        self.fault = hal_fault
        self.sim_env_ref = sim_env_ref
        self.agent_idx = agent_idx
        
        # Flight Software Components
        self.scheduler = CyclicScheduler(hz=10)
        self.fdir_fsm = FDIRStateMachine()
        self.monitor = TelemetryMonitor()
        self.supervisor = SafetySupervisor()
        self.arbiter = CommandArbiter(self.supervisor)
        self.estimator = AttitudeEstimator()
        
        # AI Brain
        self.ai = AIBrain(config=getattr(sim_env_ref, "config", None))
        self.ai_validator = AIObservationValidator()

        # Current State
        self.state = SubsystemState()
        
        # Register Tasks
        self.scheduler.add_task("TELEMETRY", self.task_read_telemetry, frequency_hz=10)
        self.scheduler.add_task("FDIR", self.task_fdir, frequency_hz=5)
        self.scheduler.add_task("GNC_AI", self.task_gnc_ai, frequency_hz=2)

    def task_read_telemetry(self):
        """Read all hardware interfaces and construct SubsystemState."""
        self.state.eps = self.eps.get_telemetry()
        self.state.thermal = self.thermal.get_telemetry()
        raw_adcs = self.adcs.get_telemetry()
        self.state.adcs = self.estimator.update(raw_adcs)
        self.state.comms = self.radio.get_telemetry()
        self.state.faults = self.fault.get_telemetry()
        
        # In a real system, these would come via ISL mesh packets received by the radio
        self.state.neighbors = self.radio.get_neighbor_telemetry()
        self.state.global_fleet = self.radio.get_global_telemetry()
        enrich = getattr(self.adcs, "enrich_state", None)
        if callable(enrich):
            enrich(self.state)

    def task_fdir(self):
        """Run telemetry monitors and FDIR state machine."""
        alarms = self.monitor.check(self.state)
        prev_mode = self.fdir_fsm.mode
        new_mode = self.fdir_fsm.update(self.state)
        if new_mode != prev_mode:
            logging.warning(f"FDIR MODE TRANSITION: {prev_mode.name} -> {new_mode.name}")

    def task_gnc_ai(self):
        """Query AI, validate, arbitrate, and send commands to HAL."""
        mode = self.fdir_fsm.mode
        
        # 1. AI Proposal
        ai_cmd = ActuatorCommand(source="AI_PLANNER")
        if self.ai_validator.validate(self.state):
            t, r, p, y = self.ai.propose_action(
                self.state, in_recovery=self.fdir_fsm.mode == OperatingMode.RECOVERY
            )
            ai_cmd.thrust, ai_cmd.roll_tq, ai_cmd.pitch_tq, ai_cmd.yaw_tq = t, r, p, y
            
        # 2. Arbiter (determines final command based on mode and safety rules)
        final_cmd = self.arbiter.process_tick(mode, self.state, ai_cmd)
        
        # 3. Send to HAL
        final_cmd.acknowledged = bool(self.adcs.command_actuators(final_cmd))
        if final_cmd.approved and not final_cmd.acknowledged:
            logging.error("FSW command was not acknowledged by the HAL: %s", final_cmd.command_id)
        
        # Logging (Optional, but good for SIL output)
        logging.info(f"FSW Actuator Command: [{final_cmd.thrust}, {final_cmd.roll_tq}, {final_cmd.pitch_tq}, {final_cmd.yaw_tq}] (Source: {final_cmd.source}, Reason: {final_cmd.reason_code})")

    def run(self, max_ticks: int = 0):
        logging.info("Onboard Autonomy Agent Booting...")
        self.scheduler.run(max_ticks=max_ticks)
