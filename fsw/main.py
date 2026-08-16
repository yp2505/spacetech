"""
fsw/main.py
-----------
Main Entry Point for the Onboard Autonomy Agent.
Binds the Scheduler, FDIR, Safety Supervisor, AI Brain, and Mission Commander together.

ARTEMIS System Identity
=======================
This is ARTEMIS — Autonomous Real-Time Embedded Mission Intelligence System.
Operating at 550 km LEO. Orbital period ≈95 min. ΔV budget 1000 m/s.
Priority: Safety > Health > Mission > Constellation > Longevity.
All ISL data relay packets are encrypted (rolling-XOR + SHA-256 key derivation).
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
from fsw.ai_brain.commander import MissionCommander
from fsw.core.commands import ActuatorCommand
from fsw.core.watchdog import WatchdogManager, WatchdogAction

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
        
        # Watchdog Manager
        self.watchdog = WatchdogManager(default_timeout_sec=2.0)
        self.watchdog.register_task("TELEMETRY", timeout_sec=0.5, action=WatchdogAction.TRIGGER_SAFE_MODE)
        self.watchdog.register_task("FDIR", timeout_sec=1.0, action=WatchdogAction.TRIGGER_SAFE_MODE)
        self.watchdog.register_task("GNC_AI", timeout_sec=1.0, action=WatchdogAction.RESET_AI_BRAIN)
        self.watchdog.register_ai_heartbeat(timeout_sec=5.0)
        
        # Flight Software Components
        self.scheduler = CyclicScheduler(hz=10)
        self.fdir_fsm = FDIRStateMachine()
        self.monitor = TelemetryMonitor()
        self.supervisor = SafetySupervisor()
        self.arbiter = CommandArbiter(self.supervisor)
        self.estimator = AttitudeEstimator()
        
        # AI Brain with watchdog integration
        self.ai = AIBrain(config=getattr(sim_env_ref, "config", None), watchdog=self.watchdog)
        self.ai_validator = AIObservationValidator()

        # Mission Commander (Hierarchical RL layer 1 — rule-based)
        _config = getattr(sim_env_ref, "config", None)
        _data_cap = getattr(_config, "data_capacity_gb", 100.0) if _config else 100.0
        self.commander = MissionCommander(data_capacity_gb=_data_cap)
        self._gnc_tick = 0   # tracks how often commander evaluate() should run

        # Current State
        self.state = SubsystemState()
        
        # Register Tasks
        self.scheduler.add_task("TELEMETRY", self.task_read_telemetry, frequency_hz=10)
        self.scheduler.add_task("FDIR", self.task_fdir, frequency_hz=5)
        self.scheduler.add_task("GNC_AI", self.task_gnc_ai, frequency_hz=2)
        self.scheduler.add_task("WATCHDOG", self.task_watchdog, frequency_hz=10)

    def task_read_telemetry(self):
        """Read all hardware interfaces and construct SubsystemState."""
        self.watchdog.kick("TELEMETRY")
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
        self.watchdog.kick("FDIR")
        alarms = self.monitor.check(self.state)
        prev_mode = self.fdir_fsm.mode
        new_mode = self.fdir_fsm.update(self.state)
        if new_mode != prev_mode:
            logging.warning(f"FDIR MODE TRANSITION: {prev_mode.name} -> {new_mode.name}")
        
        # Check if arbiter safety shield wants safe mode
        if self.arbiter.check_safety_shield():
            logging.critical("ARBITER SAFETY SHIELD: Triggering SAFE_MODE due to consecutive rejects")
            self.fdir_fsm.mode = OperatingMode.SAFE_MODE
            self.arbiter.reset_safety_shield()

    def task_gnc_ai(self):
        """Run Mission Commander, query AI, validate, arbitrate, send commands to HAL."""
        self.watchdog.kick("GNC_AI")
        mode = self.fdir_fsm.mode
        self._gnc_tick += 1

        # ── Step 1: Mission Commander sets high-level goal (every 10 GNC ticks = ~5 s) ──
        if self._gnc_tick % 10 == 0:
            goal = self.commander.evaluate(self.state, mode)
            self.state.commander_goal = goal

        # ── Step 2: AI Proposal ─────────────────────────────────────────────────
        ai_cmd = ActuatorCommand(source="AI_PLANNER")
        if self.ai_validator.validate(self.state):
            t, r, p, y, relay, hohm, avoid, deorb = self.ai.propose_action(
                self.state, in_recovery=self.fdir_fsm.mode == OperatingMode.RECOVERY
            )
            ai_cmd.thrust, ai_cmd.roll_tq, ai_cmd.pitch_tq, ai_cmd.yaw_tq = t, r, p, y
            ai_cmd.relay_action, ai_cmd.hohmann, ai_cmd.avoidance, ai_cmd.deorbit = relay, hohm, avoid, deorb

        # ── Step 3: Arbiter (determines final command based on mode and safety rules) ──
        final_cmd = self.arbiter.process_tick(mode, self.state, ai_cmd)

        # ── Step 4: Send to HAL ───────────────────────────────────────────────
        final_cmd.acknowledged = bool(self.adcs.command_actuators(final_cmd))
        if final_cmd.approved and not final_cmd.acknowledged:
            logging.error("FSW command was not acknowledged by the HAL: %s", final_cmd.command_id)

        logging.info(
            "FSW [%s] Mode=%-14s Cmd=[%.2f, %.2f, %.2f, %.2f] relay=%.2f deorbit=%.2f  (src=%s reason=%s)",
            f"tick={self._gnc_tick:05d}",
            mode.name,
            final_cmd.thrust, final_cmd.roll_tq, final_cmd.pitch_tq, final_cmd.yaw_tq,
            final_cmd.relay_action, final_cmd.deorbit,
            final_cmd.source, final_cmd.reason_code,
        )

    def task_watchdog(self):
        """Periodic watchdog check."""
        status = self.watchdog.check()
        # Log any dead tasks
        for task_name, is_alive in status.items():
            if not is_alive:
                logging.error(f"WATCHDOG: Task '{task_name}' is DEAD")

    def run(self, max_ticks: int = 0):
        logging.info("Onboard Autonomy Agent Booting...")
        self.scheduler.run(max_ticks=max_ticks)
