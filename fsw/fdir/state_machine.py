"""
fsw/fdir/state_machine.py
-------------------------
Fault Detection, Isolation, and Recovery (FDIR) State Machine.
Determines the operational mode based on confidence-voted telemetry.
"""

from enum import Enum
import logging

class OperatingMode(Enum):
    BOOT = 0
    COMMISSIONING = 1
    NOMINAL = 2
    MISSION_EXECUTION = 3
    DEGRADED = 4
    SAFE_MODE = 5
    RECOVERY = 6
    GROUND_HOLD = 7
    EMERGENCY = 8
    EOL_DEORBIT = 9   # End-of-Life: battery critically degraded → execute de-orbit

class FDIRStateMachine:
    def __init__(self):
        self.mode = OperatingMode.BOOT
        self.fault_counters = {
            "low_battery": 0,
            "over_temp": 0,
            "wheel_fault": 0,
            "thruster_fault": 0,
            "sensor_fault": 0,
            "stale_telemetry": 0,
            "eol_battery": 0,      # persistent critical-low battery counter
        }
        self.persistence_threshold = 3  # Require 3 consecutive hits to latch fault
        self.eol_threshold = 20         # 20 consecutive cycles at < 15% → EOL
        self._latched_faults = set()

    def update(self, state: 'SubsystemState') -> OperatingMode:
        """Evaluate subsystem state and transition operating modes."""

        # ── EOL is terminal — once entered we never leave ─────────────────────
        if self.mode == OperatingMode.EOL_DEORBIT:
            return self.mode

        # 1. Evaluate primitive faults and decay when clear
        if state.eps.battery_charge_percent < 20.0:
            self.fault_counters["low_battery"] += 1
        else:
            self.fault_counters["low_battery"] = max(0, self.fault_counters["low_battery"] - 1)

        if state.thermal.battery_temp_c > 50.0:
            self.fault_counters["over_temp"] += 1
        else:
            self.fault_counters["over_temp"] = max(0, self.fault_counters["over_temp"] - 1)

        if state.faults.wheel_fault:
            self.fault_counters["wheel_fault"] += 1
        else:
            self.fault_counters["wheel_fault"] = max(0, self.fault_counters["wheel_fault"] - 1)

        if state.faults.thruster_fault:
            self.fault_counters["thruster_fault"] += 1
        else:
            self.fault_counters["thruster_fault"] = max(0, self.fault_counters["thruster_fault"] - 1)

        if state.faults.sensor_fault:
            self.fault_counters["sensor_fault"] += 1
        else:
            self.fault_counters["sensor_fault"] = max(0, self.fault_counters["sensor_fault"] - 1)

        if state.any_stale:
            self.fault_counters["stale_telemetry"] += 1
        else:
            self.fault_counters["stale_telemetry"] = max(0, self.fault_counters["stale_telemetry"] - 1)

        # ── EOL battery degradation counter ──────────────────────────────────
        # If battery is critically low (< 15%) for eol_threshold consecutive
        # cycles we declare End-of-Life and initiate the de-orbit sequence.
        EOL_BAT_THRESH = 15.0
        if state.eps.battery_charge_percent < EOL_BAT_THRESH:
            self.fault_counters["eol_battery"] += 1
            if self.fault_counters["eol_battery"] >= self.eol_threshold:
                logging.critical(
                    "FDIR: EOL condition confirmed — battery < %.0f%% for %d consecutive cycles. "
                    "Initiating EOL_DEORBIT sequence.",
                    EOL_BAT_THRESH, self.eol_threshold
                )
                self.mode = OperatingMode.EOL_DEORBIT
                return self.mode
        else:
            # Reset EOL counter only if battery recovers above threshold
            self.fault_counters["eol_battery"] = max(
                0, self.fault_counters["eol_battery"] - 1
            )

        # 2. Latch persistent faults
        for fault_name, count in self.fault_counters.items():
            if (fault_name != "eol_battery"
                    and count >= self.persistence_threshold
                    and fault_name not in self._latched_faults):
                logging.error("FDIR: Latched critical fault: %s", fault_name)
                self._latched_faults.add(fault_name)

        # 3. Mode transitions
        if self.mode in [OperatingMode.BOOT, OperatingMode.COMMISSIONING]:
            if not self._latched_faults:
                self.mode = OperatingMode.NOMINAL

        if self._latched_faults:
            if {"low_battery", "over_temp", "stale_telemetry", "sensor_fault"} & self._latched_faults:
                if self.mode != OperatingMode.SAFE_MODE:
                    logging.critical("FDIR: Escalating to SAFE_MODE due to critical resource faults.")
                    self.mode = OperatingMode.SAFE_MODE
            else:
                if self.mode not in [OperatingMode.SAFE_MODE, OperatingMode.RECOVERY, OperatingMode.GROUND_HOLD]:
                    logging.warning("FDIR: Entering DEGRADED mode due to component fault.")
                    self.mode = OperatingMode.DEGRADED

        elif self.mode == OperatingMode.RECOVERY:
            # Only return to nominal when all fault counters are zero and telemetry fresh
            if sum(self.fault_counters.values()) == 0 and not state.any_stale:
                logging.info("FDIR: Recovery criteria met. Returning to NOMINAL.")
                self.mode = OperatingMode.NOMINAL

        elif self.mode == OperatingMode.DEGRADED and not self._latched_faults:
            logging.info("FDIR: Degraded condition cleared. Returning to NOMINAL.")
            self.mode = OperatingMode.NOMINAL

        return self.mode

    def ground_command_clear_fault(self, fault_name: str, *, authenticated: bool = False) -> bool:
        """Clear a latched fault only after an authenticated ground command."""
        if not authenticated or fault_name not in self._latched_faults:
            logging.warning("FDIR: rejected unauthenticated or invalid fault-clear request.")
            return False
        self._latched_faults.remove(fault_name)
        self.fault_counters[fault_name] = 0
        logging.info(f"FDIR: Ground cleared fault {fault_name}")
        if not self._latched_faults:
            self.mode = OperatingMode.RECOVERY
        return True
