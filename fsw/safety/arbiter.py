"""
fsw/safety/arbiter.py
---------------------
Command Arbiter with Safety Shield / Action Governor.
Prioritizes and routes commands.
Priority: 1. Hardware/Safety Override -> 2. FDIR Recovery -> 3. AI Planner

Mode hierarchy (highest to lowest authority):
  EOL_DEORBIT > SAFE_MODE > DEGRADED ≈ NOMINAL > RECOVERY > BOOT/COMMISSIONING/GROUND_HOLD

Safety Shield:
  - Rate limiting on actuator commands
  - Hard bounds enforcement
  - Anomaly detection on command sequences
  - Dead-man switch integration
"""

import logging
import time
from typing import Tuple, Optional
from dataclasses import dataclass
from collections import deque
from fsw.fdir.state_machine import OperatingMode
from fsw.safety.supervisor import SafetySupervisor
from fsw.core.telemetry import SubsystemState
from fsw.core.commands import ActuatorCommand
from fsw.gnc.classical import ClassicalControllers

logger = logging.getLogger(__name__)


@dataclass
class SafetyShieldConfig:
    """Configuration for the Safety Shield / Action Governor."""
    max_thrust_rate: float = 0.5          # Max change in thrust per tick
    max_torque_rate: float = 0.3          # Max change in torque per tick
    max_relay_rate: float = 1.0           # Max change in relay action per tick
    thrust_bounds: Tuple[float, float] = (-1.0, 1.0)
    torque_bounds: Tuple[float, float] = (-1.0, 1.0)
    relay_bounds: Tuple[float, float] = (0.0, 1.0)
    anomaly_window: int = 10              # Ticks to track for anomaly detection
    max_consecutive_rejects: int = 3      # Max consecutive rejects before safe mode
    deadman_timeout_sec: float = 5.0      # Max time without valid command


class SafetyShield:
    """
    Safety Shield / Action Governor.
    Enforces hard limits, rate limits, and anomaly detection on actuator commands.
    """
    
    def __init__(self, config: SafetyShieldConfig = None):
        self.config = config or SafetyShieldConfig()
        self.last_cmd: Optional[ActuatorCommand] = None
        self.consecutive_rejects = 0
        self.command_history = deque(maxlen=self.config.anomaly_window)
        self.last_valid_cmd_time = time.time()
        
    def filter(self, cmd: ActuatorCommand, state: SubsystemState) -> Tuple[ActuatorCommand, bool, str]:
        """
        Filter and validate command. Returns (filtered_cmd, approved, reason).
        """
        # Hard bounds enforcement
        filtered = self._enforce_bounds(cmd)
        
        # Rate limiting
        filtered = self._rate_limit(filtered)
        
        # Physics validation
        approved, reason = self._validate_physics(filtered, state)
        
        # Anomaly detection
        if approved:
            anomaly, anomaly_reason = self._detect_anomaly(filtered)
            if anomaly:
                approved = False
                reason = anomaly_reason
        
        # Dead-man switch check
        if time.time() - self.last_valid_cmd_time > self.config.deadman_timeout_sec:
            approved = False
            reason = "DEADMAN_TIMEOUT"
        
        if approved:
            self.consecutive_rejects = 0
            self.last_valid_cmd_time = time.time()
            self.last_cmd = filtered
            self.command_history.append((time.time(), filtered))
        else:
            self.consecutive_rejects += 1
            
        return filtered, approved, reason
    
    def _enforce_bounds(self, cmd: ActuatorCommand) -> ActuatorCommand:
        """Clamp all actuator values to hard bounds."""
        filtered = ActuatorCommand(source=cmd.source)
        filtered.thrust = max(self.config.thrust_bounds[0], min(self.config.thrust_bounds[1], cmd.thrust))
        filtered.roll_tq = max(self.config.torque_bounds[0], min(self.config.torque_bounds[1], cmd.roll_tq))
        filtered.pitch_tq = max(self.config.torque_bounds[0], min(self.config.torque_bounds[1], cmd.pitch_tq))
        filtered.yaw_tq = max(self.config.torque_bounds[0], min(self.config.torque_bounds[1], cmd.yaw_tq))
        filtered.relay_action = max(self.config.relay_bounds[0], min(self.config.relay_bounds[1], cmd.relay_action))
        filtered.hohmann = max(0.0, min(1.0, cmd.hohmann))
        filtered.avoidance = max(0.0, min(1.0, cmd.avoidance))
        filtered.deorbit = max(0.0, min(1.0, cmd.deorbit))
        filtered.command_id = cmd.command_id
        return filtered
    
    def _rate_limit(self, cmd: ActuatorCommand) -> ActuatorCommand:
        """Limit rate of change between consecutive commands."""
        if self.last_cmd is None:
            return cmd
            
        filtered = ActuatorCommand(source=cmd.source)
        filtered.thrust = self._clamp_rate(cmd.thrust, self.last_cmd.thrust, self.config.max_thrust_rate)
        filtered.roll_tq = self._clamp_rate(cmd.roll_tq, self.last_cmd.roll_tq, self.config.max_torque_rate)
        filtered.pitch_tq = self._clamp_rate(cmd.pitch_tq, self.last_cmd.pitch_tq, self.config.max_torque_rate)
        filtered.yaw_tq = self._clamp_rate(cmd.yaw_tq, self.last_cmd.yaw_tq, self.config.max_torque_rate)
        filtered.relay_action = self._clamp_rate(cmd.relay_action, self.last_cmd.relay_action, self.config.max_relay_rate)
        filtered.hohmann = cmd.hohmann
        filtered.avoidance = cmd.avoidance
        filtered.deorbit = cmd.deorbit
        filtered.command_id = cmd.command_id
        return filtered
    
    def _clamp_rate(self, new_val: float, old_val: float, max_rate: float) -> float:
        diff = new_val - old_val
        if abs(diff) <= max_rate:
            return new_val
        return old_val + (max_rate if diff > 0 else -max_rate)
    
    def _validate_physics(self, cmd: ActuatorCommand, state: SubsystemState) -> Tuple[bool, str]:
        """Validate command against current physical state."""
        # No thrust if fuel depleted
        if abs(cmd.thrust) > 0.1 and state.adcs.fuel_percent < 1.0:
            return False, "INSUFFICIENT_FUEL"
        
        # No high torque if wheel fault
        max_torque = max(abs(cmd.roll_tq), abs(cmd.pitch_tq), abs(cmd.yaw_tq))
        if max_torque > 0.3 and state.faults.wheel_fault:
            return False, "WHEEL_FAULT_LIMIT"
        
        # No relay if no ISL or buffer empty
        if cmd.relay_action > 0.1:
            has_isl = any(n.isl_active and n.link_quality > 0.3 for n in state.neighbors)
            if not has_isl or state.comms.data_buffer_gb < 0.1:
                return False, "RELAY_NOT_FEASIBLE"
        
        # No avoidance if no debris threat
        if cmd.avoidance > 0.1:
            has_debris = any(d < 10.0 for d in state.global_fleet.debris_positions)
            if not has_debris:
                return False, "NO_DEBRIS_THREAT"
        
        return True, "PHYSICS_OK"
    
    def _detect_anomaly(self, cmd: ActuatorCommand) -> Tuple[bool, str]:
        """Detect anomalous command patterns."""
        if len(self.command_history) < 3:
            return False, ""
        
        # Detect oscillation (rapid sign changes)
        thrust_signs = [c.thrust for _, c in self.command_history]
        sign_changes = sum(1 for i in range(1, len(thrust_signs)) 
                          if thrust_signs[i] * thrust_signs[i-1] < -0.1)
        if sign_changes >= 3:
            return True, "THRUST_OSCILLATION"
        
        # Detect stuck at extreme
        recent_thrust = [c.thrust for _, c in list(self.command_history)[-5:]]
        if all(abs(t) > 0.9 for t in recent_thrust):
            return True, "SUSTAINED_MAX_THRUST"
        
        return False, ""
    
    def should_trigger_safe_mode(self) -> bool:
        """Check if consecutive rejects exceed threshold."""
        return self.consecutive_rejects >= self.config.max_consecutive_rejects
    
    def reset(self):
        """Reset shield state."""
        self.last_cmd = None
        self.consecutive_rejects = 0
        self.command_history.clear()
        self.last_valid_cmd_time = time.time()


class CommandArbiter:
    def __init__(self, supervisor: SafetySupervisor):
        self.supervisor = supervisor
        self._eol_burn_executed = False
        self.safety_shield = SafetyShield()

    def process_tick(
        self,
        mode: OperatingMode,
        state: SubsystemState,
        ai_cmd: ActuatorCommand,
    ) -> ActuatorCommand:
        """
        Determine final commands for actuators for this tick.
        Returns an ActuatorCommand with approved=True if the command should be sent.
        """
        final_cmd = ActuatorCommand(source="ARBITER")

# ── EOL DE-ORBIT — highest authority, AI completely bypassed ─────────
        if mode == OperatingMode.EOL_DEORBIT:
            altitude_km = state.adcs.position_eci_km[0] / 100.0  # mock mapping
            if altitude_km <= 0.0 or altitude_km > 50000.0:
                altitude_km = 550.0

            burn = ClassicalControllers.deorbit_burn(
                altitude_km=altitude_km,
                delta_v_remaining_ms=state.adcs.delta_v_remaining,
            )

            final_cmd = ActuatorCommand(source="EOL_DEORBIT_SEQUENCE")
            if burn["feasible"]:
                final_cmd.thrust    = -1.0
                final_cmd.deorbit   = 1.0
                final_cmd.approved  = True
                final_cmd.reason_code = f"EOL_BURN dv={burn['dv_required_ms']:.1f}m/s alt={altitude_km:.0f}km"
                if not self._eol_burn_executed:
                    logger.critical("ARBITER: EOL DE-ORBIT BURN COMMANDED. ΔV=%.1f m/s altitude=%.0f km budget=%.1f m/s",
                                  burn["dv_required_ms"], altitude_km, state.adcs.delta_v_remaining)
                    self._eol_burn_executed = True
            else:
                final_cmd.thrust = 0.0
                final_cmd.deorbit = 0.0
                final_cmd.approved = False
                final_cmd.reason_code = f"EOL_INSUFFICIENT_DV need={burn['dv_required_ms']:.1f}m/s have={state.adcs.delta_v_remaining:.1f}m/s"
                logger.warning("ARBITER: EOL burn NOT feasible — coasting.")

            final_cmd.roll_tq = 0.0
            final_cmd.pitch_tq = 0.0
            final_cmd.yaw_tq = 0.0
            final_cmd.relay_action = 0.0
            final_cmd.hohmann = 0.0
            final_cmd.avoidance = 0.0
            # Apply safety shield even for EOL (rate limits still apply)
            shielded_cmd, approved, _ = self.safety_shield.filter(final_cmd, state)
            shielded_cmd.approved = approved
            return shielded_cmd

        # ── SAFE MODE — classical controllers only, AI bypassed ──────────────
        if mode == OperatingMode.SAFE_MODE:
            logger.debug("ARBITER: Classical control (SAFE_MODE).")
            roll_tq, pitch_tq, yaw_tq = ClassicalControllers.detumble(
                state.adcs.rates_deg_s
            )
            eclipse_flag = state.eps.solar_power_w == 0.0
            _, sp_pitch, _ = ClassicalControllers.sun_pointing(
                state.adcs.attitude_deg, eclipse=eclipse_flag
            )
            final_cmd = ActuatorCommand(source="CLASSICAL_SAFE_MODE")
            final_cmd.thrust = 0.0
            final_cmd.roll_tq = roll_tq
            final_cmd.pitch_tq = pitch_tq + sp_pitch
            final_cmd.yaw_tq = yaw_tq
            final_cmd.relay_action = 0.0
            final_cmd.hohmann = 0.0
            final_cmd.avoidance = 0.0
            final_cmd.deorbit = 0.0
            final_cmd.approved = True
            final_cmd.reason_code = "SAFE_MODE_OVERRIDE"
            # Apply safety shield
            shielded_cmd, approved, _ = self.safety_shield.filter(final_cmd, state)
            shielded_cmd.approved = approved
            return shielded_cmd

        # ── NOMINAL / DEGRADED — AI proposes, supervisor validates ───────────
        if mode in [OperatingMode.NOMINAL, OperatingMode.DEGRADED]:
            logger.debug("ARBITER: Evaluating AI proposed action.")
            eval_cmd = self.supervisor.evaluate_command(mode, state, ai_cmd)

            if eval_cmd.approved:
                # Apply Safety Shield before sending to actuators
                shielded_cmd, approved, reason = self.safety_shield.filter(eval_cmd, state)
                if approved:
                    shielded_cmd.source = eval_cmd.source
                    shielded_cmd.reason_code = eval_cmd.reason_code
                    return shielded_cmd
                else:
                    logger.warning("ARBITER: Safety Shield Rejected -> %s", reason)

            logger.warning("ARBITER: AI Command Rejected -> %s", eval_cmd.reason_code)
            # Fallback to zero command (coasting)
            final_cmd = ActuatorCommand(source="ARBITER_FALLBACK")
            final_cmd.approved = False
            final_cmd.reason_code = eval_cmd.reason_code
            return final_cmd

        # ── BOOT / COMMISSIONING / GROUND_HOLD / RECOVERY — coast ───────────
        final_cmd = ActuatorCommand(source=mode.name)
        final_cmd.approved = False
        final_cmd.reason_code = f"{mode.name}_COAST"
        return final_cmd

    def check_safety_shield(self) -> bool:
        """Check if safety shield should trigger safe mode."""
        return self.safety_shield.should_trigger_safe_mode()
    
    def reset_safety_shield(self):
        """Reset safety shield state."""
        self.safety_shield.reset()

