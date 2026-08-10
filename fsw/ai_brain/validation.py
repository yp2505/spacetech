"""
fsw/ai_brain/validation.py
--------------------------
AI Sandbox Validation.
Out-of-Distribution (OOD) detection for the neural network inputs.
"""

from fsw.core.telemetry import SubsystemState
import logging

class AIObservationValidator:
    def __init__(self):
        # We define bounds of what the AI was trained on
        self.bounds = {
            "battery_min": 10.0,
            "battery_max": 100.0,
            "rate_max": 10.0,
        }

    def validate(self, state: SubsystemState) -> bool:
        """Returns True if the state is within the training distribution."""
        if state.any_stale or state.faults.sensor_fault:
            logging.warning("AI Validation: telemetry is stale or navigation sensors are faulted.")
            return False
        if not self.bounds["battery_min"] <= state.eps.battery_charge_percent <= self.bounds["battery_max"]:
            logging.warning("AI Validation: battery is outside trained distribution.")
            return False
            
        if any(abs(r) > self.bounds["rate_max"] for r in state.adcs.rates_deg_s):
            logging.warning("AI Validation: Angular rates exceed trained distribution.")
            return False
            
        # The AI policy handles normalized [-1, 1] internally for most states,
        # but if we get raw telemetry that is completely crazy (e.g. Temp = 500C),
        # we shouldn't feed it to the policy.
        
        return True
