"""
fsw/gnc/classical.py
--------------------
Classical Deterministic Controllers for SAFE_MODE.
"""

import math

class ClassicalControllers:
    @staticmethod
    def detumble(rates_deg_s: tuple[float, float, float], max_torque: float = 1.0) -> tuple[float, float, float]:
        """Simple B-dot style detumble. Opposes angular rates."""
        k = 0.5
        roll_tq = max(-max_torque, min(max_torque, -k * rates_deg_s[0]))
        pitch_tq = max(-max_torque, min(max_torque, -k * rates_deg_s[1]))
        yaw_tq = max(-max_torque, min(max_torque, -k * rates_deg_s[2]))
        return roll_tq, pitch_tq, yaw_tq
