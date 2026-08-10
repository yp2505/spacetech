"""
fsw/gnc/estimators.py
---------------------
GNC State Estimators (Stubs for future UKF/EKF).
Fuses noisy sensor data into a clean state estimate.
"""

from fsw.core.telemetry import ADCSTelemetry
import numpy as np

class AttitudeEstimator:
    def __init__(self):
        self.last_estimate = None

    def update(self, raw_telemetry: ADCSTelemetry) -> ADCSTelemetry:
        """
        In a real FSW, this implements an EKF/UKF fusing Star Tracker and Gyro.
        For now, we pass through the (mocked) truth state from the Sim Backend.
        """
        # Add a tiny bit of simulated estimation noise if desired
        self.last_estimate = raw_telemetry
        return self.last_estimate
