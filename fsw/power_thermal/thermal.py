"""
fsw/power_thermal/thermal.py
----------------------------
Thermal Management System.
Controls heaters and enforces component survival limits.
"""

from fsw.core.telemetry import ThermalTelemetry

class ThermalManager:
    def __init__(self):
        self.heater_on = False
        self.heater_threshold_c = -5.0
        self.heater_off_c = 5.0

    def process(self, telemetry: ThermalTelemetry):
        """Simple bang-bang heater control."""
        if telemetry.battery_temp_c < self.heater_threshold_c:
            self.heater_on = True
        elif telemetry.battery_temp_c > self.heater_off_c:
            self.heater_on = False
            
        return self.heater_on
