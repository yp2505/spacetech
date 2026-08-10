"""
fsw/fdir/monitors.py
--------------------
Telemetry bounds monitors for detecting anomalies.
"""

from fsw.core.telemetry import SubsystemState
import logging

class TelemetryMonitor:
    def __init__(self):
        self.limits = {
            "battery_min": 10.0,
            "temp_max": 60.0,
            "temp_min": -10.0,
            "max_rate": 8.0,
        }

    def check(self, state: SubsystemState) -> list[str]:
        """Check state against limits. Returns a list of active alarms."""
        alarms = []
        
        if state.eps.battery_charge_percent < self.limits["battery_min"]:
            alarms.append(f"LOW_BATTERY: {state.eps.battery_charge_percent:.1f}%")
            
        if state.thermal.battery_temp_c > self.limits["temp_max"]:
            alarms.append(f"HIGH_TEMP: {state.thermal.battery_temp_c:.1f}C")
            
        if state.thermal.battery_temp_c < self.limits["temp_min"]:
            alarms.append(f"LOW_TEMP: {state.thermal.battery_temp_c:.1f}C")
            
        if any(abs(r) > self.limits["max_rate"] for r in state.adcs.rates_deg_s):
            alarms.append(f"HIGH_RATE: {state.adcs.rates_deg_s}")

        for alarm in alarms:
            logging.warning(f"MONITOR ALARM: {alarm}")
            
        return alarms
