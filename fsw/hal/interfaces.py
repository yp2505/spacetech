"""
fsw/hal/interfaces.py
---------------------
Hardware Abstraction Layer (HAL) base classes.
These interfaces ensure the Flight Software is completely agnostic to whether
it is running in Simulation, Hardware-in-the-Loop (HIL), or actual flight.
"""

from abc import ABC, abstractmethod
from fsw.core.telemetry import (
    EPSTelemetry, ThermalTelemetry, ADCSTelemetry, CommsTelemetry, FaultTelemetry
)
from fsw.core.commands import ActuatorCommand

class AbstractEPS(ABC):
    @abstractmethod
    def get_telemetry(self) -> EPSTelemetry:
        pass

class AbstractThermal(ABC):
    @abstractmethod
    def get_telemetry(self) -> ThermalTelemetry:
        pass

class AbstractADCS(ABC):
    @abstractmethod
    def get_telemetry(self) -> ADCSTelemetry:
        pass

    @abstractmethod
    def command_actuators(self, cmd: ActuatorCommand) -> bool:
        """Send torque/thrust commands to the reaction wheels and thrusters."""
        pass

class AbstractRadio(ABC):
    @abstractmethod
    def get_telemetry(self) -> CommsTelemetry:
        pass
        
    @abstractmethod
    def downlink_data(self, amount_gb: float) -> bool:
        """Attempt to downlink data. Returns True if successful."""
        pass

class AbstractFaultMonitor(ABC):
    @abstractmethod
    def get_telemetry(self) -> FaultTelemetry:
        pass
