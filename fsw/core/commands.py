"""
fsw/core/commands.py
--------------------
Strongly-typed command definitions for the FSW Architecture.
Ensures traceability, validation, and clear reason codes for all actuator commands.
"""

from dataclasses import dataclass, field
import time
import uuid

@dataclass
class ActuatorCommand:
    """Represents a unified command to the ADCS and Propulsion HAL."""
    source: str                 # e.g., 'AI_PLANNER', 'CLASSICAL_SAFE_MODE', 'GROUND'
    thrust: float = 0.0
    roll_tq: float = 0.0
    pitch_tq: float = 0.0
    yaw_tq: float = 0.0
    relay_action: float = 0.0
    hohmann: float = 0.0
    avoidance: float = 0.0
    deorbit: float = 0.0
    
    # Metadata for tracing
    command_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.monotonic)
    expires_after_s: float = 1.0
    authorization: str = "LOCAL_AUTONOMY"
    approved: bool = False
    acknowledged: bool = False
    reason_code: str = "PENDING"

    @property
    def is_expired(self) -> bool:
        return time.monotonic() - self.timestamp > self.expires_after_s
