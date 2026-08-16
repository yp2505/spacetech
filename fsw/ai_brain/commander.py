"""
fsw/ai_brain/commander.py
--------------------------
Hierarchical RL: Mission Commander Layer (Rule-Based).

Sets high-level `commander_goal` vectors consumed by the PPO low-level agent.
The Commander runs at a *lower* frequency (every 10 FSW ticks = 1 Hz) and acts
as a deterministic supervisor that decides *what* the satellite should focus on.
The PPO agent then figures out *how* to achieve that goal via its actuator actions.

Commander Goal Vector: 3-dim float [PRIORITY_MODE, TARGET_VALUE, URGENCY]

  PRIORITY_MODE encoding (continuous, interpolated):
    0.0 = Nominal slot-keeping / station-keeping
    0.3 = Prioritise data downlink or ISL relay
    0.5 = Debris avoidance manoeuvre
    0.6 = Energy conservation / eclipse preparation
    0.9 = De-orbit prep / terminal phase
    1.0 = Emergency EOL — execute de-orbit burn

  TARGET_VALUE: normalised [0,1] metric relevant to the current mode
    e.g. for mode=0.3 (downlink), target_value = data_buffer / capacity
    e.g. for mode=0.6 (energy), target_value = battery_soc / 100

  URGENCY: [0,1] — how quickly the low-level agent should act
    0.0 = no rush, 1.0 = act immediately this cycle

ISL Packet Crypto
-----------------
All inter-satellite DATA payloads (not just status beacons) are encrypted using
a rolling-XOR cipher with a 16-byte key derived from the satellite index and
a shared mission epoch seed. The adapter already decodes status beacons; the
MissionCommander uses ISLPacketCrypto for actual data relay payloads.
"""

from __future__ import annotations
import base64
import hashlib
import logging
import struct
import time
from typing import Optional

from fsw.core.telemetry import SubsystemState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  ISL Packet Encryption / Decryption
# ─────────────────────────────────────────────────────────────────────────────

# Mission-wide shared seed (in a real mission this would be a pre-loaded key
# uplinked during commissioning and stored in secure onboard memory).
_MISSION_KEY_SEED: bytes = b"ARTEMIS_SWARM_2025"


def _derive_key(src_sat_idx: int, dst_sat_idx: int) -> bytes:
    """
    Derive a 16-byte rolling-XOR key from the mission seed and satellite indices.
    Uses SHA-256 so the key is unique per directed link but deterministic.
    The same key is produced on both ends independently — no key exchange needed.
    """
    material = _MISSION_KEY_SEED + struct.pack(">HH", src_sat_idx, dst_sat_idx)
    digest = hashlib.sha256(material).digest()
    return digest[:16]  # 128-bit key


class ISLPacketCrypto:
    """
    Lightweight rolling-XOR cipher for ISL data relay packets.

    Design rationale:
    - Satellites are resource-constrained; AES-128 in hardware would be ideal,
      but this simulation uses a rolling-XOR keyed with SHA-256 as a proxy.
    - The cipher is symmetric: encrypt == decrypt (XOR is its own inverse).
    - A 4-byte big-endian sequence number is prepended to guard against replay
      and to ensure the key stream repeats only after 2^32 packets (>>mission life).

    Wire format (base64-encoded):
      [4 bytes: sequence_number] [N bytes: ciphertext]
    """

    def __init__(self, src_sat_idx: int, dst_sat_idx: int):
        self.key = _derive_key(src_sat_idx, dst_sat_idx)
        self._seq = 0

    def _xor_stream(self, data: bytes, seq: int) -> bytes:
        """Apply rolling-XOR using the key, salted with the sequence number."""
        key_len = len(self.key)
        seq_bytes = struct.pack(">I", seq)  # 4-byte salt
        result = bytearray(len(data))
        for i, byte in enumerate(data):
            k = self.key[(i + seq) % key_len] ^ seq_bytes[i % 4]
            result[i] = byte ^ k
        return bytes(result)

    def encrypt(self, plaintext: bytes) -> str:
        """Encrypt bytes and return a base64 string suitable for ISL transmission."""
        self._seq += 1
        ciphertext = self._xor_stream(plaintext, self._seq)
        wire = struct.pack(">I", self._seq) + ciphertext
        return base64.b64encode(wire).decode("ascii")

    def decrypt(self, b64_payload: str) -> Optional[bytes]:
        """Decrypt a base64-encoded ISL payload. Returns None on any error."""
        try:
            wire = base64.b64decode(b64_payload)
            if len(wire) < 4:
                return None
            seq = struct.unpack(">I", wire[:4])[0]
            ciphertext = wire[4:]
            return self._xor_stream(ciphertext, seq)
        except Exception as exc:
            logger.warning("ISL packet decryption failed: %s", exc)
            return None

    @staticmethod
    def pack_data_packet(src_idx: int, data_gb: float, timestamp: float) -> bytes:
        """Serialise a data relay packet into bytes."""
        # Format: "RELAY:<src>:<data_gb>:<ts>"
        return f"RELAY:{src_idx}:{data_gb:.4f}:{timestamp:.3f}".encode()

    @staticmethod
    def unpack_data_packet(raw: bytes) -> Optional[dict]:
        """Deserialise. Returns dict with keys src, data_gb, timestamp or None."""
        try:
            parts = raw.decode().split(":")
            if parts[0] != "RELAY" or len(parts) < 4:
                return None
            return {
                "src": int(parts[1]),
                "data_gb": float(parts[2]),
                "timestamp": float(parts[3]),
            }
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  Mission Commander
# ─────────────────────────────────────────────────────────────────────────────

class MissionCommander:
    """
    Rule-based hierarchical supervisor.

    Runs at ~1 Hz (every 10 FSW ticks) and evaluates spacecraft + fleet state
    to emit a `commander_goal` 3-tuple that is injected into the observation
    adapter and seen by the low-level PPO agent as extra goal-conditioning.

    Priority order (strictly evaluated top-to-bottom):
      1. EOL / emergency de-orbit
      2. Low battery — conserve power
      3. Overtemp — thermal recovery
      4. Debris proximity — avoidance
      5. Data buffer full + GS LOS — downlink now
      6. Data buffer high + no LOS — ISL relay
      7. Nominal slot-keeping
    """

    # Thresholds
    BAT_EOL          = 15.0   # % — trigger de-orbit prep
    BAT_LOW          = 30.0   # % — enter energy conservation
    TEMP_HIGH        = 55.0   # °C — thermal concern
    DATA_HIGH        = 80.0   # % capacity — downlink urgency
    DATA_RELAY       = 50.0   # % capacity — relay urgency
    DEBRIS_WARN_DEG  = 10.0   # degrees — proximity alert
    ISL_QUAL_MIN     = 0.5    # minimum link quality for relay

    def __init__(self, data_capacity_gb: float = 100.0):
        self.data_capacity_gb = max(data_capacity_gb, 1.0)
        self.last_goal: tuple[float, float, float] = (0.0, 1.0, 0.0)
        self._tick = 0

    def evaluate(
        self,
        state: SubsystemState,
        fdir_mode,
    ) -> tuple[float, float, float]:
        """
        Evaluate spacecraft state and return the commander_goal 3-tuple.
        Should be called every 10 FSW ticks (at ~1 Hz).
        """
        self._tick += 1

        bat  = state.eps.battery_charge_percent
        temp = state.thermal.battery_temp_c
        data = state.comms.data_buffer_gb
        dv   = state.adcs.delta_v_remaining
        los  = state.comms.has_ground_los
        data_pct = (data / self.data_capacity_gb) * 100.0

        # ── Rule 1: EOL emergency ────────────────────────────────────────────
        if bat < self.BAT_EOL:
            goal = (1.0, bat / 100.0, 1.0)
            self._log_transition(goal, "EOL_EMERGENCY")
            self.last_goal = goal
            return goal

        # ── Rule 2: Low battery ──────────────────────────────────────────────
        if bat < self.BAT_LOW:
            urgency = 1.0 - (bat / self.BAT_LOW)
            goal = (0.6, bat / 100.0, round(urgency, 3))
            self._log_transition(goal, "LOW_BATTERY")
            self.last_goal = goal
            return goal

        # ── Rule 3: Thermal concern ──────────────────────────────────────────
        if temp > self.TEMP_HIGH:
            urgency = min(1.0, (temp - self.TEMP_HIGH) / 20.0)
            goal = (0.6, 1.0 - temp / 100.0, round(urgency, 3))
            self._log_transition(goal, "THERMAL_CONCERN")
            self.last_goal = goal
            return goal

        # ── Rule 4: Debris proximity ─────────────────────────────────────────
        own_pos_deg = state.adcs.position_eci_km[0] / 100.0 * 360.0
        for d_pos in state.global_fleet.debris_positions:
            angular_sep = min(
                abs(own_pos_deg - d_pos),
                360.0 - abs(own_pos_deg - d_pos)
            )
            if angular_sep < self.DEBRIS_WARN_DEG:
                urgency = 1.0 - (angular_sep / self.DEBRIS_WARN_DEG)
                goal = (0.5, 1.0, round(urgency, 3))
                self._log_transition(goal, "DEBRIS_AVOIDANCE")
                self.last_goal = goal
                return goal

        # ── Rule 5: Data full + GS LOS — downlink now ────────────────────────
        if data_pct >= self.DATA_HIGH and los:
            urgency = min(1.0, (data_pct - self.DATA_HIGH) / (100.0 - self.DATA_HIGH) + 0.5)
            goal = (0.3, data_pct / 100.0, round(urgency, 3))
            self._log_transition(goal, "DOWNLINK_PRIORITY")
            self.last_goal = goal
            return goal

        # ── Rule 6: Data high + no LOS — ISL relay ───────────────────────────
        if data_pct >= self.DATA_RELAY and not los:
            relay_possible = any(
                n.isl_active and n.link_quality >= self.ISL_QUAL_MIN
                for n in state.neighbors
            )
            if relay_possible:
                urgency = min(1.0, data_pct / 100.0)
                goal = (0.3, data_pct / 100.0, round(urgency * 0.6, 3))
                self._log_transition(goal, "ISL_RELAY")
                self.last_goal = goal
                return goal

        # ── Rule 7: Nominal ──────────────────────────────────────────────────
        goal = (0.0, 1.0, 0.0)
        self.last_goal = goal
        return goal

    def _log_transition(self, goal: tuple, label: str) -> None:
        if self.last_goal[0] != goal[0]:
            logger.info(
                "COMMANDER: Goal transition → %s  goal=%s",
                label, goal
            )
