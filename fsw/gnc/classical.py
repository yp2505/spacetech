"""
fsw/gnc/classical.py
--------------------
Classical Deterministic Controllers for SAFE_MODE and EOL sequences.

All controllers are pure functions (no state). The Arbiter calls these when
the FDIR state machine transitions to SAFE_MODE or EOL_DEORBIT, completely
bypassing the AI brain.
"""

import math
import logging

# Earth gravitational parameter [km³/s²] and radius [km]
_GM_EARTH = 398600.4418
_R_EARTH_KM = 6378.1
_REENTRY_ALT_KM = 80.0   # Target perigee for destructive reentry (Point Nemo reference)


class ClassicalControllers:
    @staticmethod
    def detumble(
        rates_deg_s: tuple[float, float, float],
        max_torque: float = 1.0,
    ) -> tuple[float, float, float]:
        """
        Simple B-dot-style detumble controller.
        Opposes angular rates proportionally, clamped to ±max_torque.
        """
        k = 0.5
        roll_tq  = max(-max_torque, min(max_torque, -k * rates_deg_s[0]))
        pitch_tq = max(-max_torque, min(max_torque, -k * rates_deg_s[1]))
        yaw_tq   = max(-max_torque, min(max_torque, -k * rates_deg_s[2]))
        return roll_tq, pitch_tq, yaw_tq

    @staticmethod
    def deorbit_burn(
        altitude_km: float,
        delta_v_remaining_ms: float,
        target_perigee_km: float = _REENTRY_ALT_KM,
    ) -> dict:
        """
        Compute the ΔV required for a Hohmann-style de-orbit burn using the
        vis-viva equation (assumes circular parking orbit).

        Physics:
          - Circular orbit velocity:  v_c  = sqrt(GM / r)
          - Transfer apoapsis = current altitude, perigee = target_perigee_km
          - Semi-major axis of transfer: a   = (r + r_p) / 2
          - Velocity at apoapsis:        v_a = sqrt(GM * (2/r - 1/a))
          - Required retrograde ΔV:      ΔV  = v_c - v_a  (negative burn)

        Args:
            altitude_km:           Current circular orbit altitude above Earth [km].
            delta_v_remaining_ms:  Available ΔV budget [m/s].
            target_perigee_km:     Desired perigee altitude for reentry [km] (default 80 km).

        Returns:
            dict with keys:
              dv_required_ms  — ΔV magnitude needed [m/s]
              feasible        — True if delta_v_remaining_ms >= dv_required_ms
              burn_direction  — -1.0 (retrograde) always for de-orbit
              v_circular_ms   — Current circular orbit speed [m/s]
              v_transfer_ms   — Speed at de-orbit apoapsis [m/s]
        """
        r_km   = _R_EARTH_KM + altitude_km
        r_p_km = _R_EARTH_KM + target_perigee_km

        # Convert to metres for vis-viva (GM in km³/s², r in km → m/s * 1000)
        v_circular_ms = math.sqrt(_GM_EARTH / r_km) * 1000.0   # km/s → m/s

        a_km = (r_km + r_p_km) / 2.0
        v_transfer_ms = math.sqrt(_GM_EARTH * (2.0 / r_km - 1.0 / a_km)) * 1000.0

        dv_ms = abs(v_circular_ms - v_transfer_ms)
        feasible = delta_v_remaining_ms >= dv_ms

        if not feasible:
            logging.warning(
                "GNC: De-orbit requires %.1f m/s ΔV but only %.1f m/s available. "
                "Burn NOT feasible.",
                dv_ms, delta_v_remaining_ms,
            )
        else:
            logging.info(
                "GNC: De-orbit burn computed. ΔV=%.1f m/s  v_c=%.1f m/s  v_transfer=%.1f m/s",
                dv_ms, v_circular_ms, v_transfer_ms,
            )

        return {
            "dv_required_ms": dv_ms,
            "feasible": feasible,
            "burn_direction": -1.0,   # retrograde
            "v_circular_ms": v_circular_ms,
            "v_transfer_ms": v_transfer_ms,
        }

    @staticmethod
    def sun_pointing(
        attitude_deg: tuple[float, float, float],
        eclipse: bool,
        max_torque: float = 0.5,
    ) -> tuple[float, float, float]:
        """
        Nadir/sun-pointing controller for SAFE_MODE power recovery.
        Rotates the pitch axis toward 0° (nadir-pointing) to maximise
        solar panel area toward the sun when not in eclipse.
        """
        if eclipse:
            return 0.0, 0.0, 0.0  # No manoeuvring in eclipse to save battery
        target_pitch = 0.0
        pitch_error  = target_pitch - attitude_deg[1]
        pitch_tq     = max(-max_torque, min(max_torque, 0.3 * pitch_error))
        return 0.0, pitch_tq, 0.0

