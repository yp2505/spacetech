"""
orbital_decay.py
----------------
Orbital Salience Gating (OSG) — temporal decay helper.

Derives the memory decay rate β from Kepler's Third Law so that the
half-life of a memory equals exactly one orbital period.  This grounds
the decay rate in real physics rather than an arbitrary hyperparameter.
"""


def compute_beta(altitude_km: float, steps_per_second: float = 1.0):
    """
    Derive OSG temporal decay rate β from Kepler's Third Law.

    Kepler's Third Law gives the orbital period T for a circular orbit
    at altitude h above Earth's surface:

        T = 2π × sqrt((R_earth + h)³ / GM)

    where:
        R_earth = 6371 km
        GM      = 3.986 × 10^5 km³/s²

    We define β so that exactly one orbital period equals the memory
    half-life:

        exp(-β × T_steps) = 0.5
        therefore: β = ln(2) / T_steps

    This means a memory retains exactly 50 % of its temporal validity
    after one full orbit — grounded in real orbital mechanics, not a
    guessed hyperparameter.

    Args:
        altitude_km:        Orbital altitude above Earth's surface (km).
        steps_per_second:   How many simulation steps equal one real
                            second.  Use the sat_config step_seconds
                            value: steps_per_second = 1 / step_seconds.

    Returns:
        beta      (float): Decay rate per simulation step.
        T_seconds (float): Orbital period in seconds.
        T_steps   (float): Orbital period in simulation steps.
    """
    import numpy as np

    R_earth = 6371.0    # km
    GM      = 3.986e5   # km³/s²

    r         = R_earth + altitude_km
    T_seconds = 2 * np.pi * np.sqrt(r**3 / GM)
    T_steps   = T_seconds * steps_per_second
    beta      = np.log(2) / T_steps

    return beta, T_seconds, T_steps


# ─── Quick smoke-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # LEO 550 km — default Starlink-style orbit used by this codebase.
    # step_seconds = 15.9  ->  steps_per_second = 1/15.9
    LEO_ALT        = 550.0
    STEP_SECONDS   = 15.9                     # from sat_config ORBIT_PARAMS[LEO]
    sps            = 1.0 / STEP_SECONDS

    beta, T_seconds, T_steps = compute_beta(LEO_ALT, steps_per_second=sps)

    print("utils/orbital_decay.py created")
    print(f"beta for LEO 550km = {beta:.8f}, T = {T_seconds:.1f}s")
