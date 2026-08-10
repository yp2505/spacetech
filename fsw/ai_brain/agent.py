"""
fsw/ai_brain/agent.py
---------------------
AI Mission Planner (Agent).
Wraps the offline-trained RL model. Receives validated telemetry, builds the
observation vector, and proposes actions to the Command Arbiter.
"""

import numpy as np
import logging
from typing import Optional, Tuple
from fsw.core.telemetry import SubsystemState
from fsw.ai_brain.adapter import AIObservationAdapter
from simulation.sat_config import SatelliteConfig

class AIBrain:
    def __init__(self, model_path: str = "ppo_swarm_brain.zip", config: Optional[SatelliteConfig] = None):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.obs_adapter = AIObservationAdapter(config=config)
        self._load_model()

    def _load_model(self):
        try:
            from stable_baselines3 import PPO
            # In a real flight system, this would be a C++ ONNX runtime or similar.
            # Here we load the python SB3 model for SIL testing.
            from rl_training.ctde_policy import CTDEPolicy
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = PPO.load(self.model_path, custom_objects={'policy_class': CTDEPolicy})
            self.is_loaded = True
            logging.info(f"AI Brain loaded from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load AI Brain: {e}")
            self.is_loaded = False

    def propose_action(self, state: SubsystemState, in_recovery: bool = False) -> Tuple[float, float, float, float]:
        """
        Takes validated structured telemetry and queries the offline-trained policy.
        """
        if not self.is_loaded:
            return 0.0, 0.0, 0.0, 0.0

        if state.any_stale:
            logging.warning("AI Brain: Telemetry is stale. Proposing NO-OP.")
            return 0.0, 0.0, 0.0, 0.0

        try:
            # Construct observation solely from FSW validated telemetry
            obs_dict = self.obs_adapter.build_observation(state, is_in_recovery=in_recovery)
            
            # If we had a memory module integrated into FSW, we'd append it here.
            # For SIL/HIL we pad the 4-dim memory context to match LOCAL_DIM=53
            local_padded = np.concatenate([obs_dict["local"], np.zeros(4, dtype=np.float32)])
            final_obs = {"local": local_padded, "global": obs_dict["global"]}

            action, _ = self.model.predict(final_obs, deterministic=True)
            
            if not np.isfinite(action).all():
                raise ValueError("policy produced non-finite action")
            thrust = float(action[0])
            roll_tq = float(action[1])
            pitch_tq = float(action[2])
            yaw_tq = float(action[3])
            return thrust, roll_tq, pitch_tq, yaw_tq
            
        except Exception as e:
            logging.error(f"AI Brain inference error: {e}")
            return 0.0, 0.0, 0.0, 0.0
