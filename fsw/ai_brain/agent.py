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
from fsw.core.watchdog import WatchdogManager

class AIBrain:
    def __init__(self, model_path: str = "ppo_swarm_brain.zip", config: Optional[SatelliteConfig] = None, memory=None, watchdog: Optional[WatchdogManager] = None):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.memory = memory
        self.watchdog = watchdog
        self.obs_adapter = AIObservationAdapter(config=config)
        self._load_model()

    def _load_model(self):
        try:
            from stable_baselines3 import PPO
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

    def propose_action(self, state: SubsystemState, in_recovery: bool = False) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Takes validated structured telemetry and queries the offline-trained policy.
        """
        if not self.is_loaded:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if state.any_stale:
            logging.warning("AI Brain: Telemetry is stale. Proposing NO-OP.")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            # Construct observation solely from FSW validated telemetry
            obs_dict = self.obs_adapter.build_observation(state, is_in_recovery=in_recovery)
            
            # Use shared episodic memory if available
            mem_ctx = np.zeros(4, dtype=np.float32)
            action_bias = None
            if self.memory is not None:
                mem_ctx = self.memory.get_context()
                # Cross-satellite query using cosine similarity
                orb_state = {
                    "true_anomaly": obs_dict["local"][0] * 360.0,
                    "altitude_km": obs_dict["local"][1] * 10000.0,
                    "eclipse_fraction": obs_dict["local"][2]
                }
                similar_ep = self.memory.query_similar_episode(orb_state, threshold=0.9)
                if similar_ep and similar_ep.total_reward > 0 and len(similar_ep.action_sequence) > 0:
                    action_bias = similar_ep.action_sequence[0]

            local_padded = np.concatenate([obs_dict["local"], mem_ctx])
            final_obs = {"local": local_padded, "global": obs_dict["global"]}

            action, _ = self.model.predict(final_obs, deterministic=True)
            
            # Bias PPO action based on successful past memory
            if action_bias is not None:
                action = 0.5 * action + 0.5 * np.array(action_bias, dtype=np.float32)
            
            if not np.isfinite(action).all():
                raise ValueError("policy produced non-finite action")
            
            # Signal successful inference to watchdog
            if self.watchdog is not None:
                self.watchdog.ai_heartbeat()
            
            # Action space is now 8D:
            return tuple(float(x) for x in action)
            
        except Exception as e:
            logging.error(f"AI Brain inference error: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
