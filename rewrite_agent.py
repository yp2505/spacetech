import re

with open("fsw/ai_brain/agent.py", "r") as f:
    content = f.read()

# Add memory to AIBrain
old_init = """    def __init__(self, model_path: str = "ppo_swarm_brain.zip", config: Optional[SatelliteConfig] = None):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.obs_adapter = AIObservationAdapter(config=config)
        self._load_model()"""

new_init = """    def __init__(self, model_path: str = "ppo_swarm_brain.zip", config: Optional[SatelliteConfig] = None, memory=None):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.memory = memory
        self.obs_adapter = AIObservationAdapter(config=config)
        self._load_model()"""

content = content.replace(old_init, new_init)

old_propose = """            # If we had a memory module integrated into FSW, we'd append it here.
            # For SIL/HIL we pad the 4-dim memory context to match LOCAL_DIM=48
            local_padded = np.concatenate([obs_dict["local"], np.zeros(4, dtype=np.float32)])
            final_obs = {"local": local_padded, "global": obs_dict["global"]}

            action, _ = self.model.predict(final_obs, deterministic=True)"""

new_propose = """            # Use shared episodic memory if available
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
                action = 0.5 * action + 0.5 * np.array(action_bias, dtype=np.float32)"""

content = content.replace(old_propose, new_propose)

with open("fsw/ai_brain/agent.py", "w") as f:
    f.write(content)
