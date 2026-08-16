import re

with open("rl_training/memory.py", "r") as f:
    content = f.read()

# Add query_similar_episode using cosine similarity
query_code = """    def query_similar_episode(self, current_orbital_state: dict, threshold=0.8):
        \"\"\"
        Cross-satellite memory query: finds a past episode with a highly similar
        orbital state (true anomaly, altitude, eclipse).
        Uses cosine similarity.
        \"\"\"
        if not self.episodes:
            return None
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return None
            
        def to_vec(state):
            return np.array([[
                state.get("true_anomaly", 0.0),
                state.get("altitude_km", 500.0),
                state.get("eclipse_fraction", 0.0)
            ]])
            
        current_vec = to_vec(current_orbital_state)
        
        best_ep = None
        best_sim = -1.0
        
        for ep in self.episodes:
            if not ep.orbital_state:
                continue
            ep_vec = to_vec(ep.orbital_state)
            sim = cosine_similarity(current_vec, ep_vec)[0][0]
            if sim > best_sim and sim > threshold:
                best_sim = sim
                best_ep = ep
                
        return best_ep

"""

content = content.replace("    # ── summary ────────────────────────────────────────────────────────────────", query_code + "    # ── summary ────────────────────────────────────────────────────────────────")

with open("rl_training/memory.py", "w") as f:
    f.write(content)
