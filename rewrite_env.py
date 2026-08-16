import re

with open("simulation/satellite_env.py", "r") as f:
    content = f.read()

# Replace debris logic in step()
step_debris_find = """        # Advance debris
        for d in self.debris:
            d["pos"] = (d["pos"] + d["vel"]) % 360.0

        # Spawn / despawn debris (Phase 3+)
        if self.curriculum_phase >= 3:
            if len(self.debris) < MAX_DEBRIS and self.np_random.random() < DEBRIS_SPAWN:
                self.debris.append({
                    "pos": float(self.np_random.uniform(0, 360)),
                    "vel": float(self.np_random.uniform(-2.0, 2.0)),
                })
            self.debris = [d for d in self.debris if self.np_random.random() > DEBRIS_DESPAWN]"""

step_debris_replace = """        # Advance Keplerian debris
        for d in self.debris:
            # Mean motion n = sqrt(GM/a^3)
            n_mean = np.sqrt(GM_EARTH / (d["a"]*1000)**3)
            # Update true anomaly (simplified assuming circular-ish for delta nu)
            delta_nu = np.degrees(n_mean * self._step_sec)
            d["nu"] = (d["nu"] + delta_nu) % 360.0"""

if step_debris_find in content:
    content = content.replace(step_debris_find, step_debris_replace)
    print("Step debris updated")
else:
    print("Could not find step debris hook")

with open("simulation/satellite_env.py", "w") as f:
    f.write(content)
