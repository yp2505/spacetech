import re

with open("simulation/satellite_env.py", "r") as f:
    content = f.read()

# Replace _compute_eclipse
old_eclipse = """    def _compute_eclipse(self, pos_deg: float) -> bool:
        \"\"\"Determine if satellite is in eclipse based on sun angle and orbit arc.\"\"\"
        if self.curriculum_phase < 2:
            return False
        eclipse_half = self.config.orbit.eclipse_arc_deg / 2.0
        relative_pos = (pos_deg - self._sun_angle_deg) % 360.0
        return (180.0 - eclipse_half) < relative_pos < (180.0 + eclipse_half)"""

new_eclipse = """    def _compute_eclipse(self, pos_deg: float) -> tuple[bool, float]:
        \"\"\"Determine if satellite is in eclipse and return (is_eclipse, fraction).\"\"\"
        if self.curriculum_phase < 2:
            return False, 0.0
        eclipse_half = self.config.orbit.eclipse_arc_deg / 2.0
        relative_pos = (pos_deg - self._sun_angle_deg) % 360.0
        
        dist_to_anti_sun = abs(180.0 - relative_pos)
        
        is_eclipse = dist_to_anti_sun < eclipse_half
        
        # Calculate fraction
        if dist_to_anti_sun < eclipse_half * 0.8:
            fraction = 1.0 # deep eclipse
        elif dist_to_anti_sun < eclipse_half:
            fraction = (eclipse_half - dist_to_anti_sun) / (eclipse_half * 0.2)
        else:
            fraction = 0.0
            
        return is_eclipse, fraction"""

content = content.replace(old_eclipse, new_eclipse)

# Update call site
content = content.replace(
    "ecl = self._compute_eclipse(self.agent_pos[i])\n            self.eclipse_mode[i] = ecl",
    "ecl, frac = self._compute_eclipse(self.agent_pos[i])\n            self.eclipse_mode[i] = ecl\n            self.eclipse_fraction[i] = frac"
)

# Update _get_obs_list 
content = content.replace("float(np.mean(self.eclipse_mode)),", "float(np.mean(self.eclipse_fraction)),")

with open("simulation/satellite_env.py", "w") as f:
    f.write(content)
