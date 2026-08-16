import re

with open("simulation/satellite_env.py", "r") as f:
    content = f.read()

obs_logic_find = """        for i in range(self.num_satellites):
            # 1) Earth station visibility
            self.gs_los[i] = any(
                abs(self.agent_pos[i] - gs) < 15.0 or (360.0 - abs(self.agent_pos[i] - gs)) < 15.0
                for gs in self.config.ground_stations
            )

            # 2) Eclipse logic
            if self.config.orbit.sun_synchronous:
                self.eclipse_mode[i] = False
            else:
                sun_pos = 180.0
                dist_to_sun = abs(self.agent_pos[i] - sun_pos)
                if dist_to_sun > 180.0:
                    dist_to_sun = 360.0 - dist_to_sun
                self.eclipse_mode[i] = (dist_to_sun < 45.0)"""

obs_logic_replace = """        for i in range(self.num_satellites):
            # 1) Earth station visibility (Geometric Line of Sight)
            # using true anomaly (pos)
            self.gs_los[i] = any(
                abs(self.agent_pos[i] - gs) < 15.0 or (360.0 - abs(self.agent_pos[i] - gs)) < 15.0
                for gs in self.config.ground_stations
            )

            # 2) Eclipse fraction calculation
            if self.config.orbit.sun_synchronous:
                self.eclipse_mode[i] = False
                self.eclipse_fraction[i] = 0.0
            else:
                sun_pos = (self.current_step * 0.1) % 360.0 # moving sun
                dist_to_sun = abs(self.agent_pos[i] - sun_pos)
                if dist_to_sun > 180.0:
                    dist_to_sun = 360.0 - dist_to_sun
                self.eclipse_mode[i] = (dist_to_sun < 45.0)
                # fraction is 1.0 in deep eclipse, 0.0 in full sun
                if dist_to_sun < 35.0:
                    self.eclipse_fraction[i] = 1.0
                elif dist_to_sun < 45.0:
                    self.eclipse_fraction[i] = (45.0 - dist_to_sun) / 10.0
                else:
                    self.eclipse_fraction[i] = 0.0"""

if obs_logic_find in content:
    content = content.replace(obs_logic_find, obs_logic_replace)
    print("Obs logic updated")
else:
    print("Could not find obs logic hook")

with open("simulation/satellite_env.py", "w") as f:
    f.write(content)
