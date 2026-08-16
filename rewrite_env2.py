import re

with open("simulation/satellite_env.py", "r") as f:
    content = f.read()

visviva_find = """            if hohmann > 0 and self.agent_delta_v[i] > 10.0:
                self.agent_delta_v[i] -= 10.0
                rewards[i] += 0.5 # Example reward for valid maneuver
            if avoidance > 0 and self.agent_delta_v[i] > 20.0:
                self.agent_delta_v[i] -= 20.0
                rewards[i] += 1.0 # Reward for avoiding debris
            if deorbit > 0 and self.agent_delta_v[i] > 50.0:
                self.agent_delta_v[i] -= 50.0
                rewards[i] += 2.0 # Reward for safe end of life disposal

            # Part 4: Commander updates (runs every 10 steps)
            if self.current_step % 10 == 0:
                self.commander_goal[i] = np.array([1.0, 0.0, 0.0], dtype=np.float32)"""

visviva_replace = """            # Vis-viva equation: v = sqrt(GM * (2/r - 1/a))
            # Assume circular orbit for simplicity r = a
            current_r = (EARTH_RADIUS_KM + self.config.orbit.altitude_km) * 1000
            current_v = np.sqrt(GM_EARTH / current_r)
            
            if hohmann > 0:
                target_r = current_r + 50000 # 50km boost
                target_v = np.sqrt(GM_EARTH / target_r)
                dv_cost = abs(current_v - target_v) * 2 # 2 burns
                if self.agent_delta_v[i] > dv_cost:
                    self.agent_delta_v[i] -= dv_cost
                    rewards[i] += 0.5
                    
            if avoidance > 0:
                dv_cost = 15.0 # 15 m/s evasive burn
                if self.agent_delta_v[i] > dv_cost:
                    self.agent_delta_v[i] -= dv_cost
                    rewards[i] += 1.0
                    
            if deorbit > 0:
                target_r = EARTH_RADIUS_KM * 1000 # deorbit
                dv_cost = abs(current_v - np.sqrt(GM_EARTH / target_r))
                if self.agent_delta_v[i] > dv_cost:
                    self.agent_delta_v[i] -= dv_cost
                    rewards[i] += 2.0

            # Part 4: Commander updates (runs every 10 steps)
            if self.current_step % 10 == 0:
                # Basic heuristics for numeric commander goals
                if self.agent_battery[i] < 30.0:
                    self.commander_goal[i] = np.array([0.0, 0.0, 1.0], dtype=np.float32) # Deorbit / safe
                elif self.agent_delta_v[i] < 100.0:
                    self.commander_goal[i] = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Relay / conserve
                else:
                    self.commander_goal[i] = np.array([1.0, 0.0, 0.0], dtype=np.float32) # Prioritize imaging"""

if visviva_find in content:
    content = content.replace(visviva_find, visviva_replace)
    print("Vis-viva updated")
else:
    # If not exact match, try regex
    content = re.sub(r'if hohmann > 0.*?self\.commander_goal\[i\] = np\.array\(\[1\.0, 0\.0, 0\.0\], dtype=np\.float32\)', visviva_replace, content, flags=re.DOTALL)
    print("Regex replaced")

with open("simulation/satellite_env.py", "w") as f:
    f.write(content)
