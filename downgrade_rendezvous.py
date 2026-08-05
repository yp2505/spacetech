import re

def main():
    with open('/home/yug/My_Practice/spacetech/satellite_env.py', 'r') as f:
        content = f.read()

    # 1. Update constants
    target = 'FUEL_COST_MOVE   = 0.2        # fuel units consumed when thrusting\nSCENARIO_RENDEZVOUS = True  # Agent A is disabled, Agent B is servicer'
    replacement = 'FUEL_COST_MOVE   = 0.2        # fuel units consumed when thrusting'
    if target in content:
        content = content.replace(target, replacement)
    
    # 2. Add velocity to agent state
    target_state = '''        # ── Agent state ───────────────────────────────────────────────────────
        self.agent_pos     = [0, 5]
        self.agent_vel     = [0, 0] # velocity (degrees/step)
        self.agent_fuel    = [100, 100]
        self.agent_battery = [100, 100]'''
    replacement_state = '''        # ── Agent state ───────────────────────────────────────────────────────
        self.agent_pos     = [0, 5]
        self.agent_fuel    = [100, 100]
        self.agent_battery = [100, 100]'''
    if target_state in content:
        content = content.replace(target_state, replacement_state)

    target_reset = '''        self.agent_pos         = [s1, s2]
        self.agent_vel         = [0, 0] # Start with 0 relative velocity
        self.agent_fuel        = [100, 100]
        self.agent_battery     = [100, 100]'''
    replacement_reset = '''        self.agent_pos         = [s1, s2]
        self.agent_fuel        = [100, 100]
        self.agent_battery     = [100, 100]'''
    if target_reset in content:
        content = content.replace(target_reset, replacement_reset)

    # 3. Update Step Logic (Agent A disabled, velocity mechanics)
    target_step = '''        # 1) Execute actions (Velocity Mechanics & Rendezvous Scenario)
        for i in range(2):
            if SCENARIO_RENDEZVOUS and i == 0:
                # Agent 0 is the disabled target! It ignores actions and has no thruster control.
                pass
            else:
                # Agent 1 is the Servicer. Actions control VELOCITY, not absolute position.
                if actions[i] != 0 and self.agent_fuel[i] > 0 and self.agent_battery[i] > 0:
                    self.agent_vel[i] = np.clip(self.agent_vel[i] + actions[i], -5, 5) # Max speed 5 deg/step
                    self.agent_fuel[i] -= FUEL_COST_MOVE
            
            # Apply velocity to position
            self.agent_pos[i] = (self.agent_pos[i] + self.agent_vel[i]) % self.num_positions
            
            # Space weather random perturbation affects velocity
            if self.space_weather_active and self.np_random.random() < PERTURB_CHANCE:
                self.agent_vel[i] = np.clip(self.agent_vel[i] + self.np_random.choice([-1, 1]), -5, 5)'''
    
    replacement_step = '''        # 1) Execute actions
        for i in range(2):
            if actions[i] != 0 and self.agent_fuel[i] > 0 and self.agent_battery[i] > 0:
                self.agent_pos[i] = (self.agent_pos[i] + actions[i]) % self.num_positions
                self.agent_fuel[i] -= FUEL_COST_MOVE
            # Space weather random perturbation (wind)
            if self.space_weather_active and self.np_random.random() < PERTURB_CHANCE:
                self.agent_pos[i] = (self.agent_pos[i] + self.np_random.choice([-1, 1])) % self.num_positions'''
    
    if target_step in content:
        content = content.replace(target_step, replacement_step)

    # 4. Update Reward Logic
    target_reward = '''        # 4) Calculate Rewards (Rendezvous & Servicing)
        rewards = [0.0, 0.0]
        # Calculate angular distance (shortest path)
        diff = abs(self.agent_pos[0] - self.agent_pos[1])
        fwd_dist = min(diff, self.num_positions - diff)
        
        if SCENARIO_RENDEZVOUS:
            # Goal: Get as close to Agent A (pos 0) as possible without colliding
            # Tolerance zone is 5 degrees (hover range)
            gap_error = fwd_dist
            
            # Agent B (Servicer) gets rewarded for approaching and matching speed
            # Reward decreases if the gap is larger
            if gap_error <= 5:
                # Close enough! Now penalize for relative velocity difference
                vel_diff = abs(self.agent_vel[1] - self.agent_vel[0])
                rewards[1] += 2.0 - (gap_error * 0.1) - (vel_diff * 0.2)
            else:
                # Give small breadcrumb reward for getting closer
                rewards[1] -= gap_error * 0.01
                
            # Agent A is disabled, it receives no meaningful reward
            rewards[0] = 0.0
            
            # Penalties for Servicer
            if fuel_out_flags[1]:
                rewards[1] -= 0.5
            if collision_flags[1]:
                rewards[1] -= 2.0 # Huge penalty for crashing into target!
        else:
            # Fallback to Formation
            gap_error = abs(fwd_dist - TARGET_SLOT_GAP)
            for i in range(2):
                if gap_error <= 5:
                    rewards[i] += 1.0 - (gap_error * 0.1)
                else:
                    rewards[i] -= gap_error * 0.01
                if fuel_out_flags[i]: rewards[i] -= 0.5
                if collision_flags[i]: rewards[i] -= 0.5'''
                
    replacement_reward = '''        # 4) Calculate Rewards (Formation Flying)
        rewards = [0.0, 0.0]
        # Calculate angular distance (shortest path)
        diff = abs(self.agent_pos[0] - self.agent_pos[1])
        fwd_dist = min(diff, self.num_positions - diff)
        
        # Target is TARGET_SLOT_GAP (90 degrees). We use a 5-degree tolerance zone.
        gap_error = abs(fwd_dist - TARGET_SLOT_GAP)
        for i in range(2):
            if gap_error <= 5:
                # Inside tolerance zone: +1.0 for perfect 90, down to +0.5 for 5 degrees off.
                rewards[i] += 1.0 - (gap_error * 0.1)
            else:
                # Outside tolerance zone: linear penalty
                rewards[i] -= gap_error * 0.01
            
            if fuel_out_flags[i]:
                rewards[i] -= 0.5
            if collision_flags[i]:
                rewards[i] -= 0.5'''
    
    if target_reward in content:
        content = content.replace(target_reward, replacement_reward)

    with open('/home/yug/My_Practice/spacetech/satellite_env.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()
