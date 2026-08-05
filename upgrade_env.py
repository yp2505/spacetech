import re
import sys

def main():
    with open('/home/yug/My_Practice/spacetech/satellite_env.py', 'r') as f:
        content = f.read()

    # 1. Update defaults and constants
    content = content.replace('num_positions: int = 20, max_steps: int = 100', 'num_positions: int = 360, max_steps: int = 360')
    content = content.replace('TARGET_SLOT_GAP  = 5', 'TARGET_SLOT_GAP  = 90       # 90 degrees apart')
    content = content.replace('MAX_DEBRIS       = 3', 'MAX_DEBRIS       = 10      # more debris for 360 degree orbit')
    content = content.replace('DEBRIS_SPAWN     = 0.03', 'DEBRIS_SPAWN     = 0.10')
    
    # Update battery and fuel drain to match 360 steps (was 100 steps)
    # If 180 steps of eclipse, drain needs to be smaller to not hit 0.
    # Base drain was 2, now 0.5. Extra drain was 5, now 1. Charge was 5, now 1.5.
    content = content.replace('self.agent_battery[i] -= 2           # base shadow drain', 'self.agent_battery[i] -= 0.5         # base shadow drain')
    content = content.replace('self.agent_battery[i] -= 5', 'self.agent_battery[i] -= 1.5')
    content = content.replace('self.agent_battery[i] += 5           # solar charging', 'self.agent_battery[i] += 1.5         # solar charging')
    content = content.replace('FUEL_COST_MOVE   = 1', 'FUEL_COST_MOVE   = 0.2')

    # 2. Update Orbital tracks to be independent
    # Find _orbit_coords initialization
    init_orbit = '''        # ── Precompute orbit tracks for rendering ─────────────────────────────
        # We assume both sats share roughly the same orbital track for this formation scenario.
        # Just use sat 0's coords to build the slot track.
        xyz_ring = self.physics.get_orbit_xyz_ring(0, num_points=self.num_positions)
        self._orbit_coords = [
            (xyz_ring[:, 0], xyz_ring[:, 1], xyz_ring[:, 2])
        ]'''
    
    new_init_orbit = '''        # ── Precompute orbit tracks for both independent satellites ───────────
        self._orbit_coords = []
        for i in range(2):
            xyz = self.physics.get_orbit_xyz_ring(i, num_points=self.num_positions)
            self._orbit_coords.append((xyz[:, 0], xyz[:, 1], xyz[:, 2]))'''
            
    content = content.replace(init_orbit, new_init_orbit)

    # 3. Update Rendering loop to use independent tracks and draw Earth
    render_start = content.find('    def render(self):')
    render_end = content.find('    # ── close')
    
    new_render = '''    def render(self):
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d.art3d as art3d
        from matplotlib.patches import Circle

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 8), facecolor='#05050f')
            self.ax_env = self.fig.add_subplot(1, 2, 1, projection='3d')
            self.ax_env.set_facecolor('#05050f')
            
            # Setup right panel for charts
            self.ax_rew = self.fig.add_subplot(2, 2, 2)
            self.ax_cum = self.fig.add_subplot(2, 2, 4)
            plt.tight_layout()

        self.ax_env.clear()
        self.ax_rew.clear()
        self.ax_cum.clear()

        # ── Draw 3D Earth ─────────────────────────────────────────────────────
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        x_earth = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
        y_earth = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
        z_earth = EARTH_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax_env.plot_surface(x_earth, y_earth, z_earth, color='#113366', alpha=0.9, edgecolors='none', zorder=1)

        # ── Draw Eclipse Shadow Cylinder ──────────────────────────────────────
        # Assuming sun is roughly in the +Y direction for simplicity in this visualization
        shadow_radius = EARTH_RADIUS_KM
        z_shadow = np.linspace(-EARTH_RADIUS_KM*1.5, EARTH_RADIUS_KM*1.5, 2)
        theta_shadow = np.linspace(np.pi, 2*np.pi, 30) # Dark side
        Theta_sh, Z_sh = np.meshgrid(theta_shadow, z_shadow)
        X_sh = shadow_radius * np.cos(Theta_sh)
        Y_sh = shadow_radius * np.sin(Theta_sh) - EARTH_RADIUS_KM * 1.5 # Offset into shadow
        self.ax_env.plot_surface(X_sh, Y_sh, Z_sh, color='black', alpha=0.3, zorder=2)

        # ── Draw Independent Orbits ───────────────────────────────────────────
        sat_colors = ['#ff3333', '#ffaa00']
        for i in range(2):
            x, y, z = self._orbit_coords[i]
            # Draw the faint orbit ring
            self.ax_env.plot(x, y, z, color=sat_colors[i], alpha=0.3, linestyle='--', linewidth=1)
            
            # Draw debris
            for d in self.debris:
                dp = d['pos'] % self.num_positions
                self.ax_env.scatter(x[dp], y[dp], z[dp], color='#888888', s=20, marker='x')

            # Draw satellite
            p = self.agent_pos[i] % self.num_positions
            
            # Find the true 3D cartesian coordinates of this satellite
            sat_x, sat_y, sat_z = x[p], y[p], z[p]
            
            self.ax_env.scatter(sat_x, sat_y, sat_z,
                           color=sat_colors[i], s=180, depthshade=False, zorder=5,
                           label=f'Sat {i+1}')
            
            # Glow effect
            self.ax_env.scatter(sat_x, sat_y, sat_z,
                           color=sat_colors[i], s=600, alpha=0.2, depthshade=False, zorder=4)

        # ── HUD text & Visual Bars ────────────────────────────────────────────
        # Calculate true 3D Euclidean distance between the two satellites
        x1, y1, z1 = self._orbit_coords[0]
        x2, y2, z2 = self._orbit_coords[1]
        p1 = self.agent_pos[0] % self.num_positions
        p2 = self.agent_pos[1] % self.num_positions
        true_dist = np.sqrt((x1[p1]-x2[p2])**2 + (y1[p1]-y2[p2])**2 + (z1[p1]-z2[p2])**2)
        
        sim_t_min = (self.current_step * STEP_SECONDS) / 60.0

        bar_len = 10
        def make_bar(val):
            filled = int((val / 100.0) * bar_len)
            return '[' + '█'*filled + '-'*(bar_len - filled) + ']'

        hud = (
            f"STEP: {self.current_step:03d} / {self.max_steps}    SIM TIME: {sim_t_min:.1f} min\\n"
            f"EUCLIDEAN GAP: {true_dist:.0f} km\\n\\n"
            f"SAT 1 (Red)\\n"
            f"  Pos:  {self.agent_pos[0]:03d}°\\n"
            f"  Fuel: {make_bar(self.agent_fuel[0])} {max(0, self.agent_fuel[0]):.1f}%\\n"
            f"  Batt: {make_bar(self.agent_battery[0])} {max(0, self.agent_battery[0]):.1f}%\\n\\n"
            f"SAT 2 (Orange)\\n"
            f"  Pos:  {self.agent_pos[1]:03d}°\\n"
            f"  Fuel: {make_bar(self.agent_fuel[1])} {max(0, self.agent_fuel[1]):.1f}%\\n"
            f"  Batt: {make_bar(self.agent_battery[1])} {max(0, self.agent_battery[1]):.1f}%\\n\\n"
            f"DEBRIS: {len(self.debris)}   "
            f"{' [ECLIPSE]' if self.eclipse_mode else ' [SUNLIGHT]'}   "
            f"{'[SOLAR STORM]' if self.space_weather_active else ''}"
        )

        hud_bg = '#2b0000' if self.space_weather_active else '#000010'
        if self.eclipse_mode:
            hud_bg = '#000000'

        self.ax_env.text2D(0.02, 0.97, hud, transform=self.ax_env.transAxes,
                      color='#ccddff', fontsize=9, family='monospace',
                      va='top',
                      bbox=dict(facecolor=hud_bg, alpha=0.8, edgecolor='none'))

        # ── Charts ────────────────────────────────────────────────────────────
        for ax in [self.ax_rew, self.ax_cum]:
            ax.set_facecolor('#0f0f1f')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#333344')
            ax.grid(color='#333344', linestyle='--', alpha=0.5)

        steps = range(1, len(self.reward_history[0]) + 1)
        if steps:
            self.ax_rew.plot(steps, self.reward_history[0], color='#ff3333', label='Sat 1', linewidth=1.5)
            self.ax_rew.plot(steps, self.reward_history[1], color='#ffaa00', label='Sat 2', linewidth=1.5)
            self.ax_rew.set_title("Reward per Step", color='white', pad=8, fontsize=10)
            self.ax_rew.legend(facecolor='#0a0a1a', edgecolor='none', labelcolor='white', fontsize=8)

            self.ax_cum.plot(steps, self.cumulative_reward_history[0], color='#ff3333', label='Sat 1', linewidth=2)
            self.ax_cum.plot(steps, self.cumulative_reward_history[1], color='#ffaa00', label='Sat 2', linewidth=2)
            self.ax_cum.set_title("Cumulative Reward", color='white', pad=8, fontsize=10)

        # ── 3D camera ─────────────────────────────────────────────────────────
        avg_r = EARTH_RADIUS_KM + 550  # Roughly LEO
        lim = avg_r * 1.2
        self.ax_env.set_xlim(-lim, lim)
        self.ax_env.set_ylim(-lim, lim)
        self.ax_env.set_zlim(-lim, lim)
        self.ax_env.axis('off')
        
        # Smooth cinematic camera rotation
        self.ax_env.view_init(elev=20 + np.sin(self.current_step/20)*5, azim=45 + self.current_step * 0.5)

        self.fig.canvas.draw()
        plt.pause(0.01)

'''
    
    content = content[:render_start] + new_render + content[render_end:]

    # Write it back
    with open('/home/yug/My_Practice/spacetech/satellite_env.py', 'w') as f:
        f.write(content)
        
    print("Upgraded satellite_env.py")

if __name__ == '__main__':
    main()
