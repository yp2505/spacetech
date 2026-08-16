function visualize_project_full()
% ============================================================================
%  visualize_project_full.m — SpaceTech Live 360-Cycle Animated Dashboard
% ============================================================================
%  FIXED: 
%    - Added 'axis vis3d' and fixed 3D limits to prevent 3D Earth shrinking
%    - Full 100% screen stretch layout
% ============================================================================

clc; close all;
fprintf('============================================================\n');
fprintf('  SpaceTech Live 360-Cycle Synchronized Dashboard\n');
fprintf('============================================================\n');

% ─────────────────────────────────────────────────────────────────────────────
%  COLOR PALETTE (Deep Space Dark Theme)
% ─────────────────────────────────────────────────────────────────────────────
BG      = [0.043 0.055 0.102];   % Figure background
AX_BG   = [0.051 0.063 0.125];   % Axes background
GR      = [0.118 0.145 0.251];   % Grid color
TC      = [0.816 0.847 0.973];   % Text color
CYAN    = [0.000 0.831 1.000];
RED     = [1.000 0.267 0.267];
AMBER   = [1.000 0.863 0.239];
GREEN   = [0.267 1.000 0.533];
PURPLE  = [0.780 0.290 1.000];
ORANGE  = [1.000 0.439 0.263];
PINK    = [1.000 0.420 0.706];
TEAL    = [0.000 0.898 0.753];
GOLD    = [1.000 0.843 0.000];
WHITE   = [1.000 1.000 1.000];

CLRS = [GREEN; AMBER; RED; PURPLE; TEAL];
SAT_COL = [CYAN; RED; AMBER; GREEN; PURPLE; ORANGE; PINK; TEAL; GOLD; [0.2 0.6 1.0]];

% ─────────────────────────────────────────────────────────────────────────────
%  LOAD DATA FILES
% ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[1/3] Loading simulation data files...\n');

ep      = safe_load(fullfile('outputs','episode_data.csv'),       'ep');
rew     = safe_load(fullfile('outputs','reward_history.csv'),     'rew');
rew_new = safe_load(fullfile('outputs','reward_history_new.csv'), 'rew_new');
gs      = safe_load(fullfile('outputs','gs_positions.csv'),       'gs');

target_rew = rew;
if isempty(target_rew), target_rew = rew_new; end

if isempty(ep)
    error('Error: episode_data.csv could not be loaded. Check your folder!');
end

n_sats  = max(ep.sat_id) + 1;
n_steps = max(ep.step);
fprintf('  Loaded %d Satellites across %d Steps.\n', n_sats, n_steps);

% Compute Mean Reward array across satellites
if ~isempty(target_rew)
    sc = varnames_starting(target_rew, 'sat');
    if ~isempty(sc)
        raw_rwd = table2array(target_rew(:, sc));
        raw_rwd(raw_rwd < -5) = NaN;
        mr = mean(raw_rwd, 2, 'omitnan');
    else
        mr = zeros(n_steps, 1);
    end
else
    mr = zeros(n_steps, 1);
end

% ─────────────────────────────────────────────────────────────────────────────
%  SETUP DASHBOARD LAYOUT (Full Screen 3x3 Grid)
% ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[2/3] Building Synchronized Live Dashboard...\n');

fig = figure('Name', 'SpaceTech AI Satellite Constellation — Live 360-Cycle Animation', ...
             'NumberTitle', 'off', 'Color', BG, ...
             'Units', 'normalized', 'OuterPosition', [0 0 1 1], 'WindowState', 'maximized');

t = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'SpaceTech Constellation — Real-Time 360-Step Simulation & Telemetry Tracker', ...
      'Color', WHITE, 'FontSize', 13, 'FontWeight', 'bold');

% ── PANEL 1 (Main Center/Left 2x2): 3D Constellation Orbit Animation ────────
ax3d = nexttile(t, 1, [2 2]);
style_ax(ax3d, AX_BG, GR, TC);
set(ax3d, 'Projection', 'perspective', 'DataAspectRatio', [1 1 1]);
hold(ax3d, 'on'); grid(ax3d, 'on'); axis(ax3d, 'equal');

% Lock 3D Axes Limits & Freeze Aspect Ratio so camera rotation NEVER shrinks Earth!
xlim(ax3d, [-8200 8200]);
ylim(ax3d, [-8200 8200]);
zlim(ax3d, [-8200 8200]);
axis(ax3d, 'vis3d');  % <--- PREVENTS ZOOM/SHRINKING DURING ANIMATION

view(ax3d, 30, 22);
xlabel(ax3d, 'X (km)', 'Color', TC, 'FontSize', 8);
ylabel(ax3d, 'Y (km)', 'Color', TC, 'FontSize', 8);
zlabel(ax3d, 'Z (km)', 'Color', TC, 'FontSize', 8);

% 3D Earth Globe
[SX, SY, SZ] = sphere(28);
Re = 6371;
surf(ax3d, SX*Re, SY*Re, SZ*Re, 'FaceColor', [0.07 0.18 0.42], ...
     'EdgeColor', [0.10 0.22 0.52], 'EdgeAlpha', 0.2, 'FaceAlpha', 0.94, ...
     'FaceLighting', 'none', 'HandleVisibility', 'off');

% Equator Ring
th_eq = linspace(0, 2*pi, 90);
plot3(ax3d, Re*cos(th_eq), Re*sin(th_eq), zeros(size(th_eq)), ...
      '-', 'Color', [0.2 0.5 0.9 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');

% Debris Field
rng(42);
N_deb = 60;
r_deb  = Re + 550 + randn(N_deb, 1)*60;
az_deb = rand(N_deb, 1)*2*pi;
el_deb = (rand(N_deb, 1) - 0.5)*pi*0.3;
scatter3(ax3d, r_deb.*cos(el_deb).*cos(az_deb), ...
               r_deb.*cos(el_deb).*sin(az_deb), ...
               r_deb.*sin(el_deb), ...
               8, 'MarkerFaceColor', [1 0.25 0.08], 'MarkerEdgeColor', 'none', ...
               'MarkerFaceAlpha', 0.6, 'DisplayName', 'Debris');

% Ground Stations 3D
if ~isempty(gs)
    for g = 1:height(gs)
        gx = gs.ecef_x_km(g); gy = gs.ecef_y_km(g); gz = gs.ecef_z_km(g);
        scatter3(ax3d, gx*1.02, gy*1.02, gz*1.02, 70, 'v', 'filled', ...
                 'MarkerFaceColor', GOLD, 'MarkerEdgeColor', 'w', 'HandleVisibility', 'off');
        text(ax3d, gx*1.06, gy*1.06, gz*1.06, gs.name{g}, ...
             'Color', GOLD, 'FontSize', 7, 'FontWeight', 'bold');
    end
end

% Initial Satellites 3D Scatter Object
s1 = ep(ep.step == 1, :);
s1 = sortrows(s1, 'sat_id');
C0 = zeros(n_sats, 3);
for i = 0:n_sats-1
    row = s1(s1.sat_id == i, :);
    if ~isempty(row), C0(i+1, :) = hclr(row, CLRS); end
end
sat_sc = scatter3(ax3d, s1.ecef_x_km, s1.ecef_y_km, s1.ecef_z_km, ...
                  140, C0, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.7, ...
                  'DisplayName', 'Satellites');

legend(ax3d, 'show', 'Location', 'northwest', 'TextColor', WHITE, ...
       'Color', AX_BG, 'EdgeColor', GR, 'FontSize', 7.5);


% ── PANEL 2 (Top Right): Live AI Reward Tracker ─────────────────────────────
ax_rwd = nexttile(t, 3);
style_ax(ax_rwd, AX_BG, GR, TC);
hold(ax_rwd, 'on');
plot(ax_rwd, 1:length(mr), mr, '-', 'Color', CYAN, 'LineWidth', 1.5, 'DisplayName', 'Fleet Mean');
yline(ax_rwd, 300, '--', 'Color', GREEN, 'LineWidth', 1.0, 'HandleVisibility', 'off');
yline(ax_rwd, 0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'HandleVisibility', 'off');

rwd_trk = plot(ax_rwd, 1, mr(1), 'o', 'Color', GOLD, 'MarkerSize', 7, ...
               'MarkerFaceColor', GOLD, 'DisplayName', 'Active Step');

xlabel(ax_rwd, 'Step', 'Color', TC, 'FontSize', 8);
ylabel(ax_rwd, 'Reward', 'Color', TC, 'FontSize', 8);
title(ax_rwd, 'AI Training — Reward per Step', 'Color', TC, 'FontWeight', 'bold', 'FontSize', 9);
legend(ax_rwd, 'show', 'Location', 'northwest', 'TextColor', WHITE, 'Color', AX_BG, 'EdgeColor', GR, 'FontSize', 7);


% ── PANEL 3 (Middle Right): Live Battery, Fuel & Altitude Tracker ───────────
ax_sub = nexttile(t, 6);
style_ax(ax_sub, AX_BG, GR, TC);
grp_sub = groupsummary(ep, 'step', 'mean', {'altitude_km', 'battery_pct', 'fuel_pct'});

yyaxis(ax_sub, 'left');
plot(ax_sub, grp_sub.step, grp_sub.mean_altitude_km, '-', 'Color', CYAN, 'LineWidth', 1.5);
ylabel(ax_sub, 'Alt (km)', 'Color', CYAN, 'FontSize', 8);
ax_sub.YAxis(1).Color = CYAN;

yyaxis(ax_sub, 'right');
plot(ax_sub, grp_sub.step, grp_sub.mean_battery_pct, '--', 'Color', GOLD, 'LineWidth', 1.2);
hold(ax_sub, 'on');
plot(ax_sub, grp_sub.step, grp_sub.mean_fuel_pct, ':', 'Color', GREEN, 'LineWidth', 1.2);
ylabel(ax_sub, 'Level (%)', 'Color', GOLD, 'FontSize', 8);
ax_sub.YAxis(2).Color = GOLD;

sub_cursor = xline(ax_sub, 1, '--y', 'LineWidth', 1.2, 'HandleVisibility', 'off');

xlabel(ax_sub, 'Step', 'Color', TC, 'FontSize', 8);
title(ax_sub, 'Subsystems — Alt, Battery & Fuel', 'Color', TC, 'FontWeight', 'bold', 'FontSize', 9);


% ── PANEL 4 (Bottom Left): Live 2D Ground Track Map ─────────────────────────
ax_map = nexttile(t, 7);
style_ax(ax_map, AX_BG, GR, TC);
hold(ax_map, 'on'); grid(ax_map, 'on');

ep_s = ep(mod(ep.step, 3) == 0, :);
sids = unique(ep_s.sat_id);
for i = 1:numel(sids)
    sub_gt = ep_s(ep_s.sat_id == sids(i), :);
    r_gt   = sqrt(sub_gt.ecef_x_km.^2 + sub_gt.ecef_y_km.^2 + sub_gt.ecef_z_km.^2);
    lat_gt = rad2deg(asin(sub_gt.ecef_z_km ./ r_gt));
    lon_gt = rad2deg(atan2(sub_gt.ecef_y_km, sub_gt.ecef_x_km));
    c_gt   = SAT_COL(mod(sids(i), 10) + 1, :);
    scatter(ax_map, lon_gt, lat_gt, 1.5, c_gt, 'filled', 'MarkerFaceAlpha', 0.25, 'HandleVisibility', 'off');
end

if ~isempty(gs)
    scatter(ax_map, gs.lon_deg, gs.lat_deg, 60, '^', ...
            'MarkerFaceColor', GOLD, 'MarkerEdgeColor', 'w', 'LineWidth', 0.8, 'DisplayName', 'GS');
end

map_sats = scatter(ax_map, zeros(n_sats, 1), zeros(n_sats, 1), 35, ...
                   C0, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5, 'DisplayName', 'Satellites');

xlim(ax_map, [-180 180]); ylim(ax_map, [-90 90]);
xlabel(ax_map, 'Lon (°)', 'Color', TC, 'FontSize', 8);
ylabel(ax_map, 'Lat (°)', 'Color', TC, 'FontSize', 8);
title(ax_map, 'Live 2D Ground Track', 'Color', TC, 'FontWeight', 'bold', 'FontSize', 9);


% ── PANEL 5 (Bottom Center): Live Operational Status Tracker ─────────────────
ax_ops = nexttile(t, 8);
style_ax(ax_ops, AX_BG, GR, TC);
grp_ops = groupsummary(ep, 'step', 'mean', {'in_safe_mode', 'in_recovery', 'eclipse', 'gs_los'});
hold(ax_ops, 'on');
plot(ax_ops, grp_ops.step, grp_ops.mean_in_safe_mode*100, '-', 'Color', RED, 'LineWidth', 1.2, 'DisplayName', 'Safe Mode');
plot(ax_ops, grp_ops.step, grp_ops.mean_eclipse*100, '-', 'Color', [0.4 0.4 1.0], 'LineWidth', 1.2, 'DisplayName', 'Eclipse');
plot(ax_ops, grp_ops.step, grp_ops.mean_gs_los*100, '-', 'Color', GREEN, 'LineWidth', 1.2, 'DisplayName', 'GS LOS');

ops_cursor = xline(ax_ops, 1, '--y', 'LineWidth', 1.2, 'HandleVisibility', 'off');

ylim(ax_ops, [0 110]);
xlabel(ax_ops, 'Step', 'Color', TC, 'FontSize', 8);
ylabel(ax_ops, '% Fleet', 'Color', TC, 'FontSize', 8);
title(ax_ops, 'Operational Status (% Fleet)', 'Color', TC, 'FontWeight', 'bold', 'FontSize', 9);
legend(ax_ops, 'show', 'Location', 'northeast', 'TextColor', WHITE, 'Color', AX_BG, 'EdgeColor', GR, 'FontSize', 6.5);


% ── PANEL 6 (Bottom Right): Live Attitude Error Tracker ──────────────────────
ax_att = nexttile(t, 9);
style_ax(ax_att, AX_BG, GR, TC);
grp_att = groupsummary(ep, 'step', 'mean', {'roll_deg', 'pitch_deg', 'yaw_deg'});
wrap = @(v) v - 360*(v > 180);

hold(ax_att, 'on');
plot(ax_att, grp_att.step, wrap(grp_att.mean_roll_deg), '-', 'Color', CYAN, 'LineWidth', 1.2, 'DisplayName', 'Roll');
plot(ax_att, grp_att.step, wrap(grp_att.mean_pitch_deg), '-', 'Color', AMBER, 'LineWidth', 1.2, 'DisplayName', 'Pitch');
plot(ax_att, grp_att.step, wrap(grp_att.mean_yaw_deg), '-', 'Color', GREEN, 'LineWidth', 1.2, 'DisplayName', 'Yaw');
yline(ax_att, 0, '--', 'Color', [1 1 1 0.3], 'LineWidth', 0.8, 'HandleVisibility', 'off');

att_cursor = xline(ax_att, 1, '--y', 'LineWidth', 1.2, 'HandleVisibility', 'off');

xlabel(ax_att, 'Step', 'Color', TC, 'FontSize', 8);
ylabel(ax_att, 'Error (°)', 'Color', TC, 'FontSize', 8);
title(ax_att, 'Attitude Error (Roll/Pitch/Yaw)', 'Color', TC, 'FontWeight', 'bold', 'FontSize', 9);
legend(ax_att, 'show', 'Location', 'northwest', 'TextColor', WHITE, 'Color', AX_BG, 'EdgeColor', GR, 'FontSize', 6.5);


% ─────────────────────────────────────────────────────────────────────────────
%  REAL-TIME ANIMATION LOOP (360 CYCLES)
% ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[3/3] Starting Live Animation Loop across all %d steps...\n', n_steps);

for s = 1:n_steps
    if ~isvalid(fig), break; end
    
    sd = ep(ep.step == s, :);
    sd = sortrows(sd, 'sat_id');
    
    % 1. Update 3D Orbit Positions & Status Colors
    sat_sc.XData = sd.ecef_x_km;
    sat_sc.YData = sd.ecef_y_km;
    sat_sc.ZData = sd.ecef_z_km;
    
    C  = zeros(n_sats, 3);
    SZ = ones(n_sats, 1) * 120;
    for i = 0:n_sats-1
        row = sd(sd.sat_id == i, :);
        if isempty(row), continue; end
        C(i+1, :) = hclr(row, CLRS);
        if row.fault_wheel || row.fault_thruster || row.fault_sensor
            SZ(i+1) = 260;
        elseif row.in_safe_mode
            SZ(i+1) = 180;
        end
    end
    sat_sc.CData    = C;
    sat_sc.SizeData = SZ;
    
    % Smooth camera rotation
    view(ax3d, 30 + s*0.35, 20 + 3*sin(s/45));
    
    % Telemetry HUD in 3D Title
    n_flt  = sum(sd.fault_wheel) + sum(sd.fault_thruster) + sum(sd.fault_sensor);
    n_safe = sum(sd.in_safe_mode);
    n_los  = sum(sd.gs_los);
    n_ecl  = sum(sd.eclipse);
    title(ax3d, sprintf('3D Constellation Orbit — Step %d/%d  |  Faults: %d  Safe: %d  LOS: %d  Eclipse: %d', ...
          s, n_steps, n_flt, n_safe, n_los, n_ecl), ...
          'Color', WHITE, 'FontSize', 9, 'FontWeight', 'bold');

    % 2. Update Live Reward Marker
    if s <= length(mr)
        rwd_trk.XData = s;
        rwd_trk.YData = mr(s);
    end
    
    % 3. Update 2D Ground Track Map Positions
    r_map   = sqrt(sd.ecef_x_km.^2 + sd.ecef_y_km.^2 + sd.ecef_z_km.^2);
    lat_map = rad2deg(asin(sd.ecef_z_km ./ r_map));
    lon_map = rad2deg(atan2(sd.ecef_y_km, sd.ecef_x_km));
    map_sats.XData = lon_map;
    map_sats.YData = lat_map;
    map_sats.CData = C;

    % 4. Update Vertical Cursors on Subsystem, Operational Status & Attitude Graphs
    sub_cursor.Value = s;
    ops_cursor.Value = s;
    att_cursor.Value = s;
    
    % Render Frame
    drawnow limitrate;
    pause(0.01);
end

fprintf('\nAnimation completed successfully!\n');

end % Main Function End


% ============================================================================
%  HELPER FUNCTIONS
% ============================================================================

function style_ax(ax, ax_bg, gr_col, tc)
    set(ax, 'Color', ax_bg, ...
        'XColor', tc, 'YColor', tc, 'ZColor', tc, ...
        'GridColor', gr_col, 'MinorGridColor', gr_col, ...
        'GridAlpha', 0.5, 'MinorGridAlpha', 0.3, ...
        'Box', 'on', 'FontSize', 7.5);
    grid(ax, 'on');
end

function tbl = safe_load(fpath, name)
    target = '';
    if isfile(fpath)
        target = fpath;
    else
        [~, fname, fext] = fileparts(fpath);
        direct_file = [fname fext];
        if isfile(direct_file)
            target = direct_file;
        else
            d = dir(fullfile('**', [fname '*.csv']));
            if ~isempty(d)
                target = fullfile(d(1).folder, d(1).name);
            end
        end
    end
    
    if ~isempty(target) && isfile(target)
        try
            tbl = readtable(target, 'VariableNamingRule', 'preserve');
            tbl.Properties.VariableNames = matlab.lang.makeValidName(tbl.Properties.VariableNames);
            fprintf('  OK  %-10s <- %s (%d rows)\n', name, target, height(tbl));
        catch e
            fprintf('  ERR %-10s : %s\n', name, e.message);
            tbl = table();
        end
    else
        fprintf('  --  %-10s (file not found: %s)\n', name, fpath);
        tbl = table();
    end
end

function names = varnames_starting(tbl, prefix)
    all_names = tbl.Properties.VariableNames;
    names = all_names(startsWith(all_names, prefix));
end

function c = hclr(row, CLRS)
    if row.in_safe_mode
        c = CLRS(4,:); % Safe mode (magenta)
    elseif row.fault_wheel || row.fault_thruster || row.fault_sensor
        c = CLRS(3,:); % Fault / critical (red)
    elseif row.battery_pct < 30 || row.fuel_pct < 15
        c = CLRS(2,:); % Degraded (amber)
    elseif row.eclipse
        c = CLRS(5,:); % Eclipse (blue)
    else
        c = CLRS(1,:); % Nominal (green)
    end
end