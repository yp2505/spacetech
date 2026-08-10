% ============================================================================
%  visualize_project_full.m  —  SpaceTech Complete Dashboard v1.0
% ============================================================================
%
%  PURPOSE : Visualize ALL model data from the SpaceTech FSW + PPO RL project
%            in a professional dark-themed multi-panel dashboard.
%
%  FILES NEEDED (copy all from your project into ONE folder, then cd to it):
%    outputs/episode_data.csv            <- main simulation log
%    outputs/reward_history.csv          <- per-satellite rewards (Run 1)
%    outputs/reward_history_new.csv      <- per-satellite rewards (Run 2)
%    outputs/gs_positions.csv            <- ground station lat/lon/ECEF
%    matlab_export/training_progress.csv <- PPO training cycles
%    matlab_export/positions_history.csv <- orbital slot positions
%    matlab_export/reward_history.csv    <- reward from ML export
%    matlab_export/continual_learning_results.csv  <- EWC results
%
%  HOW TO RUN:
%    1. Open MATLAB
%    2. cd to the project root folder  (where this .m file lives)
%    3. Run:   visualize_project_full
%
%  OUTPUT : 4 separate dark-themed figure windows covering:
%    Fig 1 - PPO Training Dashboard  (4 panels)
%    Fig 2 - Satellite Health Dashboard  (4 panels)
%    Fig 3 - 3D Orbit + Ground Map  (2 panels)
%    Fig 4 - Continual Learning + EWC (2 panels)
%
% ============================================================================

clear; clc; close all;
fprintf('============================================================\n');
fprintf('  SpaceTech Full MATLAB Visualization Dashboard\n');
fprintf('============================================================\n');

% ─────────────────────────────────────────────────────────────────────────────
%  COLOUR PALETTE  (deep-space dark theme)
% ─────────────────────────────────────────────────────────────────────────────
BG      = [0.043 0.055 0.102];   % figure background
AX_BG   = [0.051 0.063 0.125];   % axes background
GR      = [0.118 0.145 0.251];   % grid colour
TC      = [0.816 0.847 0.973];   % text colour
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

SAT_COL = [CYAN; RED; AMBER; GREEN; PURPLE;
           ORANGE; PINK; TEAL; GOLD; [0.2 0.6 1.0]];

% ─────────────────────────────────────────────────────────────────────────────
%  LOAD ALL DATA FILES
% ─────────────────────────────────────────────────────────────────────────────
fprintf('\n[1/4] Loading data files...\n');

train   = safe_load(fullfile('matlab_export','training_progress.csv'),       'train');
pos     = safe_load(fullfile('matlab_export','positions_history.csv'),        'pos');
cl      = safe_load(fullfile('matlab_export','continual_learning_results.csv'),'cl');
ep      = safe_load(fullfile('outputs','episode_data.csv'),                   'ep');
rew     = safe_load(fullfile('outputs','reward_history.csv'),                 'rew');
rew_new = safe_load(fullfile('outputs','reward_history_new.csv'),             'rew_new');
gs      = safe_load(fullfile('outputs','gs_positions.csv'),                   'gs');

fprintf('\n[2/4] Building figures...\n');

% =============================================================================
%  FIGURE 1 — PPO TRAINING DASHBOARD
% =============================================================================
fig1 = make_fig('SpaceTech  |  PPO Training Dashboard', BG);
t1   = tiledlayout(fig1, 2, 2, 'TileSpacing','compact','Padding','compact');
title(t1, 'PPO Reinforcement Learning — Training Overview', ...
      'Color', WHITE, 'FontSize', 15, 'FontWeight', 'bold');

% ── 1-A  Training reward curves ───────────────────────────────────────────────
ax = nexttile(t1);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(train)
    x = train.steps_completed / 1000;
    plot(ax, x, train.sat1_reward_mean, '-',  'Color', CYAN,   'LineWidth', 2.2, ...
         'DisplayName', 'Sat-1 Reward');
    hold(ax,'on');
    plot(ax, x, train.sat2_reward_mean, '-',  'Color', RED,    'LineWidth', 2.2, ...
         'DisplayName', 'Sat-2 Reward');
    % Smoothed rolling average
    plot(ax, x, movmean(train.sat1_reward_mean,12), '--', 'Color', CYAN*0.7,   ...
         'LineWidth', 1.2, 'HandleVisibility','off');
    plot(ax, x, movmean(train.sat2_reward_mean,12), '--', 'Color', RED*0.75,   ...
         'LineWidth', 1.2, 'HandleVisibility','off');
    % Fill area
    fill(ax, [x; flipud(x)], ...
         [train.sat1_reward_mean; zeros(height(train),1)], ...
         CYAN, 'FaceAlpha',0.07,'EdgeColor','none','HandleVisibility','off');
    fill(ax, [x; flipud(x)], ...
         [train.sat2_reward_mean; zeros(height(train),1)], ...
         RED,  'FaceAlpha',0.07,'EdgeColor','none','HandleVisibility','off');
    yline(ax, 300, '--', 'Color', GREEN, 'LineWidth', 1.0, ...
          'Label','Target 300', 'LabelHorizontalAlignment','left', ...
          'HandleVisibility','off');
end
xlabel(ax,'Training Steps (x1 000)', 'Color',TC,'FontSize',9);
ylabel(ax,'Mean Episode Reward',      'Color',TC,'FontSize',9);
title(ax,'PPO Training — Cumulative Reward', 'Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northwest','TextColor',WHITE,'Color',AX_BG, ...
       'EdgeColor',GR,'FontSize',8);

% ── 1-B  Value estimates ───────────────────────────────────────────────────────
ax = nexttile(t1);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(train)
    x = train.steps_completed / 1000;
    plot(ax, x, train.sat1_V_estimate, '-','Color',AMBER,'LineWidth',2,'DisplayName','Sat-1 V(s)');
    hold(ax,'on');
    plot(ax, x, train.sat2_V_estimate, '-','Color',GREEN,'LineWidth',2,'DisplayName','Sat-2 V(s)');
    fill(ax,[x; flipud(x)],[train.sat1_V_estimate; zeros(height(train),1)], ...
         AMBER,'FaceAlpha',0.1,'EdgeColor','none','HandleVisibility','off');
    fill(ax,[x; flipud(x)],[train.sat2_V_estimate; zeros(height(train),1)], ...
         GREEN,'FaceAlpha',0.1,'EdgeColor','none','HandleVisibility','off');
end
xlabel(ax,'Training Steps (x1 000)','Color',TC,'FontSize',9);
ylabel(ax,'V-Estimate (Critic)',     'Color',TC,'FontSize',9);
title(ax,'Critic Value Estimates V(s)','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northwest','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% ── 1-C  Fleet reward comparison: Run1 vs Run2 ───────────────────────────────
ax = nexttile(t1);
style_ax(ax, AX_BG, GR, TC);
hold(ax,'on');
if ~isempty(rew)
    sc = varnames_starting(rew,'sat');
    if ~isempty(sc)
        raw = table2array(rew(:,sc));
        raw(raw < -5) = NaN;
        fm1 = mean(raw, 2, 'omitnan');
        sm1 = movmean(fm1, 20, 'omitnan');
        plot(ax, rew.step, sm1, '-','Color',CYAN,'LineWidth',2,'DisplayName','Run 1 (smoothed)');
        fill(ax,[rew.step; flipud(rew.step)],[sm1; zeros(height(rew),1)], ...
             CYAN,'FaceAlpha',0.1,'EdgeColor','none','HandleVisibility','off');
    end
end
if ~isempty(rew_new)
    sc2 = varnames_starting(rew_new,'sat');
    if ~isempty(sc2)
        raw2 = table2array(rew_new(:,sc2));
        raw2(raw2 < -5) = NaN;
        fm2  = mean(raw2, 2, 'omitnan');
        sm2  = movmean(fm2, 20, 'omitnan');
        plot(ax, rew_new.step, sm2, '-','Color',TEAL,'LineWidth',2,'DisplayName','Run 2 (smoothed)');
        fill(ax,[rew_new.step; flipud(rew_new.step)],[sm2; zeros(height(rew_new),1)], ...
             TEAL,'FaceAlpha',0.1,'EdgeColor','none','HandleVisibility','off');
    end
end
xlabel(ax,'Simulation Step',           'Color',TC,'FontSize',9);
ylabel(ax,'Fleet Mean Reward',         'Color',TC,'FontSize',9);
title(ax,'Fleet Mean Reward — Run Comparison','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northwest','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% ── 1-D  Per-satellite reward heatmap ────────────────────────────────────────
ax = nexttile(t1);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(rew)
    sc  = varnames_starting(rew,'sat');
    if ~isempty(sc)
        mat = table2array(rew(:,sc))';
        mat = max(min(mat, 1.5), -3);
        imagesc(ax, mat);
        colormap(ax, hot_cold_cmap());
        caxis(ax,[-3 1.5]);
        cb = colorbar(ax);
        cb.Color = TC;
        cb.Label.String = 'Reward';
        cb.Label.Color  = TC;
        ax.YTick     = 1:length(sc);
        ax.YTickLabel= arrayfun(@(i) sprintf('Sat-%d',i-1), 1:length(sc),'Uni',0);
        ax.XLabel.String = 'Simulation Step';
        ax.XLabel.Color  = TC;
    end
end
title(ax,'Per-Satellite Reward Heatmap (clipped -3 to +1.5)', ...
      'Color',TC,'FontWeight','bold');

% =============================================================================
%  FIGURE 2 — SATELLITE HEALTH DASHBOARD
% =============================================================================
fig2 = make_fig('SpaceTech  |  Satellite Health Dashboard', BG);
t2   = tiledlayout(fig2, 2, 2, 'TileSpacing','compact','Padding','compact');
title(t2,'Satellite Health, Faults & Thermal Monitoring', ...
      'Color',WHITE,'FontSize',15,'FontWeight','bold');

% ── 2-A  Battery + Fuel + Altitude ───────────────────────────────────────────
ax = nexttile(t2);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(ep)
    grp   = groupsummary(ep, 'step', 'mean', {'altitude_km','battery_pct','fuel_pct'});
    ax2   = yyaxis(ax,'right');
    style_ax_right(ax2, TC);
    plot(ax2, grp.step, grp.mean_battery_pct, '--', 'Color', GOLD,  'LineWidth',1.8,'DisplayName','Battery %');
    hold(ax2,'on');
    plot(ax2, grp.step, grp.mean_fuel_pct,    ':',  'Color', GREEN, 'LineWidth',1.8,'DisplayName','Fuel %');
    ylabel(ax2,'Subsystem Level (%)','Color',TC,'FontSize',9);
    yyaxis(ax,'left');
    plot(ax, grp.step, grp.mean_altitude_km, '-', 'Color',CYAN,'LineWidth',2,'DisplayName','Altitude (km)');
    hold(ax,'on');
    ylabel(ax,'Altitude (km)','Color',TC,'FontSize',9);
end
xlabel(ax,'Step','Color',TC,'FontSize',9);
title(ax,'Fleet-Mean: Altitude, Battery & Fuel','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','southeast','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% ── 2-B  Temperature (min/mean/max band) ─────────────────────────────────────
ax = nexttile(t2);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(ep)
    grp = groupsummary(ep,'step','mean','temp_c');
    grpmin = groupsummary(ep,'step','min', 'temp_c');
    grpmax = groupsummary(ep,'step','max', 'temp_c');
    hold(ax,'on');
    fill(ax,[grp.step; flipud(grp.step)], ...
         [grpmax.max_temp_c; flipud(grpmin.min_temp_c)], ...
         ORANGE,'FaceAlpha',0.2,'EdgeColor','none','HandleVisibility','off');
    plot(ax, grp.step, grp.mean_temp_c, '-','Color',ORANGE,'LineWidth',2,'DisplayName','Mean Temp');
    yline(ax, 35, '--','Color',RED,  'LineWidth',1.2,'Label','Max Safe (35C)','HandleVisibility','off','LabelHorizontalAlignment','left');
    yline(ax, 15, '--','Color',TEAL, 'LineWidth',1.2,'Label','Min Safe (15C)','HandleVisibility','off','LabelHorizontalAlignment','left');
end
xlabel(ax,'Step',         'Color',TC,'FontSize',9);
ylabel(ax,'Temperature (C)','Color',TC,'FontSize',9);
title(ax,'Thermal Dynamics — Satellite Temperature', 'Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northeast','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% ── 2-C  Fault events stacked bar (50-step bins) ─────────────────────────────
ax = nexttile(t2);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(ep)
    ep.bin = floor(ep.step / 50) * 50;
    binned = groupsummary(ep,'bin','sum',{'fault_wheel','fault_thruster','fault_sensor'});
    hold(ax,'on');
    bar(ax, binned.bin, binned.sum_fault_wheel,    'FaceColor',RED,    'EdgeColor','none','DisplayName','Wheel Fault');
    bar(ax, binned.bin, binned.sum_fault_thruster,  'FaceColor',AMBER,  'EdgeColor','none','DisplayName','Thruster Fault', ...
        'BarLayout','stacked','BaseValue',0);
    % Manual stacking
    bar(ax, binned.bin, [binned.sum_fault_wheel, binned.sum_fault_thruster, binned.sum_fault_sensor], ...
        'stacked','EdgeColor','none');
    % Redo manually with correct colors
    cla(ax);
    b1 = bar(ax, binned.bin, binned.sum_fault_wheel, 0.9, ...
             'FaceColor', RED,   'EdgeColor','none','DisplayName','Wheel');
    hold(ax,'on');
    b2 = bar(ax, binned.bin, binned.sum_fault_thruster, 0.9, ...
             'FaceColor', AMBER, 'EdgeColor','none','DisplayName','Thruster', ...
             'BottomEdge', binned.sum_fault_wheel);

    % Simplest stacked bar approach: overlay transparent bars
    cla(ax);
    bdat = [binned.sum_fault_wheel, binned.sum_fault_thruster, binned.sum_fault_sensor];
    bclr = {RED; AMBER; [1 1 0.3]};
    btm  = zeros(height(binned),1);
    for k=1:3
        bh = bar(ax, binned.bin, btm + bdat(:,k), 0.85, 'FaceColor', bclr{k}{:}, ...
                 'EdgeColor','none');
        hold(ax,'on');
        btm = btm + bdat(:,k);
    end
end
xlabel(ax,'Step Bin', 'Color',TC,'FontSize',9);
ylabel(ax,'Fault Count','Color',TC,'FontSize',9);
title(ax,'Fault Events by 50-Step Window','Color',TC,'FontWeight','bold');
legend(ax,{'Wheel','Thruster','Sensor'},'Location','northeast', ...
       'TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% ── 2-D  Safe-mode + Recovery + Eclipse + GS LOS ─────────────────────────────
ax = nexttile(t2);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(ep)
    grp = groupsummary(ep,'step','mean',{'in_safe_mode','in_recovery','eclipse','gs_los'});
    hold(ax,'on');
    area(ax, grp.step, grp.mean_in_safe_mode*100,  'FaceColor',RED,    'FaceAlpha',0.35,'EdgeColor','none','DisplayName','Safe Mode %');
    area(ax, grp.step, grp.mean_in_recovery*100,   'FaceColor',AMBER,  'FaceAlpha',0.35,'EdgeColor','none','DisplayName','Recovery %');
    plot(ax, grp.step, grp.mean_eclipse*100,  '-','Color',[0.4 0.4 1.0],'LineWidth',1.5,'DisplayName','Eclipse %');
    plot(ax, grp.step, grp.mean_gs_los*100,   '-','Color',GREEN,        'LineWidth',1.5,'DisplayName','GS LOS %');
    ylim(ax,[0 110]);
end
xlabel(ax,'Step',     'Color',TC,'FontSize',9);
ylabel(ax,'% of Fleet','Color',TC,'FontSize',9);
title(ax,'Safe-Mode / Recovery / Eclipse / GS LOS','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northeast','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% =============================================================================
%  FIGURE 3 — 3D ORBIT + GROUND TRACK MAP
% =============================================================================
fig3 = make_fig('SpaceTech  |  3D Orbit & Ground Map', BG);
t3   = tiledlayout(fig3, 1, 2, 'TileSpacing','compact','Padding','compact');
title(t3,'Constellation Orbital Geometry & Ground Track', ...
      'Color',WHITE,'FontSize',15,'FontWeight','bold');

% ── 3-A  3D ECEF orbit plot ───────────────────────────────────────────────────
ax = nexttile(t3);
style_ax(ax, AX_BG, GR, TC);
set(ax,'Projection','perspective','DataAspectRatio',[1 1 1]);
view(ax, 35, 22); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');

% Earth sphere
[SX,SY,SZ] = sphere(36);
Re = 6371;
surf(ax, SX*Re, SY*Re, SZ*Re, 'FaceColor',[0.07 0.18 0.42], ...
     'EdgeColor',[0.10 0.22 0.52],'EdgeAlpha',0.25,'FaceAlpha',0.92, ...
     'FaceLighting','none','HandleVisibility','off');

% Equator ring
th  = linspace(0,2*pi,200);
plot3(ax, Re*cos(th), Re*sin(th), zeros(size(th)), ...
      '-','Color',[0.3 0.55 1.0 0.45],'LineWidth',1,'HandleVisibility','off');

% Satellite positions (sub-sampled every 8 steps)
if ~isempty(ep)
    sids = unique(ep.sat_id);
    ep_s = ep(mod(ep.step,8)==0,:);
    for i=1:numel(sids)
        sub = ep_s(ep_s.sat_id == sids(i),:);
        c   = SAT_COL(mod(sids(i),10)+1,:);
        scatter3(ax, sub.ecef_x_km, sub.ecef_y_km, sub.ecef_z_km, ...
                 4, c, 'filled','MarkerFaceAlpha',0.65,'HandleVisibility','off');
    end
    % Final-step markers (large)
    last = ep(ep.step == max(ep.step),:);
    for i=1:numel(sids)
        sub = last(last.sat_id == sids(i),:);
        if isempty(sub), continue; end
        c = SAT_COL(mod(sids(i),10)+1,:);
        scatter3(ax, sub.ecef_x_km, sub.ecef_y_km, sub.ecef_z_km, ...
                 200, c,'filled','MarkerEdgeColor','w','LineWidth',0.8, ...
                 'DisplayName',sprintf('Sat-%d',sids(i)));
    end
end

% Ground stations
if ~isempty(gs)
    scatter3(ax, gs.ecef_x_km*1.02, gs.ecef_y_km*1.02, gs.ecef_z_km*1.02, ...
             160,'v','filled','MarkerFaceColor',GOLD,'MarkerEdgeColor','w', ...
             'LineWidth',0.8,'DisplayName','Ground Stn');
    for g=1:height(gs)
        text(ax, gs.ecef_x_km(g)*1.07, gs.ecef_y_km(g)*1.07, gs.ecef_z_km(g)*1.07, ...
             gs.name{g},'Color',GOLD,'FontSize',7,'FontWeight','bold');
    end
end

xlabel(ax,'X (km)','Color',TC,'FontSize',9);
ylabel(ax,'Y (km)','Color',TC,'FontSize',9);
zlabel(ax,'Z (km)','Color',TC,'FontSize',9);
title(ax,'3D ECEF Constellation Positions','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northwest','TextColor',WHITE,'Color',AX_BG, ...
       'EdgeColor',GR,'FontSize',7,'NumColumns',2);

% ── 3-B  Ground track + ground station 2D map ─────────────────────────────────
ax = nexttile(t3);
style_ax(ax, AX_BG, GR, TC);
hold(ax,'on');

% Draw simple coast-equivalent grid lines
for lat_line = -90:30:90
    xline(ax,lat_line,'Color',GR,'LineWidth',0.4,'HandleVisibility','off');
end
for lon_line = -180:30:180
    yline_compat(ax, lon_line, GR);
end

% Ground tracks for all satellites
if ~isempty(ep)
    ep_s = ep(mod(ep.step,3)==0,:);
    sids = unique(ep_s.sat_id);
    for i=1:numel(sids)
        sub = ep_s(ep_s.sat_id==sids(i),:);
        r   = sqrt(sub.ecef_x_km.^2 + sub.ecef_y_km.^2 + sub.ecef_z_km.^2);
        lat = rad2deg(asin(sub.ecef_z_km ./ r));
        lon = rad2deg(atan2(sub.ecef_y_km, sub.ecef_x_km));
        c   = SAT_COL(mod(sids(i),10)+1,:);
        scatter(ax, lon, lat, 3, c,'filled','MarkerFaceAlpha',0.5, ...
                'HandleVisibility','off');
    end
end

% Ground stations
if ~isempty(gs)
    scatter(ax, gs.lon_deg, gs.lat_deg, 200, '^', ...
            'MarkerFaceColor',GOLD,'MarkerEdgeColor','w','LineWidth',1.2, ...
            'DisplayName','Ground Stations');
    for g=1:height(gs)
        text(ax, gs.lon_deg(g)+3, gs.lat_deg(g)+3, gs.name{g}, ...
             'Color',GOLD,'FontSize',7.5,'FontWeight','bold');
    end
end

xlim(ax,[-180 180]); ylim(ax,[-90 90]);
xlabel(ax,'Longitude (deg)','Color',TC,'FontSize',9);
ylabel(ax,'Latitude (deg)', 'Color',TC,'FontSize',9);
title(ax,'Satellite Ground Tracks & Station Positions','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','southwest','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% =============================================================================
%  FIGURE 4 — CONTINUAL LEARNING + ATTITUDE + ORBITAL POSITIONS
% =============================================================================
fig4 = make_fig('SpaceTech  |  Continual Learning & Orbital State', BG);
t4   = tiledlayout(fig4, 2, 2, 'TileSpacing','compact','Padding','compact');
title(t4,'Continual Learning (EWC) + Attitude + Orbital Positions', ...
      'Color',WHITE,'FontSize',15,'FontWeight','bold');

% ── 4-A  EWC vs No-EWC bar chart ──────────────────────────────────────────────
ax = nexttile(t4);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(cl) && height(cl) >= 2
    scenarios = cl.scenario;
    metrics   = {'baseline_t1','after_task2_t1','task2_reward','forgetting'};
    mlabels   = {'Baseline T1','After T2 (T1)','Task-2 Reward','Forgetting'};
    n = numel(metrics);
    x = 1:n;
    w = 0.38;
    bclr = {ORANGE; TEAL};
    hold(ax,'on');
    for s=1:2
        vals = zeros(1,n);
        for m=1:n
            if ismember(metrics{m}, cl.Properties.VariableNames)
                vals(m) = cl.(metrics{m})(s);
            end
        end
        b = bar(ax, x + (s-1.5)*w, vals, w, 'FaceColor', bclr{s}{:}, ...
                'EdgeColor','none','DisplayName', scenarios{s});
        % Value labels on bars
        for m=1:n
            text(ax, x(m)+(s-1.5)*w, vals(m)+0.5, sprintf('%.1f',vals(m)), ...
                 'HorizontalAlignment','center','Color',TC,'FontSize',7.5);
        end
    end
    ax.XTick      = x;
    ax.XTickLabel = mlabels;
    ax.XTickLabelRotation = 12;
end
ylabel(ax,'Score','Color',TC,'FontSize',9);
title(ax,'Continual Learning — EWC vs No-EWC','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northeast','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% ── 4-B  Weight change magnitude comparison ────────────────────────────────────
ax = nexttile(t4);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(cl) && height(cl) >= 2 && ...
        ismember('weight_change_magnitude', cl.Properties.VariableNames)
    wc     = cl.weight_change_magnitude * 1000;
    cats   = cl.scenario;
    clrs   = {ORANGE; TEAL};
    hold(ax,'on');
    for s=1:2
        b = bar(ax, s, wc(s), 0.5, 'FaceColor', clrs{s}{:}, 'EdgeColor','none', ...
                'DisplayName', cats{s});
        text(ax, s, wc(s)+0.05, sprintf('%.3f\n(x1e-3)',wc(s)), ...
             'HorizontalAlignment','center','Color',TC,'FontSize',8);
    end
    ax.XTick      = [1 2];
    ax.XTickLabel = cats;
    ylim(ax,[0 max(wc)*1.4]);
end
ylabel(ax,'Weight Change (x1 000)','Color',TC,'FontSize',9);
title(ax,'Network Weight Change Magnitude','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northeast','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% ── 4-C  Attitude error: Roll / Pitch / Yaw ───────────────────────────────────
ax = nexttile(t4);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(ep)
    grp  = groupsummary(ep,'step','mean',{'roll_deg','pitch_deg','yaw_deg'});
    % Wrap angles: values near 360 -> negative (error interpretation)
    wrap = @(v) v - 360*(v > 180);
    hold(ax,'on');
    plot(ax, grp.step, wrap(grp.mean_roll_deg),  '-','Color',CYAN,  'LineWidth',1.6,'DisplayName','Roll');
    plot(ax, grp.step, wrap(grp.mean_pitch_deg), '-','Color',AMBER, 'LineWidth',1.6,'DisplayName','Pitch');
    plot(ax, grp.step, wrap(grp.mean_yaw_deg),   '-','Color',GREEN, 'LineWidth',1.6,'DisplayName','Yaw');
    yline(ax,0,'--','Color',[1 1 1 0.3],'LineWidth',0.8,'HandleVisibility','off');
end
xlabel(ax,'Step',          'Color',TC,'FontSize',9);
ylabel(ax,'Angle Error (deg)','Color',TC,'FontSize',9);
title(ax,'Fleet-Mean Attitude Error (Roll / Pitch / Yaw)','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northwest','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% ── 4-D  Orbital slot angular positions ───────────────────────────────────────
ax = nexttile(t4);
style_ax(ax, AX_BG, GR, TC);
if ~isempty(pos) && ismember('sat1_position',pos.Properties.VariableNames) ...
                 && ismember('sat2_position',pos.Properties.VariableNames)
    hold(ax,'on');
    plot(ax, pos.step, pos.sat1_position, '-','Color',CYAN,'LineWidth',2,'DisplayName','Sat-1 Position');
    plot(ax, pos.step, pos.sat2_position, '-','Color',RED, 'LineWidth',2,'DisplayName','Sat-2 Position');
    fill(ax,[pos.step; flipud(pos.step)], ...
         [pos.sat1_position; flipud(pos.sat2_position)], ...
         PURPLE,'FaceAlpha',0.12,'EdgeColor','none','DisplayName','Slot Separation');
    % Ideal separation line
    ideal_sep = mean(pos.sat2_position - pos.sat1_position,'omitnan');
    yline(ax, mean(pos.sat1_position,'omitnan') + ideal_sep, '--', ...
          'Color',[1 1 1 0.35],'LineWidth',0.9,'HandleVisibility','off');
end
xlabel(ax,'Step',              'Color',TC,'FontSize',9);
ylabel(ax,'Angular Position (deg)','Color',TC,'FontSize',9);
title(ax,'Orbital Slot Position History','Color',TC,'FontWeight','bold');
legend(ax,'show','Location','northwest','TextColor',WHITE,'Color',AX_BG,'EdgeColor',GR,'FontSize',8);

% =============================================================================
%  PRINT SUMMARY TABLE IN COMMAND WINDOW
% =============================================================================
fprintf('\n[3/4] Computing KPIs...\n');
fprintf('============================================================\n');
fprintf('  PROJECT KPI SUMMARY\n');
fprintf('============================================================\n');
if ~isempty(ep)
    n_sats      = max(ep.sat_id)+1;
    n_planes    = numel(unique(ep.plane_id));
    fault_total = sum(ep.fault_wheel) + sum(ep.fault_thruster) + sum(ep.fault_sensor);
    safe_pct    = mean(ep.in_safe_mode)*100;
    rec_pct     = mean(ep.in_recovery)*100;
    gs_cov      = mean(ep.gs_los)*100;
    ecl_pct     = mean(ep.eclipse)*100;
    mean_bat    = mean(ep.battery_pct);
    mean_fuel   = mean(ep.fuel_pct);
    mean_temp   = mean(ep.temp_c);
    fprintf('  Satellites           : %d\n',  n_sats);
    fprintf('  Orbital Planes       : %d\n',  n_planes);
    fprintf('  Mean Battery         : %.1f %%\n', mean_bat);
    fprintf('  Mean Fuel            : %.1f %%\n', mean_fuel);
    fprintf('  Mean Temperature     : %.2f C\n',  mean_temp);
    fprintf('  Total Faults         : %d\n',  fault_total);
    fprintf('  %% Time in Safe Mode  : %.2f %%\n', safe_pct);
    fprintf('  %% Time in Recovery   : %.2f %%\n', rec_pct);
    fprintf('  Mean GS Coverage     : %.1f %%\n', gs_cov);
    fprintf('  Mean Eclipse Fraction: %.1f %%\n', ecl_pct);
end
if ~isempty(rew)
    sc = varnames_starting(rew,'sat');
    if ~isempty(sc)
        raw = table2array(rew(:,sc));
        raw(raw < -5) = NaN;
        fprintf('  Fleet Mean Reward    : %.4f\n', mean(raw(:),'omitnan'));
        fm = mean(raw,2,'omitnan');
        fprintf('  Final Step Reward    : %.4f\n', fm(end));
    end
end
if ~isempty(train)
    fprintf('  Training Steps Done  : %d\n', max(train.steps_completed));
    fprintf('  Final Sat-1 Reward   : %.1f\n', train.sat1_reward_mean(end));
    fprintf('  Final Sat-2 Reward   : %.1f\n', train.sat2_reward_mean(end));
end
fprintf('============================================================\n');

fprintf('\n[4/4] All figures ready!\n');
fprintf('  Fig 1 - PPO Training Dashboard\n');
fprintf('  Fig 2 - Satellite Health Dashboard\n');
fprintf('  Fig 3 - 3D Orbit + Ground Map\n');
fprintf('  Fig 4 - Continual Learning + Attitude\n\n');

% ============================================================================
%  HELPER FUNCTIONS
% ============================================================================

function fig = make_fig(name, bg)
    fig = figure('Name', name, 'NumberTitle','off', ...
                 'Color', bg, 'WindowState','maximized');
end

function style_ax(ax, ax_bg, gr_col, tc)
    set(ax, 'Color',ax_bg, ...
        'XColor',tc, 'YColor',tc, 'ZColor',tc, ...
        'GridColor',gr_col, 'MinorGridColor',gr_col, ...
        'GridAlpha',0.5, 'MinorGridAlpha',0.3, ...
        'Box','on', 'FontSize',8.5);
    grid(ax,'on');
end

function style_ax_right(ax, tc)
    set(ax, 'YColor', tc, 'FontSize', 8.5);
end

function tbl = safe_load(fpath, name)
    if isfile(fpath)
        try
            tbl = readtable(fpath, 'VariableNamingRule','preserve');
            % Sanitise column names to valid MATLAB identifiers
            tbl.Properties.VariableNames = matlab.lang.makeValidName( ...
                tbl.Properties.VariableNames);
            fprintf('  OK  %-10s <- %s  (%d rows)\n', name, fpath, height(tbl));
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

function cmap = hot_cold_cmap()
    % Custom green-yellow-red colormap  (good for reward heatmaps)
    N = 256;
    r = [linspace(0,1,N/2), ones(1,N/2)];
    g = [linspace(0,1,N/2), linspace(1,0,N/2)];
    b = [zeros(1,N/2),      zeros(1,N/2)];
    cmap = [r(:), g(:), b(:)];
end

function yline_compat(ax, val, col)
    % Draw a horizontal line (lat or lon reference) compatible with older MATLAB
    yl = ylim(ax);
    plot(ax, [val val], yl, '-','Color',[col 0.35],'LineWidth',0.35,'HandleVisibility','off');
    ylim(ax, yl);
end
