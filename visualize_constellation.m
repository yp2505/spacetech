% ============================================================================
%  visualize_constellation.m  —  SpaceTech AI Constellation Visualizer v3
% ============================================================================
%  PERFORMANCE: Batch scatter (all sats in one call) + frame skipping
%  FIXES      : WebGL crash | Legend overflow | Lag
%  SHOWS      : Earth | Debris | 10 Satellites | Trails | Reward curve
%
%  Command Window: visualize_constellation
% ============================================================================

clear; clc; close all;

%% ── Settings ─────────────────────────────────────────────────────────────────
EARTH_R    = 6371.0;
FRAME_SKIP = 1;       % Animate exactly all 360 steps (1 by 1)
ANIM_PAUSE = 0.01;    % Fast framerate

%% ── Load CSVs (handles (2),(3) suffix automatically) ─────────────────────────
fprintf('Loading data...\n');
ep = load_csv('episode_data.csv');
rh = load_csv('reward_history.csv');
gs = load_csv('gs_positions.csv');

n_sats  = max(ep.sat_id) + 1;
n_steps = max(ep.step);
n_cyc   = height(rh);
fprintf('  Sats: %d  Steps: %d  Cycles: %d\n', n_sats, n_steps, n_cyc);

%% ── Colors ───────────────────────────────────────────────────────────────────
CLRS = [0.10 0.95 0.35;   % nominal   green
        1.00 0.85 0.00;   % degraded  amber
        1.00 0.20 0.15;   % critical  red
        0.90 0.10 0.90;   % safe mode magenta
        0.30 0.70 1.00];  % eclipse   blue

%% ── Figure (2 panels only for performance) ───────────────────────────────────
fig = figure('Name','SpaceTech AI Constellation', ...
             'Color',[0.04 0.04 0.10],'NumberTitle','off','WindowState','maximized');
set(fig,'Units','normalized','OuterPosition',[0.02 0.02 0.96 0.96]);

% Left: 3D orbital view
ax3d = subplot(1,2,1);
set(ax3d,'Color',[0.04 0.04 0.10],'GridColor',[0.18 0.18 0.30], ...
    'XColor',[0.5 0.5 0.7],'YColor',[0.5 0.5 0.7],'ZColor',[0.5 0.5 0.7]);
hold(ax3d,'on'); axis(ax3d,'equal'); grid(ax3d,'on');
view(ax3d,30,22);
xlabel(ax3d,'X (km)','Color',[0.7 0.7 0.9]);
ylabel(ax3d,'Y (km)','Color',[0.7 0.7 0.9]);
zlabel(ax3d,'Z (km)','Color',[0.7 0.7 0.9]);

% Right: AI reward curve
ax_rwd = subplot(1,2,2);
set(ax_rwd,'Color',[0.06 0.06 0.14],'GridColor',[0.18 0.18 0.30], ...
    'XColor',[0.7 0.7 0.9],'YColor',[0.7 0.7 0.9]);
hold(ax_rwd,'on'); grid(ax_rwd,'on');
title(ax_rwd,'AI Training — Reward per Cycle', ...
    'Color','white','FontSize',11,'FontWeight','bold');
xlabel(ax_rwd,'Cycle','Color',[0.7 0.7 0.9]);
ylabel(ax_rwd,'Reward','Color',[0.7 0.7 0.9]);

%% ── Earth (sphere 24 — minimal vertices for WebGL) ───────────────────────────
[xs,ys,zs] = sphere(24);
surf(ax3d, xs*EARTH_R, ys*EARTH_R, zs*EARTH_R, ...
    'FaceColor',[0.07 0.18 0.42],'EdgeColor',[0.10 0.22 0.52], ...
    'EdgeAlpha',0.18,'FaceAlpha',0.96,'FaceLighting','none', ...
    'HandleVisibility','off');

% Simple equator ring only (1 line, not a full grid)
th = linspace(0,2*pi,80);
plot3(ax3d, EARTH_R*cos(th), EARTH_R*sin(th), zeros(size(th)), ...
      '-','Color',[0.2 0.5 0.9 0.4],'LineWidth',1,'HandleVisibility','off');

%% ── Debris Field (60 dots — minimal) ─────────────────────────────────────────
rng(42);
N  = 60;
rd = EARTH_R + 550 + randn(N,1)*60;
az = rand(N,1)*2*pi;
el = (rand(N,1)-0.5)*pi*0.3;
scatter3(ax3d, rd.*cos(el).*cos(az), rd.*cos(el).*sin(az), rd.*sin(el), ...
         8,'MarkerFaceColor',[1 0.25 0.08],'MarkerEdgeColor','none', ...
         'MarkerFaceAlpha',0.65,'DisplayName','⚠ Debris');

%% ── Ground Stations (triangles, no animated LOS for performance) ─────────────
for g = 1:height(gs)
    gx=gs.ecef_x_km(g); gy=gs.ecef_y_km(g); gz=gs.ecef_z_km(g);
    scatter3(ax3d,gx*1.02,gy*1.02,gz*1.02,80,'v','filled', ...
             'MarkerFaceColor',[1 0.85 0],'MarkerEdgeColor','w', ...
             'HandleVisibility','off');
    text(ax3d,gx*1.06,gy*1.06,gz*1.06,gs.name{g}, ...
         'Color',[1 0.9 0.5],'FontSize',7,'FontWeight','bold');
end

%% ── AI Reward Curve (static — no update in loop) ─────────────────────────────
rc = rh.Properties.VariableNames;
sc = rc(contains(rc,'sat'));
if isempty(sc), mr=rh.(rc{2}); else, mr=mean(table2array(rh(:,sc)),2); end
cv=(1:n_cyc)';
fill(ax_rwd,[cv;flipud(cv)],[mr;zeros(n_cyc,1)],[0.1 0.5 1], ...
     'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
plot(ax_rwd,cv,mr,'-','Color',[0.3 0.8 1.0],'LineWidth',2,'DisplayName','Mean reward');
yline(ax_rwd,0,'--','Color',[0.55 0.55 0.55],'LineWidth',0.8,'HandleVisibility','off');
yline(ax_rwd,300,'-','Color',[0 1 0.4],'LineWidth',1,'Label','Target 300', ...
      'HandleVisibility','off');
ph_s=[1 11 31 61 91 131]; ph_l={'Ph1','Ph2','Ph3','Ph4','Ph5','Ph6'};
for p=1:6
    xp=ph_s(p)*n_cyc/200;
    xline(ax_rwd,xp,'--','Color',[1 0.5 0 0.5],'LineWidth',0.8,'HandleVisibility','off');
    text(ax_rwd,xp+1,max(mr)*0.88,ph_l{p},'Color',[1 0.7 0.3],'FontSize',8);
end
ax_rwd.XLim=[1 n_cyc];
rwd_step = plot(ax_rwd, 1, mr(1), 'o','Color',[1 1 0],'MarkerSize',9, ...
                'MarkerFaceColor',[1 1 0],'DisplayName','Current cycle');
legend(ax_rwd,'show','Location','northwest','TextColor','white', ...
       'Color',[0.08 0.08 0.18],'EdgeColor',[0.3 0.3 0.5]);

%% ── BATCH Satellite Scatter (ONE scatter object, all 10 sats) ─────────────────
% Initial positions at step 1
s1 = ep(ep.step==1,:);
s1 = sortrows(s1,'sat_id');
sat_colors = arrayfun(@(i) hclr(s1(i+1,:),CLRS), 0:n_sats-1, ...
                      'UniformOutput',false);
C0 = cell2mat(sat_colors');   % n_sats × 3

% Single scatter3 for ALL satellites — much faster than 10 separate ones
sat_sc = scatter3(ax3d, s1.ecef_x_km, s1.ecef_y_km, s1.ecef_z_km, ...
                  140, C0, 'filled','MarkerEdgeColor','white','LineWidth',0.7, ...
                  'DisplayName','Satellites');

% Legend (only named objects)
scatter3(ax3d,NaN,NaN,NaN,50,'filled','MarkerFaceColor',[0.1 0.95 0.35], ...
         'DisplayName','Nominal');
scatter3(ax3d,NaN,NaN,NaN,50,'filled','MarkerFaceColor',[1 0.2 0.15], ...
         'DisplayName','Fault/Critical');
scatter3(ax3d,NaN,NaN,NaN,50,'filled','MarkerFaceColor',[0.9 0.1 0.9], ...
         'DisplayName','Safe Mode');
legend(ax3d,'show','Location','northwest','TextColor','white', ...
       'Color',[0.08 0.08 0.18],'EdgeColor',[0.3 0.3 0.5],'FontSize',8);

%% ── Animation (frame-skipped for performance) ────────────────────────────────
fprintf('\nAnimating %d steps (every %d frames)...\n',n_steps,FRAME_SKIP);

for s = 1:FRAME_SKIP:n_steps
    if ~isvalid(fig), break; end
    sd = ep(ep.step==s,:);
    sd = sortrows(sd,'sat_id');

    % --- Batch update ALL satellites at once (1 call instead of 10)
    sat_sc.XData = sd.ecef_x_km;
    sat_sc.YData = sd.ecef_y_km;
    sat_sc.ZData = sd.ecef_z_km;

    % Color & size per satellite
    C = zeros(n_sats,3);
    SZ = ones(n_sats,1)*120;
    for i=0:n_sats-1
        row=sd(sd.sat_id==i,:);
        if isempty(row), continue; end
        C(i+1,:)=hclr(row,CLRS);
        if row.fault_wheel||row.fault_thruster||row.fault_sensor
            SZ(i+1)=280;
        elseif row.in_safe_mode
            SZ(i+1)=200;
        end
    end
    sat_sc.CData    = C;
    sat_sc.SizeData = SZ;

    % Reward marker on right panel
    cyc_frac = round(s/n_steps * n_cyc);
    cyc_frac = max(1,min(cyc_frac,n_cyc));
    rwd_step.XData = cyc_frac;
    rwd_step.YData = mr(cyc_frac);

    % Title
    n_flt  = sum(sd.fault_wheel)+sum(sd.fault_thruster)+sum(sd.fault_sensor);
    n_safe = sum(sd.in_safe_mode);
    n_los  = sum(sd.gs_los);
    n_ecl  = sum(sd.eclipse);
    title(ax3d,sprintf( ...
        'Walker Delta — Step %d/%d  |  Faults:%d  Safe:%d  LOS:%d  Eclipse:%d', ...
        s,n_steps,n_flt,n_safe,n_los,n_ecl), ...
        'Color','white','FontSize',9,'FontWeight','bold');

    view(ax3d, 30+s*0.3, 20+3*sin(s/50));
    drawnow limitrate;
    pause(ANIM_PAUSE);
end
fprintf('Animation complete!\n');

%% ── Helpers ──────────────────────────────────────────────────────────────────
function tbl = load_csv(name)
    if isfile(name), tbl=readtable(name); return; end
    d=dir([name(1:end-4) '*.csv']);
    if isempty(d), error('Cannot find: %s',name); end
    [~,idx]=sort([d.datenum],'descend');
    found=fullfile(d(idx(1)).folder,d(idx(1)).name);
    fprintf('  Using: %s\n',d(idx(1)).name);
    movefile(found,name);
    tbl=readtable(name);
end

function c = hclr(row, CLRS)
    if row.in_safe_mode,                                          c=CLRS(4,:);
    elseif row.fault_wheel||row.fault_thruster||row.fault_sensor, c=CLRS(3,:);
    elseif row.battery_pct<30||row.fuel_pct<15,                   c=CLRS(2,:);
    elseif row.eclipse,                                           c=CLRS(5,:);
    else,                                                         c=CLRS(1,:);
    end
end
