"""
visualize_project.py
====================
Complete Matplotlib Visualization Dashboard for the SpaceTech FSW / RL Project.

Run from the project root:
    python visualize_project.py

All data is loaded from CSV files in:
    matlab_export/   ->  training_progress.csv, positions_history.csv,
                         reward_history.csv, continual_learning_results.csv
    outputs/         ->  episode_data.csv, reward_history.csv,
                         reward_history_new.csv, gs_positions.csv
"""

import os
import sys
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Project root  (auto-detect from this script's location)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def p(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette  (deep-space dark theme)
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK  = "#0B0E1A"
BG_PANEL = "#111829"
BG_AX    = "#0D1020"
GRID_COL = "#1E2540"
TEXT_COL = "#D0D8F8"
SAT_COLS = ["#00D4FF", "#FF6B6B", "#FFD93D", "#6BCB77",
            "#C77DFF", "#FF9F1C", "#2ECC71", "#E74C3C",
            "#3498DB", "#F39C12"]
ACCENT   = "#00FFCC"
ORANGE   = "#FF7043"
PURPLE   = "#AB47BC"
GOLD     = "#FFD700"
RED      = "#FF4444"
GREEN    = "#44FF88"

mpl.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_AX,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "axes.grid":         True,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.5,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "legend.facecolor":  BG_PANEL,
    "legend.edgecolor":  GRID_COL,
    "legend.labelcolor": TEXT_COL,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
})


# ─────────────────────────────────────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return pd.DataFrame()


def load_all():
    data = {}
    data["train"]    = load(p("matlab_export", "training_progress.csv"))
    data["pos"]      = load(p("matlab_export", "positions_history.csv"))
    data["ml_rew"]   = load(p("matlab_export", "reward_history.csv"))
    data["cl"]       = load(p("matlab_export", "continual_learning_results.csv"))
    data["ep"]       = load(p("outputs", "episode_data.csv"))
    data["rew"]      = load(p("outputs", "reward_history.csv"))
    data["rew_new"]  = load(p("outputs", "reward_history_new.csv"))
    data["gs"]       = load(p("outputs", "gs_positions.csv"))
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def rolling(series, w=20):
    return series.rolling(window=w, min_periods=1).mean()


def styled_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG_AX)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8)
    ax.tick_params(colors=TEXT_COL, labelsize=8)


# ─────────────────────────────────────────────────────────────────────────────
#  Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_progress(ax, train):
    """Sat1 & Sat2 cumulative reward across training cycles."""
    if train.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=TEXT_COL)
        styled_ax(ax, "Training Progress")
        return
    steps = train["steps_completed"] / 1000
    ax.plot(steps, train["sat1_reward_mean"], color=SAT_COLS[0],
            lw=2, label="Sat-1 Reward", zorder=3)
    ax.plot(steps, train["sat2_reward_mean"], color=SAT_COLS[1],
            lw=2, label="Sat-2 Reward", zorder=3)
    ax.fill_between(steps, train["sat1_reward_mean"], alpha=0.12, color=SAT_COLS[0])
    ax.fill_between(steps, train["sat2_reward_mean"], alpha=0.12, color=SAT_COLS[1])
    ax.plot(steps, rolling(train["sat1_reward_mean"], 10),
            color=SAT_COLS[0], lw=0.8, linestyle="--", alpha=0.6)
    ax.plot(steps, rolling(train["sat2_reward_mean"], 10),
            color=SAT_COLS[1], lw=0.8, linestyle="--", alpha=0.6)
    styled_ax(ax, "PPO Training - Cumulative Reward per Cycle",
              "Steps (x1 000)", "Mean Episode Reward")
    ax.legend(fontsize=8, loc="lower right")


def plot_value_estimates(ax, train):
    """V-function (critic) estimates for both satellites."""
    if train.empty:
        styled_ax(ax, "Value Estimates"); return
    steps = train["steps_completed"] / 1000
    ax.plot(steps, train["sat1_V_estimate"], color=SAT_COLS[2],
            lw=1.8, label="Sat-1 V(s)")
    ax.plot(steps, train["sat2_V_estimate"], color=SAT_COLS[3],
            lw=1.8, label="Sat-2 V(s)")
    ax.fill_between(steps, train["sat1_V_estimate"], alpha=0.15, color=SAT_COLS[2])
    ax.fill_between(steps, train["sat2_V_estimate"], alpha=0.15, color=SAT_COLS[3])
    styled_ax(ax, "Critic Value Estimates V(s)", "Steps (x1 000)", "V-Estimate")
    ax.legend(fontsize=8, loc="upper left")


def plot_all_satellite_rewards(ax, rew):
    """Per-step per-satellite reward heatmap."""
    if rew.empty:
        styled_ax(ax, "Satellite Reward Heatmap"); return
    sat_cols = [c for c in rew.columns if c.startswith("sat")]
    mat = rew[sat_cols].T.values.astype(float)
    mat = np.clip(mat, -3, 1.5)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-3, vmax=1.5,
                   interpolation="nearest")
    ax.set_yticks(range(len(sat_cols)))
    ax.set_yticklabels([f"Sat-{i}" for i in range(len(sat_cols))],
                       color=TEXT_COL, fontsize=7)
    ax.set_xlabel("Simulation Step", color=TEXT_COL, fontsize=8)
    ax.set_title("Per-Satellite Reward Heatmap (clipped to +-3)", color=TEXT_COL,
                 fontsize=10, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cb.ax.yaxis.set_tick_params(color=TEXT_COL)
    cb.ax.tick_params(labelsize=7, colors=TEXT_COL)


def plot_reward_comparison(ax, rew, rew_new):
    """Compare mean fleet reward: original vs new training run."""
    def fleet_mean(df):
        sat_cols = [c for c in df.columns if c.startswith("sat")]
        vals = df[sat_cols].replace([-9, -8.9], np.nan)
        return vals.mean(axis=1)

    if not rew.empty:
        fm_old = fleet_mean(rew)
        ax.plot(rew["step"], rolling(fm_old, 15), color=SAT_COLS[0],
                lw=2, label="Run 1 (smoothed)")
        ax.fill_between(rew["step"], rolling(fm_old, 15), alpha=0.15,
                        color=SAT_COLS[0])
    if not rew_new.empty:
        fm_new = fleet_mean(rew_new)
        ax.plot(rew_new["step"], rolling(fm_new, 15), color=ACCENT,
                lw=2, label="Run 2 (smoothed)")
        ax.fill_between(rew_new["step"], rolling(fm_new, 15), alpha=0.15,
                        color=ACCENT)
    styled_ax(ax, "Fleet Mean Reward - Training Run Comparison",
              "Step", "Mean Reward")
    ax.legend(fontsize=8)


def plot_satellite_3d_orbits(ax, ep):
    """3D ECEF scatter of satellite positions, coloured by satellite ID."""
    if ep.empty:
        styled_ax(ax, "3D Orbit"); return
    sample = ep.iloc[::10]
    for sid in sorted(sample["sat_id"].unique()):
        sub = sample[sample["sat_id"] == sid]
        ax.scatter(sub["ecef_x_km"], sub["ecef_y_km"], sub["ecef_z_km"],
                   s=1.5, color=SAT_COLS[sid % len(SAT_COLS)],
                   alpha=0.6, label=f"Sat-{sid}")
    u = np.linspace(0, 2 * math.pi, 60)
    v = np.linspace(0, math.pi, 30)
    R = 6371.0
    xs = R * np.outer(np.cos(u), np.sin(v))
    ys = R * np.outer(np.sin(u), np.sin(v))
    zs = R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.1, color="#3A7BD5", linewidth=0)
    ax.set_xlabel("X (km)", color=TEXT_COL, fontsize=7, labelpad=2)
    ax.set_ylabel("Y (km)", color=TEXT_COL, fontsize=7, labelpad=2)
    ax.set_zlabel("Z (km)", color=TEXT_COL, fontsize=7, labelpad=2)
    ax.set_title("3D ECEF Orbital Positions", color=TEXT_COL,
                 fontsize=10, fontweight="bold")
    ax.tick_params(colors=TEXT_COL, labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COL)
    ax.yaxis.pane.set_edgecolor(GRID_COL)
    ax.zaxis.pane.set_edgecolor(GRID_COL)
    ax.set_facecolor(BG_AX)


def plot_altitude_battery(ax, ep):
    """Altitude and battery level for each satellite over time."""
    if ep.empty:
        styled_ax(ax, "Altitude & Battery"); return
    grp = ep.groupby("step")[["altitude_km", "battery_pct", "fuel_pct"]].mean()
    ax2 = ax.twinx()
    ax.plot(grp.index, grp["altitude_km"], color=SAT_COLS[0],
            lw=1.5, label="Altitude (km)")
    ax2.plot(grp.index, grp["battery_pct"], color=GOLD,
             lw=1.5, linestyle="--", label="Battery %")
    ax2.plot(grp.index, grp["fuel_pct"], color=GREEN,
             lw=1.5, linestyle=":", label="Fuel %")
    styled_ax(ax, "Fleet-Mean: Altitude & Subsystem State",
              "Step", "Altitude (km)")
    ax2.set_ylabel("% Level", color=TEXT_COL, fontsize=8)
    ax2.tick_params(colors=TEXT_COL, labelsize=8)
    ax2.set_facecolor(BG_AX)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right")


def plot_fault_events(ax, ep):
    """Stacked bar showing fault counts per simulation step bin."""
    if ep.empty:
        styled_ax(ax, "Fault Events"); return
    ep2 = ep.copy()
    ep2["bin"] = (ep2["step"] // 50) * 50
    grp = ep2.groupby("bin")[["fault_wheel", "fault_thruster", "fault_sensor"]].sum()
    x = grp.index
    ax.bar(x, grp["fault_wheel"],    color="#FF4444", width=45, label="Wheel fault")
    ax.bar(x, grp["fault_thruster"], color="#FF9900", width=45,
           bottom=grp["fault_wheel"], label="Thruster fault")
    ax.bar(x, grp["fault_sensor"],   color="#FFFF44", width=45,
           bottom=grp["fault_wheel"] + grp["fault_thruster"], label="Sensor fault")
    styled_ax(ax, "Fault Events by Step Bin (50-step window)",
              "Step", "Fault Count")
    ax.legend(fontsize=7)


def plot_safe_mode_recovery(ax, ep):
    """Fraction of satellites in safe-mode or recovery per step."""
    if ep.empty:
        styled_ax(ax, "Safe Mode / Recovery"); return
    grp = ep.groupby("step")[["in_safe_mode", "in_recovery"]].mean() * 100
    ax.fill_between(grp.index, grp["in_safe_mode"], alpha=0.4,
                    color=RED, label="Safe Mode %")
    ax.fill_between(grp.index, grp["in_recovery"], alpha=0.4,
                    color=ORANGE, label="Recovery Mode %")
    ax.plot(grp.index, grp["in_safe_mode"],  color=RED,    lw=1.2)
    ax.plot(grp.index, grp["in_recovery"],   color=ORANGE, lw=1.2)
    styled_ax(ax, "Fleet % in Safe-Mode / Recovery", "Step", "% of Satellites")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7)


def plot_attitude(ax, ep):
    """Roll / pitch / yaw deviation across fleet."""
    if ep.empty:
        styled_ax(ax, "Attitude"); return
    grp = ep.groupby("step")[["roll_deg", "pitch_deg", "yaw_deg"]].mean()
    for col in ["roll_deg", "pitch_deg", "yaw_deg"]:
        grp[col] = grp[col].apply(lambda x: x - 360 if x > 180 else x)
    ax.plot(grp.index, grp["roll_deg"],  color=SAT_COLS[0], lw=1.3, label="Roll")
    ax.plot(grp.index, grp["pitch_deg"], color=SAT_COLS[2], lw=1.3, label="Pitch")
    ax.plot(grp.index, grp["yaw_deg"],   color=SAT_COLS[3], lw=1.3, label="Yaw")
    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    styled_ax(ax, "Fleet-Mean Attitude Error (deg)", "Step", "Angle Error (deg)")
    ax.legend(fontsize=7)


def plot_temperature(ax, ep):
    """Thermal dynamics: satellite temperature over steps."""
    if ep.empty:
        styled_ax(ax, "Temperature"); return
    grp = ep.groupby("step")["temp_c"].agg(["mean", "min", "max"])
    ax.fill_between(grp.index, grp["min"], grp["max"],
                    alpha=0.25, color=ORANGE, label="Min-Max range")
    ax.plot(grp.index, grp["mean"], color=ORANGE, lw=2, label="Mean Temp (C)")
    ax.axhline(35, color=RED,    lw=1, linestyle="--", alpha=0.7, label="Max safe")
    ax.axhline(15, color=ACCENT, lw=1, linestyle="--", alpha=0.7, label="Min safe")
    styled_ax(ax, "Thermal Dynamics - Satellite Temperature",
              "Step", "Temperature (C)")
    ax.legend(fontsize=7)


def plot_eclipse_gs(ax, ep):
    """Eclipse fraction and ground-station LOS coverage per step."""
    if ep.empty:
        styled_ax(ax, "Eclipse / GS LOS"); return
    grp = ep.groupby("step")[["eclipse", "gs_los"]].mean() * 100
    ax.fill_between(grp.index, grp["eclipse"], alpha=0.35,
                    color="#4444AA", label="Eclipse %")
    ax.fill_between(grp.index, grp["gs_los"], alpha=0.35,
                    color=GREEN, label="GS LOS %")
    ax.plot(grp.index, grp["eclipse"], color="#8888FF", lw=1.2)
    ax.plot(grp.index, grp["gs_los"],  color=GREEN,    lw=1.2)
    styled_ax(ax, "Eclipse & Ground-Station LOS Coverage (%)",
              "Step", "% of Satellites")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7)


def plot_ground_stations(ax, gs, ep):
    """2D lat-lon map with ground stations and ground track."""
    if ep.empty and gs.empty:
        styled_ax(ax, "Ground Station Map"); return
    if not ep.empty:
        sample = ep[ep["sat_id"] == 0].iloc[::3]
        r = np.sqrt(sample["ecef_x_km"]**2 + sample["ecef_y_km"]**2
                    + sample["ecef_z_km"]**2)
        lat = np.degrees(np.arcsin(sample["ecef_z_km"] / r))
        lon = np.degrees(np.arctan2(sample["ecef_y_km"], sample["ecef_x_km"]))
        ax.scatter(lon, lat, s=1, color=SAT_COLS[0], alpha=0.5,
                   label="Sat-0 Ground Track", zorder=2)
    if not gs.empty:
        ax.scatter(gs["lon_deg"], gs["lat_deg"], s=120, marker="^",
                   color=RED, zorder=5, label="Ground Station")
        for _, row in gs.iterrows():
            ax.annotate(row["name"], (row["lon_deg"], row["lat_deg"]),
                        textcoords="offset points", xytext=(5, 5),
                        color=TEXT_COL, fontsize=6.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(0, color=GRID_COL, lw=0.8)
    ax.axvline(0, color=GRID_COL, lw=0.8)
    styled_ax(ax, "Ground Track & Ground-Station Map",
              "Longitude (deg)", "Latitude (deg)")
    ax.legend(fontsize=7, loc="lower left")


def plot_continual_learning(ax, cl):
    """EWC vs no-EWC continual learning bar chart."""
    if cl.empty or len(cl) < 2:
        ax.text(0.5, 0.5, "No EWC data", ha="center", va="center", color=TEXT_COL)
        styled_ax(ax, "Continual Learning"); return

    labels  = ["Baseline\nT1", "After T2\n(T1 score)", "Task-2\nReward",
                "Forgetting", "Weight\nChange x1k"]
    no_ewc  = [cl.iloc[0]["baseline_t1"], cl.iloc[0]["after_task2_t1"],
                cl.iloc[0]["task2_reward"],  cl.iloc[0]["forgetting"],
                cl.iloc[0]["weight_change_magnitude"] * 1000]
    ewc     = [cl.iloc[1]["baseline_t1"], cl.iloc[1]["after_task2_t1"],
                cl.iloc[1]["task2_reward"],  cl.iloc[1]["forgetting"],
                cl.iloc[1]["weight_change_magnitude"] * 1000]

    x = np.arange(len(labels))
    w = 0.36
    bars1 = ax.bar(x - w/2, no_ewc, w, color=ORANGE, alpha=0.85, label="No-EWC",
                    edgecolor=BG_DARK, linewidth=0.7)
    bars2 = ax.bar(x + w/2, ewc,    w, color=ACCENT, alpha=0.85, label="EWC",
                    edgecolor=BG_DARK, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_COL, fontsize=7.5)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", color=TEXT_COL, fontsize=6.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", color=TEXT_COL, fontsize=6.5)
    styled_ax(ax, "Continual Learning - EWC vs No-EWC", "", "Score / Metric")
    ax.legend(fontsize=8)


def plot_positions(ax, pos):
    """Satellite angular positions vs step."""
    if pos.empty:
        styled_ax(ax, "Position History"); return
    if "sat1_position" in pos.columns and "sat2_position" in pos.columns:
        ax.plot(pos["step"], pos["sat1_position"], color=SAT_COLS[0],
                lw=1.5, label="Sat-1 Position (deg)")
        ax.plot(pos["step"], pos["sat2_position"], color=SAT_COLS[1],
                lw=1.5, label="Sat-2 Position (deg)")
        ax.fill_between(pos["step"], pos["sat1_position"], pos["sat2_position"],
                        alpha=0.12, color=ACCENT, label="Separation")
    styled_ax(ax, "Orbital Slot Position History",
              "Step", "Angular Position (deg)")
    ax.legend(fontsize=7)


def plot_summary_kpis(ax, ep, rew):
    """Dashboard KPI summary panel."""
    ax.set_facecolor(BG_PANEL)
    ax.axis("off")

    kpis = []

    if not ep.empty:
        fault_total = (ep["fault_wheel"] + ep["fault_thruster"] + ep["fault_sensor"]).sum()
        safe_frac   = ep["in_safe_mode"].mean() * 100
        rec_frac    = ep["in_recovery"].mean() * 100
        gs_cov      = ep["gs_los"].mean() * 100
        ecl_frac    = ep["eclipse"].mean() * 100
        mean_bat    = ep["battery_pct"].mean()
        mean_fuel   = ep["fuel_pct"].mean()
        mean_temp   = ep["temp_c"].mean()
        n_sats      = ep["sat_id"].nunique()
        n_planes    = ep["plane_id"].nunique() if "plane_id" in ep.columns else "N/A"

        kpis += [
            ("Satellites",           f"{n_sats}"),
            ("Orbital Planes",       f"{n_planes}"),
            ("Mean Battery",         f"{mean_bat:.1f} %"),
            ("Mean Fuel",            f"{mean_fuel:.1f} %"),
            ("Mean Temperature",     f"{mean_temp:.2f} C"),
            ("Total Faults",         f"{int(fault_total)}"),
            ("% Time Safe-Mode",     f"{safe_frac:.2f} %"),
            ("% Time Recovery",      f"{rec_frac:.2f} %"),
            ("Mean GS Coverage",     f"{gs_cov:.1f} %"),
            ("Mean Eclipse Frac.",   f"{ecl_frac:.1f} %"),
        ]

    if not rew.empty:
        sat_cols = [c for c in rew.columns if c.startswith("sat")]
        vals = rew[sat_cols].replace([-9, -8.9], np.nan)
        fleet_mean_val = vals.mean(axis=1).mean()
        fleet_last = vals.mean(axis=1).iloc[-1]
        kpis += [
            ("Fleet Mean Reward",    f"{fleet_mean_val:.4f}"),
            ("Final Step Reward",    f"{fleet_last:.4f}"),
        ]

    y_start = 0.95
    ax.text(0.5, y_start + 0.03, "Project KPI Summary",
            ha="center", va="top", color=GOLD,
            fontsize=12, fontweight="bold",
            transform=ax.transAxes)

    step = 0.075
    for i, (label, value) in enumerate(kpis):
        y = y_start - (i + 1) * step
        ax.text(0.04, y, label, ha="left", va="top",
                color=TEXT_COL, fontsize=9, transform=ax.transAxes)
        ax.text(0.96, y, value, ha="right", va="top",
                color=ACCENT, fontsize=9, fontweight="bold",
                transform=ax.transAxes)

    ax.text(0.5, 0.01,
            "SpaceTech FSW + PPO RL Constellation Simulator",
            ha="center", va="bottom", color=PURPLE,
            fontsize=7, style="italic", transform=ax.transAxes)


# ─────────────────────────────────────────────────────────────────────────────
#  Main dashboard layout
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(data):
    fig = plt.figure(figsize=(26, 22), facecolor=BG_DARK)
    fig.suptitle(
        "SpaceTech  |  Multi-Satellite Autonomous FSW + PPO RL Dashboard",
        color=TEXT_COL, fontsize=17, fontweight="bold", y=0.985,
        fontfamily="DejaVu Sans"
    )

    gs_master = gridspec.GridSpec(
        5, 4,
        figure=fig,
        hspace=0.52, wspace=0.36,
        left=0.05, right=0.97,
        top=0.97, bottom=0.03
    )

    # Row 0: Training progress (wide) + KPI panel
    ax_train = fig.add_subplot(gs_master[0, :3])
    ax_kpi   = fig.add_subplot(gs_master[0, 3])

    # Row 1: Value estimates + Reward heatmap
    ax_val   = fig.add_subplot(gs_master[1, :2])
    ax_heat  = fig.add_subplot(gs_master[1, 2:])

    # Row 2: 3D orbit + Continual learning + Position history
    ax_3d    = fig.add_subplot(gs_master[2, :2], projection="3d")
    ax_cl    = fig.add_subplot(gs_master[2, 2])
    ax_pos   = fig.add_subplot(gs_master[2, 3])

    # Row 3: Reward comparison + Altitude/Battery + Eclipse/GS + Attitude
    ax_cmp   = fig.add_subplot(gs_master[3, 0])
    ax_alt   = fig.add_subplot(gs_master[3, 1])
    ax_ecl   = fig.add_subplot(gs_master[3, 2])
    ax_att   = fig.add_subplot(gs_master[3, 3])

    # Row 4: Temperature + Fault events + Safe-mode + Ground station map
    ax_temp  = fig.add_subplot(gs_master[4, 0])
    ax_fault = fig.add_subplot(gs_master[4, 1])
    ax_safe  = fig.add_subplot(gs_master[4, 2])
    ax_gs    = fig.add_subplot(gs_master[4, 3])

    # Populate all subplots
    plot_training_progress   (ax_train, data["train"])
    plot_summary_kpis        (ax_kpi,   data["ep"],   data["rew"])
    plot_value_estimates     (ax_val,   data["train"])
    plot_all_satellite_rewards(ax_heat, data["rew"])
    plot_satellite_3d_orbits (ax_3d,    data["ep"])
    plot_continual_learning  (ax_cl,    data["cl"])
    plot_positions           (ax_pos,   data["pos"])
    plot_reward_comparison   (ax_cmp,   data["rew"],  data["rew_new"])
    plot_altitude_battery    (ax_alt,   data["ep"])
    plot_eclipse_gs          (ax_ecl,   data["ep"])
    plot_attitude            (ax_att,   data["ep"])
    plot_temperature         (ax_temp,  data["ep"])
    plot_fault_events        (ax_fault, data["ep"])
    plot_safe_mode_recovery  (ax_safe,  data["ep"])
    plot_ground_stations     (ax_gs,    data["gs"],   data["ep"])

    fig.text(0.02, 0.005,
             "Generated by visualize_project.py  |  SpaceTech FSW RL Suite",
             color=GRID_COL, fontsize=7)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  SpaceTech Matplotlib Visualization Dashboard")
    print("=" * 62)
    print(f"  Project root : {PROJECT_ROOT}")

    data = load_all()
    for name, df in data.items():
        if not df.empty:
            print(f"  OK Loaded [{name}]  -> {len(df)} rows, {len(df.columns)} cols")
        else:
            print(f"  MISSING [{name}]")

    print("\n  Building dashboard ...")
    fig = build_dashboard(data)

    out_path = p("outputs", "project_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    print(f"  Saved -> {out_path}")

    plt.tight_layout(rect=[0, 0.01, 1, 0.985])
    plt.show()
    print("\n  Done. Close the window to exit.")


if __name__ == "__main__":
    main()
