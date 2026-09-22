"""
experiments/osg_comparison.py
==============================
Orbital Salience Gating (OSG) — Standalone Evaluation Experiment.

Runs TWO identical 10-episode evaluation sessions (same random seeds)
using the already-trained ppo_swarm_brain.zip:

  SESSION A  — BASELINE  : cosine-only memory retrieval
  SESSION B  — OSG        : physics-derived Kepler β retrieval

Records per-session metrics and saves a comparison CSV to:
  matlab_export/osg_comparison_results.csv

Usage (from the project root):
    python -m experiments.osg_comparison
    # or
    python experiments/osg_comparison.py
"""

from __future__ import annotations

import os
import sys
import csv
import pickle
import warnings
import numpy as np

# ── path setup (run from project root OR experiments/) ───────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── project imports ──────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from simulation.sat_config     import PRESETS
from simulation.satellite_env  import MultiSatelliteEnv
from rl_training.memory        import EpisodicMemory, EpisodeRecord
from utils.orbital_decay       import compute_beta

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join(_ROOT, "ppo_swarm_brain.zip")
MEM_PATTERN      = os.path.join(_ROOT, "episodic_memory_sat{i}.pkl")
NUM_SATS         = 10
NUM_EPISODES     = 10
EP_STEPS         = 500
TOP_K            = 5
CURRICULUM_PHASE = 6          # full phase (faults + thermal)
SEEDS            = list(range(42, 42 + NUM_EPISODES))   # fixed seeds for reproducibility
CSV_OUT          = os.path.join(_ROOT, "matlab_export", "osg_comparison_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(config):
    """Load the trained PPO model, creating a dummy env for observation space."""
    from simulation.satellite_env import SingleAgentWrapper
    dummy_env = SingleAgentWrapper(config=config, max_steps=EP_STEPS)
    print(f"  Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=dummy_env)
    print("  Model loaded OK.")
    return model

# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Load episodic memories
# ─────────────────────────────────────────────────────────────────────────────
def load_all_memories():
    """
    Load episodic_memory_sat0.pkl through sat9.pkl.

    Episodes with empty orbital_state ({}) are noted but kept — osg_score()
    returns 0.0 for them, which is the correct graceful-skip behaviour.
    """
    all_episodes: list[EpisodeRecord] = []
    total_skipped_orbital = 0

    for i in range(NUM_SATS):
        path = MEM_PATTERN.format(i=i)
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found — skipping sat {i}")
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            eps = data.get("episodes", [])
            for ep in eps:
                # Backward-compat: patch missing fields
                if not hasattr(ep, "orbital_state"):
                    ep.orbital_state = {}
                if not hasattr(ep, "timestamp_step"):
                    ep.timestamp_step = 0
            no_orbital = sum(1 for ep in eps if not ep.orbital_state)
            if no_orbital:
                total_skipped_orbital += no_orbital
            all_episodes.extend(eps)
            print(f"  sat{i}: {len(eps)} episodes loaded "
                  f"({no_orbital} have empty orbital_state — will score 0.0 in OSG)")
        except Exception as exc:
            print(f"  [WARN] Could not load {path}: {exc}")

    print(f"\n  Total episodes loaded: {len(all_episodes)}")
    if total_skipped_orbital:
        print(f"  Episodes with empty orbital_state: {total_skipped_orbital} "
              f"(gracefully handled — osg_score returns 0.0 for these)")
    return all_episodes

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: Compute physics-derived β
# ─────────────────────────────────────────────────────────────────────────────
def get_kepler_beta(config):
    """Derive β from Kepler's Third Law for this config's orbital altitude."""
    altitude_km   = config.orbit.altitude_km
    step_seconds  = config.orbit.step_seconds
    sps           = 1.0 / step_seconds          # steps per real second
    beta, T_sec, T_steps = compute_beta(altitude_km, steps_per_second=sps)
    print(f"\n  Orbital altitude : {altitude_km:.1f} km")
    print(f"  Orbital period   : {T_sec:.1f} s  ({T_sec/60:.1f} min)")
    print(f"  Period in steps  : {T_steps:.1f} steps")
    print(f"  Kepler β         : {beta:.8f} per step")
    return beta, altitude_km, T_sec

# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 + 5: Run evaluation sessions
# ─────────────────────────────────────────────────────────────────────────────
def _build_pool_memory(all_episodes: list[EpisodeRecord]) -> EpisodicMemory:
    """
    Create an in-memory EpisodicMemory backed by all loaded episodes.
    No file I/O — we inject episodes directly.
    """
    mem = EpisodicMemory.__new__(EpisodicMemory)
    mem.capacity     = len(all_episodes) + 1
    mem.filepath     = "__osg_experiment_no_file__"
    mem.episodes     = list(all_episodes)
    mem.total_seen   = len(all_episodes)
    if all_episodes:
        rewards          = [e.total_reward for e in all_episodes]
        mem.best_reward  = float(max(rewards))
        mem.worst_reward = float(min(rewards))
    else:
        mem.best_reward  = -1e9
        mem.worst_reward =  1e9
    return mem


def _episode_orbital_state(env: MultiSatelliteEnv, sat_idx: int = 0) -> dict:
    """
    Build a minimal orbital state dict from live env state.
    Used as the query vector when calling osg_retrieve.
    """
    return {
        "altitude_km":       env.config.orbit.altitude_km,
        "true_anomaly":      float(env.agent_pos[sat_idx]),
        "eclipse_fraction":  float(env.eclipse_fraction[sat_idx]),
    }


def run_session(
    model: PPO,
    config,
    all_episodes: list[EpisodeRecord],
    use_osg: bool,
    beta: float,
    label: str,
) -> dict:
    """
    Run NUM_EPISODES evaluation episodes under either OSG or baseline retrieval.

    Args:
        model:        Trained PPO.
        config:       SatelliteConfig.
        all_episodes: All loaded EpisodeRecords (used as retrieval pool).
        use_osg:      True = OSG retrieval, False = cosine-only baseline.
        beta:         Kepler-derived decay rate.
        label:        Human-readable session label for printing.

    Returns:
        Dict of aggregated metrics across the session.
    """
    mem = _build_pool_memory(all_episodes)

    ep_rewards         = []
    ep_targets_hit     = []    # proxy: avoidance maneuvers (positive reward events)
    ep_wasted_actions  = []    # proxy: safe-mode steps (agent unable to act usefully)
    ep_fuel_remaining  = []
    ep_memory_helped   = []    # 1 if best retrieved episode had reward > R_mean, else 0

    R_mean_pool = (float(np.mean([e.total_reward for e in all_episodes]))
                   if all_episodes else 0.0)

    print(f"\n  {'─'*60}")
    print(f"  SESSION {label}: {'OSG' if use_osg else 'BASELINE'} retrieval "
          f"({NUM_EPISODES} episodes × {EP_STEPS} steps)")
    print(f"  {'─'*60}")

    for ep_idx, seed in enumerate(SEEDS):
        env = MultiSatelliteEnv(config=config, max_steps=EP_STEPS)
        env.curriculum_phase = CURRICULUM_PHASE
        obs_raw, _ = env.reset(seed=seed)

        ctx = mem.get_context()

        # Track whether memory was helpful at the start of the episode
        ep_memory_helped_flags = []

        global_step = ep_idx * EP_STEPS  # synthetic global timestep for age calc

        for step_idx in range(EP_STEPS):
            # ── Memory retrieval (once per step for sat 0 as representative) ──
            current_state   = _episode_orbital_state(env, sat_idx=0)
            current_t       = global_step + step_idx

            if use_osg:
                osg_res, _ = mem.osg_retrieve(
                    current_state, current_t, top_k=TOP_K, beta=beta
                )
                retrieved = osg_res
            else:
                # Baseline: cosine-only (second return value from osg_retrieve)
                _, base_res = mem.osg_retrieve(
                    current_state, current_t, top_k=TOP_K, beta=beta
                )
                retrieved = base_res

            # memory_helped: top retrieved episode has reward > R_mean_pool
            if retrieved:
                top_score, top_ep = retrieved[0]
                helped = 1 if top_ep.total_reward > R_mean_pool else 0
                ep_memory_helped_flags.append(helped)

            # ── Model inference ──────────────────────────────────────────────
            actions = []
            for i in range(config.num_satellites):
                local = np.concatenate([obs_raw[i]["local"], ctx]).astype(np.float32)
                obs_d = {"local": local, "global": obs_raw[i]["global"]}
                a, _  = model.predict(obs_d, deterministic=True)
                actions.append(a)

            obs_raw, _, term, trunc, _ = env.step(np.array(actions))
            if term or trunc:
                break

        # ── Collect episode metrics ──────────────────────────────────────────
        total_reward = float(np.mean(
            [np.sum(env.reward_history[i]) for i in range(config.num_satellites)]
        ))
        # targets_hit: sum of avoidance events (faults_recovered as proxy)
        targets_hit = int(np.sum(env.faults_recovered))
        # wasted_actions: steps spent in safe mode (conservative proxy)
        wasted_actions = int(np.sum(env.in_safe_mode))
        # fuel_remaining: mean fuel % across all satellites
        fuel_remaining = float(np.mean(env.agent_fuel))

        memory_helped_rate = (
            float(np.mean(ep_memory_helped_flags))
            if ep_memory_helped_flags else 0.0
        )

        ep_rewards.append(total_reward)
        ep_targets_hit.append(targets_hit)
        ep_wasted_actions.append(wasted_actions)
        ep_fuel_remaining.append(fuel_remaining)
        ep_memory_helped.append(memory_helped_rate)

        print(f"    Ep {ep_idx+1:2d}/{NUM_EPISODES} | "
              f"Reward: {total_reward:+8.2f} | "
              f"Targets: {targets_hit:3d} | "
              f"Fuel: {fuel_remaining:5.1f}% | "
              f"MemHelped: {memory_helped_rate:.2%}")

    results = {
        "avg_total_reward":   float(np.mean(ep_rewards)),
        "avg_targets_hit":    float(np.mean(ep_targets_hit)),
        "avg_wasted_actions": float(np.mean(ep_wasted_actions)),
        "avg_fuel_remaining": float(np.mean(ep_fuel_remaining)),
        "memory_helped_rate": float(np.mean(ep_memory_helped)) * 100.0,  # as %
    }
    return results

# ─────────────────────────────────────────────────────────────────────────────
#  Step 7: Print comparison table
# ─────────────────────────────────────────────────────────────────────────────
def _pct_change(baseline_val: float, osg_val: float) -> str:
    if abs(baseline_val) < 1e-9:
        return "N/A"
    pct = (osg_val - baseline_val) / abs(baseline_val) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def print_comparison(baseline: dict, osg: dict,
                     altitude_km: float, beta: float, T_sec: float):
    metrics = [
        ("Avg Total Reward",   "avg_total_reward",   "{:+.2f}",  "{:+.2f}"),
        ("Avg Targets Hit",    "avg_targets_hit",    "{:.2f}",   "{:.2f}"),
        ("Avg Wasted Actions", "avg_wasted_actions", "{:.2f}",   "{:.2f}"),
        ("Avg Fuel Remaining", "avg_fuel_remaining", "{:.2f}%",  "{:.2f}%"),
        ("Memory Helped Rate", "memory_helped_rate", "{:.2f}%",  "{:.2f}%"),
    ]

    W = 72
    print("\n" + "=" * W)
    print("  OSG vs Baseline: 10-Episode Evaluation")
    print("=" * W)
    print(f"  Orbital altitude : {altitude_km:.1f} km")
    print(f"  Kepler β         = {beta:.6f} per step")
    print(f"  Orbital period   = {T_sec:.1f} seconds")
    print()
    print(f"  {'Metric':<22} {'Baseline':>10} {'OSG':>10} {'Change':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")

    for label, key, bfmt, ofmt in metrics:
        bval = baseline[key]
        oval = osg[key]
        print(f"  {label:<22} {bfmt.format(bval):>10} "
              f"{ofmt.format(oval):>10} {_pct_change(bval, oval):>10}")

    print("=" * W)

# ─────────────────────────────────────────────────────────────────────────────
#  Step 8: Save CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(baseline: dict, osg: dict):
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    rows = []
    for key in baseline:
        bval = baseline[key]
        oval = osg[key]
        if abs(bval) > 1e-9:
            pct = (oval - bval) / abs(bval) * 100.0
        else:
            pct = float("nan")
        rows.append({
            "metric":         key,
            "baseline":       f"{bval:.6f}",
            "osg":            f"{oval:.6f}",
            "percent_change": f"{pct:.4f}",
        })

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "baseline", "osg", "percent_change"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV saved → {CSV_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 68)
    print("  Orbital Salience Gating (OSG) Experiment")
    print("  Physics-grounded memory retrieval vs cosine baseline")
    print("=" * 68)

    # ── Config ────────────────────────────────────────────────────────────────
    config = PRESETS["starlink_leo"]
    print(f"\n  Config: {config.name}  "
          f"({config.num_satellites} satellites, {config.orbit_type.value})")

    # ── Step 1: Load model ────────────────────────────────────────────────────
    print("\n[STEP 1] Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        print("  Please train the model first: python rl_training/train.py")
        sys.exit(1)
    model = load_model(config)
    print("  ✓ Model loaded")

    # ── Step 2: Load memories ─────────────────────────────────────────────────
    print("\n[STEP 2] Loading episodic memories...")
    all_episodes = load_all_memories()
    print(f"  ✓ {len(all_episodes)} total episodes in retrieval pool")

    # ── Step 3: Compute physics-derived β ────────────────────────────────────
    print("\n[STEP 3] Computing Kepler-derived decay rate β...")
    beta, altitude_km, T_sec = get_kepler_beta(config)
    print(f"  ✓ β = {beta:.8f} per step")

    # ── Step 4/5: Run both sessions ───────────────────────────────────────────
    print("\n[STEP 4] Running SESSION A — BASELINE (cosine-only)...")
    baseline_results = run_session(
        model, config, all_episodes,
        use_osg=False, beta=beta, label="A"
    )
    print("  ✓ Baseline session complete")

    print("\n[STEP 5] Running SESSION B — OSG (Kepler β = {:.6f})...".format(beta))
    osg_results = run_session(
        model, config, all_episodes,
        use_osg=True, beta=beta, label="B"
    )
    print("  ✓ OSG session complete")

    # ── Step 6/7: Print comparison table ──────────────────────────────────────
    print("\n[STEP 6] Final comparison:")
    print_comparison(baseline_results, osg_results, altitude_km, beta, T_sec)

    # ── Step 8: Save CSV ──────────────────────────────────────────────────────
    print("\n[STEP 7] Saving results to CSV...")
    save_csv(baseline_results, osg_results)
    print("  ✓ Results saved")

    print("\n  Experiment complete.\n")


if __name__ == "__main__":
    # Suppress SB3 deprecation noise during evaluation
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
