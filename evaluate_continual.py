"""
evaluate_continual.py
---------------------
Continual Learning Comparison Experiment — Novel Research Contribution.

Proves that PPO + EWC + Prioritized Experience Replay preserves Task-1
knowledge when the agent is trained on a harder Task-2, while standard
PPO (no EWC) suffers catastrophic forgetting.

─────────────────────────────────────────────────────────────────────────────
EXPERIMENTAL DESIGN
─────────────────────────────────────────────────────────────────────────────

Task 1: Standard mission (current setup)
  • Target gap  : 90°  between two satellites
  • Debris      : random spawn (0–3 objects)
  • Conditions  : random eclipse, solar storms

Task 2: Harder mission (new learning pressure → causes forgetting)
  • Target gap  : 120° (harder — must rearrange more)
  • Initial debris: 3 objects always pre-spawned (more hostile environment)
  • Conditions  : same eclipse / storm mechanics

─────────────────────────────────────────────────────────────────────────────
SCENARIOS COMPARED
─────────────────────────────────────────────────────────────────────────────

Scenario A — Baseline (no continual learning):
  1. Load trained Task-1 models
  2. Train on Task-2 for TASK2_CYCLES cycles  (pure PPO, no EWC/replay)
  3. Evaluate Task-1 performance
  → Shows how much the agent FORGETS Task-1 while learning Task-2

Scenario B — Continual learning (EWC + Prioritized Replay):
  1. Load trained Task-1 models + EWC Fisher matrices
  2. Train on Task-2 for TASK2_CYCLES cycles  (PPO + EWC correction)
     Environment reset has 20% probability of replaying a high-reward
     Task-1 episode (Prioritized Experience Replay)
  3. Evaluate Task-1 performance
  → Shows much LESS forgetting

COMPARISON TABLE printed at the end shows:
  | Metric                              | Without EWC | With EWC+Replay |
  |-------------------------------------|-------------|-----------------|
  | Task-1 reward after Task-2 training |             |                 |
  | Task-2 learning speed               |             |                 |
  | Weight change magnitude             |             |                 |
  | Collisions (Task-1 after Task-2)    |             |                 |

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  python evaluate_continual.py

Requires:
  • ppo_satellite_1.zip / ppo_satellite_2.zip  (Task-1 trained models)
  • ewc_fisher_sat1.pkl / ewc_fisher_sat2.pkl  (computed by train.py)

Run train.py first if these files don't exist.
"""

import io
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from satellite_env import (
    MultiSatelliteEnv, SingleAgentWrapper,
    LOCAL_DIM, GLOBAL_DIM, TARGET_SLOT_GAP,
)
from ctde_policy  import CTDEPolicy
from memory       import EpisodicMemory
from ewc          import EWC
from train        import (
    _load_or_create, _augment_obs, LaggedPolicy,
    ARCH_VERSION, MODEL1_PATH, MODEL2_PATH,
    EWC1_PATH, EWC2_PATH, EWC_LAMBDA, EWC_N_SAMPLES,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Experiment parameters
# ─────────────────────────────────────────────────────────────────────────────
TASK2_TARGET_GAP    = 120.0      # degrees — harder than Task-1's 90°
TASK2_CYCLES        = 20         # Task-2 training cycles per scenario
TASK2_CHUNK_STEPS   = 5_000      # steps per cycle
EVAL_EPISODES       = 15         # evaluation episodes per task per scenario
REPLAY_PROB         = 0.20       # 20% of Task-2 resets start from Task-1 memory

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _clone_model(model, env):
    """Deep-copy a PPO model via SB3's save/load (avoids deepcopy issues)."""
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    return PPO.load(buf, env=env,
                    custom_objects={"policy_class": CTDEPolicy})


def evaluate_task(model1, model2, memory, task_name: str,
                  target_gap: float = TARGET_SLOT_GAP,
                  n_episodes: int = EVAL_EPISODES,
                  ep_steps: int = 360) -> dict:
    """
    Run n_episodes evaluation episodes and return performance metrics.

    Args:
        target_gap: the target slot gap for this task (90° = Task-1, 120° = Task-2)
    """
    total_rewards   = [[], []]
    total_coll      = [[], []]
    total_fuel      = [[], []]

    for ep in range(n_episodes):
        env = MultiSatelliteEnv(max_steps=ep_steps, target_gap=target_gap)
        obs_raw, _ = env.reset()

        for _ in range(ep_steps):
            obs1 = _augment_obs(obs_raw[0], memory)
            obs2 = _augment_obs(obs_raw[1], memory)
            a1, _ = model1.predict(obs1, deterministic=True)
            a2, _ = model2.predict(obs2, deterministic=True)
            obs_raw, _, terminated, truncated, _ = env.step([int(a1), int(a2)])
            if terminated or truncated:
                break

        for i in range(2):
            total_rewards[i].append(np.sum(env.reward_history[i]))
            total_coll[i].append(env.collisions[i])
            total_fuel[i].append(env.fuel_outs[i])

        env.close()

    return {
        "task":         task_name,
        "avg_reward":   np.mean([np.mean(total_rewards[i]) for i in range(2)]),
        "avg_coll":     np.mean([np.mean(total_coll[i])    for i in range(2)]),
        "avg_fuel":     np.mean([np.mean(total_fuel[i])    for i in range(2)]),
        "sat1_reward":  float(np.mean(total_rewards[0])),
        "sat2_reward":  float(np.mean(total_rewards[1])),
    }


def train_on_task2(model1, model2, memory, use_ewc: bool,
                   ewc1: EWC = None, ewc2: EWC = None,
                   n_cycles: int = TASK2_CYCLES) -> list:
    """
    Train model1 and model2 on Task-2 for n_cycles cycles.

    Args:
        use_ewc  : if True, apply EWC correction after each PPO update
                   and use PER (Prioritized Experience Replay) in resets.
        ewc1/ewc2: pre-computed EWC objects (only used when use_ewc=True)

    Returns:
        List of (cycle, sat1_reward, sat2_reward) tuples for plotting.
    """
    replay_p = REPLAY_PROB if use_ewc else 0.0

    env1 = SingleAgentWrapper(
        num_positions=360, max_steps=360, agent_idx=0,
        memory=memory, target_gap=TASK2_TARGET_GAP, replay_prob=replay_p,
    )
    env2 = SingleAgentWrapper(
        num_positions=360, max_steps=360, agent_idx=1,
        memory=memory, target_gap=TASK2_TARGET_GAP, replay_prob=replay_p,
    )

    model1.set_env(env1)
    model2.set_env(env2)

    lagged1 = lagged2 = None
    progress = []

    for cycle in range(1, n_cycles + 1):
        env1.set_other_model(lagged2 if lagged2 else model2)
        model1.learn(total_timesteps=TASK2_CHUNK_STEPS, reset_num_timesteps=False)
        if use_ewc and ewc1 and ewc1.is_active():
            ewc1.apply_correction(model1, n_steps=5)

        env2.set_other_model(lagged1 if lagged1 else model1)
        model2.learn(total_timesteps=TASK2_CHUNK_STEPS, reset_num_timesteps=False)
        if use_ewc and ewc2 and ewc2.is_active():
            ewc2.apply_correction(model2, n_steps=5)

        lagged1 = LaggedPolicy(model1)
        lagged2 = LaggedPolicy(model2)

        def _mean_rew(m):
            buf = m.ep_info_buffer
            return np.mean([ep["r"] for ep in buf]) if buf else float("nan")

        r1, r2 = _mean_rew(model1), _mean_rew(model2)
        progress.append((cycle, r1, r2))

        tag = "[EWC+PER]" if use_ewc else "[standard]"
        print(f"  Task-2 cycle {cycle:>3}/{n_cycles}  {tag}  "
              f"Sat1: {r1:+.1f}  Sat2: {r2:+.1f}")

    return progress


# ─────────────────────────────────────────────────────────────────────────────
#  Main experiment
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  CONTINUAL LEARNING COMPARISON EXPERIMENT")
    print("="*70)
    print(f"  Task 1: gap=90°  (standard mission, already trained)")
    print(f"  Task 2: gap={TASK2_TARGET_GAP:.0f}°  (harder mission, trained now)")
    print(f"  Task-2 cycles : {TASK2_CYCLES}  ×  {TASK2_CHUNK_STEPS:,} steps")
    print(f"  EWC λ         : {EWC_LAMBDA:.0f}  (Kirkpatrick 2017 default)")
    print(f"  PER replay    : {REPLAY_PROB*100:.0f}% of Task-2 resets from Task-1 memory")

    # ── Check prerequisites ────────────────────────────────────────────────────
    missing = []
    for f in [MODEL1_PATH + ".zip", MODEL2_PATH + ".zip"]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        print(f"\n  ERROR: Missing files: {missing}")
        print("  Please run 'python train.py' first to train Task-1 models.")
        return

    ewc_files_exist = (
        os.path.exists(EWC1_PATH) and os.path.exists(EWC2_PATH)
    )
    if not ewc_files_exist:
        print(f"\n  WARNING: EWC Fisher files not found ({EWC1_PATH}, {EWC2_PATH}).")
        print("  Scenario B (EWC) will compute Fisher matrices fresh — this is slower.")
        print("  Tip: Run 'python train.py' to completion first to pre-compute them.\n")

    memory = EpisodicMemory()
    print(f"\n  Memory: {memory.stats}\n")

    # ── Load Task-1 models for Scenario A ────────────────────────────────────
    print("─" * 70)
    print("  LOADING Task-1 models...")

    _env_dummy_1 = SingleAgentWrapper(agent_idx=0, memory=memory)
    _env_dummy_2 = SingleAgentWrapper(agent_idx=1, memory=memory)

    base_model1 = _load_or_create(MODEL1_PATH, _env_dummy_1, "Sat-1")
    base_model2 = _load_or_create(MODEL2_PATH, _env_dummy_2, "Sat-2")

    # ── Evaluate BASELINE Task-1 performance (before any Task-2 training) ────
    print("\n  Evaluating baseline Task-1 performance (before Task-2 training)...")
    baseline_t1 = evaluate_task(base_model1, base_model2, memory,
                                 "Baseline Task-1 (before Task-2)",
                                 target_gap=TARGET_SLOT_GAP)
    print(f"  Baseline Task-1: Sat1={baseline_t1['sat1_reward']:+.1f}  "
          f"Sat2={baseline_t1['sat2_reward']:+.1f}  "
          f"Avg={baseline_t1['avg_reward']:+.1f}")

    # ═════════════════════════════════════════════════════════════════════════
    # SCENARIO A — No EWC (standard PPO only)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  SCENARIO A — Task-2 training WITHOUT EWC (standard PPO)")
    print("="*70)

    # Clone models so Scenario A and B start from the SAME weights
    dummy1a = SingleAgentWrapper(agent_idx=0, memory=memory,
                                 target_gap=TASK2_TARGET_GAP)
    dummy2a = SingleAgentWrapper(agent_idx=1, memory=memory,
                                 target_gap=TASK2_TARGET_GAP)
    model1_a = _clone_model(base_model1, dummy1a)
    model2_a = _clone_model(base_model2, dummy2a)

    progress_a = train_on_task2(model1_a, model2_a, memory,
                                use_ewc=False, n_cycles=TASK2_CYCLES)

    print("\n  Evaluating Task-1 performance AFTER Task-2 training (Scenario A)...")
    after_a_t1 = evaluate_task(model1_a, model2_a, memory,
                                "After Task-2 (no EWC)",
                                target_gap=TARGET_SLOT_GAP)
    print(f"  Task-1 after A: Avg={after_a_t1['avg_reward']:+.1f}  "
          f"(Δ={after_a_t1['avg_reward'] - baseline_t1['avg_reward']:+.1f})")

    print("\n  Evaluating Task-2 performance for Scenario A...")
    after_a_t2 = evaluate_task(model1_a, model2_a, memory,
                                "Task-2 performance (no EWC)",
                                target_gap=TASK2_TARGET_GAP)
    print(f"  Task-2 score A: Avg={after_a_t2['avg_reward']:+.1f}")

    # ═════════════════════════════════════════════════════════════════════════
    # SCENARIO B — With EWC + Prioritized Replay
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  SCENARIO B — Task-2 training WITH EWC + Prioritized Replay")
    print("="*70)

    # Load / compute EWC Fisher matrices
    ewc1 = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC1_PATH)
    ewc2 = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC2_PATH)

    if not ewc1.is_active():
        print("  Computing Fisher matrices for Sat-1...")
        ewc1.compute_fisher(base_model1, _env_dummy_1, n_samples=EWC_N_SAMPLES)
    if not ewc2.is_active():
        print("  Computing Fisher matrices for Sat-2...")
        ewc2.compute_fisher(base_model2, _env_dummy_2, n_samples=EWC_N_SAMPLES)

    # Measure weight change magnitude BEFORE Task-2 (should be 0)
    wcm1_before = ewc1.weight_change_magnitude(base_model1)
    wcm2_before = ewc2.weight_change_magnitude(base_model2)

    dummy1b = SingleAgentWrapper(agent_idx=0, memory=memory,
                                 target_gap=TASK2_TARGET_GAP,
                                 replay_prob=REPLAY_PROB)
    dummy2b = SingleAgentWrapper(agent_idx=1, memory=memory,
                                 target_gap=TASK2_TARGET_GAP,
                                 replay_prob=REPLAY_PROB)
    model1_b = _clone_model(base_model1, dummy1b)
    model2_b = _clone_model(base_model2, dummy2b)

    progress_b = train_on_task2(model1_b, model2_b, memory,
                                use_ewc=True, ewc1=ewc1, ewc2=ewc2,
                                n_cycles=TASK2_CYCLES)

    print("\n  Evaluating Task-1 performance AFTER Task-2 training (Scenario B)...")
    after_b_t1 = evaluate_task(model1_b, model2_b, memory,
                                "After Task-2 (EWC+PER)",
                                target_gap=TARGET_SLOT_GAP)
    print(f"  Task-1 after B: Avg={after_b_t1['avg_reward']:+.1f}  "
          f"(Δ={after_b_t1['avg_reward'] - baseline_t1['avg_reward']:+.1f})")

    print("\n  Evaluating Task-2 performance for Scenario B...")
    after_b_t2 = evaluate_task(model1_b, model2_b, memory,
                                "Task-2 performance (EWC+PER)",
                                target_gap=TASK2_TARGET_GAP)
    print(f"  Task-2 score B: Avg={after_b_t2['avg_reward']:+.1f}")

    # Measure weight change magnitude AFTER Task-2 (EWC should limit this)
    wcm1_after_b = ewc1.weight_change_magnitude(model1_b)
    wcm2_after_b = ewc2.weight_change_magnitude(model2_b)
    wcm_b = (wcm1_after_b + wcm2_after_b) / 2

    # Compute weight change for Scenario A (no EWC protection)
    wcm1_after_a = ewc1.weight_change_magnitude(model1_a)
    wcm2_after_a = ewc2.weight_change_magnitude(model2_a)
    wcm_a = (wcm1_after_a + wcm2_after_a) / 2

    # Task-2 learning speed: cycle where reward first crosses +20
    def _cycles_to_threshold(progress, threshold=20.0) -> str:
        for cyc, r1, r2 in progress:
            if np.nanmean([r1, r2]) >= threshold:
                return str(cyc)
        return f">{len(progress)}"

    spd_a = _cycles_to_threshold(progress_a)
    spd_b = _cycles_to_threshold(progress_b)

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  CONTINUAL LEARNING COMPARISON RESULTS")
    print("="*70)

    forgetting_a = after_a_t1["avg_reward"] - baseline_t1["avg_reward"]
    forgetting_b = after_b_t1["avg_reward"] - baseline_t1["avg_reward"]

    col_w = 22
    def row(label, val_a, val_b, better="higher"):
        if isinstance(val_a, float) and isinstance(val_b, float):
            win = "✓" if (val_b > val_a) == (better == "higher") else " "
            va = f"{val_a:+.2f}"
            vb = f"{val_b:+.2f}  {win}"
        else:
            va, vb = str(val_a), str(val_b)
        print(f"  {label:<38} {va:>{col_w}} {vb:>{col_w}}")

    print(f"\n  {'Metric':<38} {'Without EWC':>{col_w}} {'With EWC+Replay':>{col_w}}")
    print(f"  {'-'*38} {'-'*col_w} {'-'*col_w}")

    row("Baseline Task-1 reward (reference)",
        baseline_t1["avg_reward"], baseline_t1["avg_reward"])
    row("Task-1 reward AFTER Task-2 training",
        after_a_t1["avg_reward"], after_b_t1["avg_reward"])
    row("Forgetting Δ  (higher = less forgetting)",
        forgetting_a, forgetting_b)
    row("Task-2 reward (new task learned)",
        after_a_t2["avg_reward"], after_b_t2["avg_reward"])
    row("Task-2 learning speed (cycles to +20)",
        spd_a, spd_b, better="lower")
    row("Weight change magnitude (↓ = better retention)",
        wcm_a, wcm_b, better="lower")
    row("Collisions Task-1 after Task-2",
        after_a_t1["avg_coll"], after_b_t1["avg_coll"], better="lower")

    print(f"\n  {'─'*70}")
    retention_pct_a = (after_a_t1["avg_reward"] / max(abs(baseline_t1["avg_reward"]), 1)) * 100
    retention_pct_b = (after_b_t1["avg_reward"] / max(abs(baseline_t1["avg_reward"]), 1)) * 100
    print(f"  Task-1 retention: {retention_pct_a:.1f}%  (without EWC)  "
          f"vs  {retention_pct_b:.1f}%  (with EWC+PER)")
    improvement = retention_pct_b - retention_pct_a
    print(f"  EWC+PER improvement in retention: {improvement:+.1f} percentage points")

    if improvement > 0:
        print(f"\n  ✓ RESULT: EWC + Prioritized Replay REDUCES catastrophic forgetting!")
        print(f"            Task-1 knowledge preserved {improvement:.1f}pp better.")
    else:
        print(f"\n  ⚠ RESULT: EWC did not help in this run. Try increasing λ or n_samples.")

    # ── Save CSV for MATLAB ───────────────────────────────────────────────────
    os.makedirs("matlab_export", exist_ok=True)
    csv_path = "matlab_export/continual_learning_results.csv"
    with open(csv_path, "w") as f:
        f.write("scenario,baseline_t1,after_task2_t1,forgetting,task2_reward,"
                "weight_change_magnitude,cycles_to_t2_threshold\n")
        f.write(f"without_ewc,{baseline_t1['avg_reward']:.4f},"
                f"{after_a_t1['avg_reward']:.4f},{forgetting_a:.4f},"
                f"{after_a_t2['avg_reward']:.4f},{wcm_a:.8f},{spd_a}\n")
        f.write(f"with_ewc_per,{baseline_t1['avg_reward']:.4f},"
                f"{after_b_t1['avg_reward']:.4f},{forgetting_b:.4f},"
                f"{after_b_t2['avg_reward']:.4f},{wcm_b:.8f},{spd_b}\n")
    print(f"\n  ✓ Results saved → {csv_path}")

    # ── Plot learning curves ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5), facecolor="#07071a")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    for ax in [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]:
        ax.set_facecolor("#0d0d22")
        ax.tick_params(colors="#8899cc", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#1a1a40")
        ax.grid(color="#0e0e22", linestyle="--", alpha=0.7)

    ax1, ax2 = fig.axes

    # Task-2 learning curve
    cyc_a = [p[0] for p in progress_a]
    rew_a = [np.nanmean([p[1], p[2]]) for p in progress_a]
    cyc_b = [p[0] for p in progress_b]
    rew_b = [np.nanmean([p[1], p[2]]) for p in progress_b]

    ax1.plot(cyc_a, rew_a, color="#ff4444", linewidth=2, label="Without EWC")
    ax1.plot(cyc_b, rew_b, color="#44ff88", linewidth=2, label="With EWC+PER")
    ax1.axhline(20, color="#666688", linewidth=0.8, linestyle=":", alpha=0.7)
    ax1.set_title("Task-2 Learning Curve", color="#8899cc", pad=6, fontsize=10)
    ax1.set_xlabel("Cycle", color="#8899cc")
    ax1.set_ylabel("Mean Reward", color="#8899cc")
    ax1.legend(facecolor="#0a0a20", edgecolor="none", labelcolor="#aabbdd",
               fontsize=8)

    # Task-1 retention bar chart
    labels  = ["Baseline\nTask-1", "Without EWC\n(after Task-2)", "With EWC+PER\n(after Task-2)"]
    heights = [baseline_t1["avg_reward"], after_a_t1["avg_reward"], after_b_t1["avg_reward"]]
    colors  = ["#3399ff", "#ff4444", "#44ff88"]
    bars    = ax2.bar(labels, heights, color=colors, width=0.55)
    for bar, h in zip(bars, heights):
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                 f"{h:+.1f}", ha="center", va="bottom",
                 color="#cce4ff", fontsize=8)
    ax2.set_title("Task-1 Retention After Task-2 Training", color="#8899cc",
                  pad=6, fontsize=10)
    ax2.set_ylabel("Mean Episode Reward", color="#8899cc")
    ax2.axhline(0, color="#333355", linewidth=0.8, linestyle=":")
    ax2.tick_params(colors="#8899cc", labelsize=7)

    fig.suptitle("Continual Learning: EWC + Prioritized Experience Replay",
                 color="#cce4ff", fontsize=12, y=1.01)

    plot_path = "matlab_export/continual_learning_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✓ Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
