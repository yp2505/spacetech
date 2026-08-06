"""
train.py
--------
Phase 3: CTDE + Episodic Memory + Realistic Orbital Mechanics.

What's new vs Phase 2
---------------------
  - 5-level thruster control (Full Retro / Light Retro / Coast / Light Pro / Full Pro)
  - Momentum-based orbital physics (angular velocity accumulates from thrusters)
  - Fully randomized spawn positions across all 360 degrees
  - Richer obs: velocity, gap error, closing rate, orbits completed (LOCAL_DIM=15)
  - Expanded Critic: sat0/sat1 velocity added (GLOBAL_DIM=13)
  - Updated reward shaping for long-range rendezvous
"""

import os
import io
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from satellite_env import (
    MultiSatelliteEnv, SingleAgentWrapper,
    LOCAL_DIM, GLOBAL_DIM, TARGET_SLOT_GAP,
)
from ctde_policy  import CTDEPolicy
from memory       import EpisodicMemory
from ewc          import EWC


# ── Architecture version marker ────────────────────────────────────────────────
# If the saved model was trained with a different policy architecture,
# loading it will cause a shape mismatch.  We track the version to
# detect this and start fresh automatically.
ARCH_VERSION    = "CTDEv3_local15_global13_thrust5"
ARCH_FILE       = "arch_version.txt"
MODEL1_PATH     = "ppo_satellite_1"
MODEL2_PATH     = "ppo_satellite_2"
EWC1_PATH       = "ewc_fisher_sat1.pkl"
EWC2_PATH       = "ewc_fisher_sat2.pkl"
EWC_LAMBDA      = 5_000.0   # Kirkpatrick 2017 default
EWC_N_SAMPLES   = 300       # states sampled to compute Fisher


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _arch_matches() -> bool:
    """True iff saved models were trained with the current CTDE architecture."""
    if not os.path.exists(ARCH_FILE):
        return False
    with open(ARCH_FILE) as f:
        return f.read().strip() == ARCH_VERSION


def _save_arch_marker():
    with open(ARCH_FILE, "w") as f:
        f.write(ARCH_VERSION)


def _load_or_create(path: str, env, label: str):
    """
    Load a saved PPO model if the architecture version matches,
    otherwise create a fresh CTDEPolicy model.
    On resume: lower the LR to 1e-4 for fine-tuning stability.
    """
    if os.path.exists(path + ".zip") and _arch_matches():
        try:
            model = PPO.load(path, env=env, device="cpu",
                             custom_objects={"learning_rate": 3e-4, "policy_class": CTDEPolicy})
            print(f"  -> Resumed:  {path}.zip  ({label})  [LR=1e-4]")
            return model
        except Exception as exc:
            print(f"  -> Load failed ({exc}).  Creating fresh {label}.")

    print(f"  -> Fresh model: {label}  (CTDE policy)")
    return PPO(
        CTDEPolicy, env,
        verbose=0,
        learning_rate=3e-4,       # initial LR; lowered to 1e-4 on next resume
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,            # entropy bonus encourages exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
    )


def _augment_obs(base_obs: dict, memory: EpisodicMemory) -> dict:
    """Append memory context to a raw MultiSatelliteEnv obs (8-dim → 12-dim local)."""
    ctx = memory.get_context()
    return {
        "local":  np.concatenate([base_obs["local"], ctx]).astype(np.float32),
        "global": base_obs["global"],
    }


class LaggedPolicy:
    """
    Frozen deep-copy of a policy for use as a stable opponent.

    WHY: When Model1 trains against the CURRENT Model2, and Model2 simultaneously
    trains against Model1, both targets are moving.  This creates oscillating
    rewards and V(s) divergence.

    FIX (used in AlphaStar / OpenAI Five): each model trains against the
    PREVIOUS cycle\'s frozen snapshot of the other model.  The opponent is
    stable for the entire cycle, giving clean gradient signal.
    """

    def __init__(self, model):
        """
        Snapshot the model using SB3's own save/load via an in-memory buffer.
        copy.deepcopy() fails on non-leaf PyTorch tensors, but save/load
        operates on plain state_dicts which are fully copyable.
        """
        buf = io.BytesIO()
        model.save(buf)
        buf.seek(0)
        self._frozen = PPO.load(
            buf,
            env=model.get_env(),
            custom_objects={"policy_class": CTDEPolicy},
        )
        self._frozen.policy.set_training_mode(False)

    def predict(self, obs, deterministic: bool = True):
        """Drop-in replacement for model.predict() — fully handles tensor conversion."""
        return self._frozen.predict(obs, deterministic=deterministic)


# ─────────────────────────────────────────────────────────────────────────────
#  Q-value proxy logging  (V(s) from centralised Critic)
# ─────────────────────────────────────────────────────────────────────────────
def log_q_values(model, env_wrapper: SingleAgentWrapper, label: str):
    """
    Estimates mean V(s) over 16 random initial states.
    V(s) is the Critic's expected return — a proxy for Q-values.
    Q(s,a) ≈ V(s) + A(s,a).  Since advantages average to 0 over the policy,
    V(s) is the single best indicator of long-term stability.
    """
    values = []
    for _ in range(16):
        obs, _ = env_wrapper.reset()
        obs_t = obs_as_tensor(
            {"local":  obs["local"][None, :],
             "global": obs["global"][None, :]},
            model.device,
        )
        with torch.no_grad():
            feats     = model.policy.extract_features(obs_t)
            latent_vf = model.policy.mlp_extractor.forward_critic(feats)
            v         = model.policy.value_net(latent_vf)
        values.append(float(v.item()))
    mean_v = np.mean(values)
    print(f"    [{label}] Q-proxy V(s): {mean_v:+.3f}  "
          f"(min {min(values):+.2f} / max {max(values):+.2f})")
    return mean_v


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(model1, model2, memory: EpisodicMemory,
                   num_episodes=10, ep_steps=360, render_last=True,
                   store_start_conditions=True):
    """
    Runs num_episodes evaluation episodes in a fresh shared environment.
    Returns per-episode stats and records every episode into episodic memory.
    store_start_conditions: if True, save starting state for PER replay.
    """
    collisions_per_ep  = [[], []]
    fuel_outs_per_ep   = [[], []]
    mean_reward_per_ep = [[], []]

    for ep in range(num_episodes):
        env        = MultiSatelliteEnv(max_steps=ep_steps)
        obs_raw, _ = env.reset()
        start_cond = env.get_start_conditions() if store_start_conditions else {}
        do_render  = render_last and (ep == num_episodes - 1)

        for step in range(ep_steps):
            if do_render:
                env.render()

            # Augment raw obs with memory context before predicting
            obs1 = _augment_obs(obs_raw[0], memory)
            obs2 = _augment_obs(obs_raw[1], memory)

            a1, _ = model1.predict(obs1, deterministic=True)
            a2, _ = model2.predict(obs2, deterministic=True)
            obs_raw, rewards, terminated, truncated, _ = env.step(
                [int(a1), int(a2)]
            )
            if terminated or truncated:
                break

        collisions_per_ep[0].append(env.collisions[0])
        collisions_per_ep[1].append(env.collisions[1])
        fuel_outs_per_ep[0].append(env.fuel_outs[0])
        fuel_outs_per_ep[1].append(env.fuel_outs[1])
        mean_reward_per_ep[0].append(np.sum(env.reward_history[0]))
        mean_reward_per_ep[1].append(np.sum(env.reward_history[1]))

        # Record each evaluation episode into long-term episodic memory
        for i in range(2):
            ep_rew = np.sum(env.reward_history[i])
            memory.record(
                total_reward=ep_rew,
                collisions=env.collisions[i],
                fuel_outs=env.fuel_outs[i],
                steps=env.current_step,
                was_eclipse=any(env.eclipse_mode),
                was_weather=env.space_weather_active,
                start_conditions=start_cond if ep_rew > 0 else {},
            )

        if do_render:
            print("Animation complete. Close the window to exit.")
            plt.ioff()
            plt.show()
            env.close()

    return collisions_per_ep, fuel_outs_per_ep, mean_reward_per_ep, env


# ─────────────────────────────────────────────────────────────────────────────
#  Summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(label, collisions_per_ep, fuel_outs_per_ep, mean_reward_per_ep):
    W = 62
    print(f"\n{'─'*W}")
    print(f"  {label} — 10-Episode Evaluation Summary")
    print(f"{'─'*W}")
    print(f"  {'Satellite':<18} {'Collisions':>10} {'Fuel Outs':>10} {'Avg Total Rew':>15}")
    print(f"  {'─'*(W-2)}")
    for i, name in enumerate(["Sat 1 (Red)   ", "Sat 2 (Orange)"]):
        avg_coll = np.mean(collisions_per_ep[i])
        avg_fuel = np.mean(fuel_outs_per_ep[i])
        avg_rew  = np.mean(mean_reward_per_ep[i])
        print(f"  {name:<18} {avg_coll:>10.2f} {avg_fuel:>10.2f} {avg_rew:>15.2f}")
    print(f"{'─'*W}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
import argparse

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true', help='Skip training and just evaluate')
    args = parser.parse_args()

    chunk_steps     = 5_000    # steps per agent per co-training cycle
    num_cycles      = 200      # total cycles → 1 Million steps total
    total_timesteps = chunk_steps * num_cycles

    # ── Episodic memory (persistent across runs) ───────────────────────────────
    memory = EpisodicMemory()
    print(f"\n[Memory] {memory.stats}")

    print("\n" + "="*62)
    print("  INITIALIZATION  (CTDE + Episodic Memory + EWC + PER)")
    print("="*62)
    print(f"  Policy : CTDEPolicy  "
          f"(Actor: local {LOCAL_DIM}-dim | Critic: global {GLOBAL_DIM}-dim)")
    print(f"  Arch   : {ARCH_VERSION}")

    # ── EWC (Elastic Weight Consolidation) ────────────────────────────────────
    ewc1 = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC1_PATH)
    ewc2 = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC2_PATH)
    ewc_active = ewc1.is_active() and ewc2.is_active()
    print(f"  EWC    : {'ACTIVE (λ={EWC_LAMBDA:.0f})  — protecting Task-1 knowledge' if ewc_active else 'inactive (Fisher not yet computed)'}")

    # ── Environments (each agent owns its own env) ─────────────────────────────
    env1 = SingleAgentWrapper(num_positions=360, max_steps=360,
                              agent_idx=0, memory=memory)
    env2 = SingleAgentWrapper(num_positions=360, max_steps=360,
                              agent_idx=1, memory=memory)

    # ── Load or create models ──────────────────────────────────────────────────
    model1 = _load_or_create(MODEL1_PATH, env1, "Sat-1")
    model2 = _load_or_create(MODEL2_PATH, env2, "Sat-2")

    # ── Baseline evaluation ────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  BASELINE: Evaluating current models (10 episodes)")
    print("="*62)
    b_coll, b_fuel, b_rew, _ = run_evaluation(
        model1, model2, memory,
        num_episodes=10, ep_steps=360, render_last=False,
    )
    print_summary("BASELINE (pre-cycle)", b_coll, b_fuel, b_rew)
    print(f"\n  Memory context: {memory.get_context()}")

    # ── Alternating CTDE co-training ───────────────────────────────────────────
    print("\n" + "="*62)
    print(f"  CO-TRAINING: {num_cycles} cycles × {chunk_steps:,} steps = "
          f"{total_timesteps:,} steps each")
    print("="*62)

    header = f"\n  {'Cycle':>5}  {'Sat1 rew':>10}  {'Sat2 rew':>10}  "
    header += f"{'V(s)_1':>8}  {'V(s)_2':>8}  {'Mem episodes':>13}"
    print(header)
    print(f"  {'─'*70}")

    # ── Co-training setup: lagged frozen opponents ─────────────────────────────
    # Each model trains against a FROZEN COPY of the previous cycle's opponent.
    # This makes the training target stable for an entire cycle → faster, smoother
    # convergence (same principle as AlphaStar / OpenAI Five league training).
    lagged1: LaggedPolicy | None = None   # frozen snapshot of model1
    lagged2: LaggedPolicy | None = None   # frozen snapshot of model2

    progress_data = []

    if not args.eval_only:
        for cycle in range(1, num_cycles + 1):
            # ── Curriculum Learning Phase progression ──────────────────────────
            if cycle <= 20:
                phase = 1
            elif cycle <= 40:
                phase = 2
            elif cycle <= 100:
                phase = 3
            else:
                phase = 4
            env1.set_curriculum_phase(phase)
            env2.set_curriculum_phase(phase)

            # Use frozen previous-cycle opponent (or current model on cycle 1)
            env1.set_other_model(lagged2 if lagged2 else model2)
            model1.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
            # EWC correction: pull important weights back toward Task-1 anchors
            ewc_loss1 = ewc1.apply_correction(model1, n_steps=5)
    
            env2.set_other_model(lagged1 if lagged1 else model1)
            model2.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
            ewc_loss2 = ewc2.apply_correction(model2, n_steps=5)

            # Freeze snapshots for NEXT cycle's opponent
            lagged1 = LaggedPolicy(model1)
            lagged2 = LaggedPolicy(model2)

            # ── Extract max episode reward (best run of this cycle) ────────────────
            def max_rew(model):
                buf = model.ep_info_buffer
                return np.max([ep["r"] for ep in buf]) if len(buf) > 0 else float("nan")

            r1 = max_rew(model1)
            r2 = max_rew(model2)

            # ── Record training episodes into episodic memory ──────────────────────
            # ep_info_buffer has {r: total_reward, l: episode_length, t: time}
            for ep in list(model1.ep_info_buffer):
                memory.record(total_reward=float(ep['r']), collisions=0,
                              fuel_outs=0, steps=int(ep['l']))
            for ep in list(model2.ep_info_buffer):
                memory.record(total_reward=float(ep['r']), collisions=0,
                              fuel_outs=0, steps=int(ep['l']))

            # ── Q-value proxy (V(s) from Critic) ─────────────────────────────────
            def get_mean_v(model, env):
                vs = []
                for _ in range(16):
                    obs, _ = env.reset()
                    obs_t = obs_as_tensor(
                        {"local":  obs["local"][None, :],
                         "global": obs["global"][None, :]},
                        model.device,
                    )
                    with torch.no_grad():
                        feats     = model.policy.extract_features(obs_t)
                        latent_vf = model.policy.mlp_extractor.forward_critic(feats)
                        v = float(model.policy.value_net(latent_vf).item())
                        vs.append(v)
                return np.mean(vs)

            v1 = get_mean_v(model1, env1)
            v2 = get_mean_v(model2, env2)

            steps_done = cycle * chunk_steps
            progress_data.append([cycle, steps_done, r1, r2, v1, v2])

            print(f"  {cycle:>5}  {r1:>+10.3f}  {r2:>+10.3f}  "
                  f"{v1:>+8.3f}  {v2:>+8.3f}  "
                  f"{len(memory.episodes):>13}  "
                  f"[{steps_done:>7,} / {total_timesteps:,} steps]")

            # ── Per-cycle save (crash-safe) ────────────────────────────────────────
            model1.save(MODEL1_PATH)
            model2.save(MODEL2_PATH)
            _save_arch_marker()
            memory.save()

    print(f"\n  Models saved → {MODEL1_PATH}.zip / {MODEL2_PATH}.zip")
    print(f"  Memory saved  → episodic_memory.pkl  ({memory.stats})")

    # ── Compute EWC Fisher matrices after Task-1 training ─────────────────────
    # This marks the "end of Task 1" checkpoint.  Fisher matrices will protect
    # this knowledge during any future Task-2 training.
    if not args.eval_only:
        print("\n" + "="*62)
        print("  COMPUTING EWC FISHER MATRICES (Task-1 checkpoint)")
        print("="*62)
        ewc1.compute_fisher(model1, env1, n_samples=EWC_N_SAMPLES)
        ewc2.compute_fisher(model2, env2, n_samples=EWC_N_SAMPLES)
        print(f"  [EWC] Fisher matrices saved → {EWC1_PATH}, {EWC2_PATH}")

    # ── Post-training evaluation ───────────────────────────────────────────────
    print("\n" + "="*62)
    print("  EVALUATION: Trained agents (10 episodes, last shown live)")
    print("="*62)

    t_coll, t_fuel, t_rew, final_env = run_evaluation(
        model1, model2, memory,
        num_episodes=10, ep_steps=360, render_last=True,
        )
    print_summary("TRAINED", t_coll, t_fuel, t_rew)

    # ── Before / After comparison ──────────────────────────────────────────────
    print("\n" + "="*62)
    print("  BEFORE vs AFTER CO-TRAINING")
    print("="*62)
    for i, name in enumerate(["Sat 1 (Red)", "Sat 2 (Orange)"]):
        b_r   = np.mean(b_rew[i])
        t_r   = np.mean(t_rew[i])
        delta = t_r - b_r
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {name}: {b_r:.1f} → {t_r:.1f}  ({arrow} {abs(delta):.1f})")

    # ── Export data for MATLAB ─────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  EXPORTING DATA TO MATLAB")
    print("="*62)

    export_dir = "matlab_export"
    os.makedirs(export_dir, exist_ok=True)

    # 1. reward_history.csv
    with open(os.path.join(export_dir, "reward_history.csv"), "w") as f:
        f.write("step,sat1_reward,sat2_reward,"
                "sat1_cumulative_reward,sat2_cumulative_reward\n")
        for s in range(len(final_env.reward_history[0])):
            f.write(
                f"{s+1},"
                f"{final_env.reward_history[0][s]:.4f},"
                f"{final_env.reward_history[1][s]:.4f},"
                f"{final_env.cumulative_reward_history[0][s]:.4f},"
                f"{final_env.cumulative_reward_history[1][s]:.4f}\n"
            )
    print(f"  ✓ Saved {export_dir}/reward_history.csv")

    # 2. positions_history.csv
    with open(os.path.join(export_dir, "positions_history.csv"), "w") as f:
        f.write("step,sat1_position,sat2_position,debris_positions\n")
        for s in range(len(final_env.agent_pos_history[0])):
            d_arr = final_env.debris_history[s]
            d_pos = [str(val) for val in d_arr]
            f.write(
                f"{s+1},"
                f"{final_env.agent_pos_history[0][s]},"
                f"{final_env.agent_pos_history[1][s]},"
                f"\"{','.join(d_pos)}\"\n"
            )
    print(f"  ✓ Saved {export_dir}/positions_history.csv")

    # 3. training_progress.csv  (includes V(s) proxy)
    with open(os.path.join(export_dir, "training_progress.csv"), "w") as f:
        f.write("cycle,steps_completed,sat1_reward_mean,sat2_reward_mean,"
                "sat1_V_estimate,sat2_V_estimate\n")
        for row in progress_data:
            f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.4f},"
                    f"{row[4]:.4f},{row[5]:.4f}\n")
    print(f"  ✓ Saved {export_dir}/training_progress.csv")
    print("  Done.")


if __name__ == "__main__":
    main()
