"""
train.py
========
Phases A-F: Universal Satellite AI Training Script.

Trains a single generalized brain that controls ANY satellite type,
across any orbit regime, mission profile, and fault scenario.

Usage:
    python train.py                         # Default: Starlink LEO (COMMS)
    python train.py --orbit gps_meo         # GPS MEO constellation
    python train.py --orbit landsat_obs     # Earth Observation mission
    python train.py --orbit cubesat_sci     # CubeSat Science mission
    python train.py --orbit geo_comms       # GEO Communications satellite
    python train.py --eval-only             # Skip training, just evaluate
    python train.py --list-orbits           # Show all available presets
"""

import os
import sys
import argparse
from dataclasses import replace
import numpy as np
import torch

# Ensure the parent directory is in the Python path so it finds 'simulation' and 'fsw'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from simulation.sat_config import PRESETS, SatelliteConfig, list_presets, parse_tle, orbit_params_from_tle
from simulation.satellite_env import SingleAgentWrapper, MultiSatelliteEnv, LOCAL_DIM, GLOBAL_DIM
from rl_training.ctde_policy import CTDEPolicy
from rl_training.memory import EpisodicMemory
from rl_training.ewc import EWC
from rl_training.network_surgery import transfer_phase_a_to_phase_b

# ─────────────────────────────────────────────────────────────────────────────
#  Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
CYCLES          = 200       # Training cycles
STEPS_PER_CYCLE = 5_000     # Steps per cycle per agent
EWC_LAMBDA      = 5_000.0   # EWC regularization strength
MODEL_PATH      = "ppo_swarm_brain"
EWC_PATH        = "ewc_fisher_swarm.pkl"
PHASE_B_ZIP     = "checkpoints/phase_b_archive/ppo_swarm_brain.zip"


# ─────────────────────────────────────────────────────────────────────────────
#  Curriculum schedule
# ─────────────────────────────────────────────────────────────────────────────
def curriculum_phase(cycle: int) -> int:
    if cycle <= 10:  return 1   # Basics: clean environment
    if cycle <= 30:  return 2   # Add eclipses + domain randomization
    if cycle <= 60:  return 3   # Full physics + debris + space weather
    if cycle <= 90:  return 4   # Adversarial: worst-case scenarios
    if cycle <= 130: return 5   # Phase 5: Thermal Physics constraints
    return 6                     # Phase 6: Full fault injection (wheel/thruster/sensor)


# ─────────────────────────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────────────────────────
def build_or_load_model(env: SingleAgentWrapper) -> tuple[PPO, bool]:
    """
    Returns (model, is_new). If a saved brain exists, load it.
    If not, create fresh and transfer Phase A knowledge into it.
    """
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"  ✓ Loading existing Swarm Brain from {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=env)
        return model, False

    print("  Creating new Phase C-F Universal Swarm Brain...")
    model = PPO(
        CTDEPolicy, env,
        verbose        = 0,
        learning_rate  = 3e-4,
        n_steps        = 2048,
        batch_size     = 256,
        n_epochs       = 10,
        gamma          = 0.995,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.01,
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
    )

    # Transfer Phase B hidden-layer knowledge into Phase C-F
    transfer_phase_a_to_phase_b(PHASE_B_ZIP, model)
    return model, True


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    model: PPO,
    config: SatelliteConfig,
    memory: EpisodicMemory,
    num_episodes: int = 10,
    ep_steps: int = 360,
) -> tuple:
    n = config.num_satellites
    collisions_ep  = [[] for _ in range(n)]
    fuel_outs_ep   = [[] for _ in range(n)]
    rewards_ep     = [[] for _ in range(n)]

    for ep in range(num_episodes):
        env = MultiSatelliteEnv(config=config, max_steps=ep_steps)
        env.curriculum_phase = 6   # Always evaluate on full Phase 6 (faults + thermal)
        obs_raw, _ = env.reset()

        ctx = memory.get_context() if memory else np.zeros(4, dtype=np.float32)

        for _ in range(ep_steps):
            actions = []
            for i in range(n):
                local = np.concatenate([obs_raw[i]["local"], ctx]).astype(np.float32)
                obs_d = {"local": local, "global": obs_raw[i]["global"]}
                a, _  = model.predict(obs_d, deterministic=True)
                actions.append(a)
            obs_raw, _, term, trunc, _ = env.step(np.array(actions))
            if term or trunc:
                break

        for i in range(n):
            collisions_ep[i].append(env.collisions[i])
            fuel_outs_ep[i].append(env.fuel_outs[i])
            rewards_ep[i].append(float(np.sum(env.reward_history[i])))
        if ep == 0:
            memory.record(
                total_reward=float(np.mean([np.sum(env.reward_history[i]) for i in range(n)])),
                collisions=int(np.sum([env.collisions[i] for i in range(n)])),
                fuel_outs=int(np.sum([env.fuel_outs[i] for i in range(n)])),
                steps=env.current_step,
                was_eclipse=bool(np.any(env.eclipse_mode)),
                was_weather=env.space_weather_active,
                was_fault=bool(np.any(env.faults_logged > 0)),
            )

    return collisions_ep, fuel_outs_ep, rewards_ep


def print_summary(label: str, config: SatelliteConfig, collisions, fuel_outs, rewards):
    n = config.num_satellites
    W = 68
    print(f"\n{'─'*W}")
    print(f"  {label} — {len(rewards[0])}-Episode Evaluation  [{config.name}]")
    print(f"{'─'*W}")
    print(f"  {'Satellite':<18} {'Collisions':>10} {'Fuel Outs':>10} {'Avg Total Rew':>15}")
    print(f"  {'─'*(W-2)}")
    for i in range(min(n, 6)):   # Print up to 6 satellites
        print(f"  Sat {i:<14} {np.mean(collisions[i]):>10.2f} "
              f"{np.mean(fuel_outs[i]):>10.2f} {np.mean(rewards[i]):>15.2f}")
    if n > 6:
        print(f"  ... [{n-6} more satellites not shown]")
    print(f"  {'Fleet Average':<18} "
          f"{'':>10} {'':>10} "
          f"{np.mean([np.mean(r) for r in rewards]):>15.2f}")
    print(f"{'─'*W}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Universal Satellite AI — Phases A-F Training")
    parser.add_argument("--orbit",       type=str, default="starlink_leo",
                        help="Satellite preset key (use --list-orbits to see options)")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Skip training, only run evaluation")
    parser.add_argument("--list-orbits", action="store_true",
                        help="List all satellite presets and exit")
    parser.add_argument("--tle-file", type=str,
                        help="Optional two- or three-line NORAD TLE file; replaces the preset orbit")
    parser.add_argument("--blackout", action="append", default=[], metavar="START:END[:STATION]",
                        help="Disable one ground station (or all stations) for a step range; repeatable")
    args = parser.parse_args()

    if args.list_orbits:
        list_presets()
        return

    if args.orbit not in PRESETS:
        print(f"Unknown orbit preset '{args.orbit}'. Use --list-orbits to see options.")
        return

    config = PRESETS[args.orbit]
    if args.tle_file:
        try:
            with open(args.tle_file, encoding="utf-8") as tle_file:
                tle = parse_tle(tle_file.read())
            config = replace(
                config, name=f"{config.name} / {tle.name}", tle=tle,
                orbit_params=orbit_params_from_tle(tle), inclination_deg=tle.inclination_deg,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"Could not load --tle-file: {exc}")

    if args.blackout:
        blackouts = []
        for value in args.blackout:
            try:
                parts = value.split(":", 2)
                start, end = int(parts[0]), int(parts[1])
                station = parts[2] if len(parts) == 3 and parts[2] else None
                blackouts.append((start, end, station))
            except (ValueError, IndexError):
                parser.error("--blackout must be START:END[:STATION], for example 30:90:Hawaii")
        try:
            config = replace(config, ground_station_blackouts=tuple(blackouts))
        except ValueError as exc:
            parser.error(f"Invalid blackout: {exc}")

    print("=" * 68)
    print(f"  PHASES A-F: UNIVERSAL SATELLITE AI — {config.name.upper()}")
    print(f"  Orbit:   {config.orbit_type.value.upper()} @ {config.orbit.altitude_km:,.0f} km  |  "
          f"Mission: {config.mission_type.value}")
    print(f"  Fleet:   {config.num_satellites} satellites  ({config.planes} plane(s))  |  "
          f"Thruster: {config.thruster_type.value}")
    print(f"  Obs:     LOCAL={LOCAL_DIM}-dim  |  GLOBAL={GLOBAL_DIM}-dim")
    print("=" * 68)

    # ── Memory ────────────────────────────────────────────────────────────────
    memory = EpisodicMemory()
    print(f"\n[Memory] {memory.stats}")

    # ── Environment & Model ───────────────────────────────────────────────────
    env   = SingleAgentWrapper(config=config, max_steps=360, agent_idx=0, memory=memory)
    model, is_new = build_or_load_model(env)
    env.set_other_model(model)   # self-play

    # ── EWC ───────────────────────────────────────────────────────────────────
    ewc = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC_PATH)
    if ewc.is_active():
        print(f"  EWC: ACTIVE (λ={EWC_LAMBDA:.0f})")
    else:
        print("  EWC: inactive (will compute after first run)")

    # ── Baseline evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  BASELINE: Evaluating initial model (5 episodes)...")
    print("=" * 68)
    b_coll, b_fuel, b_rew = run_evaluation(model, config, memory, num_episodes=5)
    print_summary("BASELINE", config, b_coll, b_fuel, b_rew)

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.eval_only:
        print("\n" + "=" * 68)
        print(f"  CO-TRAINING: {CYCLES} cycles × {STEPS_PER_CYCLE:,} steps "
              f"= {CYCLES*STEPS_PER_CYCLE:,} total steps")
        print("=" * 68)

        header  = f"\n  {'Cycle':>5}  {'Max Rew':>10}  {'Mean Rew':>10}  "
        header += f"{'V(s)':>8}  {'Phase':>6}  {'Steps'}"
        print(header)
        print(f"  {'─'*68}")

        for cycle in range(1, CYCLES + 1):
            phase = curriculum_phase(cycle)
            env.set_curriculum_phase(phase)

            model.learn(total_timesteps=STEPS_PER_CYCLE, reset_num_timesteps=False)

            if ewc.is_active():
                ewc.apply_correction(model, n_steps=5)

            model.save(MODEL_PATH)

            # Episode reward stats from SB3's internal buffer
            buf = model.ep_info_buffer
            if len(buf) > 0:
                max_r  = float(np.max([ep["r"] for ep in buf]))
                mean_r = float(np.mean([ep["r"] for ep in buf]))
            else:
                max_r = mean_r = 0.0

            # V(s) confidence estimate
            vs = []
            for _ in range(8):
                obs, _ = env.reset()
                obs_t  = obs_as_tensor(
                    {"local":  obs["local"][None, :],
                     "global": obs["global"][None, :]},
                    model.device,
                )
                with torch.no_grad():
                    feats = model.policy.extract_features(obs_t)
                    lv    = model.policy.mlp_extractor.forward_critic(feats)
                    vs.append(float(model.policy.value_net(lv).item()))
            v_conf = float(np.mean(vs))

            print(f"  {cycle:>5}  {max_r:>+10.3f}  {mean_r:>+10.3f}  "
                  f"{v_conf:>+8.3f}  Ph {phase:>1}     "
                  f"[{cycle*STEPS_PER_CYCLE:>7,}]")

        # ── Compute EWC Fisher after full training ────────────────────────────
        if not ewc.is_active():
            print("\n  Computing EWC Fisher matrices (protecting trained knowledge)...")
            ewc.compute_fisher(model, env, n_samples=400)
            print("  ✓ EWC Fisher saved.")

        memory.save()
        print(f"\n  ✓ Model saved → {MODEL_PATH}.zip")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  FINAL EVALUATION: 10 episodes (adversarial Phase 4)")
    print("=" * 68)
    t_coll, t_fuel, t_rew = run_evaluation(model, config, memory, num_episodes=10)
    print_summary("FINAL TRAINED MODEL", config, t_coll, t_fuel, t_rew)

    # ── Export to CSV ─────────────────────────────────────────────────────────
    export_dir = "outputs"
    os.makedirs(export_dir, exist_ok=True)

    # Quick standalone evaluation for CSV export
    eval_env = MultiSatelliteEnv(config=config, max_steps=360)
    eval_env.curriculum_phase = 6
    obs_raw, _ = eval_env.reset()
    ctx = memory.get_context()
    for _ in range(360):
        actions = []
        for i in range(config.num_satellites):
            local = np.concatenate([obs_raw[i]["local"], ctx]).astype(np.float32)
            a, _  = model.predict({"local": local, "global": obs_raw[i]["global"]},
                                   deterministic=True)
            actions.append(a)
        obs_raw, _, term, trunc, _ = eval_env.step(np.array(actions))
        if term or trunc:
            break

    csv_path = os.path.join(export_dir, "reward_history.csv")
    with open(csv_path, "w") as f:
        headers = ["step"] + [f"sat{i}_reward" for i in range(config.num_satellites)]
        f.write(",".join(headers) + "\n")
        for s in range(len(eval_env.reward_history[0])):
            row = [str(s + 1)] + [
                f"{eval_env.reward_history[i][s]:.4f}"
                for i in range(config.num_satellites)
            ]
            f.write(",".join(row) + "\n")
    print(f"\n  ✓ Exported reward history → {csv_path}")


if __name__ == "__main__":
    main()
