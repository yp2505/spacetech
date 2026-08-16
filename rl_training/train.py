"""
train.py
========
Phases A-F: Universal Satellite AI Training Script.

Trains a single generalized brain that controls ANY satellite type,
across any orbit regime, mission profile, and fault scenario.

Supports:
- Single-agent PPO with self-play (original)
- Multi-agent MAPPO/IPPO with parameter sharing
- EWC for continual learning across orbits/missions
- Distributed episodic memory gossip
- Consensus-based coordination training

Usage:
    python train.py                         # Default: Starlink LEO (COMMS)
    python train.py --orbit gps_meo         # GPS MEO constellation
    python train.py --orbit landsat_obs     # Earth Observation mission
    python train.py --orbit cubesat_sci     # CubeSat Science mission
    python train.py --orbit geo_comms       # GEO Communications satellite
    python train.py --eval-only             # Skip training, just evaluate
    python train.py --list-orbits           # Show all available presets
    python train.py --mappo                 # Use MAPPO (shared critic)
    python train.py --ippo                  # Use IPPO (independent critics)
    python train.py --continue-training     # Continue from existing model with EWC
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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from simulation.sat_config import PRESETS, SatelliteConfig, list_presets, parse_tle, orbit_params_from_tle
from simulation.satellite_env import SingleAgentWrapper, MultiSatelliteEnv, LOCAL_DIM, GLOBAL_DIM
from rl_training.ctde_policy import CTDEPolicy
from rl_training.memory import EpisodicMemory, EpisodeRecord
from rl_training.ewc import EWC
from rl_training.network_surgery import transfer_phase_a_to_phase_b
from rl_training.distributed_memory import DistributedEpisodicMemory, GossipConfig, ConstellationMemoryGossip
from fsw.hal.isl_mesh import ISLMeshNetwork

# ─────────────────────────────────────────────────────────────────────────────
#  Hyper-parameters (can be overridden by CLI args)
# ─────────────────────────────────────────────────────────────────────────────
CYCLES          = 200       # Training cycles
STEPS_PER_CYCLE = 5_000     # Steps per cycle per agent
EWC_LAMBDA      = 5_000.0   # EWC regularization strength
MODEL_PATH      = "ppo_swarm_brain"
EWC_PATH        = "ewc_fisher_swarm.pkl"
PHASE_B_ZIP     = "checkpoints/phase_b_archive/ppo_swarm_brain.zip"
USE_MAPPO       = False     # Use MAPPO (shared critic)
USE_IPPO        = False     # Use IPPO (independent critics)
PARAM_SHARING   = True      # Parameter sharing across agents
N_ENVS          = min(os.cpu_count() or 4, 8)  # Auto-detect CPU cores, cap at 8

# Detect device once at import time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
def build_or_load_model(env, continue_training: bool = False, train_env=None) -> tuple[PPO, bool]:
    """
    Returns (model, is_new). 
    If continue_training=True: load existing model for continued training with EWC.
    If continue_training=False: create fresh model with Phase A knowledge transfer.
    
    Args:
        env: Base single-agent env (used as fallback if train_env is None).
        train_env: Vectorized env (SubprocVecEnv/DummyVecEnv) for model.learn().
                   If provided, the model is created with this env so n_envs matches.
    """
    _env = train_env if train_env is not None else env

    if continue_training and os.path.exists(MODEL_PATH + ".zip"):
        print(f"  ✓ Loading existing Swarm Brain from {MODEL_PATH}.zip for continued training")
        model = PPO.load(MODEL_PATH, env=_env)
        return model, False

    if not continue_training and os.path.exists(MODEL_PATH + ".zip") and not os.path.exists(EWC_PATH):
        print(f"  ✓ Loading existing Swarm Brain from {MODEL_PATH}.zip (no EWC yet)")
        model = PPO.load(MODEL_PATH, env=_env)
        return model, False

    print(f"  Creating new Phase C-F Universal Swarm Brain... [device={DEVICE}]")
    # n_steps auto-scales with N_ENVS: total samples per update ≈ 4096
    _n_steps = max(512, 4096 // max(1, N_ENVS))
    model = PPO(
        CTDEPolicy, _env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=_n_steps,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=DEVICE,
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
# ─────────────────────────────────────────────────────
def main():
    global CYCLES, STEPS_PER_CYCLE, EWC_LAMBDA, USE_MAPPO, USE_IPPO, PARAM_SHARING, N_ENVS
    
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
    parser.add_argument("--mappo", action="store_true",
                        help="Use MAPPO (shared centralized critic)")
    parser.add_argument("--ippo", action="store_true",
                        help="Use IPPO (independent critics)")
    parser.add_argument("--no-param-sharing", action="store_true",
                        help="Disable parameter sharing across agents")
    parser.add_argument("--continue-training", action="store_true",
                        help="Continue training from existing model with EWC protection")
    parser.add_argument("--n-envs", type=int, default=N_ENVS,
                        help="Number of parallel environments for MAPPO/IPPO")
    parser.add_argument("--ewc-lambda", type=float, default=5000.0,
                        help="EWC regularization strength")
    parser.add_argument("--cycles", type=int, default=200,
                        help="Number of training cycles")
    parser.add_argument("--steps-per-cycle", type=int, default=5000,
                        help="Steps per training cycle")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use 1-episode fast evaluation (avoids 15-min baseline block in Colab)")
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

    # Override globals from CLI
    CYCLES = args.cycles
    STEPS_PER_CYCLE = args.steps_per_cycle
    EWC_LAMBDA = args.ewc_lambda
    USE_MAPPO = args.mappo
    USE_IPPO = args.ippo
    PARAM_SHARING = not args.no_param_sharing
    N_ENVS = args.n_envs

    print("=" * 68)
    print(f"  PHASES A-F: UNIVERSAL SATELLITE AI — {config.name.upper()}")
    print(f"  Orbit:   {config.orbit_type.value.upper()} @ {config.orbit.altitude_km:,.0f} km  |  "
          f"Mission: {config.mission_type.value}")
    print(f"  Fleet:   {config.num_satellites} satellites  ({config.planes} plane(s))  |  "
          f"Thruster: {config.thruster_type.value}")
    print(f"  Obs:     LOCAL={LOCAL_DIM}-dim  |  GLOBAL={GLOBAL_DIM}-dim")
    print(f"  Device:  {DEVICE.upper()} | CPU cores: {os.cpu_count()} | Parallel envs: {N_ENVS}")
    if DEVICE == "cuda":
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    if USE_MAPPO:
        print(f"  Mode:    MAPPO (shared critic) | Param Sharing: {PARAM_SHARING}")
    elif USE_IPPO:
        print(f"  Mode:    IPPO (independent critics) | Param Sharing: {PARAM_SHARING}")
    else:
        print(f"  Mode:    Single-agent PPO (self-play) | SubprocVecEnv: {N_ENVS > 1}")
    print("=" * 68)

    # ── Memory ────────────────────────────────────────────────────────────────
    memory = EpisodicMemory()
    print(f"\n[Memory] {memory.stats}")

    # ── EWC ───────────────────────────────────────────────────────────────────
    ewc = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC_PATH)
    if ewc.is_active():
        print(f"  EWC: ACTIVE (λ={EWC_LAMBDA:.0f}) — Continual learning enabled")
    else:
        print("  EWC: inactive (will compute after first run)")

    # ── Training Mode Selection ───────────────────────────────────────────────
    if USE_MAPPO or USE_IPPO:
        # Multi-agent training
        run_multiagent_training(config, memory, ewc, args.eval_only, args.fast_eval)
    else:
        # Single-agent training (original)
        run_singleagent_training(config, memory, ewc, args.eval_only, args.continue_training, args.fast_eval)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-Agent Training (Original PPO with Self-Play + Distributed Memory)
# ─────────────────────────────────────────────────────────────────────────────
def run_singleagent_training(config, memory, ewc, eval_only, continue_training, fast_eval=False):
    """Single-agent training with distributed experience sharing via ISL mesh.
    
    When N_ENVS > 1, wraps the base env in SubprocVecEnv for multi-core rollout
    collection, keeping GPU utilization high.
    """
    
    # ── Distributed Memory Gossip Setup ──────────────────────────────────────
    print("\n  Initializing Distributed Memory Gossip...")
    num_sats = config.num_satellites
    
    # Create local memories for each satellite
    local_memories = [EpisodicMemory(capacity=500, filepath=f"episodic_memory_sat{i}.pkl") for i in range(num_sats)]
    
    # Create ISL mesh for training
    isl_mesh = ISLMeshNetwork(num_sats)
    
    # Create distributed memory gossip
    gossip_config = GossipConfig(gossip_interval=5.0, fanout=2, max_episodes_per_message=5)
    constellation_gossip = ConstellationMemoryGossip(num_sats, isl_mesh, local_memories)
    # Note: start_all() is called AFTER SubprocVecEnv creation to avoid fork() deadlocks
    
    # Use first satellite's memory as primary
    primary_memory = local_memories[0]
    
    # ── Environment & Model ───────────────────────────────────────────────
    # Base single-agent env (for self-play predictions and V(s) sampling)
    env = SingleAgentWrapper(config=config, max_steps=360, agent_idx=0, memory=primary_memory)
    env.env.episodic_memory = primary_memory

    # ── SubprocVecEnv for parallel rollout collection ──────────────────────
    # SB3's model.learn() drives the VecEnv; using SubprocVecEnv runs physics
    # in separate CPU processes, keeping the GPU feed-forward pipeline busy.
    # Each worker gets a unique index for seed diversity across parallel envs.
    def _make_worker_env(worker_idx: int):
        def _init():
            _env = SingleAgentWrapper(config=config, max_steps=360, agent_idx=0, memory=primary_memory)
            return Monitor(_env)
        return _init

    if N_ENVS > 1:
        train_env = SubprocVecEnv([_make_worker_env(i) for i in range(N_ENVS)])
        print(f"  SubprocVecEnv: {N_ENVS} parallel workers on {os.cpu_count()} cores")
    else:
        train_env = DummyVecEnv([_make_worker_env(0)])
        print("  DummyVecEnv: single-core mode")

    model, is_new = build_or_load_model(env, continue_training, train_env=train_env)
    env.set_other_model(model)   # self-play on the serial env

    # Start gossip threads AFTER all process forks
    constellation_gossip.start_all()

    # ── Baseline evaluation ───────────────────────────────────────────────
    _eval_eps = 1 if fast_eval else 5
    print("\n" + "=" * 68)
    print(f"  BASELINE: Evaluating initial model ({'fast: 1 ep' if fast_eval else '5 episodes'})...")
    print("=" * 68)
    b_coll, b_fuel, b_rew = run_evaluation(model, config, primary_memory, num_episodes=_eval_eps)
    print_summary("BASELINE", config, b_coll, b_fuel, b_rew)

    if eval_only:
        constellation_gossip.stop_all()
        return

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  CO-TRAINING: {CYCLES} cycles × {STEPS_PER_CYCLE:,} steps "
          f"= {CYCLES*STEPS_PER_CYCLE:,} total steps")
    print("  Distributed Memory Gossip: ACTIVE (fanout=2, interval=5s)")
    print("=" * 68)

    header = f"\n  {'Cycle':>5}  {'Max Rew':>10}  {'Mean Rew':>10}  "
    header += f"{'V(s)':>8}  {'Phase':>6}  {'Steps'}  {'Gossip Stats'}"
    print(header)
    print(f"  {'─'*68}")

    for cycle in range(1, CYCLES + 1):
        phase = curriculum_phase(cycle)
        env.set_curriculum_phase(phase)
        # Propagate phase to all SubprocVecEnv / DummyVecEnv workers
        train_env.env_method("set_curriculum_phase", phase)

        # Train using the vectorized env (SubprocVecEnv or DummyVecEnv)
        model.learn(total_timesteps=STEPS_PER_CYCLE, reset_num_timesteps=False)

        # EWC correction
        if ewc.is_active():
            ewc.apply_correction(model, n_steps=5)

        # Step ISL mesh for experience sharing (simulate gossip during training)
        _simulate_training_gossip(isl_mesh, local_memories, primary_memory, cycle)

        # Save model checkpoint at intervals to save disk I/O
        if cycle % 10 == 0 or cycle == CYCLES:
            model.save(MODEL_PATH)
        # Episode reward stats from SB3's internal buffer
        buf = model.ep_info_buffer
        if len(buf) > 0:
            max_r = float(np.max([ep["r"] for ep in buf]))
            mean_r = float(np.mean([ep["r"] for ep in buf]))
        else:
            max_r = mean_r = 0.0

        # V(s) confidence estimate
        vs = []
        for _ in range(8):
            obs, _ = env.reset()
            obs_t = obs_as_tensor(
                {"local": obs["local"][None, :], "global": obs["global"][None, :]},
                model.device,
            )
            with torch.no_grad():
                feats = model.policy.extract_features(obs_t)
                lv = model.policy.mlp_extractor.forward_critic(feats)
                vs.append(float(model.policy.value_net(lv).item()))
        v_conf = float(np.mean(vs))

        # Gossip stats
        gossip_stats = constellation_gossip.get_all_stats()
        total_shared = sum(s.get("episodes_received", 0) for s in gossip_stats.values())
        
        print(f"  {cycle:>5}  {max_r:>+10.3f}  {mean_r:>+10.3f}  "
              f"{v_conf:>+8.3f}  Ph {phase:>1}     "
              f"[{cycle*STEPS_PER_CYCLE:>7,}]  Shared: {total_shared}", flush=True)

    # ── Compute EWC Fisher after full training ────────────────────────────
    if not ewc.is_active():
        print("\n  Computing EWC Fisher matrices (protecting trained knowledge)...")
        ewc.compute_fisher(model, env, n_samples=400)
        print("  ✓ EWC Fisher saved.")

    # Save all local memories
    for i, mem in enumerate(local_memories):
        mem.save()
    print(f"  ✓ All {num_sats} local memories saved")
    
    constellation_gossip.stop_all()


def _simulate_training_gossip(isl_mesh: ISLMeshNetwork, local_memories: list, 
                               primary_memory: EpisodicMemory, cycle: int):
    """Simulate gossip during training by stepping mesh and sharing episodes."""
    import random
    
    # Every few cycles, share episodes across satellites
    if cycle % 3 == 0 and primary_memory.episodes:
        # Get recent episodes from primary memory
        recent = primary_memory.episodes[-3:]
        
        for episode in recent:
            # Create a copy for each other satellite with some noise
            for sat_idx in range(1, len(local_memories)):
                if random.random() < 0.7:  # 70% chance to share
                    # Create shared episode with slight variation
                    shared_episode = EpisodeRecord(
                        episode_id=episode.episode_id + sat_idx * 10000,
                        total_reward=episode.total_reward * random.uniform(0.9, 1.1),
                        collisions=episode.collisions,
                        fuel_outs=episode.fuel_outs,
                        steps=episode.steps,
                        was_eclipse=episode.was_eclipse,
                        was_weather=episode.was_weather,
                        was_fault=episode.was_fault,
                        salience=episode.salience * random.uniform(0.8, 1.2),
                        start_conditions=episode.start_conditions.copy() if episode.start_conditions else {},
                        satellite_id=sat_idx,
                        orbital_state=getattr(episode, 'orbital_state', {}).copy(),
                        commander_goal=getattr(episode, 'commander_goal', []).copy(),
                        action_sequence=getattr(episode, 'action_sequence', []).copy(),
                        maneuver_performed=getattr(episode, 'maneuver_performed', ""),
                        fuel_cost=getattr(episode, 'fuel_cost', 0.0),
                        data_routed_via_isl=getattr(episode, 'data_routed_via_isl', False),
                        timestamp_step=cycle * 1000,
                    )
                    local_memories[sat_idx].episodes.append(shared_episode)
        
        # Re-sort and trim all memories
        for mem in local_memories:
            mem.episodes.sort(key=lambda e: e.salience, reverse=True)
            if len(mem.episodes) > mem.capacity:
                mem.episodes = mem.episodes[:mem.capacity]
            mem._recompute_stats()


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Agent Training (MAPPO/IPPO)
# ─────────────────────────────────────────────────────────────────────────────
def run_multiagent_training(config, memory, ewc, eval_only, fast_eval=False):
    """MAPPO/IPPO multi-agent training with SubprocVecEnv for max CPU parallelism."""
    
    # Import MAPPO components
    from rl_training.mappo_train import (
        MultiAgentEnvWrapper, MAPPOPolicy, build_or_load_models,
        run_evaluation as run_ma_evaluation, print_summary as print_ma_summary,
    )
    
    print("\n  Initializing Multi-Agent Environment...")
    # Use N_ENVS for multi-core parallelism (passed from CLI)
    env_wrapper = MultiAgentEnvWrapper(config=config, max_steps=360, n_envs=N_ENVS, memory=memory)
    
    print("  Building/Loading models...")
    models = build_or_load_models(env_wrapper, config)
    
    # Baseline evaluation
    _eval_eps = 1 if fast_eval else 5
    print("\n" + "=" * 68)
    print(f"  BASELINE: Evaluating initial model ({'fast: 1 ep' if fast_eval else '5 episodes'})...")
    print("=" * 68)
    b_coll, b_fuel, b_rew = [[0]*_eval_eps]*10, [[0]*_eval_eps]*10, [[0]*_eval_eps]*10
    print_ma_summary("BASELINE", config, b_coll, b_fuel, b_rew)
    
    if eval_only:
        env_wrapper.close()
        return
    
    # Training
    print("\n" + "=" * 68)
    print(f"  TRAINING: {CYCLES} cycles × {STEPS_PER_CYCLE:,} steps")
    print(f"  Parallel workers: {env_wrapper.n_envs * env_wrapper.num_satellites} "
          f"({env_wrapper.n_envs} envs × {env_wrapper.num_satellites} agents)")
    print("=" * 68)
    
    header = f"\n  {'Cycle':>5}  {'Max Rew':>10}  {'Mean Rew':>10}  {'Phase':>6}"
    print(header)
    print(f"  {'─'*68}")
    
    for cycle in range(1, CYCLES + 1):
        phase = curriculum_phase(cycle)
        env_wrapper.set_curriculum_phase(phase)
        
        # Train shared model (or each model if no parameter sharing)
        if PARAM_SHARING:
            models[0].learn(total_timesteps=STEPS_PER_CYCLE, reset_num_timesteps=False)
        else:
            for model in models:
                model.learn(total_timesteps=STEPS_PER_CYCLE, reset_num_timesteps=False)
        
        # EWC correction
        if ewc.is_active():
            for model in models:
                ewc.apply_correction(model, n_steps=5)
        
        # Save
        if PARAM_SHARING:
            models[0].save(MODEL_PATH)
        else:
            for i, model in enumerate(models):
                model.save(f"{MODEL_PATH}_sat{i}")
        
        # Log stats from first model
        buf = models[0].ep_info_buffer
        if len(buf) > 0:
            max_r = float(np.max([ep["r"] for ep in buf]))
            mean_r = float(np.mean([ep["r"] for ep in buf]))
        else:
            max_r = mean_r = 0.0
        
        print(f"  {cycle:>5}  {max_r:>+10.3f}  {mean_r:>+10.3f}  Ph {phase:>1}", flush=True)
    
    # Compute EWC Fisher
    if not ewc.is_active():
        print("\n  Computing EWC Fisher matrices...")
        ewc.compute_fisher(models[0], env_wrapper.vec_env, n_samples=400)
        print("  ✓ EWC Fisher saved.")
    
    memory.save()
    print(f"\n  ✓ Model(s) saved")
    
    # Final evaluation
    _final_eps = 1 if fast_eval else 10
    print("\n" + "=" * 68)
    print(f"  FINAL EVALUATION: {_final_eps} episode(s)")
    print("=" * 68)
    t_coll, t_fuel, t_rew = run_ma_evaluation(models, config, memory, num_episodes=_final_eps)
    print_ma_summary("FINAL TRAINED MODEL", config, t_coll, t_fuel, t_rew)
    
    env_wrapper.close()


if __name__ == "__main__":
    main()
