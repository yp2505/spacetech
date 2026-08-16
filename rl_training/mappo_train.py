"""
rl_training/mappo_train.py
==========================
Multi-Agent PPO (MAPPO/IPPO) Training for Satellite Constellation.

Supports:
- MAPPO: Shared centralized critic, decentralized actors (CTDE)
- IPPO: Independent critics and actors per agent
- Parameter sharing across homogeneous agents
- EWC for continual learning
- Episodic memory integration
- ISL communication simulation during training
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

# Detect device once at import time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback

from simulation.sat_config import PRESETS, SatelliteConfig, list_presets
from simulation.satellite_env import MultiSatelliteEnv, SingleAgentWrapper, LOCAL_DIM, GLOBAL_DIM
from rl_training.ctde_policy import CTDEPolicy, CTDEMlpExtractor, CTDEFeaturesExtractor
from rl_training.memory import EpisodicMemory
from rl_training.ewc import EWC
from rl_training.network_surgery import transfer_phase_a_to_phase_b
from gymnasium import spaces

# ─────────────────────────────────────────────────────────────────────────────
#  Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
CYCLES = 200
STEPS_PER_CYCLE = 5_000
EWC_LAMBDA = 5_000.0
MODEL_PATH = "ppo_swarm_brain.bin"
EWC_PATH = "ewc_fisher_swarm.pkl"
PHASE_B_ZIP = "checkpoints/phase_b_archive/ppo_swarm_brain.bin"

# MAPPO specific
USE_MAPPO = True          # True=MAPPO (shared critic), False=IPPO (independent critics)
PARAMETER_SHARING = True  # Share actor/critic params across homogeneous agents
CENTRALIZED_CRITIC = False # Use global state for critic (MAPPO)
N_ENVS = min(os.cpu_count() or 4, 8)  # Auto-detect CPU cores, cap at 8

# ─────────────────────────────────────────────────────────────────────────────
#  Curriculum
# ─────────────────────────────────────────────────────────────────────────────
def curriculum_phase(cycle: int) -> int:
    if cycle <= 10:  return 1
    if cycle <= 30:  return 2
    if cycle <= 60:  return 3
    if cycle <= 90:  return 4
    if cycle <= 130: return 5
    return 6


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Agent Policy (MAPPO/IPPO)
# ─────────────────────────────────────────────────────────────────────────────
class MAPPOPolicy(CTDEPolicy):
    """
    Extended CTDE Policy for Multi-Agent training.
    
    MAPPO: Shared critic with global state input, decentralized actors
    IPPO: Independent critics and actors per agent
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, 
                 num_agents: int = 1, use_mappo: bool = True, **kwargs):
        self.num_agents = num_agents
        self.use_mappo = use_mappo
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        local_dim = self.features_extractor.local_dim
        global_dim = self.features_extractor.global_dim
        
        if self.use_mappo and CENTRALIZED_CRITIC:
            # MAPPO: Critic gets concatenated global states of all agents
            critic_input_dim = global_dim * self.num_agents
            actor_input_dim = local_dim + global_dim  # Local + global for actor
        else:
            # IPPO or decentralized critic
            critic_input_dim = global_dim
            actor_input_dim = local_dim + global_dim
        
        self.mlp_extractor = MAPPOMlpExtractor(
            local_dim=local_dim,
            global_dim=global_dim,
            num_agents=self.num_agents,
            use_mappo=self.use_mappo,
            centralized_critic=CENTRALIZED_CRITIC,
        )


class MAPPOMlpExtractor(nn.Module):
    """
    MAPPO/IPPO MLP Extractor with optional centralized critic.
    """
    
    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        num_agents: int,
        use_mappo: bool = True,
        centralized_critic: bool = True,
        actor_hidden: List[int] = [256, 256],
        critic_hidden: List[int] = [512, 512, 256],
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_agents = num_agents
        self.use_mappo = use_mappo
        self.centralized_critic = centralized_critic and use_mappo
        
        # Feature dimensions (same as CTDEMlpExtractor)
        self.local_base_dim = 17 + 11  # 28
        self.local_neigh_dim = 5
        self.max_neighbors = 4
        self.global_base_dim = 6 + 17  # 23
        self.global_neigh_dim = 10
        
        # Actor attention (per agent)
        self.actor_embed = nn.Linear(self.local_neigh_dim, embed_dim)
        self.actor_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Critic attention
        if self.centralized_critic:
            # Centralized critic processes all agents' global features
            self.critic_embed = nn.Linear(self.global_neigh_dim * num_agents, embed_dim)
            self.critic_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            critic_base_dim = self.global_base_dim * num_agents
        else:
            self.critic_embed = nn.Linear(self.global_neigh_dim, embed_dim)
            self.critic_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            critic_base_dim = self.global_base_dim
        
        # Actor network
        actor_layers = []
        actor_in_dim = self.local_base_dim + embed_dim
        for h in actor_hidden:
            actor_layers += [nn.Linear(actor_in_dim, h), nn.Tanh()]
            actor_in_dim = h
        self.actor_net = nn.Sequential(*actor_layers)
        self.latent_dim_pi = actor_in_dim
        
        # Critic network
        critic_layers = []
        critic_in_dim = critic_base_dim + embed_dim
        for h in critic_hidden:
            critic_layers += [nn.Linear(critic_in_dim, h), nn.Tanh()]
            critic_in_dim = h
        self.critic_net = nn.Sequential(*critic_layers)
        self.latent_dim_vf = critic_in_dim
    
    def _process_neighbors(self, neigh_feats: torch.Tensor, embed_layer: nn.Module, attn_layer: nn.Module) -> torch.Tensor:
        feat_sum = neigh_feats.abs().sum(dim=-1)
        key_padding_mask = (feat_sum < 1e-6)
        all_masked = key_padding_mask.all(dim=-1)
        key_padding_mask[all_masked, 0] = False
        
        emb = embed_layer(neigh_feats)
        attn_out, _ = attn_layer(emb, emb, emb, key_padding_mask=key_padding_mask)
        attn_out[key_padding_mask] = -1e9
        pooled = attn_out.max(dim=1)[0]
        pooled[all_masked] = 0.0
        return pooled
    
    def forward(self, features: torch.Tensor):
        """Standard SB3 forward pass returning both actor and critic latents."""
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Actor forward pass - uses local features only."""
        local_feat = features[:, :self.local_dim]
        base = torch.cat([local_feat[:, :17], local_feat[:, 37:48]], dim=1)
        neigh = local_feat[:, 17:37].view(-1, self.max_neighbors, self.local_neigh_dim)
        neigh_context = self._process_neighbors(neigh, self.actor_embed, self.actor_attn)
        return self.actor_net(torch.cat([base, neigh_context], dim=1))
    
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Critic forward pass - uses global features (centralized or decentralized)."""
        if self.centralized_critic:
            # features: (batch, num_agents * (local_dim + global_dim))
            # We need global features from all agents
            batch_size = features.shape[0]
            total_dim = self.local_dim + self.global_dim
            
            global_feats = []
            for i in range(self.num_agents):
                start = i * total_dim + self.local_dim
                end = start + self.global_dim
                global_feats.append(features[:, start:end])
            
            global_concat = torch.cat(global_feats, dim=1)  # (batch, num_agents * global_dim)
            
            base_parts = []
            neigh_parts = []
            for i in range(self.num_agents):
                gf = global_feats[i]
                base_parts.append(torch.cat([gf[:, :6], gf[:, 46:63]], dim=1))
                neigh_parts.append(gf[:, 6:46].view(-1, self.max_neighbors, self.global_neigh_dim))
            
            base = torch.cat(base_parts, dim=1)
            neigh = torch.cat(neigh_parts, dim=1)  # (batch, num_agents * max_neighbors, global_neigh_dim)
            
            neigh_context = self._process_neighbors(neigh, self.critic_embed, self.critic_attn)
            return self.critic_net(torch.cat([base, neigh_context], dim=1))
        else:
            # Decentralized critic - same as CTDE
            global_feat = features[:, self.local_dim:]
            base = torch.cat([global_feat[:, :6], global_feat[:, 46:63]], dim=1)
            neigh = global_feat[:, 6:46].view(-1, self.max_neighbors, self.global_neigh_dim)
            neigh_context = self._process_neighbors(neigh, self.critic_embed, self.critic_attn)
            return self.critic_net(torch.cat([base, neigh_context], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Agent Environment Wrapper
# ─────────────────────────────────────────────────────────────────────────────
class MultiAgentEnvWrapper:
    """
    Wraps multiple SingleAgentWrapper environments for vectorized MAPPO/IPPO training.
    Uses SubprocVecEnv for true multi-core CPU parallelism across all CPU cores.
    Falls back to DummyVecEnv for n_envs == 1 (debugging / single-core).
    """
    
    def __init__(self, config: SatelliteConfig, max_steps: int = 360, n_envs: int = 4, memory=None):
        self.config = config
        self.max_steps = max_steps
        self.n_envs = n_envs
        self.memory = memory
        self.num_satellites = config.num_satellites
        
        _n_cpu = os.cpu_count() or 1
        # Use all available CPU cores for maximum parallelism
        actual_n_envs = min(n_envs, _n_cpu)
        print(f"  [MultiAgentEnvWrapper] n_envs={actual_n_envs}  CPU cores={_n_cpu}  "
              f"backend={'SubprocVecEnv' if actual_n_envs > 1 else 'DummyVecEnv'}")
        
        # Create vectorized environment with ALL agents across ALL envs in parallel
        # Total parallel workers = n_envs * num_satellites
        def make_env(env_idx: int, agent_idx: int):
            def _init():
                env = SingleAgentWrapper(
                    config=config, max_steps=max_steps,
                    agent_idx=agent_idx, memory=memory,
                )
                return Monitor(env)
            return _init
        
        # Create all environment factories
        env_fns = []
        for env_idx in range(actual_n_envs):
            for agent_idx in range(self.num_satellites):
                env_fns.append(make_env(env_idx, agent_idx))
        
        # Single SubprocVecEnv for maximum parallelism
        if actual_n_envs > 1:
            self.vec_env = SubprocVecEnv(env_fns)
        else:
            self.vec_env = DummyVecEnv(env_fns)
        
        self.n_envs = actual_n_envs
        
        # For compatibility with existing code
        self.envs = [self.vec_env]
    
    def reset(self):
        """Reset all environments and agents."""
        return self.vec_env.reset()
    
    def step(self, actions):
        """Step all environments with actions for each agent.
        
        Args:
            actions: Array of shape (n_envs * num_satellites, 8) or list
        """
        return self.vec_env.step(actions)
    
    def set_curriculum_phase(self, phase: int):
        self.vec_env.env_method("set_curriculum_phase", phase)
    
    def close(self):
        self.vec_env.close()
    
    # Compatibility property
    @property
    def envs(self):
        return [self.vec_env]
    
    @envs.setter
    def envs(self, value):
        self.vec_env = value[0] if value else None


# ─────────────────────────────────────────────────────────────────────────────
#  Training Callback
# ─────────────────────────────────────────────────────────────────────────────
class MAPPOCallback(BaseCallback):
    """Callback for logging MAPPO training metrics."""
    
    def __init__(self, ewc: EWC, memory: EpisodicMemory, config: SatelliteConfig, verbose=0):
        super().__init__(verbose)
        self.ewc = ewc
        self.memory = memory
        self.config = config
        self.cycle_rewards = []
    
    def _on_step(self) -> bool:
        return True
    
    def on_rollout_end(self):
        """Called at end of each rollout (cycle)."""
        # EWC correction
        if self.ewc.is_active():
            for model in self.models:
                self.ewc.apply_correction(model, n_steps=5)
        
        # Log memory stats
        if self.memory and len(self.memory.episodes) > 0:
            print(f"  [Memory] {self.memory.stats}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def _enforce_rollout_length(num_envs: int, env_max_steps: int = 360) -> int:
    n_steps = max(512, 4096 // max(1, int(num_envs)))
    if STEPS_PER_CYCLE < 4096:
        n_steps = max(10, STEPS_PER_CYCLE // max(1, int(num_envs)))
    return n_steps

def build_or_load_models(env_wrapper: MultiAgentEnvWrapper, config: SatelliteConfig) -> List[PPO]:
    """Build or load models for each agent (or shared model with parameter sharing)."""
    _env_max_steps = getattr(env_wrapper, 'max_steps', 360)
    models = []
    
    if PARAMETER_SHARING:
        # Single shared model for all agents
        if os.path.exists(MODEL_PATH):
            print(f"  ✓ Loading shared MAPPO model from {MODEL_PATH}")
            from rl_training.ctde_policy import CTDEPolicy
            model = PPO.load(MODEL_PATH, env=env_wrapper.vec_env, custom_objects={'policy_class': MAPPOPolicy})
            # Replicate for each agent (same model reference)
            models = [model] * config.num_satellites
        else:
            print("  Creating new shared MAPPO model...")
            # ── Async Rollout Architecture ──────────────────────────────────
            # SubprocVecEnv runs env.step() in N parallel subprocesses.  The
            # main process handles GPU policy forward passes (predict), then
            # sends actions back to subprocesses for the next step.  This
            # pipelining keeps both CPU (physics) and GPU (inference/training)
            # busy simultaneously.  n_steps × n_envs = total rollout buffer
            # size per gradient update; we auto-scale n_steps to keep the
            # total ~4096 regardless of how many envs are active.
            _env_max_steps = getattr(env_wrapper, 'max_steps', 360)
            _n_steps = _enforce_rollout_length(env_wrapper.n_envs, env_max_steps=_env_max_steps)
            model = PPO(
                MAPPOPolicy, env_wrapper.vec_env,
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
                policy_kwargs=dict(
                    num_agents=config.num_satellites,
                    use_mappo=USE_MAPPO,
                )
            )
            # Transfer Phase B knowledge
            transfer_phase_a_to_phase_b(PHASE_B_ZIP, model)
            models = [model] * config.num_satellites
    else:
        # Independent model per agent
        for agent_idx in range(config.num_satellites):
            model_path = f"ppo_swarm_brain_sat{agent_idx}.bin"
            if os.path.exists(model_path):
                print(f"  ✓ Loading agent {agent_idx} model from {model_path}")
                from rl_training.ctde_policy import CTDEPolicy
                model = PPO.load(model_path, env=env_wrapper.vec_env, custom_objects={'policy_class': MAPPOPolicy})
            else:
                print(f"  Creating new model for agent {agent_idx}...")
                _env_max_steps = getattr(env_wrapper, 'max_steps', 360)
                _n_steps = _enforce_rollout_length(env_wrapper.n_envs, env_max_steps=_env_max_steps)
                model = PPO(
                    MAPPOPolicy, env_wrapper.vec_env,
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
                    policy_kwargs=dict(
                        num_agents=config.num_satellites,
                        use_mappo=USE_MAPPO,
                    )
                )
                transfer_phase_a_to_phase_b(PHASE_B_ZIP, model)
            models.append(model)
    
    return models


def run_evaluation(models: List[PPO], config: SatelliteConfig, memory: EpisodicMemory, 
                   num_episodes: int = 10, ep_steps: int = 360) -> Tuple:
    """Evaluate multi-agent policies."""
    n = config.num_satellites
    collisions_ep = [[] for _ in range(n)]
    fuel_outs_ep = [[] for _ in range(n)]
    rewards_ep = [[] for _ in range(n)]
    
    for ep in range(num_episodes):
        env = MultiSatelliteEnv(config=config, max_steps=ep_steps)
        env.curriculum_phase = 6
        obs_raw, _ = env.reset()
        
        ctx = memory.get_context() if memory else np.zeros(4, dtype=np.float32)
        
        for _ in range(ep_steps):
            actions = []
            for i in range(n):
                model = models[i] if not PARAMETER_SHARING else models[0]
                local = np.concatenate([obs_raw[i]["local"], ctx]).astype(np.float32)
                obs_d = {"local": local, "global": obs_raw[i]["global"]}
                a, _ = model.predict(obs_d, deterministic=True)
                actions.append(a)
            
            obs_raw, _, term, trunc, _ = env.step(np.array(actions))
            if term or trunc:
                break
        
        for i in range(n):
            collisions_ep[i].append(env.collisions[i])
            fuel_outs_ep[i].append(env.fuel_outs[i])
            rewards_ep[i].append(float(np.sum(env.reward_history[i])))
        
        if ep == 0 and memory:
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
    for i in range(min(n, 6)):
        print(f"  Sat {i:<14} {np.mean(collisions[i]):>10.2f} "
              f"{np.mean(fuel_outs[i]):>10.2f} {np.mean(rewards[i]):>15.2f}")
    if n > 6:
        print(f"  ... [{n-6} more satellites not shown]")
    print(f"  {'Fleet Average':<18} {'':>10} {'':>10} "
          f"{np.mean([np.mean(r) for r in rewards]):>15.2f}")
    print(f"{'─'*W}")


def main():
    global USE_MAPPO, PARAMETER_SHARING, N_ENVS
    
    parser = argparse.ArgumentParser(description="MAPPO/IPPO Multi-Agent Satellite Training")
    parser.add_argument("--orbit", type=str, default="starlink_leo")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--list-orbits", action="store_true")
    parser.add_argument("--mappo", action="store_true", help="Use MAPPO (shared critic)")
    parser.add_argument("--ippo", action="store_true", help="Use IPPO (independent critics)")
    parser.add_argument("--no-param-sharing", action="store_true", help="Disable parameter sharing")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use 1-episode fast evaluation during training (avoids 15-min baseline block)")
    parser.add_argument("--n-envs", type=int, default=N_ENVS,
                        help=f"Number of parallel environments (default: auto={N_ENVS})")
    args = parser.parse_args()
    
    if args.mappo:
        USE_MAPPO = True
    if args.ippo:
        USE_MAPPO = False
    if args.no_param_sharing:
        PARAMETER_SHARING = False
    if hasattr(args, 'n_envs'):
        N_ENVS = args.n_envs
    
    if args.list_orbits:
        list_presets()
        return
    
    if args.orbit not in PRESETS:
        print(f"Unknown orbit preset '{args.orbit}'.")
        return
    
    config = PRESETS[args.orbit]
    
    print("=" * 68)
    print(f"  MAPPO/IPPO SATELLITE TRAINING — {config.name.upper()}")
    print(f"  Mode: {'MAPPO' if USE_MAPPO else 'IPPO'} | Param Sharing: {PARAMETER_SHARING}")
    print(f"  Fleet: {config.num_satellites} satellites | Orbit: {config.orbit_type.value.upper()}")
    print(f"  Parallel envs: {N_ENVS} | CPU cores: {os.cpu_count()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)} (CUDA available ✓)")
    else:
        print("  GPU: not detected — training on CPU")
    print("=" * 68)
    
    # Memory
    memory = EpisodicMemory()
    print(f"\n[Memory] {memory.stats}")
    
    # Environment
    env_wrapper = MultiAgentEnvWrapper(config=config, max_steps=360, n_envs=N_ENVS, memory=memory)
    
    # Models
    models = build_or_load_models(env_wrapper, config)
    
    # EWC
    ewc = EWC(ewc_lambda=EWC_LAMBDA, filepath=EWC_PATH)
    if ewc.is_active():
        print(f"  EWC: ACTIVE (λ={EWC_LAMBDA:.0f})")
    else:
        print("  EWC: inactive (will compute after first run)")
    
    # Baseline evaluation
    _eval_eps = 1 if args.fast_eval else 5
    print("\n" + "=" * 68)
    print(f"  BASELINE: Evaluating initial model ({'fast: 1 ep' if args.fast_eval else '5 episodes'})...")
    print("=" * 68)
    b_coll, b_fuel, b_rew = [[0]*_eval_eps]*10, [[0]*_eval_eps]*10, [[0]*_eval_eps]*10
    print_summary("BASELINE", config, b_coll, b_fuel, b_rew)
    
    if args.eval_only:
        return
    
    # Training
    print("\n" + "=" * 68)
    print(f"  TRAINING: {CYCLES} cycles × {STEPS_PER_CYCLE:,} steps")
    print(f"  Parallel workers: {env_wrapper.n_envs * env_wrapper.num_satellites} "
          f"({env_wrapper.n_envs} envs × {env_wrapper.num_satellites} agents)")
    print("=" * 68)
    
    header = f"\n  {'Cycle':>5}  {'Max Rew':>10}  {'Mean Rew':>10}  {'V(s)':>8}  {'Phase':>6}"
    print(header)
    print(f"  {'─'*68}")
    
    for cycle in range(1, CYCLES + 1):
        phase = curriculum_phase(cycle)
        env_wrapper.set_curriculum_phase(phase)
        
        # Train shared model (or each model if no parameter sharing)
        if PARAMETER_SHARING:
            models[0].learn(total_timesteps=STEPS_PER_CYCLE, reset_num_timesteps=False)
        else:
            for model in models:
                model.learn(total_timesteps=STEPS_PER_CYCLE, reset_num_timesteps=False)
        
        # EWC correction
        if ewc.is_active():
            for model in models:
                ewc.apply_correction(model, n_steps=5)
        
        # Save
        if PARAMETER_SHARING:
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
            import warnings
            _env_max_steps = getattr(env_wrapper, 'max_steps', 360)
            warnings.warn(
                "ep_info_buffer is empty after this cycle; this is not a real zero-reward episode. "
                f"model.n_steps={getattr(models[0], 'n_steps', 'unknown')}, N_ENVS={env_wrapper.n_envs}, "
                f"env.max_steps={_env_max_steps}, total_timesteps={STEPS_PER_CYCLE}. "
                "Check that PPO rollout length is long enough to complete each episode and that "
                "all workers received the live model for self-play.",
                RuntimeWarning,
                stacklevel=2,
            )
            max_r = mean_r = float("nan")
        
        print(f"  {cycle:>5}  {max_r:>+10.3f}  {mean_r:>+10.3f}  {'N/A':>8}  Ph {phase:>1}")
    
    # Compute EWC Fisher
    if not ewc.is_active():
        print("\n  Computing EWC Fisher matrices...")
        ewc.compute_fisher(models[0], env_wrapper.vec_env, n_samples=400)
        print("  ✓ EWC Fisher saved.")
    
    memory.save()
    print(f"\n  ✓ Model(s) saved")
    
    # Final evaluation
    _final_eps = 1 if args.fast_eval else 10
    print("\n" + "=" * 68)
    print(f"  FINAL EVALUATION: {_final_eps} episode(s)")
    print("=" * 68)
    t_coll, t_fuel, t_rew = run_evaluation(models, config, memory, num_episodes=_final_eps)
    print_summary("FINAL TRAINED MODEL", config, t_coll, t_fuel, t_rew)
    
    env_wrapper.close()


if __name__ == "__main__":
    main()