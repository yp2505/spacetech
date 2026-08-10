"""
ctde_policy.py
--------------
Centralised Training with Decentralised Execution (CTDE) Policy for SB3.
Phases C-F: Universal Configuration Engine with Multi-Mission, Thermal,
and Multi-Plane support.

Observation dims (must match simulation.satellite_env constants):
  LOCAL_DIM      = 53   (49 base + 4 memory context)
  GLOBAL_DIM     = 47
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

LOCAL_DIM  = 53
GLOBAL_DIM = 47

class CTDEFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        local_dim  = observation_space["local"].shape[0]
        global_dim = observation_space["global"].shape[0]
        super().__init__(observation_space, features_dim=local_dim + global_dim)
        self.local_dim  = local_dim
        self.global_dim = global_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        local    = observations["local"].float()
        global_s = observations["global"].float()
        return torch.cat([local, global_s], dim=-1)

class CTDEMlpExtractor(nn.Module):
    def __init__(
        self,
        local_dim:     int,
        global_dim:    int,
        actor_hidden:  List[int] = [256, 256],
        critic_hidden: List[int] = [512, 512, 256],
    ):
        super().__init__()
        self.local_dim = local_dim

        # ── Actor: small network, only local obs ──────────────────────────────
        actor_layers = []
        in_dim = local_dim
        for h in actor_hidden:
            actor_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.actor_net      = nn.Sequential(*actor_layers)
        self.latent_dim_pi  = in_dim

        # ── Critic: larger network, full global state ─────────────────────────
        critic_layers = []
        in_dim = global_dim
        for h in critic_hidden:
            critic_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.critic_net     = nn.Sequential(*critic_layers)
        self.latent_dim_vf  = in_dim

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        local_feat  = features[:, :self.local_dim]
        global_feat = features[:, self.local_dim:]
        return self.actor_net(local_feat), self.critic_net(global_feat)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_net(features[:, :self.local_dim])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_net(features[:, self.local_dim:])

class CTDEPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CTDEFeaturesExtractor,
            features_extractor_kwargs={},
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        local_dim  = self.features_extractor.local_dim
        global_dim = self.features_extractor.global_dim
        self.mlp_extractor = CTDEMlpExtractor(local_dim, global_dim)

    def get_value_estimate(self, obs_dict: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            features  = self.extract_features(obs_dict)
            latent_vf = self.mlp_extractor.forward_critic(features)
            values    = self.value_net(latent_vf)
        return float(values.mean().item())

def load_old_weights_with_padding(model, old_model_path="ppo_satellite_1.zip"):
    """
    Safely load old 15/13-dim model weights into the new 46/31-dim model.
    The new inputs (neighbor states, attitude) will have weights initialized to zero,
    meaning the network ignores them initially, preserving exact old behavior.
    """
    import os
    if not os.path.exists(old_model_path):
        print(f"Skipping weight padding, {old_model_path} not found.")
        return

    print(f"Attempting to load and pad old weights from {old_model_path}...")
    from stable_baselines3 import PPO
    try:
        old_model = PPO.load(old_model_path, custom_objects={'policy_class': CTDEPolicy})
        old_state_dict = old_model.policy.state_dict()
        new_state_dict = model.policy.state_dict()
        
        for name, param in old_state_dict.items():
            if name in new_state_dict:
                new_param = new_state_dict[name]
                if param.shape == new_param.shape:
                    new_state_dict[name].copy_(param)
                else:
                    # Shape mismatch. If it's a 2D weight matrix (e.g. Linear layer)
                    if len(param.shape) == 2 and len(new_param.shape) == 2:
                        old_out, old_in = param.shape
                        new_out, new_in = new_param.shape
                        if new_out == old_out and new_in > old_in:
                            print(f"  Padding weights for {name}: {old_in} -> {new_in}")
                            # Copy old weights into the first 'old_in' columns
                            new_param.data[:, :old_in] = param.data
                            # Zero out the rest so new features don't disrupt output
                            new_param.data[:, old_in:] = 0.0
                        else:
                            print(f"  Skipping shape mismatch {name}: {param.shape} != {new_param.shape}")
                    else:
                        print(f"  Skipping shape mismatch {name}: {param.shape} != {new_param.shape}")
                        
        model.policy.load_state_dict(new_state_dict)
        print("✓ Successfully padded old weights into new architecture!")
    except Exception as e:
        print(f"Error loading old weights: {e}")
