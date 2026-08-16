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

LOCAL_DIM = 48
GLOBAL_DIM = 63

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
        embed_dim:     int = 64,
        num_heads:     int = 4,
    ):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        
        # Base features dimensions
        # local: 0:17 (base 17), 17:37 (neighbors 20), 37:48 (base 11)
        self.local_base_dim = 17 + 11
        self.local_neigh_dim = 5
        self.max_neighbors = 4
        
        # global: 0:6 (base 6), 6:46 (neighbors 40), 46:63 (base 17)
        self.global_base_dim = 6 + 17
        self.global_neigh_dim = 10

        # ── Actor Attention ───────────────────────────────────────────────────
        self.actor_embed = nn.Linear(self.local_neigh_dim, embed_dim)
        self.actor_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # ── Critic Attention ──────────────────────────────────────────────────
        self.critic_embed = nn.Linear(self.global_neigh_dim, embed_dim)
        self.critic_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # ── Actor MLP ─────────────────────────────────────────────────────────
        actor_layers = []
        in_dim = self.local_base_dim + embed_dim
        for h in actor_hidden:
            actor_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.actor_net      = nn.Sequential(*actor_layers)
        self.latent_dim_pi  = in_dim

        # ── Critic MLP ────────────────────────────────────────────────────────
        critic_layers = []
        in_dim = self.global_base_dim + embed_dim
        for h in critic_hidden:
            critic_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.critic_net     = nn.Sequential(*critic_layers)
        self.latent_dim_vf  = in_dim

    def _process_neighbors(self, neigh_feats: torch.Tensor, embed_layer: nn.Module, attn_layer: nn.Module) -> torch.Tensor:
        # neigh_feats: (batch_size, num_neighbors, feat_dim)
        
        # Create key_padding_mask: True where neighbor vector is all zeros (dummy)
        feat_sum = neigh_feats.abs().sum(dim=-1)
        key_padding_mask = (feat_sum < 1e-6)
        
        # If a whole batch item has no neighbors, unmask the first dummy to avoid NaN in attention
        all_masked = key_padding_mask.all(dim=-1)
        key_padding_mask[all_masked, 0] = False

        # Embed and attend
        emb = embed_layer(neigh_feats) # (batch, seq, embed_dim)
        attn_out, _ = attn_layer(emb, emb, emb, key_padding_mask=key_padding_mask)
        
        # Max pool over sequence, explicitly ignoring masked elements
        attn_out[key_padding_mask] = -1e9
        pooled = attn_out.max(dim=1)[0]
        
        # For completely masked items (where we unmasked dummy 0 to prevent NaN), zero out the output
        pooled[all_masked] = 0.0
        
        return pooled

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        local_feat = features[:, :self.local_dim]
        base = torch.cat([local_feat[:, :17], local_feat[:, 37:48]], dim=1)
        neigh = local_feat[:, 17:37].view(-1, self.max_neighbors, self.local_neigh_dim)
        
        neigh_context = self._process_neighbors(neigh, self.actor_embed, self.actor_attn)
        return self.actor_net(torch.cat([base, neigh_context], dim=1))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        global_feat = features[:, self.local_dim:]
        base = torch.cat([global_feat[:, :6], global_feat[:, 46:63]], dim=1)
        neigh = global_feat[:, 6:46].view(-1, self.max_neighbors, self.global_neigh_dim)
        
        neigh_context = self._process_neighbors(neigh, self.critic_embed, self.critic_attn)
        return self.critic_net(torch.cat([base, neigh_context], dim=1))

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

def load_old_weights_with_padding(model, old_model_path="ppo_satellite_1.bin"):
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
                    # Shape mismatch handling
                    if 'mlp_extractor.actor_net.0.weight' in name and param.shape[1] == 48:
                        print(f"  Mapping old actor weights into new attention-based actor...")
                        # base 0-17
                        new_param.data[:, :17] = param.data[:, :17]
                        # base 37-48 -> new 17-28
                        new_param.data[:, 17:28] = param.data[:, 37:48]
                        # Zero out attention features
                        new_param.data[:, 28:] = 0.0
                    elif 'mlp_extractor.critic_net.0.weight' in name and param.shape[1] == 63:
                        print(f"  Mapping old critic weights into new attention-based critic...")
                        # base 0-6
                        new_param.data[:, :6] = param.data[:, :6]
                        # base 46-63 -> new 6-23
                        new_param.data[:, 6:23] = param.data[:, 46:63]
                        # Zero out attention features
                        new_param.data[:, 23:] = 0.0
                    elif len(param.shape) == 2 and len(new_param.shape) == 2:
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
