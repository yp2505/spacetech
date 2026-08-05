"""
ctde_policy.py
--------------
Centralised Training with Decentralised Execution (CTDE) Policy for SB3.

KEY IDEA
--------
During TRAINING  → Critic sees the GLOBAL state (both satellites + full env)
                   This gives stable, low-variance value estimates.
During DEPLOYMENT → Actor only sees LOCAL obs (own sensors — pos, fuel, battery,
                    velocity, gap error, closing rate, debris, eclipse, weather,
                    orbits completed, memory context).

Phase 3 Network Architecture
-----------------------------
  Actor  MLP [128 → 128, Tanh] — takes LOCAL  obs (15 dims) → 5 action logits
  Critic MLP [256 → 256 → 128, Tanh] — takes GLOBAL state (13 dims) → V(s)

Observation dims (must match satellite_env.py constants):
  BASE_LOCAL_DIM = 11  (pos, vel, fuel, bat, gap_err, closing_rate, debris×3, eclipse, weather)
  MEMORY_CTX_DIM = 4   (episodic memory context)
  LOCAL_DIM      = 15  (total actor input)
  GLOBAL_DIM     = 13  (sat0×4 + sat1×4 + debris×3 + eclipse + weather)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

import numpy as np
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Obs dims — kept in sync with satellite_env.py ─────────────────────────────
LOCAL_DIM  = 15   # Phase 3: pos,vel,fuel,bat,gap_err,closing_rate,debris×3,eclipse,weather,orbits + 4 memory
GLOBAL_DIM = 13   # Phase 3: sat0[pos,vel,fuel,bat] + sat1[pos,vel,fuel,bat] + debris×3 + eclipse + weather


# ─────────────────────────────────────────────────────────────────────────────
#  Features extractor: concatenates local + global into a flat vector
# ─────────────────────────────────────────────────────────────────────────────
class CTDEFeaturesExtractor(BaseFeaturesExtractor):
    """
    Takes the Dict observation {"local": ..., "global": ...} and
    concatenates them into one flat tensor [local | global].

    The downstream CTDEMlpExtractor then splits this tensor and
    routes each portion to the correct (Actor or Critic) network.
    """

    def __init__(self, observation_space: spaces.Dict):
        local_dim  = observation_space["local"].shape[0]
        global_dim = observation_space["global"].shape[0]
        # features_dim = total concatenated size
        super().__init__(observation_space, features_dim=local_dim + global_dim)
        self.local_dim  = local_dim
        self.global_dim = global_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        local    = observations["local"].float()
        global_s = observations["global"].float()
        return torch.cat([local, global_s], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Split MLP extractor: Actor uses local slice, Critic uses global slice
# ─────────────────────────────────────────────────────────────────────────────
class CTDEMlpExtractor(nn.Module):
    """
    True CTDE MLP extractor with separate Actor and Critic branches.

    SB3 calls:
      forward(features)         → (latent_pi, latent_vf)  [during training]
      forward_actor(features)   → latent_pi               [during model.predict()]
      forward_critic(features)  → latent_vf               [for value estimates]
    """

    def __init__(
        self,
        local_dim:     int,
        global_dim:    int,
        actor_hidden:  List[int] = [128, 128],      # Phase 3: keep same capacity
        critic_hidden: List[int] = [256, 256, 128], # Phase 3: keep same capacity
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
        self.latent_dim_pi  = in_dim   # output dim (expected by SB3)

        # ── Critic: larger network, full global state ─────────────────────────
        critic_layers = []
        in_dim = global_dim
        for h in critic_hidden:
            critic_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.critic_net     = nn.Sequential(*critic_layers)
        self.latent_dim_vf  = in_dim   # output dim (expected by SB3)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass — both branches (used during training rollouts)."""
        local_feat  = features[:, :self.local_dim]
        global_feat = features[:, self.local_dim:]
        return self.actor_net(local_feat), self.critic_net(global_feat)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Actor-only path — called by SB3 during model.predict() (deployment)."""
        return self.actor_net(features[:, :self.local_dim])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Critic-only path — called by SB3 for value estimates."""
        return self.critic_net(features[:, self.local_dim:])


# ─────────────────────────────────────────────────────────────────────────────
#  Full CTDE Policy (wires extractor + mlp_extractor into SB3)
# ─────────────────────────────────────────────────────────────────────────────
class CTDEPolicy(ActorCriticPolicy):
    """
    SB3-compatible ActorCriticPolicy implementing true CTDE.

    Training:   Actor sees local obs (12-dim), Critic sees global state (11-dim).
    Deployment: model.predict() only uses Actor → purely local, no partner data.

    Usage:
        model = PPO(CTDEPolicy, env, verbose=0)
    """

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
        """
        Override: replace SB3's shared MlpExtractor with our split CTDE version.
        Called automatically by _build() after the features extractor is created.
        """
        local_dim  = self.features_extractor.local_dim
        global_dim = self.features_extractor.global_dim
        self.mlp_extractor = CTDEMlpExtractor(local_dim, global_dim)

    # ── Q-value proxy (V(s) from centralised Critic) ──────────────────────────
    def get_value_estimate(self, obs_dict: Dict[str, torch.Tensor]) -> float:
        """
        Returns the mean V(s) estimate over a batch of observations.
        Used as a Q-value proxy for stability monitoring each cycle.

        Q(s,a) ≈ V(s) + A(s,a).  Since advantages average to 0, V(s) is the
        best single-number summary of the Critic's confidence in the current policy.
        """
        with torch.no_grad():
            features  = self.extract_features(obs_dict)
            latent_vf = self.mlp_extractor.forward_critic(features)
            values    = self.value_net(latent_vf)
        return float(values.mean().item())
