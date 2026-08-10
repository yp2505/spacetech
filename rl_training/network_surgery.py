"""
network_surgery.py
==================
Phase B: Transfer Phase A knowledge into the Phase B architecture.

Phase A model (2-satellite, discrete 5-action, 15-dim local, 13-dim global):
  - Actor hidden layers [256, 256] are IDENTICAL in shape → copy exactly
  - Critic hidden layers [512, 512, 256] are IDENTICAL in shape → copy exactly
  - Input layers (15-dim, 13-dim) → zero-pad to new Phase B dims (46, 34)
  - Action output head (5-discrete) → reinitialize for 4-continuous

Result: Phase B model starts with Phase A's learned orbital representations
        baked into the hidden layers, and only needs to learn the new
        input features and continuous action head from scratch.
"""

import torch
import zipfile
import os
import shutil
import tempfile
import numpy as np


def pad_weight(old_tensor: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    """
    Zero-pad an old weight tensor to new_shape.
    Old values are placed in the top-left corner, preserving all knowledge.
    """
    new_t = torch.zeros(new_shape, dtype=old_tensor.dtype)
    slices = tuple(slice(0, s) for s in old_tensor.shape)
    new_t[slices] = old_tensor
    return new_t


def transfer_phase_a_to_phase_b(
    phase_a_zip: str,
    phase_b_model,          # a freshly created SB3 PPO model with Phase B env
    new_local_dim:  int = 46,
    new_global_dim: int = 34,
) -> None:
    """
    Transfer Phase A knowledge into Phase B model in-place.

    Strategy:
    - Input layers:  zero-pad (new features start neutral)
    - Hidden layers: copy exactly (preserve learned representations)
    - Output head:   leave fresh (action space changed: discrete→continuous)
    - Value net:     copy exactly (critic output unchanged)
    """
    if not os.path.exists(phase_a_zip):
        print(f"[Surgery] Phase A checkpoint not found at {phase_a_zip}. Skipping transfer.")
        return

    print(f"[Surgery] Loading Phase A weights from {phase_a_zip}...")
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(phase_a_zip, "r") as zf:
            zf.extractall(temp_dir)

        policy_path = os.path.join(temp_dir, "policy.pth")
        if not os.path.exists(policy_path):
            print("[Surgery] policy.pth not found inside zip. Skipping transfer.")
            return

        old_sd = torch.load(policy_path, map_location="cpu", weights_only=True)
        new_sd = phase_b_model.policy.state_dict()

        transferred, padded, skipped = [], [], []

        for name, new_param in new_sd.items():
            if name not in old_sd:
                skipped.append(name)
                continue

            old_param = old_sd[name]
            if old_param.shape == new_param.shape:
                # ── Exact match: copy directly ────────────────────────────────
                new_sd[name] = old_param.clone()
                transferred.append(name)
            elif all(o <= n for o, n in zip(old_param.shape, new_param.shape)):
                # ── Old is smaller: zero-pad ──────────────────────────────────
                new_sd[name] = pad_weight(old_param, tuple(new_param.shape))
                padded.append(f"{name}: {list(old_param.shape)} → {list(new_param.shape)}")
            else:
                # ── Incompatible: skip (e.g., action_net for discrete→continuous)
                skipped.append(f"{name}: old={list(old_param.shape)} vs new={list(new_param.shape)}")

        phase_b_model.policy.load_state_dict(new_sd)
        print(f"[Surgery] ✓ Transferred {len(transferred)} layers exactly")
        print(f"[Surgery] ✓ Zero-padded {len(padded)} input layers:")
        for p in padded:
            print(f"            {p}")
        print(f"[Surgery]   Skipped {len(skipped)} layers (action head + new params):")
        for s in skipped[:5]:
            print(f"            {s}")
        if len(skipped) > 5:
            print(f"            ... and {len(skipped)-5} more")
        print("[Surgery] Knowledge transfer complete! Phase B brain is ready.")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Run this via train.py --transfer flag.")
