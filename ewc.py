"""
ewc.py
------
Elastic Weight Consolidation (EWC) for Continual Learning.

Prevents catastrophic forgetting when the agent learns new tasks (e.g. a
harder orbit scenario) after mastering the original mission.

Reference: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in
neural networks" — https://arxiv.org/abs/1612.00796

─────────────────────────────────────────────────────────────────────────────
ALGORITHM
─────────────────────────────────────────────────────────────────────────────

Phase A  (after Task-1 training is complete):
    1. Sample n random states from the environment.
    2. For each state s, compute ∂ log π(a|s) / ∂θᵢ for every parameter θᵢ.
    3. Fisher diagonal: F_ii ≈ (1/n) × Σ (∂ log π(a|s) / ∂θᵢ)²
       F_ii is large when weight θᵢ strongly shapes the policy → protect it.
    4. Store "anchor" weights θ* = current weights after Task-1 training.

Phase B  (while training on Task-2):
    After each PPO update, apply an EWC correction gradient step:
        L_ewc = λ × Σᵢ F_ii × (θᵢ − θ*ᵢ)²
    This pulls important weights back toward θ* if PPO pushed them away.

    Combined effect: Task-2 learning flows mainly through weights that are
    NOT important for Task-1 → old knowledge is preserved.

─────────────────────────────────────────────────────────────────────────────
USAGE IN TRAIN.PY
─────────────────────────────────────────────────────────────────────────────
    from ewc import EWC

    ewc1 = EWC(ewc_lambda=5000, filepath="ewc_fisher_sat1.pkl")
    ewc2 = EWC(ewc_lambda=5000, filepath="ewc_fisher_sat2.pkl")

    # After Task-1 training finishes:
    ewc1.compute_fisher(model1, env1_wrapper, n_samples=300)
    ewc2.compute_fisher(model2, env2_wrapper, n_samples=300)

    # During every Task-2 training cycle (after model.learn()):
    ewc_loss1 = ewc1.apply_correction(model1, n_steps=5)
    ewc_loss2 = ewc2.apply_correction(model2, n_steps=5)
"""

import os
import pickle
import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor


# ─────────────────────────────────────────────────────────────────────────────
class EWC:
    """
    Elastic Weight Consolidation for a single SB3 PPO model.

    One EWC instance per satellite model (ewc1 for Sat-1, ewc2 for Sat-2).
    """

    def __init__(self, ewc_lambda: float = 5_000.0, filepath: str = None):
        """
        Args:
            ewc_lambda: EWC penalty strength λ.
                • < 100  : virtually no protection (good for warm-up)
                • 5_000  : Kirkpatrick 2017 default — solid Task-1 protection
                • > 50_000: rigid memory, may block Task-2 learning completely
            filepath: where to persist Fisher matrices between sessions.
                      Pass None to disable persistence.
        """
        self.ewc_lambda  = ewc_lambda
        self.filepath    = filepath
        self.fisher: dict = {}    # {param_name: diagonal FIM tensor (CPU)}
        self.anchors: dict = {}   # {param_name: θ* tensor (CPU)}
        self._active      = False  # True once Fisher has been computed

        if filepath and os.path.exists(filepath):
            self._load()

    # ── public API ─────────────────────────────────────────────────────────────
    def is_active(self) -> bool:
        """True once Fisher matrices exist and EWC protection is live."""
        return self._active and bool(self.fisher)

    # ── Fisher Information Matrix ───────────────────────────────────────────────
    def compute_fisher(self, model, env_wrapper, n_samples: int = 300) -> None:
        """
        Compute diagonal Fisher Information Matrix from n_samples random states.

        F_ii ≈ E_{s,a~π} [ (∂ log π(a|s) / ∂θᵢ)² ]

        After this call, apply_correction() becomes active.
        """
        print(f"\n  [EWC] ━━ Computing Fisher Information Matrix ━━")
        print(f"  [EWC] Sampling {n_samples} policy roll-outs...")

        model.policy.set_training_mode(True)
        device = next(model.policy.parameters()).device

        # Zero-initialise accumulator (keep on CPU to save GPU VRAM)
        fisher_accum = {
            name: torch.zeros_like(param.data, device="cpu")
            for name, param in model.policy.named_parameters()
        }

        valid = 0
        for _ in range(n_samples):
            try:
                obs, _ = env_wrapper.reset()
                obs_t  = obs_as_tensor(
                    {"local":  obs["local"][None, :].astype(np.float32),
                     "global": obs["global"][None, :].astype(np.float32)},
                    device,
                )

                # Get action distribution and sample
                dist   = model.policy.get_distribution(obs_t)
                action = dist.distribution.sample()
                log_p  = dist.log_prob(action)

                # Backprop ONLY to read gradients — do NOT call optimizer.step()
                model.policy.zero_grad()
                log_p.mean().backward()

                # Accumulate squared gradients → diagonal Fisher estimate
                for name, param in model.policy.named_parameters():
                    if param.grad is not None:
                        fisher_accum[name].add_(param.grad.detach().cpu() ** 2)

                valid += 1
            except Exception:
                continue

        n = max(valid, 1)
        # Normalise and store
        self.fisher  = {name: v / n  for name, v in fisher_accum.items()}
        self.anchors = {
            name: param.data.detach().cpu().clone()
            for name, param in model.policy.named_parameters()
        }
        self._active = True
        model.policy.set_training_mode(False)

        # Diagnostic: mean Fisher magnitude across all parameters
        total_f = sum(f.sum().item() for f in self.fisher.values())
        n_param = sum(f.numel()      for f in self.fisher.values())
        print(f"  [EWC] ✓ Fisher computed from {valid}/{n_samples} samples.")
        print(f"  [EWC]   Mean |F|: {total_f / max(n_param, 1):.8f}  "
              f"({n_param:,} parameters protected  λ={self.ewc_lambda:.0f})")

        if self.filepath:
            self.save()

    # ── EWC Penalty ─────────────────────────────────────────────────────────────
    def penalty(self, model) -> torch.Tensor:
        """
        Compute the EWC penalty for the current model weights.

            L_ewc = λ × Σᵢ F_ii × (θᵢ − θ*ᵢ)²

        Returns a scalar tensor with grad attached (ready for .backward()).
        Returns torch.tensor(0.0) when EWC is inactive (no penalty).
        """
        if not self.is_active():
            return torch.tensor(0.0)

        device   = next(model.policy.parameters()).device
        terms    = []

        for name, param in model.policy.named_parameters():
            if name in self.fisher and name in self.anchors:
                fisher = self.fisher[name].to(device)
                anchor = self.anchors[name].to(device)
                terms.append((fisher * (param - anchor).pow(2)).sum())

        if not terms:
            return torch.tensor(0.0, device=device)

        return self.ewc_lambda * torch.stack(terms).sum()

    def apply_correction(self, model, n_steps: int = 5) -> float:
        """
        Apply n_steps of EWC-only gradient correction after a PPO update.

        This post-hoc correction pulls important weights back toward their
        Task-1 anchor values if PPO pushed them too far.  Mathematically
        equivalent to joint optimisation when the learning rate is small.

        Returns the mean EWC penalty value (for logging).
        """
        if not self.is_active():
            return 0.0

        total = 0.0
        for _ in range(n_steps):
            pen = self.penalty(model)
            val = pen.item()
            if val < 1e-12:
                break
            model.policy.optimizer.zero_grad()
            pen.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
            total += val

        return total / max(n_steps, 1)

    def weight_change_magnitude(self, model) -> float:
        """
        Compute mean |Δθ| = mean|θ_current − θ*| (weighted by Fisher importance).

        Used as a research metric: measures how much the model has drifted
        from its Task-1 knowledge.  Lower = better memory retention.
        """
        if not self.is_active():
            return float("nan")

        device = next(model.policy.parameters()).device
        total_w_delta = 0.0
        total_fisher  = 0.0

        for name, param in model.policy.named_parameters():
            if name in self.fisher and name in self.anchors:
                F = self.fisher[name].to(device)
                a = self.anchors[name].to(device)
                delta          = (param.detach() - a).abs()
                total_w_delta += (F * delta).sum().item()
                total_fisher  += F.sum().item()

        return total_w_delta / max(total_fisher, 1e-12)

    # ── Persistence ─────────────────────────────────────────────────────────────
    def save(self) -> None:
        """Persist Fisher matrices and anchors to disk."""
        if not self.filepath:
            return
        with open(self.filepath, "wb") as fh:
            pickle.dump({
                "fisher":     self.fisher,   # already on CPU
                "anchors":    self.anchors,  # already on CPU
                "ewc_lambda": self.ewc_lambda,
                "_active":    self._active,
            }, fh)
        print(f"  [EWC] Saved → {self.filepath}")

    def _load(self) -> None:
        try:
            with open(self.filepath, "rb") as fh:
                data = pickle.load(fh)
            self.fisher     = data["fisher"]
            self.anchors    = data["anchors"]
            self.ewc_lambda = data.get("ewc_lambda", self.ewc_lambda)
            self._active    = data.get("_active", True)
            n_param = sum(v.numel() for v in self.fisher.values())
            print(f"  [EWC] ✓ Loaded {self.filepath}  "
                  f"({n_param:,} params  λ={self.ewc_lambda:.0f})")
        except Exception as exc:
            print(f"  [EWC] ⚠ Could not load '{self.filepath}': {exc}. Fresh start.")
