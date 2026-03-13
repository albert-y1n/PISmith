"""
Adaptive GRPO Trainer for PISmith.

Extends TRL's GRPOTrainer with:
1. Dynamic entropy coefficient based on batch ASR
2. Success weightinging for rare successful samples
3. Asymmetric advantage scaling (amplify successes, dampen failures)
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
from dataclasses import dataclass

from trl import GRPOTrainer
from trl.trainer.utils import entropy_from_logits, selective_log_softmax

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Distributed Helpers
# =============================================================================

def is_main_process() -> bool:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def all_reduce_sum(value: float) -> float:
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, dtype=torch.float32,
                     device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


# =============================================================================
# Adaptive Loss Configuration
# =============================================================================

@dataclass
class AdaptiveLossConfig:
    """Configuration for adaptive GRPO loss mechanisms."""

    enable_adaptive_loss: bool = True

    # Entropy adaptation
    enable_adaptive_entropy: bool = True
    use_fixed_entropy_coef: bool = False   # Ablation: constant entropy coefficient
    base_entropy_coef: float = 0.001       # Coefficient when ASR >= 50%
    max_entropy_coef: float = 0.01   
    entropy_asr_threshold: float = 0.5
    very_low_asr_threshold: float = 0.0

    # Success weightinging
    enable_success_weighting: bool = True
    max_success_weighting: float = 5.0
    weighting_asr_threshold: float = 0.5

# =============================================================================
# Adaptive GRPO Trainer
# =============================================================================

class AdaptiveGRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer with adaptive loss mechanisms.

    Key features for sparse binary reward settings:
    - Dynamic entropy coefficient: increases when global ASR is low
    - Success weightinging: upweight rare successful samples
    - Asymmetric advantage scaling
    """

    def __init__(self, *args, adaptive_config: Optional[AdaptiveLossConfig] = None, **kwargs):
        self.adaptive_config = adaptive_config or AdaptiveLossConfig()
        # Normalize use_fixed_entropy_coef (may come as string from YAML)
        uf = self.adaptive_config.use_fixed_entropy_coef
        self.adaptive_config.use_fixed_entropy_coef = (
            uf is True or (isinstance(uf, str) and uf.strip().lower() in ("true", "1", "yes"))
        )
        super().__init__(*args, **kwargs)
        self._train_step = 0

        if is_main_process() and self.adaptive_config.enable_adaptive_loss:
            print("Adaptive loss enabled:")
            print(f"  entropy [{self.adaptive_config.base_entropy_coef} — {self.adaptive_config.max_entropy_coef}]")
            print(f"  success weighting max: {self.adaptive_config.max_success_weighting}")

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt", "image", "images",
                "context", "target_inst", "injected_task",
                # InjectAgent fields
                "Attacker Instruction", "User Instruction",
                "Tool Response Template", "User Tool", "Attacker Tools",
            ]

    # ------------------------------------------------------------------
    # Adaptive helpers
    # ------------------------------------------------------------------

    def _compute_global_asr(self, rewards: torch.Tensor) -> float:
        local_sum = rewards.float().sum().item()
        local_count = float(rewards.numel())
        global_sum = all_reduce_sum(local_sum)
        global_count = all_reduce_sum(local_count)
        return global_sum / global_count if global_count > 0 else 0.0

    def _get_global_asr_from_metrics(self) -> float:
        mode = "train" if self.model.training else "eval"
        metrics = getattr(self, "_metrics", None)
        if metrics and mode in metrics:
            for name in getattr(self, "reward_func_names", []):
                key = f"rewards/{name}/mean"
                vals = metrics[mode].get(key)
                if vals:
                    return float(vals[-1])
            vals = metrics[mode].get("reward")
            if vals:
                return float(vals[-1])
        return 0.0

    def _compute_adaptive_entropy_coef(self, asr: float) -> float:
        cfg = self.adaptive_config
        if asr < cfg.very_low_asr_threshold:
            return cfg.max_entropy_coef
        if asr < cfg.entropy_asr_threshold:
            ratio = (cfg.entropy_asr_threshold - asr) / max(
                cfg.entropy_asr_threshold - cfg.very_low_asr_threshold, 1e-8
            )
            ratio = max(0.0, min(1.0, ratio))
            return cfg.base_entropy_coef + (cfg.max_entropy_coef - cfg.base_entropy_coef) * ratio
        return cfg.base_entropy_coef

    def _compute_success_weighting(self, asr: float) -> float:
        cfg = self.adaptive_config
        if not cfg.enable_success_weighting or asr >= cfg.weighting_asr_threshold:
            return 1.0
        ratio = 1.0 - asr / cfg.weighting_asr_threshold
        return min(1.0 + (cfg.max_success_weighting - 1.0) * ratio, cfg.max_success_weighting)

    def _modify_advantages(
        self, advantages: torch.Tensor, rewards: torch.Tensor, asr: float
    ) -> torch.Tensor:
        weighting = self._compute_success_weighting(asr)
        pos = rewards > 0
        modified = advantages.clone()
        modified = torch.where(pos, advantages * weighting, modified)
        return modified

    # ------------------------------------------------------------------
    # Forward override to capture entropies
    # ------------------------------------------------------------------

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep,
        batch_size=None, compute_entropy=False,
        pixel_values=None, image_grid_thw=None, num_images=None,
        pixel_attention_mask=None, image_sizes=None, token_type_ids=None,
    ):
        batch_size = batch_size or input_ids.size(0)
        all_logps, all_entropies = [], []
        for start in range(0, input_ids.size(0), batch_size):
            b_ids = input_ids[start: start + batch_size]
            b_mask = attention_mask[start: start + batch_size]
            model_inputs = {"input_ids": b_ids, "attention_mask": b_mask, "use_cache": False}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start: start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start: start + batch_size]
            logits = model(**model_inputs).logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :] / self.temperature
            comp_ids = b_ids[:, -logits_to_keep:]
            all_logps.append(selective_log_softmax(logits, comp_ids))
            if compute_entropy:
                all_entropies.append(entropy_from_logits(logits))

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        self._last_entropies = entropies
        return logps, entropies

    # ------------------------------------------------------------------
    # compute_loss override
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cfg = self.adaptive_config
        global_asr = 0.0
        entropy_coef = cfg.base_entropy_coef

        success_weighting = 1.0
        if cfg.enable_adaptive_loss:
            if "rewards" in inputs:
                global_asr = self._compute_global_asr(inputs["rewards"])
            else:
                global_asr = self._get_global_asr_from_metrics()

            if cfg.use_fixed_entropy_coef:
                entropy_coef = cfg.base_entropy_coef
            elif cfg.enable_adaptive_entropy:
                entropy_coef = self._compute_adaptive_entropy_coef(global_asr)
            else:
                entropy_coef = 0.0

            success_weighting = self._compute_success_weighting(global_asr)

            if self._train_step % 10 == 0 and is_main_process():
                print(f"  [Adaptive] step={self._train_step} ASR={global_asr:.4f} "
                      f"entropy_coef={entropy_coef:.4f} weighting={success_weighting:.2f}")

        # Modify advantages
        original_advantages = None
        if cfg.enable_adaptive_loss and "advantages" in inputs and "rewards" in inputs:
            original_advantages = inputs["advantages"].clone()
            inputs["advantages"] = self._modify_advantages(
                inputs["advantages"], inputs["rewards"], global_asr
            )

        # Main GRPO loss
        self._current_entropy_coef = entropy_coef
        self._last_entropies = None
        loss = super().compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # Entropy bonus
        entropy_loss = torch.tensor(0.0, device=loss.device)
        mean_entropy = torch.tensor(0.0, device=loss.device)
        if (
            cfg.enable_adaptive_loss
            and cfg.enable_adaptive_entropy
            and entropy_coef > 0
            and getattr(self, "_last_entropies", None) is not None
            and "completion_mask" in inputs
        ):
            comp_mask = inputs["completion_mask"]
            tok_count = comp_mask.sum().clamp(min=1.0)
            mean_entropy = (self._last_entropies * comp_mask).sum() / tok_count
            if mean_entropy < 0.5:
                entropy_loss = -entropy_coef * mean_entropy
                loss = loss + entropy_loss
        self._last_entropies = None

        # Logging
        self._train_step += 1
        if is_main_process() and WANDB_AVAILABLE and wandb.run is not None:
            me = mean_entropy.item() if isinstance(mean_entropy, torch.Tensor) else float(mean_entropy)
            log_dict = {
                "adaptive/global_asr": global_asr,
                "adaptive/entropy_coef": float(entropy_coef),
                "adaptive/fixed_entropy_mode": float(cfg.use_fixed_entropy_coef),
                "adaptive/success_weighting": success_weighting,
                "adaptive/entropy_loss": entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else float(entropy_loss),
                "adaptive/mean_entropy": me,
            }
            if original_advantages is not None and "rewards" in inputs:
                pos_mask = inputs["rewards"] > 0
                if pos_mask.any():
                    log_dict["adaptive/pos_advantage_mean"] = inputs["advantages"][pos_mask].mean().item()
                    log_dict["adaptive/pos_advantage_original"] = original_advantages[pos_mask].mean().item()
            wandb.log(log_dict, commit=False)

        if return_outputs:
            return loss, None
        return loss
