"""GRPO configuration for IPI Arena OS."""

from dataclasses import dataclass, field
from typing import List, Optional

from trl import GRPOConfig


DEFAULT_LUNA_MODEL = "gpt-5.6-luna"


@dataclass
class IPIArenaOSGRPOConfig(GRPOConfig):
    """Configuration shared by IPI Arena OS training and rollout evaluation."""

    categories: str = field(
        default="tool,coding,browser",
        metadata={"help": "Comma-separated behavior categories."},
    )
    behavior_ids: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated behavior IDs."},
    )
    waves: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated release waves (1,2,3)."},
    )
    eval_categories: Optional[str] = field(
        default=None,
        metadata={"help": "Optional categories for a separate trainer eval dataset."},
    )
    eval_behavior_ids: Optional[str] = field(
        default=None,
        metadata={"help": "Optional behavior IDs for a separate trainer eval dataset."},
    )
    eval_waves: Optional[str] = field(
        default=None,
        metadata={"help": "Optional waves for a separate trainer eval dataset."},
    )
    behaviors_path: Optional[str] = field(
        default=None,
        metadata={"help": "Override the vendored official behavior data directory."},
    )

    attacker_model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen3-4B-Instruct-2507"
    )
    model_dtype: Optional[str] = field(default="bfloat16")
    format_prompt: bool = field(default=True)

    target_provider: str = field(default="openai")
    target_model: str = field(default=DEFAULT_LUNA_MODEL)
    target_api_key_env: Optional[str] = field(default=None)
    target_base_url: Optional[str] = field(default=None)
    target_reasoning_effort: Optional[str] = field(default="medium")

    judge_provider: str = field(default="openai")
    judge_model: str = field(default=DEFAULT_LUNA_MODEL)
    judge_api_key_env: Optional[str] = field(default=None)
    judge_base_url: Optional[str] = field(default=None)
    judge_reasoning_effort: Optional[str] = field(default="medium")

    worldsim_provider: str = field(default="openai")
    worldsim_model: str = field(default=DEFAULT_LUNA_MODEL)
    worldsim_api_key_env: Optional[str] = field(default=None)
    worldsim_base_url: Optional[str] = field(default=None)
    worldsim_reasoning_effort: Optional[str] = field(default="medium")

    max_steps: int = field(default=5)
    eval_max_workers: int = field(default=8)
    soft_rewards: bool = field(default=False)
    reward_functions: List[str] = field(
        default_factory=lambda: ["IPIArenaOSAttackReward"]
    )

    max_completion_length: int = field(default=512)
    adaptive: bool = field(default=False)
    seed: int = field(default=42)
    save_total_limit: int = field(default=10)
    max_grad_norm: float = field(default=1.0)
    loss_type: str = field(default="grpo")
    epsilon: float = field(default=0.5)
