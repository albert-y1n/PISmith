"""
GRPO configuration for AgentDojo benchmark.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from trl import GRPOConfig


@dataclass
class AgentDojoGRPOConfig(GRPOConfig):
    """GRPO configuration for AgentDojo RL attacker training."""

    # --- Suite/Task Selection ---
    train_suites: str = field(
        default="workspace",
        metadata={"help": "Comma-separated suite names for training (e.g., 'workspace' or 'workspace,banking')"},
    )
    eval_suites: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated suites for evaluation. If None, uses train_suites."},
    )
    train_injection_tasks: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated injection task IDs for training. None = all."},
    )
    eval_injection_tasks: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated injection task IDs for evaluation. None = all."},
    )
    train_user_tasks: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated user task IDs for training. None = all."},
    )
    eval_user_tasks: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated user task IDs for evaluation. None = all."},
    )
    benchmark_version: str = field(
        default="v1.2.2",
        metadata={"help": "AgentDojo benchmark version"},
    )

    # --- Target Model (the agent being attacked) ---
    target_model: str = field(
        default="gpt-4o-mini-2024-07-18",
        metadata={"help": "Target model for AgentDojo pipeline (ModelsEnum value or 'local')"},
    )
    target_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Model ID for local target models; sets AGENTDOJO_VLLM_MODEL env var"},
    )
    target_model_url: Optional[str] = field(
        default=None,
        metadata={"help": "vLLM URL for local target models; sets AGENTDOJO_VLLM_URL env var"},
    )
    target_defense: Optional[str] = field(
        default=None,
        metadata={"help": "Defense method for target pipeline (e.g., 'tool_filter'). None = no defense."},
    )

    # --- Attacker Model ---
    attacker_model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-4B-Instruct-2507")
    model_dtype: Optional[str] = field(default="bfloat16")

    # --- Generation ---
    max_completion_length: int = field(default=512)

    format_prompt: bool = field(
        default=True,
        metadata={"help": "If True, require <prompt></prompt> tags in attacker output"},
    )

    # --- Reward ---
    reward_functions: List[str] = field(
        default_factory=lambda: ["AgentDojoAttackReward"]
    )
    soft_rewards: bool = field(default=False)

    # --- Adaptive ---
    adaptive: bool = field(default=False)

    # --- Training ---
    seed: int = field(default=1024)
    save_total_limit: int = field(default=10)
    max_grad_norm: float = field(default=1.0)
    loss_type: str = field(default="grpo")
    epsilon: float = field(default=0.5)

    # --- Evaluation ---
    eval_max_workers: int = field(default=16)
    agentdojo_logdir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save agentdojo evaluation logs. None = no logging."},
    )
