"""
GRPO configuration for InjectAgent benchmark.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from trl import GRPOConfig


@dataclass
class InjecAgentGRPOConfig(GRPOConfig):
    """GRPO config for InjectAgent tool-calling attack training."""

    # --- Dataset ---
    dataset: str = field(
        default=None,
        metadata={"help": "Path to InjectAgent JSON training data file"},
    )
    tools_path: str = field(
        default="rl_attack/data/injecagent/tools.json",
        metadata={"help": "Path to InjecAgent tools.json"},
    )

    # --- Models ---
    attacker_model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-4B-Instruct-2507")
    model_dtype: Optional[str] = field(default="bfloat16")
    target_model_name_or_path: str = field(
        default="",
        metadata={"help": "Target model name(s); semicolon-separated for multiple"},
    )
    target_model_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "vLLM URL(s); semicolon-separated for multiple"},
    )
    target_openai_config: Optional[str] = field(
        default=None,
        metadata={"help": "OpenAI config YAML for API targets"},
    )

    # --- Generation ---
    max_completion_length: int = field(default=1024)
    target_model_max_completion_length: int = field(default=512)
    target_model_temperature: Optional[float] = field(default=None)

    # --- Reward ---
    reward_functions: List[str] = field(
        default_factory=lambda: ["InjecAgentToolCallingReward"]
    )
    soft_rewards: bool = field(default=True)
    model_wise_reward_weights: Optional[List[float]] = field(default=None)
    reasoning_effort: str = field(default="minimal")

    # --- Format ---
    format_prompt: bool = field(default=True)
    prompt_strategy: str = field(
        default="recent_update",
        metadata={"help": "'default' (original InjectAgent prompt) or 'recent_update'"},
    )

    # --- Training ---
    seed: int = field(default=1024)
    save_total_limit: int = field(default=10)
    max_grad_norm: float = field(default=1.0)
    loss_type: str = field(default="grpo")
    epsilon: float = field(default=0.5)

    # --- Adaptive ---
    adaptive: bool = field(default=False)
