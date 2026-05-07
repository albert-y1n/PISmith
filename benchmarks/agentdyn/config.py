"""
GRPO configuration for AgentDyn benchmark.
"""

from dataclasses import dataclass, field
from typing import Optional

from benchmarks.agentdojo.config import AgentDojoGRPOConfig


@dataclass
class AgentDynGRPOConfig(AgentDojoGRPOConfig):
    """GRPO configuration for AgentDyn RL attacker training."""

    train_suites: str = field(
        default="workspace",
        metadata={
            "help": (
                "Comma-separated AgentDyn suite names for training "
                "(e.g., 'workspace', 'github', 'dailylife', 'shopping', or combinations)."
            )
        },
    )
    eval_suites: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated AgentDyn suites for evaluation. "
                "If None, uses train_suites."
            )
        },
    )
    benchmark_version: str = field(
        default="v1.2.2",
        metadata={"help": "AgentDyn benchmark version"},
    )
