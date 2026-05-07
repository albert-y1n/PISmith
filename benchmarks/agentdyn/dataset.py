"""
Dataset for AgentDyn RL attacker training.
"""

from benchmarks.agentdojo.dataset import (
    AgentDojoDataset,
    compute_injection_candidates,
    format_attacker_prompt,
    get_injection_vector_descriptions,
)


class AgentDynDataset(AgentDojoDataset):
    """Dataset for AgentDyn RL attacker training."""

    benchmark_name = "AgentDyn"


__all__ = [
    "AgentDynDataset",
    "compute_injection_candidates",
    "format_attacker_prompt",
    "get_injection_vector_descriptions",
]
