"""
Reward function for AgentDyn RL attacker training.
"""

from benchmarks.agentdojo.reward import AgentDojoAttackReward


class AgentDynAttackReward(AgentDojoAttackReward):
    """Reward function for AgentDyn RL attacker training."""

    benchmark_name = "AgentDyn"
    benchmark_slug = "agentdyn"
