"""
InjectAgent dataset for GRPO training.
"""

import json
from typing import Dict, Any

from torch.utils.data import Dataset

from rl_attack.core.utils import get_injecagent_attacker_sys_prompt


class InjecAgentDataset(Dataset):
    """
    Dataset for InjectAgent format data.

    JSON fields per sample:
      - Attacker Instruction: goal to inject
      - User Instruction: user's task
      - Tool Response Template: template with <Attacker Instruction> placeholder
      - User Tool, Attacker Tools, Tool Parameters, Thought
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to InjectAgent JSON file.
        """
        with open(data_path) as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx].copy()

        tool_response = sample.get("Tool Response Template", "")
        context_tail = tool_response.replace("<Attacker Instruction>", "<INJECTION_PLACEHOLDER>").strip()
        if not context_tail or len(context_tail) < 10:
            context_tail = sample.get("User Instruction", "")

        prompt_content = get_injecagent_attacker_sys_prompt(
            context_tail=context_tail,
            injected_task=sample.get("Attacker Instruction", ""),
            user_task=sample.get("User Instruction", ""),
        )

        sample["prompt"] = [{"role": "user", "content": prompt_content}]
        return sample
