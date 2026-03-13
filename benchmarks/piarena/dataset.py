"""
PIArena dataset for GRPO training.
"""

import os
import json
import glob
from typing import Dict, List, Any

from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset

from rl_attack.core.utils import (
    format_attack_prompt,
    ATTACKER_SYS_PROMPT,
    ATTACK_TEMPLATES_FORMATTED,
)


class PIArenaDataset(Dataset):
    """
    PyTorch Dataset for PIArena prompt injection attack training.

    Each sample:
      - prompt: attacker LLM input (list of message dicts)
      - context: original document
      - target_inst: user's question
      - injected_task: task to inject
    """

    def __init__(
        self,
        data_path: str = "sleeepeer/PIArena",
        split: str = "squad_v2",
        start_idx: int = 0,
        end_idx: int = 100,
        attack_template: str = "recent_update",
        max_context_chars: int = 500,
        format_prompt: bool = True,
    ):
        self.attack_template = attack_template
        self.max_context_chars = max_context_chars
        self.format_prompt = format_prompt

        is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

        if os.path.isdir(data_path):
            self.dataset = self._load_from_dir(data_path, split, start_idx, end_idx, is_main)
        elif os.path.isfile(data_path):
            with open(data_path) as f:
                data = json.load(f)
            self.dataset = data[start_idx:end_idx]
            if is_main:
                print(f"Loaded {len(self.dataset)} samples from {data_path}")
        else:
            # HuggingFace dataset id
            if is_main:
                hf_ds = hf_load_dataset(data_path, split=split, verification_mode="no_checks")
                print(f"Loaded {data_path}/{split} from HuggingFace")
            else:
                hf_ds = hf_load_dataset(
                    data_path, split=split, verification_mode="no_checks",
                    download_mode="reuse_cache_if_exists"
                )
            self.dataset = [hf_ds[i] for i in range(start_idx, min(end_idx, len(hf_ds)))]

        # Validate template
        if attack_template not in ATTACK_TEMPLATES_FORMATTED:
            raise ValueError(f"Unknown attack template: {attack_template}. Available: {sorted(ATTACK_TEMPLATES_FORMATTED)}")

    def _load_from_dir(self, data_path, split, start_idx, end_idx, is_main):
        # Check for parquet files in data_path or data_path/data/
        parquet_dir = None
        for candidate in [data_path, os.path.join(data_path, "data")]:
            if os.path.isdir(candidate) and any(f.endswith(".parquet") for f in os.listdir(candidate)):
                parquet_dir = candidate
                break

        if parquet_dir:
            pattern = os.path.join(parquet_dir, f"{split}-*-of-*.parquet")
            files = sorted(glob.glob(pattern))
            if not files:
                single = os.path.join(parquet_dir, f"{split}-00000-of-00001.parquet")
                if os.path.isfile(single):
                    files = [single]
            if not files:
                raise FileNotFoundError(f"No parquet files for split '{split}' in {parquet_dir}")
            hf_ds = hf_load_dataset("parquet", data_files=files, split="train", verification_mode="no_checks")
            if is_main:
                print(f"Loaded {split} from local parquet: {parquet_dir}")
            return [hf_ds[i] for i in range(start_idx, min(end_idx, len(hf_ds)))]

        # Treat as HF cache directory
        dataset_name = "sleeepeer/PIArena"
        cache_dir = None
        if "sleeepeer" in data_path or "pi_arena" in data_path.lower():
            parent = os.path.dirname(data_path)
            cache_dir = os.path.dirname(parent) if os.path.basename(parent) == "datasets" else parent
        hf_ds = hf_load_dataset(
            dataset_name, split=split, cache_dir=cache_dir,
            verification_mode="no_checks", download_mode="reuse_cache_if_exists"
        )
        if is_main:
            print(f"Loaded {dataset_name}/{split} from HF cache")
        return [hf_ds[i] for i in range(start_idx, min(end_idx, len(hf_ds)))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        context = sample.get("context") or (
            sample.get("contexts", [""])[0] if isinstance(sample.get("contexts"), list) else ""
        )
        target_inst = sample.get("target_inst") or sample.get("question", "")
        injected_task = sample.get("injected_task", "")

        user_prompt = format_attack_prompt(
            template_name=self.attack_template,
            context=context,
            user_task=target_inst,
            injected_task=injected_task,
            max_context_chars=self.max_context_chars,
            format_prompt=self.format_prompt,
        )
        prompt = [{"role": "user", "content": ATTACKER_SYS_PROMPT + "\n\n" + user_prompt}]

        return {
            "prompt": prompt,
            "context": context,
            "target_inst": target_inst,
            "injected_task": injected_task,
        }
