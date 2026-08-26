"""GRPO attack reward backed by the official IPI Arena OS runner."""

from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from .client import build_runner


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    return completion[0].get("content", "") if completion else ""


def extract_injection_text(completion: str, format_prompt: bool = True) -> str:
    if format_prompt and "<prompt>" in completion and "</prompt>" in completion:
        return completion.split("<prompt>", 1)[1].split("</prompt>", 1)[0].strip()
    return completion.strip()


def valid_prompt_format(completion: str) -> bool:
    return (
        completion.count("<prompt>") == 1
        and completion.count("</prompt>") == 1
        and completion.index("<prompt>") < completion.index("</prompt>")
        if "<prompt>" in completion and "</prompt>" in completion
        else False
    )


class IPIArenaOSAttackReward:
    """Return 1 when every official criterion passes, otherwise 0."""

    benchmark_name = "IPI Arena OS"
    benchmark_slug = "ipi_arena_os"

    def __init__(self, config: Any):
        self.__name__ = self.__class__.__name__
        self.config = config
        self.format_prompt = getattr(config, "format_prompt", True)
        self.soft_rewards = getattr(config, "soft_rewards", False)
        self.max_workers = getattr(config, "eval_max_workers", 8)
        self.runner = build_runner(config, verbose=False)
        self._trainer = None
        self._step_counter = 0

        output_dir = getattr(config, "output_dir", "checkpoints/ipi_arena_os")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def set_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    def _current_step(self) -> int:
        state = getattr(self._trainer, "state", None)
        return getattr(state, "global_step", self._step_counter)

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        batch_size = len(completions)
        behavior_ids = kwargs.get("behavior_id", [""] * batch_size)
        categories = kwargs.get("category", [""] * batch_size)

        texts = [completion_text(completion) for completion in completions]
        injections = [
            extract_injection_text(text, self.format_prompt) for text in texts
        ]
        format_ok = [
            valid_prompt_format(text) if self.format_prompt else True for text in texts
        ]
        results: list[Any] = [None] * batch_size

        def evaluate(index: int):
            if not format_ok[index] or not injections[index]:
                return index, None
            return index, self.runner.run_one(behavior_ids[index], injections[index])

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(evaluate, index) for index in range(batch_size)]
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as exc:
                    warnings.warn(f"IPI Arena OS rollout failed: {exc}")

        rewards = []
        for index, result in enumerate(results):
            if not format_ok[index] or result is None or result.error:
                rewards.append(0.0)
            elif self.soft_rewards and result.criteria:
                passed = sum(int(value["passed"]) for value in result.criteria.values())
                rewards.append(passed / len(result.criteria))
            else:
                rewards.append(float(result.is_break))

        self._step_counter += 1
        step = self._current_step()
        successes = sum(reward >= 1.0 for reward in rewards)
        print(
            f"  [IPI Arena OS] Step {step} ASR: {successes / max(batch_size, 1):.1%} "
            f"({successes}/{batch_size}), Format OK: {sum(format_ok)}/{batch_size}"
        )

        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "batch_size": batch_size,
            "asr": successes / max(batch_size, 1),
            "samples": [
                {
                    "behavior_id": behavior_ids[index],
                    "category": categories[index],
                    "injection_text": injections[index],
                    "reward": rewards[index],
                    "error": getattr(results[index], "error", None),
                }
                for index in range(batch_size)
            ],
        }
        try:
            path = os.path.join(self.log_dir, "ipi_arena_os_training_samples.jsonl")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            warnings.warn(f"Could not write IPI Arena OS training log: {exc}")

        return rewards
