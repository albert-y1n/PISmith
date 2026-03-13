"""
Reward function for AgentDojo RL attacker training.
"""

import os
import json
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig

from rl_attack.benchmarks.agentdojo.dataset import compute_injection_candidates


AGENTDOJO_MODEL_ALIASES = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-05-13",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview": "gpt-4-0125-preview",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "gpt-5-nano": "gpt-5-nano",
}


def _completion_text(c):
    """Normalize TRL completion to string."""
    if isinstance(c, str):
        return c
    return c[0]["content"] if c else ""


def _resolve_agentdojo_model(target_model: str) -> str:
    """Resolve short model names to exact AgentDojo ModelsEnum values."""
    return AGENTDOJO_MODEL_ALIASES.get(target_model, target_model)


def extract_injection_text(completion: str, format_prompt: bool = True) -> str:
    """Extract injection text from attacker's completion."""
    if format_prompt:
        if "<prompt>" in completion and "</prompt>" in completion:
            try:
                text = completion.split("<prompt>", 1)[1].split("</prompt>", 1)[0]
                return text.strip()
            except (IndexError, ValueError):
                pass
    return completion.strip()


def check_injection_text_format(completion: str) -> bool:
    """Check if completion follows <prompt></prompt> format."""
    return (
        "<prompt>" in completion
        and "</prompt>" in completion
        and completion.count("<prompt>") == 1
        and completion.count("</prompt>") == 1
        and completion.index("<prompt>") < completion.index("</prompt>")
    )


class AgentDojoAttackReward:
    """
    Reward function for AgentDojo RL attacker training.

    Evaluates attack success by:
    1. Extracting injection text from attacker's completion
    2. Placing it into the relevant injection vectors in the environment
    3. Running the target agent pipeline through AgentDojo
    4. Checking if the injection task was executed (security check)

    Reward: 1.0 if attack succeeds (security=True), 0.0 otherwise.
    """

    def __init__(self, config):
        self.__name__ = "AgentDojoAttackReward"
        self.config = config
        self._trainer = None
        self._step_counter = 0

        self.format_prompt = getattr(config, "format_prompt", True)
        self.max_workers = getattr(config, "eval_max_workers", 8)
        self.benchmark_version = getattr(config, "benchmark_version", "v1.2.2")

        # Initialize logging
        log_dir = getattr(config, "output_dir", "logs")
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(log_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # Set up environment variables for local models if needed
        target_model_raw = getattr(config, "target_model", "gpt-4o-mini-2024-07-18")
        target_model = _resolve_agentdojo_model(target_model_raw)
        target_model_id = getattr(config, "target_model_id", None)
        target_model_url = getattr(config, "target_model_url", None)

        if target_model_id:
            os.environ["AGENTDOJO_VLLM_MODEL"] = target_model_id
        if target_model_url:
            os.environ["AGENTDOJO_VLLM_URL"] = target_model_url

        print(f"\nCreating AgentDojo target pipeline...")
        print(f"  Target model: {target_model}")
        if target_model_id:
            print(f"  Model ID: {target_model_id}")
        if target_model_url:
            print(f"  Model URL: {target_model_url}")

        target_defense = getattr(config, "target_defense", None)
        if target_defense:
            print(f"  Defense: {target_defense}")

        self.pipeline_config = PipelineConfig(
            llm=target_model,
            model_id=target_model_id,
            defense=target_defense,
            system_message_name=None,
            system_message=None,
        )
        self.pipeline = AgentPipeline.from_config(self.pipeline_config)
        print(f"  Pipeline name: {self.pipeline.name}")
        print(f"  Pipeline created successfully")

        self._suites: Dict[str, TaskSuite] = {}
        self.agentdojo_logdir = getattr(config, "agentdojo_logdir", None)

    def _get_suite(self, suite_name: str) -> TaskSuite:
        if suite_name not in self._suites:
            self._suites[suite_name] = get_suite(self.benchmark_version, suite_name)
        return self._suites[suite_name]

    def set_trainer(self, trainer):
        """Set trainer reference for step tracking."""
        self._trainer = trainer

    def _get_current_step(self) -> int:
        if self._trainer is not None:
            if hasattr(self._trainer, "state") and hasattr(
                self._trainer.state, "global_step"
            ):
                return self._trainer.state.global_step
        return self._step_counter

    def _evaluate_single(
        self,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str,
        injection_text: str,
        injection_candidates: List[str],
    ) -> Tuple[bool, bool]:
        """Evaluate a single attack by running the AgentDojo pipeline."""
        suite = self._get_suite(suite_name)
        user_task = suite.get_user_task_by_id(user_task_id)
        injection_task = suite.get_injection_task_by_id(injection_task_id)

        injections = {candidate: injection_text for candidate in injection_candidates}

        try:
            utility, security = suite.run_task_with_pipeline(
                agent_pipeline=self.pipeline,
                user_task=user_task,
                injection_task=injection_task,
                injections=injections,
            )
            return utility, security
        except Exception as e:
            warnings.warn(
                f"AgentDojo evaluation failed for {user_task_id}+{injection_task_id}: {e}"
            )
            return False, False

    def __call__(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        batch_size = len(prompts)

        suite_names = kwargs.get("suite_name", ["workspace"] * batch_size)
        user_task_ids = kwargs.get("user_task_id", [""] * batch_size)
        injection_task_ids = kwargs.get("injection_task_id", [""] * batch_size)
        injection_candidates_json = kwargs.get("injection_candidates", ["[]"] * batch_size)
        injection_task_goals = kwargs.get("injection_task_goal", [""] * batch_size)

        # Extract injection text from each completion
        injection_texts = []
        format_ok = []
        for i in range(batch_size):
            completion_text = _completion_text(completions[i])
            injection_text = extract_injection_text(completion_text, self.format_prompt)
            injection_texts.append(injection_text)
            format_ok.append(
                check_injection_text_format(completion_text)
                if self.format_prompt
                else True
            )

        # Parse injection candidates
        all_candidates = []
        for i in range(batch_size):
            try:
                candidates = json.loads(injection_candidates_json[i])
            except (json.JSONDecodeError, TypeError):
                candidates = []
            all_candidates.append(candidates)

        # Run evaluations in parallel
        results = [None] * batch_size

        def eval_task(idx):
            if not format_ok[idx]:
                return idx, False, False
            if not all_candidates[idx]:
                return idx, False, False
            if not injection_texts[idx]:
                return idx, False, False

            utility, security = self._evaluate_single(
                suite_names[idx],
                user_task_ids[idx],
                injection_task_ids[idx],
                injection_texts[idx],
                all_candidates[idx],
            )
            return idx, utility, security

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(eval_task, i) for i in range(batch_size)]
            for future in as_completed(futures):
                try:
                    idx, utility, security = future.result()
                    results[idx] = (utility, security)
                except Exception as e:
                    for i in range(batch_size):
                        if results[i] is None:
                            results[i] = (False, False)
                    warnings.warn(f"Evaluation task failed: {e}")

        for i in range(batch_size):
            if results[i] is None:
                results[i] = (False, False)

        # Compute rewards: security=True means injection was carried out = attack succeeded
        rewards = []
        for i in range(batch_size):
            utility, security = results[i]
            if not format_ok[i]:
                rewards.append(0.0)
            elif security:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        # Logging
        self._step_counter += 1
        current_step = self._get_current_step()

        num_success = sum(1 for r in rewards if r > 0)
        asr = num_success / batch_size if batch_size > 0 else 0
        num_format_ok = sum(format_ok)

        print(
            f"  [AgentDojo] Step {current_step} ASR: {asr:.1%} "
            f"({num_success}/{batch_size}), Format OK: {num_format_ok}/{batch_size}"
        )

        if batch_size > 0:
            print(f"  Suite: {suite_names[0]}, User: {user_task_ids[0]}, Inj: {injection_task_ids[0]}")
            print(f"  Injection: {injection_texts[0][:150]}...")
            print(f"  Reward: {rewards[0]}")

        # Save log
        log_file = os.path.join(self.log_dir, "agentdojo_training_samples.jsonl")
        try:
            log_entry = {
                "step": current_step,
                "timestamp": datetime.now().isoformat(),
                "batch_size": batch_size,
                "asr": asr,
                "num_success": num_success,
                "num_format_ok": num_format_ok,
                "first_sample": {
                    "suite_name": suite_names[0] if batch_size > 0 else "",
                    "user_task_id": user_task_ids[0] if batch_size > 0 else "",
                    "injection_task_id": injection_task_ids[0] if batch_size > 0 else "",
                    "injection_text": injection_texts[0][:500] if batch_size > 0 else "",
                    "reward": rewards[0] if batch_size > 0 else 0,
                },
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return rewards
