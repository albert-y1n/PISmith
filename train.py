#!/usr/bin/env python3
"""
Unified GRPO Training Script for PISmith

Trains an attacker LLM using Group Relative Policy Optimization (GRPO)
on one of three benchmarks: PIArena, InjecAgent, or AgentDojo.

Usage:
    # PIArena
    accelerate launch -m train \\
        --benchmark piarena \\
        --config_file configs/piarena.yaml

    # InjecAgent
    accelerate launch -m train \\
        --benchmark injecagent \\
        --config_file configs/injecagent.yaml

    # AgentDojo
    accelerate launch -m train \\
        --benchmark agentdojo \\
        --config_file configs/agentdojo.yaml
"""

import os
import sys
import argparse

from trl import ModelConfig
from peft import LoraConfig

from core.utils import set_random_seed


BENCHMARKS = ["piarena", "injecagent", "agentdojo"]


def _pop_benchmark_arg(argv):
    """Extract --benchmark from argv before TrlParser sees it."""
    benchmark = "piarena"
    clean_argv = []
    i = 0
    while i < len(argv):
        if argv[i] == "--benchmark" and i + 1 < len(argv):
            benchmark = argv[i + 1]
            i += 2
        elif argv[i].startswith("--benchmark="):
            benchmark = argv[i].split("=", 1)[1]
            i += 1
        else:
            clean_argv.append(argv[i])
            i += 1
    return benchmark, clean_argv


def _rename_config_file_to_config(argv):
    """Rename --config_file to --config so TrlParser can find it.

    TrlParser.parse_args_and_config() looks for '--config', not '--config_file'.
    This lets shell scripts keep using --config_file while TrlParser works correctly.
    """
    result = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config_file" and i + 1 < len(argv):
            result.append("--config")
            result.append(argv[i + 1])
            i += 2
        elif argv[i].startswith("--config_file="):
            result.append("--config=" + argv[i].split("=", 1)[1])
            i += 1
        else:
            result.append(argv[i])
            i += 1
    return result


def main():
    benchmark, remaining_argv = _pop_benchmark_arg(sys.argv[1:])
    remaining_argv = _rename_config_file_to_config(remaining_argv)

    if benchmark not in BENCHMARKS:
        print(f"Unknown benchmark: {benchmark}. Choose from {BENCHMARKS}")
        sys.exit(1)

    sys.argv = [sys.argv[0]] + remaining_argv

    if benchmark == "piarena":
        _train_piarena()
    elif benchmark == "injecagent":
        _train_injecagent()
    elif benchmark == "agentdojo":
        _train_agentdojo()


def _build_peft_config(model_config: ModelConfig):
    if not model_config.use_peft:
        return None
    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )


def _apply_model_init_kwargs(grpo_config, model_config: ModelConfig):
    """Set standard model init kwargs on grpo_config."""
    grpo_config.gradient_checkpointing_kwargs = {"use_reentrant": False}
    grpo_config.ddp_find_unused_parameters = False
    grpo_config.model_init_kwargs = {"device_map": None}

    if model_config.torch_dtype is not None:
        grpo_config.model_init_kwargs["torch_dtype"] = model_config.torch_dtype
    else:
        grpo_config.model_init_kwargs["torch_dtype"] = "bfloat16"

    if model_config.attn_implementation is not None:
        grpo_config.model_init_kwargs["attn_implementation"] = model_config.attn_implementation
    else:
        grpo_config.model_init_kwargs["attn_implementation"] = "sdpa"


def _make_trainer(grpo_config, reward_functions, train_dataset, peft_config, use_adaptive,
                  eval_dataset=None):
    """Create the appropriate GRPO trainer."""
    from core.trainer import AdaptiveGRPOTrainer, AdaptiveLossConfig
    from trl import GRPOTrainer

    extra = {}
    if eval_dataset is not None:
        extra["eval_dataset"] = eval_dataset

    if use_adaptive:
        adaptive_config = AdaptiveLossConfig(enable_adaptive_loss=True)
        trainer = AdaptiveGRPOTrainer(
            args=grpo_config,
            model=grpo_config.attacker_model_name_or_path,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_dataset,
            adaptive_config=adaptive_config,
            **extra,
        )
        print("AdaptiveGRPOTrainer initialized")
    else:
        trainer = GRPOTrainer(
            args=grpo_config,
            model=grpo_config.attacker_model_name_or_path,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_dataset,
            **extra,
        )
        print("GRPOTrainer initialized")

    return trainer


def _train_piarena():
    from trl import TrlParser
    from benchmarks.piarena.config import PIArenaGRPOConfig
    from benchmarks.piarena.dataset import PIArenaDataset
    from benchmarks.piarena.reward import PIArenaAttackReward

    parser = TrlParser((PIArenaGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()

    print("=" * 70)
    print("PIArena RL Attacker Training")
    print("=" * 70)
    set_random_seed(grpo_config.seed)

    train_dataset = PIArenaDataset(
        data_path=grpo_config.dataset_path,
        split=grpo_config.dataset_split,
        start_idx=grpo_config.train_start_idx,
        end_idx=grpo_config.train_end_idx,
        attack_template=grpo_config.attack_template,
        format_prompt=grpo_config.format_prompt,
    )
    print(f"Loaded {len(train_dataset)} PIArena training samples")

    reward_functions = [PIArenaAttackReward(grpo_config)]

    peft_config = _build_peft_config(model_config)
    _apply_model_init_kwargs(grpo_config, model_config)

    print(f"  Attacker: {grpo_config.attacker_model_name_or_path}")
    print(f"  Target: {grpo_config.target_model_name_or_path}")
    print(f"  Defense: {grpo_config.defense_method}")
    print(f"  Output: {grpo_config.output_dir}")

    trainer = _make_trainer(
        grpo_config, reward_functions, train_dataset, peft_config,
        use_adaptive=getattr(grpo_config, "adaptive", False),
    )
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)
    print(f"\nTraining complete. Checkpoints: {grpo_config.output_dir}")


def _train_injecagent():
    from trl import TrlParser
    from benchmarks.injecagent.config import InjecAgentGRPOConfig
    from benchmarks.injecagent.dataset import InjecAgentDataset
    from benchmarks.injecagent.reward import InjecAgentToolCallingReward

    parser = TrlParser((InjecAgentGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()

    print("=" * 70)
    print("InjecAgent RL Attacker Training")
    print("=" * 70)
    set_random_seed(grpo_config.seed)

    train_dataset = InjecAgentDataset(
        data_path=grpo_config.dataset,
    )
    print(f"Loaded {len(train_dataset)} InjecAgent training samples")

    reward_functions = [InjecAgentToolCallingReward(grpo_config)]

    peft_config = _build_peft_config(model_config)
    _apply_model_init_kwargs(grpo_config, model_config)

    print(f"  Attacker: {grpo_config.attacker_model_name_or_path}")
    print(f"  Targets: {grpo_config.target_model_name_or_path}")
    print(f"  Output: {grpo_config.output_dir}")

    trainer = _make_trainer(
        grpo_config, reward_functions, train_dataset, peft_config,
        use_adaptive=getattr(grpo_config, "adaptive", False),
    )
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)
    print(f"\nTraining complete. Checkpoints: {grpo_config.output_dir}")


def _parse_task_list(s):
    """Parse comma-separated task IDs into a list, or return None if empty."""
    if not s:
        return None
    return [t.strip() for t in s.split(",") if t.strip()]


def _train_agentdojo():
    from trl import TrlParser
    from benchmarks.agentdojo.config import AgentDojoGRPOConfig
    from benchmarks.agentdojo.dataset import AgentDojoDataset
    from benchmarks.agentdojo.reward import AgentDojoAttackReward

    parser = TrlParser((AgentDojoGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()

    print("=" * 70)
    print("AgentDojo RL Attacker Training")
    print("=" * 70)
    set_random_seed(grpo_config.seed)

    # ── Training dataset ──────────────────────────────────────
    train_user_tasks     = _parse_task_list(grpo_config.train_user_tasks)
    train_injection_tasks = _parse_task_list(grpo_config.train_injection_tasks)

    print(f"\nLoading AgentDojo train dataset...")
    print(f"  Suites     : {grpo_config.train_suites}")
    if train_user_tasks:
        print(f"  User tasks : {train_user_tasks}")
    if train_injection_tasks:
        print(f"  Inj tasks  : {train_injection_tasks}")

    train_dataset = AgentDojoDataset(
        suites=grpo_config.train_suites,
        benchmark_version=grpo_config.benchmark_version,
        user_tasks=train_user_tasks,
        injection_tasks=train_injection_tasks,
    )
    print(f"Loaded {len(train_dataset)} AgentDojo training samples")

    # Per-suite breakdown
    suite_counts = {}
    for s in train_dataset.samples:
        suite_counts[s["suite_name"]] = suite_counts.get(s["suite_name"], 0) + 1
    for sn, cnt in suite_counts.items():
        print(f"  {sn}: {cnt} samples")

    if len(train_dataset) == 0:
        raise ValueError(
            "No training samples found. Check suite/task configuration.\n"
            f"  train_suites: {grpo_config.train_suites}\n"
            f"  train_user_tasks: {train_user_tasks}\n"
            f"  train_injection_tasks: {train_injection_tasks}"
        )

    # ── Eval dataset (optional separate split) ────────────────
    eval_dataset = None
    has_separate_eval = (
        grpo_config.eval_suites is not None
        or grpo_config.eval_injection_tasks is not None
        or grpo_config.eval_user_tasks is not None
    )
    if has_separate_eval:
        eval_suites_str = grpo_config.eval_suites or grpo_config.train_suites
        eval_user_tasks      = _parse_task_list(grpo_config.eval_user_tasks)
        eval_injection_tasks = _parse_task_list(grpo_config.eval_injection_tasks)

        print(f"\nLoading AgentDojo eval dataset...")
        print(f"  Suites     : {eval_suites_str}")
        if eval_user_tasks:
            print(f"  User tasks : {eval_user_tasks}")
        if eval_injection_tasks:
            print(f"  Inj tasks  : {eval_injection_tasks}")

        eval_dataset = AgentDojoDataset(
            suites=eval_suites_str,
            benchmark_version=grpo_config.benchmark_version,
            user_tasks=eval_user_tasks,
            injection_tasks=eval_injection_tasks,
        )
        if len(eval_dataset) == 0:
            print("WARNING: No eval samples found, disabling eval_dataset.")
            eval_dataset = None
        else:
            print(f"Loaded {len(eval_dataset)} AgentDojo eval samples")

    # ── Trainer setup ──────────────────────────────────────────
    reward_functions = [AgentDojoAttackReward(grpo_config)]

    peft_config = _build_peft_config(model_config)
    _apply_model_init_kwargs(grpo_config, model_config)

    print(f"\n  Attacker : {grpo_config.attacker_model_name_or_path}")
    print(f"  Target   : {grpo_config.target_model}")
    if getattr(grpo_config, "target_defense", None):
        print(f"  Defense  : {grpo_config.target_defense}")
    print(f"  Suites   : {grpo_config.train_suites}")
    if eval_dataset is not None:
        print(f"  Eval set : {len(eval_dataset)} samples")
    print(f"  Output   : {grpo_config.output_dir}")

    trainer = _make_trainer(
        grpo_config, reward_functions, train_dataset, peft_config,
        use_adaptive=getattr(grpo_config, "adaptive", False),
        eval_dataset=eval_dataset,
    )
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)
    print(f"\nTraining complete. Checkpoints: {grpo_config.output_dir}")


if __name__ == "__main__":
    main()
